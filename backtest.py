"""
╔══════════════════════════════════════════════════════════════════╗
║  BACKTEST HARNESS — Intraday Signal Engine                       ║
║                                                                  ║
║  Replays SignalGen + OptPicker over historical 5-min candles     ║
║  with realistic slippage + brokerage. Produces per-trade and     ║
║  aggregate P&L so you can answer "does this strategy work?"      ║
║  before risking real capital.                                    ║
║                                                                  ║
║  Usage:                                                          ║
║    python backtest.py --instrument NIFTY --days 60               ║
║    python backtest.py --all --days 90 --csv results.csv          ║
║                                                                  ║
║  Notes:                                                          ║
║  - Uses the LIVE Angel One API (server.py's AngelClient) to      ║
║    fetch historical 5-min candles. Requires the same env vars.   ║
║  - Option premiums are estimated from live spot + delta ladder   ║
║    (no historical option chain available via Angel One API).     ║
║    This is a known limitation — back-of-envelope, not penny-     ║
║    accurate. Use the win-rate as the primary signal of edge.     ║
║  - Skips Claude AI layers (B/C/D) entirely — those depend on     ║
║    real-time chain analytics that we can't replay. The result    ║
║    is a "scanner-only" backtest, which is what you want first    ║
║    anyway: prove the base strategy has edge, THEN evaluate AI.   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import csv as _csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

# Reuse the existing engine modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import (
    AngelClient, SignalGen, OptPicker, TA, INSTRUMENTS,
    PREMIUM_RANGES, fallback_delta, CONFIG, IST,
    estimate_costs, _master,
)
import pandas as pd
import numpy as np


# ─── Helpers ──────────────────────────────────────────────────────────

def estimate_option_premium(spot, strike, opt_type, dte, atr):
    """Rough Black-Scholes-free option premium estimate from spot, ATR, DTE.

    Used because Angel One's API does NOT expose historical option chains.
    Approximates the premium as `intrinsic + time_value` where:
      intrinsic = max(spot-strike, 0) for CE, max(strike-spot, 0) for PE
      time_value ~ 0.4 * ATR * sqrt(DTE) * delta_at_strike
    Calibrated to match observed NIFTY weekly ATM premiums in the ₹40-90
    range. Will misprice deep ITM/OTM strikes — fine for relative scoring,
    not fine for accurate P&L. Treat absolute rupee figures as ±20%.
    """
    if opt_type == "CE":
        intrinsic = max(spot - strike, 0)
    else:
        intrinsic = max(strike - spot, 0)
    moneyness = abs(strike - spot) / spot if spot else 0
    delta = fallback_delta(moneyness, dte=max(dte, 0.5),
                           right_side=(opt_type == "CE" and strike >= spot)
                                       or (opt_type == "PE" and strike <= spot))
    time_value = 0.4 * max(atr, 1) * (max(dte, 0.5) ** 0.5) * delta
    return round(max(intrinsic + time_value, 5), 2)


def simulate_trade(future_candles, opt_entry, opt_sl, opt_t1, opt_t2,
                   delta, sl_pts_index, t1_pts_index, t2_pts_index,
                   direction, max_minutes=120):
    """Walk forward through future 5-min candles. Exit when option price
    crosses SL or T1. Returns (result, exit_price_option, bars_held).

    Approximates option price at each future bar as:
      opt_price[i] = opt_entry + (index_close[i] - index_entry) * delta
    Capped at max_minutes (24 bars). If neither hit, exit at last bar."""
    if future_candles.empty:
        return "TIMEOUT", opt_entry, 0
    entry_idx_close = future_candles["close"].iloc[0]
    bars = min(len(future_candles), max_minutes // 5)
    for i in range(1, bars):
        idx_now = future_candles["close"].iloc[i]
        if direction == "LONG":
            opt_now = opt_entry + (idx_now - entry_idx_close) * delta
        else:
            opt_now = opt_entry + (entry_idx_close - idx_now) * delta
        if opt_now <= opt_sl:
            return "LOSS", round(opt_sl, 2), i
        if opt_now >= opt_t1:
            return "WIN", round(opt_t1, 2), i
    # Timeout — close at last bar's implied option price
    last_close = future_candles["close"].iloc[bars - 1]
    if direction == "LONG":
        opt_final = opt_entry + (last_close - entry_idx_close) * delta
    else:
        opt_final = opt_entry + (entry_idx_close - last_close) * delta
    if opt_final >= opt_entry:
        return "WIN_TIMEOUT", round(opt_final, 2), bars - 1
    return "LOSS_TIMEOUT", round(opt_final, 2), bars - 1


# ─── Main backtest loop ───────────────────────────────────────────────

def run_backtest(instrument: str, days: int, *, verbose: bool = False):
    """Replay the scanner over `days` of 5-min candles for one instrument."""
    inst = INSTRUMENTS.get(instrument)
    if not inst:
        print(f"Unknown instrument: {instrument}")
        return None

    print(f"\n═══ Backtest: {instrument} over last {days} days ═══")
    client = AngelClient()
    if not client.login():
        print("❌ Angel One login failed — check env vars")
        return None
    if not _master.ensure():
        print("⚠ Instrument master load failed — option premiums use fallback delta only")

    df = client.candles(inst["token"], inst["exchange"],
                        interval="FIVE_MINUTE", days=days, force_refresh=True)
    if df.empty:
        print("❌ No candles returned. Check market hours / API auth.")
        return None
    df = df.reset_index(drop=True)
    print(f"  Loaded {len(df)} candles ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]})")

    sgen = SignalGen()
    picker = OptPicker()
    trades = []
    last_signal_ts = None  # 15-min cooldown
    skipped_cooldown = 0
    skipped_no_signal = 0
    skipped_low_conf = 0

    # Walk forward bar by bar (need 30+ history bars, leave 24 future bars for trade simulation)
    for i in range(30, len(df) - 24):
        slice_df = df.iloc[: i + 1].copy()
        ts = slice_df["timestamp"].iloc[-1]
        # Cooldown gate
        if last_signal_ts is not None and (ts - last_signal_ts).total_seconds() < 900:
            skipped_cooldown += 1
            continue
        # Time gates: only between 09:20 and 14:50
        try:
            ts_py = pd.Timestamp(ts).to_pydatetime()
            if ts_py.tzinfo is not None:
                ts_py = ts_py.replace(tzinfo=None)
            hr, mn = ts_py.hour, ts_py.minute
        except Exception:
            continue
        if hr < 9 or (hr == 9 and mn < 20): continue
        if hr >= 15 or (hr == 14 and mn >= 50): continue

        sig = sgen.analyze(slice_df)
        if sig is None:
            skipped_no_signal += 1
            continue
        if sig["confidence"] < CONFIG.get("min_confidence", 45):
            skipped_low_conf += 1
            continue
        if sig.get("risk_reward", 0) < 1.5:
            continue

        # Pick a strike from the (estimated) chain
        spot = sig["price"]
        gap = inst["strike_gap"]
        atm = round(spot / gap) * gap
        ot = "CE" if sig["direction"] == "LONG" else "PE"
        strikes = [atm + j * gap for j in range(-3, 4)]
        # Estimate ATR from the slice — used for premium time-value
        atr_now = TA.atr(slice_df).iloc[-1]
        # Pick strike: ATM or 1 OTM
        chosen_strike = atm if ot == "CE" else atm
        # Use moneyness=0 fallback delta for ATM
        delta = fallback_delta(0, dte=5, right_side=True)
        opt_premium = estimate_option_premium(spot, chosen_strike, ot, dte=5, atr=atr_now)

        # Build entry / SL / T1 from index points scaled by delta
        idx_sl_pts = abs(sig["sl"] - sig["entry"])
        idx_t1_pts = abs(sig["target1"] - sig["entry"])
        idx_t2_pts = abs(sig["target2"] - sig["entry"])
        opt_sl = round(max(opt_premium - idx_sl_pts * delta, opt_premium * 0.65), 2)
        opt_t1 = round(opt_premium + idx_t1_pts * delta, 2)
        opt_t2 = round(opt_premium + idx_t2_pts * delta, 2)

        # Simulate forward
        future = df.iloc[i + 1: i + 25].reset_index(drop=True)
        result, exit_price, bars = simulate_trade(
            future, opt_premium, opt_sl, opt_t1, opt_t2,
            delta, idx_sl_pts, idx_t1_pts, idx_t2_pts,
            sig["direction"])

        # Sizing: assume 50% of budget, max 3 lots
        lot = inst["lot_size"]
        max_cap = CONFIG.get("budget", 20000) * 0.5
        cost_1 = opt_premium * lot
        lots = max(1, min(int(max_cap / cost_1), 3)) if cost_1 <= max_cap else 1
        qty = lots * lot
        gross_pnl = round((exit_price - opt_premium) * qty, 0)
        brokerage_rs, slippage_rs, _ = estimate_costs(opt_premium, exit_price, qty, lots)
        net_pnl = round(gross_pnl - brokerage_rs - slippage_rs, 0)

        trade = {
            "timestamp": str(ts),
            "instrument": instrument,
            "direction": sig["direction"],
            "confidence": sig["confidence"],
            "rr": sig.get("risk_reward", 0),
            "spot_entry": sig["entry"],
            "strike": chosen_strike,
            "type": ot,
            "opt_entry": opt_premium,
            "opt_exit": exit_price,
            "opt_sl": opt_sl, "opt_t1": opt_t1,
            "lots": lots, "qty": qty,
            "result": result,
            "bars_held": bars,
            "gross_pnl": gross_pnl,
            "brokerage_rs": brokerage_rs,
            "slippage_rs": slippage_rs,
            "net_pnl": net_pnl,
        }
        trades.append(trade)
        last_signal_ts = ts
        if verbose:
            print(f"  {ts}  {sig['direction']:<5}  conf={sig['confidence']}%  "
                  f"strike={chosen_strike}  result={result:<14}  net=₹{net_pnl}")

    return _summarise(instrument, trades, days, skipped_cooldown, skipped_no_signal, skipped_low_conf)


def _summarise(instrument, trades, days, skip_cool, skip_none, skip_low):
    if not trades:
        print("No trades generated.")
        return {"instrument": instrument, "trades": []}
    df = pd.DataFrame(trades)
    wins   = df[df["result"].isin(["WIN", "WIN_TIMEOUT"])]
    losses = df[df["result"].isin(["LOSS", "LOSS_TIMEOUT"])]
    total_gross = df["gross_pnl"].sum()
    total_net   = df["net_pnl"].sum()
    total_costs = df["brokerage_rs"].sum() + df["slippage_rs"].sum()
    win_rate = len(wins) / len(df) * 100 if len(df) else 0
    avg_win  = wins["net_pnl"].mean() if len(wins) else 0
    avg_loss = losses["net_pnl"].mean() if len(losses) else 0
    expectancy = (win_rate / 100) * avg_win + ((100 - win_rate) / 100) * avg_loss
    # Max drawdown on cumulative net P&L
    cum = df["net_pnl"].cumsum()
    running_max = cum.cummax()
    dd = (cum - running_max).min()

    print(f"\n══ Results: {instrument} ({days}d) ══")
    print(f"  Trades:        {len(df)}  (wins {len(wins)}  losses {len(losses)})")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Gross P&L:     ₹{total_gross:,.0f}")
    print(f"  Costs:         ₹{total_costs:,.0f}  (brokerage+slippage)")
    print(f"  Net P&L:       ₹{total_net:,.0f}")
    print(f"  Avg win:       ₹{avg_win:,.0f}")
    print(f"  Avg loss:      ₹{avg_loss:,.0f}")
    print(f"  Expectancy:    ₹{expectancy:,.0f} per trade (net)")
    print(f"  Max drawdown:  ₹{dd:,.0f}")
    print(f"  Skipped — cooldown: {skip_cool}  no-signal: {skip_none}  low-conf: {skip_low}")
    return {
        "instrument": instrument, "days": days,
        "trades_count": len(df),
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(win_rate, 1),
        "gross_pnl": round(float(total_gross), 0),
        "net_pnl":   round(float(total_net), 0),
        "total_costs": round(float(total_costs), 0),
        "expectancy": round(float(expectancy), 0),
        "max_drawdown": round(float(dd), 0),
        "skipped_cooldown": skip_cool,
        "skipped_no_signal": skip_none,
        "skipped_low_conf": skip_low,
        "trades": trades,
    }


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest the intraday signal engine.")
    parser.add_argument("--instrument", choices=list(INSTRUMENTS.keys()),
                        help="Single instrument to backtest")
    parser.add_argument("--all", action="store_true",
                        help="Run all three instruments (NIFTY, BANKNIFTY, FINNIFTY)")
    parser.add_argument("--days", type=int, default=30,
                        help="Lookback days (default 30; max ~90 due to Angel API)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Write per-trade CSV to this path")
    parser.add_argument("--json", type=str, default=None,
                        help="Write summary JSON to this path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print every trade as it's evaluated")
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Specify --instrument NIFTY (or --all)")

    instruments = list(INSTRUMENTS.keys()) if args.all else [args.instrument]
    all_results = []
    all_trades = []
    for inst in instruments:
        result = run_backtest(inst, args.days, verbose=args.verbose)
        if result:
            all_results.append(result)
            all_trades.extend(result.get("trades") or [])

    if args.csv and all_trades:
        with open(args.csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(all_trades[0].keys()))
            writer.writeheader()
            writer.writerows(all_trades)
        print(f"\n📄 Per-trade CSV → {args.csv}  ({len(all_trades)} rows)")

    if args.json:
        # Strip the heavy 'trades' lists from the per-instrument summaries; keep aggregate
        slim = [{k: v for k, v in r.items() if k != "trades"} for r in all_results]
        with open(args.json, "w") as f:
            json.dump({"runs": slim, "generated_at": datetime.now(IST).isoformat()}, f, indent=2)
        print(f"📄 Summary JSON → {args.json}")

    # Aggregate across instruments
    if len(all_results) > 1:
        total_trades = sum(r["trades_count"] for r in all_results)
        total_net    = sum(r["net_pnl"]      for r in all_results)
        total_wins   = sum(r["wins"]         for r in all_results)
        agg_winrate  = total_wins / total_trades * 100 if total_trades else 0
        print(f"\n═══ Aggregate ({len(all_results)} instruments, {args.days}d) ═══")
        print(f"  Total trades:  {total_trades}")
        print(f"  Net P&L:       ₹{total_net:,.0f}")
        print(f"  Win rate:      {agg_winrate:.1f}%")


if __name__ == "__main__":
    main()
