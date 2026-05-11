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
    last_taken_ts = None  # cooldown applies ONLY to TAKEN trades, not filtered ones

    # Walk forward bar by bar (need 30+ history bars, leave 24 future bars for trade simulation)
    for i in range(30, len(df) - 24):
        slice_df = df.iloc[: i + 1].copy()
        ts = slice_df["timestamp"].iloc[-1]
        # Time gates apply to BOTH paths
        try:
            ts_py = pd.Timestamp(ts).to_pydatetime()
            if ts_py.tzinfo is not None:
                ts_py = ts_py.replace(tzinfo=None)
            hr, mn = ts_py.hour, ts_py.minute
        except Exception:
            continue
        if hr < 9 or (hr == 9 and mn < 20): continue

        sig = sgen.analyze(slice_df)
        if sig is None:
            # The analyzer returns None for late-day, low-data, or blocked-window bars.
            # No signal = no missed opportunity to evaluate.
            continue

        # ── Classify against the engine's filter chain ─────────────────────
        filter_reasons = []
        if hr >= 15 or (hr == 14 and mn >= 50):
            filter_reasons.append("LATE_DAY")
        if sig["confidence"] < CONFIG.get("min_confidence", 45):
            filter_reasons.append("LOW_CONFIDENCE")
        if sig.get("risk_reward", 0) < 1.5:
            filter_reasons.append("LOW_RR")
        # Cooldown only counts against the TAKEN path. If the engine WOULD take this
        # but the cooldown is active, it's a different kind of "missed" — we record it
        # but mark it specially.
        cooldown_active = (last_taken_ts is not None and
                           (ts - last_taken_ts).total_seconds() < 900)
        if cooldown_active and not filter_reasons:
            filter_reasons.append("COOLDOWN")

        would_be_taken = (len(filter_reasons) == 0)

        # ── Forward simulation runs EITHER WAY ─────────────────────────────
        spot = sig["price"]
        gap = inst["strike_gap"]
        atm = round(spot / gap) * gap
        ot = "CE" if sig["direction"] == "LONG" else "PE"
        atr_now = TA.atr(slice_df).iloc[-1]
        chosen_strike = atm
        delta = fallback_delta(0, dte=5, right_side=True)
        opt_premium = estimate_option_premium(spot, chosen_strike, ot, dte=5, atr=atr_now)

        # Build SL/T1 using the SAME premium-pct mode the engine now defaults to (step 14)
        exit_mode = CONFIG.get("opt_exit_mode", "premium_pct")
        if exit_mode == "premium_pct":
            opt_sl = round(opt_premium * (1 - float(CONFIG.get("opt_sl_pct", 0.35))), 2)
            opt_t1 = round(opt_premium * (1 + float(CONFIG.get("opt_t1_pct", 0.50))), 2)
            opt_t2 = round(opt_premium * (1 + float(CONFIG.get("opt_t2_pct", 1.00))), 2)
        else:
            idx_sl_pts = abs(sig["sl"] - sig["entry"])
            idx_t1_pts = abs(sig["target1"] - sig["entry"])
            idx_t2_pts = abs(sig["target2"] - sig["entry"])
            opt_sl = round(max(opt_premium - idx_sl_pts * delta, opt_premium * 0.65), 2)
            opt_t1 = round(opt_premium + idx_t1_pts * delta, 2)
            opt_t2 = round(opt_premium + idx_t2_pts * delta, 2)

        idx_sl_pts = abs(sig["sl"] - sig["entry"])
        idx_t1_pts = abs(sig["target1"] - sig["entry"])
        idx_t2_pts = abs(sig["target2"] - sig["entry"])

        future = df.iloc[i + 1: i + 25].reset_index(drop=True)
        result, exit_price, bars = simulate_trade(
            future, opt_premium, opt_sl, opt_t1, opt_t2,
            delta, idx_sl_pts, idx_t1_pts, idx_t2_pts,
            sig["direction"])
        won = result in ("WIN", "WIN_TIMEOUT")

        # Sizing
        lot = inst["lot_size"]
        max_cap = CONFIG.get("budget", 20000) * 0.5
        cost_1 = opt_premium * lot
        lots = max(1, min(int(max_cap / cost_1), 3)) if cost_1 <= max_cap else 1
        qty = lots * lot
        gross_pnl = round((exit_price - opt_premium) * qty, 0)
        brokerage_rs, slippage_rs, _ = estimate_costs(opt_premium, exit_price, qty, lots)
        net_pnl = round(gross_pnl - brokerage_rs - slippage_rs, 0)

        # ── 4-bucket classification ─────────────────────────────────────────
        if would_be_taken and won:        bucket = "TAKEN_WIN"
        elif would_be_taken and not won:  bucket = "TAKEN_LOSS"
        elif not would_be_taken and won:  bucket = "FILTERED_WIN"   # ← missed opportunity
        else:                              bucket = "FILTERED_LOSS"  # ← filter worked

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
            "bucket": bucket,
            "would_be_taken": would_be_taken,
            "filtered_by": ",".join(filter_reasons) if filter_reasons else None,
        }
        trades.append(trade)
        if would_be_taken:
            last_taken_ts = ts
        if verbose:
            tag = bucket.ljust(13)
            print(f"  {ts}  {sig['direction']:<5}  conf={sig['confidence']}%  {tag}  net=₹{net_pnl}")

    return _summarise(instrument, trades, days)


def _summarise(instrument, trades, days):
    if not trades:
        print("No signals generated.")
        return {"instrument": instrument, "trades": []}
    df = pd.DataFrame(trades)

    taken    = df[df["would_be_taken"] == True]
    filtered = df[df["would_be_taken"] == False]
    taken_w  = df[df["bucket"] == "TAKEN_WIN"]
    taken_l  = df[df["bucket"] == "TAKEN_LOSS"]
    filt_w   = df[df["bucket"] == "FILTERED_WIN"]    # missed opportunities
    filt_l   = df[df["bucket"] == "FILTERED_LOSS"]   # filters that worked

    print(f"\n══ Backtest: {instrument} ({days}d) ══")
    print(f"  Total signals scanned: {len(df)}")
    print()

    # ── TAKEN path: what the engine would have alerted ──
    taken_net = float(taken["net_pnl"].sum()) if len(taken) else 0
    taken_wr  = len(taken_w) / max(len(taken), 1) * 100
    print(f"══ TAKEN (engine would alert) ══")
    print(f"  Trades:     {len(taken)}  win {len(taken_w)}  loss {len(taken_l)}  "
          f"win-rate {taken_wr:.1f}%")
    print(f"  Net P&L:    ₹{taken_net:,.0f}")
    if len(taken):
        avg_w = taken_w["net_pnl"].mean() if len(taken_w) else 0
        avg_l = taken_l["net_pnl"].mean() if len(taken_l) else 0
        expectancy = (taken_wr / 100) * avg_w + ((100 - taken_wr) / 100) * avg_l
        cum = taken["net_pnl"].cumsum()
        dd  = (cum - cum.cummax()).min() if len(cum) else 0
        print(f"  Avg win:    ₹{avg_w:,.0f}    Avg loss: ₹{avg_l:,.0f}")
        print(f"  Expectancy: ₹{expectancy:,.0f}/trade")
        print(f"  Max DD:     ₹{dd:,.0f}")
    print()

    # ── FILTERED — MISSED OPPORTUNITIES ──
    missed_net = float(filt_w["net_pnl"].sum()) if len(filt_w) else 0
    print(f"══ FILTERED — MISSED OPPORTUNITIES ══")
    print(f"  Trades engine SKIPPED that WOULD HAVE WON: {len(filt_w)}")
    print(f"  P&L missed:  ₹{missed_net:,.0f}  (this is the cost of your filters)")
    if len(filt_w):
        print(f"  Top filters causing misses:")
        for reason, count in filt_w["filtered_by"].value_counts().head(5).items():
            sub_net = float(filt_w[filt_w["filtered_by"] == reason]["net_pnl"].sum())
            print(f"    {reason:<35} {count:3d} winners filtered · ₹{sub_net:,.0f} missed")
    print()

    # ── FILTERED — CORRECT REJECTS ──
    saved_net = abs(float(filt_l["net_pnl"].sum())) if len(filt_l) else 0
    print(f"══ FILTERED — CORRECT REJECTS ══")
    print(f"  Trades engine SKIPPED that WOULD HAVE LOST: {len(filt_l)}")
    print(f"  Loss avoided: ₹{saved_net:,.0f}")
    print()

    # ── NET FILTER VALUE ──
    net_filter = saved_net - missed_net
    print(f"══ NET FILTER VALUE ══")
    print(f"  Filters saved   ₹{saved_net:,.0f} on bad trades")
    print(f"  Filters cost    ₹{missed_net:,.0f} on missed winners")
    print(f"  Net filter:     {'+' if net_filter >= 0 else ''}₹{net_filter:,.0f}  "
          f"({'filters HELP' if net_filter > 0 else 'filters HURT'})")

    return {
        "instrument": instrument, "days": days,
        "trades_scanned":  len(df),
        "taken":           {"count": len(taken),   "wins": len(taken_w),
                            "losses": len(taken_l), "net_pnl": round(taken_net, 0),
                            "win_rate_pct": round(taken_wr, 1)},
        "missed_winners":  {"count": len(filt_w),  "net_pnl_missed": round(missed_net, 0),
                            "by_filter": filt_w["filtered_by"].value_counts().to_dict() if len(filt_w) else {}},
        "correct_rejects": {"count": len(filt_l),  "loss_avoided": round(saved_net, 0)},
        "net_filter_value": round(net_filter, 0),
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
                        help="Print every signal evaluated, with its bucket")
    parser.add_argument("--show-missed", action="store_true",
                        help="After the summary, list the top 10 'missed opportunity' trades "
                             "(signals the engine filtered that would have won) with their "
                             "timestamps + the specific filter that blocked them")
    parser.add_argument("--top-n", type=int, default=10,
                        help="How many missed opportunities to show with --show-missed (default 10)")
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

    # ── Show missed opportunities (step 16 — the headline feature) ──
    if args.show_missed and all_trades:
        missed = [t for t in all_trades if t.get("bucket") == "FILTERED_WIN"]
        missed.sort(key=lambda t: -(t.get("net_pnl") or 0))
        if not missed:
            print(f"\n🎯 No missed opportunities found — your filters caught all winners.")
        else:
            top = missed[:args.top_n]
            print(f"\n🎯 Top {len(top)} missed opportunities (winners the engine filtered out)")
            print(f"   {'─' * 100}")
            for t in top:
                print(f"   {t['timestamp']}  {t['instrument']:<10} {t['direction']:<5}  "
                      f"conf={t['confidence']:>2}%  rr={t.get('rr',0):.1f}  "
                      f"filtered_by={t['filtered_by']:<25}  would-have-won ₹{t.get('net_pnl',0):,.0f}")

    # ── Aggregate across instruments ──
    if len(all_results) > 1:
        total_scanned = sum(r["trades_scanned"] for r in all_results)
        total_taken_net = sum(r["taken"]["net_pnl"] for r in all_results)
        total_taken     = sum(r["taken"]["count"]    for r in all_results)
        total_taken_w   = sum(r["taken"]["wins"]     for r in all_results)
        total_missed    = sum(r["missed_winners"]["count"] for r in all_results)
        total_missed_net= sum(r["missed_winners"]["net_pnl_missed"] for r in all_results)
        total_saved     = sum(r["correct_rejects"]["loss_avoided"]  for r in all_results)
        agg_winrate     = total_taken_w / total_taken * 100 if total_taken else 0
        print(f"\n═══ Aggregate ({len(all_results)} instruments, {args.days}d) ═══")
        print(f"  Signals scanned:   {total_scanned}")
        print(f"  Trades taken:      {total_taken}  win-rate {agg_winrate:.1f}%  net ₹{total_taken_net:,.0f}")
        print(f"  Missed winners:    {total_missed}  P&L missed ₹{total_missed_net:,.0f}")
        print(f"  Losses avoided:    ₹{total_saved:,.0f}")
        print(f"  Net filter value:  ₹{total_saved - total_missed_net:,.0f}")


if __name__ == "__main__":
    main()
