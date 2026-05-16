"""
╔══════════════════════════════════════════════════════════════════╗
║  backtest_v2.py — REAL-DATA BACKTEST                              ║
║                                                                  ║
║  Walks every 5-min spot bar over the last N trading days,         ║
║  runs SignalGenV2 on each, captures EVERY signal into one of      ║
║  4 buckets, and looks up REAL option premiums for entry/exit.    ║
║                                                                  ║
║  Buckets:                                                         ║
║    TAKEN_WIN    — engine would alert (passed all gates) + won     ║
║    TAKEN_LOSS   — engine would alert + lost                       ║
║    FILTERED_WIN — engine skipped (regime/etc.) but would have won ║
║    FILTERED_LOSS — engine skipped, would have lost                ║
║                                                                  ║
║  Per-trade CSV columns (the journal you asked for):               ║
║    date, time, instrument, direction, score,                      ║
║    strike, opt_type, expiry,                                      ║
║    opt_entry_price, opt_exit_price, exit_reason,                  ║
║    qty, gross_pnl, brokerage, slippage, net_pnl,                  ║
║    bucket, filter_reason, price_source,                           ║
║    rsi, vwap, vwap_dev_pct, ema20, ema50, atr, range_ratio        ║
║                                                                  ║
║  Every price has a `price_source` so you see what's real          ║
║  (angel / cache) vs estimated (bs_interpolated).                  ║
║                                                                  ║
║  USAGE                                                            ║
║    python3 backtest_v2.py --instrument NIFTY --days 10            ║
║    python3 backtest_v2.py --all --days 10 --csv journal.csv       ║
║    python3 backtest_v2.py --all --days 10 --html report.html      ║
║                                                                  ║
║  REQUIRES Angel One credentials in env (same as server.py).       ║
║  Pulls historical spot data via Angel; option premiums via the    ║
║  data_layer cascade.                                              ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import sys
import time as _time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IST = timezone(timedelta(hours=5, minutes=30))


# ─── Helpers ──────────────────────────────────────────────────────────

def _is_market_hours(ts: datetime) -> bool:
    """True if `ts` is within NSE intraday window (09:15-15:30 IST, Mon-Fri)."""
    if ts.weekday() >= 5:   # weekend
        return False
    hm = ts.hour * 100 + ts.minute
    return 915 <= hm <= 1530


def _next_expiry_after(ts: datetime, symbol: str = "NIFTY") -> date:
    """Find the next NIFTY weekly expiry (Tuesday) after `ts`.

    BANKNIFTY/FINNIFTY: monthly last-Tuesday only (post-Nov-2024 SEBI change).
    For backtest simplicity, we use the next Tuesday for both — most signals
    are 2-7 days from expiry which is the right zone for option buying.
    """
    d = ts.date() if isinstance(ts, datetime) else ts
    # NIFTY weekly: next Tuesday >= today
    days_ahead = (1 - d.weekday()) % 7   # Tuesday = 1 in Python's weekday
    if days_ahead == 0:
        # Today is Tuesday — if before market open, expiry is today; else next Tue
        if isinstance(ts, datetime) and ts.hour >= 15:
            days_ahead = 7
    return d + timedelta(days=days_ahead)


def _atm_strike(spot: float, gap: int) -> int:
    return round(spot / gap) * gap


def _opt_symbol(prefix: str, expiry: date, strike: int, opt_type: str) -> str:
    """Build the standard NSE NFO symbol: NIFTY07MAY26 25600 CE → 'NIFTY07MAY2625600CE'"""
    return f"{prefix}{expiry.strftime('%d%b%y').upper()}{strike}{opt_type}"


# ─── Trade simulation ─────────────────────────────────────────────────

@dataclass
class Trade:
    """One simulated round-trip trade. Maps 1:1 with one CSV row."""
    date:           str
    time:           str
    instrument:     str
    direction:      str
    score:          int
    strike:         int
    opt_type:       str
    expiry:         str
    opt_entry:      float
    opt_exit:       float
    exit_reason:    str
    bars_held:      int
    qty:            int
    gross_pnl:      float
    brokerage:      float
    slippage:       float
    net_pnl:        float
    bucket:         str
    filter_reason:  str   # empty for TAKEN_*; populated for FILTERED_*
    price_source:   str   # angel / cache / bs_interpolated / missing
    rsi:            float
    vwap:           float
    vwap_dev_pct:   float
    ema20:          float
    ema50:          float
    atr:            float
    range_ratio:    float


def _simulate_forward(spot_df, start_idx: int, opt_entry: float, opt_sl: float,
                       opt_t1: float, opt_t2: float, direction: str,
                       max_bars: int = 24,
                       data_layer_mod=None,
                       symbol: str = "", strike: int = 0, opt_type: str = "",
                       expiry: date = None, angel_client=None) -> tuple:
    """Walk forward through `spot_df` bars looking up REAL option premium at each
    timestamp. Exit when SL/T1/T2 hit or time runs out.

    Returns: (exit_price, exit_reason, bars_held, source_summary)
    """
    sources_used = []
    for i in range(1, max_bars + 1):
        if start_idx + i >= len(spot_df):
            break
        bar = spot_df.iloc[start_idx + i]
        bar_ts = bar["ts"] if "ts" in spot_df.columns else bar.name
        if isinstance(bar_ts, str):
            bar_ts = datetime.fromisoformat(bar_ts.replace("Z", "").split("+")[0])
        # Stop if past market close
        if isinstance(bar_ts, datetime) and (bar_ts.hour > 15 or (bar_ts.hour == 15 and bar_ts.minute >= 15)):
            # Force exit at end of day
            opt_now = data_layer_mod.get_option_premium_at(
                symbol, strike, opt_type, expiry, bar_ts, angel_client=angel_client
            )
            if opt_now:
                sources_used.append(opt_now.get("source", "?"))
                return opt_now["price"], "EOD_CLOSE", i, ",".join(sources_used)
            break

        opt_now = data_layer_mod.get_option_premium_at(
            symbol, strike, opt_type, expiry, bar_ts, angel_client=angel_client
        )
        if opt_now is None or opt_now.get("price") is None:
            continue
        sources_used.append(opt_now.get("source", "?"))
        price = float(opt_now["price"])

        # Exit checks — premium-percentage mode (matches OptPicker v2 logic)
        if price <= opt_sl:
            return opt_sl, "SL_HIT", i, ",".join(sources_used[-3:])
        if price >= opt_t2:
            return opt_t2, "T2_HIT", i, ",".join(sources_used[-3:])
        if price >= opt_t1:
            return opt_t1, "T1_HIT", i, ",".join(sources_used[-3:])

    # Ran out of bars → exit at last observed price (or original if no data)
    return opt_entry, "TIMEOUT", max_bars, ",".join(sources_used[-3:]) if sources_used else "none"


# ─── Main backtest loop ───────────────────────────────────────────────

def run_backtest(symbol: str, from_date: date, to_date: date,
                 budget: float = 50000.0,
                 verbose: bool = False) -> list[Trade]:
    """Replay the v2 strategy + regime filter over the date range.

    For each 5-min spot bar:
      1. Run SignalGenV2.analyze(rolling_window)
      2. If signal fires:
         a. Apply RegimeFilter to determine TAKEN or FILTERED + reason
         b. Determine ATM strike, expiry
         c. Look up REAL option entry price via data_layer
         d. Compute SL/T1/T2 via premium-pct mode (same as live engine)
         e. Walk forward, look up REAL option price each bar, exit at SL/T1/T2/EOD
         f. Classify into bucket: TAKEN_WIN / TAKEN_LOSS / FILTERED_WIN / FILTERED_LOSS
         g. Record Trade dataclass row

    Returns: list of Trade objects.
    """
    import pandas as pd
    from server import AngelClient, INSTRUMENTS, CONFIG, estimate_costs
    from signal_v2 import SignalGenV2
    from regime import RegimeFilter
    import data_layer

    inst = INSTRUMENTS.get(symbol.upper())
    if not inst:
        print(f"❌ Unknown instrument: {symbol}")
        return []
    gap = inst["strike_gap"]
    lot = inst["lot_size"]
    prefix = inst["expiry_prefix"]

    print(f"\n══ Backtest: {symbol}  {from_date} → {to_date} ══")

    # Boot Angel client for spot + option data
    print("  Logging into Angel One ...")
    client = AngelClient()
    if not client.login():
        print(f"  ✗ Angel login failed: {client.last_login_error}")
        return []
    print("  ✓ Logged in")

    # Pull spot bars for the full range (data_layer caches)
    print(f"  Fetching spot 5-min bars for {symbol} ...")
    from_dt = datetime.combine(from_date, datetime.min.time()).replace(hour=9, minute=15)
    to_dt   = datetime.combine(to_date,   datetime.min.time()).replace(hour=15, minute=30)
    spot_df = data_layer.get_spot_bars(symbol.upper(), from_dt, to_dt,
                                       interval="5min", angel_client=client)
    if spot_df.empty:
        print("  ✗ No spot bars returned")
        return []
    if "ts" not in spot_df.columns and "timestamp" in spot_df.columns:
        spot_df = spot_df.rename(columns={"timestamp": "ts"})
    spot_df = spot_df.sort_values("ts").reset_index(drop=True)
    spot_df["ts"] = pd.to_datetime(spot_df["ts"])
    print(f"  ✓ Got {len(spot_df)} spot bars")

    # Premium-pct exit levels (matches OptPicker v2 step 14)
    sl_pct = float(CONFIG.get("opt_sl_pct", 0.35))
    t1_pct = float(CONFIG.get("opt_t1_pct", 0.50))
    t2_pct = float(CONFIG.get("opt_t2_pct", 1.00))

    trades: list[Trade] = []
    last_taken_ts = None     # 15-min cooldown only applies to TAKEN

    # Walk every bar (need at least 30 bars of history for analyze)
    n_total = len(spot_df)
    n_scanned = 0
    n_signals = 0
    n_processed = 0

    for i in range(30, n_total - 1):
        bar = spot_df.iloc[i]
        ts = bar["ts"]
        if not isinstance(ts, datetime):
            try: ts = ts.to_pydatetime()
            except Exception: continue
        # Only process bars during market hours
        if not _is_market_hours(ts):
            continue
        # No new entries after 14:50 (matches engine hard gate)
        if ts.hour > 14 or (ts.hour == 14 and ts.minute >= 50):
            continue

        n_scanned += 1

        # Pass rolling window (last 60 bars) to analyzer
        window = spot_df.iloc[max(0, i - 59):i + 1].copy()
        if len(window) < 30:
            continue
        sig = SignalGenV2.analyze(window)
        if sig is None:
            continue
        n_signals += 1

        # Apply regime filter — determines TAKEN vs FILTERED
        ok, reason = RegimeFilter.should_trade(angel_client=None, symbol=symbol.upper(), now=ts)

        # Cooldown — only blocks TAKEN
        if last_taken_ts is not None and (ts - last_taken_ts).total_seconds() < 900:
            if ok:
                ok = False
                reason = "COOLDOWN_15MIN"

        would_be_taken = ok

        # Pick strike (ATM) + expiry
        spot = float(bar["close"])
        strike = _atm_strike(spot, gap)
        opt_type = "CE" if sig["direction"] == "LONG" else "PE"
        expiry = _next_expiry_after(ts, symbol.upper())

        # Look up REAL entry premium
        entry_data = data_layer.get_option_premium_at(
            symbol.upper(), strike, opt_type, expiry, ts, angel_client=client
        )
        if entry_data is None or entry_data.get("price") is None or entry_data.get("price") <= 0:
            # Can't price this trade — skip
            if verbose:
                print(f"    {ts}  {sig['direction']} no entry price → skip")
            continue
        opt_entry = float(entry_data["price"])
        if opt_entry < 5:   # too cheap, skip
            continue
        entry_source = entry_data.get("source", "?")

        # Premium-pct exits (real prices, matches v2 OptPicker)
        opt_sl = round(opt_entry * (1 - sl_pct), 2)
        opt_t1 = round(opt_entry * (1 + t1_pct), 2)
        opt_t2 = round(opt_entry * (1 + t2_pct), 2)

        # Walk forward — pull REAL option price each bar, exit at SL/T1/T2/EOD
        exit_price, exit_reason, bars_held, exit_source = _simulate_forward(
            spot_df, i, opt_entry, opt_sl, opt_t1, opt_t2, sig["direction"],
            max_bars=24, data_layer_mod=data_layer,
            symbol=symbol.upper(), strike=strike, opt_type=opt_type,
            expiry=expiry, angel_client=client,
        )

        # Position size (matches v1 OptPicker: 50% of budget, max 3 lots)
        cost_1lot = opt_entry * lot
        max_cap = budget * 0.5
        lots = max(1, min(int(max_cap / cost_1lot), 3)) if cost_1lot <= max_cap else 1
        qty = lots * lot
        gross_pnl = round((exit_price - opt_entry) * qty, 0)
        brokerage_rs, slippage_rs, _ = estimate_costs(opt_entry, exit_price, qty, lots)
        net_pnl = round(gross_pnl - brokerage_rs - slippage_rs, 0)

        # Classify
        won = net_pnl > 0
        if would_be_taken and won:        bucket = "TAKEN_WIN"
        elif would_be_taken and not won:  bucket = "TAKEN_LOSS"
        elif not would_be_taken and won:  bucket = "FILTERED_WIN"   # ← missed opportunity
        else:                              bucket = "FILTERED_LOSS"

        if would_be_taken:
            last_taken_ts = ts

        ind = sig.get("indicators", {})
        trade = Trade(
            date=ts.strftime("%Y-%m-%d"),
            time=ts.strftime("%H:%M:%S"),
            instrument=symbol.upper(),
            direction=sig["direction"],
            score=sig.get("v2_score", 0),
            strike=strike,
            opt_type=opt_type,
            expiry=expiry.strftime("%Y-%m-%d"),
            opt_entry=opt_entry,
            opt_exit=exit_price,
            exit_reason=exit_reason,
            bars_held=bars_held,
            qty=qty,
            gross_pnl=gross_pnl,
            brokerage=brokerage_rs,
            slippage=slippage_rs,
            net_pnl=net_pnl,
            bucket=bucket,
            filter_reason="" if would_be_taken else reason,
            price_source=f"{entry_source}→{exit_source}",
            rsi=ind.get("rsi", 0),
            vwap=ind.get("vwap", 0),
            vwap_dev_pct=ind.get("vwap_dev_pct", 0),
            ema20=ind.get("ema20", 0),
            ema50=ind.get("ema50", 0),
            atr=ind.get("atr", 0),
            range_ratio=ind.get("range_ratio", 0),
        )
        trades.append(trade)
        n_processed += 1

        if verbose:
            marker = {"TAKEN_WIN":"🟢","TAKEN_LOSS":"🔴","FILTERED_WIN":"🟡","FILTERED_LOSS":"⚪"}[bucket]
            print(f"  {marker} {trade.date} {trade.time}  {symbol}  {sig['direction']:<5}  "
                  f"{strike}{opt_type}  ₹{opt_entry:.1f}→₹{exit_price:.1f}  "
                  f"[{exit_reason}]  net=₹{net_pnl:+.0f}  bucket={bucket}  "
                  f"src={trade.price_source}")

    print(f"  ✓ Scanned {n_scanned} bars · {n_signals} v2 signals fired · {n_processed} priced trades")
    return trades


# ─── Reporting ────────────────────────────────────────────────────────

def summarise(trades: list[Trade]):
    if not trades:
        print("\n⚠ No trades to summarise.")
        return

    n = len(trades)
    taken_w = [t for t in trades if t.bucket == "TAKEN_WIN"]
    taken_l = [t for t in trades if t.bucket == "TAKEN_LOSS"]
    filt_w  = [t for t in trades if t.bucket == "FILTERED_WIN"]
    filt_l  = [t for t in trades if t.bucket == "FILTERED_LOSS"]

    taken_n = len(taken_w) + len(taken_l)
    taken_net = sum(t.net_pnl for t in taken_w + taken_l)
    taken_wr  = (len(taken_w) / taken_n * 100) if taken_n else 0
    avg_win   = (sum(t.net_pnl for t in taken_w) / len(taken_w)) if taken_w else 0
    avg_loss  = (sum(t.net_pnl for t in taken_l) / len(taken_l)) if taken_l else 0
    expectancy = (taken_wr/100) * avg_win + ((100-taken_wr)/100) * avg_loss

    # Max drawdown on TAKEN cumulative P&L
    taken_ordered = sorted(taken_w + taken_l, key=lambda t: (t.date, t.time))
    cum = 0.0; peak = 0.0; dd = 0.0
    for t in taken_ordered:
        cum += t.net_pnl
        peak = max(peak, cum)
        dd = min(dd, cum - peak)

    missed_net = sum(t.net_pnl for t in filt_w)
    saved_net  = abs(sum(t.net_pnl for t in filt_l))

    print("\n" + "═" * 70)
    print(f"  BACKTEST SUMMARY  ({n} signals total)")
    print("═" * 70)

    print(f"\n  ┌─ TAKEN (engine would alert) ─────────────────────────────┐")
    print(f"  │  Trades:     {taken_n:4d}  win {len(taken_w):3d}  loss {len(taken_l):3d}  win-rate {taken_wr:5.1f}%  │")
    print(f"  │  Net P&L:    ₹{taken_net:>+10,.0f}                                    │")
    print(f"  │  Avg win:    ₹{avg_win:>+10,.0f}   Avg loss: ₹{avg_loss:>+10,.0f}    │")
    print(f"  │  Expectancy: ₹{expectancy:>+10,.0f} per trade (net of costs)        │")
    print(f"  │  Max DD:     ₹{dd:>+10,.0f}                                     │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ FILTERED — MISSED OPPORTUNITIES (would have won) ─────────┐")
    print(f"  │  Count:     {len(filt_w):4d}                                            │")
    print(f"  │  P&L missed: ₹{missed_net:>+10,.0f}   ← this is the cost of filters  │")
    if filt_w:
        # Per-filter breakdown
        from collections import Counter
        cnt = Counter(t.filter_reason for t in filt_w)
        for reason, c in cnt.most_common(5):
            sub = sum(t.net_pnl for t in filt_w if t.filter_reason == reason)
            print(f"  │    {reason[:35]:<35} {c:>3d} × ₹{sub:>+8,.0f}    │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    print(f"\n  ┌─ FILTERED — CORRECT REJECTS (would have lost) ────────────┐")
    print(f"  │  Count:      {len(filt_l):4d}                                            │")
    print(f"  │  Loss avoided: ₹{saved_net:>10,.0f}                                  │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    net_filter = saved_net - missed_net
    verdict = "HELP" if net_filter > 0 else "HURT"
    print(f"\n  ► Filters net value: {'+' if net_filter >= 0 else ''}₹{net_filter:,.0f}  → filters currently {verdict}")

    # Price-source breakdown — REAL vs ESTIMATED
    from collections import Counter
    src_counter = Counter()
    for t in trades:
        # price_source format: "entry_src→exit_src" — count both halves separately
        for src in (t.price_source.split("→") if t.price_source else ()):
            for sub_src in src.split(","):
                sub_src = sub_src.strip().split(":")[-1]
                src_counter[sub_src] += 1
    real = sum(c for s, c in src_counter.items() if s in ("angel", "jugaad"))
    interp = src_counter.get("bs_interpolated", 0)
    total_src = sum(src_counter.values()) or 1
    real_pct = real / total_src * 100
    print(f"\n  ► Price data integrity: {real_pct:.1f}% real exchange data · "
          f"{interp/total_src*100:.1f}% BS-interpolated")
    if real_pct < 70:
        print(f"    ⚠ Less than 70% real prices — interpret results with caution")


def write_csv(trades: list[Trade], path: str):
    if not trades:
        print(f"  ⚠ No trades to write")
        return
    with open(path, "w", newline="") as f:
        fieldnames = list(trades[0].__dict__.keys())
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in trades:
            w.writerow(t.__dict__)
    print(f"\n  📄 Per-trade journal → {path}  ({len(trades)} rows)")


def write_html(trades: list[Trade], path: str):
    """Generate a simple HTML report with equity curve + per-trade table.
    Uses matplotlib only if available, else falls back to a plain table."""
    if not trades:
        return
    # Cumulative TAKEN P&L
    taken = [t for t in trades if t.bucket in ("TAKEN_WIN", "TAKEN_LOSS")]
    taken_ordered = sorted(taken, key=lambda t: (t.date, t.time))
    cum = []
    running = 0.0
    for t in taken_ordered:
        running += t.net_pnl
        cum.append((f"{t.date} {t.time}", running))

    img_html = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if cum:
            xs = list(range(len(cum)))
            ys = [c[1] for c in cum]
            plt.figure(figsize=(10, 4))
            plt.plot(xs, ys, color="#0a84ff", linewidth=2)
            plt.fill_between(xs, ys, 0, alpha=0.15, color="#0a84ff")
            plt.axhline(0, color="#888", linestyle="--", alpha=0.5)
            plt.title("Cumulative TAKEN net P&L over backtest period")
            plt.xlabel("Trade #"); plt.ylabel("Cumulative ₹")
            plt.tight_layout()
            img_path = path.replace(".html", "_equity.png")
            plt.savefig(img_path)
            img_html = f"<img src='{os.path.basename(img_path)}' style='max-width:100%; border-radius:8px'/>"
    except Exception as e:
        img_html = f"<p>Equity curve unavailable: {e}</p>"

    # Per-trade table
    rows_html = []
    for t in trades:
        cls = {"TAKEN_WIN":"win", "TAKEN_LOSS":"loss",
               "FILTERED_WIN":"miss", "FILTERED_LOSS":"avoid"}.get(t.bucket, "")
        rows_html.append(
            f"<tr class='{cls}'>"
            f"<td>{t.date}</td><td>{t.time}</td><td>{t.instrument}</td>"
            f"<td class='{'long' if t.direction=='LONG' else 'short'}'>{t.direction}</td>"
            f"<td>{t.score}/4</td><td>{t.strike}{t.opt_type}</td>"
            f"<td>{t.expiry}</td><td>₹{t.opt_entry:.1f}</td><td>₹{t.opt_exit:.1f}</td>"
            f"<td>{t.exit_reason}</td>"
            f"<td class='{'pos' if t.net_pnl>=0 else 'neg'}'>₹{t.net_pnl:+,.0f}</td>"
            f"<td><span class='chip {cls}'>{t.bucket}</span></td>"
            f"<td>{t.filter_reason}</td><td style='font-size:10px'>{t.price_source}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Backtest v2 — Journal</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #fafafa; }}
  h1 {{ font-size: 22px; margin-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: 8px 10px; text-align: left; font-size: 12.5px; border-bottom: 1px solid #eee; }}
  th {{ background: #f4f5f7; font-weight: 600; font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: #666; }}
  tr.win td   {{ background: #f0fff4; }}
  tr.loss td  {{ background: #fff5f5; }}
  tr.miss td  {{ background: #fffbe6; }}
  tr.avoid td {{ background: #f5f5f5; }}
  .pos {{ color: #1ea54d; font-weight: 600; font-family: monospace; }}
  .neg {{ color: #c0392b; font-weight: 600; font-family: monospace; }}
  .long  {{ color: #1ea54d; font-weight: 600; }}
  .short {{ color: #c0392b; font-weight: 600; }}
  .chip {{ padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
  .chip.win   {{ background: #1ea54d20; color: #1ea54d; }}
  .chip.loss  {{ background: #c0392b20; color: #c0392b; }}
  .chip.miss  {{ background: #ffaf0030; color: #c47900; }}
  .chip.avoid {{ background: #00000010; color: #666; }}
  .summary {{ background: #fff; padding: 16px; border-radius: 6px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
</style>
</head>
<body>
<h1>📈 Backtest v2 — Per-Trade Journal</h1>
<p style="color:#666">Generated {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')} · {len(trades)} signals · v2.1 strategy + regime filter</p>
<div class="summary">{img_html}</div>
<table>
<thead><tr>
  <th>Date</th><th>Time</th><th>Inst</th><th>Dir</th><th>Score</th>
  <th>Option</th><th>Expiry</th><th>Entry</th><th>Exit</th><th>Reason</th>
  <th>Net P&amp;L</th><th>Bucket</th><th>Filter</th><th>Source</th>
</tr></thead>
<tbody>{''.join(rows_html)}</tbody>
</table>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    print(f"  📄 HTML report → {path}")


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-data backtest for v2 strategy. Produces full per-trade journal.")
    parser.add_argument("--instrument", choices=["NIFTY", "BANKNIFTY", "FINNIFTY"],
                        help="Single instrument to backtest")
    parser.add_argument("--all", action="store_true",
                        help="Run all three indices")
    parser.add_argument("--days", type=int, default=10,
                        help="Lookback days (default 10; Angel API limits ~30)")
    parser.add_argument("--from-date", type=str, default=None,
                        help="Override: YYYY-MM-DD start date")
    parser.add_argument("--to-date", type=str, default=None,
                        help="Override: YYYY-MM-DD end date")
    parser.add_argument("--budget", type=float, default=50000.0,
                        help="Account size in ₹ for position sizing (default 50000)")
    parser.add_argument("--csv", type=str, default="backtest_journal.csv",
                        help="CSV output path (default backtest_journal.csv)")
    parser.add_argument("--html", type=str, default=None,
                        help="Optional HTML report path")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print every trade as it runs")
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Specify --instrument NIFTY (or --all)")

    # Date range
    if args.from_date and args.to_date:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
        to_date   = datetime.strptime(args.to_date,   "%Y-%m-%d").date()
    else:
        to_date = datetime.now(IST).date()
        from_date = to_date - timedelta(days=args.days)

    instruments = ["NIFTY", "BANKNIFTY", "FINNIFTY"] if args.all else [args.instrument]

    all_trades: list[Trade] = []
    for sym in instruments:
        trades = run_backtest(sym, from_date, to_date,
                              budget=args.budget, verbose=args.verbose)
        all_trades.extend(trades)

    if not all_trades:
        print("\n⚠ Zero trades generated across the requested period.")
        print("  Common reasons:")
        print("    - Wrong date range (weekends/holidays)")
        print("    - v2 strategy too strict for the period (check /api/v2-diag)")
        print("    - Angel API didn't return enough spot bars")
        return

    # Sort chronologically for journal readability
    all_trades.sort(key=lambda t: (t.date, t.time, t.instrument))

    summarise(all_trades)
    write_csv(all_trades, args.csv)
    if args.html:
        write_html(all_trades, args.html)


if __name__ == "__main__":
    main()
