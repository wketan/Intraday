"""Opening Range Breakout (ORB-15) strategy.

Marks the high/low of the first 15 minutes of the session (9:15-9:30 IST,
which is the first 3 bars on a 5-min timeframe) and trades a breakout in
the direction of the move with the opposite boundary as stop.

Why this strategy:
- Published 8.5-year NIFTY backtest (Intraday Lab, n=2,122 trades):
  win rate 48.7%, profit factor 1.23. Strongest documented retail edge.
- Mechanism: retail + institutional orders concentrate at session open,
  creating a measurable range whose break tends to follow through.
- Simple, parameter-light: only one knob is the minimum ORB range filter
  to avoid false-breakout days when the open is uninteresting.

Compared to v2:
- v2 uses 6 redundant trend indicators (all measure the same thing).
- ORB uses ONE structural level (the morning range) and ONE confirmation
  (volume / range-expansion on the breakout bar). Orthogonal signals.

Interface mirrors SignalGenV2: stateless analyze(df) returns either a
signal dict or None, with diagnostics in `last_decision`.

ORB enforces ONE TRADE PER DAY PER INSTRUMENT at the engine/backtest
layer (the one-shot gate is not in the analyzer — analyze() can fire
the same setup repeatedly until cooldown / day-cap blocks it).
"""

from __future__ import annotations

import os
from datetime import datetime, time, timezone, timedelta
from typing import Optional

try:
    import pandas as pd
except ImportError:
    pd = None

# ── IST timezone (mirrors signal_v2) ────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


def _env_float(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)))
    except: return default

def _env_int(key: str, default: int) -> int:
    try: return int(os.environ.get(key, str(default)))
    except: return default


class SignalGenORB:
    """Opening Range Breakout v1. Stateless analyzer."""

    last_decision: dict = {}

    @staticmethod
    def _config() -> dict:
        """All knobs in one place. Env-tunable; sensible defaults below."""
        return {
            # ORB window: first N bars of the session form the range.
            # 3 bars × 5min = 15min (9:15-9:30). Standard ORB-15.
            "orb_bars":          _env_int  ("ORB_BARS",            3),

            # Minimum ORB range in spot points to bother trading.
            # Tight ranges (<NIFTY 30 pts) tend to produce false breakouts
            # because the open was indecisive. Filter cuts low-edge days.
            "min_orb_range_pts": _env_float("ORB_MIN_RANGE_PTS",  30.0),

            # Breakout confirmation: current bar's range must expand vs
            # day's average bar range so far. Acts as a volume proxy on
            # indices (where volume data is unreliable). 1.2× is gentle —
            # not a strict filter, just kicks out lazy breakouts.
            "breakout_range_mult": _env_float("ORB_BREAKOUT_RANGE_MULT", 1.2),

            # Earliest time to consider a breakout (after ORB completes).
            # Defaults to 09:30 IST = 9 hours + 30 min from midnight.
            "earliest_h":        _env_int  ("ORB_EARLIEST_H",     9),
            "earliest_m":        _env_int  ("ORB_EARLIEST_M",    30),

            # Latest time to ENTER a new ORB trade. After this, no new
            # signals fire — leaves enough runway to hit T1/T2 before
            # the 15:30 close. Default 13:00.
            "latest_h":          _env_int  ("ORB_LATEST_H",      13),
            "latest_m":          _env_int  ("ORB_LATEST_M",       0),

            # R:R levels
            "rr_target":         _env_float("ORB_RR_TARGET",     1.5),
            "rr_target2":        _env_float("ORB_RR_TARGET2",    2.5),
        }

    @staticmethod
    def analyze(df, **ignored_kwargs):
        """Detect an ORB breakout on the most recent 5-min bar.

        Args:
            df: DataFrame with columns [ts, open, high, low, close, volume].
                Must include today's ORB window (9:15-9:30) plus the current
                bar. Backtest passes a 60-bar rolling window which always
                contains both within the trading day.

        Returns:
            signal dict (same shape as SignalGenV2 emits) or None.
            Diagnostics emitted to SignalGenORB.last_decision either way.
        """
        cfg = SignalGenORB._config()

        if df is None or len(df) < cfg["orb_bars"] + 1:
            SignalGenORB.last_decision = {"verdict": "INSUFFICIENT_BARS",
                                          "bars": len(df) if df is not None else 0}
            return None
        if pd is None:
            SignalGenORB.last_decision = {"verdict": "PANDAS_MISSING"}
            return None

        # ── Identify the current bar + today's date ──────────────────────
        n = len(df) - 1
        current_ts = df["ts"].iloc[n]
        if isinstance(current_ts, str):
            current_ts = pd.to_datetime(current_ts)
        try:
            current_ts = current_ts.to_pydatetime() if hasattr(current_ts, "to_pydatetime") else current_ts
        except Exception:
            pass
        current_date = current_ts.date() if hasattr(current_ts, "date") else None
        current_time = current_ts.time() if hasattr(current_ts, "time") else None

        if current_date is None or current_time is None:
            SignalGenORB.last_decision = {"verdict": "BAD_TIMESTAMP"}
            return None

        # ── Time gates ───────────────────────────────────────────────────
        earliest = time(cfg["earliest_h"], cfg["earliest_m"])
        latest   = time(cfg["latest_h"],   cfg["latest_m"])
        if current_time < earliest:
            SignalGenORB.last_decision = {"verdict": "BEFORE_ORB_COMPLETE",
                                          "current_time": current_time.strftime("%H:%M")}
            return None
        if current_time >= latest:
            SignalGenORB.last_decision = {"verdict": "AFTER_LATEST_ENTRY",
                                          "current_time": current_time.strftime("%H:%M")}
            return None

        # ── Extract today's bars from the rolling window ─────────────────
        # df['ts'] may already be tz-naive (data_layer strips IST); compare
        # by date only to be safe.
        ts_series = pd.to_datetime(df["ts"])
        today_mask = ts_series.dt.date == current_date
        today_bars = df.loc[today_mask].copy()
        today_bars["ts"] = ts_series.loc[today_mask]
        if len(today_bars) < cfg["orb_bars"] + 1:
            SignalGenORB.last_decision = {"verdict": "INSUFFICIENT_TODAY_BARS",
                                          "today_bars": len(today_bars)}
            return None

        # ── Establish the opening range (first N bars of the day) ────────
        orb_bars = today_bars.iloc[: cfg["orb_bars"]]
        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())
        orb_range = orb_high - orb_low
        orb_mid   = (orb_high + orb_low) / 2.0

        if orb_range < cfg["min_orb_range_pts"]:
            SignalGenORB.last_decision = {
                "verdict": "ORB_RANGE_TOO_TIGHT",
                "orb_range": round(orb_range, 1),
                "min_required": cfg["min_orb_range_pts"],
            }
            return None

        # ── Current bar + breakout confirmation ──────────────────────────
        cur_open  = float(df["open"].iloc[n])
        cur_close = float(df["close"].iloc[n])
        cur_high  = float(df["high"].iloc[n])
        cur_low   = float(df["low"].iloc[n])
        cur_range = cur_high - cur_low

        # Average bar range so far today (excluding ORB bars + current bar)
        post_orb_bars = today_bars.iloc[cfg["orb_bars"]:-1]
        if len(post_orb_bars) >= 1:
            avg_range = float((post_orb_bars["high"] - post_orb_bars["low"]).mean())
        else:
            avg_range = orb_range / cfg["orb_bars"]  # fall back to ORB-bar avg

        range_expansion = cur_range >= cfg["breakout_range_mult"] * avg_range if avg_range > 0 else True

        # ── Breakout detection ───────────────────────────────────────────
        # LONG: current bar closes above ORB high.
        # SHORT: current bar closes below ORB low.
        # Use close (not high) so we don't fire on wicks that get rejected.
        broke_up   = cur_close > orb_high
        broke_down = cur_close < orb_low

        diag = {
            "verdict":         None,
            "orb_high":        round(orb_high, 1),
            "orb_low":         round(orb_low, 1),
            "orb_range":       round(orb_range, 1),
            "orb_mid":         round(orb_mid, 1),
            "cur_close":       round(cur_close, 1),
            "cur_range":       round(cur_range, 1),
            "avg_range_today": round(avg_range, 1),
            "range_expansion": range_expansion,
            "broke_up":        broke_up,
            "broke_down":      broke_down,
        }

        if not (broke_up or broke_down):
            diag["verdict"] = "NO_BREAKOUT"
            SignalGenORB.last_decision = diag
            return None

        if not range_expansion:
            diag["verdict"] = "BREAKOUT_NO_RANGE_EXPANSION"
            SignalGenORB.last_decision = diag
            return None

        # ── Build the signal ─────────────────────────────────────────────
        if broke_up:
            direction = "LONG"
            entry = round(cur_close, 2)
            sl    = round(orb_low, 2)              # opposite ORB boundary
            risk  = entry - sl
            t1    = round(entry + risk * cfg["rr_target"],  2)
            t2    = round(entry + risk * cfg["rr_target2"], 2)
        else:
            direction = "SHORT"
            entry = round(cur_close, 2)
            sl    = round(orb_high, 2)             # opposite ORB boundary
            risk  = sl - entry
            t1    = round(entry - risk * cfg["rr_target"],  2)
            t2    = round(entry - risk * cfg["rr_target2"], 2)

        # ORB doesn't have a noisy "score" the way v2 does — confidence is
        # binary. Set it based on the size of the breakout move relative
        # to the ORB range, so deeper breakouts read as higher conviction.
        breakout_depth = (cur_close - orb_high) if direction == "LONG" else (orb_low - cur_close)
        breakout_pct_of_range = (breakout_depth / orb_range) if orb_range > 0 else 0.0
        confidence = max(60, min(90, int(70 + breakout_pct_of_range * 40)))

        reward = abs(t1 - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

        reasons = [
            f"ORB-15 {direction} breakout — close ₹{cur_close:.0f} "
            f"{'>' if direction == 'LONG' else '<'} "
            f"ORB {'high' if direction == 'LONG' else 'low'} ₹{orb_high if direction == 'LONG' else orb_low:.0f}",
            f"ORB range ₹{orb_range:.0f} (min ₹{cfg['min_orb_range_pts']:.0f})",
            f"Range expansion {cur_range:.1f} ≥ {cfg['breakout_range_mult']:.1f}× today's avg {avg_range:.1f}",
            f"SL ₹{sl:.0f} = opposite ORB boundary · 1R={risk:.0f} pts",
        ]

        diag["verdict"] = f"TRIGGER {direction} ORB-15"
        SignalGenORB.last_decision = diag

        return {
            "direction":    direction,
            "confidence":   confidence,
            "price":        round(cur_close, 2),
            "entry":        entry,
            "sl":           sl,
            "target1":      t1,
            "target2":      t2,
            "risk":         round(risk, 2),
            "reward":       round(reward, 2),
            "risk_reward":  rr,
            "reasons":      reasons,
            "indicators": {
                "orb_high":         round(orb_high, 1),
                "orb_low":          round(orb_low, 1),
                "orb_range":        round(orb_range, 1),
                "orb_mid":          round(orb_mid, 1),
                "cur_close":        round(cur_close, 1),
                "breakout_depth":   round(breakout_depth, 1),
                "breakout_pct":     round(breakout_pct_of_range * 100, 1),
                "range_expansion":  round(cur_range / avg_range, 2) if avg_range > 0 else 1.0,
            },
            "strategy":   "orb-15",
            "v2_score":   1,      # legacy compat — ORB doesn't have a 6-confluence score
            "v2_diag":    diag,
            "timestamp":  current_ts.strftime("%H:%M:%S") if hasattr(current_ts, "strftime") else "",
        }
