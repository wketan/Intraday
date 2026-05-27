"""Gamma blast strategy — expiry-day only, gated tight.

The mechanics: on expiry day, ATM/1-OTM index options have gamma 3-5×
normal-day values (Quantsapp). When spot suddenly moves toward a strike,
delta accelerates (0.05 → 0.20 → 0.50 → 0.80) and short option sellers
cover by buying spot, pushing it further. Premium can multiply 2-5×
in minutes. Documented in EnrichMoney, Pushkarraj Thakur, ORB Trader,
AlgoTest, Sensibull.

This implementation is DISCIPLINED (not the deep-OTM lottery version):
- Expiry days only (NIFTY weekly Tue, BANKNIFTY/FINNIFTY monthly)
- 14:00-15:15 IST window only (peak gamma squeeze, before close gamma noise)
- Compressed-range precondition: day's spot range <1% by entry time
  (compressed = sellers complacent = more squeeze fuel)
- Breakout candle: current bar body ≥ 2× avg of prior 5 bars
- Range expansion confirmation: current bar range ≥ 1.5× prior 5-bar avg
- Direction: CE on bullish break, PE on bearish break

Sizing/SL/TP (handled by execution layer, not the signal):
- ATM or 1-OTM strike only (never deeper)
- Premium ₹5-75 range (avoid pure lottery + avoid expensive)
- SL 30% of premium, TP1 2× (scale half), trail rest
- Hard exit 15:15

Realistic edge: 40-45% WR with 2-3× payoffs (AlgoTest), ~1-2 trades
per expiry day. NOT a primary strategy — a tactical add-on. The
naive deep-OTM version has ~5% WR and is negative EV. This version
filters tight so only the high-conviction setups fire.

Interface mirrors SignalGenV2 and SignalGenORB: stateless analyze(df)
returns a signal dict or None.
"""

from __future__ import annotations

import os
from datetime import datetime, date, time, timezone, timedelta
from typing import Optional

try:
    import pandas as pd
except ImportError:
    pd = None

IST = timezone(timedelta(hours=5, minutes=30))


def _env_float(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)))
    except: return default

def _env_int(key: str, default: int) -> int:
    try: return int(os.environ.get(key, str(default)))
    except: return default


def is_expiry_day(d: date, symbol: str) -> bool:
    """Return True if `d` is an expiry day for the given index.

    NIFTY post-Nov-2024: weekly expiry every Tuesday.
    BANKNIFTY / FINNIFTY post-March-2025: monthly only (last Tuesday of month).

    Holidays not handled here — caller should also verify the date is a
    trading day. False positives on holiday-shifted expiries are OK
    because the backtest will just find no spot bars for that day.
    """
    if d.weekday() != 1:   # not Tuesday — no expiry
        return False
    sym = symbol.upper()
    if sym == "NIFTY":
        return True  # every Tuesday
    if sym in ("BANKNIFTY", "FINNIFTY"):
        # Last Tuesday of the month: next Tuesday is in a different month.
        next_tue = d + timedelta(days=7)
        return next_tue.month != d.month
    return False


class SignalGenGamma:
    """Gamma Blast — expiry-day only. Stateless analyzer."""

    last_decision: dict = {}

    @staticmethod
    def _config() -> dict:
        return {
            # Time window for entry (IST)
            "earliest_h":           _env_int  ("GAMMA_EARLIEST_H",   14),
            "earliest_m":           _env_int  ("GAMMA_EARLIEST_M",    0),
            "latest_h":             _env_int  ("GAMMA_LATEST_H",     15),
            "latest_m":             _env_int  ("GAMMA_LATEST_M",     15),

            # Day must be compressed (range <X% of open by entry time).
            # Compressed days = sellers complacent = more squeeze fuel.
            "max_day_range_pct":    _env_float("GAMMA_MAX_DAY_RANGE_PCT", 1.0),  # 1%

            # Breakout candle filters
            "body_mult":            _env_float("GAMMA_BODY_MULT",        2.0),  # 2× avg body
            "range_mult":           _env_float("GAMMA_RANGE_MULT",       1.5),  # 1.5× avg range
            "lookback_bars":        _env_int  ("GAMMA_LOOKBACK_BARS",      5),

            # Min absolute body size to count as a "real" breakout — prevents
            # the body_mult check from firing on 5×0.1 = 0.5 pt moves.
            "min_body_pts":         _env_float("GAMMA_MIN_BODY_PTS",     8.0),

            # SL/TP as % of premium (used by execution layer)
            "sl_pct":               _env_float("GAMMA_SL_PCT",          0.30),  # 30% loss
            "tp1_pct":              _env_float("GAMMA_TP1_PCT",         1.00),  # 100% (2×)
            "tp2_pct":              _env_float("GAMMA_TP2_PCT",         2.00),  # 200% (3×)
        }

    @staticmethod
    def analyze(df, symbol: str = "", **ignored_kwargs):
        """Detect a gamma-blast-worthy breakout on expiry day.

        Args:
            df: DataFrame with [ts, open, high, low, close, volume].
            symbol: instrument name ("NIFTY", "BANKNIFTY", "FINNIFTY") —
                    required to check expiry-day rules per index.

        Returns:
            signal dict or None. Note: T1/T2 here are at premium-pct levels
            (not spot levels) because gamma blast trades by premium movement
            not spot points. Execution layer handles strike picking + SL/TP.
        """
        cfg = SignalGenGamma._config()

        if df is None or len(df) < cfg["lookback_bars"] + 1:
            SignalGenGamma.last_decision = {"verdict": "INSUFFICIENT_BARS",
                                            "bars": len(df) if df is not None else 0}
            return None
        if pd is None:
            SignalGenGamma.last_decision = {"verdict": "PANDAS_MISSING"}
            return None
        if not symbol:
            SignalGenGamma.last_decision = {"verdict": "NO_SYMBOL"}
            return None

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
            SignalGenGamma.last_decision = {"verdict": "BAD_TIMESTAMP"}
            return None

        # ── Gate 1: expiry day only ─────────────────────────────────────
        if not is_expiry_day(current_date, symbol):
            SignalGenGamma.last_decision = {"verdict": "NOT_EXPIRY_DAY",
                                            "date": str(current_date),
                                            "symbol": symbol}
            return None

        # ── Gate 2: time window ─────────────────────────────────────────
        earliest = time(cfg["earliest_h"], cfg["earliest_m"])
        latest   = time(cfg["latest_h"],   cfg["latest_m"])
        if current_time < earliest:
            SignalGenGamma.last_decision = {"verdict": "BEFORE_GAMMA_WINDOW",
                                            "current_time": current_time.strftime("%H:%M")}
            return None
        if current_time >= latest:
            SignalGenGamma.last_decision = {"verdict": "AFTER_GAMMA_WINDOW",
                                            "current_time": current_time.strftime("%H:%M")}
            return None

        # ── Gate 3: compressed-day precondition ─────────────────────────
        # Pull all of today's bars from the rolling window to compute day range.
        ts_series = pd.to_datetime(df["ts"])
        today_mask = ts_series.dt.date == current_date
        today_bars = df.loc[today_mask].copy()
        if len(today_bars) < cfg["lookback_bars"] + 1:
            SignalGenGamma.last_decision = {"verdict": "INSUFFICIENT_TODAY_BARS",
                                            "today_bars": len(today_bars)}
            return None

        day_open  = float(today_bars["open"].iloc[0])
        day_high  = float(today_bars["high"].max())
        day_low   = float(today_bars["low"].min())
        day_range_pct = ((day_high - day_low) / day_open * 100.0) if day_open > 0 else 999

        if day_range_pct >= cfg["max_day_range_pct"]:
            SignalGenGamma.last_decision = {
                "verdict": "DAY_NOT_COMPRESSED",
                "day_range_pct": round(day_range_pct, 2),
                "max_allowed": cfg["max_day_range_pct"],
            }
            return None

        # ── Gate 4: breakout candle (body + range expansion) ────────────
        cur_open  = float(df["open"].iloc[n])
        cur_close = float(df["close"].iloc[n])
        cur_high  = float(df["high"].iloc[n])
        cur_low   = float(df["low"].iloc[n])
        cur_body  = abs(cur_close - cur_open)
        cur_range = cur_high - cur_low

        lookback = cfg["lookback_bars"]
        prior = df.iloc[max(0, n - lookback): n]   # prior N bars, excluding current
        avg_body  = float((prior["close"] - prior["open"]).abs().mean()) if len(prior) else 0.0
        avg_range = float((prior["high"]  - prior["low"]).mean())        if len(prior) else 0.0

        diag = {
            "verdict":       None,
            "date":          str(current_date),
            "symbol":        symbol,
            "current_time":  current_time.strftime("%H:%M"),
            "day_range_pct": round(day_range_pct, 2),
            "cur_body":      round(cur_body, 1),
            "avg_body":      round(avg_body, 1),
            "body_ratio":    round(cur_body / avg_body, 2) if avg_body > 0 else 0,
            "cur_range":     round(cur_range, 1),
            "avg_range":     round(avg_range, 1),
            "range_ratio":   round(cur_range / avg_range, 2) if avg_range > 0 else 0,
        }

        # Body must be ≥ 2× avg AND ≥ min absolute pts
        if cur_body < cfg["min_body_pts"]:
            diag["verdict"] = "BODY_TOO_SMALL_ABS"
            SignalGenGamma.last_decision = diag
            return None
        if avg_body > 0 and cur_body < cfg["body_mult"] * avg_body:
            diag["verdict"] = "BODY_NOT_EXPANDED"
            SignalGenGamma.last_decision = diag
            return None
        if avg_range > 0 and cur_range < cfg["range_mult"] * avg_range:
            diag["verdict"] = "RANGE_NOT_EXPANDED"
            SignalGenGamma.last_decision = diag
            return None

        # ── Direction from candle ───────────────────────────────────────
        if cur_close > cur_open:
            direction = "LONG"     # bullish break → buy CE
        elif cur_close < cur_open:
            direction = "SHORT"    # bearish break → buy PE
        else:
            diag["verdict"] = "DOJI_NO_DIRECTION"
            SignalGenGamma.last_decision = diag
            return None

        # ── Build the signal ────────────────────────────────────────────
        # Spot-level SL/T1/T2 are placeholders here — gamma blast trades
        # premium-pct exits (handled by execution). We expose the breakout
        # boundary as SL so the engine has SOMETHING to compute risk from.
        if direction == "LONG":
            entry = round(cur_close, 2)
            sl    = round(min(cur_low, cur_open), 2)   # below breakout candle low
            risk  = entry - sl
            t1    = round(entry + risk * 2.0, 2)       # 2R = first scale-out
            t2    = round(entry + risk * 3.0, 2)       # 3R = trail target
        else:
            entry = round(cur_close, 2)
            sl    = round(max(cur_high, cur_open), 2)
            risk  = sl - entry
            t1    = round(entry - risk * 2.0, 2)
            t2    = round(entry - risk * 3.0, 2)

        # Confidence scales with how much the day was compressed (more
        # compressed = stronger signal) and how big the body was.
        compression_score = max(0, (cfg["max_day_range_pct"] - day_range_pct) / cfg["max_day_range_pct"])  # 0..1
        body_score = min(1.0, (cur_body / avg_body / cfg["body_mult"])) if avg_body > 0 else 0.5
        confidence = max(60, min(90, int(60 + (compression_score + body_score) * 15)))

        reasons = [
            f"Expiry-day gamma blast {direction} on {current_ts.strftime('%H:%M')}",
            f"Day compressed: range {day_range_pct:.2f}% (<{cfg['max_day_range_pct']:.1f}%)",
            f"Breakout body {cur_body:.0f} pts = {cur_body/avg_body:.1f}× avg of last {lookback} bars",
            f"Range expansion {cur_range:.0f} = {cur_range/avg_range:.1f}× avg",
            f"SL ₹{sl:.0f} = breakout candle {'low' if direction=='LONG' else 'high'} · 1R={risk:.0f} pts",
        ]

        diag["verdict"] = f"TRIGGER {direction} GAMMA-BLAST"
        SignalGenGamma.last_decision = diag

        return {
            "direction":    direction,
            "confidence":   confidence,
            "price":        round(cur_close, 2),
            "entry":        entry,
            "sl":           sl,
            "target1":      t1,
            "target2":      t2,
            "risk":         round(risk, 2),
            "reward":       round(abs(t1 - entry), 2),
            "risk_reward":  round(abs(t1 - entry) / risk, 2) if risk > 0 else 0,
            "reasons":      reasons,
            "indicators": {
                "day_range_pct":  round(day_range_pct, 2),
                "cur_body":       round(cur_body, 1),
                "body_ratio":     round(cur_body / avg_body, 2) if avg_body > 0 else 0,
                "cur_range":      round(cur_range, 1),
                "range_ratio":    round(cur_range / avg_range, 2) if avg_range > 0 else 0,
                "compression":    round(compression_score, 2),
                # Gamma-specific premium-pct exit hints (execution uses these)
                "premium_sl_pct":  cfg["sl_pct"],
                "premium_tp1_pct": cfg["tp1_pct"],
                "premium_tp2_pct": cfg["tp2_pct"],
            },
            "strategy":   "gamma-blast",
            "v2_score":   1,
            "v2_diag":    diag,
            "timestamp":  current_ts.strftime("%H:%M:%S") if hasattr(current_ts, "strftime") else "",
        }
