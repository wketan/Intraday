"""Scalper strategy — fast-trigger crossover scalps.

Models how the user actually trades manually:
  • Watch MACD histogram + EMA9/EMA21 crossovers on 5-min chart
  • Fire on confirmed crossover candle (green close for LONG, red for SHORT)
  • Target 5-10 spot points (~₹600-700 per lot on NIFTY)
  • Tight stop ~6-7 points
  • Hold 5-15 minutes max — exit on time if neither SL/T1 hits
  • 3-4 trades per day

This is NOT swing-style ORB or 5-dim confluence. Scalping is a different
beast — small targets, small stops, high frequency, time-bounded holds.
Statistical edge comes from win rate (target ≥55%) at near-1:1 R:R,
multiplied by trade count.

What counts as edge here:
  • At 55% WR with 1:1 R:R and ₹600 wins / ₹600 losses:
    EV = 0.55 × 600 + 0.45 × (-600) = +₹60/trade
    × 4 trades/day × 20 days = ₹4,800/month — meh
  • At 60% WR with 1.3:1 R:R (₹650 win, ₹500 loss):
    EV = 0.60 × 650 + 0.40 × (-500) = +₹190/trade
    × 4 × 20 = ₹15,200/month — getting there
  • At 65% WR with 1.5:1 R:R (₹700 win, ₹500 loss):
    EV = 0.65 × 700 + 0.35 × (-500) = +₹280/trade
    × 4 × 20 = ₹22,400/month — close to target
  • Sizing 2 lots (NIFTY) at same %s: ~₹44,800/month — target hit

So scalping NEEDS ≥60% win rate to be worthwhile. The crossover-based
entry is the historical-edge path: when MACD histogram flips zero AND
EMAs cross within 1-2 bars, the next 5-15 min has a directional bias.

Interface mirrors SignalGenV2/ORB/Gamma/Conductor.
"""

from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
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


class SignalGenScalper:
    """Stateless scalper analyzer. Returns signal dict or None per bar."""

    last_decision: dict = {}

    @staticmethod
    def _config() -> dict:
        return {
            # EMA periods (default 9/21 — most-cited scalping combo)
            "ema_fast":         _env_int  ("SCALP_EMA_FAST",       9),
            "ema_slow":         _env_int  ("SCALP_EMA_SLOW",      21),

            # MACD params (default 12,26,9 — standard)
            "macd_fast":        _env_int  ("SCALP_MACD_FAST",     12),
            "macd_slow":        _env_int  ("SCALP_MACD_SLOW",     26),
            "macd_signal":      _env_int  ("SCALP_MACD_SIGNAL",    9),

            # How many bars back to look for a crossover. 1 = current bar
            # only. 2 = current OR previous bar. Wider = more setups but
            # later entries (worse R:R).
            "crossover_lookback": _env_int("SCALP_CROSSOVER_LOOKBACK", 2),

            # Trigger requires BOTH? or EITHER?
            "require_both":     os.environ.get("SCALP_REQUIRE_BOTH", "false").lower() == "true",

            # Confirmation candle: body must be ≥ this fraction of bar range
            "min_body_pct":     _env_float("SCALP_MIN_BODY_PCT",  0.40),

            # Stops and targets (in SPOT points)
            "sl_pts":           _env_float("SCALP_SL_PTS",         7.0),
            "t1_pts":           _env_float("SCALP_T1_PTS",        10.0),
            "t2_pts":           _env_float("SCALP_T2_PTS",        18.0),

            # Time stop in 5-min bars (3 bars = 15 min)
            "time_stop_bars":   _env_int  ("SCALP_TIME_STOP_BARS",  3),

            # Window
            "earliest_h":       _env_int  ("SCALP_EARLIEST_H",      9),
            "earliest_m":       _env_int  ("SCALP_EARLIEST_M",     30),
            "latest_h":         _env_int  ("SCALP_LATEST_H",       14),
            "latest_m":         _env_int  ("SCALP_LATEST_M",       30),

            # Avoid trading the very flat midday window where crossovers
            # are noise. Block 12:00-13:00 by default (set to 0 to disable).
            "block_midday_start_h": _env_int("SCALP_MIDDAY_START_H", 12),
            "block_midday_end_h":   _env_int("SCALP_MIDDAY_END_H",   13),
        }

    @staticmethod
    def _ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _macd(series, fast, slow, signal):
        ema_f = SignalGenScalper._ema(series, fast)
        ema_s = SignalGenScalper._ema(series, slow)
        macd  = ema_f - ema_s
        sig   = SignalGenScalper._ema(macd, signal)
        hist  = macd - sig
        return macd, sig, hist

    @staticmethod
    def analyze(df, **ignored):
        """Score current bar for a scalp entry trigger.

        Returns signal dict if (MACD histogram crossover OR EMA crossover
        within lookback) AND confirmation candle in same direction.
        Otherwise returns None.
        """
        cfg = SignalGenScalper._config()
        if df is None or len(df) < max(cfg["ema_slow"], cfg["macd_slow"]) + 3:
            SignalGenScalper.last_decision = {"verdict": "INSUFFICIENT_BARS",
                                              "bars": len(df) if df is not None else 0}
            return None
        if pd is None:
            SignalGenScalper.last_decision = {"verdict": "PANDAS_MISSING"}
            return None

        n = len(df) - 1
        cur_ts = df["ts"].iloc[n]
        if isinstance(cur_ts, str):
            cur_ts = pd.to_datetime(cur_ts)
        try:
            cur_ts = cur_ts.to_pydatetime() if hasattr(cur_ts, "to_pydatetime") else cur_ts
        except Exception:
            pass
        cur_time = cur_ts.time() if hasattr(cur_ts, "time") else None
        if cur_time is None:
            SignalGenScalper.last_decision = {"verdict": "BAD_TIMESTAMP"}
            return None

        # ── Time gates ─────────────────────────────────────────────────
        if cur_time < time(cfg["earliest_h"], cfg["earliest_m"]):
            SignalGenScalper.last_decision = {"verdict": "BEFORE_WINDOW",
                                              "time": cur_time.strftime("%H:%M")}
            return None
        if cur_time >= time(cfg["latest_h"], cfg["latest_m"]):
            SignalGenScalper.last_decision = {"verdict": "AFTER_WINDOW",
                                              "time": cur_time.strftime("%H:%M")}
            return None
        # Midday chop block
        if cfg["block_midday_start_h"] > 0:
            mid_start = time(cfg["block_midday_start_h"], 0)
            mid_end   = time(cfg["block_midday_end_h"],   0)
            if mid_start <= cur_time < mid_end:
                SignalGenScalper.last_decision = {"verdict": "MIDDAY_CHOP_BLOCK",
                                                  "time": cur_time.strftime("%H:%M")}
                return None

        # ── Compute indicators ─────────────────────────────────────────
        close = df["close"]
        ema_f = SignalGenScalper._ema(close, cfg["ema_fast"])
        ema_s = SignalGenScalper._ema(close, cfg["ema_slow"])
        _, _, hist = SignalGenScalper._macd(close,
                                             cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])

        # Current and recent values
        lookback = cfg["crossover_lookback"]
        ema_f_now  = float(ema_f.iloc[n])
        ema_s_now  = float(ema_s.iloc[n])
        hist_now   = float(hist.iloc[n])

        # ── Detect crossovers within lookback window ───────────────────
        macd_cross_up_recent   = False
        macd_cross_down_recent = False
        ema_cross_up_recent    = False
        ema_cross_down_recent  = False

        for k in range(1, lookback + 1):
            if n - k < 0: break
            # MACD histogram zero-line cross
            h_now  = float(hist.iloc[n - k + 1])
            h_prev = float(hist.iloc[n - k])
            if h_prev <= 0 and h_now > 0: macd_cross_up_recent   = True
            if h_prev >= 0 and h_now < 0: macd_cross_down_recent = True
            # EMA fast/slow cross
            f_now  = float(ema_f.iloc[n - k + 1])
            f_prev = float(ema_f.iloc[n - k])
            s_now  = float(ema_s.iloc[n - k + 1])
            s_prev = float(ema_s.iloc[n - k])
            if f_prev <= s_prev and f_now > s_now: ema_cross_up_recent   = True
            if f_prev >= s_prev and f_now < s_now: ema_cross_down_recent = True

        if cfg["require_both"]:
            long_trigger  = macd_cross_up_recent   and ema_cross_up_recent
            short_trigger = macd_cross_down_recent and ema_cross_down_recent
        else:
            long_trigger  = macd_cross_up_recent   or  ema_cross_up_recent
            short_trigger = macd_cross_down_recent or  ema_cross_down_recent

        # ── Confirmation candle ────────────────────────────────────────
        cur_open  = float(df["open"].iloc[n])
        cur_close = float(df["close"].iloc[n])
        cur_high  = float(df["high"].iloc[n])
        cur_low   = float(df["low"].iloc[n])
        cur_range = cur_high - cur_low
        cur_body  = abs(cur_close - cur_open)
        body_pct  = (cur_body / cur_range) if cur_range > 0 else 0
        is_green  = cur_close > cur_open
        is_red    = cur_close < cur_open

        diag = {
            "verdict":           None,
            "time":              cur_time.strftime("%H:%M"),
            "ema_f":             round(ema_f_now, 1),
            "ema_s":             round(ema_s_now, 1),
            "hist":              round(hist_now, 2),
            "macd_cross_up":     macd_cross_up_recent,
            "macd_cross_down":   macd_cross_down_recent,
            "ema_cross_up":      ema_cross_up_recent,
            "ema_cross_down":    ema_cross_down_recent,
            "long_trigger":      long_trigger,
            "short_trigger":     short_trigger,
            "body_pct":          round(body_pct * 100, 1),
            "is_green":          is_green,
            "is_red":            is_red,
        }

        # Decide direction
        direction = None
        if long_trigger and is_green and body_pct >= cfg["min_body_pct"]:
            direction = "LONG"
        elif short_trigger and is_red and body_pct >= cfg["min_body_pct"]:
            direction = "SHORT"
        else:
            if not (long_trigger or short_trigger):
                diag["verdict"] = "NO_CROSSOVER"
            elif long_trigger and not is_green:
                diag["verdict"] = "LONG_TRIGGER_NO_GREEN_CONFIRM"
            elif short_trigger and not is_red:
                diag["verdict"] = "SHORT_TRIGGER_NO_RED_CONFIRM"
            elif body_pct < cfg["min_body_pct"]:
                diag["verdict"] = "WEAK_BODY"
            else:
                diag["verdict"] = "AMBIGUOUS"
            SignalGenScalper.last_decision = diag
            return None

        # ── Build entry/SL/T1/T2 in SPOT points ────────────────────────
        # Scalper uses fixed tight pt SL/T1 — not ATR-based. The whole
        # point is small, defined risk per trade.
        price = round(cur_close, 2)
        if direction == "LONG":
            entry = price
            sl    = round(entry - cfg["sl_pts"], 2)
            t1    = round(entry + cfg["t1_pts"], 2)
            t2    = round(entry + cfg["t2_pts"], 2)
        else:
            entry = price
            sl    = round(entry + cfg["sl_pts"], 2)
            t1    = round(entry - cfg["t1_pts"], 2)
            t2    = round(entry - cfg["t2_pts"], 2)

        risk   = abs(entry - sl)
        reward = abs(t1 - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

        # Confidence: both triggers + body strength
        confidence = 60
        if macd_cross_up_recent and ema_cross_up_recent and direction == "LONG":
            confidence += 15
        if macd_cross_down_recent and ema_cross_down_recent and direction == "SHORT":
            confidence += 15
        if body_pct >= 0.7:
            confidence += 10

        reasons = []
        if direction == "LONG":
            if macd_cross_up_recent:
                reasons.append(f"MACD histogram flipped GREEN (hist now {hist_now:+.2f})")
            if ema_cross_up_recent:
                reasons.append(f"EMA{cfg['ema_fast']} crossed above EMA{cfg['ema_slow']} ({ema_f_now:.0f} > {ema_s_now:.0f})")
            reasons.append(f"Green confirmation bar (body {body_pct*100:.0f}% of range)")
        else:
            if macd_cross_down_recent:
                reasons.append(f"MACD histogram flipped RED (hist now {hist_now:+.2f})")
            if ema_cross_down_recent:
                reasons.append(f"EMA{cfg['ema_fast']} crossed below EMA{cfg['ema_slow']} ({ema_f_now:.0f} < {ema_s_now:.0f})")
            reasons.append(f"Red confirmation bar (body {body_pct*100:.0f}% of range)")
        reasons.append(f"Scalp: SL {cfg['sl_pts']:.0f} pts, T1 {cfg['t1_pts']:.0f} pts, time-stop {cfg['time_stop_bars']} bars")

        diag["verdict"] = f"TRIGGER {direction} SCALP"
        SignalGenScalper.last_decision = diag

        return {
            "direction":    direction,
            "confidence":   confidence,
            "price":        price,
            "entry":        entry,
            "sl":           sl,
            "target1":      t1,
            "target2":      t2,
            "risk":         round(risk, 2),
            "reward":       round(reward, 2),
            "risk_reward":  rr,
            "reasons":      reasons,
            "indicators": {
                "ema_fast":  round(ema_f_now, 1),
                "ema_slow":  round(ema_s_now, 1),
                "macd_hist": round(hist_now, 2),
                "body_pct":  round(body_pct * 100, 1),
                "rr":        rr,
                "sl_pts":    cfg["sl_pts"],
                "t1_pts":    cfg["t1_pts"],
                "time_stop_bars": cfg["time_stop_bars"],
            },
            "strategy":   "scalper",
            "v2_score":   1,
            "v2_diag":    diag,
            "timestamp":  cur_ts.strftime("%H:%M:%S") if hasattr(cur_ts, "strftime") else "",
        }
