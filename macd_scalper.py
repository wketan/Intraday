"""MACD Scalper — fires on MACD histogram zero-line crossover + EMA9/EMA21 alignment.

Entry logic (per 5-min bar):
  LONG  : MACD histogram was negative last bar, flipped positive this bar
           AND EMA9 > EMA21 (fast EMA above slow = upward bias)
           AND RSI 35-70 (not extreme — room to run)

  SHORT : MACD histogram was positive last bar, flipped negative this bar
           AND EMA9 < EMA21
           AND RSI 30-65

Trade window: 9:45 – 14:30 (same as Conductor)
Target: 15 pts (T1), 25 pts (T2)
SL    : 10 pts minimum (or 0.5 * ATR14, whichever is larger)

This module runs alongside Conductor — no strategy switch needed. Engine calls both
every scan cycle and emits whichever fires.
"""

import pandas as pd
import numpy as np

_isnan = lambda x: x != x  # avoids importing math

EARLIEST_H, EARLIEST_M = 9, 45
LATEST_H,   LATEST_M   = 14, 30

T1_PTS = 15
T2_PTS = 25
SL_MIN = 10


class MACDScalper:
    last_decision: dict = {}

    # ── Indicator helpers (mirrored from Conductor for consistency) ──────────

    @staticmethod
    def _ema(s: pd.Series, period: int) -> pd.Series:
        return s.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
        delta = s.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        ag = gain.ewm(com=period - 1, adjust=False).mean()
        al = loss.ewm(com=period - 1, adjust=False).mean()
        rs = ag / al.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(s: pd.Series, fast=12, slow=26, signal=9):
        ema_f = MACDScalper._ema(s, fast)
        ema_s = MACDScalper._ema(s, slow)
        macd  = ema_f - ema_s
        sig   = MACDScalper._ema(macd, signal)
        return macd, sig, macd - sig   # (macd_line, signal_line, histogram)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        tr = pd.concat([h - l,
                        (h - c.shift()).abs(),
                        (l - c.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    # ── Main entry point ─────────────────────────────────────────────────────

    @staticmethod
    def analyze(df: pd.DataFrame, symbol: str = "",
                chain_analytics: dict = None, **_ignored) -> dict | None:
        """
        Returns a signal dict (same schema as Conductor) or None.
        Called on every scan cycle when strategy == 'conductor'.
        """
        if df is None or len(df) < 30:
            MACDScalper.last_decision = {
                "verdict": "INSUFFICIENT_BARS",
                "bars": len(df) if df is not None else 0,
            }
            return None

        n = len(df) - 1

        # ── Time gate (use candle timestamp, works in backtest too) ──────────
        cur_ts = df["ts"].iloc[n]
        if isinstance(cur_ts, str):
            cur_ts = pd.to_datetime(cur_ts)
        try:
            cur_ts = cur_ts.to_pydatetime() if hasattr(cur_ts, "to_pydatetime") else cur_ts
        except Exception:
            pass
        cur_time = cur_ts.time() if hasattr(cur_ts, "time") else None
        if cur_time is None:
            MACDScalper.last_decision = {"verdict": "BAD_TIMESTAMP"}
            return None

        from datetime import time as dtime
        earliest = dtime(EARLIEST_H, EARLIEST_M)
        latest   = dtime(LATEST_H, LATEST_M)

        if cur_time < earliest:
            MACDScalper.last_decision = {"verdict": "BEFORE_WINDOW",
                                          "time": cur_time.strftime("%H:%M")}
            return None
        if cur_time >= latest:
            MACDScalper.last_decision = {"verdict": "AFTER_WINDOW",
                                          "time": cur_time.strftime("%H:%M")}
            return None

        # ── Indicators ───────────────────────────────────────────────────────
        close = df["close"].astype(float)

        _, _, macd_hist = MACDScalper._macd(close)
        ema9  = MACDScalper._ema(close, 9)
        ema21 = MACDScalper._ema(close, 21)
        rsi   = MACDScalper._rsi(close, 14)
        atr   = MACDScalper._atr(df, 14)

        mh_now  = float(macd_hist.iloc[n])
        mh_prev = float(macd_hist.iloc[n - 1]) if n > 0 else 0.0
        e9      = float(ema9.iloc[n])
        e21     = float(ema21.iloc[n])
        rsi_now = float(rsi.iloc[n]) if not _isnan(rsi.iloc[n]) else 50.0
        atr_now = float(atr.iloc[n]) if not _isnan(atr.iloc[n]) else 0.0
        price   = float(close.iloc[n])

        # ── Crossover detection ──────────────────────────────────────────────
        # Strictly ONE bar: prev negative (or zero), current strictly positive
        flipped_long  = mh_prev <= 0 and mh_now > 0
        flipped_short = mh_prev >= 0 and mh_now < 0

        ema_long  = e9 > e21
        ema_short = e9 < e21

        rsi_ok_long  = 35 < rsi_now < 70
        rsi_ok_short = 30 < rsi_now < 65

        # ── Decide direction ─────────────────────────────────────────────────
        direction = None
        reasons   = []
        blocks    = []

        if flipped_long:
            if ema_long and rsi_ok_long:
                direction = "LONG"
                reasons = [
                    f"MACD histogram flipped green ({mh_prev:+.1f} → {mh_now:+.1f})",
                    f"EMA9 {e9:.0f} above EMA21 {e21:.0f} — uptrend confirmed",
                    f"RSI {rsi_now:.0f} — momentum zone, room to run",
                ]
            else:
                if not ema_long:
                    blocks.append(f"EMA9 {e9:.0f} still below EMA21 {e21:.0f}")
                if not rsi_ok_long:
                    blocks.append(f"RSI {rsi_now:.0f} outside 35-70 range")

        elif flipped_short:
            if ema_short and rsi_ok_short:
                direction = "SHORT"
                reasons = [
                    f"MACD histogram flipped red ({mh_prev:+.1f} → {mh_now:+.1f})",
                    f"EMA9 {e9:.0f} below EMA21 {e21:.0f} — downtrend confirmed",
                    f"RSI {rsi_now:.0f} — bearish momentum zone",
                ]
            else:
                if not ema_short:
                    blocks.append(f"EMA9 {e9:.0f} still above EMA21 {e21:.0f}")
                if not rsi_ok_short:
                    blocks.append(f"RSI {rsi_now:.0f} outside 30-65 range")

        if not direction:
            verdict = "NO_CROSSOVER"
            if flipped_long or flipped_short:
                verdict = "CROSSOVER_BLOCKED"
            MACDScalper.last_decision = {
                "verdict": verdict,
                "macd_h": round(mh_now, 2),
                "macd_h_prev": round(mh_prev, 2),
                "ema9": round(e9, 0),
                "ema21": round(e21, 0),
                "rsi": round(rsi_now, 1),
                "blocks": blocks,
            }
            return None

        # ── SL / target ──────────────────────────────────────────────────────
        sl_pts = max(SL_MIN, round(atr_now * 0.5))

        if direction == "LONG":
            entry  = price
            sl     = round(entry - sl_pts, 2)
            t1     = round(entry + T1_PTS, 2)
            t2     = round(entry + T2_PTS, 2)
        else:
            entry  = price
            sl     = round(entry + sl_pts, 2)
            t1     = round(entry - T1_PTS, 2)
            t2     = round(entry - T2_PTS, 2)

        rr = round(T1_PTS / sl_pts, 2)

        MACDScalper.last_decision = {
            "verdict": "TRIGGER",
            "direction": direction,
            "macd_h": round(mh_now, 2),
            "macd_h_prev": round(mh_prev, 2),
            "ema9": round(e9, 0),
            "ema21": round(e21, 0),
            "rsi": round(rsi_now, 1),
            "sl_pts": sl_pts,
        }

        return {
            "direction":   direction,
            "confidence":  72,
            "price":       round(price, 2),
            "entry":       round(entry, 2),
            "sl":          round(sl, 2),
            "target1":     t1,
            "target2":     t2,
            "risk_reward": rr,
            "reasons":     reasons,
            "indicators": {
                "strategy":       "macd_scalper",
                "macd_hist":      round(mh_now, 2),
                "macd_hist_prev": round(mh_prev, 2),
                "ema9":           round(e9, 0),
                "ema21":          round(e21, 0),
                "rsi":            round(rsi_now, 1),
                "atr":            round(atr_now, 1),
                "sl_pts":         sl_pts,
            },
        }
