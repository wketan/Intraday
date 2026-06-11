"""NIFTY regime-windowed strategies — time-of-day gated wrappers.

Why this module exists (2026-06-11)
───────────────────────────────────
Honest re-validation (post-e55eb4b lookahead fix) killed every NIFTY
strategy: Scalper 2.6% WR, ScalperV3 0 signals in 90d, Reverter 0/12,
Conductor negative on NIFTY. Deep research (102-agent verified sweep)
found NO replicable published NIFTY intraday option-buying edge — the
only fact that survived 3-0 adversarial verification is the U-shaped
intraday volatility curve (Singh & Gangwar 2018, MPRA 89689, 1-min
NIFTY futures 2011-2018): volatile open, dead 11:00-12:00, volatile
close.

Hypothesis under test: prior NIFTY failures came from trading the same
logic at ALL hours — momentum logic whipsawed in the midday dead zone,
mean-reversion fading the trending open. These wrappers apply the
regime split:

  NiftyWindows  — Conductor's multi-dim confluence (the only validated
                  engine, +₹68.6k/90d on BANKNIFTY honest harness) but
                  fires ONLY in the volatile windows: 09:30-11:00 and
                  13:30-14:45. Conductor's own internal gates (09:45
                  earliest, 14:30 latest by default) trim further.
  DeadzoneFade  — Reverter's VWAP-fade (extension + RSI extreme +
                  reversal candle) but fires ONLY inside the 11:00-13:15
                  dead zone, where mean reversion is the regime.

Both are STATELESS thin gates — all signal logic stays in the parent
module so a fix there propagates here. Both must clear the honest
90-day backtest gate (≥30 trades, +EV after costs, no outlier
concentration, max DD < ₹15k) before any live routing.

Window knobs (env-tunable, defaults from the verified U-shape):
  NW_OPEN_START / NW_OPEN_END     (default 09:30 / 11:00)
  NW_CLOSE_START / NW_CLOSE_END   (default 13:30 / 14:45)
  DZ_START / DZ_END               (default 11:00 / 13:15)
"""

from __future__ import annotations

import os
from datetime import time

try:
    import pandas as pd
except ImportError:
    pd = None


def _env_time(key: str, default_hm: str) -> time:
    """Parse 'HH:MM' env override; loud-fail on malformed values."""
    raw = os.environ.get(key, default_hm).strip()
    try:
        h, m = raw.split(":")
        return time(int(h), int(m))
    except Exception as e:
        raise ValueError(f"{key}='{raw}' is not HH:MM: {e}")


def _bar_dt(df):
    """Extract the current (last) bar's datetime, or None. Handles the
    'ts' column convention (dispatch normalizes 'timestamp' → 'ts')."""
    if df is None or len(df) == 0 or "ts" not in df.columns:
        return None
    cur = df["ts"].iloc[len(df) - 1]
    if isinstance(cur, str):
        cur = pd.to_datetime(cur)
    try:
        cur = cur.to_pydatetime() if hasattr(cur, "to_pydatetime") else cur
    except Exception:
        pass
    return cur if hasattr(cur, "time") else None


class NiftyWindows:
    """Conductor confluence, restricted to the volatile open/close windows."""

    last_decision: dict = {}

    OPEN_START  = _env_time("NW_OPEN_START",  "09:30")
    OPEN_END    = _env_time("NW_OPEN_END",    "11:00")
    CLOSE_START = _env_time("NW_CLOSE_START", "13:30")
    CLOSE_END   = _env_time("NW_CLOSE_END",   "14:45")

    @staticmethod
    def _set_decision(d: dict):
        NiftyWindows.last_decision = d

    @staticmethod
    def analyze(df, symbol: str = "", chain_analytics: dict = None,
                **ignored):
        cur = _bar_dt(df)
        if cur is None:
            NiftyWindows._set_decision({"verdict": "BAD_TIMESTAMP"})
            return None
        t = cur.time()
        in_open  = NiftyWindows.OPEN_START  <= t < NiftyWindows.OPEN_END
        in_close = NiftyWindows.CLOSE_START <= t < NiftyWindows.CLOSE_END
        if not (in_open or in_close):
            NiftyWindows._set_decision({
                "verdict": "OUTSIDE_VOL_WINDOW",
                "time": t.strftime("%H:%M"),
            })
            return None

        from conductor import Conductor
        result = Conductor.analyze(df, symbol=symbol,
                                   chain_analytics=chain_analytics)
        # Surface the delegate's rejection reason for diagnostics
        NiftyWindows._set_decision(dict(Conductor.last_decision or {}))
        return result


class DeadzoneFade:
    """Reverter VWAP-fade, restricted to the 11:00-13:15 dead zone."""

    last_decision: dict = {}

    DZ_START = _env_time("DZ_START", "11:00")
    DZ_END   = _env_time("DZ_END",   "13:15")

    @staticmethod
    def _set_decision(d: dict):
        DeadzoneFade.last_decision = d

    @staticmethod
    def analyze(df, symbol: str = "", chain_analytics: dict = None,
                **ignored):
        cur = _bar_dt(df)
        if cur is None:
            DeadzoneFade._set_decision({"verdict": "BAD_TIMESTAMP"})
            return None
        t = cur.time()
        if not (DeadzoneFade.DZ_START <= t < DeadzoneFade.DZ_END):
            DeadzoneFade._set_decision({
                "verdict": "OUTSIDE_DEADZONE",
                "time": t.strftime("%H:%M"),
            })
            return None

        from signal_reverter import Reverter
        result = Reverter.analyze(df, symbol=symbol,
                                  chain_analytics=chain_analytics)
        DeadzoneFade._set_decision(dict(Reverter.last_decision or {}))
        return result
