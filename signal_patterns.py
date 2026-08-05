"""PatternScanner — deterministic chart-pattern detector (intraday, 5-min).

Detects classic price patterns by GEOMETRY (swing-point relationships),
not by AI. Zero token cost, instant, reproducible. Emits the standard
signal dict so it plugs into backtest_v2 + the live engine exactly like
Conductor/Reverter.

⚠️ EVIDENCE WARNING (2026-06-16): chart patterns are the weakest-evidence
approach in trading. A 67,041-trade backtest of head & shoulders showed
47% win rate and ~zero average return; large multi-rule studies (incl.
Indian market) find technical patterns don't survive out-of-sample. This
module exists to TEST patterns on our honest 90d harness — not because
they're known to work. It must clear the same gate Conductor passed
(≥30 trades, +EV after costs, no outlier concentration, survivable DD)
before any live routing. Expect it to fail; let the backtest decide.

Patterns implemented (well-defined on a 60-bar intraday window):
  Reversal   : Head & Shoulders, Inverse H&S, Double Top, Double Bottom
  Continuation: Ascending / Descending / Symmetrical Triangle, Bull/Bear Flag
Deferred (need more bars than an intraday window gives, or too noisy to
detect without heavy false positives): Rising/Falling Wedge, Cup & Handle.
If the core set shows no edge, there is no reason to add these.

Mechanism for every pattern: locate confirmed swing pivots, verify the
geometric relationship, then fire ONLY on the bar that breaks the
neckline / boundary (a discrete event, so it can't re-fire while the
condition stands). SL = pattern-invalidation level. Target = measured
move (pattern height projected from the breakout).
"""

from __future__ import annotations

import os
from datetime import time

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


def _env_float(key, default):
    try: return float(os.environ.get(key, str(default)))
    except: return default

def _env_int(key, default):
    try: return int(os.environ.get(key, str(default)))
    except: return default


PAT_EARLIEST_H   = _env_int  ("PATTERN_EARLIEST_H", 9)
PAT_EARLIEST_M   = _env_int  ("PATTERN_EARLIEST_M", 45)
PAT_LATEST_H     = _env_int  ("PATTERN_LATEST_H", 14)
PAT_LATEST_M     = _env_int  ("PATTERN_LATEST_M", 30)
PAT_MIN_RR       = _env_float("PATTERN_MIN_RR", 1.5)
PAT_TARGET_RS    = _env_float("PATTERN_TARGET_PROFIT_RS", 1000.0)
PAT_PIVOT_K      = _env_int  ("PATTERN_PIVOT_K", 3)       # bars each side for a confirmed pivot
PAT_LEVEL_TOL    = _env_float("PATTERN_LEVEL_TOL_PCT", 0.25)  # "same level" tolerance, % of price
PAT_MIN_HEIGHT   = _env_float("PATTERN_MIN_HEIGHT_PCT", 0.15)  # min pattern height, % of price

_LOTS = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60}


def _isnan(x):
    try: return x != x
    except: return False


class PatternScanner:
    last_decision: dict = {}

    @staticmethod
    def _set(d): PatternScanner.last_decision = d

    # ── swing pivots ────────────────────────────────────────────────
    @staticmethod
    def _pivots(highs, lows, k):
        """Return (pivot_highs, pivot_lows) as lists of (idx, price), only
        for bars that have k confirmed bars on BOTH sides (so the last k
        bars — including the current/breakout bar — are never pivots)."""
        ph, pl = [], []
        n = len(highs)
        for i in range(k, n - k):
            hw = highs[i - k:i + k + 1]
            lw = lows[i - k:i + k + 1]
            if highs[i] >= hw.max() and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                ph.append((i, float(highs[i])))
            if lows[i] <= lw.min() and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                pl.append((i, float(lows[i])))
        return ph, pl

    @staticmethod
    def _atr(df, period=14):
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def analyze(df, symbol: str = "", chain_analytics: dict = None, **ignored):
        if pd is None:
            PatternScanner._set({"verdict": "PANDAS_MISSING"}); return None
        if df is None or len(df) < 30:
            PatternScanner._set({"verdict": "INSUFFICIENT_BARS",
                                 "bars": 0 if df is None else len(df)}); return None

        # Time gate (same window as the other intraday strategies)
        n = len(df) - 1
        cur_ts = df["ts"].iloc[n]
        if isinstance(cur_ts, str): cur_ts = pd.to_datetime(cur_ts)
        try: cur_ts = cur_ts.to_pydatetime() if hasattr(cur_ts, "to_pydatetime") else cur_ts
        except Exception: pass
        ct = cur_ts.time() if hasattr(cur_ts, "time") else None
        if ct is None:
            PatternScanner._set({"verdict": "BAD_TIMESTAMP"}); return None
        if ct < time(PAT_EARLIEST_H, PAT_EARLIEST_M):
            PatternScanner._set({"verdict": "BEFORE_WINDOW", "time": ct.strftime("%H:%M")}); return None
        if ct >= time(PAT_LATEST_H, PAT_LATEST_M):
            PatternScanner._set({"verdict": "AFTER_WINDOW", "time": ct.strftime("%H:%M")}); return None

        highs = df["high"].to_numpy(dtype=float)
        lows  = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        price = float(closes[n])
        prev_close = float(closes[n - 1])
        if price <= 0:
            PatternScanner._set({"verdict": "BAD_PRICE"}); return None

        atr_series = PatternScanner._atr(df)
        atr_now = float(atr_series.iloc[n]) if not _isnan(atr_series.iloc[n]) else 0.0
        tol = price * PAT_LEVEL_TOL / 100.0
        min_h = max(price * PAT_MIN_HEIGHT / 100.0, 0.5 * atr_now)

        ph, pl = PatternScanner._pivots(highs, lows, PAT_PIVOT_K)
        if len(ph) < 2 and len(pl) < 2:
            PatternScanner._set({"verdict": "NO_PIVOTS",
                                 "ph": len(ph), "pl": len(pl)}); return None

        # Each detector returns dict(name, direction, neckline, height,
        # invalidation, quality) on a CURRENT-BAR breakout, else None.
        cands = []
        for fn in (PatternScanner._double_top, PatternScanner._double_bottom,
                   PatternScanner._hns, PatternScanner._inv_hns,
                   PatternScanner._asc_tri, PatternScanner._desc_tri,
                   PatternScanner._sym_tri, PatternScanner._flag):
            try:
                c = fn(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr_now)
            except Exception:
                c = None
            if c: cands.append(c)

        if not cands:
            PatternScanner._set({"verdict": "NO_PATTERN",
                                 "ph": len(ph), "pl": len(pl)}); return None

        # Pick highest-quality breakout
        best = max(cands, key=lambda c: c["quality"])
        direction = best["direction"]
        height = best["height"]
        neck = best["neckline"]
        entry = round(price, 2)

        # Textbook measured-move construction: target = pattern height
        # projected from the NECKLINE; stop = "breakout failed" just back
        # through the neckline (buffer = max 0.4×ATR or 10% of height).
        # NOT a full-pattern-depth stop — that gives ~1:1 RR and is a big
        # reason naive pattern trading loses. (The invalidation extreme is
        # still respected as a hard outer cap on the stop.)
        buf = max(0.4 * atr_now, 0.10 * height, 1.0)
        if direction == "LONG":
            sl = round(max(neck - buf, best["invalidation"]), 2)
            t1 = round(neck + height, 2)
            t2 = round(neck + height * 1.5, 2)
            risk = entry - sl
        else:
            sl = round(min(neck + buf, best["invalidation"]), 2)
            t1 = round(neck - height, 2)
            t2 = round(neck - height * 1.5, 2)
            risk = sl - entry

        if risk <= 0:
            PatternScanner._set({"verdict": "BAD_RISK", "pattern": best["name"]}); return None
        rr = round(abs(t1 - entry) / risk, 2)
        if rr < PAT_MIN_RR:
            PatternScanner._set({"verdict": "RR_BELOW_MIN", "rr": rr,
                                 "min": PAT_MIN_RR, "pattern": best["name"]}); return None

        lot = _LOTS.get(symbol.upper(), 75)
        est_t1_rs = abs(t1 - entry) * 0.5 * lot
        if est_t1_rs < PAT_TARGET_RS:
            PatternScanner._set({"verdict": "T1_BELOW_TARGET_RS",
                                 "est_t1_profit": round(est_t1_rs),
                                 "target": PAT_TARGET_RS, "pattern": best["name"]}); return None

        confidence = int(max(55, min(85, best["quality"])))
        diag = {"verdict": f"TRIGGER {best['name']} {direction}",
                "pattern": best["name"], "rr": rr, "est_t1_profit": round(est_t1_rs)}
        PatternScanner._set(diag)

        return {
            "direction": direction, "confidence": confidence,
            "price": entry, "entry": entry, "sl": sl,
            "target1": t1, "target2": t2,
            "risk": round(risk, 2), "reward": round(abs(t1 - entry), 2),
            "risk_reward": rr,
            "reasons": [f"{best['name']} breakout · height {round(height,1)}pt · "
                        f"target {round(abs(t1-entry),1)}pt"],
            "indicators": {"atr": round(atr_now, 2), "pattern": best["name"],
                           "height": round(height, 2)},
            "strategy": "patterns", "v2_score": confidence,
            "v2_diag": diag,
            "timestamp": cur_ts.strftime("%H:%M:%S") if hasattr(cur_ts, "strftime") else "",
        }

    # ── individual detectors ────────────────────────────────────────
    # Each fires only when the CURRENT bar's close breaks the neckline/
    # boundary AND the prior bar had not yet broken it (fresh breakout).

    @staticmethod
    def _double_top(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        if len(ph) < 2: return None
        (i1, p1), (i2, p2) = ph[-2], ph[-1]
        if abs(p1 - p2) > tol: return None
        troughs = [pr for (j, pr) in pl if i1 < j < i2]
        if not troughs: return None
        neck = min(troughs)
        height = min(p1, p2) - neck
        if height < min_h: return None
        if not (prev_close >= neck and price < neck): return None  # fresh down-break
        sym = 1 - abs(p1 - p2) / max(tol, 1e-9)
        return {"name": "Double Top", "direction": "SHORT", "neckline": neck,
                "height": height, "invalidation": max(p1, p2),
                "quality": 60 + 10 * max(0, sym) + min(10, height / max(atr, 1e-9) * 3)}

    @staticmethod
    def _double_bottom(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        if len(pl) < 2: return None
        (i1, p1), (i2, p2) = pl[-2], pl[-1]
        if abs(p1 - p2) > tol: return None
        peaks = [pr for (j, pr) in ph if i1 < j < i2]
        if not peaks: return None
        neck = max(peaks)
        height = neck - max(p1, p2)
        if height < min_h: return None
        if not (prev_close <= neck and price > neck): return None  # fresh up-break
        sym = 1 - abs(p1 - p2) / max(tol, 1e-9)
        return {"name": "Double Bottom", "direction": "LONG", "neckline": neck,
                "height": height, "invalidation": min(p1, p2),
                "quality": 60 + 10 * max(0, sym) + min(10, height / max(atr, 1e-9) * 3)}

    @staticmethod
    def _hns(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        if len(ph) < 3 or len(pl) < 2: return None
        (iL, L), (iH, H), (iR, R) = ph[-3], ph[-2], ph[-1]
        if not (H > L and H > R): return None          # head is highest
        if abs(L - R) > tol * 1.5: return None          # shoulders ~level
        troughs = [(j, pr) for (j, pr) in pl if iL < j < iR]
        if len(troughs) < 2: return None
        neck = sum(pr for _, pr in troughs) / len(troughs)
        height = H - neck
        if height < min_h: return None
        if not (prev_close >= neck and price < neck): return None
        return {"name": "Head & Shoulders", "direction": "SHORT", "neckline": neck,
                "height": height, "invalidation": H,
                "quality": 62 + 8 * (1 - abs(L - R) / max(tol * 1.5, 1e-9))
                           + min(10, height / max(atr, 1e-9) * 2)}

    @staticmethod
    def _inv_hns(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        if len(pl) < 3 or len(ph) < 2: return None
        (iL, L), (iH, H), (iR, R) = pl[-3], pl[-2], pl[-1]
        if not (H < L and H < R): return None          # head is lowest
        if abs(L - R) > tol * 1.5: return None
        peaks = [(j, pr) for (j, pr) in ph if iL < j < iR]
        if len(peaks) < 2: return None
        neck = sum(pr for _, pr in peaks) / len(peaks)
        height = neck - H
        if height < min_h: return None
        if not (prev_close <= neck and price > neck): return None
        return {"name": "Inverse H&S", "direction": "LONG", "neckline": neck,
                "height": height, "invalidation": H,
                "quality": 62 + 8 * (1 - abs(L - R) / max(tol * 1.5, 1e-9))
                           + min(10, height / max(atr, 1e-9) * 2)}

    @staticmethod
    def _asc_tri(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        if len(ph) < 2 or len(pl) < 2: return None
        (i1, h1), (i2, h2) = ph[-2], ph[-1]
        (j1, l1), (j2, l2) = pl[-2], pl[-1]
        if abs(h1 - h2) > tol: return None              # flat resistance
        if not (l2 > l1 + tol * 0.5): return None        # rising support
        res = max(h1, h2)
        height = res - l1
        if height < min_h: return None
        if not (prev_close <= res and price > res): return None
        return {"name": "Ascending Triangle", "direction": "LONG", "neckline": res,
                "height": height, "invalidation": l2,
                "quality": 60 + min(12, (l2 - l1) / max(atr, 1e-9) * 4)}

    @staticmethod
    def _desc_tri(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        if len(ph) < 2 or len(pl) < 2: return None
        (i1, h1), (i2, h2) = ph[-2], ph[-1]
        (j1, l1), (j2, l2) = pl[-2], pl[-1]
        if abs(l1 - l2) > tol: return None              # flat support
        if not (h2 < h1 - tol * 0.5): return None        # falling resistance
        sup = min(l1, l2)
        height = h1 - sup
        if height < min_h: return None
        if not (prev_close >= sup and price < sup): return None
        return {"name": "Descending Triangle", "direction": "SHORT", "neckline": sup,
                "height": height, "invalidation": h2,
                "quality": 60 + min(12, (h1 - h2) / max(atr, 1e-9) * 4)}

    @staticmethod
    def _sym_tri(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        if len(ph) < 2 or len(pl) < 2: return None
        (i1, h1), (i2, h2) = ph[-2], ph[-1]
        (j1, l1), (j2, l2) = pl[-2], pl[-1]
        if not (h2 < h1 - tol * 0.5 and l2 > l1 + tol * 0.5): return None  # converging
        height = h1 - l1
        if height < min_h: return None
        if prev_close <= h2 and price > h2:
            return {"name": "Symmetrical Triangle", "direction": "LONG",
                    "neckline": h2, "height": height, "invalidation": l2,
                    "quality": 58 + min(10, height / max(atr, 1e-9) * 2)}
        if prev_close >= l2 and price < l2:
            return {"name": "Symmetrical Triangle", "direction": "SHORT",
                    "neckline": l2, "height": height, "invalidation": h2,
                    "quality": 58 + min(10, height / max(atr, 1e-9) * 2)}
        return None

    @staticmethod
    def _flag(ph, pl, highs, lows, closes, price, prev_close, tol, min_h, atr):
        """Flagpole = strong directional thrust over the last ~10 bars,
        then a tight consolidation, then current bar breaks consolidation
        in the pole's direction."""
        n = len(closes) - 1
        if n < 14: return None
        pole_lk = 8; cons_lk = 5
        pole_start = n - pole_lk - cons_lk
        pole_end = n - cons_lk
        if pole_start < 0: return None
        pole_move = closes[pole_end] - closes[pole_start]
        pole_h = abs(pole_move)
        if pole_h < max(2.0 * atr, min_h * 2): return None     # need a real pole
        cons_hi = highs[pole_end:n + 1].max()
        cons_lo = lows[pole_end:n + 1].min()
        cons_range = cons_hi - cons_lo
        if cons_range > pole_h * 0.6: return None              # consolidation must be tight
        if pole_move > 0:                                      # bull flag
            if prev_close <= cons_hi and price > cons_hi:
                return {"name": "Bull Flag", "direction": "LONG", "neckline": cons_hi,
                        "height": pole_h, "invalidation": cons_lo,
                        "quality": 60 + min(12, pole_h / max(atr, 1e-9) * 2)}
        else:                                                  # bear flag
            if prev_close >= cons_lo and price < cons_lo:
                return {"name": "Bear Flag", "direction": "SHORT", "neckline": cons_lo,
                        "height": pole_h, "invalidation": cons_hi,
                        "quality": 60 + min(12, pole_h / max(atr, 1e-9) * 2)}
        return None
