"""
╔══════════════════════════════════════════════════════════════════╗
║  signal_v2.py — 3-of-4 confluence trend-momentum strategy         ║
║  v2.1 — fixed-for-indices                                         ║
║                                                                  ║
║  HISTORY                                                          ║
║   v2.0 (2026-05-11): used volume > 1.5× avg as condition 4.       ║
║                       Bug: NIFTY/BANKNIFTY are indices, Angel One ║
║                       returns volume=0. Condition NEVER triggered.║
║                       Max possible score = 3/4. RSI 45-55 zone    ║
║                       killed every other signal. Two days live,   ║
║                       zero alerts.                                ║
║   v2.1 (2026-05-12): replaced volume with RANGE EXPANSION (close- ║
║                       to-close volatility surrogate that DOES work║
║                       on indices). Loosened RSI 55/45 → 52/48.    ║
║                       Added VWAP-deviation noise filter.          ║
║                       Always returns diagnostics so the engine    ║
║                       can log WHY each bar didn't fire.           ║
║                                                                  ║
║  THE 4 CONDITIONS (now)                                           ║
║   1. CLOSE vs VWAP — price must deviate from VWAP by ≥ 0.05% in   ║
║       the trade direction (filters mean-reversion at VWAP)        ║
║   2. RSI(14) directional — > 52 for LONG, < 48 for SHORT          ║
║   3. EMA(20) vs EMA(50) — stacked in trade direction              ║
║   4. RANGE EXPANSION — current bar's (high-low) > 1.2× avg of     ║
║       last 20 bars' (high-low). Works without volume data because ║
║       price-range expansion is a robust momentum signal.          ║
║                                                                  ║
║  Returns SAME dict shape as v1 SignalGen.analyze() so OptPicker,  ║
║  Slack, PLTracker, kill-switch all work unchanged.                ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))


def _env_float(key: str, default: float) -> float:
    try:    return float(os.environ.get(key, default))
    except: return default


def _env_int(key: str, default: int) -> int:
    try:    return int(os.environ.get(key, default))
    except: return default


class SignalGenV2:
    """Stateless. Every call independent. Returns either a signal dict
    (when the 3-of-4 threshold is met) or None, plus emits diagnostics
    via `last_decision` so the engine can log every bar's score.

    All thresholds are env-tunable. Sensible defaults below.
    """

    @staticmethod
    def _config() -> dict:
        """All knobs in one place. Read fresh each call so dashboard /
        engine_state mutations apply without a restart."""
        return {
            "trigger_score":     _env_int  ("V2_TRIGGER_SCORE",     3),
            "rsi_long":          _env_float("V2_RSI_LONG",         52),
            "rsi_short":         _env_float("V2_RSI_SHORT",        48),
            "vwap_dev_pct":      _env_float("V2_VWAP_DEV_PCT",   0.05),  # 0.05% min deviation
            "range_mult":        _env_float("V2_RANGE_MULT",      1.2),  # 1.2× 20-bar mean range
            "atr_sl_mult":       _env_float("V2_ATR_SL_MULT",     1.2),
            "rr_target":         _env_float("V2_RR_TARGET",       1.5),
            "rr_target2":        _env_float("V2_RR_TARGET2",      2.5),
        }

    # Last-decision diagnostic dict, keyed per process. The engine reads this
    # after every analyze() call (signal or not) to log why a bar fired/skipped.
    last_decision: dict = {}

    @staticmethod
    def analyze(df, **ignored_kwargs):
        """Score the latest 5-min bar.

        Returns a signal dict on trigger, else None. EITHER WAY, populates
        `SignalGenV2.last_decision` with the score breakdown so the engine
        loop can log every scan tick — no more black-box silence.
        """
        cfg = SignalGenV2._config()

        if df is None or len(df) < 30:
            SignalGenV2.last_decision = {"verdict": "INSUFFICIENT_BARS", "bars": len(df) if df is not None else 0}
            return None

        from server import TA

        close = df["close"]
        n = len(df) - 1
        price = float(close.iloc[n])

        # ── Indicators ──────────────────────────────────────────────────
        vwap   = TA.vwap(df)
        rsi    = TA.rsi(close, 14)
        ema20  = TA.ema(close, 20)
        ema50  = TA.ema(close, 50)
        atr    = TA.atr(df, 14)

        vwap_now  = float(vwap.iloc[n])
        rsi_now   = float(rsi.iloc[n])     if not _isnan(rsi.iloc[n])   else 50.0
        ema20_now = float(ema20.iloc[n])
        ema50_now = float(ema50.iloc[n])
        atr_now   = float(atr.iloc[n])     if not _isnan(atr.iloc[n])   else 0.0

        # ── Volume-substitute: range expansion ──────────────────────────
        bar_ranges = (df["high"] - df["low"]).iloc[max(0, n - 20):n]
        avg_range  = float(bar_ranges.mean()) if len(bar_ranges) else 0.0
        cur_range  = float(df["high"].iloc[n] - df["low"].iloc[n])
        range_ratio = (cur_range / avg_range) if avg_range > 0 else 1.0

        # ── Score the 4 conditions per direction ────────────────────────
        long_score = 0
        short_score = 0
        long_checks: dict[str, bool] = {}
        short_checks: dict[str, bool] = {}

        # 1. VWAP position with noise filter
        vwap_dev_abs = (price - vwap_now) / vwap_now * 100.0 if vwap_now else 0.0
        long_checks["close_above_vwap"]  = vwap_dev_abs >  cfg["vwap_dev_pct"]
        short_checks["close_below_vwap"] = vwap_dev_abs < -cfg["vwap_dev_pct"]
        if long_checks["close_above_vwap"]:  long_score  += 1
        if short_checks["close_below_vwap"]: short_score += 1

        # 2. RSI directional
        long_checks["rsi_bullish"]  = rsi_now > cfg["rsi_long"]
        short_checks["rsi_bearish"] = rsi_now < cfg["rsi_short"]
        if long_checks["rsi_bullish"]:  long_score  += 1
        if short_checks["rsi_bearish"]: short_score += 1

        # 3. EMA stack
        long_checks["ema_bullish"]  = ema20_now > ema50_now
        short_checks["ema_bearish"] = ema20_now < ema50_now
        if long_checks["ema_bullish"]:  long_score  += 1
        if short_checks["ema_bearish"]: short_score += 1

        # 4. Range expansion — works on indices (unlike volume)
        range_expanding = range_ratio > cfg["range_mult"]
        # Attribute to whichever direction already has score > 0
        if range_expanding:
            if long_score > short_score:
                long_score += 1
                long_checks["range_expansion"] = True
            elif short_score > long_score:
                short_score += 1
                short_checks["range_expansion"] = True
            else:
                long_checks["range_expansion"] = False
                short_checks["range_expansion"] = False
        else:
            long_checks["range_expansion"] = False
            short_checks["range_expansion"] = False

        trigger = cfg["trigger_score"]

        # ── Build diagnostics regardless of fire/skip ───────────────────
        diag = {
            "ts":         datetime.now(IST).strftime("%H:%M:%S"),
            "price":      round(price, 2),
            "vwap":       round(vwap_now, 2),
            "vwap_dev_pct": round(vwap_dev_abs, 3),
            "rsi":        round(rsi_now, 1),
            "ema20":      round(ema20_now, 2),
            "ema50":      round(ema50_now, 2),
            "atr":        round(atr_now, 2),
            "range_ratio": round(range_ratio, 2),
            "long_score":  long_score,
            "short_score": short_score,
            "trigger":     trigger,
            "long_checks":  long_checks,
            "short_checks": short_checks,
        }

        # ── Trigger ─────────────────────────────────────────────────────
        if long_score >= trigger and long_score > short_score:
            direction = "LONG"; score = long_score; checks = long_checks
        elif short_score >= trigger and short_score > long_score:
            direction = "SHORT"; score = short_score; checks = short_checks
        else:
            diag["verdict"] = f"NO_TRIGGER (long={long_score} short={short_score} need≥{trigger})"
            SignalGenV2.last_decision = diag
            return None

        # ── Confidence: 3/4 = 60, 4/4 = 85 ─────────────────────────────
        confidence = 60 if score == 3 else 85
        if score > 4: confidence = 90

        # ── Index-level SL/T1/T2 (premium-pct mode in OptPicker will translate) ─
        if atr_now <= 0:
            diag["verdict"] = "NO_ATR"
            SignalGenV2.last_decision = diag
            return None

        sl_dist = atr_now * cfg["atr_sl_mult"]
        if direction == "LONG":
            entry = round(price + atr_now * 0.1, 2)
            sl    = round(price - sl_dist, 2)
            actual_risk = entry - sl
            t1    = round(entry + actual_risk * cfg["rr_target"], 2)
            t2    = round(entry + actual_risk * cfg["rr_target2"], 2)
        else:
            entry = round(price - atr_now * 0.1, 2)
            sl    = round(price + sl_dist, 2)
            actual_risk = sl - entry
            t1    = round(entry - actual_risk * cfg["rr_target"], 2)
            t2    = round(entry - actual_risk * cfg["rr_target2"], 2)

        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

        # Reasons for Slack
        reasons = []
        if direction == "LONG":
            if checks.get("close_above_vwap"): reasons.append(f"Close ₹{price:.0f} > VWAP ₹{vwap_now:.0f} ({vwap_dev_abs:+.2f}%)")
            if checks.get("rsi_bullish"):       reasons.append(f"RSI {rsi_now:.0f} > {cfg['rsi_long']:.0f}")
            if checks.get("ema_bullish"):       reasons.append(f"EMA20 ₹{ema20_now:.0f} > EMA50 ₹{ema50_now:.0f}")
            if checks.get("range_expansion"):   reasons.append(f"Range {range_ratio:.1f}× 20-bar avg")
        else:
            if checks.get("close_below_vwap"): reasons.append(f"Close ₹{price:.0f} < VWAP ₹{vwap_now:.0f} ({vwap_dev_abs:+.2f}%)")
            if checks.get("rsi_bearish"):       reasons.append(f"RSI {rsi_now:.0f} < {cfg['rsi_short']:.0f}")
            if checks.get("ema_bearish"):       reasons.append(f"EMA20 ₹{ema20_now:.0f} < EMA50 ₹{ema50_now:.0f}")
            if checks.get("range_expansion"):   reasons.append(f"Range {range_ratio:.1f}× 20-bar avg")

        diag["verdict"] = f"TRIGGER {direction} {score}/4"
        SignalGenV2.last_decision = diag

        return {
            "direction":    direction,
            "confidence":   confidence,
            "price":        round(price, 2),
            "entry":        entry,
            "sl":           sl,
            "target1":      t1,
            "target2":      t2,
            "risk":         round(risk, 2),
            "reward":       round(reward, 2),
            "risk_reward":  rr,
            "reasons":      reasons,
            "indicators": {
                "vwap":         round(vwap_now, 2),
                "vwap_dev_pct": round(vwap_dev_abs, 3),
                "rsi":          round(rsi_now, 1),
                "ema20":        round(ema20_now, 2),
                "ema50":        round(ema50_now, 2),
                "atr":          round(atr_now, 2),
                "range_ratio":  round(range_ratio, 2),
                "score":        f"{score}/4",
            },
            "strategy":   "v2.1",
            "v2_score":   score,
            "v2_diag":    diag,
            "timestamp":  datetime.now(IST).strftime("%H:%M:%S"),
        }


def _isnan(x) -> bool:
    try:    return x != x
    except: return False


# ─── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    import random

    print("=" * 60)
    print("signal_v2.1 self-test — three regimes")
    print("=" * 60)

    def _build(seed: int, drift: float, vol: float, n: int = 60):
        random.seed(seed)
        base = 25000.0
        rows = []
        for i in range(n):
            base *= (1 + random.gauss(drift, vol))
            o = base * (1 - random.uniform(0, 0.001))
            h = base * (1 + random.uniform(0, 0.003))
            l = base * (1 - random.uniform(0, 0.003))
            c = base * (1 + random.uniform(-0.001, 0.001))
            rows.append({
                "timestamp": datetime.now(IST) - timedelta(minutes=(n - i) * 5),
                "open": o, "high": max(o, h, c), "low": min(o, l, c),
                "close": c, "volume": 0,   # ← INDEX: zero volume, the bug fixer
            })
        # Force a range-expansion on the last bar
        rows[-1]["high"] = rows[-1]["high"] * 1.005
        rows[-1]["low"]  = rows[-1]["low"]  * 0.995
        return pd.DataFrame(rows)

    for label, seed, drift, vol in [
        ("Bullish trend",  42, +0.0007, 0.0015),
        ("Bearish trend",   7, -0.0007, 0.0015),
        ("Choppy/sideways", 3,  0.0000, 0.0025),
    ]:
        df = _build(seed, drift, vol)
        sig = SignalGenV2.analyze(df)
        d = SignalGenV2.last_decision
        if sig:
            print(f"\n● {label}:  🟢 {sig['direction']} score={sig['v2_score']}/4  conf={sig['confidence']}%  RR={sig['risk_reward']}")
            print(f"   {'; '.join(sig['reasons'])}")
        else:
            print(f"\n● {label}:  🔴 NO SIGNAL  →  long={d.get('long_score')} short={d.get('short_score')}")
            print(f"   long_checks:  {d.get('long_checks')}")
            print(f"   short_checks: {d.get('short_checks')}")
            print(f"   diag: rsi={d.get('rsi')} vwap_dev%={d.get('vwap_dev_pct')} range_ratio={d.get('range_ratio')}")
