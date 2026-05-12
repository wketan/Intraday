"""
╔══════════════════════════════════════════════════════════════════╗
║  signal_v2.py — 3-of-4 confluence trend-momentum strategy         ║
║                                                                  ║
║  Based on aaryansinha16/AI-trader's two highest-WR strategies:    ║
║    - vwap_momentum_breakout (CALL)  — 100% WR (small N)           ║
║    - bearish_momentum (PUT)         — 68% WR  over 33 trades      ║
║                                                                  ║
║  CALL trigger (need ≥3 of 4):                                     ║
║    1. close > VWAP                                               ║
║    2. RSI(14) > 55                                               ║
║    3. EMA(20) > EMA(50)                                          ║
║    4. volume > 1.5 × 20-bar avg                                  ║
║                                                                  ║
║  PUT trigger (need ≥3 of 4, all inverted)                         ║
║                                                                  ║
║  Returns the SAME dict shape as v1 SignalGen.analyze(), so the    ║
║  rest of the pipeline (OptPicker, Slack, kill-switch, DB) works   ║
║  unchanged.                                                       ║
║                                                                  ║
║  No indicator-weight tuning. No 13-indicator soup. One rule,      ║
║  mechanically defined, backtest-verifiable.                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


class SignalGenV2:
    """3-of-4 confluence trend-momentum option-buying signal generator.

    Stateless — every call is independent. Returns either a signal dict
    (same shape as v1) or None.
    """

    # Score threshold for a valid signal (out of 4)
    TRIGGER_SCORE = 3

    # Confidence mapping: 3/4 = decent, 4/4 = strong
    CONFIDENCE = {3: 60, 4: 85}

    # SL / target distances on the underlying (ATR-multiples)
    SL_ATR_MULT  = 1.2
    T1_RR_MULT   = 1.5   # T1 = SL distance × 1.5 (so RR = 1.5)
    T2_RR_MULT   = 2.5   # T2 = SL distance × 2.5

    @staticmethod
    def analyze(df, **ignored_kwargs):
        """Score the latest 5-min bar against the 3-of-4 rule.

        Args:
            df: 5-min OHLCV DataFrame with columns [timestamp, open, high, low,
                close, volume]. Must have ≥30 bars.
            **ignored_kwargs: v1 took weight_adj, blocked_windows — accepted but
                              ignored so the dispatcher can pass them blindly.

        Returns:
            dict (same shape as v1 SignalGen.analyze) or None if no setup.
        """
        if df is None or len(df) < 30:
            return None

        # Reuse v1's TA helpers (they're proven and shared).
        from server import TA

        close = df["close"]
        n = len(df) - 1
        price = float(close.iloc[n])

        # ── Indicators ──────────────────────────────────────────────────
        vwap = TA.vwap(df)
        rsi = TA.rsi(close, 14)
        ema20 = TA.ema(close, 20)
        ema50 = TA.ema(close, 50)
        atr = TA.atr(df, 14)

        vwap_now = float(vwap.iloc[n])
        rsi_now = float(rsi.iloc[n]) if not _isnan(rsi.iloc[n]) else 50.0
        ema20_now = float(ema20.iloc[n])
        ema50_now = float(ema50.iloc[n])
        atr_now = float(atr.iloc[n]) if not _isnan(atr.iloc[n]) else 0.0

        # Volume relative to trailing 20-bar mean (excluding the current bar)
        vol_avg = float(df["volume"].iloc[max(0, n - 20):n].mean() or 1.0)
        vol_now = float(df["volume"].iloc[n])
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

        # ── Score the 4 conditions per direction ────────────────────────
        long_score = 0
        short_score = 0
        long_reasons = []
        short_reasons = []

        # 1. VWAP position
        if price > vwap_now:
            long_score += 1
            long_reasons.append(f"Close ₹{price:.0f} > VWAP ₹{vwap_now:.0f}")
        elif price < vwap_now:
            short_score += 1
            short_reasons.append(f"Close ₹{price:.0f} < VWAP ₹{vwap_now:.0f}")

        # 2. RSI(14) momentum zone
        if rsi_now > 55:
            long_score += 1
            long_reasons.append(f"RSI {rsi_now:.0f} > 55 (bullish momentum)")
        elif rsi_now < 45:
            short_score += 1
            short_reasons.append(f"RSI {rsi_now:.0f} < 45 (bearish momentum)")

        # 3. EMA(20) vs EMA(50) trend stack
        if ema20_now > ema50_now:
            long_score += 1
            long_reasons.append(f"EMA20 ₹{ema20_now:.0f} > EMA50 ₹{ema50_now:.0f}")
        elif ema20_now < ema50_now:
            short_score += 1
            short_reasons.append(f"EMA20 ₹{ema20_now:.0f} < EMA50 ₹{ema50_now:.0f}")

        # 4. Volume surge — attributed to whichever direction is otherwise winning
        if vol_ratio > 1.5:
            note = f"Volume {vol_ratio:.1f}× 20-bar avg (conviction)"
            if long_score > short_score:
                long_score += 1
                long_reasons.append(note)
            elif short_score > long_score:
                short_score += 1
                short_reasons.append(note)

        # ── Trigger check ───────────────────────────────────────────────
        if (long_score >= SignalGenV2.TRIGGER_SCORE
                and long_score > short_score):
            direction = "LONG"
            score = long_score
            reasons = long_reasons
        elif (short_score >= SignalGenV2.TRIGGER_SCORE
                and short_score > long_score):
            direction = "SHORT"
            score = short_score
            reasons = short_reasons
        else:
            return None   # No clear setup

        confidence = SignalGenV2.CONFIDENCE.get(score, 60)

        # ── Compute index-level SL/T1/T2 ────────────────────────────────
        # SL distance = 1.2 × ATR. T1 = 1.5 × SL distance. T2 = 2.5 × SL distance.
        # (OptPicker downstream will translate these into option-premium levels
        # via the OPT_EXIT_MODE config — either premium_pct or delta_scaled.)
        if atr_now <= 0:
            return None   # Can't size the trade without ATR

        sl_dist = atr_now * SignalGenV2.SL_ATR_MULT
        # Compute entry first; then size t1/t2 off the ACTUAL entry-to-SL distance
        # so RR is exactly T1_RR_MULT (not influenced by the entry offset from price).
        if direction == "LONG":
            entry = round(price + atr_now * 0.1, 2)
            sl    = round(price - sl_dist, 2)
            actual_risk = entry - sl
            t1    = round(entry + actual_risk * SignalGenV2.T1_RR_MULT, 2)
            t2    = round(entry + actual_risk * SignalGenV2.T2_RR_MULT, 2)
        else:
            entry = round(price - atr_now * 0.1, 2)
            sl    = round(price + sl_dist, 2)
            actual_risk = sl - entry
            t1    = round(entry - actual_risk * SignalGenV2.T1_RR_MULT, 2)
            t2    = round(entry - actual_risk * SignalGenV2.T2_RR_MULT, 2)

        risk = abs(entry - sl)
        reward = abs(t1 - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0

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
                "vwap":      round(vwap_now, 2),
                "rsi":       round(rsi_now, 1),
                "ema20":     round(ema20_now, 2),
                "ema50":     round(ema50_now, 2),
                "atr":       round(atr_now, 2),
                "vol_ratio": round(vol_ratio, 2),
                "score":     f"{score}/4",
            },
            "strategy":  "v2",
            "v2_score":  score,
            "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
        }


def _isnan(x) -> bool:
    """NaN check that works for both numpy and native floats without importing numpy."""
    try:
        return x != x
    except Exception:
        return False


# ─── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    import random

    # Synthesise a bullish trend bar series
    random.seed(42)
    n = 60
    base = 25000
    rows = []
    for i in range(n):
        # Drift up with noise
        base = base * (1 + random.gauss(0.0005, 0.002))
        o = base * (1 - random.uniform(0, 0.001))
        h = base * (1 + random.uniform(0, 0.002))
        l = base * (1 - random.uniform(0, 0.002))
        c = base * (1 + random.uniform(-0.001, 0.001))
        v = random.uniform(1_000_000, 3_000_000)
        rows.append({
            "timestamp": datetime.now(IST) - timedelta(minutes=(n - i) * 5),
            "open": o, "high": max(o, h, c), "low": min(o, l, c),
            "close": c, "volume": v,
        })
    # Force volume surge on last bar
    rows[-1]["volume"] = rows[-1]["volume"] * 3
    df = pd.DataFrame(rows)

    sig = SignalGenV2.analyze(df)
    print("Synthetic bullish-trend test:")
    if sig:
        print(f"  ✓ Signal: {sig['direction']} confidence={sig['confidence']}% score={sig['v2_score']}/4")
        print(f"    Price ₹{sig['price']}  Entry ₹{sig['entry']}  SL ₹{sig['sl']}  T1 ₹{sig['target1']}")
        print(f"    R:R = {sig['risk_reward']}")
        print(f"    Reasons: {sig['reasons']}")
    else:
        print(f"  No signal fired (this can happen on random data).")
