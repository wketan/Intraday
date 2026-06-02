"""Reverter — mean-reversion intraday strategy for range-bound regimes.

This is Conductor's complement. Conductor looks for trend continuation
and confluence-driven breakouts; Reverter looks for over-extensions
that reverse back to the mean (VWAP).

Why a separate module
─────────────────────
The 90d Feb-May 2026 validation showed Conductor lost money on NIFTY
(only worked on BANKNIFTY). ORB, Scalper, and v2 all also underperformed
on NIFTY. The common thread: NIFTY in this regime ranges 200-350 pts/day
without clean directional follow-through. Trend strategies enter near
the top/bottom of the range and get faded.

Mean reversion targets the OPPOSITE setup: wait for price to extend
beyond a normal-distance band from VWAP, confirm with an RSI extreme
AND a reversal candle, then take the trade BACK toward VWAP.

Edge sources (literature + Indian-market practice):
  1. VWAP is a magnet on range-bound days. Stat: NIFTY closes within
     0.3% of VWAP on roughly 60% of range-bound days.
  2. RSI > 75 or < 25 in the 5-min timeframe marks exhaustion, not
     continuation, on non-trending days.
  3. Reversal candle (engulfing or pin-bar / long wick) at extreme +
     VWAP-extension is a higher-probability fade than RSI alone.

Hard rules
──────────
  • Time gate: 09:45 ≤ t < 14:30 IST. Skip the volatile open and the
    EOD theta-decay window.
  • Direction LONG fires when:
      - price < VWAP × (1 - extension_pct)  AND
      - RSI ≤ rsi_oversold  AND
      - current bar is bullish reversal (close>open AND lower-wick
        > 1.5× body AND lower-wick > 30% of range)  AND
      - day's high-low range ≥ min_range_pts (need volatility to play out)
      - NOT already at day's absolute low (need exhaustion sign, not collapse)
  • Direction SHORT mirrors above
  • SL: tighter of (recent swing high/low) or (0.35% from entry)
  • T1: return to VWAP (the mean target)
  • T2: opposite side of day's range (capped at 2× T1 distance)
  • Per-instrument max 2 trades/day (built into per-day dedup in caller)

Estimated ₹ profit (1 lot NIFTY = 75)
─────────────────────────────────────
For NIFTY at ATM, option delta ≈ 0.5 means:
  T1 distance = VWAP - entry (in spot pts)
  Estimated option premium move ≈ T1_dist × 0.5
  Estimated profit per lot = T1_dist × 0.5 × 75

For a reasonable trade: spot extends 30-60 pts from VWAP →
T1_dist = 30-60 → est_profit = 1,125-2,250 per lot. Hits the ₹1k
per-trade target user set.

Mirrors the SignalGen interface so it plugs into both the live engine
dispatch (via signal_v2-style dispatcher) and the backtest_v2 framework.
"""

from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone

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


# Tunable knobs (env-overridable)
REV_EXTENSION_PCT     = _env_float("REVERTER_EXTENSION_PCT",     0.40)   # how far below VWAP for LONG, above for SHORT
REV_RSI_OVERSOLD      = _env_int  ("REVERTER_RSI_OVERSOLD",        32)
REV_RSI_OVERBOUGHT    = _env_int  ("REVERTER_RSI_OVERBOUGHT",      68)
REV_MIN_RANGE_PCT     = _env_float("REVERTER_MIN_RANGE_PCT",     0.40)   # day's range as % of open
REV_TARGET_PROFIT_RS  = _env_float("REVERTER_TARGET_PROFIT_RS", 1000.0)
REV_SL_PCT            = _env_float("REVERTER_SL_PCT",            0.35)
REV_EARLIEST_H        = _env_int  ("REVERTER_EARLIEST_H",           9)
REV_EARLIEST_M        = _env_int  ("REVERTER_EARLIEST_M",          45)
REV_LATEST_H          = _env_int  ("REVERTER_LATEST_H",            14)
REV_LATEST_M          = _env_int  ("REVERTER_LATEST_M",            30)


def _isnan(x) -> bool:
    try:    return x != x
    except: return False


class Reverter:
    """Stateless analyze() per bar — returns signal dict or None.

    Same interface as Conductor / SignalGenV2 / SignalGenORB so the
    dispatcher in SignalGen.analyze can route to it transparently.
    """

    last_decision: dict = {}

    @staticmethod
    def _set_decision(d: dict):
        Reverter.last_decision = d

    @staticmethod
    def _ema(s, period: int):
        return s.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _rsi(s, period: int = 14):
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _vwap(df):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        vol = vol.replace(0, 1).fillna(1)
        return (tp * vol).cumsum() / vol.cumsum()

    @staticmethod
    def analyze(df, symbol: str = "", chain_analytics: dict = None,
                **ignored):
        if df is None or len(df) < 30:
            Reverter._set_decision({"verdict": "INSUFFICIENT_BARS",
                                     "bars": len(df) if df is not None else 0})
            return None
        if pd is None:
            Reverter._set_decision({"verdict": "PANDAS_MISSING"})
            return None

        # Time gate
        n = len(df) - 1
        cur_ts = df["ts"].iloc[n]
        if isinstance(cur_ts, str):
            cur_ts = pd.to_datetime(cur_ts)
        try:
            cur_ts = cur_ts.to_pydatetime() if hasattr(cur_ts, "to_pydatetime") else cur_ts
        except Exception:
            pass
        cur_time = cur_ts.time() if hasattr(cur_ts, "time") else None
        cur_date = cur_ts.date() if hasattr(cur_ts, "date") else None
        earliest = time(REV_EARLIEST_H, REV_EARLIEST_M)
        latest = time(REV_LATEST_H, REV_LATEST_M)
        if cur_time is None or cur_date is None:
            Reverter._set_decision({"verdict": "BAD_TIMESTAMP"})
            return None
        if cur_time < earliest:
            Reverter._set_decision({"verdict": "BEFORE_WINDOW", "time": cur_time.strftime("%H:%M")})
            return None
        if cur_time >= latest:
            Reverter._set_decision({"verdict": "AFTER_WINDOW", "time": cur_time.strftime("%H:%M")})
            return None

        # Today's bars (need enough for VWAP to stabilize)
        ts_series = pd.to_datetime(df["ts"])
        today_mask = ts_series.dt.date == cur_date
        today_bars = df.loc[today_mask]
        if len(today_bars) < 6:
            Reverter._set_decision({"verdict": "INSUFFICIENT_TODAY_BARS",
                                     "bars": len(today_bars)})
            return None

        # Build today's VWAP from today's bars only (true intraday VWAP)
        tdy_vwap = Reverter._vwap(today_bars)
        vwap_now = float(tdy_vwap.iloc[-1])

        close = df["close"]
        price = float(close.iloc[n])
        rsi = Reverter._rsi(close, 14)
        rsi_now = float(rsi.iloc[n]) if not _isnan(rsi.iloc[n]) else 50.0

        day_open = float(today_bars["open"].iloc[0])
        day_high = float(today_bars["high"].max())
        day_low  = float(today_bars["low"].min())
        day_range = day_high - day_low
        day_range_pct = (day_range / day_open * 100.0) if day_open > 0 else 0.0

        # Need enough intraday range to mean-revert into
        if day_range_pct < REV_MIN_RANGE_PCT:
            Reverter._set_decision({
                "verdict": "RANGE_TOO_TIGHT",
                "day_range_pct": round(day_range_pct, 3),
                "min": REV_MIN_RANGE_PCT,
            })
            return None

        # Current bar shape
        cur_open  = float(df["open"].iloc[n])
        cur_close = float(df["close"].iloc[n])
        cur_high  = float(df["high"].iloc[n])
        cur_low   = float(df["low"].iloc[n])
        cur_body  = abs(cur_close - cur_open)
        cur_range = cur_high - cur_low
        if cur_range <= 0:
            Reverter._set_decision({"verdict": "ZERO_RANGE_BAR"})
            return None

        upper_wick = cur_high - max(cur_close, cur_open)
        lower_wick = min(cur_close, cur_open) - cur_low

        # Reversal candle tests:
        # LONG: bullish close, big lower wick (rejected lower prices)
        bullish_reversal = (
            cur_close > cur_open and
            cur_body > 0 and
            lower_wick > 1.5 * cur_body and
            lower_wick > 0.30 * cur_range
        )
        # SHORT: bearish close, big upper wick (rejected higher prices)
        bearish_reversal = (
            cur_close < cur_open and
            cur_body > 0 and
            upper_wick > 1.5 * cur_body and
            upper_wick > 0.30 * cur_range
        )

        # VWAP extension test
        vwap_dev_pct = ((price - vwap_now) / vwap_now * 100.0) if vwap_now > 0 else 0.0
        extended_low  = vwap_dev_pct < -REV_EXTENSION_PCT
        extended_high = vwap_dev_pct >  REV_EXTENSION_PCT

        # RSI test
        rsi_oversold   = rsi_now <= REV_RSI_OVERSOLD
        rsi_overbought = rsi_now >= REV_RSI_OVERBOUGHT

        # Exhaustion (not collapse): current bar shouldn't be at the *exact* day extreme.
        # If price = day_low exactly (within 2 pts), still falling — wait for confirmation.
        EXACT_EXTREME_TOLERANCE = max(2.0, 0.0005 * price)
        at_day_low  = (price - day_low)  < EXACT_EXTREME_TOLERANCE
        at_day_high = (day_high - price) < EXACT_EXTREME_TOLERANCE

        long_setup  = extended_low  and rsi_oversold   and bullish_reversal and not at_day_low
        short_setup = extended_high and rsi_overbought and bearish_reversal and not at_day_high

        if not long_setup and not short_setup:
            Reverter._set_decision({
                "verdict": "NO_REVERSAL_SETUP",
                "vwap_dev_pct": round(vwap_dev_pct, 3),
                "rsi": round(rsi_now, 1),
                "bullish_reversal": bullish_reversal,
                "bearish_reversal": bearish_reversal,
                "extended_low": extended_low,
                "extended_high": extended_high,
                "day_range_pct": round(day_range_pct, 3),
            })
            return None

        direction = "LONG" if long_setup else "SHORT"
        entry = round(price, 2)

        # SL: tighter of recent swing or fixed pct
        SWING_LOOKBACK = 5
        prior = today_bars.iloc[-SWING_LOOKBACK-1:-1] if len(today_bars) > SWING_LOOKBACK else today_bars
        recent_swing_low  = float(prior["low"].min())  if len(prior) else cur_low
        recent_swing_high = float(prior["high"].max()) if len(prior) else cur_high

        if direction == "LONG":
            # SL below current bar's low or recent swing low — whichever is tighter
            sl_candidate_swing = min(recent_swing_low, cur_low) - 1.0
            sl_candidate_pct   = entry * (1 - REV_SL_PCT/100.0)
            sl = round(max(sl_candidate_swing, sl_candidate_pct), 2)
            # Sanity: SL must be below entry
            if sl >= entry:
                sl = round(entry * (1 - REV_SL_PCT/100.0), 2)
            risk = entry - sl
            t1 = round(vwap_now, 2)   # mean target
            if t1 <= entry:
                Reverter._set_decision({"verdict": "VWAP_BELOW_ENTRY_LONG",
                                         "vwap": round(vwap_now,2), "entry": entry})
                return None
            reward = t1 - entry
            t2_dist = min(reward * 1.5, (day_high - entry))
            t2 = round(entry + max(reward, t2_dist), 2)
        else:
            sl_candidate_swing = max(recent_swing_high, cur_high) + 1.0
            sl_candidate_pct   = entry * (1 + REV_SL_PCT/100.0)
            sl = round(min(sl_candidate_swing, sl_candidate_pct), 2)
            if sl <= entry:
                sl = round(entry * (1 + REV_SL_PCT/100.0), 2)
            risk = sl - entry
            t1 = round(vwap_now, 2)
            if t1 >= entry:
                Reverter._set_decision({"verdict": "VWAP_ABOVE_ENTRY_SHORT",
                                         "vwap": round(vwap_now,2), "entry": entry})
                return None
            reward = entry - t1
            t2_dist = min(reward * 1.5, (entry - day_low))
            t2 = round(entry - max(reward, t2_dist), 2)

        rr = round(reward / risk, 2) if risk > 0 else 0
        # Mean reversion: don't require 2:1 R:R. VWAP fade typically gives 1:1
        # to 1.5:1, but win rate is higher than trend strategies.
        MIN_RR = float(os.environ.get("REVERTER_MIN_RR", "1.1"))
        if rr < MIN_RR:
            Reverter._set_decision({"verdict": "RR_BELOW_MIN", "rr": rr, "min": MIN_RR})
            return None

        # ₹ profit estimate (delta ≈ 0.5 ATM, lot=75 NIFTY)
        lot_size = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 65}.get(symbol.upper(), 75)
        est_t1_profit = abs(t1 - entry) * 0.5 * lot_size
        if est_t1_profit < REV_TARGET_PROFIT_RS:
            Reverter._set_decision({
                "verdict": "T1_BELOW_TARGET_RS",
                "est_t1_profit": round(est_t1_profit),
                "target": REV_TARGET_PROFIT_RS,
                "t1_dist": round(abs(t1 - entry), 2),
            })
            return None

        confidence = 65
        if abs(vwap_dev_pct) > REV_EXTENSION_PCT * 1.5: confidence += 5
        if (direction == "LONG" and rsi_now < 25) or (direction == "SHORT" and rsi_now > 75): confidence += 5
        if rr >= 1.5: confidence += 5
        confidence = min(85, confidence)

        reasons = [
            f"VWAP {direction.lower()} fade: price {vwap_dev_pct:+.2f}% vs VWAP",
            f"RSI {rsi_now:.0f} ({'oversold' if direction == 'LONG' else 'overbought'})",
            f"Reversal candle: {'bullish' if direction == 'LONG' else 'bearish'} wick {lower_wick if direction == 'LONG' else upper_wick:.1f}pts > 1.5× body",
            f"Day range {day_range_pct:.2f}% — room to revert",
            f"T1 = VWAP @ {t1:.2f} ({reward:.1f}pts away, R:R {rr})",
        ]

        diag = {
            "verdict": f"TRIGGER {direction}",
            "rr": rr,
            "vwap_dev_pct": round(vwap_dev_pct, 3),
            "rsi": round(rsi_now, 1),
            "day_range_pct": round(day_range_pct, 3),
            "est_t1_profit": round(est_t1_profit),
        }
        Reverter._set_decision(diag)

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
                "rsi":           round(rsi_now, 1),
                "vwap":          round(vwap_now, 2),
                "vwap_dev_pct":  round(vwap_dev_pct, 3),
                "day_range_pct": round(day_range_pct, 3),
                "lower_wick":    round(lower_wick, 2),
                "upper_wick":    round(upper_wick, 2),
                "est_t1_rs":     round(est_t1_profit),
            },
            "strategy":   "reverter",
            "v2_score":   3,
            "v2_diag":    diag,
            "timestamp":  cur_ts.strftime("%H:%M:%S") if hasattr(cur_ts, "strftime") else "",
        }
