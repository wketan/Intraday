"""ScalperV3 — price-action scalping for NIFTY (and any low-vol index).

Why a v3
────────
v1 (raw MACD/EMA crossover) and v2 (Sahi 8-filter stack) both ran the
crossover-of-indicator triggers. Both failed on NIFTY in the 90d Feb-May
2026 sample:
  v1: 28.7% WR, -₹73,857 over 171 trades
  v2: 28.6% WR, -₹3,369 over 35 trades after the filters

The diagnosis: 8 filters cut 80% of v1's signals AND kept the same 28.6%
win rate. That means the filters remove good and bad signals at equal
rates — the underlying MACD/EMA crossover signal itself doesn't have
positive edge on NIFTY in this regime.

v3 changes the TRIGGER, not the filter. Crossovers are noise on NIFTY's
choppy 5-min bars; the entry signals that DO have documented edge are
price-action and structural:

  1. VWAP bounce — price touches VWAP from above/below and shows a
     reversal candle + volume. VWAP is a magnet on range-bound days.

  2. Yesterday's level bounce — price tests yesterday's high or low
     (now today's S/R) and reverses. Documented standard among Indian
     intraday traders.

  3. 3-bar reversal at intraday extreme — 3 same-color bars in a row
     followed by an opposing bar with body >50% of range. Classic
     exhaustion-then-reversal pattern.

ANY ONE of the three triggers fires a signal — orthogonal triggers,
not stacked filters.

Targets are tight (matches user's stated 5-15 min hold, ~10 pt move
playbook):
  SL = 7 pts | T1 = 14 pts (2:1 R:R) | T2 = 21 pts

At NIFTY ATM (delta ≈ 0.5, lot = 75):
  T1 profit per lot ≈ 14 × 0.5 × 75 = ₹525
  T2 profit per lot ≈ 21 × 0.5 × 75 = ₹790

These are SCALP profits — the strategy targets 2-4 trades/day at 1 lot
to hit ₹1.5-2k/day on NIFTY scalps. Stacking lots scales linearly.

Mirrors Conductor / Reverter interface for transparent dispatch.
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


# Tunable scalp knobs
SCALP_SL_PTS              = _env_float("SCALPER_V3_SL_PTS",            7.0)
SCALP_T1_PTS              = _env_float("SCALPER_V3_T1_PTS",           14.0)
SCALP_T2_PTS              = _env_float("SCALPER_V3_T2_PTS",           21.0)
SCALP_MIN_DAY_RANGE_PCT   = _env_float("SCALPER_V3_MIN_RANGE_PCT",   0.30)
SCALP_TARGET_PROFIT_RS    = _env_float("SCALPER_V3_TARGET_PROFIT_RS", 400.0)
SCALP_VOL_MULT            = _env_float("SCALPER_V3_VOL_MULT",         1.15)
SCALP_VWAP_TOLERANCE_PCT  = _env_float("SCALPER_V3_VWAP_TOL_PCT",    0.10)
SCALP_PIVOT_TOLERANCE_PCT = _env_float("SCALPER_V3_PIVOT_TOL_PCT",   0.10)
SCALP_EARLIEST_H          = _env_int  ("SCALPER_V3_EARLIEST_H",         9)
SCALP_EARLIEST_M          = _env_int  ("SCALPER_V3_EARLIEST_M",        30)
SCALP_LATEST_H            = _env_int  ("SCALPER_V3_LATEST_H",          14)
SCALP_LATEST_M            = _env_int  ("SCALPER_V3_LATEST_M",          30)


def _isnan(x) -> bool:
    try:    return x != x
    except: return False


class ScalperV3:
    """Stateless analyze() — returns signal dict or None.

    Same interface as Conductor / SignalGenV2 / Reverter / SignalGenORB.
    """

    last_decision: dict = {}

    @staticmethod
    def _set_decision(d: dict):
        ScalperV3.last_decision = d

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
            ScalperV3._set_decision({"verdict": "INSUFFICIENT_BARS",
                                       "bars": len(df) if df is not None else 0})
            return None
        if pd is None:
            ScalperV3._set_decision({"verdict": "PANDAS_MISSING"})
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
        cur_date = cur_ts.date() if hasattr(cur_ts, "date") else None
        if cur_time is None or cur_date is None:
            ScalperV3._set_decision({"verdict": "BAD_TIMESTAMP"})
            return None
        earliest = time(SCALP_EARLIEST_H, SCALP_EARLIEST_M)
        latest = time(SCALP_LATEST_H, SCALP_LATEST_M)
        if cur_time < earliest:
            ScalperV3._set_decision({"verdict": "BEFORE_WINDOW", "time": cur_time.strftime("%H:%M")})
            return None
        if cur_time >= latest:
            ScalperV3._set_decision({"verdict": "AFTER_WINDOW", "time": cur_time.strftime("%H:%M")})
            return None

        # Today's bars
        ts_series = pd.to_datetime(df["ts"])
        today_mask = ts_series.dt.date == cur_date
        today_bars = df.loc[today_mask]
        if len(today_bars) < 6:
            ScalperV3._set_decision({"verdict": "INSUFFICIENT_TODAY_BARS", "bars": len(today_bars)})
            return None

        # Yesterday's high/low (last bar not today)
        not_today = df.loc[~today_mask]
        if len(not_today) >= 6:
            # take "yesterday" = the most recent prior date
            prev_date = pd.to_datetime(not_today["ts"]).dt.date.iloc[-1]
            yest = not_today.loc[pd.to_datetime(not_today["ts"]).dt.date == prev_date]
            prev_high = float(yest["high"].max()) if len(yest) else None
            prev_low  = float(yest["low"].min())  if len(yest) else None
        else:
            prev_high = prev_low = None

        # Indicators on today's bars
        tdy_vwap = ScalperV3._vwap(today_bars)
        vwap_now = float(tdy_vwap.iloc[-1])

        price = float(df["close"].iloc[n])
        day_open = float(today_bars["open"].iloc[0])
        day_high = float(today_bars["high"].max())
        day_low  = float(today_bars["low"].min())
        day_range_pct = ((day_high - day_low) / day_open * 100.0) if day_open > 0 else 0.0
        if day_range_pct < SCALP_MIN_DAY_RANGE_PCT:
            ScalperV3._set_decision({
                "verdict": "RANGE_TOO_TIGHT",
                "day_range_pct": round(day_range_pct, 3),
                "min": SCALP_MIN_DAY_RANGE_PCT,
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
            ScalperV3._set_decision({"verdict": "ZERO_RANGE_BAR"})
            return None
        body_pct  = cur_body / cur_range

        upper_wick = cur_high - max(cur_close, cur_open)
        lower_wick = min(cur_close, cur_open) - cur_low
        is_bullish_bar = cur_close > cur_open
        is_bearish_bar = cur_close < cur_open

        # Volume confirmation (this bar vs prior 10-bar avg)
        recent = df.iloc[max(0, n-10):n]
        avg_vol = float(recent["volume"].mean()) if len(recent) and "volume" in recent.columns else 0.0
        cur_vol = float(df["volume"].iloc[n]) if "volume" in df.columns else 0.0
        vol_ratio = (cur_vol / avg_vol) if avg_vol > 0 else 1.0
        vol_ok = vol_ratio >= SCALP_VOL_MULT

        # ── Trigger 1: VWAP bounce ────────────────────────────────────
        vwap_dev_pct = ((price - vwap_now) / vwap_now * 100.0) if vwap_now > 0 else 0.0
        near_vwap = abs(vwap_dev_pct) <= SCALP_VWAP_TOLERANCE_PCT
        # LONG: price dipped below VWAP intra-bar then closed above → bounce up
        vwap_bounce_long  = (near_vwap and is_bullish_bar and lower_wick > 0.5 * cur_body
                              and cur_low < vwap_now * (1 - 0.0005))
        # SHORT: price popped above VWAP intra-bar then closed below → bounce down
        vwap_bounce_short = (near_vwap and is_bearish_bar and upper_wick > 0.5 * cur_body
                              and cur_high > vwap_now * (1 + 0.0005))

        # ── Trigger 2: Yesterday's level bounce ───────────────────────
        pivot_long  = pivot_short = False
        pivot_note  = "no prev day"
        if prev_low is not None and prev_low > 0:
            pl_tol = prev_low * SCALP_PIVOT_TOLERANCE_PCT / 100.0
            if abs(price - prev_low) <= pl_tol and is_bullish_bar and lower_wick > 0.4 * cur_body:
                pivot_long = True
                pivot_note = f"bounced off yesterday low {prev_low:.1f}"
        if prev_high is not None and prev_high > 0:
            ph_tol = prev_high * SCALP_PIVOT_TOLERANCE_PCT / 100.0
            if abs(price - prev_high) <= ph_tol and is_bearish_bar and upper_wick > 0.4 * cur_body:
                pivot_short = True
                pivot_note = f"rejected at yesterday high {prev_high:.1f}"

        # ── Trigger 3: 3-bar reversal at intraday extreme ─────────────
        # 3 consecutive bars in same direction, then current bar reverses
        # with body >50% of range. Strongest at day's low/high zone.
        reversal_long = reversal_short = False
        reversal_note = ""
        if n >= 3:
            prev3 = df.iloc[n-3:n]
            all_bearish = all(prev3["close"].iloc[i] < prev3["open"].iloc[i] for i in range(3))
            all_bullish = all(prev3["close"].iloc[i] > prev3["open"].iloc[i] for i in range(3))
            near_low  = (price - day_low) / max(1.0, day_range_pct * day_open / 100.0) < 0.30
            near_high = (day_high - price) / max(1.0, day_range_pct * day_open / 100.0) < 0.30
            if all_bearish and is_bullish_bar and body_pct > 0.55 and near_low:
                reversal_long = True
                reversal_note = "3-bear-then-bull at intraday low"
            if all_bullish and is_bearish_bar and body_pct > 0.55 and near_high:
                reversal_short = True
                reversal_note = "3-bull-then-bear at intraday high"

        # Aggregate trigger
        long_triggers  = [t for t in [vwap_bounce_long, pivot_long, reversal_long] if t]
        short_triggers = [t for t in [vwap_bounce_short, pivot_short, reversal_short] if t]

        if not long_triggers and not short_triggers:
            ScalperV3._set_decision({
                "verdict": "NO_TRIGGER",
                "vwap_dev_pct": round(vwap_dev_pct, 3),
                "near_vwap": near_vwap,
                "vol_ratio": round(vol_ratio, 2),
                "day_range_pct": round(day_range_pct, 3),
                "pivot_note": pivot_note,
                "reversal_note": reversal_note,
            })
            return None

        # Volume gate AFTER trigger detection (some scalps with no vol are still real)
        if not vol_ok:
            ScalperV3._set_decision({
                "verdict": "WEAK_VOLUME",
                "vol_ratio": round(vol_ratio, 2),
                "min": SCALP_VOL_MULT,
                "would_trigger": "LONG" if long_triggers else "SHORT",
            })
            return None

        direction = "LONG" if long_triggers else "SHORT"
        entry = round(price, 2)
        if direction == "LONG":
            sl = round(entry - SCALP_SL_PTS, 2)
            t1 = round(entry + SCALP_T1_PTS, 2)
            t2 = round(entry + SCALP_T2_PTS, 2)
            risk = entry - sl
            reward = t1 - entry
        else:
            sl = round(entry + SCALP_SL_PTS, 2)
            t1 = round(entry - SCALP_T1_PTS, 2)
            t2 = round(entry - SCALP_T2_PTS, 2)
            risk = sl - entry
            reward = entry - t1

        rr = round(reward / risk, 2) if risk > 0 else 0
        if rr < 1.5:
            ScalperV3._set_decision({"verdict": "RR_BELOW_MIN", "rr": rr})
            return None

        # ₹ profit estimate at ATM delta 0.5
        lot_size = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 65}.get(symbol.upper(), 75)
        est_t1_profit = SCALP_T1_PTS * 0.5 * lot_size
        if est_t1_profit < SCALP_TARGET_PROFIT_RS:
            ScalperV3._set_decision({
                "verdict": "T1_BELOW_TARGET_RS",
                "est_t1_profit": round(est_t1_profit),
                "target": SCALP_TARGET_PROFIT_RS,
            })
            return None

        # Confidence based on which + how many triggers fired
        n_triggers = len(long_triggers) if direction == "LONG" else len(short_triggers)
        confidence = 65 + (n_triggers - 1) * 8  # 65 for 1, 73 for 2, 81 for 3
        if vol_ratio >= 1.5: confidence += 4
        confidence = min(85, confidence)

        reasons = []
        if direction == "LONG":
            if vwap_bounce_long:  reasons.append(f"VWAP bounce: dipped to {cur_low:.1f}, closed {cur_close:.1f} (VWAP {vwap_now:.1f})")
            if pivot_long:        reasons.append(f"Pivot bounce: {pivot_note}")
            if reversal_long:     reasons.append(f"3-bar reversal: {reversal_note}")
        else:
            if vwap_bounce_short: reasons.append(f"VWAP rejection: spiked to {cur_high:.1f}, closed {cur_close:.1f} (VWAP {vwap_now:.1f})")
            if pivot_short:       reasons.append(f"Pivot rejection: {pivot_note}")
            if reversal_short:    reasons.append(f"3-bar reversal: {reversal_note}")
        reasons.append(f"Volume {vol_ratio:.2f}× avg")
        reasons.append(f"Scalp targets: SL -{SCALP_SL_PTS:.0f}pts / T1 +{SCALP_T1_PTS:.0f}pts / T2 +{SCALP_T2_PTS:.0f}pts (R:R {rr})")

        diag = {
            "verdict": f"TRIGGER {direction} x{n_triggers}",
            "n_triggers": n_triggers,
            "vol_ratio": round(vol_ratio, 2),
            "vwap_dev_pct": round(vwap_dev_pct, 3),
            "day_range_pct": round(day_range_pct, 3),
            "est_t1_profit": round(est_t1_profit),
        }
        ScalperV3._set_decision(diag)

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
                "vwap":          round(vwap_now, 2),
                "vwap_dev_pct":  round(vwap_dev_pct, 3),
                "day_range_pct": round(day_range_pct, 3),
                "vol_ratio":     round(vol_ratio, 2),
                "n_triggers":    n_triggers,
                "est_t1_rs":     round(est_t1_profit),
            },
            "strategy":   "scalper_v3",
            "v2_score":   n_triggers,
            "v2_diag":    diag,
            "timestamp":  cur_ts.strftime("%H:%M:%S") if hasattr(cur_ts, "strftime") else "",
        }
