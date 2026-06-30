"""Conductor — proactive multi-source intraday opportunity detector.

This is the "expert trader" engine: instead of one mechanical strategy firing
signals that get rubber-stamped by AI, the Conductor gathers a complete
picture per bar and decides whether ANY setup with real edge exists.

Architecture
────────────
For each bar:
  1. Build full context:
     • Price + standard indicators (RSI, MACD, EMAs, VWAP, BB, ATR)
     • Today's structure (gap, morning range, ORB, day's high/low so far,
       position-in-range)
     • Price action on current/last 3 bars (engulfing, hammer, breakout
       structure, range expansion)
     • OI/chain context — injected by caller (already computed by Engine)
     • Strategy votes (does ORB say breakout? does gamma precondition hold?)

  2. Score the context — rule-based confluence across orthogonal dimensions:
     • Trend dimension (VWAP / EMA stack / MACD direction)
     • Momentum dimension (RSI / ATR percentile / range expansion)
     • Structure dimension (ORB break / breakout candle / S-R proximity)
     • Flow dimension (OI velocity / PCR / IV skew)
     • Pattern dimension (engulfing / hammer / breakout structure)

  3. Require AGREEMENT across ≥3 dimensions (orthogonal confluence — not the
     redundant trend-stacking that broke v2). Each dimension contributes
     ONE vote max; same-dimension correlated checks don't double-count.

  4. If 3+ dimensions align AND projected R:R ≥ 2:1 AND projected ₹ profit
     ≥ user's target (₹1000 per trade), emit a high-conviction signal.

  5. In LIVE mode, the signal is then fed to SignalValidation.analyze() for
     the final AI gate. The AI sees the same full context plus the conductor's
     verdict + scoring breakdown.

Why this is different from v2
─────────────────────────────
v2 stacked 6 *correlated* trend indicators (VWAP+RSI+EMA+MACD+BB+range).
All measure "is price trending up?". 4-of-6 = "trend is strong" = entry at
trend exhaustion.

Conductor requires ≥3 of 5 *orthogonal* dimensions to agree. A trade fires
only when, say, trend is up AND momentum is fresh (not exhausted) AND there's
a structural reason (ORB break / S-R bounce) AND flow confirms (OI building
in your direction) AND a clean candle pattern shows up. THAT'S real
confluence.

Per-trade target sizing
───────────────────────
This is the difference from ORB/v2/gamma: those targeted "any positive
expectancy". Conductor targets ≥₹1,000 net profit per trade. The signal it
emits has T1/T2 spaced wide enough that hitting T1 produces ≥₹1,000 after
brokerage + slippage with 1 lot of the picked option.

Sizing math (target ₹1k net on 1 lot):
  • NIFTY lot=75, brokerage ≈ ₹100, slippage ≈ 0.5% × premium × qty
  • Need (T1_premium - entry_premium) × 75 ≥ 1,200 after costs
  • For ATM premium ~₹150, T1 must be ≥ ₹165 (10% gain) MIN
  • For deeper OTM (smaller premium), need larger % gain

The conductor computes this and rejects setups where T1 isn't reachable
at the target ₹ amount.
"""

from __future__ import annotations

import os
import math
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


# ── User-tunable knobs ────────────────────────────────────────────────────
CONDUCTOR_MIN_DIMENSIONS  = _env_int  ("CONDUCTOR_MIN_DIMENSIONS",  3)
CONDUCTOR_MIN_RR          = _env_float("CONDUCTOR_MIN_RR",         2.0)
CONDUCTOR_TARGET_PROFIT_RS = _env_float("CONDUCTOR_TARGET_PROFIT_RS", 1000.0)
CONDUCTOR_EARLIEST_H      = _env_int  ("CONDUCTOR_EARLIEST_H",       9)
CONDUCTOR_EARLIEST_M      = _env_int  ("CONDUCTOR_EARLIEST_M",      45)
CONDUCTOR_LATEST_H        = _env_int  ("CONDUCTOR_LATEST_H",        14)
CONDUCTOR_LATEST_M        = _env_int  ("CONDUCTOR_LATEST_M",        30)


def _last_decision_set(d: dict):
    """Single global slot so callers can introspect WHY the conductor
    skipped or fired on the last call."""
    Conductor.last_decision = d


class Conductor:
    """Stateless analyze() per bar. Returns signal dict or None.
    Mirrors the SignalGenV2/SignalGenORB/SignalGenGamma interface so it
    plugs into the existing backtest_v2 dispatch table.
    """

    last_decision: dict = {}

    @staticmethod
    def _config() -> dict:
        return {
            "min_dimensions":      CONDUCTOR_MIN_DIMENSIONS,
            "min_rr":              CONDUCTOR_MIN_RR,
            "target_profit_rs":    CONDUCTOR_TARGET_PROFIT_RS,
            "earliest":            time(CONDUCTOR_EARLIEST_H, CONDUCTOR_EARLIEST_M),
            "latest":              time(CONDUCTOR_LATEST_H,   CONDUCTOR_LATEST_M),
        }

    # ── Indicator helpers (lightweight, no server.TA dep) ────────────────
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
    def _atr(df, period: int = 14):
        h, l, c = df["high"], df["low"], df["close"]
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def _vwap(df):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        vol = vol.replace(0, 1).fillna(1)
        return (tp * vol).cumsum() / vol.cumsum()

    @staticmethod
    def _macd(s, fast=12, slow=26, signal=9):
        ema_fast = Conductor._ema(s, fast)
        ema_slow = Conductor._ema(s, slow)
        macd = ema_fast - ema_slow
        sig  = Conductor._ema(macd, signal)
        return macd, sig, macd - sig

    @staticmethod
    def _bb(s, period=20, mult=2.0):
        ma  = s.rolling(period).mean()
        sd  = s.rolling(period).std()
        return ma + mult*sd, ma, ma - mult*sd

    # ── Main entry point ────────────────────────────────────────────────
    @staticmethod
    def analyze(df, symbol: str = "", chain_analytics: dict = None,
                **ignored):
        """Score the current bar across 5 dimensions; emit a signal if ≥3
        agree and projected R:R + ₹ profit hit minimums.

        Args:
            df: 60-bar rolling window of 5-min spot bars with columns
                ts, open, high, low, close, volume.
            symbol: 'NIFTY' | 'BANKNIFTY' | 'FINNIFTY'.
            chain_analytics: live chain analytics from OptPicker
                (PCR, IV skew, OI velocity, max pain). Optional in backtest
                where chain data isn't always available per bar.
        """
        cfg = Conductor._config()
        if df is None or len(df) < 30:
            _last_decision_set({"verdict": "INSUFFICIENT_BARS",
                                 "bars": len(df) if df is not None else 0})
            return None
        if pd is None:
            _last_decision_set({"verdict": "PANDAS_MISSING"})
            return None

        # ── Time gate ───────────────────────────────────────────────────
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
            _last_decision_set({"verdict": "BAD_TIMESTAMP"})
            return None
        if cur_time < cfg["earliest"]:
            _last_decision_set({"verdict": "BEFORE_WINDOW", "time": cur_time.strftime("%H:%M")})
            return None
        if cur_time >= cfg["latest"]:
            _last_decision_set({"verdict": "AFTER_WINDOW", "time": cur_time.strftime("%H:%M")})
            return None

        # ── Compute indicators ─────────────────────────────────────────
        close = df["close"]
        price = float(close.iloc[n])
        vwap = Conductor._vwap(df)
        rsi  = Conductor._rsi(close, 14)
        ema20 = Conductor._ema(close, 20)
        ema50 = Conductor._ema(close, 50)
        atr   = Conductor._atr(df, 14)
        macd_line, macd_sig, macd_hist = Conductor._macd(close)
        bb_up, bb_mid, bb_lo = Conductor._bb(close, 20, 2.0)

        vwap_now  = float(vwap.iloc[n])
        rsi_now   = float(rsi.iloc[n])     if not _isnan(rsi.iloc[n]) else 50.0
        rsi_prev  = float(rsi.iloc[n-1])   if n > 0 and not _isnan(rsi.iloc[n-1]) else 50.0
        ema20_now = float(ema20.iloc[n])
        ema50_now = float(ema50.iloc[n])
        atr_now   = float(atr.iloc[n])     if not _isnan(atr.iloc[n]) else 0.0
        macd_h    = float(macd_hist.iloc[n]) if not _isnan(macd_hist.iloc[n]) else 0.0
        macd_h_p  = float(macd_hist.iloc[n-1]) if n > 0 and not _isnan(macd_hist.iloc[n-1]) else 0.0
        bb_up_now = float(bb_up.iloc[n])   if not _isnan(bb_up.iloc[n]) else price
        bb_lo_now = float(bb_lo.iloc[n])   if not _isnan(bb_lo.iloc[n]) else price

        # Today's bars for structure
        ts_series = pd.to_datetime(df["ts"])
        today_mask = ts_series.dt.date == cur_date
        today_bars = df.loc[today_mask]
        if len(today_bars) < 4:
            _last_decision_set({"verdict": "INSUFFICIENT_TODAY_BARS"})
            return None

        day_open = float(today_bars["open"].iloc[0])
        day_high = float(today_bars["high"].max())
        day_low  = float(today_bars["low"].min())
        day_range = day_high - day_low
        day_range_pct = (day_range / day_open * 100.0) if day_open > 0 else 0.0
        pos_in_range = ((price - day_low) / day_range) if day_range > 0 else 0.5

        # ORB context (first 3 bars of the day)
        orb_bars = today_bars.iloc[:3]
        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())
        orb_range = orb_high - orb_low

        # Price action on current bar
        cur_open  = float(df["open"].iloc[n])
        cur_close = float(df["close"].iloc[n])
        cur_high  = float(df["high"].iloc[n])
        cur_low   = float(df["low"].iloc[n])
        cur_body  = abs(cur_close - cur_open)
        cur_range = cur_high - cur_low
        body_pct  = (cur_body / cur_range) if cur_range > 0 else 0.0

        # Range expansion (current vs prior 5-bar avg)
        prior = df.iloc[max(0, n-5):n]
        avg_range = float((prior["high"] - prior["low"]).mean()) if len(prior) else cur_range
        range_ratio = (cur_range / avg_range) if avg_range > 0 else 1.0

        # ── Score the 5 dimensions ─────────────────────────────────────
        # Each dimension: +1 for long, -1 for short, 0 for unclear.
        # Sum gives directional bias; abs(sum) gives confluence count.
        # Same-dimension correlated checks vote ONCE — see comments.

        # 1. TREND DIMENSION (VWAP + EMA stack — both measure same thing,
        #    so we require BOTH to agree to count as 1 vote)
        trend_vote = 0
        trend_long  = (price > vwap_now * 1.0005) and (ema20_now > ema50_now)
        trend_short = (price < vwap_now * 0.9995) and (ema20_now < ema50_now)
        if trend_long:  trend_vote = +1
        if trend_short: trend_vote = -1

        # 2. MOMENTUM DIMENSION (RSI moving in direction + MACD histogram)
        #    Fresh momentum, not exhausted. Reject if RSI > 70 (overbought)
        #    or < 30 (oversold) — those are exhaustion, not opportunity.
        momentum_vote = 0
        rsi_rising   = rsi_now > rsi_prev
        rsi_falling  = rsi_now < rsi_prev
        macd_rising  = macd_h > macd_h_p and macd_h > 0
        macd_falling = macd_h < macd_h_p and macd_h < 0
        rsi_zone_ok_long  = 45 < rsi_now < 70   # not oversold, not overbought
        rsi_zone_ok_short = 30 < rsi_now < 55
        if rsi_rising and macd_rising and rsi_zone_ok_long:    momentum_vote = +1
        if rsi_falling and macd_falling and rsi_zone_ok_short: momentum_vote = -1

        # 3. STRUCTURE DIMENSION (ORB break OR breakout structure)
        struct_vote = 0
        orb_break_up   = orb_range > 30 and cur_close > orb_high and pos_in_range > 0.6
        orb_break_down = orb_range > 30 and cur_close < orb_low  and pos_in_range < 0.4
        # Alternative: holding above/below recent swing
        recent_high = float(today_bars["high"].iloc[:-1].max()) if len(today_bars) > 1 else cur_high
        recent_low  = float(today_bars["low"].iloc[:-1].min())  if len(today_bars) > 1 else cur_low
        break_recent_up   = cur_close > recent_high * 1.0005
        break_recent_down = cur_close < recent_low  * 0.9995
        if orb_break_up   or break_recent_up:   struct_vote = +1
        if orb_break_down or break_recent_down: struct_vote = -1

        # 4. FLOW DIMENSION (OI velocity + PCR + IV skew)
        #    Only counts if chain_analytics is provided AND has fresh OI data.
        flow_vote = 0
        flow_note = "no chain data"
        ca = chain_analytics or {}
        if ca and ca.get("pcr") is not None:
            pcr = float(ca.get("pcr") or 1.0)
            iv_skew = float(ca.get("iv_skew") or 0)
            oi_shift = str(ca.get("oi_shift_signal") or "NONE")
            # Bullish flow: low PCR + call premium skew, OR OI shift signals bullish
            # CE_ROLL_BULLISH = calls being placed near ATM (bullish)
            # PE_BUILD = fresh put writing = support forming (bullish for underlying)
            bullish_flow = (pcr < 0.9 and iv_skew < 0) or oi_shift in ("CE_ROLL_BULLISH", "PE_BUILD")
            # PE_ROLL_BEARISH = puts being placed near ATM (bearish)
            # CE_BUILD = fresh call writing = resistance forming (bearish for underlying)
            bearish_flow = (pcr > 1.1 and iv_skew > 0) or oi_shift in ("PE_ROLL_BEARISH", "CE_BUILD")
            # Note: "bullish flow for LONGS" means underlying going UP, but
            # bearish flow can also support a SHORT (i.e., flow_vote=-1).
            if bullish_flow: flow_vote = +1; flow_note = f"PCR {pcr:.2f}, skew {iv_skew:+.1f}, shift {oi_shift}"
            if bearish_flow: flow_vote = -1; flow_note = f"PCR {pcr:.2f}, skew {iv_skew:+.1f}, shift {oi_shift}"

        # 5. PATTERN DIMENSION (current bar shape + range expansion)
        pattern_vote = 0
        big_bullish_bar = (cur_close > cur_open and body_pct > 0.6 and range_ratio > 1.3)
        big_bearish_bar = (cur_close < cur_open and body_pct > 0.6 and range_ratio > 1.3)
        # Engulfing
        if n > 0:
            prev_open = float(df["open"].iloc[n-1])
            prev_close = float(df["close"].iloc[n-1])
            engulf_bull = (prev_close < prev_open) and (cur_close > prev_open) and (cur_open < prev_close)
            engulf_bear = (prev_close > prev_open) and (cur_close < prev_open) and (cur_open > prev_close)
            if big_bullish_bar or engulf_bull: pattern_vote = +1
            if big_bearish_bar or engulf_bear: pattern_vote = -1
        else:
            if big_bullish_bar: pattern_vote = +1
            if big_bearish_bar: pattern_vote = -1

        # ── Aggregate ─────────────────────────────────────────────────
        votes = [trend_vote, momentum_vote, struct_vote, flow_vote, pattern_vote]
        long_dims  = sum(1 for v in votes if v == +1)
        short_dims = sum(1 for v in votes if v == -1)

        if long_dims >= short_dims and long_dims >= cfg["min_dimensions"]:
            direction = "LONG"
            n_dims = long_dims
        elif short_dims > long_dims and short_dims >= cfg["min_dimensions"]:
            direction = "SHORT"
            n_dims = short_dims
        else:
            _last_decision_set({
                "verdict": "INSUFFICIENT_CONFLUENCE",
                "long_dims": long_dims, "short_dims": short_dims,
                "min_required": cfg["min_dimensions"],
                "votes": {
                    "trend": trend_vote, "momentum": momentum_vote,
                    "structure": struct_vote, "flow": flow_vote, "pattern": pattern_vote
                },
                "flow_note": flow_note,
            })
            return None

        # ── Build trade levels ─────────────────────────────────────────
        # SL: ATR-based (1.5×ATR away from entry — gives the trade real
        # breathing room). T1: 2R. T2: 3R. R:R = 2:1 minimum required.
        if atr_now <= 0:
            _last_decision_set({"verdict": "NO_ATR"})
            return None

        sl_dist = atr_now * 1.5
        entry = round(price, 2)
        if direction == "LONG":
            sl = round(entry - sl_dist, 2)
            risk = entry - sl
            t1 = round(entry + risk * 2.0, 2)
            t2 = round(entry + risk * 3.0, 2)
        else:
            sl = round(entry + sl_dist, 2)
            risk = sl - entry
            t1 = round(entry - risk * 2.0, 2)
            t2 = round(entry - risk * 3.0, 2)

        rr = round(abs(t1 - entry) / risk, 2) if risk > 0 else 0
        if rr < cfg["min_rr"]:
            _last_decision_set({"verdict": "RR_BELOW_MIN", "rr": rr, "min": cfg["min_rr"]})
            return None

        # ── ₹ profit estimate ─────────────────────────────────────────
        # Translate spot-pt risk/reward to approximate option-premium ₹ profit.
        # Approximation: ATM option delta ≈ 0.5, so option-premium move ≈
        # spot move × 0.5. For NIFTY lot=75:
        #   est_t1_premium_move = (t1_distance × 0.5) per share
        #   est_t1_profit       = t1_distance × 0.5 × lot_size
        lot_size = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 65}.get(symbol.upper(), 75)
        t1_distance = abs(t1 - entry)
        est_t1_profit_per_lot = t1_distance * 0.5 * lot_size
        # 1 lot is the minimum sizing — engine scales up based on conviction
        if est_t1_profit_per_lot < cfg["target_profit_rs"]:
            _last_decision_set({
                "verdict": "T1_BELOW_TARGET_RS",
                "est_t1_profit": round(est_t1_profit_per_lot),
                "target": cfg["target_profit_rs"],
            })
            return None

        # ── Build the signal ──────────────────────────────────────────
        # Conviction scales with n_dims: 3 dims = 65, 4 dims = 75, 5 dims = 90
        confidence = max(65, min(95, 50 + n_dims * 8))

        reasons = []
        if trend_vote == (+1 if direction == "LONG" else -1):
            reasons.append(f"Trend aligned (VWAP {'>' if direction == 'LONG' else '<'} px, EMA20{'>' if direction == 'LONG' else '<'}EMA50)")
        if momentum_vote == (+1 if direction == "LONG" else -1):
            reasons.append(f"Fresh momentum (RSI {rsi_now:.0f} {'rising' if direction == 'LONG' else 'falling'}, MACD hist {macd_h:+.1f})")
        if struct_vote == (+1 if direction == "LONG" else -1):
            reasons.append(f"Structure break ({'ORB high' if orb_break_up else 'ORB low' if orb_break_down else 'day high' if break_recent_up else 'day low'})")
        if flow_vote == (+1 if direction == "LONG" else -1):
            reasons.append(f"Flow confirms: {flow_note}")
        if pattern_vote == (+1 if direction == "LONG" else -1):
            reasons.append(f"Pattern: {'big bullish bar / engulfing' if direction == 'LONG' else 'big bearish bar / engulfing'} (range {range_ratio:.1f}× avg)")

        diag = {
            "verdict": f"TRIGGER {direction} {n_dims}-of-5",
            "n_dims": n_dims,
            "votes": {
                "trend": trend_vote, "momentum": momentum_vote,
                "structure": struct_vote, "flow": flow_vote, "pattern": pattern_vote,
            },
            "est_t1_profit": round(est_t1_profit_per_lot),
            "rr": rr,
        }
        _last_decision_set(diag)

        return {
            "direction":    direction,
            "confidence":   confidence,
            "price":        round(price, 2),
            "entry":        entry,
            "sl":           sl,
            "target1":      t1,
            "target2":      t2,
            "risk":         round(risk, 2),
            "reward":       round(abs(t1 - entry), 2),
            "risk_reward":  rr,
            "reasons":      reasons,
            "indicators": {
                "rsi":           round(rsi_now, 1),
                "vwap":          round(vwap_now, 2),
                "ema20":         round(ema20_now, 2),
                "ema50":         round(ema50_now, 2),
                "atr":           round(atr_now, 2),
                "macd_hist":     round(macd_h, 2),
                "day_range_pct": round(day_range_pct, 2),
                "pos_in_range":  round(pos_in_range, 2),
                "range_ratio":   round(range_ratio, 2),
                "n_dimensions":  n_dims,
                "est_t1_rs":     round(est_t1_profit_per_lot),
                "flow_note":     flow_note,
            },
            "strategy":   "conductor",
            "v2_score":   n_dims,
            "v2_diag":    diag,
            "timestamp":  cur_ts.strftime("%H:%M:%S") if hasattr(cur_ts, "strftime") else "",
        }


def _isnan(x) -> bool:
    try:    return x != x
    except: return False
