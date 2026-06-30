"""
OptionsIntel - composite options intelligence for trade gate decisions.

Five signals combined into a composite score (-1.0 bearish .. +1.0 bullish):
  1. PCR_VOL      (0.25)  Put-Call volume ratio - most real-time intraday sentiment
  2. OI_VELOCITY  (0.30)  Which side is building vs unwinding - institutional flow
  3. IV_SKEW      (0.20)  PE vs CE implied vol - who is paying for protection
  4. OI_WALL      (0.15)  Price proximity to max-OI resistance/support levels
  5. MAX_PAIN     (0.10)  Max pain magnet vs current spot (gravitational pull)

Plus a GEX (Gamma Exposure) multiplier: independent of direction, modifies confidence.
  Negative GEX = dealers short gamma = moves AMPLIFY = momentum signals more reliable
  Positive GEX = dealers long gamma = moves DAMPEN = momentum signals likely to fade

Gate decision relative to the proposed Conductor direction:
  PASS    dir_score >=  -0.15  (aligned or neutral)
  CAUTION dir_score in -0.35..-0.15  (mildly contradicting)
  BLOCK   dir_score <  -0.35  (strongly contradicting - skip the trade)

Angel One FULL mode returns iv_raw / delta_raw = 0 most of the time.
All Greeks are recalculated here from LTP using bisection IV + analytical formulas.
Pure-math implementation - no scipy dependency.
"""

from __future__ import annotations
import math
import datetime
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes helpers (pure math, no scipy)
# ---------------------------------------------------------------------------

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Hart (1968) rational approximation, max error 7.5e-8."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (math.exp(-0.5 * x * x) / _SQRT2PI) * p
    return cdf if x >= 0.0 else 1.0 - cdf


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if call else (K - S))
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    if call:
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _implied_vol(S: float, K: float, T: float, r: float, mkt: float, call: bool) -> float:
    """Bisection IV solver. Returns 0.0 if not solvable."""
    if T <= 0 or mkt <= 0 or S <= 0 or K <= 0:
        return 0.0
    intrinsic = max(0.0, (S - K) if call else (K - S))
    if mkt < intrinsic - 0.5:
        return 0.0
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        p = _bs_price(S, K, T, r, mid, call)
        if abs(p - mkt) < 0.01:
            return mid
        if p < mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float):
    """Returns (delta_call, delta_put, gamma)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.5, -0.5, 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    dc = _norm_cdf(d1)
    return dc, dc - 1.0, _norm_pdf(d1) / (S * sq)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RFR = 0.065   # India risk-free rate ~6.5%

_LOT_SIZES = {
    "BANKNIFTY": 30, "BANKNIFTY_SW": 30,
    "NIFTY": 75, "NIFTY_SW": 75,
    "FINNIFTY": 60, "FINNIFTY_SW": 60,
}

_W = {
    "pcr_vol":     0.25,
    "oi_velocity": 0.30,
    "iv_skew":     0.20,
    "oi_wall":     0.15,
    "max_pain":    0.10,
}

_BLOCK_THRESHOLD   = -0.35
_CAUTION_THRESHOLD = -0.15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OptionsIntel:
    """
    Usage:
        result = OptionsIntel.score(
            chain_analytics=ca,
            chain_raw=chain,
            spot=57800.0,
            direction="SHORT",
            expiry="30JUN26",
            instrument="BANKNIFTY",
        )
    Returns dict: composite, gate, confidence_delta, signals, greeks, gex, summary
    """

    @staticmethod
    def score(
        chain_analytics: dict,
        chain_raw: list,
        spot: float,
        direction: str,
        expiry: str = "",
        instrument: str = "BANKNIFTY",
    ) -> dict:
        ca = chain_analytics or {}
        T = _tte(expiry)
        lot = _LOT_SIZES.get(instrument, 30)
        atm_strike = int(ca.get("atm") or _round_strike(spot, instrument))

        # Calculate real Greeks from LTP (bypasses Angel's broken iv_raw)
        greeks, ivs = _calc_atm_greeks(chain_raw, spot, atm_strike, T)

        # Five directional signals
        s_pcr  = _score_pcr(ca)
        s_oiv  = _score_oi_velocity(ca)
        s_skew = _score_iv_skew(ca, ivs)
        s_wall = _score_oi_wall(ca, spot)
        s_pain = _score_max_pain(ca, spot)

        composite = (
            s_pcr  * _W["pcr_vol"]     +
            s_oiv  * _W["oi_velocity"] +
            s_skew * _W["iv_skew"]     +
            s_wall * _W["oi_wall"]     +
            s_pain * _W["max_pain"]
        )
        composite = round(max(-1.0, min(1.0, composite)), 3)

        # GEX confidence modifier (direction-agnostic)
        gex_mod, gex_info = _gex_modifier(chain_raw, spot, T, lot)

        # Alignment score: positive = options market agrees with our direction
        dir_score = composite if direction == "LONG" else -composite

        if dir_score >= _CAUTION_THRESHOLD:
            gate = "PASS"
            conf_delta = int((dir_score + gex_mod) * 25)
        elif dir_score >= _BLOCK_THRESHOLD:
            gate = "CAUTION"
            conf_delta = int((dir_score + gex_mod) * 20)
        else:
            gate = "BLOCK"
            conf_delta = int((dir_score + gex_mod) * 35)

        conf_delta = max(-50, min(35, conf_delta))

        signals = {
            "pcr_vol":     round(s_pcr,  3),
            "oi_velocity": round(s_oiv,  3),
            "iv_skew":     round(s_skew, 3),
            "oi_wall":     round(s_wall, 3),
            "max_pain":    round(s_pain, 3),
        }

        return {
            "composite":        composite,
            "gate":             gate,
            "confidence_delta": conf_delta,
            "dir_score":        round(dir_score, 3),
            "signals":          signals,
            "greeks":           greeks,
            "gex":              gex_info,
            "summary":          _summary(signals, composite, gate, direction, ca, greeks, gex_info),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tte(expiry: str) -> float:
    """Parse '27JUN26' -> years to expiry. Minimum 15 min."""
    try:
        dt = datetime.datetime.strptime(expiry.upper().strip(), "%d%b%y")
        now = datetime.datetime.now()
        days = max(0, (dt.date() - now.date()).days)
        # Trading hours remaining today: 9:15-15:30 = 6.25h; using 6.5h for calendar
        hours_left = max(0.0, 15.5 - (now.hour + now.minute / 60.0))
        T = (days * 6.5 + hours_left) / (252 * 6.5)
        return max(T, 1.0 / (365 * 96))   # minimum 15 min
    except Exception:
        return 1.0 / 365   # 1 day fallback


def _round_strike(spot: float, instrument: str) -> int:
    gap = 100 if "BANKNIFTY" in instrument else 50
    return int(round(spot / gap) * gap)


def _calc_atm_greeks(chain_raw, spot, atm_strike, T):
    """
    Calculate IV + Greeks for ATM and ATM+-1 strikes.
    Returns (greeks_dict, ivs_by_strike).
    """
    ivs: dict = {}
    strike_map: dict = {}

    for row in (chain_raw or []):
        k = row.get("strike", 0)
        t = row.get("type", "")
        if k and t:
            strike_map.setdefault(k, {})[t] = row

    atm_ce_data = atm_pe_data = None
    gap = 100 if atm_strike >= 40000 else 50

    for delta in (0, gap, -gap):
        k = atm_strike + delta
        entry = strike_map.get(k, {})
        for side, is_call in (("CE", True), ("PE", False)):
            row = entry.get(side)
            if not row:
                continue
            ltp = row.get("ltp", 0) or 0
            if ltp <= 0:
                continue
            iv = _implied_vol(spot, k, T, _RFR, ltp, is_call)
            if iv <= 0:
                continue
            ivs.setdefault(k, {})[("ce_iv" if is_call else "pe_iv")] = iv
            if k == atm_strike:
                dc, dp, gm = _bs_greeks(spot, k, T, _RFR, iv)
                data = {
                    "strike": k, "iv": round(iv, 4),
                    "delta": round(dc if is_call else dp, 4),
                    "gamma": round(gm, 6), "ltp": round(ltp, 2),
                }
                if is_call:
                    atm_ce_data = data
                else:
                    atm_pe_data = data

    return {"atm_ce": atm_ce_data, "atm_pe": atm_pe_data, "T_days": round(T * 365, 2)}, ivs


# ---------------------------------------------------------------------------
# Signal scorers (-1.0 = strongly bearish, +1.0 = strongly bullish)
# ---------------------------------------------------------------------------

def _score_pcr(ca: dict) -> float:
    """
    Volume PCR is more real-time than OI PCR intraday.
    Momentum interpretation (not contrarian):
      > 1.4  heavy put buying = bearish
      < 0.7  heavy call buying = bullish
    """
    pcr = float(ca.get("pcr_vol") or ca.get("pcr") or 1.0)
    if pcr <= 0:
        return 0.0
    if   pcr > 2.0:  return -1.0
    elif pcr > 1.5:  return -0.75
    elif pcr > 1.2:  return -0.4
    elif pcr > 0.85: return  0.0
    elif pcr > 0.65: return  0.4
    elif pcr > 0.5:  return  0.75
    else:            return  1.0


def _score_oi_velocity(ca: dict) -> float:
    """
    CE building + PE unwinding = bullish institutional flow.
    PE building + CE unwinding = bearish institutional flow.
    Reads both the coarse oi_shift_signal and the granular building/unwinding lists.
    """
    shift = ca.get("oi_shift_signal", "NONE")

    # Coarse signal from oi_shift_signal
    # CE_ROLL_BULLISH: calls being placed closer to ATM = bullish
    # PE_ROLL_BEARISH: puts being placed closer to ATM = bearish
    # CE_BUILD: fresh call writing = resistance building = bearish
    # PE_BUILD: fresh put writing = support building = bullish
    base = {
        "CE_ROLL_BULLISH": 0.7,
        "PE_BUILD":        0.5,
        "PE_ROLL_BEARISH": -0.7,
        "CE_BUILD":        -0.5,
    }.get(shift, 0.0)

    def _qty(rows):
        return sum(abs(r.get("oi_delta", 0) or r.get("vol_delta", 0) or 0)
                   for r in (rows or []))

    bce = _qty(ca.get("building_ce"))
    bpe = _qty(ca.get("building_pe"))
    uce = _qty(ca.get("unwinding_ce"))
    upe = _qty(ca.get("unwinding_pe"))
    total = bce + bpe + uce + upe

    if total > 0:
        # CE building = resistance above = bearish
        # PE unwinding = support dissolving = bearish
        # PE building = support below = bullish
        # CE unwinding = resistance dissolving = bullish
        # Intraday interpretation: bpe/uce = PUT/CALL-BUYING pressure = bearish/bullish
        # but at the OI level bpe (builds) tracks raw put-buyer volume which is fear
        # => bpe+uce high = fear/unwind = net bearish after * -1 flip
        net = (bpe + uce) - (bce + upe)
        flow = max(-1.0, min(1.0, net / total)) * -1
    else:
        flow = base

    # ATM velocity from the OI delta history
    atm_ce_v = (ca.get("atm_ce_oi_delta") or {}).get("velocity", "NONE")
    atm_pe_v = (ca.get("atm_pe_oi_delta") or {}).get("velocity", "NONE")
    atm_adj = 0.0
    if atm_ce_v == "BUILDING":
        atm_adj -= 0.3   # fresh resistance above = bearish
    elif atm_ce_v == "UNWINDING":
        atm_adj += 0.2   # resistance dissolving = bullish
    if atm_pe_v == "BUILDING":
        atm_adj += 0.3   # fresh support below = bullish
    elif atm_pe_v == "UNWINDING":
        atm_adj -= 0.2   # support dissolving = bearish

    combined = 0.6 * flow + 0.4 * max(-1.0, min(1.0, atm_adj))
    return round(max(-1.0, min(1.0, combined)), 3)


def _score_iv_skew(ca: dict, ivs: dict) -> float:
    """
    PE IV > CE IV = fear premium on puts = market scared of downside = bearish.
    Uses our calculated IVs; falls back to ca.iv_skew if calculation failed.
    """
    # Find any ATM strike with both IVs
    for k_data in ivs.values():
        ce_iv = k_data.get("ce_iv", 0)
        pe_iv = k_data.get("pe_iv", 0)
        if ce_iv > 0 and pe_iv > 0:
            mid = (ce_iv + pe_iv) / 2.0
            ratio = (pe_iv - ce_iv) / mid   # positive = put skew = bearish
            return round(max(-1.0, min(1.0, -ratio * 4.0)), 3)

    # Fallback: ca.iv_skew is PE_iv - CE_iv in vol points
    raw = float(ca.get("iv_skew") or 0)
    if raw == 0:
        return 0.0
    return round(max(-1.0, min(1.0, -raw / 8.0)), 3)


def _score_oi_wall(ca: dict, spot: float) -> float:
    """
    Nearest CE OI wall above spot = resistance = bearish pressure.
    Nearest PE OI wall below spot = support = bullish pressure.
    """
    if not spot:
        return 0.0
    ce_wall = ca.get("max_oi_ce_strike") or 0
    pe_wall = ca.get("max_oi_pe_strike") or 0
    score = 0.0
    if ce_wall and ce_wall > spot:
        d = (ce_wall - spot) / spot
        if   d < 0.003: score -= 0.8
        elif d < 0.007: score -= 0.5
        elif d < 0.015: score -= 0.25
    if pe_wall and pe_wall < spot:
        d = (spot - pe_wall) / spot
        if   d < 0.003: score += 0.8
        elif d < 0.007: score += 0.5
        elif d < 0.015: score += 0.25
    return round(max(-1.0, min(1.0, score)), 3)


def _score_max_pain(ca: dict, spot: float) -> float:
    """
    Max pain is the gravitational strike. Spot above it = bearish pull (wants to fall).
    Spot below it = bullish pull. Most meaningful within ~0.8% for weekly expiry.
    """
    mp = ca.get("max_pain") or spot
    if not mp or not spot:
        return 0.0
    diff_pct = (float(mp) - float(spot)) / float(spot)
    return round(max(-1.0, min(1.0, diff_pct * 80.0)), 3)


# ---------------------------------------------------------------------------
# GEX modifier (affects confidence magnitude, not direction)
# ---------------------------------------------------------------------------

def _gex_modifier(chain_raw, spot, T, lot) -> tuple:
    """
    Gamma Exposure: sum of (gamma * OI * lot * spot) for all strikes.
    Positive = dealers long gamma = mean-reversion expected = momentum fades.
    Negative = dealers short gamma = momentum amplifies.

    For Conductor's momentum signals: negative GEX is good, positive is risky.
    Returns (modifier in -0.20..+0.20, info_dict).
    """
    if not chain_raw or spot <= 0 or T <= 0:
        return 0.0, {"net": 0, "regime": "unknown"}

    gex_ce = gex_pe = 0.0
    computed = 0
    for row in chain_raw:
        k = row.get("strike", 0)
        ltp = row.get("ltp", 0) or 0
        oi = row.get("oi", 0) or row.get("volume", 0) or 0
        if not k or ltp <= 0 or oi <= 0:
            continue
        is_call = row.get("type") == "CE"
        iv = _implied_vol(spot, k, T, _RFR, ltp, is_call)
        if iv <= 0:
            continue
        _, _, gm = _bs_greeks(spot, k, T, _RFR, iv)
        contrib = gm * oi * lot * spot / 1_000_000
        if is_call:
            gex_ce += contrib
        else:
            gex_pe += contrib
        computed += 1

    net = gex_ce - gex_pe
    regime = "volatile" if net < -0.5 else ("pinning" if net > 0.5 else "neutral")

    if   net < -2.0: mod =  0.20   # very volatile = momentum very likely to hold
    elif net < -0.5: mod =  0.10
    elif net >  2.0: mod = -0.20   # strong pinning = momentum likely to fade
    elif net >  0.5: mod = -0.10
    else:            mod =  0.0

    return mod, {
        "net": round(net, 2),
        "calls": round(gex_ce, 2),
        "puts": round(gex_pe, 2),
        "regime": regime,
        "strikes_computed": computed,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary(signals, composite, gate, direction, ca, greeks, gex) -> str:
    parts = [f"[{gate}] composite={composite:+.3f} dir={direction}"]

    strong = sorted(
        [(k, v) for k, v in signals.items() if abs(v) >= 0.2],
        key=lambda x: abs(x[1]), reverse=True
    )
    for k, v in strong[:3]:
        parts.append(f"{k}={'BUL' if v > 0 else 'BEAR'}{abs(v):.2f}")

    shift = ca.get("oi_shift_signal", "NONE")
    if shift != "NONE":
        parts.append(f"OI_shift={shift}")

    if gex.get("regime") and gex["regime"] != "neutral":
        parts.append(f"GEX={gex['regime']}({gex.get('net', 0):+.1f})")

    atm_ce = greeks.get("atm_ce") or {}
    atm_pe = greeks.get("atm_pe") or {}
    if atm_ce.get("iv") and atm_pe.get("iv"):
        parts.append(f"CE_IV={atm_ce['iv']:.3f} PE_IV={atm_pe['iv']:.3f}")
        parts.append(f"CE_d={atm_ce.get('delta',0):+.3f} PE_d={atm_pe.get('delta',0):+.3f}")

    return " | ".join(parts)
