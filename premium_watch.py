"""Premium Crossover Watch: Ketan's own scalp setup, read on the OPTION PREMIUM chart.

The setup (from his NIFTY 26050 CE chart, 26 Nov 2026, 10:20 bar):
  * the fast EMA (blue, 9) crosses up through the slow EMA (red, 21) on the 5-minute
    premium chart,
  * the MACD histogram flips from red to green at the same time,
  * volume on that bar is clearly above normal,
  * and for the next 15 to 30 minutes the premium usually runs far enough for
    Rs 2,000 to 3,000 per lot.

This module scans ATM and nearby CE/PE contracts of each watched index every
5 minutes, right after a bar closes, and posts a WATCH alert to Slack when every
part of the setup lines up. It is an alert for Ketan to look at the chart and
decide; it never opens a position in the engine and never counts toward the
engine's kill switch or open-position caps.

Every alert is followed to its outcome (target, stop, or the 30-minute time
stop) using the same premium candles, so the hit rate can be checked later.

Rules on the CLOSED 5-minute bar (all must pass):
  1. EMA9 crossed above EMA21 on this bar or the one before.
  2. MACD(12,26,9) histogram is green now, was red or zero within the last 3 bars,
     and is rising.
  3. Bar volume >= VOL_MULT x the average of the previous 20 bars.
  4. Close above the day's VWAP of the premium (buyers in control today).
  5. Green bar that closes in the upper half of its range.
  6. Not chasing: close no more than MAX_EXT above EMA9.
  7. Between 09:30 and 14:45 IST (13:30 on the contract's expiry day).
Ranking only (never blocks): the index's own 5-minute EMA9/21 points the same way
(up for a CE, down for a PE), Smart Money tape agrees, RSI 50 to 72, bigger volume.

Levels per lot (Ketan's ask: Rs 2,000 to 3,000 per lot is enough):
  T1 = entry + Rs 2,000 / lot size, T2 = entry + Rs 3,000 / lot size,
  SL = the lower of the last 3 bars' low, but never risking more than
  Rs 1,500 per lot and never more than 25% of the premium.
  Time stop: 30 minutes after the alert.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

EMA_FAST, EMA_SLOW = 9, 21
VOL_MULT = float(os.environ.get("PW_VOL_MULT", "1.3"))
MAX_EXT = float(os.environ.get("PW_MAX_EXT", "0.20"))        # close <= EMA9 * 1.20 (premiums move fast)
NEAR_GAP = float(os.environ.get("PW_NEAR_GAP", "0.01"))      # blue within 1% under red and closing in
MACD_LOOKBACK = 6                                             # histogram turned green within 30 min
T1_RS = float(os.environ.get("PW_T1_RS", "2000"))            # per lot
T2_RS = float(os.environ.get("PW_T2_RS", "3000"))
SL_RS_CAP = float(os.environ.get("PW_SL_RS", "1500"))
SL_PCT_CAP = 0.25
HOLD_MIN = int(os.environ.get("PW_HOLD_MIN", "30"))
START_HM, END_HM, EXPIRY_END_HM = "09:30", "14:45", "13:30"
STRIKES_EACH_SIDE = int(os.environ.get("PW_STRIKES", "2"))   # ATM +/- 2
MIN_PREMIUM = float(os.environ.get("PW_MIN_PREMIUM", "20"))
CONTRACT_COOLDOWN_MIN = 45
# "Forming" = crossover about to happen. Replay of NIFTY weekly options, 9-26 Nov 2026
# (292 forming bars, 12 contracts): on its own no better than any bar (23% rose 20+ pts
# in 30 min). With premium above VWAP, volume >= 1.2x and MACD green, before 14:00:
# 37 cases, 62% rose 20+ pts, typical reach 1.7x the premium's 5-min ATR. Small sample.
FORM_GAP = 0.03
FORM_VOL = 1.2
FORM_END_HM = "13:59"
FORM_REACH_ATR = float(os.environ.get("PW_REACH_ATR", "1.5"))
MIN_POTENTIAL = float(os.environ.get("PW_MIN_POTENTIAL", "20"))     # premium points
FORM_ODDS = "In past NIFTY setups like this, about 6 in 10 rose 20+ points within 30 min (37 cases)"
INDEX_COOLDOWN_MIN = 15


# ── indicators ──────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df: timestamp (naive IST), open, high, low, close, volume. Returns a copy with
    ema9, ema21, hist, rsi, vwap (resets each day), vavg20."""
    d = df.copy()
    for k in ("open", "high", "low", "close", "volume"):
        d[k] = d[k].astype(float)
    c = d["close"]
    d["ema9"] = _ema(c, EMA_FAST)
    d["ema21"] = _ema(c, EMA_SLOW)
    macd = _ema(c, 12) - _ema(c, 26)
    d["hist"] = macd - _ema(macd, 9)
    delta = c.diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    day = d["timestamp"].dt.date
    tp = (d["high"] + d["low"] + d["close"]) / 3
    pv = (tp * d["volume"]).groupby(day).cumsum()
    vv = d["volume"].groupby(day).cumsum()
    d["vwap"] = (pv / vv.replace(0, np.nan)).fillna(tp)
    d["vavg20"] = d["volume"].shift(1).rolling(20, min_periods=10).mean()
    return d


def check_bar(d: pd.DataFrame, i: int, expiry_day: bool = False) -> dict:
    """Evaluate the setup on closed bar i of an indicators() frame.
    Returns {"ok": bool, "fails": [...], "ind": {...}}."""
    r = d.iloc[i]
    fails = []
    hm = r["timestamp"].strftime("%H:%M")
    if not (START_HM <= hm <= (EXPIRY_END_HM if expiry_day else END_HM)):
        fails.append("time")
    if i < 30:
        fails.append("warmup")
        return {"ok": False, "fails": fails, "ind": {}}
    e9, e21 = d["ema9"], d["ema21"]
    crossed = e9.iloc[i] > e21.iloc[i] and any(
        e9.iloc[i - k] <= e21.iloc[i - k] for k in (1, 2))
    # "Crossover starting": blue still just under red but closing in fast for 2 bars.
    # On 26 Nov this is the 10:20 bar of NIFTY 26050 CE, one bar before the real cross.
    gap = [e21.iloc[i - k] - e9.iloc[i - k] for k in (0, 1, 2)]
    starting = 0 < gap[0] <= e21.iloc[i] * NEAR_GAP and gap[0] < gap[1] < gap[2]
    crossed_at = None
    if not (crossed or starting):
        if e9.iloc[i] > e21.iloc[i]:
            # Blue went over red earlier than 2 bars ago: the entry is gone, not coming.
            j = cross_index(d, i)
            crossed_at = d["timestamp"].iloc[j].strftime("%H:%M") if j is not None else None
            fails.append("crossed_earlier")
        else:
            fails.append("no_cross")
    h = d["hist"]
    flip = h.iloc[i] > 0 and h.iloc[i] > h.iloc[i - 1] and any(
        h.iloc[i - k] <= 0 for k in range(1, MACD_LOOKBACK + 1))
    if not flip:
        fails.append("macd")
    green_since = None
    if h.iloc[i] > 0:
        j = i
        while j > 0 and h.iloc[j - 1] > 0:
            j -= 1
        green_since = d["timestamp"].iloc[j].strftime("%H:%M")
    vavg = r["vavg20"]
    vr = float(r["volume"] / vavg) if vavg and vavg == vavg and vavg > 0 else 0.0
    if vr < VOL_MULT:
        fails.append("volume")
    if not r["close"] > r["vwap"]:
        fails.append("below_vwap")
    rng = r["high"] - r["low"]
    if not (r["close"] > r["open"] and (rng <= 0 or (r["close"] - r["low"]) / rng >= 0.5)):
        fails.append("weak_bar")
    if r["close"] > r["ema9"] * (1 + MAX_EXT):
        fails.append("extended")
    ind = {"close": round(r["close"], 2), "ema9": round(r["ema9"], 2), "ema21": round(r["ema21"], 2),
           "hist": round(r["hist"], 2), "hist_prev": round(h.iloc[i - 1], 2),
           "rsi": round(float(r["rsi"]), 1) if r["rsi"] == r["rsi"] else None,
           "vwap": round(r["vwap"], 2), "vol": int(r["volume"]), "vol_ratio": round(vr, 2),
           "bar": r["timestamp"].strftime("%Y-%m-%d %H:%M"), "stage": "crossed" if crossed else "starting",
           "crossed_at": crossed_at, "macd_green_since": green_since}
    return {"ok": not fails, "fails": fails, "ind": ind}


def cross_index(d: pd.DataFrame, i: int):
    """Bar where blue last went over red, today only. None if blue is under red now, or it
    went over on an earlier day (that cross is history, not a live setup)."""
    e9, e21 = d["ema9"], d["ema21"]
    if not e9.iloc[i] > e21.iloc[i]:
        return None
    day = d["timestamp"].iloc[i].date()
    j = i
    while j > 0 and e9.iloc[j - 1] > e21.iloc[j - 1]:
        j -= 1
        if d["timestamp"].iloc[j].date() != day:
            return None
    return j if j > 0 else None


def atr_at(d: pd.DataFrame, i: int) -> float:
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(com=13, adjust=False).mean().iloc[i])


def check_forming(d: pd.DataFrame, i: int) -> dict | None:
    """Crossover about to happen on bar i: blue under red but closing in for 2 bars, MACD
    green and rising, price already over the blue line, above VWAP, volume up. Returns
    {"atr", "gap_pct", "vol_ratio"} or None."""
    if i < 30:
        return None
    r = d.iloc[i]
    hm = r["timestamp"].strftime("%H:%M")
    if not (START_HM <= hm <= FORM_END_HM):
        return None
    e9, e21, h = d["ema9"], d["ema21"], d["hist"]
    gap = [e21.iloc[i - k] - e9.iloc[i - k] for k in (0, 1, 2)]
    if not (0 < gap[0] <= e21.iloc[i] * FORM_GAP and gap[0] < gap[1] < gap[2]):
        return None
    if not (h.iloc[i] > 0 and h.iloc[i] > h.iloc[i - 1] > h.iloc[i - 2]):
        return None
    vr = float(r["volume"] / r["vavg20"]) if r["vavg20"] == r["vavg20"] and r["vavg20"] else 0.0
    if not (r["close"] > e9.iloc[i] and r["close"] > r["vwap"] and vr >= FORM_VOL):
        return None
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(),
                    (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(com=13, adjust=False).mean().iloc[i])
    return {"atr": round(atr, 2), "gap_pct": round(gap[0] / e21.iloc[i] * 100, 2), "vol_ratio": round(vr, 2)}


def oi_read(chain: list, an: dict, spot: float, strike: float, otype: str, gap: float) -> dict:
    """Where the option writers sit, from the live chain. For a CE the biggest call-OI strike
    above spot is the ceiling; for a PE the biggest put-OI strike below spot is the floor.
    Returns room in index points, a rough delta, and whether OI flow helps or hurts."""
    out = {"room": None, "wall": None, "delta": None, "flow": "neutral", "note": ""}
    if not chain:
        return out
    if otype == "CE":
        side = [o for o in chain if o.get("type") == "CE" and (o.get("strike") or 0) > spot]
    else:
        side = [o for o in chain if o.get("type") == "PE" and 0 < (o.get("strike") or 0) < spot]
    side = [o for o in side if (o.get("oi") or 0) > 0]
    if side:
        w = max(side, key=lambda o: o.get("oi") or 0)
        out["wall"] = w["strike"]
        out["room"] = round(abs(w["strike"] - spot), 1)
    # Rough delta from moneyness (chain greeks are often missing): ATM ~0.5, ~0.1 per strike step.
    steps = (spot - strike) / gap if otype == "CE" else (strike - spot) / gap
    out["delta"] = round(min(0.85, max(0.15, 0.5 + 0.1 * steps)), 2)
    near = lambda lst: [x for x in (an.get(lst) or []) if abs((x.get("strike") or 0) - spot) <= 2 * gap]
    if otype == "CE":
        good, bad = near("unwinding_ce") + near("building_pe"), near("building_ce")
        gtxt, btxt = "call writers leaving / put writers adding", "call writers adding near spot"
    else:
        good, bad = near("unwinding_pe") + near("building_ce"), near("building_pe")
        gtxt, btxt = "put writers leaving / call writers adding", "put writers adding near spot"
    if bad and not good:
        out["flow"], out["note"] = "against", btxt
    elif good and not bad:
        out["flow"], out["note"] = "with", gtxt
    return out


def levels(d: pd.DataFrame, i: int, lot: int) -> dict:
    entry = float(d["close"].iloc[i])
    swing = float(d["low"].iloc[max(0, i - 2):i + 1].min())
    floor = max(entry - SL_RS_CAP / lot, entry * (1 - SL_PCT_CAP))
    sl = round(max(swing - 0.5, floor), 2)
    return {"entry": round(entry, 2), "sl": sl,
            "t1": round(entry + T1_RS / lot, 2), "t2": round(entry + T2_RS / lot, 2),
            "risk_per_lot": round((entry - sl) * lot)}


def follow(d: pd.DataFrame, i: int, lv: dict, lot: int) -> dict:
    """Walk the bars after alert bar i (up to HOLD_MIN) and say what happened.
    Stop is checked before target inside one bar (conservative)."""
    t0 = d["timestamp"].iloc[i]
    after = d[(d["timestamp"] > t0) & (d["timestamp"] <= t0 + timedelta(minutes=HOLD_MIN))]
    after = after[after["timestamp"].dt.date == t0.date()]
    best = lv["entry"]
    for _, b in after.iterrows():
        best = max(best, b["high"])
        if b["low"] <= lv["sl"]:
            return {"result": "SL", "at": b["timestamp"].strftime("%H:%M"),
                    "pnl_per_lot": round((lv["sl"] - lv["entry"]) * lot), "max_seen": round(best, 2)}
        if b["high"] >= lv["t2"]:
            return {"result": "T2", "at": b["timestamp"].strftime("%H:%M"),
                    "pnl_per_lot": round((lv["t2"] - lv["entry"]) * lot), "max_seen": round(best, 2)}
        if b["high"] >= lv["t1"]:
            # T1 reached: book it (Ketan's plan is a small fixed grab)
            return {"result": "T1", "at": b["timestamp"].strftime("%H:%M"),
                    "pnl_per_lot": round((lv["t1"] - lv["entry"]) * lot), "max_seen": round(best, 2)}
    if len(after) < HOLD_MIN // 5:
        return {"result": "OPEN", "max_seen": round(best, 2),
                "pnl_per_lot": round((float(after["close"].iloc[-1]) - lv["entry"]) * lot) if len(after) else 0}
    last = float(after["close"].iloc[-1])
    return {"result": "TIME", "at": after["timestamp"].iloc[-1].strftime("%H:%M"),
            "pnl_per_lot": round((last - lv["entry"]) * lot), "max_seen": round(best, 2)}


def replay(df: pd.DataFrame, lot: int, day: str | None = None, expiry_day: bool = False,
           detail: bool = False) -> dict:
    """Run the setup over every closed bar (optionally one day) and follow each hit."""
    d = indicators(_norm(df))
    hits, near = [], []
    last_hit = None
    for i in range(len(d)):
        if day and d["timestamp"].iloc[i].strftime("%Y-%m-%d") != day:
            continue
        chk = check_bar(d, i, expiry_day)
        if chk["ok"]:
            t = d["timestamp"].iloc[i]
            if last_hit is not None and t - last_hit < timedelta(minutes=CONTRACT_COOLDOWN_MIN):
                continue
            last_hit = t
            lv = levels(d, i, lot)
            hits.append({**chk["ind"], **lv, "outcome": follow(d, i, lv, lot)})
        elif not {"no_cross", "crossed_earlier", "warmup"} & set(chk["fails"]):
            near.append({"bar": chk["ind"].get("bar"), "missing": chk["fails"]})
    out = {"hits": hits, "near_misses": near[-12:]}
    if detail:
        dd = d if not day else d[d["timestamp"].dt.strftime("%Y-%m-%d") == day]
        out["bars"] = [{"t": r.timestamp.strftime("%H:%M"), "o": round(r.open, 2), "h": round(r.high, 2),
                        "l": round(r.low, 2), "c": round(r.close, 2), "v": int(r.volume),
                        "e9": round(r.ema9, 2), "e21": round(r.ema21, 2), "hist": round(r.hist, 2),
                        "vwap": round(r.vwap, 2), "vr": round(float(r.volume / r.vavg20), 2) if r.vavg20 == r.vavg20 and r.vavg20 else None}
                       for r in dd.itertuples()]
    return out


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    d["timestamp"] = ts
    return d.sort_values("timestamp").reset_index(drop=True)


def follow_forming(d: pd.DataFrame, i: int, entry: float, target: float, lot: int) -> dict:
    """Did the premium reach the forecast price within HOLD_MIN after forming bar i?"""
    t0 = d["timestamp"].iloc[i]
    after = d[(d["timestamp"] > t0) & (d["timestamp"] <= t0 + timedelta(minutes=HOLD_MIN))]
    after = after[after["timestamp"].dt.date == t0.date()]
    if not len(after):
        return {"result": "OPEN"}
    hi, lo = float(after["high"].max()), float(after["low"].min())
    if hi >= target:
        at = after[after["high"] >= target]["timestamp"].iloc[0].strftime("%H:%M")
        return {"result": "HIT", "at": at, "max_seen": round(hi, 2), "min_seen": round(lo, 2),
                "gain_pts": round(hi - entry, 1), "pnl_per_lot": round((target - entry) * lot)}
    if len(after) < HOLD_MIN // 5:
        return {"result": "OPEN", "max_seen": round(hi, 2)}
    last = float(after["close"].iloc[-1])
    return {"result": "MISS", "at": after["timestamp"].iloc[-1].strftime("%H:%M"), "max_seen": round(hi, 2),
            "min_seen": round(lo, 2), "gain_pts": round(hi - entry, 1), "pnl_per_lot": round((last - entry) * lot)}


class EventStore:
    """Every crossover alert, forming forecast and skipped cross, with its outcome, so the
    success rate can be counted across days. One SQLite table; payload kept as JSON."""

    def __init__(self, path):
        self.path = path
        try:
            with sqlite3.connect(self.path) as c:
                c.execute("""CREATE TABLE IF NOT EXISTS pw_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, time TEXT, kind TEXT,
                    contract TEXT, bar TEXT, result TEXT DEFAULT 'OPEN', payload TEXT,
                    UNIQUE(kind, contract, day, bar))""")
        except Exception:
            self.path = None

    def add(self, kind, ev):
        if not self.path:
            return
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR IGNORE INTO pw_events (day, time, kind, contract, bar, result, payload) "
                      "VALUES (?,?,?,?,?,?,?)", (ev["day"], ev["time"], kind, ev["contract"], ev["bar"],
                                                 (ev.get("outcome") or {}).get("result", "OPEN"),
                                                 json.dumps(ev, default=str)))

    def update(self, kind, ev):
        if not self.path:
            return
        with sqlite3.connect(self.path) as c:
            c.execute("UPDATE pw_events SET result=?, payload=? WHERE kind=? AND contract=? AND day=? AND bar=?",
                      ((ev.get("outcome") or {}).get("result", "OPEN"), json.dumps(ev, default=str),
                       kind, ev["contract"], ev["day"], ev["bar"]))

    def rows(self, where="1=1", args=()):
        if not self.path:
            return []
        with sqlite3.connect(self.path) as c:
            out = []
            for kind, payload in c.execute(f"SELECT kind, payload FROM pw_events WHERE {where} ORDER BY day, time", args):
                ev = json.loads(payload); ev["kind"] = kind; out.append(ev)
            return out

    def restore(self, path):
        """Re-seed from pw_backfill.json after a deploy wiped the DB (only when empty)."""
        if not self.path or not os.path.exists(path):
            return 0
        with sqlite3.connect(self.path) as c:
            if c.execute("SELECT COUNT(*) FROM pw_events").fetchone()[0]:
                return 0
        n = 0
        for ev in json.load(open(path)):
            k = ev.pop("kind", None)
            if k and (ev.get("outcome") or {}).get("result", "OPEN") != "OPEN":
                self.add(k, ev); n += 1
        return n


def score(rows) -> dict:
    """Success rates per kind. watch: T1/T2 = success. forming: reached the forecast.
    skipped: the filter was right when the what-if trade would NOT have hit T1."""
    out = {}
    for kind in ("watch", "forming", "cross", "skipped"):
        rs = [r for r in rows if r["kind"] == kind and (r.get("outcome") or {}).get("result") not in (None, "OPEN")]
        if kind == "watch":
            ok = [r for r in rs if r["outcome"]["result"] in ("T1", "T2")]
        elif kind in ("forming", "cross"):
            ok = [r for r in rs if r["outcome"]["result"] == "HIT"]
        else:
            ok = [r for r in rs if r["outcome"]["result"] not in ("T1", "T2")]
        out[kind] = {"closed": len(rs), "success": len(ok),
                     "pct": round(len(ok) / len(rs) * 100) if rs else None,
                     "pnl_per_lot": sum((r["outcome"].get("pnl_per_lot") or 0) for r in rs),
                     "open": sum(1 for r in rows if r["kind"] == kind and (r.get("outcome") or {}).get("result", "OPEN") == "OPEN")}
    return out


# ── live scanner ────────────────────────────────────────────────────────────

class PremiumWatch:
    """Own thread. Scans once per 5-minute bar, ~25 s after it closes."""

    def __init__(self, client, master, instruments: dict, send_slack, bias_fn=None, log=None,
                 tz=None, chain_fn=None, db_path=None):
        self.client, self.master, self.instruments = client, master, instruments
        self.send_slack, self.bias_fn, self.log, self.tz = send_slack, bias_fn, log, tz
        self.chain_fn = chain_fn
        self.store = EventStore(db_path) if db_path else None
        self._scorecard_day = None
        self.reach_factor = FORM_REACH_ATR
        self.ltp_now: dict = {}               # token -> (live price, "HH:MM:SS") from the quote API
        self._live_tokens: dict = {}          # index -> {token: contract label}
        # 5-minute bars built from the live quotes, so a scan needs no candle download
        # (Angel throttles those and a scan took 4+ minutes). token -> {bar_start: bar}
        self._tbars: dict = {}
        self._tlast: dict = {}                # token -> (time, cumulative volume) of the last quote
        self._base: dict = {}                 # token -> candle history from Angel (fetched once)
        self.bar_source: dict = {}            # contract -> "live quotes" / "Angel candles"
        self.learned: dict = {}
        self._form_last: dict = {}
        self.watch_list = [s.strip().upper() for s in
                           os.environ.get("PW_INSTRUMENTS", "NIFTY").split(",") if s.strip()]
        self.enabled = os.environ.get("PW_ENABLED", "1") != "0"
        self.lock = threading.Lock()
        self.alerts: list[dict] = []          # today's alerts, newest last
        self.last_scan: dict = {}             # per index: time, contracts, near misses
        self._last_bar = None
        self._contract_last: dict = {}
        self._index_last: dict = {}
        self.running = False
        self._last_follow = 0.0
        self.skips: dict = {}                 # contract -> last fresh cross that failed a check

    # helpers
    def _now(self):
        return datetime.now(self.tz)

    def _info(self, msg):
        if self.log:
            self.log.info(f"[PW] {msg}")

    def _warn(self, msg):
        if self.log:
            self.log.warning(f"[PW] {msg}")

    def start(self):
        if self.running or not self.enabled:
            return
        self.running = True
        if self.store:
            try:
                n = self.store.restore(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pw_backfill.json"))
                if n:
                    self._info(f"restored {n} scored events from pw_backfill.json")
                today = self._now().strftime("%Y-%m-%d")
                self.alerts = [r for r in self.store.rows("kind='watch' AND day=?", (today,))]
                for a in self.alerts:
                    self._contract_last[a["contract"]] = self._now()
            except Exception as e:
                self._warn(f"event store load: {e}")
            self._learn()
        threading.Thread(target=self._loop, daemon=True, name="PremiumWatch").start()
        threading.Thread(target=self._tick_loop, daemon=True, name="PremiumWatchLTP").start()
        threading.Thread(target=self._follow_loop, daemon=True, name="PremiumWatchFollow").start()
        self._info(f"running for {', '.join(self.watch_list)}")

    def _loop(self):
        while self.running:
            try:
                now = self._now()
                if now.weekday() < 5 and "09:15" <= now.strftime("%H:%M") <= "15:30":
                    # the bar that closed most recently, scanned once, 25 s after its close
                    bar_close = now.replace(second=0, microsecond=0,
                                            minute=now.minute - now.minute % 5)
                    if (now - bar_close).total_seconds() >= 25 and self._last_bar != bar_close:
                        self._last_bar = bar_close
                        self.scan(bar_close)
                elif now.strftime("%H:%M") < "09:00" and self.alerts and \
                        self.alerts[0].get("day") != now.strftime("%Y-%m-%d"):
                    with self.lock:
                        self.alerts = []
                        self._contract_last, self._index_last = {}, {}
            except Exception as e:
                self._warn(f"loop error: {e}")
            time.sleep(10)

    def _market_open(self):
        n = self._now()
        return n.weekday() < 5 and "09:15" <= n.strftime("%H:%M") <= "15:31"

    def _tick_loop(self):
        """Live option prices every PW_TICK_S seconds (one batched quote call), so the Desk
        shows the real current premium, not the close of the last 5-minute candle."""
        every = float(os.environ.get("PW_TICK_S", "5"))
        warned = 0.0
        while self.running:
            try:
                if self._market_open():
                    toks = {}
                    for m in list(self._live_tokens.values()):
                        toks.update(m)
                    with self.lock:
                        toks.update({str(a["token"]): a["contract"] for a in self.alerts
                                     if a.get("token") and (a.get("outcome") or {}).get("result") == "OPEN"})
                    if toks:
                        r = self.client.api.getMarketData(mode="FULL", exchangeTokens={"NFO": list(toks)})
                        now = self._now()
                        stamp = now.strftime("%H:%M:%S")
                        for it in (((r or {}).get("data") or {}).get("fetched") or []):
                            tok = str(it.get("symbolToken") or it.get("symboltoken") or "")
                            if tok in toks and it.get("ltp") not in (None, "", 0):
                                self.ltp_now[tok] = (float(it["ltp"]), stamp)
                                vol = it.get("tradeVolume") or it.get("totalTradedVolume")
                                self._tick_bar(tok, now.replace(tzinfo=None), float(it["ltp"]),
                                               float(vol) if vol not in (None, "") else None)
            except Exception as e:
                if time.time() - warned > 300:
                    warned = time.time()
                    self._warn(f"live prices: {e}")
            time.sleep(every)

    def _tick_bar(self, tok, t, px, vcum):
        """Fold one quote into its 5-minute bar. A bar is 'full' only if quotes were already
        flowing when it opened (previous quote under 15 s before), so its open and volume are right."""
        start = t.replace(second=0, microsecond=0, minute=t.minute - t.minute % 5)
        bars = self._tbars.setdefault(tok, {})
        prev = self._tlast.get(tok)
        b = bars.get(start)
        if b is None:
            full = bool(prev and (t - prev[0]).total_seconds() <= 15 and (prev[0] >= start - timedelta(minutes=5)))
            b = bars[start] = {"o": px, "h": px, "l": px, "c": px, "full": full and prev[1] is not None and vcum is not None,
                               "v0": prev[1] if prev else None, "v1": vcum}
            for k in [k for k in bars if k < start - timedelta(hours=8)]:
                bars.pop(k, None)
        b["h"], b["l"], b["c"] = max(b["h"], px), min(b["l"], px), px
        if vcum is not None:
            b["v1"] = vcum
        self._tlast[tok] = (t, vcum)

    def _bars_for(self, tok, label, bar_close):
        """Candles for one contract up to the bar that just closed. Uses Angel's history once
        (and again only if the live-quote bars have a gap), then the bars built from quotes."""
        tok = str(tok)
        last_start = bar_close.replace(tzinfo=None) - timedelta(minutes=5)
        base = self._base.get(tok)
        tb = self._tbars.get(tok, {})
        if base is not None and len(base):
            b_end = base["timestamp"].iloc[-1]
            need, t = [], b_end + timedelta(minutes=5)
            while t <= last_start:
                if t.date() == last_start.date() and t.strftime("%H:%M") >= "09:15":
                    need.append(t)
                t += timedelta(minutes=5)
            if all(t in tb and tb[t]["full"] for t in need):
                if need:
                    add = pd.DataFrame([{"timestamp": t, "open": tb[t]["o"], "high": tb[t]["h"], "low": tb[t]["l"],
                                         "close": tb[t]["c"], "volume": max(0.0, (tb[t]["v1"] or 0) - (tb[t]["v0"] or 0))}
                                        for t in need])
                    self._base[tok] = base = pd.concat([base, add], ignore_index=True)
                self.bar_source[label] = "live quotes"
                return base
        df = self.client.candles(tok, "NFO", "FIVE_MINUTE", 3, force_refresh=True)
        if df is None or df.empty:
            return df
        d = _norm(df)[["timestamp", "open", "high", "low", "close", "volume"]]
        # Keep only closed bars; the bar still forming would be counted twice later.
        d = d[d["timestamp"] <= last_start].reset_index(drop=True)
        for k in ("open", "high", "low", "close", "volume"):
            d[k] = d[k].astype(float)
        self._base[tok] = d
        self.bar_source[label] = "Angel candles"
        return d

    def _follow_loop(self):
        """Scores open events every minute, on its own thread so a slow candle call never
        holds up the 5-minute scan."""
        while self.running:
            try:
                n = self._now()
                if n.weekday() < 5 and "09:15" <= n.strftime("%H:%M") <= "15:50":
                    self._update_outcomes()
            except Exception as e:
                self._warn(f"follow loop: {e}")
            time.sleep(60)

    def _live_price(self, token, max_age_s=60):
        v = self.ltp_now.get(str(token)) if token else None
        if not v:
            return None, None
        try:
            t = datetime.strptime(self._now().strftime("%Y-%m-%d ") + v[1], "%Y-%m-%d %H:%M:%S")
            if (self._now().replace(tzinfo=None) - t).total_seconds() > max_age_s:
                return None, None
        except Exception:
            pass
        return v

    def _contracts(self, name: str, spot: float):
        """Nearest usable expiry (skip today's expiry after 11:00), ATM +/- N, CE and PE."""
        info = self.instruments[name]
        prefix, gap = info["expiry_prefix"], info["strike_gap"]
        today = self._now().date()
        exps = {}
        for (nm, strike, otype, exp), row in self.master.nfo.items():
            if nm != prefix:
                continue
            try:
                dt = datetime.strptime(exp, "%d%b%Y").date()
            except Exception:
                continue
            if dt >= today:
                exps[dt] = exp
        if not exps:
            return [], None
        dates = sorted(exps)
        pick = dates[0]
        if pick == today and self._now().strftime("%H:%M") >= "11:00" and len(dates) > 1:
            pick = dates[1]
        exp = exps[pick]
        atm = round(spot / gap) * gap
        out = []
        for k in range(-STRIKES_EACH_SIDE, STRIKES_EACH_SIDE + 1):
            for ot in ("CE", "PE"):
                row = self.master.nfo.get((prefix, float(atm + k * gap), ot, exp))
                if row:
                    out.append(row)
        return out, pick

    def _spot_and_trend(self, name: str):
        info = self.instruments[name]
        df = self.client.candles(info["token"], info["exchange"], "FIVE_MINUTE", 3)
        if df is None or df.empty:
            return None, 0
        d = _norm(df)
        c = d["close"].astype(float)
        e9, e21 = _ema(c, EMA_FAST).iloc[-1], _ema(c, EMA_SLOW).iloc[-1]
        return float(c.iloc[-1]), (1 if e9 > e21 else -1)

    def _bar_index(self, d: pd.DataFrame, bar_close):
        """Index of the last bar that has fully closed by bar_close (bars are stamped
        with their open time)."""
        cut = bar_close.replace(tzinfo=None) - timedelta(minutes=5)
        idx = d.index[d["timestamp"] <= cut]
        return int(idx[-1]) if len(idx) else None

    def scan(self, bar_close):
        for name in self.watch_list:
            if name not in self.instruments:
                continue
            try:
                self._scan_index(name, bar_close)
            except Exception as e:
                self._warn(f"{name} scan failed: {e}")

    def _scan_index(self, name, bar_close):
        t_start = time.time()
        spot, idx_trend = self._spot_and_trend(name)
        if not spot:
            return
        contracts, exp_date = self._contracts(name, spot)
        self._live_tokens[name] = {str(r["token"]): f"{name} {int(r['strike'])} {r['type']}" for r in contracts}
        expiry_day = exp_date == self._now().date()
        bias = None
        if self.bias_fn:
            b = self.bias_fn(name)
            bias = b.get("score") if isinstance(b, dict) else None
        cands, rows, forming, live = [], [], [], []
        ch = None
        if self.chain_fn:
            try:
                ch = self.chain_fn(name)
            except Exception as e:
                self._warn(f"{name} chain read: {e}")
        gap_pts = self.instruments[name]["strike_gap"]
        for row in contracts:
            label = f"{name} {int(row['strike'])} {row['type']}"
            df = self._bars_for(row["token"], label, bar_close)
            if df is None or df.empty or len(df) < 35:
                continue
            d = indicators(_norm(df))
            i = self._bar_index(d, bar_close)
            if i is None or d["timestamp"].iloc[i].date() != self._now().date():
                continue
            chk = check_bar(d, i, expiry_day)
            label = f"{name} {int(row['strike'])} {row['type']}"
            fr = set(chk["fails"])
            if fr and not fr & {"no_cross", "crossed_earlier", "warmup", "time"}:
                # A real crossover happened on this bar but a check stopped the alert: keep why.
                self.skips[label] = {"day": self._now().strftime("%Y-%m-%d"), "bar": chk["ind"]["bar"][-5:],
                                     "why": chk["fails"], "ltp": chk["ind"]["close"],
                                     "macd_green_since": chk["ind"].get("macd_green_since"),
                                     "vwap": chk["ind"].get("vwap"), "vol_ratio": chk["ind"].get("vol_ratio")}
                nw = self._now()
                recent = [self._contract_last.get(label), self._form_last.get("skip:" + label)]
                if self.store and chk["ind"]["close"] >= MIN_PREMIUM and \
                        not any(x and nw - x < timedelta(minutes=CONTRACT_COOLDOWN_MIN) for x in recent):
                    self._form_last["skip:" + label] = nw
                    lot0 = int(row.get("lotsize") or self.instruments[name]["lot_size"])
                    self.store.add("skipped", {"day": self._now().strftime("%Y-%m-%d"), "time": self._now().strftime("%H:%M"),
                                               "contract": label, "token": row["token"], "bar": chk["ind"]["bar"][-5:],
                                               "lot": lot0, "why": chk["fails"], **levels(d, i, lot0),
                                               "outcome": {"result": "OPEN"}})
            if "crossed_earlier" in fr and label not in self.skips:
                # After a restart: look back at the cross bar itself to say why it was skipped.
                today_s = self._now().strftime("%Y-%m-%d")
                js = d.index[(d["timestamp"].dt.strftime("%Y-%m-%d") == today_s) &
                             (d["timestamp"].dt.strftime("%H:%M") == chk["ind"].get("crossed_at"))]
                if len(js):
                    c2 = check_bar(d, int(js[-1]), expiry_day)
                    if c2["fails"] and c2["ind"]:
                        self.skips[label] = {"day": today_s, "bar": c2["ind"]["bar"][-5:], "why": c2["fails"],
                                             "ltp": c2["ind"]["close"], "macd_green_since": c2["ind"].get("macd_green_since"),
                                             "vwap": c2["ind"].get("vwap"), "vol_ratio": c2["ind"].get("vol_ratio")}
            sk = self.skips.get(label)
            rows.append({"contract": label, "token": str(row["token"]), "ltp": chk["ind"].get("close"),
                         "missing": chk["fails"], "vol_ratio": chk["ind"].get("vol_ratio"),
                         "crossed_at": chk["ind"].get("crossed_at"),
                         "macd_green_since": chk["ind"].get("macd_green_since"),
                         "vwap": chk["ind"].get("vwap"),
                         "skip": sk if sk and sk["day"] == self._now().strftime("%Y-%m-%d") else None})
            fm = check_forming(d, i) if not chk["ok"] else None
            if fm and chk["ind"]["close"] >= MIN_PREMIUM:
                lot = int(row.get("lotsize") or self.instruments[name]["lot_size"])
                ltp = chk["ind"]["close"]
                pt = self._potential(d, i, ch, spot, row, gap_pts, ltp)
                forming.append({"stage": "forming", "contract": label, "token": row["token"],
                                "bar": chk["ind"]["bar"][-5:], "lot": lot, "ltp": ltp,
                                **pt, "per_lot": round(pt["potential"] * lot),
                                "gap_pct": fm["gap_pct"], "vol_ratio": fm["vol_ratio"],
                                "show": pt["potential"] >= MIN_POTENTIAL,
                                "chain_age_s": int((ch or {}).get("age") or 0) if ch else None})
            # Live crossover: blue went over red today, under 30 min ago. Older ones are history.
            j = cross_index(d, i)
            if j is not None and (i - j) * 5 < HOLD_MIN and d["close"].iloc[j] >= MIN_PREMIUM:
                lot = int(row.get("lotsize") or self.instruments[name]["lot_size"])
                cp, ltp = float(d["close"].iloc[j]), float(chk["ind"]["close"])
                c_at = d["timestamp"].iloc[j].strftime("%H:%M")
                pt = self._potential(d, j, ch, spot, row, gap_pts, cp,
                                     start=d["timestamp"].iloc[j] + timedelta(minutes=5))
                vr_j = d["volume"].iloc[j] / d["vavg20"].iloc[j] if d["vavg20"].iloc[j] == d["vavg20"].iloc[j] and d["vavg20"].iloc[j] else 0
                sk = self.skips.get(label)
                lv = {"stage": "just" if i - j <= 1 else "running", "contract": label, "token": str(row["token"]),
                      "crossed_at": c_at, "cross_price": round(cp, 2), "ltp": round(ltp, 2),
                      "lot": lot, **pt, "per_lot": round(pt["potential"] * lot),
                      "left": round(pt["target"] - ltp, 1), "mins": (i - j) * 5 + 5,
                      "done_pct": round((ltp - cp) / pt["potential"] * 100) if pt["potential"] else None,
                      "vol_ratio": round(float(vr_j), 2),
                      "alerted": any(a["contract"] == label and a["day"] == self._now().strftime("%Y-%m-%d") and
                                     a["bar"] >= (d["timestamp"].iloc[j] - timedelta(minutes=10)).strftime("%H:%M")
                                     for a in self.alerts),
                      "skip_why": sk["why"] if sk and sk["day"] == self._now().strftime("%Y-%m-%d") and sk["bar"] >= c_at else None,
                      "skip": sk if sk and sk["day"] == self._now().strftime("%Y-%m-%d") else None}
                live.append(lv)
                if self.store and START_HM <= c_at <= END_HM:
                    self.store.add("cross", {"day": self._now().strftime("%Y-%m-%d"), "time": self._now().strftime("%H:%M"),
                                             "contract": label, "token": row["token"], "bar": c_at, "lot": lot,
                                             "entry": round(cp, 2), "target": pt["target"], "potential": pt["potential"],
                                             "atr": pt["atr"], "reach_factor": pt["reach_factor"],
                                             "flow": pt["flow"], "flow_note": pt["flow_note"], "wall": pt["wall"],
                                             "room": pt["room"], "vol_ratio": lv["vol_ratio"],
                                             "outcome": {"result": "OPEN"}})
            if not chk["ok"] or chk["ind"]["close"] < MIN_PREMIUM:
                continue
            lot = int(row.get("lotsize") or self.instruments[name]["lot_size"])
            want = 1 if row["type"] == "CE" else -1
            score = 0
            extras = []
            if idx_trend == want:
                score += 2; extras.append(f"{name} 5m trend agrees")
            if bias is not None and bias * want > 0.1:
                score += 1; extras.append(f"Smart Money {bias:+.2f} agrees")
            if chk["ind"]["rsi"] and 50 <= chk["ind"]["rsi"] <= 72:
                score += 1; extras.append(f"RSI {chk['ind']['rsi']:.0f}")
            if chk["ind"]["vol_ratio"] >= 2:
                score += 1; extras.append(f"volume {chk['ind']['vol_ratio']:.1f}x")
            pt = self._potential(d, i, ch, spot, row, gap_pts, chk["ind"]["close"])
            if pt["flow"] == "with":
                score += 1; extras.append(f"OI: {pt['flow_note']}")
            elif pt["flow"] == "against":
                extras.append(f"OI against: {pt['flow_note']}")
            cands.append({"row": row, "d": d, "i": i, "lot": lot, "score": score, "pot": pt,
                          "extras": extras, "ind": chk["ind"], "label": label})
        with self.lock:
            self.last_scan[name] = {"bar": (bar_close - timedelta(minutes=5)).strftime("%H:%M"),
                                    "spot": spot, "index_trend": idx_trend, "bias": bias,
                                    "expiry": exp_date.isoformat() if exp_date else None,
                                    "contracts": rows, "forming": forming, "live": live,
                                    "reach_factor": self.reach_factor, "learn": self.learned,
                                    "checked_at": self._now().strftime("%H:%M:%S"),
                                    "took_s": round(time.time() - t_start, 1),
                                    "bar_source": {k: v for k, v in self.bar_source.items() if k.startswith(name)},
                                    "odds": FORM_ODDS}
        self._post_forming(name, forming)
        if self.store:
            now = self._now()
            for f in forming:
                if not f["show"]:
                    continue
                last = self._form_last.get("rec:" + f["contract"])
                if last and now - last < timedelta(minutes=HOLD_MIN):
                    continue
                self._form_last["rec:" + f["contract"]] = now
                self.store.add("forming", {**f, "day": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"),
                                           "entry": f["ltp"], "outcome": {"result": "OPEN"}})
        if not cands:
            return
        now = self._now()
        li = self._index_last.get(name)
        if li and now - li < timedelta(minutes=INDEX_COOLDOWN_MIN):
            return
        cands = [c for c in cands if not (self._contract_last.get(c["label"]) and
                                          now - self._contract_last[c["label"]] <
                                          timedelta(minutes=CONTRACT_COOLDOWN_MIN))]
        if not cands:
            return
        best = max(cands, key=lambda c: (c["score"], c["ind"]["vol_ratio"]))
        lv = levels(best["d"], best["i"], best["lot"])
        grade = "A" if best["score"] >= 3 else "B"
        alert = {"day": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"),
                 "bar": best["ind"]["bar"][-5:], "index": name, "contract": best["label"],
                 "symbol": best["row"]["symbol"], "token": best["row"]["token"],
                 "expiry": best["row"]["expiry"], "lot": best["lot"], "grade": grade,
                 "score": best["score"], "extras": best["extras"], "ind": best["ind"],
                 **lv, "exit_by": (now + timedelta(minutes=HOLD_MIN)).strftime("%H:%M"),
                 "outcome": {"result": "OPEN"}, "others": len(cands) - 1, "pot": best["pot"]}
        with self.lock:
            self.alerts.append(alert)
        if self.store:
            self.store.add("watch", {**alert, "bar": alert["bar"]})
        self._contract_last[best["label"]] = now
        self._index_last[name] = now
        self._info(f"WATCH {best['label']} grade {grade} entry {lv['entry']}")
        self._post(alert)

    def _potential(self, d, i, ch, spot, row, gap_pts, base, start=None):
        """How far the premium can run in the next 30 min from `base`: the premium's normal
        5-min swing (ATR x the factor learned from past crosses), capped by the room to the
        nearest big OI wall (index points x delta), halved when OI flow is against it."""
        atr = atr_at(d, i)
        oi = oi_read((ch or {}).get("chain") or [], (ch or {}).get("an") or {}, spot,
                     row["strike"], row["type"], gap_pts)
        reach = self.reach_factor * atr
        cap = oi["delta"] * oi["room"] if oi["room"] else None
        pot = min(reach, cap) if cap else reach
        if oi["flow"] == "against":
            pot *= 0.5
        return {"potential": round(pot, 1), "target": round(base + pot, 1), "wall": oi["wall"],
                "room": oi["room"], "flow": oi["flow"], "flow_note": oi["note"],
                "atr": round(atr, 2), "reach_factor": self.reach_factor,
                "oi_cap": round(cap, 1) if cap else None,
                "by": ((start or self._now()) + timedelta(minutes=HOLD_MIN)).strftime("%H:%M")}

    def _post_forming(self, name, forming):
        """Heads-up before the cross: best forming contract with real potential, 30-min cooldown
        per contract and 10 min per index so Slack stays readable."""
        now = self._now()
        ok = [f for f in forming if f["show"] and f["flow"] != "against"]
        if not ok:
            return
        li = self._form_last.get(name)
        if li and now - li < timedelta(minutes=10):
            return
        ok = [f for f in ok if not (self._form_last.get(f["contract"]) and
                                    now - self._form_last[f["contract"]] < timedelta(minutes=30))]
        if not ok:
            return
        f = max(ok, key=lambda x: x["potential"])
        self._form_last[name] = self._form_last[f["contract"]] = now
        f["slack"] = True
        wall = (f"next {'call' if f['contract'].endswith('CE') else 'put'} OI wall {int(f['wall'])} "
                f"({f['room']:.0f} pts away)") if f["wall"] else "no OI wall data"
        flow = f" · OI: {f['flow_note']}" if f["flow_note"] else ""
        try:
            self.send_slack(f"🟡 *FORMING {f['contract']}*: blue is {f['gap_pct']}% under red and closing in, "
                            f"MACD green, volume {f['vol_ratio']:.1f}x\n"
                            f"Now ₹{f['ltp']:.1f} · *could reach ₹{f['target']:.1f} (+{f['potential']:.0f} pts, "
                            f"+₹{f['per_lot']:,}/lot) by {f['by']}*\n"
                            f"_{wall}{flow}. {FORM_ODDS}. Wait for the cross alert or your own read of the chart._")
        except Exception as e:
            self._warn(f"slack forming: {e}")

    def _post(self, a):
        ind = a["ind"]
        lot = a["lot"]
        extras = " · ".join(a["extras"]) if a["extras"] else "index trend not confirming yet"
        msg = (f"👀 *WATCH {a['contract']}* ({a['expiry']}) · grade {a['grade']}\n"
               f"5m crossover {'done' if ind.get('stage') == 'crossed' else 'starting (blue meeting red)'} on the premium chart, MACD green "
               f"({ind['hist_prev']:+.1f} → {ind['hist']:+.1f}), volume {ind['vol_ratio']:.1f}x, "
               f"above VWAP {ind['vwap']:.1f}\n"
               f"*Buy near* ₹{a['entry']:.1f} · *SL* ₹{a['sl']:.1f} (₹{a['risk_per_lot']:,}/lot) · "
               f"*T1* ₹{a['t1']:.1f} (+₹{int(T1_RS):,}/lot) · *T2* ₹{a['t2']:.1f} (+₹{int(T2_RS):,}/lot)\n"
               f"Exit by {a['exit_by']} if T1 is not hit · lot {lot}\n"
               f"Could reach ₹{a['pot']['target']:.1f} (+{a['pot']['potential']:.0f} pts) by {a['pot']['by']}"
               f"{' · OI wall ' + str(int(a['pot']['wall'])) + ' (' + str(int(a['pot']['room'])) + ' pts away)' if a['pot'].get('wall') else ''}"
               f"{' · short of T1, think about booking near ₹' + format(a['pot']['target'], '.0f') if a['pot']['target'] < a['t1'] else ''}\n"
               f"_{extras}. Look at the chart first: this is a watch alert, not an engine trade._")
        try:
            self.send_slack(msg)
        except Exception as e:
            self._warn(f"slack failed: {e}")

    def _update_outcomes(self):
        """Score every open event (any kind, any day still in the candle window)."""
        if self.store:
            evs = self.store.rows("result='OPEN'")
        else:
            with self.lock:
                evs = [dict(a, kind="watch") for a in self.alerts if a["outcome"].get("result") == "OPEN"]
        for a in evs:
            try:
                kind = a.pop("kind")
                df = self.client.candles(a["token"], "NFO", "FIVE_MINUTE", 4)
                if df is None or df.empty:
                    continue
                d = _norm(df)
                hit = [j for j in d.index[d["timestamp"].dt.strftime("%H:%M") == a["bar"]]
                       if d["timestamp"].iloc[j].strftime("%Y-%m-%d") == a["day"]]
                if not hit:
                    continue
                if kind in ("forming", "cross"):
                    res = follow_forming(d, hit[-1], a["entry"], a["target"], a["lot"])
                else:
                    res = follow(d, hit[-1], {k: a[k] for k in ("entry", "sl", "t1", "t2")}, a["lot"])
                if res["result"] == "OPEN":
                    continue
                a["outcome"] = res
                if self.store:
                    self.store.update(kind, a)
                if kind == "watch":
                    with self.lock:
                        for m in self.alerts:
                            if m["contract"] == a["contract"] and m["bar"] == a["bar"]:
                                m["outcome"] = res
                    icon = {"T1": "✅", "T2": "✅", "SL": "❌", "TIME": "⏱"}.get(res["result"], "")
                    self.send_slack(f"{icon} WATCH result {a['contract']} ({a['time']}): {res['result']} "
                                    f"at {res.get('at')} · {'+' if res['pnl_per_lot'] >= 0 else ''}"
                                    f"₹{res['pnl_per_lot']:,}/lot · best ₹{res['max_seen']:.1f}")
                elif kind == "forming" and a.get("slack"):
                    self.send_slack(f"{'✅' if res['result'] == 'HIT' else '❌'} FORMING result {a['contract']} ({a['time']}): "
                                    f"{'reached' if res['result'] == 'HIT' else 'did not reach'} ₹{a['target']:.1f} "
                                    f"(best ₹{res['max_seen']:.1f}, {res['gain_pts']:+.0f} pts)")
            except Exception as e:
                self._warn(f"outcome {a.get('contract')}: {e}")
        self._maybe_scorecard()

    def _learn(self):
        """Learn from every scored crossover and forecast so the estimates get better daily.
        1) Reach factor: how many premium ATRs a setup really ran in 30 min. Set to the 40th
           percentile of past runs, so about 6 in 10 targets should be reached (once 15+ cases).
        2) Hit rate by condition (OI flow, time of day, volume, alert sent or not), so the
           scorecard shows which conditions are working and which are not."""
        if not self.store:
            return
        try:
            rows = self.store.rows("kind IN ('forming','cross') AND result!='OPEN'")
            runs = []
            for r in rows:
                o = r.get("outcome") or {}
                atr = r.get("atr") or ((r.get("reach_atr") or 0) / FORM_REACH_ATR)
                if atr and o.get("max_seen") is not None:
                    runs.append((o["max_seen"] - r["entry"]) / atr)
            if len(runs) >= 15:
                self.reach_factor = round(min(3.0, max(0.8, float(np.percentile(runs, 40)))), 2)
            alerted = {(w["contract"], w["day"]) for w in self.store.rows("kind='watch'")}
            groups = {
                "OI with": lambda r: r.get("flow") == "with",
                "OI neutral": lambda r: r.get("flow") == "neutral",
                "OI against": lambda r: r.get("flow") == "against",
                "before 11:00": lambda r: r.get("bar", "") < "11:00",
                "11:00 to 13:00": lambda r: "11:00" <= r.get("bar", "") < "13:00",
                "after 13:00": lambda r: r.get("bar", "") >= "13:00",
                "volume 2x+": lambda r: (r.get("vol_ratio") or 0) >= 2,
                "volume under 2x": lambda r: (r.get("vol_ratio") or 0) < 2,
                "cross with alert": lambda r: r["kind"] == "cross" and (r["contract"], r["day"]) in alerted,
                "cross, no alert": lambda r: r["kind"] == "cross" and (r["contract"], r["day"]) not in alerted,
            }
            out = {}
            for g, fn in groups.items():
                rs = [r for r in rows if fn(r)]
                if rs:
                    h = sum(1 for r in rs if r["outcome"]["result"] == "HIT")
                    out[g] = {"n": len(rs), "hit": h, "pct": round(h / len(rs) * 100)}
            self.learned = {"cases": len(runs), "reach_factor": self.reach_factor,
                            "default": FORM_REACH_ATR, "active": len(runs) >= 15, "groups": out,
                            "median_run_atr": round(float(np.median(runs)), 2) if runs else None}
        except Exception as e:
            self._warn(f"learn: {e}")

    def _maybe_scorecard(self):
        """Once a day after 15:35 IST: today's and all-time success rates to Slack."""
        now = self._now()
        if not self.store or now.strftime("%H:%M") < "15:35" or self._scorecard_day == now.strftime("%Y-%m-%d"):
            return
        self._scorecard_day = now.strftime("%Y-%m-%d")
        self._learn()
        today = score(self.store.rows("day=?", (self._scorecard_day,)))
        if not any(v["closed"] for v in today.values()):
            return
        allt = score(self.store.rows())
        def line(lbl, k, what):
            t, a = today[k], allt[k]
            return (f"*{lbl}*: today {t['success']} of {t['closed']}{' (' + str(t['pct']) + '%)' if t['closed'] else ''} {what}"
                    f" · all-time {a['success']} of {a['closed']}{' (' + str(a['pct']) + '%)' if a['closed'] else ''}")
        self.send_slack("📊 *Crossover watch scorecard*\n"
                        + line("Crossover alerts", "watch", f"hit T1 · {'+' if today['watch']['pnl_per_lot'] >= 0 else ''}₹{today['watch']['pnl_per_lot']:,}/lot") + "\n"
                        + line("Every crossover", "cross", "reached the expected price") + "\n"
                        + line("Forming forecasts", "forming", "reached the forecast price") + "\n"
                        + line("Skipped crosses", "skipped", "were right to skip")
                        + self._learn_line())

    def _learn_line(self):
        L = self.learned or {}
        if not L.get("cases"):
            return ""
        g = [(k, v) for k, v in (L.get("groups") or {}).items() if v["n"] >= 3]
        best = max(g, key=lambda x: x[1]["pct"]) if g else None
        worst = min(g, key=lambda x: x[1]["pct"]) if g else None
        txt = (f"\n🧠 *Learning*: {L['cases']} scored setups, typical 30-min run {L['median_run_atr']}x the premium's 5-min swing. "
               + (f"Estimates now use {L['reach_factor']}x (was {L['default']}x)." if L["active"]
                  else f"Estimates keep {L['default']}x until 15 cases."))
        if best and worst and best[0] != worst[0]:
            txt += (f" Working best: {best[0]} {best[1]['pct']}% ({best[1]['n']}). "
                    f"Weakest: {worst[0]} {worst[1]['pct']}% ({worst[1]['n']}).")
        return txt

    def stats(self):
        if not self.store:
            return {"error": "no store"}
        rows = self.store.rows()
        days = sorted({r["day"] for r in rows})
        return {"all_time": score(rows), "today": score([r for r in rows if r["day"] == self._now().strftime("%Y-%m-%d")]),
                "days": len(days), "since": days[0] if days else None, "learn": self.learned,
                "events": [{k: v for k, v in r.items() if k != "token"} for r in rows[-200:]]}

    def _with_live(self, scans):
        """Copy of last_scan with every price replaced by the live quote. The 5-minute
        candle close stays as ltp_bar; 'left' and 'done' are worked out from the live price."""
        out = {}
        for name, sc in scans.items():
            sc = dict(sc)
            for key in ("contracts", "live", "forming"):
                items = []
                for x in sc.get(key) or []:
                    x = dict(x)
                    px, at = self._live_price(x.get("token"))
                    if px is not None:
                        x["ltp_bar"], x["ltp"], x["ltp_at"] = x.get("ltp"), round(px, 2), at
                        if key == "live":
                            x["left"] = round(x["target"] - px, 1)
                            x["done_pct"] = round((px - x["cross_price"]) / x["potential"] * 100) if x.get("potential") else None
                    x.pop("token", None)
                    items.append(x)
                sc[key] = items
            out[name] = sc
        return out

    def _alert_live(self, a):
        x = {k: v for k, v in a.items() if k != "token"}
        px, at = self._live_price(a.get("token"))
        if px is not None:
            x["ltp"], x["ltp_at"] = round(px, 2), at
        return x

    def snapshot(self):
        with self.lock:
            done = [a for a in self.alerts if a["outcome"].get("result") in ("T1", "T2", "SL", "TIME")]
            wins = [a for a in done if a["outcome"]["result"] in ("T1", "T2")]
            return {"enabled": self.enabled, "running": self.running, "watch": self.watch_list,
                    "rules": {"ema": [EMA_FAST, EMA_SLOW], "vol_mult": VOL_MULT, "t1_rs": T1_RS,
                              "t2_rs": T2_RS, "sl_rs_cap": SL_RS_CAP, "hold_min": HOLD_MIN},
                    "alerts": [self._alert_live(a) for a in self.alerts],
                    "today": {"alerts": len(self.alerts), "closed": len(done), "wins": len(wins),
                              "pnl_per_lot": sum(a["outcome"]["pnl_per_lot"] for a in done)},
                    "last_scan": self._with_live(self.last_scan),
                    "tick_bars": {tok: [{"t": k.strftime("%H:%M"), "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"],
                                         "v": (b["v1"] or 0) - (b["v0"] or 0) if b["v0"] is not None else None, "full": b["full"]}
                                        for k, b in sorted(bars.items())[-4:]]
                                  for tok, bars in list(self._tbars.items())[:2]},
                    "live_prices": {"every_s": float(os.environ.get("PW_TICK_S", "5")),
                                    "last": max((v[1] for v in self.ltp_now.values()), default=None)},
                    "score": ({k: v for k, v in self.stats().items() if k != "events"} if self.store else None)}
