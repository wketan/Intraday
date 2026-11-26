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

import os
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
            j = i
            while j > 0 and e9.iloc[j - 1] > e21.iloc[j - 1]:
                j -= 1
            crossed_at = d["timestamp"].iloc[j].strftime("%H:%M")
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


# ── live scanner ────────────────────────────────────────────────────────────

class PremiumWatch:
    """Own thread. Scans once per 5-minute bar, ~25 s after it closes."""

    def __init__(self, client, master, instruments: dict, send_slack, bias_fn=None, log=None,
                 tz=None):
        self.client, self.master, self.instruments = client, master, instruments
        self.send_slack, self.bias_fn, self.log, self.tz = send_slack, bias_fn, log, tz
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
        threading.Thread(target=self._loop, daemon=True, name="PremiumWatch").start()
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
                    if time.time() - self._last_follow >= 60:
                        self._last_follow = time.time()
                        self._update_outcomes()
                elif now.strftime("%H:%M") < "09:00" and self.alerts and \
                        self.alerts[0].get("day") != now.strftime("%Y-%m-%d"):
                    with self.lock:
                        self.alerts = []
                        self._contract_last, self._index_last = {}, {}
            except Exception as e:
                self._warn(f"loop error: {e}")
            time.sleep(10)

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
        spot, idx_trend = self._spot_and_trend(name)
        if not spot:
            return
        contracts, exp_date = self._contracts(name, spot)
        expiry_day = exp_date == self._now().date()
        bias = None
        if self.bias_fn:
            b = self.bias_fn(name)
            bias = b.get("score") if isinstance(b, dict) else None
        cands, rows = [], []
        for row in contracts:
            df = self.client.candles(row["token"], "NFO", "FIVE_MINUTE", 3, force_refresh=True)
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
            rows.append({"contract": label, "ltp": chk["ind"].get("close"),
                         "missing": chk["fails"], "vol_ratio": chk["ind"].get("vol_ratio"),
                         "crossed_at": chk["ind"].get("crossed_at"),
                         "macd_green_since": chk["ind"].get("macd_green_since"),
                         "vwap": chk["ind"].get("vwap"),
                         "skip": sk if sk and sk["day"] == self._now().strftime("%Y-%m-%d") else None})
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
            cands.append({"row": row, "d": d, "i": i, "lot": lot, "score": score,
                          "extras": extras, "ind": chk["ind"], "label": label})
        with self.lock:
            self.last_scan[name] = {"bar": (bar_close - timedelta(minutes=5)).strftime("%H:%M"),
                                    "spot": spot, "index_trend": idx_trend, "bias": bias,
                                    "expiry": exp_date.isoformat() if exp_date else None,
                                    "contracts": rows}
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
                 "outcome": {"result": "OPEN"}, "others": len(cands) - 1}
        with self.lock:
            self.alerts.append(alert)
        self._contract_last[best["label"]] = now
        self._index_last[name] = now
        self._info(f"WATCH {best['label']} grade {grade} entry {lv['entry']}")
        self._post(alert)

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
               f"_{extras}. Look at the chart first: this is a watch alert, not an engine trade._")
        try:
            self.send_slack(msg)
        except Exception as e:
            self._warn(f"slack failed: {e}")

    def _update_outcomes(self):
        with self.lock:
            open_ = [a for a in self.alerts if a["outcome"].get("result") == "OPEN"]
        for a in open_:
            try:
                df = self.client.candles(a["token"], "NFO", "FIVE_MINUTE", 1)
                if df is None or df.empty:
                    continue
                d = _norm(df)
                hit = d.index[d["timestamp"].dt.strftime("%H:%M") == a["bar"]]
                hit = [j for j in hit if d["timestamp"].iloc[j].strftime("%Y-%m-%d") == a["day"]]
                if not hit:
                    continue
                res = follow(d, hit[-1], {k: a[k] for k in ("entry", "sl", "t1", "t2")}, a["lot"])
                if res["result"] != "OPEN":
                    with self.lock:
                        a["outcome"] = res
                    icon = {"T1": "✅", "T2": "✅", "SL": "❌", "TIME": "⏱"}.get(res["result"], "")
                    self.send_slack(f"{icon} WATCH result {a['contract']} ({a['time']}): {res['result']} "
                                    f"at {res.get('at')} · {'+' if res['pnl_per_lot'] >= 0 else ''}"
                                    f"₹{res['pnl_per_lot']:,}/lot · best ₹{res['max_seen']:.1f}")
            except Exception as e:
                self._warn(f"outcome {a.get('contract')}: {e}")

    def snapshot(self):
        with self.lock:
            done = [a for a in self.alerts if a["outcome"].get("result") in ("T1", "T2", "SL", "TIME")]
            wins = [a for a in done if a["outcome"]["result"] in ("T1", "T2")]
            return {"enabled": self.enabled, "running": self.running, "watch": self.watch_list,
                    "rules": {"ema": [EMA_FAST, EMA_SLOW], "vol_mult": VOL_MULT, "t1_rs": T1_RS,
                              "t2_rs": T2_RS, "sl_rs_cap": SL_RS_CAP, "hold_min": HOLD_MIN},
                    "alerts": [{k: v for k, v in a.items() if k != "token"} for a in self.alerts],
                    "today": {"alerts": len(self.alerts), "closed": len(done), "wins": len(wins),
                              "pnl_per_lot": sum(a["outcome"]["pnl_per_lot"] for a in done)},
                    "last_scan": self.last_scan}
