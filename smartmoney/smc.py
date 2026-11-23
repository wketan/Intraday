"""Smart-money read of one chart: structure breaks (BOS / CHoCH), order blocks, fair value
gaps, liquidity (previous day high / low, opening range) and their sweeps, VWAP, and the
day's events in order. Works on any OHLC frame with a UTC index (IST trading days)."""
import numpy as np
import pandas as pd

from . import structure as S
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")


def _day(idx):
    return pd.Index(idx.tz_convert(IST).date)


def _hm(t):
    return pd.Timestamp(t).tz_convert(IST).strftime("%H:%M")


def fvgs(df, lookback=300):
    """Three-candle gaps: bar i's low above bar i-2's high (bullish) or its high below bar i-2's
    low (bearish). Each one is tracked until price trades back into it (touched) or through it (filled)."""
    d = df.iloc[-lookback:]
    h, l, c = d["high"].values, d["low"].values, d["close"].values
    out = []
    for i in range(2, len(d)):
        if l[i] > h[i - 2]:
            out.append({"dir": 1, "lo": float(h[i - 2]), "hi": float(l[i]), "i": i})
        elif h[i] < l[i - 2]:
            out.append({"dir": -1, "lo": float(h[i]), "hi": float(l[i - 2]), "i": i})
    for g in out:
        after = slice(g["i"] + 1, len(d))
        lo_after, hi_after = l[after], h[after]
        if g["dir"] > 0:
            g["touched"] = bool((lo_after <= g["hi"]).any())
            g["filled"] = bool((lo_after <= g["lo"]).any())
        else:
            g["touched"] = bool((hi_after >= g["lo"]).any())
            g["filled"] = bool((hi_after >= g["hi"]).any())
        g["time"] = d.index[g["i"]]
        g["status"] = "filled" if g["filled"] else "touched, not filled" if g["touched"] else "untouched"
    return out


def order_blocks(df, events, lookback=300):
    """For every structure break, the last opposite-colour candle before the move that broke it:
    where the big orders went in. Tested once price comes back into it; broken on a close through it."""
    d = df.iloc[-lookback:]
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    idx = {t: i for i, t in enumerate(d.index)}
    out = []
    for _, e in events.iterrows():
        k = idx.get(e["time"])
        if k is None:
            continue
        s = 1 if e["dir"] == "bull" else -1
        j = None
        for m in range(k - 1, max(k - 15, 0), -1):        # walk back to the start of the impulse
            if (c[m] < o[m]) if s > 0 else (c[m] > o[m]):
                j = m
                break
        if j is None:
            continue
        ob = {"dir": s, "lo": float(l[j]), "hi": float(h[j]), "time": d.index[j], "formed": d.index[k], "type": e["type"]}
        after = slice(k + 1, len(d))
        ob["tested"] = bool(((l[after] <= ob["hi"]) if s > 0 else (h[after] >= ob["lo"])).any())
        ob["broken"] = bool(((c[after] < ob["lo"]) if s > 0 else (c[after] > ob["hi"])).any())
        out.append(ob)
    return out


def vwap(df):
    """Volume-weighted average price of each IST day (needs volume; the index has none)."""
    if "volume" not in df or df["volume"].fillna(0).sum() == 0:
        return None
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = _day(df.index)
    v = df["volume"].fillna(0)
    return (tp * v).groupby(day.values).cumsum() / v.groupby(day.values).cumsum().replace(0, np.nan)


def read(df, swing=2, daily=None):
    """Everything the levels panel shows for one intraday frame."""
    if len(df) < 30:
        return None
    a = S.analyse(df, swing)
    d = a["df"]
    ev = a["events"]
    day = _day(d.index)
    today = d[np.asarray(day == day[-1])]
    prev = d[np.asarray(day == sorted(set(day))[-2])] if len(set(day)) > 1 else None
    last = float(d["close"].iloc[-1])
    out = {"price": last, "bias": a["bias"], "atr": a["atr"]}
    tev = ev[ev["time"] >= today.index[0]] if len(ev) else ev
    le = ev.iloc[-1] if len(ev) else None
    out["structure"] = {"last": None if le is None else {"type": le["type"], "dir": le["dir"], "level": round(float(le["level"]), 2),
                                                          "time": _hm(le["time"]), "today": bool(le["time"] >= today.index[0])},
                        "breaks_today": int(len(tev))}
    obs = [x for x in order_blocks(d, ev) if not x["broken"]]
    fv = [x for x in fvgs(d) if not x["filled"]]
    near = lambda zs: sorted(zs, key=lambda z: abs((z["lo"] + z["hi"]) / 2 - last))[:3]
    fmt = lambda z: {"dir": z["dir"], "lo": round(z["lo"], 2), "hi": round(z["hi"], 2), "time": int(z["time"].timestamp()),
                     "dist_pct": round(((z["hi"] if last > z["hi"] else z["lo"] if last < z["lo"] else last) / last - 1) * 100, 2),
                     **({"tested": z["tested"]} if "tested" in z else {"status": z["status"]})}
    out["order_blocks"] = [fmt(z) for z in near(obs)]
    out["fvgs"] = [fmt(z) for z in near(fv)]
    liq = {}
    if daily is not None and len(daily) >= 2:
        ddays = _day(daily.index)
        pdrow = daily[np.asarray(ddays < day[-1])].iloc[-1]
        liq["pdh"], liq["pdl"], liq["pdc"] = float(pdrow["high"]), float(pdrow["low"]), float(pdrow["close"])
    elif prev is not None and len(prev):
        liq["pdh"], liq["pdl"], liq["pdc"] = float(prev["high"].max()), float(prev["low"].min()), float(prev["close"].iloc[-1])
    m = (today.index.tz_convert(IST).hour * 60 + today.index.tz_convert(IST).minute) - 555
    orng = today[np.asarray(m < 15)]
    if len(orng):
        liq["or_hi"], liq["or_lo"] = float(orng["high"].max()), float(orng["low"].min())
    swept = []
    for k, side in (("pdh", 1), ("pdl", -1)):
        if k in liq:
            hit = today[today["high"] > liq[k]] if side > 0 else today[today["low"] < liq[k]]
            if len(hit):
                swept.append(f"{k.upper()} {_hm(hit.index[0])}")
    liq["swept"] = swept
    out["liquidity"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in liq.items()}
    vw = vwap(d)
    if vw is not None:
        out["vwap"] = round(float(vw.iloc[-1]), 2)
        up = today[today["close"] >= today["open"]]["volume"].sum()
        tot = today["volume"].sum()
        out["buyers_pct"] = round(float(up / tot * 100)) if tot else None
    # the day's events in order
    evs = []
    if "pdc" in liq and len(today):
        o = float(today["open"].iloc[0])
        gap = (o / liq["pdc"] - 1) * 100
        if abs(gap) >= 0.2:
            fill = today[today["low"] <= liq["pdc"]] if gap > 0 else today[today["high"] >= liq["pdc"]]
            evs.append((today.index[0], f"Gap {'up' if gap > 0 else 'down'} {gap:+.1f}%"))
            if len(fill):
                evs.append((fill.index[0], "Gap filled"))
    for _, e in tev.iterrows():
        evs.append((e["time"], f"{e['type']} {'up' if e['dir'] == 'bull' else 'down'}"))
    for z in fvgs(d):
        if z["time"] >= today.index[0]:
            evs.append((z["time"], f"FVG formed ({'bull' if z['dir'] > 0 else 'bear'})"))
    for z in order_blocks(d, ev):
        if z["formed"] >= today.index[0]:
            evs.append((z["formed"], f"Order block formed ({'bull' if z['dir'] > 0 else 'bear'})"))
    for s in swept:
        k, hm = s.split(" ")
        evs.append((pd.Timestamp(f"{day[-1]} {hm}", tz=IST), f"{k} swept"))
    evs.sort(key=lambda x: x[0])
    gaps = [(t, x) for t, x in evs if x.startswith("Gap")]
    rest = [(t, x) for t, x in evs if not x.startswith("Gap")]
    out["events"] = [{"time": _hm(t), "text": x} for t, x in gaps + rest[-(16 - len(gaps)):]]
    out["chart"] = {"events": [{"time": int(r["time"].timestamp()), "type": r["type"], "dir": r["dir"], "level": float(r["level"])} for _, r in ev.iterrows()][-40:],
                    "obs": [{**fmt(z), "end": None} for z in obs[-6:]], "fvgs": [fmt(z) for z in fv[-8:]],
                    "vwap": [{"time": int(t.timestamp()), "value": round(float(v), 2)} for t, v in vw.items() if t >= today.index[0] and not np.isnan(v)] if vw is not None else []}
    return out
