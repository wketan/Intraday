"""Market structure: swings, HH/HL/LH/LL, break of structure (BOS) and change of
character (CHoCH), support / resistance zones, session ranges and volatility regime."""
import numpy as np
import pandas as pd


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def add_indicators(df):
    df = df.copy()
    df["atr"] = atr(df)
    for n in (20, 50, 200):
        df[f"ema{n}"] = ema(df["close"], n)
    df["rsi"] = rsi(df["close"])
    return df


def swings(df, n=3):
    """Fractal pivots: a high higher than the n bars each side (and the mirror for lows).
    A pivot is only known n bars later, so `confirmed_at` is when it could be traded."""
    h, l = df["high"].values, df["low"].values
    out = []
    for i in range(n, len(df) - n):
        if h[i] == h[i - n:i + n + 1].max():
            out.append((df.index[i], "H", float(h[i]), df.index[i + n]))
        if l[i] == l[i - n:i + n + 1].min():
            out.append((df.index[i], "L", float(l[i]), df.index[i + n]))
    return pd.DataFrame(out, columns=["time", "kind", "price", "confirmed_at"])


def label_swings(sw):
    """Tag each swing HH / LH (highs) or HL / LL (lows) against the previous one of its kind."""
    sw = sw.copy()
    labels = []
    last = {"H": None, "L": None}
    for k, p in zip(sw["kind"], sw["price"]):
        prev = last[k]
        if prev is None:
            labels.append(k)
        elif k == "H":
            labels.append("HH" if p > prev else "LH")
        else:
            labels.append("HL" if p > prev else "LL")
        last[k] = p
    sw["label"] = labels
    return sw


def structure_events(df, sw):
    """Walk the closes: a close above the last confirmed swing high is a bullish break,
    below the last swing low a bearish one. Against the running trend it is a CHoCH,
    with it a BOS."""
    events, trend = [], 0
    hi = lo = None
    used_hi = used_lo = None
    sw = sw.sort_values("confirmed_at")
    j = 0
    times, closes = df.index, df["close"].values
    for i, t in enumerate(times):
        while j < len(sw) and sw["confirmed_at"].iloc[j] <= t:
            r = sw.iloc[j]
            if r["kind"] == "H":
                hi = r
            else:
                lo = r
            j += 1
        c = closes[i]
        if hi is not None and c > hi["price"] and used_hi is not hi["time"]:
            events.append((t, "CHoCH" if trend < 0 else "BOS", "bull", hi["price"]))
            trend, used_hi = 1, hi["time"]
        elif lo is not None and c < lo["price"] and used_lo is not lo["time"]:
            events.append((t, "CHoCH" if trend > 0 else "BOS", "bear", lo["price"]))
            trend, used_lo = -1, lo["time"]
    return pd.DataFrame(events, columns=["time", "type", "dir", "level"]), trend


def zones(df, sw, max_zones=8):
    """Cluster swing prices that sit within half an ATR of each other into S/R zones,
    scored by how many times price turned there and how recently."""
    if sw.empty:
        return []
    a = float(df["atr"].iloc[-1])
    price = float(df["close"].iloc[-1])
    end = df.index[-1]
    span = (end - df.index[0]).total_seconds() or 1
    pts = sw.sort_values("price")
    clusters, cur = [], [pts.iloc[0]]
    for _, r in pts.iloc[1:].iterrows():
        # chain neighbours within half an ATR, but never let one zone grow past one ATR wide
        if r["price"] - cur[-1]["price"] <= a * 0.5 and r["price"] - cur[0]["price"] <= a:
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]
    clusters.append(cur)
    out = []
    for c in clusters:
        ps = [r["price"] for r in c]
        recency = max(1 - (end - r["time"]).total_seconds() / span for r in c)
        score = len(c) + 2 * recency
        mid = float(np.mean(ps))
        out.append({"low": round(float(min(ps)), 2), "high": round(float(max(ps)), 2), "mid": round(mid, 2),
                    "touches": len(c), "score": round(score, 2),
                    "type": "resistance" if mid > price else "support",
                    "dist_atr": round((mid - price) / a, 2)})
    out = [z for z in out if z["touches"] >= 2]
    out.sort(key=lambda z: (abs(z["dist_atr"]) > 6, -z["score"]))
    return sorted(out[:max_zones], key=lambda z: -z["mid"])


def session_levels(m15):
    """Today's opening range and range, yesterday's high / low / close with its central pivot
    range, and last week's high / low (IST trading days)."""
    if m15.empty:
        return {}
    from .market import IST
    day = pd.Index(m15.index.tz_convert(IST).date)
    today = m15[day == day[-1]]
    out = {}
    if len(today):
        out["Opening 15m"] = {"high": round(float(today["high"].iloc[:1].max()), 2), "low": round(float(today["low"].iloc[:1].min()), 2)}
        out["Today"] = {"high": round(float(today["high"].max()), 2), "low": round(float(today["low"].min()), 2)}
    daily = m15.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    if len(daily) > 1:
        y = daily.iloc[-2]
        out["Prev day"] = {"high": round(float(y["high"]), 2), "low": round(float(y["low"]), 2), "close": round(float(y["close"]), 2)}
        pv = (y["high"] + y["low"] + y["close"]) / 3
        bc = (y["high"] + y["low"]) / 2
        tc = 2 * pv - bc
        out["CPR"] = {"high": round(float(max(tc, bc)), 2), "low": round(float(min(tc, bc)), 2), "pivot": round(float(pv), 2),
                      "width_pct": round(float(abs(tc - bc) / pv * 100), 3)}
    daily.index = pd.to_datetime(daily.index)
    weekly = daily.resample("W-FRI").agg({"high": "max", "low": "min"}).dropna()
    if len(weekly) > 1:
        out["Prev week"] = {"high": round(float(weekly["high"].iloc[-2]), 2), "low": round(float(weekly["low"].iloc[-2]), 2)}
    return out


def regime(df):
    """Volatility: where today's ATR sits among the last year of ATRs.
    Trend: price vs the 50/200 EMAs and the slope of the 50."""
    a = df["atr"].dropna()
    pct = float((a.iloc[-260:] < a.iloc[-1]).mean() * 100)
    vol = "low" if pct < 30 else "high" if pct > 75 else "normal"
    c, e50, e200 = df["close"].iloc[-1], df["ema50"].iloc[-1], df["ema200"].iloc[-1]
    slope = (df["ema50"].iloc[-1] / df["ema50"].iloc[-10] - 1) * 100
    if c > e50 > e200 and slope > 0:
        trend = "uptrend"
    elif c < e50 < e200 and slope < 0:
        trend = "downtrend"
    else:
        trend = "range"
    return {"volatility": vol, "atr_pctile": round(pct), "trend": trend, "ema50_slope": round(float(slope), 2)}


def analyse(df, swing_n=3):
    """Full structure read of one timeframe."""
    df = add_indicators(df)
    sw = label_swings(swings(df, swing_n))
    ev, trend = structure_events(df, sw)
    last = df.iloc[-1]
    bias = {1: "bullish", -1: "bearish", 0: "neutral"}[trend]
    return {
        "df": df,
        "swings": sw,
        "events": ev,
        "bias": bias,
        "last_event": ev.iloc[-1].to_dict() if len(ev) else None,
        "recent_swings": sw.tail(6)[["time", "label", "price"]].to_dict("records"),
        "zones": zones(df, sw.tail(120)),
        "regime": regime(df),
        "price": round(float(last["close"]), 2),
        "atr": round(float(last["atr"]), 2),
        "rsi": round(float(last["rsi"]), 1),
    }
