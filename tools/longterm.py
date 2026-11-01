#!/usr/bin/env python3
"""Daily long-term stock analysis for the Desk. Run by .github/workflows/longterm.yml.

Writes longterm/picks.json (today's ranking and ranges) and longterm/log.json
(every pick, with the price when it joined and left, so the picks are tracked to outcome).

Method and evidence are in SWING_STRATEGY.md, "Long-term picks". In short:
  rank    : 12-1 month price return across NIFTY 200, top 20 are the picks
  ranges  : log return ~ N(mu * T, sigma^2 * T); sigma is the stock's own 36-month volatility,
            mu is 13% a year plus a small excess for picks only, up to 3 years. Past 80% ranges
            built this way held 78-82% of the time at 3m, 1y, 3y and 5y. 20y cannot be tested.
Read-only, standard library only. Refuses to overwrite good data with a thin run.
"""
import csv, io, json, math, os, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "longterm", "picks.json")
LOG = os.path.join(ROOT, "longterm", "log.json")
IST = timezone(timedelta(hours=5, minutes=30))
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"

TOP_N = 20
Z = 1.2816                # 10th to 90th percentile
BASE_MU = 0.13            # Sensex median 11-13% a year over 1-20y windows, plus about 1% dividends
HORIZONS = [("3m", 3), ("1y", 12), ("3y", 36), ("5y", 60), ("20y", 240)]
# Yearly excess of the top 20 over an equal-weight basket: the lower of the pre-2016 and post-2016
# walk-forward results. Nothing at 5y or beyond, because the tests found no reliable edge there.
PICK_EXCESS = {3: 0.09, 12: 0.055, 36: 0.037, 60: 0.0, 240: 0.0}
BANK_LIKE = ("Financial Services",)


def get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def chart(symbol):
    """Dividend-adjusted daily closes for 5 years, oldest first."""
    y = symbol if symbol.startswith("^") else symbol + ".NS"
    d = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(y)}?range=5y&interval=1d"))
    res = d["chart"]["result"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    out = [(datetime.fromtimestamp(t, IST).strftime("%Y-%m-%d"), a) for t, a in zip(res["timestamp"], adj) if a and a > 0]
    # Yahoo's old adjusted closes contain split glitches: keep only what follows the last one.
    for i in range(len(out) - 1, 0, -1):
        if not 0.5 < out[i][1] / out[i - 1][1] < 2:
            return out[i:]
    return out


def momentum(px):
    """Return from 12 months ago to 1 month ago, the last month skipped as in the backtest."""
    return px[-22] / px[-253] - 1 if len(px) >= 253 else None


def volatility(px):
    """Yearly volatility from 36 monthly returns (21 sessions apart)."""
    if len(px) < 21 * 36 + 1:
        return None
    m = [px[-1 - 21 * i] for i in range(37)][::-1]
    r = [math.log(m[i] / m[i - 1]) for i in range(1, 37)]
    mean = sum(r) / len(r)
    return math.sqrt(sum((x - mean) ** 2 for x in r) / len(r)) * math.sqrt(12)


def ranges(sigma, is_pick):
    """Growth multiples per horizon: low (10th), typical (50th), high (90th), and chance of a loss."""
    out = {}
    for label, months in HORIZONS:
        t = months / 12
        mu = math.log(1 + BASE_MU + (PICK_EXCESS[months] if is_pick else 0)) * t
        sd = sigma * math.sqrt(t)
        out[label] = {"low": round(math.exp(mu - Z * sd), 3), "typical": round(math.exp(mu), 3),
                      "high": round(math.exp(mu + Z * sd), 3),
                      "loss_pct": round(50 * (1 + math.erf(-mu / (sd * math.sqrt(2))))),}
    return out


def financials(symbol, sector):
    """Latest annual figures from Yahoo. Descriptive only: these checks are not backtested."""
    types = "annualTotalRevenue,annualNetIncome,annualStockholdersEquity,annualTotalDebt,annualDilutedEPS,annualFreeCashFlow"
    u = (f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}.NS"
         f"?symbol={symbol}.NS&type={types}&period1=1400000000&period2={int(time.time()) + 86400}")
    series = {}
    for x in json.loads(get(u))["timeseries"]["result"]:
        key = x["meta"]["type"][0]
        series[key] = [i["reportedValue"]["raw"] for i in (x.get(key) or []) if i]
    last = lambda k: (series.get(k) or [None])[-1]
    def cagr(k):
        s = series.get(k) or []
        if len(s) < 2 or s[0] <= 0 or s[-1] <= 0:
            return None
        return round(((s[-1] / s[0]) ** (1 / (len(s) - 1)) - 1) * 100, 1)
    ni, eq, debt = last("annualNetIncome"), last("annualStockholdersEquity"), last("annualTotalDebt")
    out = {"revenue_growth": cagr("annualTotalRevenue"), "profit_growth": cagr("annualNetIncome"),
           "roe": round(ni / eq * 100, 1) if ni is not None and eq and eq > 0 else None,
           "debt_equity": round(debt / eq, 2) if debt is not None and eq and eq > 0 else None,
           "eps": last("annualDilutedEPS"), "fcf_positive": (last("annualFreeCashFlow") or 0) > 0 if last("annualFreeCashFlow") is not None else None,
           "years": len(series.get("annualTotalRevenue") or [])}
    flags = []
    loss_years = sum(1 for x in (series.get("annualNetIncome") or []) if x <= 0)
    if eq is not None and eq <= 0: flags.append("Owes more than it owns")
    if ni is not None and ni <= 0: flags.append("Made a loss last year")
    elif loss_years: flags.append(f"Loss in {loss_years} of the last {len(series['annualNetIncome'])} years")
    if out["profit_growth"] is not None and out["profit_growth"] < 0: flags.append("Profit is shrinking")
    if out["revenue_growth"] is not None and out["revenue_growth"] < 0: flags.append("Sales are shrinking")
    if sector not in BANK_LIKE:
        if out["debt_equity"] is not None and out["debt_equity"] > 1.5: flags.append("High debt")
        if out["fcf_positive"] is False: flags.append("Cash flow negative")
    out["flags"] = flags
    return out


def load(path, default):
    try:
        return json.load(open(path))
    except Exception:
        return default


def universe(previous):
    try:
        rows = list(csv.DictReader(io.StringIO(get(LIST_URL).decode())))
        u = {r["Symbol"]: {"name": r["Company Name"], "sector": r["Industry"]} for r in rows}
        if len(u) >= 150:
            return u
    except Exception as e:
        print(f"NSE list unavailable ({e}), using the last saved list", file=sys.stderr)
    u = {s["stock"]: {"name": s["name"], "sector": s["sector"]} for s in previous.get("stocks", [])}
    if not u:
        sys.exit("No stock list: NSE unreachable and no saved list")
    return u


def main():
    prev = load(OUT, {})
    uni = universe(prev)
    rows, failed = [], []
    for sym, meta in uni.items():
        try:
            bars = chart(sym)
            px = [b[1] for b in bars]
            mom, vol = momentum(px), volatility(px)
            if mom is None or vol is None:
                failed.append(f"{sym} (under 3 years of clean history)")
                continue
            rows.append({"stock": sym, **meta, "price": round(px[-1], 2), "price_date": bars[-1][0],
                         "mom12": round(mom * 100, 1), "vol": round(vol * 100, 1), "sigma": vol})
        except Exception as e:
            failed.append(f"{sym} ({type(e).__name__})")
        time.sleep(0.12)
    if len(rows) < 120:
        sys.exit(f"Only {len(rows)} stocks fetched, keeping yesterday's data. Failures: {failed[:10]}")

    rows.sort(key=lambda r: -r["mom12"])
    old_fin = {s["stock"]: s for s in prev.get("stocks", []) if s.get("fin")}
    for i, r in enumerate(rows):
        r["rank"] = i + 1
        r["pick"] = i < TOP_N
        r["ranges"] = ranges(r.pop("sigma"), r["pick"])
        if r["vol"] > 45: r["very_volatile"] = True
        if i < 40:
            try:
                r["fin"] = financials(r["stock"], r["sector"])
                time.sleep(0.12)
            except Exception:
                if r["stock"] in old_fin:
                    r["fin"] = old_fin[r["stock"]]["fin"]

    nifty = chart("^NSEI")
    nifty_now = nifty[-1][1]
    today = max(r["price_date"] for r in rows)
    px_now = {r["stock"]: r["price"] for r in rows}
    picks = [r["stock"] for r in rows if r["pick"]]

    # Track every pick to its outcome.
    log = load(LOG, {"picks": []})
    open_rows = {p["stock"]: p for p in log["picks"] if not p.get("left")}
    added, removed = [], []
    for s in picks:
        if s not in open_rows:
            log["picks"].append({"stock": s, "joined": today, "join_px": px_now[s], "join_nifty": round(nifty_now, 2)})
            added.append(s)
    for s, p in open_rows.items():
        if s not in picks:
            p.update(left=today, left_px=px_now.get(s), left_nifty=round(nifty_now, 2))
            removed.append(s)
    first_run = not open_rows
    def result(p):
        end = p.get("left_px") or px_now.get(p["stock"])
        end_n = p.get("left_nifty") or nifty_now
        return {**p, "ret": round((end / p["join_px"] - 1) * 100, 1) if end else None,
                "nifty_ret": round((end_n / p["join_nifty"] - 1) * 100, 1)}
    tracked = [result(p) for p in log["picks"]]

    data = {"updated": datetime.now(IST).strftime("%Y-%m-%d %H:%M"), "prices_through": today,
            "scanned": len(rows), "universe": len(uni), "failed": failed,
            "added_today": [] if first_run else added, "removed_today": removed,
            "nifty": round(nifty_now, 2), "stocks": rows, "tracked": tracked}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, "w"), separators=(",", ":"))
    json.dump(log, open(LOG, "w"), indent=0)
    print(f"{len(rows)}/{len(uni)} analysed, prices through {today}. Picks: {', '.join(picks)}. "
          f"Added {added if not first_run else '(first run)'}, removed {removed}. Failed {len(failed)}.")


if __name__ == "__main__":
    main()
