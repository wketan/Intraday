"""Smart Money tape: every F&O stock plus NIFTY, BANKNIFTY, FINNIFTY and SENSEX.

Port of market-lab's sm/feed.py (MIT, see LICENSE-market-lab) onto Angel One:
NSE blocks cloud IPs, so the engine cannot use NSE's market watch the way market-lab
does on a home computer. Instead, every POLL_SEC seconds we pull Angel batch quotes
(mode FULL, 50 tokens a call) for every F&O stock's cash price and its near-month
future. That gives, per stock:

  open / high / low / previous close / LTP / volume      -> move since open, displacement
  total buy qty vs total sell qty in the order book       -> order-book pressure
  near-month futures price and open interest              -> OI build-up (who is adding positions)

One-minute closes are recorded from those polls and turned into 5-minute bars, which
feed market-lab's own structure detectors (smc.py / structure.py, copied unchanged):
break of structure, change of character, order blocks, fair-value gaps, PDH/PDL sweeps.

Daily levels (PDH, PDL, a normal day's range) come from Yahoo once a day, with Angel
daily candles as the fallback for the indices.

Nothing here places orders. It is read-only market data.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from . import smc

log = logging.getLogger("smartmoney")
IST = ZoneInfo("Asia/Kolkata")
ORIGIN = pd.Timestamp("2000-01-03 03:45", tz="UTC")      # 09:15 IST, aligns 5-minute buckets
POLL_SEC = int(os.environ.get("SM_POLL_SEC", "30"))
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0"}

# Angel index tokens (exchange, token) and the Yahoo ticker used only for daily levels
INDICES = {
    "NIFTY": ("NSE", "99926000", "^NSEI"),
    "BANKNIFTY": ("NSE", "99926009", "^NSEBANK"),
    "FINNIFTY": ("NSE", "99926037", "NIFTY_FIN_SERVICE.NS"),
    "SENSEX": ("BSE", "99919000", "^BSESN"),
}

# ── trading groups for the sector chips (verbatim from market-lab sm/feed.py:28-54) ──
INSURANCE = {"SBILIFE", "HDFCLIFE", "ICICIPRULI", "ICICIGI", "LICI", "POLICYBZR", "NIACL", "GICRE", "STARHEALTH"}
CAP_MKTS = {"BSE", "MCX", "CDSL", "CAMS", "ANGELONE", "KFINTECH", "NUVAMA", "360ONE", "HDFCAMC", "NAM-INDIA", "MOTILALOFS", "IIFL", "UTIAMC", "ABSLAMC"}
PSU_BANK = {"SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "IOB", "UCOBANK", "CENTRALBK", "MAHABANK"}
DEFENCE = {"HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "BEML", "DATAPATTNS", "SOLARINDS", "ASTRAMICRO", "PARAS", "MIDHANI"}
NEW_AGE = {"ETERNAL", "ZOMATO", "PAYTM", "NYKAA", "SWIGGY", "IXIGO", "CARTRADE", "NAUKRI"}
LOGISTICS = {"DELHIVERY", "CONCOR", "BLUEDART", "TCI", "ADANIPORTS", "GMRAIRPORT", "JSWINFRA", "INDIGO"}
TELECOM = {"BHARTIARTL", "IDEA", "INDUSTOWER", "TATACOMM", "HFCL", "TEJASNET"}
GROUP = {"Oil Gas & Consumable Fuels": "OIL & GAS", "Consumer Durables": "CONSUMER", "Consumer Services": "CONSUMER", "Textiles": "CONSUMER",
         "Capital Goods": "CAP GOODS", "Construction": "CAP GOODS", "Metals & Mining": "METAL", "Construction Materials": "CEMENT",
         "Power": "POWER", "Automobile and Auto Components": "AUTO", "Healthcare": "HEALTHCARE", "Chemicals": "CHEM", "Realty": "REALTY",
         "Information Technology": "IT", "Fast Moving Consumer Goods": "FMCG", "Telecommunication": "TELECOM", "Services": "LOGISTICS",
         "Media Entertainment & Publication": "OTHERS", "Diversified": "OTHERS", "Forest Materials": "OTHERS"}
PHARMA_HINT = {"SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AUROPHARMA", "ZYDUSLIFE", "ALKEM", "TORNTPHARM", "BIOCON",
               "GLENMARK", "LAURUSLABS", "MANKIND", "IPCALAB", "GRANULES", "PPLPHARMA", "AJANTPHARM", "NATCOPHARM", "ABBOTINDIA"}
BANKS = {"HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB", "BANDHANBNK", "AUBANK", "RBLBANK", "YESBANK", "CUB"}

# NIFTY 50 / BANK NIFTY heavyweights, used for the index-level "buyers vs sellers" read
NIFTY_HEAVY = {"HDFCBANK": 13, "ICICIBANK": 9, "RELIANCE": 9, "INFY": 5, "BHARTIARTL": 5, "LT": 4, "ITC": 3.5, "TCS": 3.5,
               "AXISBANK": 3, "SBIN": 3, "KOTAKBANK": 3, "M&M": 2.5, "BAJFINANCE": 2.5, "HINDUNILVR": 2, "SUNPHARMA": 1.7,
               "HCLTECH": 1.7, "MARUTI": 1.6, "NTPC": 1.4, "TITAN": 1.3, "ULTRACEMCO": 1.2, "TATAMOTORS": 1.2, "ETERNAL": 1.2,
               "POWERGRID": 1.1, "TRENT": 1.1, "TATASTEEL": 1.1, "ASIANPAINT": 1, "BEL": 1}
BANK_HEAVY = {"HDFCBANK": 28, "ICICIBANK": 25, "SBIN": 9, "KOTAKBANK": 9, "AXISBANK": 8, "INDUSINDBK": 3, "FEDERALBNK": 3,
              "BANKBARODA": 3, "IDFCFIRSTB": 2.5, "AUBANK": 2.5, "PNB": 2.5, "CANBK": 2.5}

try:
    INDUSTRY = json.load(open(os.path.join(HERE, "industries.json")))    # symbol -> [company, NSE industry]
except Exception:
    INDUSTRY = {}


def sector_of(sym):
    industry = (INDUSTRY.get(sym) or ["", ""])[1]
    for s, g in ((INSURANCE, "INSURANCE"), (CAP_MKTS, "CAP MKTS"), (PSU_BANK, "PSU BANK"), (DEFENCE, "DEFENCE"),
                 (NEW_AGE, "NEW-AGE"), (LOGISTICS, "LOGISTICS"), (TELECOM, "TELECOM")):
        if sym in s:
            return g
    if industry == "Financial Services":
        return "BANK" if sym.endswith("BANK") or sym in BANKS else "NBFC"
    if industry == "Healthcare":
        return "PHARMA" if sym in PHARMA_HINT else "HEALTHCARE"
    return GROUP.get(industry, "OTHERS")


def now_ist():
    return datetime.now(IST)


def minute_now():
    t = now_ist()
    return t.hour * 60 + t.minute - 555          # minutes since 09:15


def market_status():
    t = now_ist()
    if t.weekday() >= 5:
        return "closed"
    m = t.hour * 60 + t.minute
    if m < 555:
        return "pre-open"
    if m <= 930:
        return "open"
    return "closed"


def buildup(price_chg, oi_chg):
    """Standard F&O read of price vs open-interest change (both in %)."""
    if price_chg is None or oi_chg is None or abs(oi_chg) < 0.5:
        return None
    if price_chg >= 0 and oi_chg > 0:
        return "LONG_BUILDUP"        # fresh buying
    if price_chg < 0 and oi_chg > 0:
        return "SHORT_BUILDUP"       # fresh selling
    if price_chg >= 0 and oi_chg < 0:
        return "SHORT_COVERING"      # sellers exiting
    return "LONG_UNWINDING"          # buyers exiting


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


class SmartMoneyFeed:
    def __init__(self, client, master):
        self.client = client            # AngelClient (has .api SmartConnect)
        self.master = master            # InstrumentMaster
        self.lock = threading.Lock()
        self.universe = {}              # sym -> {eq_token, fut_token, fut_symbol, name}
        self.closes = {}                # sym -> {minute: ltp}  (today's 1-minute closes)
        self.daily = {}                 # sym -> {pdh, pdl, pdc, range}
        self.daily_idx = {}             # index key -> daily DataFrame for smc.read
        self.oi_base = {}               # fut symbol -> first OI seen today
        self.stocks, self.indices = {}, {}
        self.day = None
        self.time = None
        self.errors = []
        self.running = False
        self.last_struct = 0.0
        self.base_path = os.path.join(os.path.dirname(os.environ.get("DB_PATH", "signals.db")) or ".", "sm_oi_base.json")

    # ── universe ────────────────────────────────────────────────────────
    def build_universe(self):
        if not getattr(self.master, "loaded", False):
            return False
        today = now_ist().date()
        futs = {}
        for it in self.master.data:
            if it.get("exch_seg") != "NFO" or it.get("instrumenttype") != "FUTSTK":
                continue
            name = (it.get("name") or "").upper()
            try:
                exp = datetime.strptime(it.get("expiry", ""), "%d%b%Y").date()
            except ValueError:
                continue
            if exp < today:
                continue
            if name not in futs or exp < futs[name][0]:
                futs[name] = (exp, it.get("token"), it.get("symbol"))
        uni = {}
        for name, (exp, ftok, fsym) in futs.items():
            eq = self.master.nse_eq.get(name + "-EQ") or self.master.nse_eq.get(name)
            if not eq:
                continue
            uni[name] = {"eq_token": str(eq["token"]), "fut_token": str(ftok), "fut_symbol": fsym,
                         "name": (INDUSTRY.get(name) or [name.title()])[0], "fut_expiry": exp.isoformat()}
        self.universe = uni
        log.info(f"[SM] universe: {len(uni)} F&O stocks")
        return bool(uni)

    # ── data ────────────────────────────────────────────────────────────
    def _quote(self, exch_tokens):
        """Angel FULL quotes, 50 tokens a call. Returns {(exchange, token): row}."""
        api = getattr(self.client, "api", None)
        if api is None:
            return {}
        out = {}
        for exch, toks in exch_tokens.items():
            for i in range(0, len(toks), 50):
                try:
                    r = api.getMarketData(mode="FULL", exchangeTokens={exch: toks[i:i + 50]})
                    for it in ((r or {}).get("data") or {}).get("fetched", []) or []:
                        out[(it.get("exchange") or exch, str(it.get("symbolToken")))] = it
                except Exception as e:
                    self._err(f"quote {exch}: {e}")
                time.sleep(0.5)
        return out

    def _err(self, msg):
        self.errors = (self.errors + [f"{now_ist():%H:%M:%S} {msg}"])[-20:]
        log.warning(f"[SM] {msg}")

    def _yahoo_daily(self, ticker, rng="3mo"):
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}?interval=1d&range={rng}"
        d = requests.get(u, headers=UA, timeout=15).json()["chart"]["result"][0]
        o = d["indicators"]["quote"][0]
        return pd.DataFrame({k: o[k] for k in ("open", "high", "low", "close", "volume")},
                            index=pd.to_datetime(d["timestamp"], unit="s", utc=True)).dropna(subset=["close"])

    def load_daily(self):
        today = now_ist().date()
        for k, (exch, tok, yt) in INDICES.items():
            try:
                self.daily_idx[k] = self._yahoo_daily(yt)
            except Exception as e:
                self._err(f"daily {k}: {e}")
        n = 0
        for s in list(self.universe):
            try:
                df = self._yahoo_daily(s + ".NS")
                past = df[[t.tz_convert(IST).date() < today for t in df.index]]
                if len(past) >= 15:
                    rng = (past["high"] - past["low"]) / past["close"] * 100
                    self.daily[s] = {"pdh": float(past["high"].iloc[-1]), "pdl": float(past["low"].iloc[-1]),
                                     "pdc": float(past["close"].iloc[-1]), "range": float(rng.iloc[-20:].mean())}
                    n += 1
            except Exception:
                pass
            time.sleep(0.15)
        log.info(f"[SM] daily levels for {n}/{len(self.universe)} stocks")

    def _load_oi_base(self):
        try:
            d = json.load(open(self.base_path))
            if d.get("day") == str(self.day):
                self.oi_base = d.get("oi", {})
        except Exception:
            self.oi_base = {}

    def _save_oi_base(self):
        try:
            json.dump({"day": str(self.day), "oi": self.oi_base}, open(self.base_path, "w"))
        except Exception:
            pass

    # ── one poll ────────────────────────────────────────────────────────
    def cycle(self, structure):
        uni = self.universe
        req = {"NSE": [u["eq_token"] for u in uni.values()] + [t for e, t, _ in INDICES.values() if e == "NSE"],
               "NFO": [u["fut_token"] for u in uni.values()],
               "BSE": [t for e, t, _ in INDICES.values() if e == "BSE"]}
        q = self._quote(req)
        if not q:
            return
        m = minute_now()
        stocks = {}
        new_base = False
        for s, u in uni.items():
            r = q.get(("NSE", u["eq_token"]))
            if not r:
                continue
            ltp, opn, prev = _f(r.get("ltp")), _f(r.get("open")), _f(r.get("close"))
            hi, lo, vol = _f(r.get("high")), _f(r.get("low")), _f(r.get("tradeVolume")) or 0
            if not (ltp and opn and prev):
                continue
            cl = self.closes.setdefault(s, {})
            if 0 <= m <= 375:
                cl[m] = ltp
            fo = (ltp / opn - 1) * 100
            dly = self.daily.get(s) or {}
            tb, ts = _f(r.get("totBuyQuan")) or 0, _f(r.get("totSellQuan")) or 0
            row = {"sym": s, "name": u["name"], "sector": sector_of(s), "ltp": round(ltp, 2),
                   "pct": round((ltp / prev - 1) * 100, 2), "from_open": round(fo, 2),
                   "disp": round(fo / (dly.get("range") or 2.0), 2),
                   "value_cr": round(ltp * vol / 1e7, 1), "ffmc": 0,
                   "book": round(tb / (tb + ts) * 100) if tb + ts > 0 else None}
            f = q.get(("NFO", u["fut_token"]))
            if f:
                fl, oi = _f(f.get("ltp")), _f(f.get("opnInterest"))
                fopen = _f(f.get("open"))
                if oi:
                    if u["fut_symbol"] not in self.oi_base:
                        self.oi_base[u["fut_symbol"]] = oi
                        new_base = True
                    base = self.oi_base[u["fut_symbol"]]
                    oi_chg = (oi / base - 1) * 100 if base else None
                    p_chg = (fl / fopen - 1) * 100 if fl and fopen else fo
                    row.update(oi=int(oi), oi_chg=round(oi_chg, 2) if oi_chg is not None else None,
                               fut=round(fl, 2) if fl else None, buildup=buildup(p_chg, oi_chg))
            series = self._series(cl)
            row["path"] = self._path(series, opn)
            b = self._bars5(series, opn)
            if b is not None:
                row["bars"] = [[round(float(x), 2) for x in rr] for rr in b[["open", "high", "low", "close"]].values]
            if dly:
                row["pdh"], row["pdl"] = round(dly["pdh"], 2), round(dly["pdl"], 2)
                row["liq"] = [k for k, hit in (("PDH", (hi or 0) > dly["pdh"]), ("PDL", 0 < (lo or 1e12) < dly["pdl"])) if hit]
            old = self.stocks.get(s, {})
            if structure or "read" not in old:
                rd = self._read(b)
                row["read"] = rd and {k: rd[k] for k in ("struct", "ob", "fvg", "ev")}
            else:
                row["read"] = old.get("read")
            p = row["path"]
            row["trail"] = next((v for mm, v in reversed(p) if mm <= m - 15), None) if p else None
            stocks[s] = row
        if new_base:
            self._save_oi_base()
        bull = sorted([r for r in stocks.values() if r["from_open"] > 0], key=lambda r: -r["disp"])
        bear = sorted([r for r in stocks.values() if r["from_open"] <= 0], key=lambda r: r["disp"])
        for i, r in enumerate(bull):
            r["rank"], r["side"] = i + 1, "bull"
        for i, r in enumerate(bear):
            r["rank"], r["side"] = i + 1, "bear"

        idx = {}
        for k, (exch, tok, _) in INDICES.items():
            r = q.get((exch, tok))
            if not r:
                continue
            ltp, opn, prev = _f(r.get("ltp")), _f(r.get("open")), _f(r.get("close"))
            if not ltp:
                continue
            cl = self.closes.setdefault("^" + k, {})
            if 0 <= m <= 375:
                cl[m] = ltp
            series = self._series(cl)
            base = opn or (float(series.iloc[0]) if len(series) else ltp)
            b = self._bars5(series, base)
            old = self.indices.get(k, {})
            rd = self._read(b, self.daily_idx.get(k)) if structure or not old.get("read") else old.get("read")
            idx[k] = {"ltp": round(ltp, 2), "pct": round((ltp / prev - 1) * 100, 2) if prev else None,
                      "from_open": round((ltp / base - 1) * 100, 2), "path": self._path(series, base),
                      "bars": [[round(float(x), 2) for x in rr] for rr in b[["open", "high", "low", "close"]].values] if b is not None else [],
                      "read": rd, "expiry": next_expiry(k), "delay": 0}
        with self.lock:
            self.stocks, self.indices = stocks, idx
            self.time = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── helpers (same maths as market-lab) ──────────────────────────────
    def _series(self, cl):
        if not cl:
            return pd.Series(dtype=float)
        d0 = pd.Timestamp(now_ist().date().isoformat() + " 03:45", tz="UTC")
        ks = sorted(cl)
        return pd.Series([cl[k] for k in ks], index=[d0 + pd.Timedelta(minutes=k) for k in ks])

    @staticmethod
    def _bars5(series, open_px=None):
        if series is None or len(series) == 0:
            return None
        df = series.resample("5min", origin=ORIGIN).agg(["first", "max", "min", "last"]).dropna()
        df.columns = ["open", "high", "low", "close"]
        if open_px and len(df):
            df.iloc[0, 0] = open_px
            df.iloc[0, 1] = max(df.iloc[0, 1], open_px)
            df.iloc[0, 2] = min(df.iloc[0, 2], open_px)
        df["volume"] = 0.0
        return df

    @staticmethod
    def _path(series, base):
        if series is None or not len(series) or not base:
            return []
        mins = ((series.index - series.index[0].normalize()).total_seconds() // 60 - 225).astype(int)
        return [[int(a), round(float(v), 3)] for a, v in zip(mins, (series.values / base - 1) * 100) if 0 <= a <= 375]

    @staticmethod
    def _read(df, daily=None):
        try:
            r = smc.read(df, 2, daily) if df is not None and len(df) >= 12 else None
        except Exception:
            return None
        if not r:
            return None
        st = r["structure"]["last"]
        d0 = df.index[0]
        return {"struct": st and {"type": st["type"], "dir": st["dir"], "time": st["time"]},
                "ob": (r["order_blocks"] or [None])[0], "fvg": (r["fvgs"] or [None])[0], "liq": r["liquidity"],
                "bias": r.get("bias"),
                "ev": [[int((e["time"] - int(d0.timestamp())) // 60), 1 if e["dir"] == "bull" else -1]
                       for e in r["chart"]["events"] if e["time"] >= int(d0.timestamp())]}

    # ── loop ────────────────────────────────────────────────────────────
    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True, name="SmartMoney").start()

    def _loop(self):
        while self.running:
            try:
                today = now_ist().date()
                if self.day != today and getattr(self.master, "loaded", False) and getattr(self.client, "api", None):
                    if self.build_universe():
                        self.day = today
                        self.closes, self.oi_base = {}, {}
                        self._load_oi_base()
                        self.cycle(True)
                        threading.Thread(target=self.load_daily, daemon=True, name="SMDaily").start()
                st = market_status()
                if self.day and (st == "open" or not self.stocks):
                    struct = time.time() - self.last_struct >= 60
                    self.cycle(struct)
                    if struct:
                        self.last_struct = time.time()
            except Exception as e:
                self._err(f"loop: {e}")
            time.sleep(POLL_SEC if market_status() == "open" else 120)

    # ── outputs ─────────────────────────────────────────────────────────
    def snapshot(self, paths=True):
        with self.lock:
            stocks = list(self.stocks.values())
            idx = dict(self.indices)
            t = self.time
        if not paths:
            stocks = [{k: v for k, v in r.items() if k != "path"} for r in stocks]
            idx = {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in idx.items()}
        return {"time": t, "market": market_status(), "errors": len(self.errors), "indices": idx, "stocks": stocks,
                "sectors": sector_table(stocks), "flow": self.flow(stocks),
                "bull": [r["sym"] for r in sorted(stocks, key=lambda r: r.get("rank", 999)) if r.get("side") == "bull"],
                "bear": [r["sym"] for r in sorted(stocks, key=lambda r: r.get("rank", 999)) if r.get("side") == "bear"]}

    def flow(self, stocks=None):
        """Where buying and selling is happening, from three independent reads."""
        if stocks is None:
            with self.lock:
                stocks = list(self.stocks.values())
        bu = {"LONG_BUILDUP": [], "SHORT_BUILDUP": [], "SHORT_COVERING": [], "LONG_UNWINDING": []}
        for r in stocks:
            if r.get("buildup"):
                bu[r["buildup"]].append(r)
        top = lambda lst: [{"sym": r["sym"], "pct": r["pct"], "oi_chg": r.get("oi_chg"), "sector": r["sector"]}
                           for r in sorted(lst, key=lambda r: -abs(r.get("oi_chg") or 0))[:8]]
        out = {k.lower(): {"n": len(v), "top": top(v)} for k, v in bu.items()}
        out["breadth"] = {"up": sum(r["from_open"] > 0 for r in stocks), "down": sum(r["from_open"] <= 0 for r in stocks)}
        for key, heavy in (("NIFTY", NIFTY_HEAVY), ("BANKNIFTY", BANK_HEAVY)):
            got = [(w, r) for s, w in heavy.items() for r in stocks if r["sym"] == s]
            wsum = sum(w for w, _ in got) or 1
            out[key.lower()] = {
                "heavy_from_open": round(sum(w * r["from_open"] for w, r in got) / wsum, 2),
                "heavy_up": sum(1 for _, r in got if r["from_open"] > 0), "heavy_n": len(got),
                "book_buy_pct": round(sum(w * (r.get("book") or 50) for w, r in got) / wsum, 1),
                "longs": sum(1 for _, r in got if r.get("buildup") in ("LONG_BUILDUP", "SHORT_COVERING")),
                "shorts": sum(1 for _, r in got if r.get("buildup") in ("SHORT_BUILDUP", "LONG_UNWINDING")),
            }
        return out

    def bias(self, instrument):
        """Signed read for the signal engine: +1 strong buyers ... -1 strong sellers.

        Combines (a) the index's own 5-minute structure, (b) weighted heavyweight move since
        open, (c) heavyweight futures build-up, (d) whole-market breadth. Returns None when the
        tape is stale or empty so callers fall back to their normal logic."""
        key = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY", "FINNIFTY": "FINNIFTY"}.get(instrument)
        with self.lock:
            stocks = list(self.stocks.values())
            ix = self.indices.get(key or "", {})
            t = self.time
        if not stocks or not t:
            return None
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(t)).total_seconds()
        if age > 180:
            return None
        fl = self.flow(stocks)
        h = fl.get("banknifty" if instrument == "BANKNIFTY" else "nifty", {})
        parts, why = [], []
        rd = (ix.get("read") or {})
        st = rd.get("struct")
        if st:
            v = 1 if st["dir"] == "bull" else -1
            parts.append(v * (1.0 if st["type"] == "BOS" else 0.6))
            why.append(f"{key} 5m {st['type']} {st['dir']}")
        if h.get("heavy_n"):
            v = max(-1, min(1, h["heavy_from_open"] / 0.6))
            parts.append(v)
            why.append(f"heavyweights {h['heavy_from_open']:+.2f}% from open ({h['heavy_up']}/{h['heavy_n']} up)")
            n = h["longs"] + h["shorts"]
            if n >= 3:
                parts.append((h["longs"] - h["shorts"]) / n)
                why.append(f"heavy futures {h['longs']} long-side vs {h['shorts']} short-side build-up")
        b = fl["breadth"]
        tot = b["up"] + b["down"]
        if tot >= 50:
            parts.append((b["up"] - b["down"]) / tot)
            why.append(f"breadth {b['up']} up / {b['down']} down")
        if not parts:
            return None
        score = round(float(np.mean(parts)), 2)
        return {"score": score, "why": why, "age_s": int(age)}


def sector_table(stocks):
    secs = {}
    for r in stocks:
        secs.setdefault(r["sector"], []).append(r)
    out = []
    for k, v in secs.items():
        bu = [r.get("buildup") for r in v]
        out.append({"name": k, "pct": round(float(np.mean([r["pct"] for r in v])), 2),
                    "from_open": round(float(np.mean([r["from_open"] for r in v])), 2),
                    "up": sum(r["pct"] > 0 for r in v), "down": sum(r["pct"] < 0 for r in v), "n": len(v),
                    "value_cr": round(sum(r.get("value_cr") or 0 for r in v), 1),
                    "longs": sum(x in ("LONG_BUILDUP", "SHORT_COVERING") for x in bu),
                    "shorts": sum(x in ("SHORT_BUILDUP", "LONG_UNWINDING") for x in bu)})
    return sorted(out, key=lambda s: -s["pct"])


def next_expiry(key):
    """NIFTY weekly Tuesday, BANKNIFTY/FINNIFTY last Tuesday of the month, SENSEX weekly Thursday.
    Holidays are not shifted here (display only)."""
    d = now_ist().date()
    for k in range(45):
        x = d + timedelta(days=k)
        if x.weekday() != (3 if key == "SENSEX" else 1):
            continue
        if key in ("BANKNIFTY", "FINNIFTY") and (x + timedelta(days=7)).month == x.month:
            continue
        return x.isoformat()
    return None
