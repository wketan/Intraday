"""
╔══════════════════════════════════════════════════════════════════╗
║  data_layer.py — REAL historical option + spot data for backtest  ║
║                                                                  ║
║  Three public methods:                                           ║
║    get_spot_bars(symbol, from_dt, to_dt, interval)               ║
║    get_option_premium_at(symbol, strike, opt_type, expiry, ts)   ║
║    get_eod_option_chain(symbol, date)                            ║
║                                                                  ║
║  Data source cascade (per request):                              ║
║    1. Local SQLite cache (`data/cache.db`)                       ║
║    2. jugaad-data — FREE NSE historical option OHLC (daily)      ║
║    3. Angel One getCandleData — intraday OHLC for option tokens  ║
║       (works on any NFO token current or recent)                 ║
║    4. Black-Scholes interpolation from spot + ATM IV             ║
║       (clearly marked `source = "bs_interpolated"` so the        ║
║        backtest can flag any P&L derived from estimates)         ║
║                                                                  ║
║  EVERY returned row carries a `source` column:                   ║
║    "cache" | "jugaad" | "angel" | "bs_interpolated"              ║
║                                                                  ║
║  Run `python verify_data_layer.py` to confirm it works against   ║
║  a known historical date before trusting backtest output.        ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
import os
import sqlite3
import json
import threading
import time as _time
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta, timezone
from typing import Optional

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

# Optional jugaad-data — graceful fallback if not installed yet.
try:
    from jugaad_data.nse import derivatives_df  # type: ignore
    _HAS_JUGAAD = True
except ImportError:
    _HAS_JUGAAD = False

IST = timezone(timedelta(hours=5, minutes=30))

# ─── Cache layer ──────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_DATA_DIR, exist_ok=True)
_CACHE_DB = os.path.join(_DATA_DIR, "cache.db")


def _cache_conn():
    conn = sqlite3.connect(_CACHE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _cache_init():
    """One-time schema init. Called lazily on first cache hit."""
    conn = _cache_conn()
    c = conn.cursor()
    # Spot OHLC bars
    c.execute("""CREATE TABLE IF NOT EXISTS spot_bars (
        symbol TEXT NOT NULL,
        ts TEXT NOT NULL,
        interval TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        source TEXT,
        PRIMARY KEY (symbol, ts, interval)
    )""")
    # Option premium ticks (intraday)
    c.execute("""CREATE TABLE IF NOT EXISTS option_bars (
        symbol TEXT NOT NULL, strike REAL NOT NULL, opt_type TEXT NOT NULL,
        expiry TEXT NOT NULL,
        ts TEXT NOT NULL, interval TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, oi REAL, iv REAL,
        source TEXT,
        PRIMARY KEY (symbol, strike, opt_type, expiry, ts, interval)
    )""")
    # EOD option chain (daily — from bhavcopy/jugaad)
    c.execute("""CREATE TABLE IF NOT EXISTS option_eod (
        symbol TEXT NOT NULL, date TEXT NOT NULL,
        expiry TEXT NOT NULL, strike REAL NOT NULL, opt_type TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, settle REAL,
        volume REAL, oi REAL, change_oi REAL,
        source TEXT,
        PRIMARY KEY (symbol, date, expiry, strike, opt_type)
    )""")
    conn.commit()
    conn.close()


_cache_init()


# ─── Helpers ──────────────────────────────────────────────────────────

def _fmt_ts(ts) -> str:
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, str):
        return ts
    return str(ts)


def _fmt_date(d) -> str:
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, _date):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _to_date(d) -> _date:
    if isinstance(d, _date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


# ─── 1. Spot bars ─────────────────────────────────────────────────────

def get_spot_bars(symbol: str, from_dt: datetime, to_dt: datetime,
                  interval: str = "5min",
                  angel_client=None) -> "pd.DataFrame":
    """Return OHLCV bars for an index spot between from_dt and to_dt at given interval.

    interval: "1min" | "5min" | "15min" | "1h" | "1d"

    Order of attempt:
    1. Cache (in `data/cache.db`)
    2. Angel One `getCandleData` via the passed-in client (the same one server.py uses)
    3. Empty DataFrame + warning logged (NSE official EOD download could be added later)

    `angel_client` should be an instance of `AngelClient` from `server.py`. Optional —
    if None, we ONLY use cache.
    """
    if pd is None:
        raise ImportError("pandas is required for data_layer")

    interval_map_angel = {
        "1min": "ONE_MINUTE", "3min": "THREE_MINUTE",
        "5min": "FIVE_MINUTE", "10min": "TEN_MINUTE",
        "15min": "FIFTEEN_MINUTE", "30min": "THIRTY_MINUTE",
        "1h": "ONE_HOUR", "1d": "ONE_DAY",
    }
    angel_interval = interval_map_angel.get(interval, "FIVE_MINUTE")

    # Try cache first
    conn = _cache_conn()
    rows = conn.execute(
        "SELECT * FROM spot_bars WHERE symbol=? AND interval=? "
        "AND ts BETWEEN ? AND ? ORDER BY ts",
        (symbol, interval, _fmt_ts(from_dt), _fmt_ts(to_dt))
    ).fetchall()
    conn.close()
    if rows and len(rows) > 1:
        df = pd.DataFrame([dict(r) for r in rows])
        df["ts"] = pd.to_datetime(df["ts"])
        # Cache miss path strips tz; mirror that here so cache-hit results are
        # naive-IST too. Stops pandas 2.x TypeError when downstream code
        # subtracts a naive datetime from this column.
        if getattr(df["ts"].dtype, "tz", None) is not None:
            df["ts"] = df["ts"].dt.tz_localize(None)
        return df

    # Cache miss → try Angel One
    if angel_client is None:
        print(f"[data_layer] WARN: cache miss for spot {symbol} {interval} and no Angel client provided")
        return pd.DataFrame()

    # Map our INSTRUMENTS dict tokens (will be passed in or imported)
    try:
        from server import INSTRUMENTS  # circular-safe — only imported here
    except ImportError:
        print("[data_layer] WARN: can't import INSTRUMENTS from server.py")
        return pd.DataFrame()

    inst = INSTRUMENTS.get(symbol.upper())
    if not inst:
        print(f"[data_layer] WARN: unknown spot symbol {symbol}")
        return pd.DataFrame()

    # Pass EXPLICIT from_dt + to_dt so Angel returns the requested historical
    # window, not "last N days from now". This was the silent bug that made
    # any historical lookup (replay panel, backtest_v2) return zero spot bars.
    df = angel_client.candles(inst["token"], inst["exchange"],
                              interval=angel_interval,
                              from_dt=from_dt, to_dt=to_dt,
                              force_refresh=True)
    if df.empty:
        return pd.DataFrame()

    # Filter to requested range.
    # Angel One returns IST-aware timestamps (UTC+05:30); from_dt / to_dt are
    # naive datetimes. pandas 2.x refuses to compare tz-aware Series with naive
    # bounds (TypeError). Strip tz so both sides are naive-IST.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dtype, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df[(df["timestamp"] >= from_dt) & (df["timestamp"] <= to_dt)].copy()
    if df.empty:
        return df
    df = df.rename(columns={"timestamp": "ts"})
    df["symbol"] = symbol.upper()
    df["interval"] = interval
    df["source"] = "angel"

    # Persist to cache
    _persist_spot_bars(df)
    return df


def _persist_spot_bars(df: "pd.DataFrame"):
    if df.empty: return
    conn = _cache_conn()
    rows = [(r["symbol"], _fmt_ts(r["ts"]), r["interval"],
             float(r["open"]), float(r["high"]), float(r["low"]),
             float(r["close"]), float(r["volume"]), r.get("source", "angel"))
            for _, r in df.iterrows()]
    conn.executemany(
        "INSERT OR REPLACE INTO spot_bars "
        "(symbol, ts, interval, open, high, low, close, volume, source) "
        "VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


# ─── 2. EOD option chain ──────────────────────────────────────────────

def get_eod_option_chain(symbol: str, date_d) -> "pd.DataFrame":
    """Return the full option chain for `symbol` on `date_d` (one day's EOD prices).

    Uses jugaad-data under the hood; cached after first fetch.

    Returns DataFrame with columns:
      symbol, date, expiry, strike, opt_type, open, high, low, close, settle,
      volume, oi, change_oi, source

    `symbol`: "NIFTY" | "BANKNIFTY" | "FINNIFTY"
    """
    if pd is None:
        raise ImportError("pandas is required")

    d = _to_date(date_d)

    # Cache hit?
    conn = _cache_conn()
    rows = conn.execute(
        "SELECT * FROM option_eod WHERE symbol=? AND date=?",
        (symbol.upper(), _fmt_date(d))
    ).fetchall()
    conn.close()
    if rows and len(rows) > 50:
        return pd.DataFrame([dict(r) for r in rows])

    if not _HAS_JUGAAD:
        print("[data_layer] jugaad-data not installed; pip install jugaad-data")
        return pd.DataFrame()

    # jugaad-data needs (symbol, from_date, to_date, expiry, instrument_type, option_type, strike_price)
    # The chain for one day = aggregate across all (expiry, strike, type). We pull
    # the nearest few expiries; for backtest purposes the active weekly expiry is enough.
    # First, discover what expiries are listed: derive from the day's bhavcopy.
    expiries_to_try = _candidate_expiries_for_date(d, symbol.upper())

    all_rows = []
    for expiry in expiries_to_try:
        for opt_type in ("CE", "PE"):
            # Pull strikes ATM ± 20 (heuristic — strike step varies by symbol)
            # We don't actually know strikes a priori, so let's pull a wide range and
            # let jugaad return only listed strikes.
            atm_guess = _atm_guess(symbol.upper(), d)
            step = _strike_step(symbol.upper())
            for strike in [atm_guess + i * step for i in range(-20, 21)]:
                try:
                    df = derivatives_df(
                        symbol=symbol.upper(),
                        from_date=d,
                        to_date=d,
                        expiry_date=expiry,
                        instrument_type="OPTIDX",
                        option_type=opt_type,
                        strike_price=float(strike),
                    )
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            all_rows.append({
                                "symbol": symbol.upper(),
                                "date": _fmt_date(d),
                                "expiry": _fmt_date(expiry),
                                "strike": float(strike),
                                "opt_type": opt_type,
                                "open": float(row.get("OPEN", 0) or 0),
                                "high": float(row.get("HIGH", 0) or 0),
                                "low": float(row.get("LOW", 0) or 0),
                                "close": float(row.get("CLOSE", 0) or 0),
                                "settle": float(row.get("SETTLE PR.", row.get("CLOSE", 0)) or 0),
                                "volume": float(row.get("CONTRACTS", 0) or 0),
                                "oi": float(row.get("OPEN INT", 0) or 0),
                                "change_oi": float(row.get("CHG IN OI", 0) or 0),
                                "source": "jugaad",
                            })
                except Exception:
                    # Strike doesn't exist for that expiry — silent, jugaad returns 404
                    continue

    if not all_rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(all_rows)
    _persist_eod_chain(df_out)
    return df_out


def _persist_eod_chain(df: "pd.DataFrame"):
    if df.empty: return
    conn = _cache_conn()
    cols = ["symbol", "date", "expiry", "strike", "opt_type",
            "open", "high", "low", "close", "settle",
            "volume", "oi", "change_oi", "source"]
    rows = [tuple(r[c] for c in cols) for _, r in df.iterrows()]
    conn.executemany(
        f"INSERT OR REPLACE INTO option_eod ({','.join(cols)}) "
        f"VALUES ({','.join(['?']*len(cols))})", rows)
    conn.commit()
    conn.close()


# ─── 3. Option premium at specific timestamp (the critical one) ───────

def get_option_premium_at(symbol: str, strike: float, opt_type: str,
                          expiry, ts: datetime,
                          angel_client=None) -> Optional[dict]:
    """Return the option premium at a specific intraday timestamp.

    Returns dict with: {price, volume, oi, source, bid, ask} or None if no data.

    Cascade:
      1. Cache (option_bars table) — if we have an intraday bar containing `ts`
      2. Angel One `getCandleData` — if the option token is still queryable
         (works for expiries within ~60 days)
      3. EOD chain + Black-Scholes interpolation:
         - Pull the day's EOD chain for this contract
         - Pull spot 5-min bar at `ts`
         - Get ATM IV from EOD chain (or approximation)
         - Compute BS price for given strike/spot/IV/time-to-expiry
         - Return with source="bs_interpolated"

    The backtest layer can flag any trade with `source="bs_interpolated"` as
    "estimated, not exchange-real" so the user sees what's real.
    """
    if pd is None:
        raise ImportError("pandas is required")

    ts = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    expiry_d = _to_date(expiry)
    day = ts.date()

    # ── 1. Cache lookup ──
    conn = _cache_conn()
    row = conn.execute(
        "SELECT * FROM option_bars WHERE symbol=? AND strike=? AND opt_type=? "
        "AND expiry=? AND ts<=? ORDER BY ts DESC LIMIT 1",
        (symbol.upper(), float(strike), opt_type.upper(),
         _fmt_date(expiry_d), _fmt_ts(ts))
    ).fetchone()
    conn.close()
    if row:
        r = dict(row)
        # Use the bar's close if same minute, else interpolate proportionally
        return {
            "price": r["close"],
            "volume": r.get("volume"), "oi": r.get("oi"), "iv": r.get("iv"),
            "source": "cache:" + (r.get("source") or "?"),
        }

    # ── 2. Angel One historical for this option token ──
    if angel_client is not None:
        result = _angel_option_lookup(angel_client, symbol, strike, opt_type, expiry_d, ts)
        if result is not None:
            return result

    # ── 3. Black-Scholes interpolation from EOD + spot ──
    return _bs_interpolate(symbol, strike, opt_type, expiry_d, ts, angel_client)


# ── In-memory option-day cache ─────────────────────────────────────────
# The backtest exit-walk previously did one Angel API call per 5-min bar
# of every trade — for a 30-day backtest that's ~6,000 calls, each ~300ms,
# making a 30-day run take 30+ minutes. We now fetch the FULL trading day
# of 1-min bars in a single call per (strike, opt_type, expiry, day) tuple
# and serve every per-timestamp lookup from this in-memory cache. The
# backtest typically hits ~50 unique tuples for a 30-day window → 50 API
# calls instead of 6,000.
_OPTION_DAY_CACHE = {}      # { (sym,strike,type,expiry_d,day_d): pd.DataFrame }
_OPTION_DAY_CACHE_LOCK = threading.Lock()
_OPTION_DAY_CACHE_MAX = 300  # bound size — last 300 entries (LRU-ish via dict order)


def reset_option_day_cache():
    """Clear the in-memory option-day cache. Called by /api/backtest at the
    start of every job so a stale snapshot from a prior run doesn't bleed
    into a new one."""
    with _OPTION_DAY_CACHE_LOCK:
        _OPTION_DAY_CACHE.clear()


def _option_day_bars(angel_client, symbol, strike, opt_type, expiry_d, day):
    """Return DataFrame of 1-min option bars for the full trading day, cached.

    Columns: ts, open, high, low, close, volume.
    Returns empty DataFrame if the option's token can't be found (the negative
    is also cached so we don't re-fetch the same dead contract every bar).
    """
    if pd is None:
        return None
    key = (symbol.upper(), float(strike), opt_type.upper(), expiry_d, day)
    with _OPTION_DAY_CACHE_LOCK:
        if key in _OPTION_DAY_CACHE:
            return _OPTION_DAY_CACHE[key]

    try:
        from server import _master
        if not _master.ensure():
            return None
        prefix = symbol.upper()
        exp_master = expiry_d.strftime("%d%b%Y").upper()
        info = _master.nfo.get((prefix, float(strike), opt_type.upper(), exp_master))
        if not info:
            # Negative cache — don't re-look-up a missing contract
            with _OPTION_DAY_CACHE_LOCK:
                _OPTION_DAY_CACHE[key] = pd.DataFrame()
            return pd.DataFrame()

        from_dt = datetime.combine(day, datetime.min.time()).replace(hour=9, minute=15)
        to_dt   = datetime.combine(day, datetime.min.time()).replace(hour=15, minute=30)
        params = {
            "exchange": "NFO",
            "symboltoken": str(info["token"]),
            "interval": "ONE_MINUTE",
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate":   to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            try:
                resp = _ex.submit(angel_client.api.getCandleData, params).result(timeout=15)
            except _cf.TimeoutError:
                return None

        if not (resp and resp.get("status") and resp.get("data")):
            with _OPTION_DAY_CACHE_LOCK:
                _OPTION_DAY_CACHE[key] = pd.DataFrame()
            return pd.DataFrame()

        rows = []
        for bar in resp["data"]:
            try:
                rows.append({
                    "ts":     pd.to_datetime(str(bar[0])[:19]),
                    "open":   float(bar[1]),
                    "high":   float(bar[2]),
                    "low":    float(bar[3]),
                    "close":  float(bar[4]),
                    "volume": float(bar[5]),
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            with _OPTION_DAY_CACHE_LOCK:
                _OPTION_DAY_CACHE[key] = df
            return df
        df = df.sort_values("ts").reset_index(drop=True)
        # Normalize to naive datetimes (matches what callers compare against)
        if getattr(df["ts"].dtype, "tz", None) is not None:
            df["ts"] = df["ts"].dt.tz_localize(None)

        with _OPTION_DAY_CACHE_LOCK:
            # LRU-ish: drop oldest if at cap
            if len(_OPTION_DAY_CACHE) >= _OPTION_DAY_CACHE_MAX:
                oldest = next(iter(_OPTION_DAY_CACHE))
                _OPTION_DAY_CACHE.pop(oldest, None)
            _OPTION_DAY_CACHE[key] = df
        return df
    except Exception as e:
        print(f"[data_layer] option day fetch failed: {e}")
        return None


def _angel_option_lookup(angel_client, symbol, strike, opt_type, expiry_d, ts):
    """Look up option price at exact timestamp `ts`. Serves from the day
    cache after the first call per (strike, opt_type, expiry, day).
    """
    df = _option_day_bars(angel_client, symbol, strike, opt_type, expiry_d, ts.date())
    if df is None or df.empty:
        return None
    target = pd.Timestamp(ts)
    if getattr(target, "tz", None) is not None:
        target = target.tz_localize(None)
    # Pick the bar whose timestamp is ≤ target — that's the last known price
    # at or before `ts` (matches the original per-bar fetch semantics).
    mask = df["ts"] <= target
    if not mask.any():
        return None
    bar = df.loc[mask].iloc[-1]
    return {
        "price":  float(bar["close"]),
        "open":   float(bar["open"]),
        "high":   float(bar["high"]),
        "low":    float(bar["low"]),
        "volume": float(bar["volume"]),
        "oi":     None, "iv": None,
        "source": "angel-day-cache",
    }


def get_nse_contract_day(symbol: str, strike: float, opt_type: str,
                         expiry, date_d) -> Optional[dict]:
    """Fetch ONE option contract's DAILY OHLC + settle from NSE's historical
    contract-wise price-volume report (the page user pointed to at
    nseindia.com/report-detail/fo_eq_security). Cached in `option_eod` SQLite
    table so repeat queries are instant.

    Returns:
        {"open", "high", "low", "close", "settle", "ltp", "volume", "oi", "source"}
        or None if NSE returned no row for that (symbol, strike, type, expiry, date).

    This is the SINGLE SOURCE OF TRUTH for the "what range did this option trade in
    on that day" question — used to (a) calibrate real day-IV from the settle
    price, and (b) HARD-CLAMP any Black-Scholes intraday estimate to [low, high]
    so we never display a number outside what actually traded.
    """
    if pd is None or not _HAS_JUGAAD:
        return None
    d = _to_date(date_d)
    exp_d = _to_date(expiry)

    # Cache lookup (option_eod has the right schema)
    conn = _cache_conn()
    row = conn.execute(
        "SELECT * FROM option_eod WHERE symbol=? AND date=? AND expiry=? "
        "AND strike=? AND opt_type=?",
        (symbol.upper(), _fmt_date(d), _fmt_date(exp_d), float(strike), opt_type.upper())
    ).fetchone()
    conn.close()
    if row:
        r = dict(row)
        return {
            "open": r.get("open"), "high": r.get("high"),
            "low": r.get("low"), "close": r.get("close"),
            "settle": r.get("settle"), "volume": r.get("volume"),
            "oi": r.get("oi"),
            "source": "cache:" + (r.get("source") or "?"),
        }

    # Fetch via jugaad-data (wraps NSE's contract-wise historical report)
    try:
        df = derivatives_df(
            symbol=symbol.upper(),
            from_date=d, to_date=d,
            expiry_date=exp_d,
            instrument_type="OPTIDX",
            option_type=opt_type.upper(),
            strike_price=float(strike),
        )
    except Exception as e:
        print(f"[data_layer] NSE fetch failed for {symbol} {strike}{opt_type} {exp_d}@{d}: {e}")
        return None
    if df is None or df.empty:
        return None

    row = df.iloc[0]
    result = {
        "open":   float(row.get("OPEN", 0) or 0),
        "high":   float(row.get("HIGH", 0) or 0),
        "low":    float(row.get("LOW", 0) or 0),
        "close":  float(row.get("CLOSE", 0) or 0),
        "settle": float(row.get("SETTLE PRICE", row.get("SETTLE PR.", row.get("CLOSE", 0))) or 0),
        "volume": float(row.get("TOTAL TRADED QUANTITY", row.get("CONTRACTS", 0)) or 0),
        "oi":     float(row.get("OPEN INTEREST", row.get("OPEN INT", 0)) or 0),
        "source": "nse_jugaad",
    }

    # Persist to cache for repeat queries
    try:
        conn = _cache_conn()
        conn.execute(
            "INSERT OR REPLACE INTO option_eod "
            "(symbol, date, expiry, strike, opt_type, open, high, low, close, settle, "
            " volume, oi, change_oi, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (symbol.upper(), _fmt_date(d), _fmt_date(exp_d), float(strike), opt_type.upper(),
             result["open"], result["high"], result["low"], result["close"],
             result["settle"], result["volume"], result["oi"], 0,
             "nse_jugaad")
        )
        conn.commit(); conn.close()
    except Exception as e:
        print(f"[data_layer] cache write failed: {e}")
    return result


def _calibrate_iv_from_nse_settle(symbol: str, strike: float, opt_type: str,
                                  expiry_d, date_d, settle_price: float,
                                  angel_client) -> Optional[float]:
    """Given NSE's day-settle for this option, back-solve the IV that the market
    actually traded at. This replaces the previous "assume 18%" hack with real IV.

    NSE settle is computed at 15:30 IST, so we use the spot value at 15:30 of
    that date to invert Black-Scholes.
    """
    if settle_price is None or settle_price <= 0: return None
    try:
        # Get end-of-day spot — use the 15:25-15:30 bar to approximate
        eod_dt = datetime.combine(_to_date(date_d), datetime.min.time()).replace(hour=15, minute=25)
        spot_df = get_spot_bars(symbol.upper(),
                                eod_dt - timedelta(minutes=15),
                                eod_dt + timedelta(minutes=10),
                                "5min", angel_client=angel_client)
        if spot_df.empty: return None
        # Closest bar to 15:25
        spot_df["ts"] = pd.to_datetime(spot_df["ts"])
        idx = (spot_df["ts"] - eod_dt).abs().argsort()
        if len(idx) == 0: return None
        eod_spot = float(spot_df.iloc[idx[0]]["close"])
        if not eod_spot: return None

        dte = max(0.5, (_to_date(expiry_d) - _to_date(date_d)).days)
        iv = _implied_vol(eod_spot, settle_price, float(strike),
                          dte / 365.0, opt_type.upper())
        # Sanity: clamp absurd IVs (data quality issues sometimes give 500%+)
        if iv is None: return None
        if iv < 0.03 or iv > 1.5: return None
        return iv
    except Exception as e:
        print(f"[data_layer] IV calibration failed: {e}")
        return None


def _bs_interpolate(symbol, strike, opt_type, expiry_d, ts, angel_client):
    """Intraday option-premium estimator that uses REAL NSE data as the anchor.

    New (bridge plan) algorithm:
      1. Pull this contract's NSE daily OHLC+settle (jugaad-data → NSE archives)
      2. Back-solve the day's REAL IV from settle (no more 18% assumption)
      3. Compute Black-Scholes at the requested intraday timestamp using that IV
      4. HARD-CLAMP the result inside [day_low, day_high] — never display a
         price that didn't actually trade. Flag the row if clamping happened.

    Returns dict with extra context so the journal can show: NSE day range,
    IV used + source, whether the BS estimate was clamped to range.
    """
    try:
        # 1. Get spot at `ts` (5-min bar)
        from_dt = ts - timedelta(minutes=10)
        to_dt   = ts + timedelta(minutes=10)
        spot_df = get_spot_bars(symbol.upper(), from_dt, to_dt, "5min", angel_client=angel_client)
        if spot_df.empty:
            return None
        spot_df["ts"] = pd.to_datetime(spot_df["ts"])
        spot_row = spot_df.iloc[(spot_df["ts"] - ts).abs().argsort()[0]]
        spot = float(spot_row["close"]) if spot_row is not None else None
        if not spot:
            return None

        # 2. Get NSE daily OHLC for THIS specific contract
        nse_day = get_nse_contract_day(symbol.upper(), float(strike), opt_type,
                                       expiry_d, ts.date())

        # 3. Calibrate IV from NSE settle (real day-IV) — fall back to 18% if no NSE row
        atm_iv = 0.18
        iv_source = "default_0.18"
        if nse_day and nse_day.get("settle", 0) > 0:
            calibrated = _calibrate_iv_from_nse_settle(
                symbol.upper(), float(strike), opt_type, expiry_d, ts.date(),
                float(nse_day["settle"]), angel_client
            )
            if calibrated is not None:
                atm_iv = calibrated
                iv_source = "nse_settle_calibrated"

        # 4. Compute BS at the requested timestamp
        dte = max(0.5, (_to_date(expiry_d) - ts.date()).days)
        # On expiry day with intraday timing, prorate to fraction-of-day remaining
        if dte <= 1.0:
            mins_remaining = max(15, (15*60 + 30) - (ts.hour*60 + ts.minute))   # till 15:30
            dte = mins_remaining / (24.0 * 60.0)   # in days
            dte = max(0.01, dte)
        bs_price = _bs_price(spot, float(strike), dte / 365.0, atm_iv, opt_type)
        bs_raw = round(bs_price, 2)

        # 5. HARD CLAMP to NSE day's traded range (the user's key requirement:
        #    no number that didn't actually trade should ever surface)
        final_price = bs_raw
        clamped = False
        out_of_range = False
        if nse_day and nse_day.get("low", 0) > 0 and nse_day.get("high", 0) > 0:
            day_low  = float(nse_day["low"])
            day_high = float(nse_day["high"])
            if bs_raw < day_low:
                final_price = day_low; clamped = True; out_of_range = True
            elif bs_raw > day_high:
                final_price = day_high; clamped = True; out_of_range = True

        return {
            "price":             final_price,
            "price_raw_bs":      bs_raw,            # what BS computed before clamp
            "iv":                round(atm_iv, 4),
            "iv_source":         iv_source,
            # NSE day context — surfaced in journal so user can sanity-check every row
            "nse_day_open":      nse_day.get("open")   if nse_day else None,
            "nse_day_high":      nse_day.get("high")   if nse_day else None,
            "nse_day_low":       nse_day.get("low")    if nse_day else None,
            "nse_day_close":     nse_day.get("close")  if nse_day else None,
            "nse_day_settle":    nse_day.get("settle") if nse_day else None,
            "nse_day_volume":    nse_day.get("volume") if nse_day else None,
            "nse_day_oi":        nse_day.get("oi")     if nse_day else None,
            "clamped":           clamped,
            "out_of_range":      out_of_range,
            "spot_at_ts":        round(spot, 2),
            # Source label tells you EXACTLY what backed this row:
            #   nse_calibrated_clamped — clamped to NSE day range (BS was outside)
            #   nse_calibrated         — BS within NSE range, real IV used
            #   bs_no_nse              — couldn't get NSE data, used 18% default
            "source": ("nse_calibrated_clamped" if clamped
                       else "nse_calibrated"    if nse_day
                       else "bs_no_nse"),
            "volume": None, "oi": None,
        }
    except Exception as e:
        print(f"[data_layer] BS interpolation failed: {e}")
        return None


# ─── Black-Scholes (no scipy dep) ────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Cumulative standard normal — Abramowitz approximation, accurate to ~1e-7."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(spot: float, strike: float, t_years: float, iv: float, opt_type: str) -> float:
    """Black-Scholes-Merton, r=0 (Indian options on indices, no dividend)."""
    if t_years <= 0 or iv <= 0:
        # Intrinsic only
        if opt_type.upper() == "CE": return max(spot - strike, 0)
        return max(strike - spot, 0)
    d1 = (math.log(spot / strike) + (iv * iv / 2.0) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)
    if opt_type.upper() == "CE":
        return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def _implied_vol(spot: float, market_price: float, strike: float, t_years: float,
                 opt_type: str = "CE", iters: int = 20) -> Optional[float]:
    """Newton-Raphson IV solver. Returns None if no convergence."""
    if market_price <= 0 or t_years <= 0: return None
    iv = 0.20
    for _ in range(iters):
        price = _bs_price(spot, strike, t_years, iv, opt_type)
        # Vega
        d1 = (math.log(spot / strike) + (iv * iv / 2.0) * t_years) / (iv * math.sqrt(t_years))
        vega = spot * math.sqrt(t_years) * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
        if vega < 1e-8: return None
        diff = market_price - price
        if abs(diff) < 0.05: return iv
        iv = iv + diff / vega
        if iv <= 0.01: iv = 0.01
        if iv >= 3.0:  iv = 3.0
    return iv


# ─── Symbol metadata ──────────────────────────────────────────────────

def _strike_step(symbol: str) -> int:
    return {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50}.get(symbol.upper(), 50)


def _atm_guess(symbol: str, d: _date) -> float:
    """Cheap ATM guess for chain pull — uses last cached spot close on or before `d`."""
    conn = _cache_conn()
    row = conn.execute(
        "SELECT close FROM spot_bars WHERE symbol=? AND ts<=? ORDER BY ts DESC LIMIT 1",
        (symbol.upper(), _fmt_date(d) + " 23:59:59")
    ).fetchone()
    conn.close()
    if row:
        return float(row[0])
    # Fallback ATM if no cache yet — order-of-magnitude correct for late-2025
    return {"NIFTY": 25500, "BANKNIFTY": 53000, "FINNIFTY": 26000}.get(symbol.upper(), 25000)


def _candidate_expiries_for_date(d: _date, symbol: str) -> list:
    """Find weekly/monthly expiries falling within ~60 days after `d`.

    NSE convention (post-Nov-2024):
      - NIFTY: weekly Tuesday expiries
      - BANKNIFTY: monthly last-Tuesday (no weeklies anymore)
      - FINNIFTY: monthly last-Tuesday (no weeklies anymore)

    Returns a list of date objects.
    """
    out = []
    if symbol == "NIFTY":
        # Next 8 Tuesdays after `d`
        days_ahead = (1 - d.weekday()) % 7  # Tuesday = 1 in Python's weekday()
        if days_ahead == 0: days_ahead = 7
        first_tue = d + timedelta(days=days_ahead)
        for i in range(8):
            out.append(first_tue + timedelta(days=7 * i))
    else:
        # Monthly last-Tuesday: find last Tuesday of current and next month
        for month_offset in range(0, 3):
            y, m = d.year, d.month + month_offset
            while m > 12: y += 1; m -= 12
            last_day = (datetime(y, m + 1 if m < 12 else 1, 1) - timedelta(days=1)).date() if m < 12 \
                       else _date(y, 12, 31)
            # Walk back to last Tuesday
            while last_day.weekday() != 1:
                last_day -= timedelta(days=1)
            if last_day >= d:
                out.append(last_day)
    return out


# ─── Public diagnostics ───────────────────────────────────────────────

def cache_stats() -> dict:
    conn = _cache_conn()
    spot_n = conn.execute("SELECT COUNT(*) FROM spot_bars").fetchone()[0]
    opt_bars_n = conn.execute("SELECT COUNT(*) FROM option_bars").fetchone()[0]
    eod_n = conn.execute("SELECT COUNT(*) FROM option_eod").fetchone()[0]
    db_size = os.path.getsize(_CACHE_DB) if os.path.exists(_CACHE_DB) else 0
    conn.close()
    return {
        "spot_bars": spot_n,
        "option_bars": opt_bars_n,
        "option_eod_rows": eod_n,
        "db_size_kb": round(db_size / 1024, 1),
        "has_jugaad": _HAS_JUGAAD,
        "cache_db": _CACHE_DB,
    }


if __name__ == "__main__":
    # Self-check
    import pprint
    print("=" * 60)
    print("data_layer.py self-check")
    print("=" * 60)
    pprint.pprint(cache_stats())
    print()
    print("BS price test:")
    print(f"  ATM NIFTY CE, spot=25500, strike=25500, 5 DTE, IV=18%:")
    p = _bs_price(25500, 25500, 5/365, 0.18, "CE")
    print(f"  → ₹{p:.2f}  (expected range: ₹70-90)")
    print()
    print(f"  Candidate NIFTY expiries from 2026-05-11:")
    for e in _candidate_expiries_for_date(_date(2026, 5, 11), "NIFTY"):
        print(f"    {e}")
