"""
verify_data_layer.py — Prove the data layer works against REAL data before any
backtest trusts it.

What it does:
  1. Checks dependencies (jugaad-data, pandas, Angel One creds)
  2. Pulls real spot bars for last 7 days via Angel One
  3. Pulls EOD option chain for last weekly expiry via jugaad-data
  4. For 10 sample timestamps, retrieves the option premium and records which
     data source supplied it (cache / angel / jugaad / bs_interpolated)
  5. Writes data_layer_verify.csv with side-by-side results
  6. Prints a verdict — GO / NO-GO with reasoning

Run:
    python3 verify_data_layer.py

If GO: data layer is ready for backtest_v2.
If NO-GO: read the verdict reasoning and fix what's missing.
"""

from __future__ import annotations

import csv
import os
import sys
import traceback
from datetime import date, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def _ok(msg):   print(f"  ✓ {msg}")
def _warn(msg): print(f"  ⚠ {msg}")
def _err(msg):  print(f"  ✗ {msg}")


def check_deps() -> dict:
    print("=" * 64)
    print("1. Dependency check")
    print("=" * 64)
    deps = {}
    try:
        import pandas
        _ok(f"pandas {pandas.__version__}")
        deps["pandas"] = True
    except ImportError:
        _err("pandas missing — `pip install pandas`")
        deps["pandas"] = False

    try:
        import jugaad_data
        _ok(f"jugaad-data installed")
        deps["jugaad"] = True
    except ImportError:
        _warn("jugaad-data missing — `pip install jugaad-data` (needed for free NSE chain history)")
        deps["jugaad"] = False

    angel_keys = ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PASSWORD", "ANGEL_TOTP_SECRET")
    missing = [k for k in angel_keys if not os.environ.get(k)]
    if missing:
        _warn(f"Angel One env vars missing: {missing} — intraday option fallback won't work locally")
        deps["angel"] = False
    else:
        _ok("Angel One env vars present")
        deps["angel"] = True
    return deps


def test_spot(angel_client) -> bool:
    print()
    print("=" * 64)
    print("2. Spot bars (last 7 days, 5-min NIFTY)")
    print("=" * 64)
    try:
        from data_layer import get_spot_bars
        to_dt = datetime.now(IST).replace(tzinfo=None)
        from_dt = to_dt - timedelta(days=7)
        df = get_spot_bars("NIFTY", from_dt, to_dt, "5min", angel_client=angel_client)
        if df.empty:
            _err("No spot bars returned")
            return False
        _ok(f"Got {len(df)} bars · range {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")
        _ok(f"  First close: ₹{df['close'].iloc[0]:.2f}  Last close: ₹{df['close'].iloc[-1]:.2f}")
        _ok(f"  Sources: {df['source'].value_counts().to_dict()}")
        return len(df) > 10
    except Exception as e:
        _err(f"Spot fetch crashed: {e}")
        traceback.print_exc()
        return False


def test_eod_chain() -> tuple[bool, list]:
    print()
    print("=" * 64)
    print("3. EOD option chain (most recent Tuesday weekly expiry)")
    print("=" * 64)
    try:
        from data_layer import get_eod_option_chain, _candidate_expiries_for_date
        # Pick the most recent past Tuesday
        today = date.today()
        days_back = (today.weekday() - 1) % 7
        if days_back == 0: days_back = 7   # if today IS Tuesday, use last Tuesday
        sample_date = today - timedelta(days=days_back)
        _ok(f"Sample date (last Tuesday): {sample_date}")
        df = get_eod_option_chain("NIFTY", sample_date)
        if df.empty:
            _warn("EOD chain empty — jugaad-data may have failed or this date had no trade")
            return False, []
        _ok(f"Got {len(df)} option rows from EOD chain")
        _ok(f"  Sample row: {df.iloc[0].to_dict()}")
        return True, df.head(20).to_dict("records")
    except Exception as e:
        _err(f"EOD chain crashed: {e}")
        traceback.print_exc()
        return False, []


def test_premium_lookup(angel_client) -> list:
    print()
    print("=" * 64)
    print("4. Premium at specific timestamps (last 7 days × 10 samples)")
    print("=" * 64)
    try:
        from data_layer import get_option_premium_at, _atm_guess
        # Sample 10 timestamps spread across last 7 trading days
        now = datetime.now(IST).replace(tzinfo=None)
        results = []
        for i in range(10):
            sample_ts = now - timedelta(days=i + 1, hours=2)
            # Land on a market-hours timestamp (between 09:30 and 14:30)
            if sample_ts.weekday() in (5, 6):  # skip weekend
                continue
            sample_ts = sample_ts.replace(hour=10 + (i % 4), minute=15 + (i % 4) * 10, second=0)
            # Use atm + 0 strike for the upcoming Tuesday
            atm = _atm_guess("NIFTY", sample_ts.date())
            # Round to nearest 50
            strike = round(atm / 50) * 50
            # Next Tuesday after sample_ts
            days_ahead = (1 - sample_ts.weekday()) % 7
            if days_ahead == 0: days_ahead = 7
            expiry = sample_ts.date() + timedelta(days=days_ahead)
            result = get_option_premium_at("NIFTY", strike, "CE", expiry, sample_ts,
                                            angel_client=angel_client)
            line = {
                "ts": sample_ts.strftime("%Y-%m-%d %H:%M"),
                "symbol": "NIFTY",
                "strike": strike,
                "opt_type": "CE",
                "expiry": expiry.isoformat(),
                "price": result.get("price") if result else None,
                "source": result.get("source") if result else "NONE",
            }
            results.append(line)
            color = "  ✓" if line["price"] else "  ✗"
            print(f"{color} {line['ts']}  NIFTY{strike}CE exp {expiry}  → ₹{line['price']}  [{line['source']}]")
        return results
    except Exception as e:
        _err(f"Premium lookup crashed: {e}")
        traceback.print_exc()
        return []


def write_csv(rows: list, path: str):
    if not rows:
        _warn(f"No rows to write to {path}")
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    _ok(f"Wrote {len(rows)} rows → {path}")


def verdict(deps: dict, spot_ok: bool, chain_ok: bool, samples: list) -> bool:
    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    real = [s for s in samples if s.get("source") in ("angel", "cache:angel")]
    interp = [s for s in samples if s.get("source") == "bs_interpolated"]
    none = [s for s in samples if not s.get("price") or s.get("source") == "NONE"]
    print(f"  Spot fetch:    {'OK' if spot_ok else 'FAIL'}")
    print(f"  EOD chain:     {'OK' if chain_ok else 'FAIL (jugaad-data unavailable?)'}")
    print(f"  Premium samples: {len(samples)} total")
    print(f"    real-source (angel/cache): {len(real)}")
    print(f"    BS-interpolated (estimated): {len(interp)}")
    print(f"    no data: {len(none)}")
    print()

    # GO if at least 70% of samples came from real sources (angel or cached angel)
    if not samples:
        _err("No samples retrieved — fix dependencies first")
        return False
    real_pct = len(real) / len(samples) * 100
    if real_pct >= 70:
        print(f"  🟢 GO — {real_pct:.0f}% of premium samples came from real exchange data")
        print("     Backtest_v2 can be trusted to produce real, dated, priced trades.")
        return True
    elif real_pct >= 30:
        print(f"  🟡 PARTIAL — only {real_pct:.0f}% real. Backtest will work but some")
        print("     trades will be BS-interpolated. Mark those in the output.")
        return True
    else:
        print(f"  🔴 NO-GO — only {real_pct:.0f}% real. Cannot trust backtest output.")
        print("     Fix: install jugaad-data + set ANGEL_* env vars + retry.")
        return False


def main():
    print()
    print("Data Layer Verification — proves backtest can use REAL data")
    print()
    deps = check_deps()

    # Try to bring up an Angel client (using same env vars as server.py)
    angel_client = None
    if deps["angel"]:
        try:
            from server import AngelClient
            angel_client = AngelClient()
            if angel_client.login():
                _ok("Angel One login successful")
            else:
                _warn(f"Angel One login failed: {angel_client.last_login_error}")
                angel_client = None
        except Exception as e:
            _warn(f"AngelClient instantiation failed: {e}")

    spot_ok = test_spot(angel_client)
    chain_ok, _ = test_eod_chain()
    samples = test_premium_lookup(angel_client)

    write_csv(samples, "data_layer_verify.csv")

    ok = verdict(deps, spot_ok, chain_ok, samples)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
