"""
╔══════════════════════════════════════════════════════════════════╗
║  regime.py — Pre-trade regime gate for v2                         ║
║                                                                  ║
║  The 3-of-4 confluence rule fires signals; this filter decides    ║
║  whether the CURRENT MOMENT is favorable for trading them.        ║
║                                                                  ║
║  Returns (should_trade: bool, reason: str)                        ║
║                                                                  ║
║  Filter chain (any failure blocks the trade):                     ║
║    1. MONDAY_BLOCK         — weekend gap risk                     ║
║    2. AFTER_1230_NO_ENTRY  — no new entries late in session       ║
║    3. EXPIRY_AFTERNOON     — Tue 11:00+ on NIFTY weekly = chaos   ║
║    4. HIGH_VIX             — India VIX > 22 = event/extreme       ║
║    5. EVENT_CALENDAR       — defers to existing EventCalendar     ║
║                                                                  ║
║  All thresholds are env-tunable so the user can dial without      ║
║  redeploying.                                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))


class RegimeFilter:
    """Stateless filter. Pass an `angel_client` (the same one server.py uses) for
    the VIX fetch. If None, VIX check is skipped (assumes OK)."""

    # India VIX is an NSE index. Token confirmed from Angel One's instrument master.
    VIX_TOKEN = "26017"
    VIX_SYMBOL = "India VIX"

    @staticmethod
    def should_trade(angel_client=None, symbol: str = "NIFTY",
                     now: datetime = None) -> tuple[bool, str]:
        """Returns (True, "OK") if all filters pass, else (False, "<REASON>").

        Args:
            angel_client: server.py's AngelClient instance; optional. If None, VIX skipped.
            symbol: instrument symbol ("NIFTY", "BANKNIFTY", "FINNIFTY").
            now: override current time for testing. Defaults to datetime.now(IST).
        """
        if now is None:
            now = datetime.now(IST)

        # ── Filter 1: Skip Monday (weekend gap risk) ─────────────────
        if os.environ.get("V2_BLOCK_MONDAY", "true").lower() == "true":
            if now.weekday() == 0:   # 0 = Monday
                return False, "MONDAY_BLOCK"

        # ── Filter 2: No new entries after 12:30 IST ──────────────────
        # Reverted from 14:50 → 12:30 after backtest showed the wider window
        # made v2 strictly worse (-₹17k → -₹92k). The afternoon-trend
        # hypothesis was right, but v2 doesn't have edge to capture it —
        # late-day trades just hit EOD theta exits at deeper losses. Keeping
        # the 12:30 cutoff on v2 (which is parked anyway); ORB and gamma
        # blast will get their own time gates that aren't tied to this.
        cutoff_h = int(os.environ.get("V2_NO_ENTRY_HOUR", "12"))
        cutoff_m = int(os.environ.get("V2_NO_ENTRY_MINUTE", "30"))
        if now.hour > cutoff_h or (now.hour == cutoff_h and now.minute >= cutoff_m):
            return False, f"AFTER_{cutoff_h:02d}{cutoff_m:02d}_NO_NEW_ENTRY"

        # ── Filter 3: Expiry-day afternoon block ─────────────────────
        # NIFTY weekly: Tuesday expiry (post Nov-2024 SEBI change).
        # BANKNIFTY/FINNIFTY: monthly Tuesday only.
        # On expiry day, after 11:00 IST: gamma + theta = unpredictable.
        if RegimeFilter._is_expiry_day(now, symbol):
            block_h = int(os.environ.get("V2_EXPIRY_BLOCK_HOUR", "11"))
            if now.hour >= block_h:
                return False, "EXPIRY_AFTERNOON_BLOCK"

        # ── Filter 4: India VIX > 22 (event/extreme volatility) ──────
        if angel_client is not None:
            vix = RegimeFilter._fetch_vix(angel_client)
            if vix is not None:
                max_vix = float(os.environ.get("V2_MAX_VIX", "22"))
                if vix > max_vix:
                    return False, f"HIGH_VIX_{vix:.1f}"

        # ── Filter 5: Event calendar — defer to EventCalendar ───────
        try:
            from server import EventCalendar
            blackout, event = EventCalendar.in_blackout()
            if blackout:
                return False, f"EVENT_BLACKOUT_{(event.get('name','?') or '?')[:30]}"
        except ImportError:
            pass   # EventCalendar not available — skip

        return True, "OK"

    @staticmethod
    def _is_expiry_day(now: datetime, symbol: str) -> bool:
        """Tuesday = weekday() 1.

        NIFTY: every Tuesday is a weekly expiry day.
        BANKNIFTY/FINNIFTY: only the LAST Tuesday of the month is expiry.
        """
        if now.weekday() != 1:
            return False
        if symbol.upper() == "NIFTY":
            return True
        # For monthly-only instruments, check it's the LAST Tuesday of the month
        next_week = now + timedelta(days=7)
        return next_week.month != now.month

    @staticmethod
    def _fetch_vix(angel_client) -> float | None:
        """Fetch India VIX LTP. Returns None on any failure (filter falls open).

        Cached for 5 min — VIX moves slowly intraday.
        """
        cache = getattr(RegimeFilter, "_vix_cache", None)
        now_ts = datetime.now(IST).timestamp()
        if cache and (now_ts - cache["ts"]) < 300:
            return cache["value"]
        try:
            data = angel_client.ltp("NSE", RegimeFilter.VIX_SYMBOL, RegimeFilter.VIX_TOKEN)
            if data and "ltp" in data:
                vix = float(data["ltp"])
                RegimeFilter._vix_cache = {"ts": now_ts, "value": vix}
                return vix
        except Exception:
            pass
        return None


# ─── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test against a few synthetic times — no angel_client so VIX skipped.

    test_cases = [
        ("Tuesday 10:00",    datetime(2026, 5, 12, 10,  0, tzinfo=IST), "NIFTY"),    # expiry day morning OK
        ("Tuesday 11:30",    datetime(2026, 5, 12, 11, 30, tzinfo=IST), "NIFTY"),    # expiry day afternoon — BLOCKED
        ("Monday 10:00",     datetime(2026, 5, 11, 10,  0, tzinfo=IST), "NIFTY"),    # BLOCKED — Monday
        ("Wednesday 09:30",  datetime(2026, 5, 13, 9,  30, tzinfo=IST), "NIFTY"),    # OK
        ("Wednesday 13:00",  datetime(2026, 5, 13, 13,  0, tzinfo=IST), "NIFTY"),    # BLOCKED — too late
        ("Thursday 10:00",   datetime(2026, 5, 14, 10,  0, tzinfo=IST), "BANKNIFTY"),# OK
    ]

    print("Regime filter test:")
    for label, ts, sym in test_cases:
        ok, reason = RegimeFilter.should_trade(angel_client=None, symbol=sym, now=ts)
        marker = "🟢" if ok else "🔴"
        print(f"  {marker} {label:<22} {sym:<10} → {reason}")
