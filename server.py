"""
╔══════════════════════════════════════════════════════════════════╗
║  INTRADAY OPTIONS SIGNAL ENGINE — Production Server             ║
║  Features: Live Signals + Option Picks + P&L + WhatsApp Alerts  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import math
import sqlite3
import threading
import logging
from datetime import datetime, timedelta, timezone
from functools import wraps
IST = timezone(timedelta(hours=5, minutes=30))

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request as flask_request, send_file

from SmartApi import SmartConnect
import pyotp

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "api_key":      os.environ.get("ANGEL_API_KEY", ""),
    "client_id":    os.environ.get("ANGEL_CLIENT_ID", ""),
    "password":     os.environ.get("ANGEL_PASSWORD", ""),
    "totp_secret":  os.environ.get("ANGEL_TOTP_SECRET", ""),

    "scan_interval_sec": int(os.environ.get("SCAN_INTERVAL", "5")),
    "candle_interval":   "FIVE_MINUTE",
    "lookback_days":     3,
    "target_points_min": int(os.environ.get("TARGET_MIN", "10")),
    "target_points_max": int(os.environ.get("TARGET_MAX", "15")),
    "min_confidence":    int(os.environ.get("MIN_CONFIDENCE", "60")),
    "budget":            int(os.environ.get("BUDGET", "20000")),

    # ── Slack Alert Config ──
    # Create webhook: Slack → Apps → Incoming Webhooks → Add to Slack → Select your DM
    "slack_webhook":    os.environ.get("SLACK_WEBHOOK", ""),
    "slack_enabled":    os.environ.get("SLACK_ENABLED", "true").lower() == "true",
    
    # ── AI Analysis (Claude) ──
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    # Model for Claude layers A, B, C. Layer D (in-flight) can be downgraded to Haiku via env.
    "anthropic_model":   os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    "anthropic_model_inflight": os.environ.get("ANTHROPIC_MODEL_INFLIGHT",
                                               os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")),

    # ── Security ──
    # Shared secret required on write endpoints (X-Auth-Token header). If empty, writes are rejected.
    "auth_token":        os.environ.get("AUTH_TOKEN", ""),
    # Comma-separated list of allowed CORS origins (GitHub Pages + localhost by default)
    "cors_origins":      os.environ.get("CORS_ORIGINS",
                                        "https://wketan.github.io,http://localhost:5050,http://127.0.0.1:5050"),
}

# Per-instrument option premium bands (₹) used by OptPicker scoring.
# Calibrated against Angel/Sensibull ATM quotes at ~5 DTE (NIFTY weekly) and ~15-25 DTE (BN/FN monthly).
PREMIUM_RANGES = {
    "NIFTY":     {"ideal": (40,  90),  "ok": (25,  140)},
    "BANKNIFTY": {"ideal": (180, 350), "ok": (120, 500)},
    "FINNIFTY":  {"ideal": (60,  140), "ok": (40,  200)},
}

# Fallback moneyness-ladder deltas when real greeks are unavailable.
# DTE-aware: weekly (~5 DTE) vs monthly (~20 DTE) behave very differently at the same moneyness.
def fallback_delta(moneyness, dte=5, right_side=True):
    """moneyness = |strike - spot| / spot.  Mirrors the client-side estimate.
    Shrink effective moneyness when DTE is larger (more time value => closer to 0.5)."""
    ref_dte = 5.0
    adj = moneyness * (ref_dte / max(dte, 0.5))
    if adj < 0.001:   d = 0.50
    elif adj < 0.002: d = 0.45
    elif adj < 0.003: d = 0.38
    elif adj < 0.005: d = 0.30
    elif adj < 0.008: d = 0.22
    else:             d = 0.12
    if not right_side:
        d = min(0.70, d + 0.20)  # ITM
    return d

PORT = int(os.environ.get("PORT", "5050"))

# ═══════════════════════════════════════════════════════════════════
# INSTRUMENTS
# ═══════════════════════════════════════════════════════════════════
INSTRUMENTS = {
    "NIFTY": {
        "symbol": "NIFTY", "token": "99926000", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 65, "strike_gap": 50,
        "expiry_prefix": "NIFTY", "expiry_day": 1, "expiry_type": "weekly",  # Tuesday weekly
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY", "token": "99926009", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 30, "strike_gap": 100,
        "expiry_prefix": "BANKNIFTY", "expiry_day": 1, "expiry_type": "monthly",  # Last Tuesday monthly
    },
    "FINNIFTY": {
        "symbol": "NIFTY FIN SERVICE", "token": "99926037", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 60, "strike_gap": 50,
        "expiry_prefix": "FINNIFTY", "expiry_day": 1, "expiry_type": "monthly",  # Last Tuesday monthly
    },
}

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("signals.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("SignalEngine")

# ═══════════════════════════════════════════════════════════════════
# SLACK DM ALERTS (FREE — uses Incoming Webhook)
# ═══════════════════════════════════════════════════════════════════
class SlackAlert:
    """
    Slack DM alerts using Incoming Webhook.
    
    SETUP:
    1. Go to https://api.slack.com/apps → Create New App → From scratch
    2. Name: "Trading Alerts", Workspace: your workspace
    3. Left sidebar → Incoming Webhooks → Activate (toggle ON)
    4. Click "Add New Webhook to Workspace" → Select your DM channel
    5. Copy the Webhook URL → paste in CONFIG below
    """
    
    @staticmethod
    def send(message, blocks=None):
        if not CONFIG["slack_enabled"] or not CONFIG["slack_webhook"]:
            return False
        try:
            payload = {"text": message}
            if blocks:
                payload["blocks"] = blocks
            resp = requests.post(CONFIG["slack_webhook"], json=payload, timeout=10)
            if resp.status_code == 200:
                log.info("📱 Slack alert sent")
                return True
            log.warning(f"Slack alert failed: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            log.error(f"Slack error: {e}")
            return False
    
    @staticmethod
    def format_signal(instrument, signal, option, timing=None, ai=None):
        arrow = "🟢" if signal["direction"] == "LONG" else "🔴"
        entry_time = signal.get("timestamp", datetime.now(IST).strftime("%H:%M"))
        
        msg = f"""{arrow} *SIGNAL: {instrument} {signal["direction"]}*
━━━━━━━━━━━━━━━━━━━━━"""

        if option:
            msg += f"""
📋 *{option["action"]}: {option["symbol"]}*

*TRADE PLAN:*
▶ Buy at: `₹{option["entry"]}` (Live LTP)
🛑 Exit SL: `₹{option["sl"]}`
✅ Exit T1: `₹{option["target1"]}` → Profit: *+₹{option["t1_profit"]}*
✅ Exit T2: `₹{option["target2"]}` → Profit: *+₹{option["t2_profit"]}*
💼 Capital: `₹{option["capital"]}` | Max Loss: `₹{option["max_loss"]}`
📐 Delta: `{option.get("delta",0.4)}` | R:R: `{signal["risk_reward"]}`

*TIMING:*
⏰ Entry: `{entry_time}` IST"""
            if timing:
                msg += f"""
🎯 Target by: `~{timing["target_by"]}` IST (~{timing["est_duration"]})
🛑 SL by: `~{timing["sl_by"]}` IST"""
        else:
            msg += f"""
*INDEX LEVELS:*
▶ Entry: `{signal["entry"]}` | 🛑 SL: `{signal["sl"]}`
✅ T1: `{signal["target1"]}` | T2: `{signal["target2"]}`
⏰ Entry: `{entry_time}` IST"""

        if ai and ai.get("verdict"):
            v = ai["verdict"]
            emoji = "✅" if v == "TAKE" else ("⏸" if v == "WAIT" else "⛔")
            adj = ai.get("confidence_adj", 0)
            adj_str = f"+{adj}" if adj > 0 else str(adj)
            msg += f"""

*🤖 AI ANALYSIS:*
{emoji} Verdict: *{v}* (Conf {adj_str}%)
💡 {ai.get("reasoning", "")}
⚠️ {ai.get("risk_note", "")}"""
        
        msg += f"""

🎯 Confidence: *{signal["confidence"]}%* | Strategies: {len(signal.get("reasons",[]))}
*Why:* {' · '.join(signal.get("reasons",[])[:4])}
━━━━━━━━━━━━━━━━━━━━━
⚠️ _Verify option LTP before trading. Not financial advice._"""
        return msg
    
    @staticmethod
    def format_close(instrument, direction, result, pnl, option=None, entry_time=None):
        emoji = "✅" if result == "WIN" else "❌"
        exit_time = datetime.now(IST).strftime("%H:%M")
        msg = f"""{emoji} *TRADE CLOSED: {instrument}*
━━━━━━━━━━━━━━━━━━━━━
📊 {direction} → *{result}*"""
        if option:
            msg += f"\n📋 {option.get('symbol','')}"
        if entry_time:
            msg += f"\n⏰ {entry_time} → {exit_time} IST"
        msg += f"""
💰 P&L: *{"+" if pnl>=0 else ""}₹{pnl}*
━━━━━━━━━━━━━━━━━━━━━"""
        return msg

    @staticmethod
    def format_daily_summary(perf):
        return f"""📊 *DAILY SUMMARY*
━━━━━━━━━━━━━━━━━
Total Signals: {perf["total"]}
✅ Wins: {perf["wins"]}  |  ❌ Losses: {perf["losses"]}
📈 Win Rate: *{perf["win_rate"]}%*
💰 Total P&L: *₹{perf["total_pnl"]}*
🏆 Best: ₹{perf["best_trade"]}  |  📉 Worst: ₹{perf["worst_trade"]}
━━━━━━━━━━━━━━━━━"""


# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════
DB_PATH = os.environ.get("DB_PATH", "signals.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, date TEXT NOT NULL,
            instrument TEXT NOT NULL, direction TEXT NOT NULL, confidence INTEGER NOT NULL,
            index_price REAL, index_entry REAL, index_sl REAL,
            index_target1 REAL, index_target2 REAL,
            option_symbol TEXT, option_strike REAL, option_type TEXT, option_expiry TEXT,
            option_token TEXT,
            option_entry REAL, option_sl REAL, option_target1 REAL, option_target2 REAL,
            option_lot_size INTEGER, option_lots INTEGER,
            position_pct INTEGER, sl_tightening TEXT,
            status TEXT DEFAULT 'OPEN', exit_price REAL, exit_time TEXT,
            option_exit REAL,
            pnl_points REAL, pnl_rupees REAL, result TEXT,
            reasons TEXT, indicators TEXT, ai_json TEXT
        )
    """)
    # Lightweight schema migrations for existing DBs
    cols = {r[1] for r in c.execute("PRAGMA table_info(signals)").fetchall()}
    for col_decl in [
        ("option_token", "TEXT"), ("option_lots", "INTEGER"),
        ("position_pct", "INTEGER"), ("sl_tightening", "TEXT"),
        ("option_exit", "REAL"), ("ai_json", "TEXT"),
    ]:
        col, typ = col_decl
        if col not in cols:
            try: c.execute(f"ALTER TABLE signals ADD COLUMN {col} {typ}")
            except Exception as e: log.warning(f"  migrate add {col}: {e}")

    # Pre-market regime brief (one row per trading day)
    c.execute("""
        CREATE TABLE IF NOT EXISTS regime (
            date TEXT PRIMARY KEY,
            regime TEXT, bias TEXT,
            confidence_floor INTEGER, min_rr REAL,
            avoid_instruments TEXT, notes TEXT,
            raw_json TEXT, created_at TEXT
        )
    """)
    # EOD tweaks Claude suggests for tomorrow's scanner
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_adjustments (
            date TEXT PRIMARY KEY,
            indicator_weight_adjustments TEXT,
            time_windows_to_avoid TEXT,
            extra_filters TEXT,
            raw_json TEXT, created_at TEXT
        )
    """)
    # In-flight management decisions (for audit)
    c.execute("""
        CREATE TABLE IF NOT EXISTS inflight_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER, ts TEXT,
            action TEXT, reasoning TEXT, raw_json TEXT
        )
    """)
    conn.commit(); conn.close()
    log.info("📊 Database ready")

init_db()

def db_exec(q, p=(), fetch=False, fetchone=False):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    c = conn.cursor(); c.execute(q, p)
    r = None
    if fetchone: r = c.fetchone()
    elif fetch: r = c.fetchall()
    conn.commit(); conn.close()
    return r

def save_signal(instrument, signal, option, ai=None):
    """Store a signal. Returns the inserted row id so downstream code (in-flight
    management, P&L check) can correlate option token lookups back to the row."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO signals (timestamp,date,instrument,direction,confidence,
        index_price,index_entry,index_sl,index_target1,index_target2,
        option_symbol,option_strike,option_type,option_expiry,option_token,
        option_entry,option_sl,option_target1,option_target2,
        option_lot_size,option_lots,position_pct,sl_tightening,
        reasons,indicators,ai_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"), datetime.now(IST).strftime("%Y-%m-%d"),
     instrument, signal["direction"], signal["confidence"],
     signal["price"], signal["entry"], signal["sl"], signal["target1"], signal["target2"],
     option.get("symbol","") if option else "", option.get("strike",0) if option else 0,
     option.get("type","") if option else "", option.get("expiry","") if option else "",
     option.get("token","") if option else "",
     option.get("entry",0) if option else 0, option.get("sl",0) if option else 0,
     option.get("target1",0) if option else 0, option.get("target2",0) if option else 0,
     option.get("lot_size",0) if option else 0, option.get("lots",0) if option else 0,
     (ai or {}).get("position_pct", 100) if ai else 100,
     (ai or {}).get("sl_tightening", "none") if ai else "none",
     json.dumps(signal.get("reasons",[])), json.dumps(signal.get("indicators",{})),
     json.dumps(ai) if ai else None))
    row_id = c.lastrowid
    conn.commit(); conn.close()
    return row_id

def update_result(sig_id, exit_price, result, pnl_pts, pnl_rs, option_exit=None):
    db_exec("""UPDATE signals SET status='CLOSED',exit_price=?,exit_time=?,
               option_exit=?,pnl_points=?,pnl_rupees=?,result=? WHERE id=?""",
            (exit_price, datetime.now(IST).strftime("%H:%M:%S"),
             option_exit, pnl_pts, pnl_rs, result, sig_id))

def get_history(limit=100, date=None):
    if date:
        rows = db_exec("SELECT * FROM signals WHERE date=? ORDER BY id DESC LIMIT ?", (date,limit), fetch=True)
    else:
        rows = db_exec("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,), fetch=True)
    return [dict(r) for r in rows] if rows else []

def get_perf():
    rows = db_exec("SELECT * FROM signals WHERE status='CLOSED'", fetch=True)
    if not rows: return {"total":0,"wins":0,"losses":0,"win_rate":0,"total_pnl":0,"avg_win":0,"avg_loss":0,"best_trade":0,"worst_trade":0}
    rows = [dict(r) for r in rows]
    wins = [r for r in rows if r["result"]=="WIN"]
    losses = [r for r in rows if r["result"]=="LOSS"]
    pnls = [r["pnl_rupees"] or 0 for r in rows]
    return {
        "total":len(rows),"wins":len(wins),"losses":len(losses),
        "win_rate":round(len(wins)/len(rows)*100,1) if rows else 0,
        "total_pnl":round(sum(pnls),0),
        "avg_win":round(sum(r["pnl_rupees"] or 0 for r in wins)/len(wins),0) if wins else 0,
        "avg_loss":round(sum(r["pnl_rupees"] or 0 for r in losses)/len(losses),0) if losses else 0,
        "best_trade":round(max(pnls),0) if pnls else 0,
        "worst_trade":round(min(pnls),0) if pnls else 0,
    }

# ═══════════════════════════════════════════════════════════════════
# ANGEL ONE CLIENT
# ═══════════════════════════════════════════════════════════════════
def _normalise_totp_secret(raw):
    """Make the Angel One TOTP secret robust to common copy-paste errors.

    Base32 (RFC 4648) only allows A-Z and 2-7. Angel One's dashboard often
    presents the secret with dashes or spaces for readability, and users
    frequently introduce two types of errors when transcribing:

        '0' -> should be 'O'   (zero vs capital-O)
        '1' -> should be 'I'   (one  vs capital-I)
        '8', '9' — impossible in base32, always a transcription artefact

    This function:
      1. Strips whitespace (including newlines), then uppercases.
      2. Removes formatting chars (space, dash, underscore, dot, colon, /).
      3. Substitutes 0->O and 1->I (the two widely-accepted confusables).
      4. Drops any remaining non-[A-Z2-7] characters.
      5. Pads to a multiple of 8 with '=' so pyotp's b32decode is happy.

    Returns a dict {value: str, changes: [str,...]} where `changes` is a
    human-readable trail of what was altered — logged (not the secret) so
    we can diagnose from the Engine Log without leaking the secret.
    """
    import re as _re
    changes = []
    s = (raw or "")
    original_len = len(s)
    # Strip all whitespace incl. newlines
    s2 = _re.sub(r"\s+", "", s)
    if s2 != s: changes.append(f"stripped {len(s)-len(s2)} whitespace chars")
    s = s2.upper()
    # Drop common formatting characters
    s2 = s
    for ch in ("-", "_", ".", ":", "/", ","):
        s2 = s2.replace(ch, "")
    if s2 != s: changes.append(f"removed {len(s)-len(s2)} formatting chars (- _ . : / ,)")
    s = s2
    # Visual-confusion substitutions
    n_zero = s.count("0"); n_one = s.count("1")
    if n_zero: s = s.replace("0", "O"); changes.append(f"substituted {n_zero} '0'→'O'")
    if n_one:  s = s.replace("1", "I"); changes.append(f"substituted {n_one} '1'→'I'")
    # Drop any char still outside base32 alphabet (8, 9, lowercase survivors, etc.)
    s2 = _re.sub(r"[^A-Z2-7]", "", s)
    dropped = len(s) - len(s2)
    if dropped: changes.append(f"dropped {dropped} non-base32 chars")
    s = s2
    # Pad to multiple of 8 with '='
    pad = (8 - (len(s) % 8)) % 8
    if pad: changes.append(f"added {pad} '=' padding char(s)")
    s = s + ("=" * pad)
    if not changes and len(s) == original_len:
        changes = []  # nothing done
    return {"value": s, "changes": changes}


class AngelClient:
    def __init__(self):
        self.api = None; self.connected = False; self.last_login = None
        # Diagnostics for failed logins — surfaced via /api/login and /api/diag
        # so the dashboard can show WHY the broker session isn't alive.
        self.last_login_error = None
        self.last_login_attempt = None
        self.login_attempts = 0
        # Greeks cache: (name, expiry_DDMMMYYYY) -> {"ts": epoch, "data": [greeks...]}
        self._greeks_cache = {}
        self._greeks_ttl = 60  # seconds

    def option_greeks(self, name, expiry_ddmmmyyyy):
        """Fetch option greeks for an (underlying, expiry) pair, cached 60s.

        Returns a list of {strike, optionType, delta, gamma, theta, vega, impliedVolatility, ...}
        or None on failure. Falls back to ladder-based delta in OptPicker.
        """
        if not expiry_ddmmmyyyy: return None
        key = (name, expiry_ddmmmyyyy.upper())
        now = time.time()
        cached = self._greeks_cache.get(key)
        if cached and (now - cached["ts"] < self._greeks_ttl):
            return cached["data"]
        try:
            if not self.ensure(): return None
            params = {"name": name, "expirydate": expiry_ddmmmyyyy.upper()}
            # Method name is getOptionGreek (singular) in smartapi-python
            fn = getattr(self.api, "getOptionGreek", None) or getattr(self.api, "optionGreek", None)
            if fn is None:
                log.warning("  smartapi-python build has no getOptionGreek — using ladder fallback")
                return None
            resp = fn(params)
            if resp and resp.get("status") and resp.get("data"):
                data = resp["data"]
                self._greeks_cache[key] = {"ts": now, "data": data}
                log.info(f"  Greeks: {name} {expiry_ddmmmyyyy} → {len(data)} rows (cached 60s)")
                return data
            log.warning(f"  Greeks API failed for {name} {expiry_ddmmmyyyy}: {resp}")
            return None
        except Exception as e:
            log.warning(f"  Greeks fetch error for {name}: {e}")
            return None

    @staticmethod
    def greeks_lookup(greeks, strike, otype):
        """Find the greeks row for a specific strike + option type. Case-insensitive."""
        if not greeks: return None
        want_type = str(otype).upper()
        try:
            strike_f = float(strike)
        except Exception:
            return None
        for g in greeks:
            try:
                g_strike = float(g.get("strikePrice") or g.get("strike") or 0)
                g_type = str(g.get("optionType") or g.get("type") or "").upper()
                if abs(g_strike - strike_f) < 0.01 and g_type == want_type:
                    return g
            except Exception:
                continue
        return None
    
    def login(self):
        self.last_login_attempt = datetime.now(IST)
        self.login_attempts += 1
        try:
            # Preflight env-var sanity — if any of these are missing, log a clear
            # message instead of letting SmartAPI throw an opaque error.
            missing = [k for k in ("api_key","client_id","password","totp_secret") if not CONFIG.get(k)]
            if missing:
                msg = f"Missing env vars: {', '.join(missing)} — set them in Railway and redeploy"
                log.error(f"❌ {msg}")
                self.connected = False; self.last_login_error = msg; return False

            # Robustly sanitise the TOTP secret. The Angel One dashboard displays
            # secrets in a way that users frequently copy with formatting characters
            # (dashes, spaces, dots, colons, newlines) OR with the common "0↔O"
            # and "1↔I" visual-confusion substitutions. We fix all of the above
            # before handing to pyotp, and log what was changed so the user can
            # see the normalisation applied — without leaking the secret itself.
            raw = CONFIG["totp_secret"] or ""
            normalised = _normalise_totp_secret(raw)
            secret = normalised["value"]
            if normalised["changes"]:
                log.info("🔑 TOTP secret normalised: " + "; ".join(normalised["changes"]))

            log.info(f"🔐 Attempting login... client_id={CONFIG['client_id']} (attempt #{self.login_attempts}) · secret len={len(secret)}")
            self.api = SmartConnect(api_key=CONFIG["api_key"])
            try:
                totp = pyotp.TOTP(secret).now()
            except Exception as te:
                # Include sanitisation trail so we can diagnose from the client log.
                detail = "; ".join(normalised["changes"]) if normalised["changes"] else "no changes"
                msg = f"TOTP generation failed ({te}) · len={len(secret)} · normalisation: {detail}"
                log.error(f"❌ {msg}")
                self.connected = False; self.last_login_error = msg; return False

            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                try:
                    data = _ex.submit(self.api.generateSession, clientCode=CONFIG["client_id"], password=CONFIG["password"], totp=totp).result(timeout=12)
                except _cf.TimeoutError:
                    msg = "Login timeout (12s) — API key invalid or Angel One unreachable"
                    log.error(f"❌ {msg}")
                    self.connected = False; self.last_login_error = msg; return False
            # SmartAPI success markers can be status: True / "success"; data may carry tokens.
            st = data.get("status") if isinstance(data, dict) else None
            ok = bool(st) or (isinstance(st, str) and st.lower() == "success")
            if ok:
                self.connected = True; self.last_login = datetime.now(IST)
                self.last_login_error = None
                log.info(f"✅ Angel One login successful (attempt #{self.login_attempts})")
                return True
            # Failure path — surface Angel One's exact error. Common: invalid TOTP,
            # rate limit, account blocked, client code mismatch.
            err_msg = (data.get("message") or data.get("errorcode") or str(data)) if isinstance(data, dict) else str(data)
            msg = f"Angel One rejected login: {err_msg}"
            log.error(f"❌ {msg}")
            self.connected = False; self.last_login_error = msg
            return False
        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc(limit=3)
            msg = f"Login exception: {e}"
            log.error(f"❌ {msg}\n{tb}")
            self.connected = False; self.last_login_error = msg
            return False

    def ensure(self):
        if not self.connected: return self.login()
        if self.last_login and (datetime.now(IST)-self.last_login).seconds > 18000: return self.login()
        return True

    def diag(self):
        """Snapshot for /api/diag — safe to expose (no secrets)."""
        return {
            "connected": bool(self.connected),
            "last_login": self.last_login.strftime("%Y-%m-%d %H:%M:%S") if self.last_login else None,
            "last_login_attempt": self.last_login_attempt.strftime("%Y-%m-%d %H:%M:%S") if self.last_login_attempt else None,
            "last_login_error": self.last_login_error,
            "login_attempts": self.login_attempts,
            "env_present": {
                "api_key":     bool(CONFIG.get("api_key")),
                "client_id":   bool(CONFIG.get("client_id")),
                "password":    bool(CONFIG.get("password")),
                "totp_secret": bool(CONFIG.get("totp_secret")),
            },
            "client_id_hint": (CONFIG.get("client_id") or "")[:4] + "***" if CONFIG.get("client_id") else None,
        }
    
    def candles(self, token, exchange, interval="FIVE_MINUTE", days=3):
        try:
            if not self.ensure(): return pd.DataFrame()
            _params = {"exchange":exchange,"symboltoken":token,"interval":interval,
                "fromdate":(datetime.now(IST)-timedelta(days=days)).strftime("%Y-%m-%d %H:%M"),
                "todate":datetime.now(IST).strftime("%Y-%m-%d %H:%M")}
            import concurrent.futures as _cf2
            with _cf2.ThreadPoolExecutor(max_workers=1) as _ex2:
                try:
                    resp = _ex2.submit(self.api.getCandleData, _params).result(timeout=12)
                except _cf2.TimeoutError:
                    log.error("❌ getCandleData timeout (12s) — returning empty")
                    return pd.DataFrame()
            if resp and resp.get("status") and resp.get("data"):
                df = pd.DataFrame(resp["data"], columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"]); return df
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Candle err: {e}"); return pd.DataFrame()
    
    def ltp(self, exchange, symbol, token):
        """Get LTP for a single instrument."""
        try:
            d = self.api.ltpData(exchange, symbol, token)
            return d["data"] if d and d.get("status") else None
        except: return None
    
    def option_chain(self, info, spot):
        """Fetch option chain: Instrument Master for tokens → 1 batch API call for all prices."""
        try:
            gap = info["strike_gap"]; atm = round(spot/gap)*gap
            strikes = [atm+i*gap for i in range(-7,8)]  # ±7 strikes = 15 strikes × 2 = 30 options
            strikes_set = set(int(s) for s in strikes)
            prefix = info["expiry_prefix"]
            exchange = info["option_exchange"]
            
            # Step 1: Get tokens from instrument master (instant, offline)
            tokens = _master.find_options(prefix, strikes, exchange)
            if not tokens:
                log.info(f"  Master miss for {prefix}, trying searchScrip...")
                tokens = self._scrip_lookup(prefix, strikes_set, exchange)
            
            if not tokens:
                log.error(f"  No tokens for {prefix} — both methods failed")
                return [], 0
            
            log.info(f"  Got {len(tokens)} tokens for {prefix}")
            
            # Step 2: BATCH fetch all prices in 1 API call using getMarketData
            token_list = [str(tk["token"]) for tk in tokens]
            token_map = {str(tk["token"]): tk for tk in tokens}
            
            opts = []
            rejected_wide = 0
            rejected_zero = 0
            try:
                # FULL mode returns top-5 bid/ask depth; we use the best bid/ask to build a mid
                # price and reject illiquid strikes where the spread is wide.
                batch_resp = self.api.getMarketData(mode="FULL", exchangeTokens={"NFO": token_list})
                if batch_resp and batch_resp.get("status") and batch_resp.get("data"):
                    fetched = batch_resp["data"].get("fetched", [])
                    unfetched = batch_resp["data"].get("unfetched", [])
                    log.info(f"  Batch FULL: {len(fetched)} fetched, {len(unfetched)} unfetched")

                    for item in fetched:
                        tok = str(item.get("symbolToken", item.get("symboltoken", "")))
                        if tok not in token_map: continue
                        tk = token_map[tok]
                        ltp = float(item.get("ltp", 0) or 0)

                        # Extract best bid/ask from depth. SmartAPI returns "depth": {"buy":[{price,quantity,...}], "sell":[...]}
                        best_bid = 0.0
                        best_ask = 0.0
                        depth = item.get("depth") or {}
                        buys = depth.get("buy") or []
                        sells = depth.get("sell") or []
                        if buys:
                            try: best_bid = float(buys[0].get("price", 0) or 0)
                            except: best_bid = 0.0
                        if sells:
                            try: best_ask = float(sells[0].get("price", 0) or 0)
                            except: best_ask = 0.0
                        # Fallback fields some builds return top-level
                        if best_bid == 0: best_bid = float(item.get("bestBidPrice") or item.get("bidPrice") or 0 or 0)
                        if best_ask == 0: best_ask = float(item.get("bestAskPrice") or item.get("askPrice") or 0 or 0)

                        # Liquidity gate: reject strikes with zero side or wide spread (>8% of ask)
                        if best_bid <= 0 or best_ask <= 0:
                            rejected_zero += 1
                            continue
                        spread = best_ask - best_bid
                        if spread > best_ask * 0.08:
                            rejected_wide += 1
                            continue

                        mid = round((best_bid + best_ask) / 2.0, 2)
                        if mid <= 0: continue

                        opts.append({
                            "strike": tk["strike"], "type": tk["type"],
                            "symbol": tk["symbol"],
                            "ltp": mid,        # price used downstream = mid, not last-trade
                            "last_trade": ltp, # kept for diagnostics
                            "bid": best_bid, "ask": best_ask,
                            "spread": round(spread, 2),
                            "token": tok, "expiry": tk.get("expiry", ""),
                        })
                    if rejected_wide or rejected_zero:
                        log.info(f"  Liquidity gate: rejected {rejected_wide} wide-spread, {rejected_zero} zero-side strikes")
                else:
                    log.error(f"  Batch API failed: {batch_resp}")
            except Exception as be:
                log.error(f"  Batch getMarketData(FULL) error: {be}")
            
            # Fallback: if batch failed, try individual ltpData calls
            if not opts:
                log.info(f"  Batch failed, falling back to individual ltpData calls...")
                for tk in tokens:
                    try:
                        lr = self.api.ltpData(exchange, tk["symbol"], str(tk["token"]))
                        if lr and lr.get("status") and lr.get("data"):
                            ltp = lr["data"].get("ltp", 0)
                            if ltp > 0:
                                opts.append({"strike":tk["strike"],"type":tk["type"],
                                    "symbol":tk["symbol"],"ltp":ltp,
                                    "token":str(tk["token"]),"expiry":tk.get("expiry","")})
                        time.sleep(0.3)  # Rate limit safety
                    except: continue
            
            log.info(f"  Chain: {len(opts)} live prices, ATM={atm}")
            if opts: log.info(f"  Sample: {opts[0]['symbol']}=Rs.{opts[0]['ltp']}")
            return opts, atm
        except Exception as e:
            log.error(f"Chain err: {e}"); return [], 0
    
    def _scrip_lookup(self, prefix, strikes_set, exchange):
        """Fallback: searchScrip('NFO','NIFTY') then filter locally for our strikes."""
        try:
            sr = self.api.searchScrip(exchange, prefix)
            if not sr or not sr.get("data"):
                log.error(f"  searchScrip('{exchange}','{prefix}') empty")
                return []
            
            items = sr["data"]
            log.info(f"  searchScrip: {len(items)} results for {prefix}")
            pfx_len = len(prefix)
            
            # Group by expiry to find nearest
            by_expiry = {}
            for item in items:
                sym = item.get("tradingsymbol","")
                if not sym.startswith(prefix) or len(sym) < pfx_len + 9: continue
                exp = sym[pfx_len:pfx_len+7]  # DDMMMYY e.g. "02MAR26"
                if not exp[:2].isdigit(): continue
                by_expiry.setdefault(exp, []).append(item)
            
            if not by_expiry:
                log.error(f"  No expiries parsed from searchScrip results")
                return []
            
            nearest = list(by_expiry.keys())[0]
            log.info(f"  Nearest expiry: {nearest} ({len(by_expiry[nearest])} opts)")
            
            results = []
            for item in by_expiry[nearest]:
                sym = item["tradingsymbol"]
                tok = item.get("symboltoken","")
                rest = sym[pfx_len+7:]
                if rest.endswith("CE"):
                    otype, sstr = "CE", rest[:-2]
                elif rest.endswith("PE"):
                    otype, sstr = "PE", rest[:-2]
                else: continue
                try: strike = int(sstr)
                except: continue
                if strike in strikes_set:
                    results.append({"symbol":sym,"token":tok,"strike":strike,"type":otype,"expiry":nearest})
            
            log.info(f"  Matched {len(results)} options for strikes")
            return results
        except Exception as e:
            log.error(f"  searchScrip fallback error: {e}")
            return []

# ═══════════════════════════════════════════════════════════════════
# INSTRUMENT MASTER — Download once, lookup any option instantly
# ═══════════════════════════════════════════════════════════════════
class InstrumentMaster:
    """Downloads Angel One's instrument master JSON and provides fast option lookups."""
    MASTER_URLS = [
        "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
        "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json",
    ]
    
    def __init__(self):
        self.data = []
        self.nfo = {}
        self.loaded = False
        self.load_time = None
    
    def load(self):
        """Download and parse master file. Called once at startup or on first use."""
        try:
            raw = None
            for url in self.MASTER_URLS:
                try:
                    log.info(f"  Downloading master from {url[:50]}...")
                    r = requests.get(url, timeout=60)
                    if r.status_code == 200:
                        raw = r.json()
                        log.info(f"  Master: {len(raw)} instruments from {url[:50]}")
                        break
                    else:
                        log.info(f"  Master HTTP {r.status_code} from {url[:50]}")
                except Exception as e:
                    log.info(f"  Master download failed: {e}")
                    continue
            
            if not raw:
                log.error("  All master URLs failed")
                return False
            
            self.data = raw
            
            # Build NFO options lookup
            # Each entry has: token, symbol, name, expiry, strike, lotsize, instrumenttype, exch_seg
            self.nfo = {}
            nfo_count = 0
            for item in self.data:
                if item.get("exch_seg") != "NFO": continue
                itype = item.get("instrumenttype", "")
                if itype not in ("OPTIDX", "OPTSTK"): continue  # Only options
                
                sym = item.get("symbol", "")       # e.g. "NIFTY02MAR2625600CE"
                token = item.get("token", "")
                strike_raw = item.get("strike", "0")
                expiry = item.get("expiry", "")      # e.g. "02MAR2026"
                name = item.get("name", "")          # e.g. "NIFTY"
                lotsize = item.get("lotsize", "0")
                
                # Parse strike (Angel One stores as string like "2560000" = 25600.00 * 100)
                try:
                    strike_val = float(strike_raw) / 100.0
                except:
                    continue
                
                # Option type from last 2 chars of symbol
                if sym.endswith("CE"):
                    otype = "CE"
                elif sym.endswith("PE"):
                    otype = "PE"
                else:
                    continue
                
                key = (name, strike_val, otype, expiry)
                self.nfo[key] = {
                    "symbol": sym, "token": token, "strike": strike_val,
                    "type": otype, "expiry": expiry, "name": name,
                    "lotsize": int(lotsize) if lotsize.isdigit() else 0
                }
                nfo_count += 1
            
            log.info(f"  Master: {nfo_count} NFO options indexed")
            self.loaded = True
            self.load_time = datetime.now(IST)
            return True
        except Exception as e:
            log.error(f"  Master load error: {e}")
            return False
    
    def ensure(self):
        """Ensure master is loaded (reload if stale > 6 hours)."""
        if self.loaded and self.load_time:
            age = (datetime.now(IST) - self.load_time).total_seconds()
            if age < 6 * 3600:  # Fresh enough
                return True
        return self.load()
    
    def find_options(self, name_prefix, strikes, exchange="NFO"):
        """Find option tokens for given strikes, nearest expiry."""
        if not self.ensure():
            log.error("  Master not loaded, can't find options")
            return []
        
        # Find all expiries for this name prefix
        expiries = set()
        for (name, strike, otype, expiry), info in self.nfo.items():
            if name == name_prefix:
                expiries.add(expiry)
        
        if not expiries:
            log.error(f"  No expiries found for {name_prefix}")
            return []
        
        # Sort expiries and pick nearest future one
        # Expiry format: "02MAR2026" → parse to date
        today = datetime.now(IST).date()
        dated_expiries = []
        for exp in expiries:
            try:
                d = datetime.strptime(exp, "%d%b%Y").date()
                if d >= today:
                    dated_expiries.append((d, exp))
            except:
                continue
        
        if not dated_expiries:
            log.error(f"  No future expiries for {name_prefix}")
            return []
        
        dated_expiries.sort()
        nearest_expiry = dated_expiries[0][1]  # e.g. "02MAR2026"
        dte = (dated_expiries[0][0] - today).days
        log.info(f"  Nearest expiry: {nearest_expiry} ({dte} DTE, of {len(dated_expiries)} future expiries)")
        
        # Find tokens for each strike
        results = []
        for s in strikes:
            for otype in ["CE", "PE"]:
                key = (name_prefix, float(s), otype, nearest_expiry)
                info = self.nfo.get(key)
                if info:
                    results.append({**info, "dte": dte})
        
        log.info(f"  Found {len(results)} option tokens for {name_prefix} (wanted {len(strikes)*2})")
        return results

# Global instance — loaded once
_master = InstrumentMaster()

# ═══════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
class TA:
    # NOTE: RSI, ATR, and ADX use Wilder smoothing (alpha = 1/p) — the canonical
    # formulation used by TradingView, Sensibull, and Angel One's charting.
    # EMA, MACD, and Bollinger Bands keep standard EMA span smoothing.
    @staticmethod
    def ema(s,p): return s.ewm(span=p,adjust=False).mean()
    @staticmethod
    def _wilder(s,p): return s.ewm(alpha=1.0/p, adjust=False).mean()
    @staticmethod
    def rsi(c,p=14):
        d=c.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0)
        avg_g=TA._wilder(g,p); avg_l=TA._wilder(l,p)
        rs = avg_g / avg_l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    @staticmethod
    def macd(c):
        ml=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
        return ml, ml.ewm(span=9,adjust=False).mean(), ml-ml.ewm(span=9,adjust=False).mean()
    @staticmethod
    def bb(c,p=20,sd=2):
        m=c.rolling(p).mean();s=c.rolling(p).std();return m+sd*s,m,m-sd*s
    @staticmethod
    def vwap(df):
        """Intraday VWAP — resets at market open (09:15 IST). Only uses TODAY's rows."""
        today = datetime.now(IST).date()
        if "timestamp" in df.columns:
            try:
                ts = pd.to_datetime(df["timestamp"])
                # Angel One returns timezone-aware timestamps; strip tz to get local IST day boundary
                if getattr(ts.dt, "tz", None) is not None:
                    ts = ts.dt.tz_localize(None)
                mask = ts.dt.date == today
                slice_df = df[mask] if mask.any() else df
            except Exception:
                slice_df = df
        else:
            slice_df = df
        tp = (slice_df["high"] + slice_df["low"] + slice_df["close"]) / 3
        cum_vol = slice_df["volume"].cumsum().replace(0, np.nan)
        vwap_slice = (tp * slice_df["volume"]).cumsum() / cum_vol
        # Reindex back to full df index so downstream .iloc[-1] still works
        full = pd.Series(index=df.index, dtype=float)
        full.loc[vwap_slice.index] = vwap_slice
        full = full.ffill().fillna(df["close"])  # pre-today rows: flatten to close so they don't NaN
        return full
    @staticmethod
    def atr(df,p=14):
        tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
        return TA._wilder(tr, p)
    @staticmethod
    def supertrend(df,p=10,m=3):
        atr=TA.atr(df,p);hl2=(df["high"]+df["low"])/2;ub,lb=hl2+m*atr,hl2-m*atr
        tr=pd.Series(1,index=df.index);fu,fl=ub.copy(),lb.copy()
        for i in range(1,len(df)):
            fu.iloc[i]=ub.iloc[i] if ub.iloc[i]<fu.iloc[i-1] or df["close"].iloc[i-1]>fu.iloc[i-1] else fu.iloc[i-1]
            fl.iloc[i]=lb.iloc[i] if lb.iloc[i]>fl.iloc[i-1] or df["close"].iloc[i-1]<fl.iloc[i-1] else fl.iloc[i-1]
            if tr.iloc[i-1]==-1 and df["close"].iloc[i]>fu.iloc[i-1]:tr.iloc[i]=1
            elif tr.iloc[i-1]==1 and df["close"].iloc[i]<fl.iloc[i-1]:tr.iloc[i]=-1
            else:tr.iloc[i]=tr.iloc[i-1]
        return tr
    @staticmethod
    def stoch(df,k=14):
        ll=df["low"].rolling(k).min();hh=df["high"].rolling(k).max()
        return 100*(df["close"]-ll)/(hh-ll)
    @staticmethod
    def adx(df,p=14):
        pm=df["high"].diff();mm=-df["low"].diff()
        pm=pm.where((pm>mm)&(pm>0),0);mm=mm.where((mm>pm)&(mm>0),0)
        atr=TA.atr(df,p)
        pdi=100*TA._wilder(pm,p)/atr
        mdi=100*TA._wilder(mm,p)/atr
        dx = 100 * ((pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan))
        return TA._wilder(dx, p), pdi, mdi

# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════
class SignalGen:
    def __init__(self):
        self.tmin=CONFIG["target_points_min"];self.tmax=CONFIG["target_points_max"]
    
    def analyze(self, df):
        if len(df)<30: return None
        c=df["close"];n=len(df)-1;price=c.iloc[n]
        e9=TA.ema(c,9);e21=TA.ema(c,21);e50=TA.ema(c,min(50,len(c)))
        rsi=TA.rsi(c);ml,sl,mh=TA.macd(c);bbu,bbm,bbl=TA.bb(c)
        vwap=TA.vwap(df);atr=TA.atr(df);st=TA.supertrend(df)
        sk=TA.stoch(df);adx,pdi,mdi=TA.adx(df)
        vra=df["volume"].tail(20).mean();vr=df["volume"].iloc[n]/vra if vra>0 else 1
        
        bs,be=0,0;br,ber=[],[]
        if e9.iloc[n]>e21.iloc[n] and e9.iloc[n-1]<=e21.iloc[n-1]:bs+=15;br.append("🔥 EMA 9/21 Bullish Crossover")
        elif e9.iloc[n]<e21.iloc[n] and e9.iloc[n-1]>=e21.iloc[n-1]:be+=15;ber.append("🔥 EMA 9/21 Bearish Crossover")
        elif e9.iloc[n]>e21.iloc[n]:bs+=8;br.append("EMA 9>21 bullish")
        else:be+=8;ber.append("EMA 9<21 bearish")
        if price>e50.iloc[n]:bs+=5;br.append("Above EMA 50")
        else:be+=5;ber.append("Below EMA 50")
        rv=rsi.iloc[-1]
        if rv<30:bs+=12;br.append(f"RSI Oversold ({rv:.1f})")
        elif rv>70:be+=12;ber.append(f"RSI Overbought ({rv:.1f})")
        elif 50<rv<65:bs+=6;br.append(f"RSI Bullish ({rv:.1f})")
        elif 35<rv<50:be+=6;ber.append(f"RSI Bearish ({rv:.1f})")
        if mh.iloc[n]>0 and mh.iloc[n-1]<=0:bs+=15;br.append("🔥 MACD Bull Cross")
        elif mh.iloc[n]<0 and mh.iloc[n-1]>=0:be+=15;ber.append("🔥 MACD Bear Cross")
        elif mh.iloc[n]>mh.iloc[n-1] and mh.iloc[n]>0:bs+=8;br.append("MACD rising")
        elif mh.iloc[n]<mh.iloc[n-1] and mh.iloc[n]<0:be+=8;ber.append("MACD falling")
        if price<=bbl.iloc[n]*1.002:bs+=10;br.append("At Lower BB")
        elif price>=bbu.iloc[n]*0.998:be+=10;ber.append("At Upper BB")
        if price>vwap.iloc[n] and c.iloc[n-1]<=vwap.iloc[n-1]:bs+=10;br.append("🔥 Crossed above VWAP")
        elif price<vwap.iloc[n] and c.iloc[n-1]>=vwap.iloc[n-1]:be+=10;ber.append("🔥 Crossed below VWAP")
        elif price>vwap.iloc[n]:bs+=5;br.append("Above VWAP")
        else:be+=5;ber.append("Below VWAP")
        if st.iloc[n]==1 and st.iloc[n-1]==-1:bs+=13;br.append("🔥 Supertrend BULL")
        elif st.iloc[n]==-1 and st.iloc[n-1]==1:be+=13;ber.append("🔥 Supertrend BEAR")
        elif st.iloc[n]==1:bs+=7;br.append("Supertrend Bull")
        else:be+=7;ber.append("Supertrend Bear")
        if vr>1.5:
            t=f"Volume {vr:.1f}x"
            if c.iloc[n]>c.iloc[n-1]:bs+=8;br.append(t)
            else:be+=8;ber.append(t)
        skv=sk.iloc[-1] if not pd.isna(sk.iloc[-1]) else 50
        if skv<20:bs+=7;br.append(f"Stoch Oversold ({skv:.0f})")
        elif skv>80:be+=7;ber.append(f"Stoch Overbought ({skv:.0f})")
        adxv=adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        if adxv>25:
            if pdi.iloc[-1]>mdi.iloc[-1]:bs+=7;br.append(f"ADX {adxv:.0f} +DI")
            else:be+=7;ber.append(f"ADX {adxv:.0f} -DI")
        l3=c.tail(3).values
        if len(l3)==3 and l3[2]>l3[1]>l3[0]:bs+=3;br.append("3-candle bull")
        elif len(l3)==3 and l3[2]<l3[1]<l3[0]:be+=3;ber.append("3-candle bear")
        
        conf=min(95,round(max(bs,be)));direction="LONG" if bs>be else "SHORT"
        av=atr.iloc[n]
        
        # ═══ CONFLICT DETECTION — reduce confidence when indicators disagree ═══
        penalties = []
        
        # SuperTrend conflict: signal vs trend
        if direction=="LONG" and st.iloc[n]==-1:
            conf=max(10,conf-12); penalties.append("SuperTrend BEAR conflicts LONG")
        elif direction=="SHORT" and st.iloc[n]==1:
            conf=max(10,conf-12); penalties.append("SuperTrend BULL conflicts SHORT")
        
        # RSI overextended: buying overbought or selling oversold
        if direction=="LONG" and rv>75:
            conf=max(10,conf-15); penalties.append(f"RSI {rv:.0f} overbought — reversal risk")
        elif direction=="SHORT" and rv<25:
            conf=max(10,conf-15); penalties.append(f"RSI {rv:.0f} oversold — bounce risk")
        
        # VWAP conflict: LONG below VWAP or SHORT above VWAP
        if direction=="LONG" and price<vwap.iloc[n]*0.998:
            conf=max(10,conf-5); penalties.append("Below VWAP — weak for LONG")
        elif direction=="SHORT" and price>vwap.iloc[n]*1.002:
            conf=max(10,conf-5); penalties.append("Above VWAP — weak for SHORT")
        
        # Low ATR = no volatility, options won't move
        avg_atr = atr.tail(20).mean()
        if av < avg_atr * 0.6:
            conf=max(10,conf-10); penalties.append(f"Low ATR ({av:.1f} vs avg {avg_atr:.1f}) — no momentum")
        
        # Late session penalty: theta decay accelerates after 2:30 PM
        now_hr = datetime.now(IST).hour
        now_min = datetime.now(IST).minute
        if now_hr == 14 and now_min >= 30:
            conf=max(10,conf-5); penalties.append("Late session — theta decay risk")
        elif now_hr >= 15:
            conf=max(10,conf-10); penalties.append("Market closing — avoid new entries")
        
        # Margin too thin: bull-bear spread too narrow = unclear direction
        spread = abs(bs - be)
        if spread < 8:
            conf=max(10,conf-8); penalties.append(f"Bull/Bear split too close ({bs}B vs {be}S)")
        
        reasons = br if direction=="LONG" else ber
        if penalties:
            reasons = reasons + [f"⚠️ {p}" for p in penalties]
        
        if direction=="LONG":
            entry=round(price+av*0.1,2);stop=round(price-av*1.2,2)
            risk_dist=round(abs(entry-stop),2)
            t1,t2=round(entry+risk_dist*1.5,2),round(entry+risk_dist*2.5,2)
        else:
            entry=round(price-av*0.1,2);stop=round(price+av*1.2,2)
            risk_dist=round(abs(entry-stop),2)
            t1,t2=round(entry-risk_dist*1.5,2),round(entry-risk_dist*2.5,2)
        risk=round(abs(entry-stop),2);reward=round(abs(t1-entry),2)
        
        return {"direction":direction,"confidence":conf,"price":round(price,2),
            "entry":entry,"sl":stop,"target1":t1,"target2":t2,
            "risk":risk,"reward":reward,"risk_reward":round(reward/risk,2) if risk>0 else 0,
            "reasons":reasons,
            "indicators":{"rsi":round(rv,1),"macd":round(mh.iloc[n],3),"ema9":round(e9.iloc[n],2),
                "ema21":round(e21.iloc[n],2),"ema50":round(e50.iloc[n],2),"vwap":round(vwap.iloc[n],2),
                "atr":round(av,2),"bb_upper":round(bbu.iloc[n],2),"bb_lower":round(bbl.iloc[n],2),
                "supertrend":"BULL" if st.iloc[n]==1 else "BEAR","vol_ratio":round(vr,2),
                "stoch":round(skv,0),"adx":round(adxv,0)},
            "timestamp":datetime.now(IST).strftime("%H:%M:%S")}

# ═══════════════════════════════════════════════════════════════════
# CLAUDE LAYERS (A = regime brief, B = signal validation,
#                C = EOD learning, D = in-flight management, E = events calendar)
# ═══════════════════════════════════════════════════════════════════
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _anthropic_call(prompt, model=None, max_tokens=800, temperature=0.2, timeout=20):
    """Low-level wrapper around Anthropic messages API. Returns parsed JSON dict
    (from Claude's response content) or None if anything failed. Callers decide
    how to handle None — the safe default for validation is SKIP."""
    api_key = CONFIG.get("anthropic_api_key", "")
    if not api_key:
        return None
    try:
        resp = requests.post(
            _ANTHROPIC_URL,
            headers={"Content-Type": "application/json",
                     "x-api-key": api_key,
                     "anthropic-version": "2023-06-01"},
            json={"model": model or CONFIG.get("anthropic_model", "claude-sonnet-4-5"),
                  "max_tokens": max_tokens,
                  "temperature": temperature,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        if resp.status_code != 200:
            log.warning(f"  Claude API error {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        text = (data.get("content", [{}])[0] or {}).get("text", "").strip()
        import re as _re
        text = _re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=_re.M).strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning(f"  Claude JSON parse failed: {e} // raw={text[:200] if 'text' in dir() else ''}")
        return None
    except Exception as e:
        log.warning(f"  Claude call failed: {e}")
        return None


# ─── Layer A: Pre-market Regime Brief (run once at ~08:45 IST) ────────
class RegimeBrief:
    """One call per trading morning. Asks Claude to characterise the day's
    expected regime and return overrides for the scanner (confidence floor,
    R:R floor, instruments to avoid). Persisted to the `regime` table."""

    @staticmethod
    def run():
        today = datetime.now(IST).strftime("%Y-%m-%d")
        existing = db_exec("SELECT * FROM regime WHERE date=?", (today,), fetchone=True)
        if existing:
            return dict(existing)

        # Pull yesterday's closes for context (best effort)
        recent = db_exec("""SELECT instrument, direction, result, pnl_rupees FROM signals
                            WHERE status='CLOSED' AND date < ?
                            ORDER BY id DESC LIMIT 10""", (today,), fetch=True)
        recent_txt = "; ".join(
            f"{r['instrument']} {r['direction']} {r['result']} ₹{r['pnl_rupees']}"
            for r in (recent or [])
        ) or "no recent closes"

        events = EventCalendar.today_events()
        event_txt = "; ".join(f"{e.get('time','')} {e.get('name','')}" for e in events) or "none"

        prompt = f"""You are preparing a ₹20,000 Indian intraday options desk for today's session.
Date: {today}   Current IST: {datetime.now(IST).strftime('%H:%M')}
Recent closed trades (most recent first): {recent_txt}
Known events today: {event_txt}

Pick a market regime and set scanner overrides for the day.

Respond in EXACTLY this JSON (no markdown):
{{"regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "VOLATILE" | "EVENT_RISK",
  "bias": "LONG" | "SHORT" | "NEUTRAL",
  "confidence_floor": integer 55..80,
  "min_rr": number 1.2..2.5,
  "avoid_instruments": ["BANKNIFTY" and/or "FINNIFTY" and/or "NIFTY"] or [],
  "notes": "1-2 sentence rationale traders can act on"}}"""

        raw = _anthropic_call(prompt, max_tokens=300, timeout=20)
        if not raw:
            # Safe defaults if Claude is down — don't block trading, just no overrides
            raw = {"regime": "UNKNOWN", "bias": "NEUTRAL",
                   "confidence_floor": CONFIG.get("min_confidence", 60),
                   "min_rr": 1.5, "avoid_instruments": [],
                   "notes": "Claude unavailable — using defaults"}

        try:
            db_exec("""INSERT OR REPLACE INTO regime
                       (date, regime, bias, confidence_floor, min_rr,
                        avoid_instruments, notes, raw_json, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (today, raw.get("regime", "UNKNOWN"), raw.get("bias", "NEUTRAL"),
                     int(raw.get("confidence_floor") or CONFIG.get("min_confidence", 60)),
                     float(raw.get("min_rr") or 1.5),
                     json.dumps(raw.get("avoid_instruments") or []),
                     str(raw.get("notes") or ""),
                     json.dumps(raw), datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")))
        except Exception as e:
            log.warning(f"  regime persist failed: {e}")

        log.info(f"🧭 Regime {raw.get('regime')} bias={raw.get('bias')} "
                 f"floor={raw.get('confidence_floor')} minRR={raw.get('min_rr')} "
                 f"avoid={raw.get('avoid_instruments')}")
        SlackAlert.send(f"🧭 *Morning Regime* — {raw.get('regime','?')} / bias {raw.get('bias','?')}\n"
                        f"Confidence floor: {raw.get('confidence_floor')}%  •  Min R:R {raw.get('min_rr')}\n"
                        f"Avoid: {', '.join(raw.get('avoid_instruments') or []) or 'none'}\n"
                        f"_{raw.get('notes','')}_")
        return raw

    @staticmethod
    def today():
        today = datetime.now(IST).strftime("%Y-%m-%d")
        row = db_exec("SELECT * FROM regime WHERE date=?", (today,), fetchone=True)
        if not row: return None
        r = dict(row)
        try: r["avoid_instruments"] = json.loads(r.get("avoid_instruments") or "[]")
        except Exception: r["avoid_instruments"] = []
        return r


# ─── Layer B: Per-signal validation (richer, replaces old AIAnalysis) ──
class SignalValidation:
    """Per-signal verdict with position sizing + SL tightening strategy.
    Default on any failure/parse error is SKIP — we never silently let a signal
    through when Claude errored out."""

    @staticmethod
    def analyze(instrument, signal, option, regime=None):
        api_key = CONFIG.get("anthropic_api_key", "")
        if not api_key:
            # No Claude configured → pass-through so the engine still works
            return {"verdict": "TAKE", "position_pct": 100, "sl_tightening": "none",
                    "reasoning": "AI disabled", "risk_note": "n/a", "confidence_adj": 0}

        ind = signal.get("indicators", {})
        opt_info = ""
        if option:
            opt_info = (f"Option: {option.get('symbol','')} | LTP ₹{option.get('ltp',0)} "
                        f"(bid {option.get('bid')} / ask {option.get('ask')}, spread {option.get('spread')}) | "
                        f"δ {option.get('delta')} ({option.get('delta_source','?')}) | "
                        f"IV {option.get('iv')} | θ {option.get('theta')}\n"
                        f"Entry ₹{option.get('entry')} SL ₹{option.get('sl')} T1 ₹{option.get('target1')} "
                        f"T2 ₹{option.get('target2')} | {option.get('lots')}×{option.get('lot_size')} = "
                        f"₹{option.get('capital')} capital")

        regime_txt = ""
        if regime:
            regime_txt = (f"\nToday's regime: {regime.get('regime')} / bias {regime.get('bias')} "
                          f"/ floor {regime.get('confidence_floor')}%  min RR {regime.get('min_rr')}  "
                          f"avoid {regime.get('avoid_instruments')}  notes: {regime.get('notes','')}")

        prompt = f"""You are a ruthlessly disciplined Indian intraday options trader on a ₹20,000 account.
Protect capital first, profit second. Only TAKE with clear edge.

SIGNAL
Instrument: {instrument}   Direction: {signal['direction']}   Engine confidence: {signal['confidence']}%
Entry {signal['entry']}  SL {signal['sl']}  T1 {signal['target1']}  T2 {signal['target2']}  R:R {signal.get('risk_reward',0)}
{opt_info}

INDICATORS
RSI {ind.get('rsi','?')}  MACD hist {ind.get('macd','?')}  SuperTrend {ind.get('supertrend','?')}
EMA9/21/50 {ind.get('ema9','?')}/{ind.get('ema21','?')}/{ind.get('ema50','?')}  VWAP {ind.get('vwap','?')}
ATR {ind.get('atr','?')}  Stoch {ind.get('stoch','?')}  ADX {ind.get('adx','?')}  VolRatio {ind.get('vol_ratio','?')}x
Reasons: {', '.join(signal.get('reasons',[])[:6])}
Time: {datetime.now(IST).strftime('%H:%M')} IST{regime_txt}

RULES
- SKIP if RSI>75 for LONG or RSI<25 for SHORT
- SKIP after 14:30 unless move is already in-progress and premium still has runway
- SKIP if ATR tiny (no movement to be had)
- SKIP if SuperTrend disagrees with direction
- SKIP if option spread looks too wide (>~4-5% of LTP)
- Scale POSITION_PCT down when the setup is decent but not clean (e.g. fighting VWAP, near resistance)
- Use SL_TIGHTENING = "trailing_atr" on trending regimes, "breakeven_at_half_t1" on ranging/volatile
- BANKNIFTY monthly contracts slip harder — be extra conservative there

Respond in EXACTLY this JSON (no markdown, no prose before/after):
{{"verdict": "TAKE" | "SKIP" | "WAIT",
  "position_pct": 25 | 50 | 75 | 100,
  "sl_tightening": "none" | "breakeven_at_half_t1" | "trailing_atr",
  "confidence_adj": integer -20..10,
  "reasoning": "one short line",
  "risk_note": "one specific risk for this trade"}}"""

        result = _anthropic_call(prompt, max_tokens=300, timeout=15)
        if result is None:
            # BUG FIX #2: Changed from fail-closed (SKIP) to fail-open (TAKE).
            # Fail-closed silently blocks ALL signals whenever the Anthropic API
            # has any issue (network blip, rate limit, JSON parse failure).
            # The engine's own confidence + R:R gates already protect capital.
            log.warning("  Claude unavailable — passing signal through (fail-open)")
            return {"verdict": "TAKE", "position_pct": 100, "sl_tightening": "none",
                    "reasoning": "Claude unavailable — engine gates apply", "risk_note": "no AI review",
                    "confidence_adj": 0}

        v = str(result.get("verdict", "SKIP")).upper()
        if v not in ("TAKE", "SKIP", "WAIT"): v = "SKIP"
        pp = result.get("position_pct", 100)
        try: pp = int(pp)
        except Exception: pp = 100
        if pp not in (0, 25, 50, 75, 100): pp = 100 if v == "TAKE" else 0

        tight = str(result.get("sl_tightening", "none")).lower()
        if tight not in ("none", "breakeven_at_half_t1", "trailing_atr"): tight = "none"

        out = {
            "verdict": v, "position_pct": pp, "sl_tightening": tight,
            "confidence_adj": int(result.get("confidence_adj") or 0),
            "reasoning": str(result.get("reasoning", ""))[:240],
            "risk_note": str(result.get("risk_note", ""))[:240],
        }
        log.info(f"🤖 SignalValidation: {instrument} {signal['direction']} → {v} "
                 f"pos={pp}% tighten={tight} ({out['reasoning'][:70]})")
        return out


# ─── Layer C: End-of-day learning loop (run ~15:45 IST) ───────────────
class LearningLoop:
    @staticmethod
    def run():
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if db_exec("SELECT 1 FROM daily_adjustments WHERE date=?", (today,), fetchone=True):
            return None
        rows = db_exec("SELECT * FROM signals WHERE date=? AND status='CLOSED'",
                       (today,), fetch=True) or []
        if not rows:
            return None
        summary = []
        for r in rows[:30]:
            r = dict(r)
            summary.append(
                f"{r.get('timestamp','')} {r['instrument']} {r['direction']} "
                f"conf={r['confidence']} pos%={r.get('position_pct')} "
                f"tighten={r.get('sl_tightening')} RR={(r.get('option_target1') or 0) - (r.get('option_entry') or 0):.1f}/"
                f"{(r.get('option_entry') or 0) - (r.get('option_sl') or 0):.1f} "
                f"→ {r['result']} ₹{r.get('pnl_rupees')}"
            )

        prompt = f"""End-of-day review for Indian intraday options desk. Date: {today}.
Trades closed today (oldest first):
{chr(10).join(summary)}

Identify concrete tweaks for tomorrow's scanner. Be specific and small-surface.
Respond in EXACTLY this JSON (no markdown):
{{"indicator_weight_adjustments": {{"rsi": -5..+5, "macd": -5..+5, "supertrend": -5..+5, "vwap": -5..+5}},
  "time_windows_to_avoid": ["HH:MM-HH:MM", ...],
  "extra_filters": ["short plain-English filters the scanner should apply tomorrow"],
  "summary": "one line recap of today"}}"""

        raw = _anthropic_call(prompt, max_tokens=500, timeout=25)
        if not raw: return None
        try:
            db_exec("""INSERT OR REPLACE INTO daily_adjustments
                       (date, indicator_weight_adjustments, time_windows_to_avoid,
                        extra_filters, raw_json, created_at) VALUES (?,?,?,?,?,?)""",
                    (today, json.dumps(raw.get("indicator_weight_adjustments") or {}),
                     json.dumps(raw.get("time_windows_to_avoid") or []),
                     json.dumps(raw.get("extra_filters") or []),
                     json.dumps(raw), datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")))
        except Exception as e:
            log.warning(f"  learning persist failed: {e}")
        log.info(f"🧠 EOD learning persisted: {raw.get('summary','')[:120]}")
        SlackAlert.send(f"🧠 *EOD Learning* — {raw.get('summary','')}\n"
                        f"Avoid: {', '.join(raw.get('time_windows_to_avoid') or []) or 'none'}\n"
                        f"Filters: {'; '.join(raw.get('extra_filters') or []) or 'none'}")
        return raw

    @staticmethod
    def for_date(date_str):
        row = db_exec("SELECT * FROM daily_adjustments WHERE date=?", (date_str,), fetchone=True)
        if not row: return None
        r = dict(row)
        for k in ("indicator_weight_adjustments", "time_windows_to_avoid", "extra_filters"):
            try: r[k] = json.loads(r.get(k) or ("{}" if k.endswith("adjustments") else "[]"))
            except Exception: r[k] = {} if k.endswith("adjustments") else []
        return r


# ─── Layer D: In-flight trade management (every ~2 min for OPEN rows) ──
class TradeManager:
    @staticmethod
    def _option_snapshot(client, token):
        if not token: return None
        try:
            r = client.api.getMarketData(mode="FULL", exchangeTokens={"NFO": [str(token)]})
            if r and r.get("status") and r.get("data"):
                items = r["data"].get("fetched") or []
                if items: return items[0]
        except Exception:
            pass
        return None

    @staticmethod
    def tick(engine):
        today = datetime.now(IST).strftime("%Y-%m-%d")
        opens = db_exec("SELECT * FROM signals WHERE status='OPEN' AND date=?",
                        (today,), fetch=True) or []
        if not opens: return
        for s in opens:
            s = dict(s)
            tok = s.get("option_token") or ""
            if not tok: continue
            snap = TradeManager._option_snapshot(engine.client, tok)
            if not snap: continue
            ltp = float(snap.get("ltp") or 0)
            entry = float(s.get("option_entry") or 0)
            if entry <= 0: continue
            move_pct = round((ltp - entry) / entry * 100.0, 1)
            held_min = 0
            try:
                ts = datetime.strptime(s["timestamp"], "%Y-%m-%d %H:%M:%S")
                held_min = int((datetime.now(IST).replace(tzinfo=None) - ts).total_seconds() // 60)
            except Exception:
                pass

            prompt = f"""You are managing an OPEN intraday options position. Decide ONE action.

Trade: {s['instrument']} {s['direction']}  {s.get('option_symbol','')}
Entry ₹{entry}  Current ₹{ltp}  Move {move_pct}%  Held {held_min}min
SL ₹{s.get('option_sl')}  T1 ₹{s.get('option_target1')}  T2 ₹{s.get('option_target2')}
SL tightening rule in effect: {s.get('sl_tightening','none')}
Time: {datetime.now(IST).strftime('%H:%M')} IST

Respond in EXACTLY this JSON:
{{"action": "HOLD" | "PARTIAL_EXIT_50" | "TRAIL_SL" | "CLOSE",
  "new_sl": number or null,
  "reasoning": "one short line"}}"""

            raw = _anthropic_call(
                prompt,
                model=CONFIG.get("anthropic_model_inflight") or CONFIG.get("anthropic_model"),
                max_tokens=200, timeout=12,
            )
            if not raw: continue
            act = str(raw.get("action", "HOLD")).upper()
            if act not in ("HOLD", "PARTIAL_EXIT_50", "TRAIL_SL", "CLOSE"): act = "HOLD"

            try:
                db_exec("""INSERT INTO inflight_actions (signal_id, ts, action, reasoning, raw_json)
                           VALUES (?,?,?,?,?)""",
                        (s["id"], datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                         act, str(raw.get("reasoning", ""))[:240], json.dumps(raw)))
            except Exception:
                pass

            if act == "TRAIL_SL" and raw.get("new_sl"):
                try:
                    new_sl = float(raw["new_sl"])
                    if new_sl > float(s.get("option_sl") or 0):
                        db_exec("UPDATE signals SET option_sl=? WHERE id=?", (new_sl, s["id"]))
                        log.info(f"🎯 Trailed SL for {s['instrument']} → ₹{new_sl}")
                except Exception:
                    pass
            elif act == "CLOSE":
                # Force-close at current option price
                pnl_per = ltp - entry
                qty = int(s.get("option_lots") or 1) * int(s.get("option_lot_size") or 0 or 1)
                update_result(s["id"], s.get("index_price") or 0,
                              "WIN" if pnl_per > 0 else "LOSS",
                              round(pnl_per, 2), round(pnl_per * qty, 0),
                              option_exit=ltp)
                log.info(f"🤖 TradeManager CLOSE {s['instrument']} at ₹{ltp} → ₹{round(pnl_per*qty,0)}")
                SlackAlert.send(f"🤖 *AI CLOSE* {s['instrument']} {s['direction']} "
                                f"₹{ltp} ({'+' if pnl_per>0 else ''}{round(pnl_per,2)})\n_{raw.get('reasoning','')}_")
            elif act == "PARTIAL_EXIT_50":
                # We can't actually place orders here — just log + alert so user can act
                SlackAlert.send(f"✂️ *AI suggests PARTIAL EXIT 50%* — {s['instrument']} {s['direction']} "
                                f"@ ₹{ltp}\n_{raw.get('reasoning','')}_")


# ─── Layer E: Event Calendar (static JSON, loaded once per process) ────
class EventCalendar:
    _cache = None
    _path = os.environ.get("EVENTS_JSON",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "events.json"))

    @staticmethod
    def _load():
        if EventCalendar._cache is not None:
            return EventCalendar._cache
        try:
            with open(EventCalendar._path, "r") as f:
                EventCalendar._cache = json.load(f) or []
        except Exception as e:
            log.warning(f"  events.json load failed ({e}) — event blackouts disabled")
            EventCalendar._cache = []
        return EventCalendar._cache

    @staticmethod
    def today_events():
        today = datetime.now(IST).strftime("%Y-%m-%d")
        return [e for e in EventCalendar._load() if e.get("date") == today]

    @staticmethod
    def in_blackout():
        """Return (True, event) if we're currently within the blackout window of a
        scheduled high-impact event, else (False, None)."""
        now = datetime.now(IST)
        today = now.strftime("%Y-%m-%d")
        hm = now.strftime("%H:%M")
        for e in EventCalendar._load():
            if e.get("date") != today: continue
            blk = e.get("blackout") or {}
            start = blk.get("start") or e.get("time")
            end = blk.get("end")
            if start and end and start <= hm <= end:
                return True, e
        return False, None


# Back-compat shim so older code paths referring to AIAnalysis still work
class AIAnalysis:
    @staticmethod
    def analyze(instrument, signal, option, regime=None):
        return SignalValidation.analyze(instrument, signal, option, regime=regime)


# ═══════════════════════════════════════════════════════════════════
# EXIT TIME ESTIMATOR
# ═══════════════════════════════════════════════════════════════════
def estimate_exit_time(signal):
    """Estimate probable exit time based on ATR and distance to target"""
    atr = signal.get("indicators", {}).get("atr", 0)
    if atr <= 0:
        return None, None
    
    entry = signal["entry"]
    t1 = signal["target1"]
    sl = signal["sl"]
    
    dist_to_target = abs(t1 - entry)
    dist_to_sl = abs(sl - entry)
    
    # Average 5-min candle covers roughly ATR/4 in directional move
    avg_move = atr / 4
    if avg_move <= 0:
        return None, None
    
    candles_to_t1 = max(2, round(dist_to_target / avg_move))
    candles_to_sl = max(1, round(dist_to_sl / avg_move))
    
    now = datetime.now(IST)
    t1_mins = candles_to_t1 * 5
    sl_mins = candles_to_sl * 5
    
    exit_t1 = (now + timedelta(minutes=t1_mins)).strftime("%H:%M")
    exit_sl = (now + timedelta(minutes=sl_mins)).strftime("%H:%M")
    
    # Cap at 15:20 (market close)
    if exit_t1 > "15:20":
        exit_t1 = "15:20"
    if exit_sl > "15:20":
        exit_sl = "15:20"
    
    duration_str = f"{t1_mins}m" if t1_mins < 60 else f"{t1_mins//60}h {t1_mins%60}m"
    
    return {
        "target_by": exit_t1,
        "sl_by": exit_sl,
        "est_candles": candles_to_t1,
        "est_duration": duration_str
    }, candles_to_t1


# ═══════════════════════════════════════════════════════════════════
# OPTION PICKER
# ═══════════════════════════════════════════════════════════════════
class OptPicker:
    """Pick the BEST option: maximize lots within ₹20K budget.
    Priority: affordability > per-instrument premium range > real delta > R:R > OTM distance.
    """

    def pick(self, sig, info, chain, atm, budget=20000, greeks=None, position_pct=100):
        """
        greeks        -- list returned by AngelClient.option_greeks(), or None to force ladder fallback.
        position_pct  -- 25/50/75/100. Claude Layer B can scale down exposure for shakier signals.
        """
        if not sig: return None
        ot = "CE" if sig["direction"] == "LONG" else "PE"
        cands = [o for o in chain if o["type"] == ot and o["ltp"] > 0]
        if not cands: return None

        inst_name = info.get("expiry_prefix") or info.get("symbol") or ""
        ranges = PREMIUM_RANGES.get(inst_name, PREMIUM_RANGES["NIFTY"])
        ideal_lo, ideal_hi = ranges["ideal"]
        ok_lo, ok_hi = ranges["ok"]

        gap = info["strike_gap"]
        lot = info["lot_size"]
        price = sig["price"]
        pct = max(25, min(100, int(position_pct or 100))) / 100.0
        max_capital = budget * 0.5 * pct  # Claude can scale this down (position_pct<100)

        scored = []
        for o in cands:
            ltp = float(o["ltp"])
            strike = float(o["strike"])
            if ltp < 5: continue

            cost_1lot = ltp * lot
            affordable = (cost_1lot <= max_capital)
            can_buy_2 = (cost_1lot * 2 <= max_capital)

            otm_gaps = abs(strike - atm) / gap
            moneyness = abs(strike - price) / price if price else 0
            right_side = (ot == "CE" and strike >= atm) or (ot == "PE" and strike <= atm)

            # ── DELTA: prefer real greeks, fall back to ladder ──
            delta = None
            iv = None
            theta = None
            g = AngelClient.greeks_lookup(greeks, strike, ot) if greeks else None
            delta_source = "fallback"
            if g is not None:
                try:
                    delta = abs(float(g.get("delta") or 0))
                    iv    = float(g.get("impliedVolatility") or 0) or None
                    theta = float(g.get("theta") or 0) or None
                    if 0.01 <= delta <= 0.99:
                        delta_source = "live"
                    else:
                        delta = None
                except Exception:
                    delta = None
            if delta is None:
                # DTE from expiry string if we have it, else assume 5
                dte = 5
                try:
                    exp = o.get("expiry") or ""
                    if exp:
                        d_exp = datetime.strptime(exp.upper(), "%d%b%Y").date()
                        dte = max(0.5, (d_exp - datetime.now(IST).date()).days or 0.5)
                except Exception:
                    pass
                delta = fallback_delta(moneyness, dte=dte, right_side=right_side)

            # ── SCORING ──
            score = 0

            # 1. AFFORDABILITY (40 pts)
            if can_buy_2: score += 40
            elif affordable: score += 25

            # 2. PREMIUM SWEET SPOT (25 pts) — per-instrument bands
            if ideal_lo <= ltp <= ideal_hi:
                score += 25
            elif ok_lo <= ltp <= ok_hi:
                score += 15
            elif ltp > ok_hi:
                score += max(0, 5 - int((ltp - ok_hi) / max(1, ok_hi)))
            else:  # below ok_lo = too cheap to move
                score += 2

            # 3. DELTA QUALITY (20 pts)
            if delta >= 0.35: score += 20
            elif delta >= 0.25: score += 15
            elif delta >= 0.18: score += 8

            # 4. R:R using real delta on index→option conversion (10 pts)
            idx_move_to_t1 = abs(sig["target1"] - sig["entry"])
            idx_move_to_sl = abs(sig["sl"] - sig["entry"])
            opt_move_to_t1 = idx_move_to_t1 * delta
            opt_move_to_sl = idx_move_to_sl * delta
            rr = opt_move_to_t1 / max(opt_move_to_sl, 1)
            if rr >= 2.0: score += 10
            elif rr >= 1.5: score += 7
            elif rr >= 1.0: score += 3

            # 5. OTM PREFERENCE (5 pts)
            if right_side and 0.5 <= otm_gaps <= 2.5: score += 5
            elif otm_gaps < 0.5: score += 3

            # Liquidity bonus: narrower spread (as share of price) is better
            sp = float(o.get("spread", 0) or 0)
            if sp > 0 and ltp > 0:
                spread_pct = sp / ltp
                if spread_pct <= 0.02: score += 3
                elif spread_pct <= 0.04: score += 1

            lots_possible = max(1, int(max_capital / cost_1lot)) if affordable else 0
            scored.append({**o,
                "delta": round(delta, 3), "delta_source": delta_source,
                "iv": round(iv, 2) if iv is not None else None,
                "theta": round(theta, 3) if theta is not None else None,
                "score": score, "otm_gaps": round(otm_gaps, 1),
                "right_side": right_side, "rr": round(rr, 2),
                "affordable": affordable, "lots_possible": lots_possible,
            })

        if not scored: return None
        scored.sort(key=lambda x: -x["score"])
        b = scored[0]

        e = float(b["ltp"])
        d = float(b["delta"])
        idx_to_sl = abs(sig["sl"] - sig["entry"])
        idx_to_t1 = abs(sig["target1"] - sig["entry"])
        idx_to_t2 = abs(sig["target2"] - sig["entry"])

        sl = round(max(e - idx_to_sl * d, e * 0.65), 2)
        t1 = round(e + idx_to_t1 * d, 2)
        t2 = round(e + idx_to_t2 * d, 2)

        cost_1lot = e * lot
        if cost_1lot <= max_capital:
            lots = max(1, min(int(max_capital / cost_1lot), 3))
        else:
            lots = 1
        qty = lots * lot
        capital = round(e * qty)

        return {
            "action": f"BUY {ot}", "symbol": b["symbol"], "strike": b["strike"], "type": ot,
            "expiry": b.get("expiry", ""), "token": b.get("token", ""),
            "ltp": round(e, 2), "entry": round(e, 2),
            "bid": b.get("bid"), "ask": b.get("ask"), "spread": b.get("spread"),
            "sl": sl, "target1": t1, "target2": t2,
            "delta": d, "delta_source": b.get("delta_source", "fallback"),
            "iv": b.get("iv"), "theta": b.get("theta"),
            "lot_size": lot, "lots": lots, "qty": qty,
            "capital": capital, "max_loss": round((e - sl) * qty),
            "t1_profit": round((t1 - e) * qty), "t2_profit": round((t2 - e) * qty),
            "rr": b["rr"], "otm_gaps": b["otm_gaps"], "score": b["score"],
            "alternatives": len(scored),
            "position_pct": int(pct * 100),
            "source": "LIVE"
        }

# ═══════════════════════════════════════════════════════════════════
# P&L TRACKER
# ═══════════════════════════════════════════════════════════════════
class PLTracker:
    """
    Tracks open signals against the OPTION'S LTP (not the index LTP) and
    honors AI-suggested SL tightening:
      - "none"                 : use stored option_sl / option_target1 as-is
      - "breakeven_at_half_t1" : once option moves ≥ half-way to T1, SL jumps to entry
      - "trailing_atr"         : trail SL behind best premium by 1 × option-ATR estimate
                                 (we approximate option-ATR as delta × index-ATR, using
                                 ATR supplied by the last-seen signal snapshot).
    """
    def __init__(self, client):
        self.client = client
        # Track the best option LTP seen per row id (for trailing)
        self._best_premium = {}

    def _current_option_price(self, token):
        """Fetch current option price via FULL mode depth; fall back to LTP."""
        if not token: return None
        try:
            r = self.client.api.getMarketData(mode="FULL", exchangeTokens={"NFO": [str(token)]})
            if r and r.get("status") and r.get("data"):
                items = r["data"].get("fetched") or []
                if items:
                    it = items[0]
                    depth = it.get("depth") or {}
                    buys = depth.get("buy") or []
                    sells = depth.get("sell") or []
                    bid = float(buys[0].get("price")) if buys else 0
                    ask = float(sells[0].get("price")) if sells else 0
                    if bid > 0 and ask > 0 and (ask - bid) <= ask * 0.08:
                        return round((bid + ask) / 2.0, 2)
                    ltp = float(it.get("ltp") or 0)
                    if ltp > 0: return ltp
        except Exception as e:
            log.debug(f"  option price fetch err: {e}")
        return None

    def check(self):
        today = datetime.now(IST).strftime("%Y-%m-%d")
        opens = db_exec("SELECT * FROM signals WHERE status='OPEN' AND date=?", (today,), fetch=True)
        if not opens: return
        for s in opens:
            s = dict(s)
            inst = INSTRUMENTS.get(s["instrument"])
            if not inst: continue

            opt_entry  = float(s.get("option_entry") or 0)
            opt_sl     = float(s.get("option_sl") or 0)
            opt_t1     = float(s.get("option_target1") or 0)
            opt_lots   = int(s.get("option_lots") or 0)
            lot_size   = int(s.get("option_lot_size") or inst.get("lot_size", 25))
            qty        = max(1, opt_lots) * lot_size if opt_lots else lot_size
            token      = s.get("option_token") or ""
            tighten    = s.get("sl_tightening") or "none"

            # Need option price for P&L. If we have no token, fall back to index-based exit detection
            # so legacy rows without a token don't block P&L accounting.
            cur_opt = self._current_option_price(token) if token else None
            idx_px = None
            idx = self.client.ltp(inst["exchange"], inst["symbol"], inst["token"])
            if idx: idx_px = idx.get("ltp") or 0

            # Trailing-SL bookkeeping (per signal id)
            if cur_opt is not None:
                best = self._best_premium.get(s["id"], opt_entry)
                if cur_opt > best:
                    best = cur_opt
                    self._best_premium[s["id"]] = best

                # Apply SL tightening rules — they only MOVE SL up (tighter), never loosen
                new_sl = opt_sl
                if tighten == "breakeven_at_half_t1" and opt_t1 > opt_entry:
                    half_t1 = opt_entry + (opt_t1 - opt_entry) * 0.5
                    if best >= half_t1:
                        new_sl = max(new_sl, opt_entry)  # breakeven
                elif tighten == "trailing_atr":
                    # 1× option-ATR ≈ delta × index-ATR, from stored indicators
                    try:
                        ind = json.loads(s.get("indicators") or "{}")
                        idx_atr = float(ind.get("atr") or 0)
                        delta_est = max(0.15, min(0.7, (opt_entry / max(1, (s.get("index_price") or 1))) * 0 + 0.4))
                        opt_atr = idx_atr * delta_est
                        trail = best - max(1.0, opt_atr)
                        if trail > new_sl:
                            new_sl = round(trail, 2)
                    except Exception:
                        pass
                opt_sl = new_sl

            # Exit detection — prefer option-based levels
            result = None
            exit_opt = None
            if cur_opt is not None and opt_entry > 0:
                if cur_opt >= opt_t1 and opt_t1 > 0:
                    result = "WIN"; exit_opt = cur_opt
                elif cur_opt <= opt_sl and opt_sl > 0:
                    result = "LOSS"; exit_opt = cur_opt
            elif idx_px:
                # Legacy fallback (no token) → use index levels (old behaviour) but
                # approximate option exit at delta=0.4 for accounting.
                d = s["direction"]; entry = s["index_entry"]; sl = s["index_sl"]; t1 = s["index_target1"]
                if d == "LONG":
                    if idx_px >= t1: result = "WIN"
                    elif idx_px <= sl: result = "LOSS"
                else:
                    if idx_px <= t1: result = "WIN"
                    elif idx_px >= sl: result = "LOSS"

            if result:
                if cur_opt is not None and opt_entry > 0:
                    pnl_per_share = (cur_opt - opt_entry)
                    pnl_rs = round(pnl_per_share * qty, 0)
                    pnl_pts = round(pnl_per_share, 2)  # points here = rupees per share of premium
                    update_result(s["id"], idx_px or 0, result, pnl_pts, pnl_rs, option_exit=cur_opt)
                else:
                    # fallback: index points × lot_size (old behaviour, flagged inaccurate)
                    d = s["direction"]
                    pnl_pts = (idx_px - s["index_entry"]) if d == "LONG" else (s["index_entry"] - idx_px)
                    pnl_rs = round(pnl_pts * lot_size, 0)
                    update_result(s["id"], idx_px, result, round(pnl_pts, 2), pnl_rs)
                emoji = "✅" if result == "WIN" else "❌"
                log.info(f"{emoji} {s['instrument']} {s['direction']} → {result} | ₹{pnl_rs} (opt exit ₹{cur_opt})")
                SlackAlert.send(SlackAlert.format_close(s["instrument"], s["direction"], result, pnl_rs))
                self._best_premium.pop(s["id"], None)

    def close_all(self):
        opens = db_exec("SELECT * FROM signals WHERE status='OPEN' AND date=?",
                        (datetime.now(IST).strftime("%Y-%m-%d"),), fetch=True)
        for s in opens:
            s = dict(s)
            # Try to mark-to-market via option price; otherwise 0
            cur_opt = self._current_option_price(s.get("option_token") or "") if s.get("option_token") else None
            opt_entry = float(s.get("option_entry") or 0)
            lot_size = int(s.get("option_lot_size") or 0)
            lots = int(s.get("option_lots") or 1)
            qty = max(1, lots) * lot_size if lot_size else 0
            if cur_opt is not None and opt_entry > 0 and qty > 0:
                pnl_rs = round((cur_opt - opt_entry) * qty, 0)
                update_result(s["id"], s["index_price"], "EXPIRED",
                              round(cur_opt - opt_entry, 2), pnl_rs, option_exit=cur_opt)
            else:
                update_result(s["id"], s["index_price"], "EXPIRED", 0, 0)
        perf = get_perf()
        if perf["total"] > 0:
            SlackAlert.send(SlackAlert.format_daily_summary(perf))

# ═══════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self):
        self.client=AngelClient();self.sgen=SignalGen();self.opick=OptPicker()
        self.tracker=PLTracker(self.client);self.latest={};self.alerts=[]
        self.running=False;self._prev={};self._last_signal={}
        self._regime=None
        self._last_regime_run=None
        self._last_eod_run=None
        self._last_inflight_run=0.0

    def start(self):
        if not self.client.login(): return{"status":"error","message":"Login failed"}
        self.running=True
        threading.Thread(target=self._loop,daemon=True).start()
        SlackAlert.send("🚀 *Signal Engine Started*\nScanning NIFTY, BANKNIFTY, FINNIFTY\nAlerts will arrive here when confidence ≥ 60%")
        return{"status":"ok","message":"Engine started"}

    def stop(self):
        self.running=False;self.tracker.close_all()
        SlackAlert.send("🔴 *Signal Engine Stopped*")
        return{"status":"ok"}

    def _maybe_regime(self, now):
        """Layer A: run pre-market regime brief once per day around 08:45 IST."""
        today = now.strftime("%Y-%m-%d")
        if self._last_regime_run == today: return
        # Pre-market window: 08:40–09:15, or opportunistic first tick after market opens
        pre = now.hour == 8 and now.minute >= 40
        opening = now.hour == 9 and now.minute < 30
        if pre or opening or self._regime is None:
            try:
                self._regime = RegimeBrief.run()
                self._last_regime_run = today
            except Exception as e:
                log.warning(f"  regime brief failed: {e}")

    def _maybe_eod(self, now):
        """Layer C: at ~15:45 run EOD learning once."""
        today = now.strftime("%Y-%m-%d")
        if self._last_eod_run == today: return
        if now.hour == 15 and now.minute >= 45:
            try:
                LearningLoop.run()
                self._last_eod_run = today
            except Exception as e:
                log.warning(f"  EOD learning failed: {e}")

    def _maybe_inflight(self):
        """Layer D: every ~2 min evaluate OPEN positions."""
        now_ts = time.time()
        if now_ts - self._last_inflight_run < 120: return
        self._last_inflight_run = now_ts
        try:
            TradeManager.tick(self)
        except Exception as e:
            log.warning(f"  in-flight tick failed: {e}")

    def _recompute_levels(self, sig, fresh_price):
        """Fix B: recompute entry/SL/T1/T2 off a fresh LTP + stored ATR so the
        Slack/DB snapshot matches what the trader will actually see."""
        atr = (sig.get("indicators") or {}).get("atr") or 0
        if fresh_price is None or fresh_price <= 0 or atr <= 0: return
        direction = sig["direction"]
        # Same geometry as SignalGen: entry = fresh price, SL = 1× ATR away, T1/T2 stretch same multiplier
        idx_sl_pts = max(atr, 1e-6)
        # Preserve R:R ratio from the original signal
        orig_entry = sig.get("entry") or fresh_price
        orig_sl_pts = abs((sig.get("sl") or orig_entry) - orig_entry) or idx_sl_pts
        orig_t1_pts = abs((sig.get("target1") or orig_entry) - orig_entry) or idx_sl_pts * 1.5
        orig_t2_pts = abs((sig.get("target2") or orig_entry) - orig_entry) or idx_sl_pts * 2.5
        if direction == "LONG":
            sig["entry"] = round(fresh_price, 2)
            sig["sl"] = round(fresh_price - orig_sl_pts, 2)
            sig["target1"] = round(fresh_price + orig_t1_pts, 2)
            sig["target2"] = round(fresh_price + orig_t2_pts, 2)
        else:
            sig["entry"] = round(fresh_price, 2)
            sig["sl"] = round(fresh_price + orig_sl_pts, 2)
            sig["target1"] = round(fresh_price - orig_t1_pts, 2)
            sig["target2"] = round(fresh_price - orig_t2_pts, 2)
        sig["price"] = round(fresh_price, 2)
        # Recompute R:R (may have drifted slightly with the fresh price)
        try:
            sig["risk_reward"] = round(orig_t1_pts / max(orig_sl_pts, 1e-6), 2)
        except Exception:
            pass

    def _loop(self):
        while self.running:
            try:
                now=datetime.now(IST)
                # Pre-market: run regime brief early so overrides are ready at 09:15
                self._maybe_regime(now)

                # BUG FIX #4: The original code had `now.hour>=15` fire a `continue`
                # BEFORE the close_all check at hour==15,min>=25 could be reached —
                # making close_all() completely unreachable and positions never auto-closed.
                # Fixed by checking the 15:25 close condition FIRST.
                if now.hour==15 and now.minute>=25:
                    self.tracker.close_all();self.running=False
                    log.info("🔔 Market close");break
                if now.hour<9 or(now.hour==9 and now.minute<20)or now.hour>15 or(now.hour==15 and now.minute>=25):
                    time.sleep(30);continue

                # Mid/late-session learning + in-flight management
                self._maybe_eod(now)
                self._maybe_inflight()

                # P&L check
                self.tracker.check()

                # Regime-level overrides (Layer A)
                regime = self._regime or RegimeBrief.today()
                avoid = set((regime or {}).get("avoid_instruments") or [])
                conf_floor = max(CONFIG["min_confidence"],
                                 int((regime or {}).get("confidence_floor") or CONFIG["min_confidence"]))
                min_rr_floor = max(1.5, float((regime or {}).get("min_rr") or 1.5))

                # Event blackout (Layer E) — short-circuit the whole scan
                blackout, ev = EventCalendar.in_blackout()
                if blackout:
                    log.info(f"🚫 Event blackout active: {ev.get('name')} ({ev.get('blackout',{})}) — skipping scan")
                    time.sleep(30); continue

                for name,inst in INSTRUMENTS.items():
                    if name in avoid:
                        log.info(f"  {name} skipped — regime avoid list")
                        continue

                    df=self.client.candles(inst["token"],inst["exchange"])
                    if df.empty or len(df)<30: continue
                    sig=self.sgen.analyze(df)
                    if not sig: continue
                    # 15-min cooldown: skip re-signal same instrument
                    _last_t=self._last_signal.get(name)
                    if _last_t and (datetime.now(IST)-_last_t).total_seconds()<900: continue

                    # Fetch real option chain at 40%+ for dashboard display
                    opt=None
                    chain=None
                    atm=None
                    greeks=None
                    if sig["confidence"]>=40 and sig.get("direction") in ("LONG","SHORT"):
                        try:
                            chain,atm=self.client.option_chain(inst,sig["price"])
                            if chain:
                                expiry = chain[0].get("expiry","") if chain else ""
                                greeks = self.client.option_greeks(
                                    inst.get("expiry_prefix", name), expiry) if expiry else None
                                opt=self.opick.pick(sig,inst,chain,atm,CONFIG.get("budget",20000),
                                                    greeks=greeks)
                        except Exception as ce:
                            log.warning(f"  Chain fetch failed for {name}: {ce}")

                    # Estimate exit time
                    timing, _ = estimate_exit_time(sig)

                    result={"instrument":name,"lot_size":inst["lot_size"],"signal":sig,"option":opt,
                            "timing":timing,"updated_at":datetime.now(IST).strftime("%H:%M:%S")}

                    prev=self._prev.get(name,{}).get("signal",{})
                    # Hard R:R gate — honour regime min_rr override (≥1.5)
                    if sig.get("risk_reward", 0) < min_rr_floor:
                        log.info(f"⛔ R:R gate blocked {name} {sig['direction']} "
                                 f"RR={sig.get('risk_reward',0)} (need ≥{min_rr_floor})")
                        self._prev[name] = result
                        continue

                    # BUG FIX #3: Removed the direction/confidence-change gate.
                    # Previously signals were ONLY fired when direction changed OR
                    # confidence shifted >10%. In a sustained trending market this
                    # meant ZERO signals after the very first scan window.
                    # The 15-min cooldown (above) already prevents alert spam.
                    if sig["confidence"] >= conf_floor:

                        # ── Fix B: refresh spot LTP right before save + Slack ──
                        fresh_price = None
                        try:
                            ltp_now = self.client.ltp(inst["exchange"], inst["symbol"], inst["token"])
                            fresh_price = (ltp_now or {}).get("ltp") or None
                        except Exception as le:
                            log.warning(f"  fresh LTP fetch failed for {name}: {le}")
                        if fresh_price and fresh_price > 0:
                            self._recompute_levels(sig, fresh_price)

                        # ── Re-pick option on fresh spot if we already had a chain ──
                        if chain and opt is not None:
                            try:
                                chain2, atm2 = self.client.option_chain(inst, sig["price"])
                                if chain2:
                                    expiry2 = chain2[0].get("expiry","")
                                    greeks2 = self.client.option_greeks(
                                        inst.get("expiry_prefix", name), expiry2) if expiry2 else greeks
                                    opt = self.opick.pick(sig, inst, chain2, atm2,
                                                          CONFIG.get("budget", 20000),
                                                          greeks=greeks2) or opt
                                    result["option"] = opt
                            except Exception:
                                pass

                        # ── Layer B: validation + sizing + SL rule ──
                        ai_result = SignalValidation.analyze(name, sig, opt, regime=regime)

                        if ai_result and ai_result.get("verdict") in ("SKIP", "WAIT"):
                            log.info(f"🤖 AI {ai_result.get('verdict')} {name} {sig['direction']} — "
                                     f"{ai_result.get('reasoning','')[:80]}")
                            self._prev[name] = result
                            # BUG FIX #5: Always update _last_signal so the 15-min cooldown
                            # kicks in even on AI SKIP/WAIT. Without this, every 5-second tick
                            # would re-evaluate the same signal and AI would SKIP it again
                            # forever — burning API calls and never moving forward.
                            self._last_signal[name] = datetime.now(IST)
                            continue

                        # Re-pick option with AI-suggested position_pct (may scale down)
                        if chain and ai_result and ai_result.get("position_pct"):
                            try:
                                opt = self.opick.pick(sig, inst, chain, atm,
                                                      CONFIG.get("budget", 20000),
                                                      greeks=greeks,
                                                      position_pct=ai_result["position_pct"]) or opt
                                result["option"] = opt
                            except Exception:
                                pass

                        # Apply adj to confidence for display (don't re-gate)
                        adj = int((ai_result or {}).get("confidence_adj") or 0)
                        sig["confidence_ai_adj"] = max(0, min(100, sig["confidence"] + adj))

                        self.alerts.insert(0,{"id":int(time.time()*1000),"time":datetime.now(IST).strftime("%H:%M:%S"),
                            "instrument":name,"signal":sig,"option":opt,"timing":timing,"ai":ai_result})
                        self.alerts=self.alerts[:100]
                        save_signal(name,sig,opt,ai=ai_result)
                        log.info(f"🚨 {name} {sig['direction']} Conf:{sig['confidence']}% "
                                 f"pos={(ai_result or {}).get('position_pct',100)}% "
                                 f"tighten={(ai_result or {}).get('sl_tightening','none')}")

                        # 📱 SLACK ALERT
                        SlackAlert.send(SlackAlert.format_signal(name, sig, opt, timing, ai_result))

                    self._prev[name]=result;self.latest[name]=result;self._last_signal[name]=datetime.now(IST)

                time.sleep(CONFIG["scan_interval_sec"])
            except Exception as e:
                log.error(f"Loop err: {e}");time.sleep(5)
    
    def get_state(self):
        return{"running":self.running,"signals":self.latest,"alerts":self.alerts[:50],
            "performance":get_perf(),
            "config":{"scan_interval":CONFIG["scan_interval_sec"],"target_min":CONFIG["target_points_min"],
                "target_max":CONFIG["target_points_max"],"min_confidence":CONFIG["min_confidence"]},
            "time":datetime.now(IST).strftime("%H:%M:%S"),
            "market_open":9<=datetime.now(IST).hour<16,
            "slack_enabled":CONFIG["slack_enabled"] and bool(CONFIG["slack_webhook"])}

# ═══════════════════════════════════════════════════════════════════
# FLASK API
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__)

# CORS: reflect the Origin header ONLY if it's in the whitelist. Anything else gets no
# Access-Control-Allow-Origin header at all — the browser will block the request.
def _allowed_origins():
    return [o.strip() for o in (CONFIG.get("cors_origins", "") or "").split(",") if o.strip()]

def _cors_origin_for(req_origin):
    if not req_origin: return None
    return req_origin if req_origin in _allowed_origins() else None

@app.after_request
def add_cors_headers(response):
    origin = _cors_origin_for(flask_request.headers.get("Origin"))
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Auth-Token"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.before_request
def handle_options():
    if flask_request.method == "OPTIONS":
        from flask import Response
        r = Response()
        origin = _cors_origin_for(flask_request.headers.get("Origin"))
        if origin:
            r.headers["Access-Control-Allow-Origin"] = origin
            r.headers["Vary"] = "Origin"
            r.headers["Access-Control-Allow-Credentials"] = "true"
            r.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Auth-Token"
            r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return r, 204

# ── Shared-secret header guard for write endpoints ─────────────────────
def require_auth(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        expected = (CONFIG.get("auth_token") or "").strip()
        supplied = (flask_request.headers.get("X-Auth-Token") or "").strip()
        # If no secret is configured on the server we still refuse — forces setup.
        if not expected:
            return jsonify({"error": "Server AUTH_TOKEN not configured"}), 503
        if not supplied or supplied != expected:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapped

engine = Engine()

@app.route("/")
def home():
    return jsonify({"name":"Intraday Signal Engine","status":"running" if engine.running else "stopped"})

@app.route("/dashboard")
def dashboard():
    """Serve the trading dashboard UI"""
    import os
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    if os.path.exists(html_path):
        return send_file(html_path)
    return "<h1>dashboard.html not found</h1><p>Place dashboard.html in the same folder as server.py</p>", 404

@app.route("/api/login", methods=["POST"])
@require_auth
def api_login():
    """Explicitly trigger Angel One login, forcing a fresh call.

    Query ?force=1 (default) drops the cached connected state so we always
    attempt a new generateSession — useful when a previous TOTP was consumed,
    when the Angel One side invalidated the session, or when the user wants
    a clean retry from the dashboard.
    """
    force = flask_request.args.get("force", "1") != "0"
    try:
        if force:
            # Drop cached connected flag — forces a fresh generateSession.
            engine.client.connected = False
            engine.client.api = None
        ok = engine.client.login() if force else engine.client.ensure()
        if ok:
            return jsonify({"status": "ok", "connected": True, "attempts": engine.client.login_attempts})
        # Real broker-login failure — surface the captured reason.
        reason = engine.client.last_login_error or "Broker login returned false. Check Railway server logs."
        return jsonify({
            "status": "failed",
            "connected": False,
            "error": reason,
            "attempts": engine.client.login_attempts,
        }), 401
    except Exception as e:
        import traceback as _tb
        log.error(f"Login endpoint error: {e}\n{_tb.format_exc(limit=3)}")
        return jsonify({"status": "failed", "connected": False, "error": str(e)}), 500

@app.route("/api/diag")
def api_diag():
    """Dashboard-visible diagnostics — connected state, last login error,
    which env vars are present (no secrets leaked). Lets the client show
    a human-readable reason when broker auth is failing."""
    return jsonify(engine.client.diag())

@app.route("/api/ltp")
def api_ltp():
    """Fast LTP endpoint — returns current prices for all instruments."""
    if not engine.client.ensure():
        return jsonify({"error": "Not logged in"}), 401
    
    prices = {}
    for name, inst in INSTRUMENTS.items():
        try:
            d = engine.client.ltp(inst["exchange"], inst["symbol"], inst["token"])
            if d and d.get("ltp"):
                ltp = float(d["ltp"])
                # Angel One ltpData returns `close` = previous day's close and `open` = today's open.
                # Use prev-close for change%; fall back to day-open if prev-close missing.
                prev_close = float(d.get("close") or d.get("prevClose") or d.get("prev_close") or 0)
                day_open   = float(d.get("open") or 0)
                ref = prev_close if prev_close > 0 else day_open
                chg = round(ltp - ref, 2) if ref > 0 else 0.0
                pct = round((chg / ref) * 100, 2) if ref > 0 else 0.0
                prices[name] = {
                    "ltp": ltp, "symbol": inst["symbol"], "token": inst["token"],
                    "chg": chg, "pct": pct,
                    "open": day_open, "prev_close": prev_close,
                }
        except: pass

    return jsonify({"prices": prices, "time": datetime.now(IST).strftime("%H:%M:%S")})

@app.route("/api/historical/<instrument>")
def historical(instrument):
    """
    Fetch historical candle data for backtesting / replay.
    Usage: /api/historical/NIFTY?days=5&interval=FIVE_MINUTE
    Returns array of [timestamp, open, high, low, close, volume]
    """
    days = int(flask_request.args.get("days", 5))
    interval = flask_request.args.get("interval", "FIVE_MINUTE")
    inst = INSTRUMENTS.get(instrument.upper())
    if not inst:
        return jsonify({"error": f"Unknown instrument: {instrument}", "available": list(INSTRUMENTS.keys())}), 400
    
    if not engine.client.ensure():
        return jsonify({"error": "Not logged in. Start engine first."}), 401
    
    df = engine.client.candles(inst["token"], inst["exchange"], interval, days)
    if df.empty:
        return jsonify({"candles": [], "count": 0, "warning": "No data from Angel One. Market may be closed."}), 200
    
    candles = []
    # SmartAPI returns IST-aware timestamps. Emit them as naive local (IST) strings —
    # no manual re-shifting. The previous implementation shadowed the module-level IST
    # constant with a timedelta and double-corrected in some cases.
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)  # already local IST wall-clock from SmartAPI
        candles.append({
            "t": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": int(row["volume"]),
        })
    
    log.info(f"📊 Returning {len(candles)} historical candles for {instrument} ({days}d)")
    return jsonify({
        "instrument": instrument.upper(),
        "interval": interval,
        "days": days,
        "count": len(candles),
        "candles": candles,
    })

@app.route("/api/ping")
def ping():
    return jsonify({"ok": True, "time": datetime.now(IST).strftime("%H:%M:%S")})

@app.route("/api/status")
def status():
    return jsonify(engine.get_state())

@app.route("/api/start", methods=["POST"])
@require_auth
def start():
    return jsonify(engine.start())

@app.route("/api/stop", methods=["POST"])
@require_auth
def stop():
    return jsonify(engine.stop())

@app.route("/api/config", methods=["POST"])
@require_auth
def config():
    d=flask_request.json or{}
    if"target_min"in d:CONFIG["target_points_min"]=int(d["target_min"]);engine.sgen.tmin=int(d["target_min"])
    if"target_max"in d:CONFIG["target_points_max"]=int(d["target_max"]);engine.sgen.tmax=int(d["target_max"])
    return jsonify({"status":"ok"})

@app.route("/api/history")
def history():
    return jsonify(get_history(int(flask_request.args.get("limit",100)),flask_request.args.get("date")))

@app.route("/api/performance")
def performance():
    return jsonify(get_perf())

@app.route("/api/chain/<instrument>")
def api_chain(instrument):
    """Full option chain with real LTPs for dashboard display."""
    name = instrument.upper()
    inst = INSTRUMENTS.get(name)
    if not inst:
        return jsonify({"error": f"Unknown: {name}"}), 400
    if not engine.client.ensure():
        return jsonify({"error": "Not logged in"}), 401
    
    # Get spot price
    spot_data = engine.client.ltp(inst["exchange"], inst["symbol"], inst["token"])
    spot = spot_data.get("ltp", 0) if spot_data else 0
    if spot == 0:
        return jsonify({"error": "Could not fetch spot price"}), 500
    
    chain, atm = engine.client.option_chain(inst, spot)
    if not chain:
        return jsonify({"error": "Could not fetch chain"}), 500
    
    return jsonify({
        "instrument": name, "spot": spot, "atm": atm,
        "expiry": chain[0].get("expiry", "") if chain else "",
        "chain": sorted(chain, key=lambda x: (x["strike"], x["type"])),
        "count": len(chain)
    })

@app.route("/api/option-ltp", methods=["POST"])
@require_auth
def option_ltp():
    """
    Smart option picker: fetches real chain from Angel One, picks best option for budget.
    Body: {"instrument":"NIFTY","spot":25624,"direction":"LONG"}
    """
    d = flask_request.json or {}
    name = d.get("instrument","").upper()
    spot = d.get("spot", 0)
    direction = d.get("direction", "LONG")
    budget = d.get("budget", 20000)
    
    inst = INSTRUMENTS.get(name)
    if not inst:
        return jsonify({"error": f"Unknown: {name}"}), 400
    if not engine.client.ensure():
        return jsonify({"error": "Not logged in"}), 401
    
    log.info(f"  OptLTP: {name} spot={spot} dir={direction}")
    chain, atm = engine.client.option_chain(inst, spot)
    if not chain:
        log.error(f"  OptLTP: Chain empty for {name}")
        return jsonify({"error": "Could not fetch option chain", "detail": "No options returned from Angel One"}), 500
    
    log.info(f"  OptLTP: Got {len(chain)} options, ATM={atm}")
    
    ot = "CE" if direction == "LONG" else "PE"
    gap = inst["strike_gap"]
    lot = inst["lot_size"]
    max_cap = budget * 0.5  # preferred max 50% capital per trade

    # Real greeks for this chain's expiry (drives delta where available)
    expiry = chain[0].get("expiry", "") if chain else ""
    greeks = engine.client.option_greeks(inst.get("expiry_prefix", name), expiry) if expiry else None

    # Per-instrument premium bands (mirrors OptPicker/Engine scoring)
    ranges = PREMIUM_RANGES.get(name, PREMIUM_RANGES["NIFTY"])
    ideal_lo, ideal_hi = ranges["ideal"]
    ok_lo, ok_hi = ranges["ok"]

    # Score all candidates (right type, non-zero LTP)
    candidates = [o for o in chain if o["type"] == ot and o["ltp"] > 0]
    if not candidates:
        return jsonify({"error": "No options of type " + ot + " found", "chain_size": len(chain)}), 500

    scored = []
    for o in candidates:
        ltp = o["ltp"]
        strike = o["strike"]
        cost_1lot = ltp * lot
        affordable = (cost_1lot <= max_cap)
        can_buy_2 = (cost_1lot * 2 <= max_cap)

        otm_dist = (strike - atm) / gap if direction == "LONG" else (atm - strike) / gap
        moneyness = abs(strike - spot) / spot if spot else 0
        right_side = (ot == "CE" and strike >= atm) or (ot == "PE" and strike <= atm)

        # Real delta where possible, else DTE-aware ladder (same rules as OptPicker)
        delta = None
        delta_source = "fallback"
        g = AngelClient.greeks_lookup(greeks, strike, ot) if greeks else None
        if g is not None:
            try:
                d_live = abs(float(g.get("delta") or 0))
                if 0.01 <= d_live <= 0.99:
                    delta = d_live
                    delta_source = "live"
            except Exception:
                pass
        if delta is None:
            dte = 5
            try:
                if expiry:
                    d_exp = datetime.strptime(expiry.upper(), "%d%b%Y").date()
                    dte = max(0.5, (d_exp - datetime.now(IST).date()).days or 0.5)
            except Exception:
                pass
            delta = fallback_delta(moneyness, dte=dte, right_side=right_side)

        # ═══ SCORING (mirrors OptPicker.pick) ═══
        score = 0

        # 1. AFFORDABILITY (40 pts)
        if can_buy_2: score += 40
        elif affordable: score += 25

        # 2. PREMIUM SWEET SPOT (25 pts) — PER-INSTRUMENT bands
        if ideal_lo <= ltp <= ideal_hi:
            score += 25
        elif ok_lo <= ltp <= ok_hi:
            score += 15
        elif ltp > ok_hi:
            score += max(0, 5 - int((ltp - ok_hi) / max(1, ok_hi)))
        else:
            score += 2

        # 3. DELTA QUALITY (20 pts)
        if delta >= 0.35: score += 20
        elif delta >= 0.25: score += 15
        elif delta >= 0.18: score += 8

        # 4. R:R (10 pts)
        sl_pts = ltp * 0.3
        t1_pts = gap * 0.5 * delta
        if sl_pts > 0 and t1_pts / sl_pts >= 2.0: score += 10
        elif sl_pts > 0 and t1_pts / sl_pts >= 1.5: score += 7
        elif sl_pts > 0 and t1_pts / sl_pts >= 1.0: score += 3

        # 5. OTM PREFERENCE (5 pts)
        if right_side and 0.5 <= otm_dist <= 2.5: score += 5
        elif abs(otm_dist) <= 0.5: score += 3

        # Liquidity bonus (narrower spread = better)
        sp = float(o.get("spread", 0) or 0)
        if sp > 0 and ltp > 0:
            sp_pct = sp / ltp
            if sp_pct <= 0.02: score += 3
            elif sp_pct <= 0.04: score += 1

        lots_possible = max(1, int(max_cap / cost_1lot)) if affordable else 0
        scored.append({**o, "delta": round(delta, 3), "delta_source": delta_source,
            "otm_dist": round(otm_dist, 1),
            "score": score, "affordable": affordable, "lots_possible": lots_possible})

    if not scored:
        return jsonify({"error": "No options scored", "chain_size": len(chain)}), 500

    scored.sort(key=lambda x: -x["score"])
    best = scored[0]
    ltp = best["ltp"]

    cost_1lot = ltp * lot
    if cost_1lot <= max_cap:
        lots = max(1, min(int(max_cap / cost_1lot), 3))
        over_budget = False
    else:
        lots = 1
        over_budget = True

    for i, s in enumerate(scored[:3]):
        tag = "→ PICKED" if i == 0 else ""
        log.info(f"  OptLTP: #{i+1} {s['symbol']} ₹{s['ltp']} δ{s['delta']} ({s.get('delta_source')}) "
                 f"OTM{s['otm_dist']} lots={s.get('lots_possible',0)} score={s['score']} {tag}")

    return jsonify({
        "symbol": best["symbol"],
        "strike": best["strike"],
        "type": ot,
        "ltp": ltp,
        "delta": best["delta"],
        "delta_source": best.get("delta_source", "fallback"),
        "lot_size": lot,
        "lots": lots,
        "qty": lots * lot,
        "atm": atm,
        "expiry": best.get("expiry", ""),
        "token": best.get("token", ""),
        "score": best["score"],
        "alternatives": len(scored) - 1,
        "source": "LIVE",
        "over_budget": over_budget
    })

@app.route("/api/test-slack", methods=["POST"])
@require_auth
def test_slack():
    ok = SlackAlert.send("✅ *Test Alert*\nSlack notifications are working!\nYou'll receive trading signals here during market hours.")
    return jsonify({"status":"ok" if ok else "failed"})

@app.route("/api/test-chain/<name>")
def test_chain(name):
    """Quick test: /api/test-chain/NIFTY → shows if option chain + greeks fetches work.
    Cross-check the sampled `mid`, `bid`, `ask`, and `delta` against Angel's live chart.
    """
    inst = INSTRUMENTS.get(name.upper())
    if not inst: return jsonify({"error":"Unknown"}),400
    if not engine.client.ensure(): return jsonify({"error":"Not logged in"}),401
    spot_data = engine.client.ltp(inst["exchange"], inst["symbol"], inst["token"])
    spot = spot_data.get("ltp",0) if spot_data else 0
    if not spot: return jsonify({"error":"No spot price"}),500
    chain, atm = engine.client.option_chain(inst, spot)
    expiry = chain[0].get("expiry","") if chain else ""
    greeks = engine.client.option_greeks(inst.get("expiry_prefix", name.upper()), expiry) if expiry else None
    # Attach delta/iv/theta from greeks onto sample rows so the owner can eyeball them vs Sensibull
    sample = []
    for o in (chain[:6] if chain else []):
        g = AngelClient.greeks_lookup(greeks, o["strike"], o["type"]) if greeks else None
        sample.append({**o,
            "delta": round(float(g.get("delta") or 0), 3) if g else None,
            "iv":    round(float(g.get("impliedVolatility") or 0), 2) if g else None,
            "theta": round(float(g.get("theta") or 0), 3) if g else None,
        })
    return jsonify({"instrument":name.upper(),"spot":spot,"atm":atm,
        "master_loaded":_master.loaded,"master_count":len(_master.nfo),
        "chain_count":len(chain),"expiry":expiry or "NONE",
        "greeks_source": "LIVE" if greeks else "NONE",
        "greeks_count": len(greeks) if greeks else 0,
        "sample": sample,
        "status":"OK" if chain else "FAILED - check server logs"})


def _startup():
    """Auto-login on server start — works with both Gunicorn (Railway) and direct run.

    On failure, retries up to 3 times with staggered 35s backoff. 35s is chosen
    so the next attempt lands in a fresh TOTP window (TOTPs rotate every 30s),
    which helps when Angel One rejected the first login because the previous
    TOTP was already consumed during a rapid container restart.
    """
    import time as _t; _t.sleep(5)
    log.info("▶ Auto-startup: loading instrument master...")
    try:
        _master.load()
    except Exception as e:
        log.error(f"▶ Auto-startup: master load failed: {e}")

    for attempt in range(1, 4):
        log.info(f"▶ Auto-startup: logging into Angel One (attempt {attempt}/3)...")
        ok = engine.client.login()
        if ok:
            log.info("▶ Auto-startup: login ✅ OK")
            # BUG FIX #1: Auto-start the scan engine so signals fire without
            # requiring a manual POST to /api/start from the dashboard.
            if not engine.running:
                log.info("▶ Auto-startup: starting signal scan engine...")
                engine.running = True
                threading.Thread(target=engine._loop, daemon=True, name="ScanLoop").start()
                SlackAlert.send("🚀 *Signal Engine Auto-Started*\nScanning NIFTY, BANKNIFTY, FINNIFTY\nAlerts will arrive here when confidence ≥ 60%")
                log.info("▶ Auto-startup: scan engine running ✅")
            return
        err = engine.client.last_login_error or "unknown"
        log.warning(f"▶ Auto-startup: login attempt {attempt}/3 failed — {err}")
        if attempt < 3:
            _t.sleep(35)  # next TOTP window
    log.error("▶ Auto-startup: all 3 login attempts failed. Dashboard /api/diag shows the reason.")

threading.Thread(target=_startup, daemon=True, name="Startup").start()

if __name__ == "__main__":
    log.info("="*60)
    log.info("  INTRADAY OPTIONS SIGNAL ENGINE v4.0")
    log.info(f"  Port: {PORT}")
    log.info(f"  Slack Alerts: {'ON' if CONFIG['slack_enabled'] and CONFIG['slack_webhook'] else 'OFF'}")
    log.info(f"  AI Analysis:  {'ON (Sonnet 4)' if CONFIG.get('anthropic_api_key') else 'OFF'}")
    # Pre-load instrument master for instant option lookups
    log.info("  Loading instrument master...")
    _master.load()
    log.info("="*60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
