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
from datetime import datetime, timedelta

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request as flask_request, send_file
from flask_cors import CORS

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
    "min_confidence":    int(os.environ.get("MIN_CONFIDENCE", "40")),
    "budget":            int(os.environ.get("BUDGET", "20000")),

    # ── Slack Alert Config ──
    # Create webhook: Slack → Apps → Incoming Webhooks → Add to Slack → Select your DM
    "slack_webhook":    os.environ.get("SLACK_WEBHOOK", ""),
    "slack_enabled":    os.environ.get("SLACK_ENABLED", "true").lower() == "true",
    
    # ── AI Analysis (Claude Sonnet 4) ──
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
}

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
        entry_time = signal.get("timestamp", datetime.now().strftime("%H:%M"))
        
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
        exit_time = datetime.now().strftime("%H:%M")
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
            option_entry REAL, option_sl REAL, option_target1 REAL, option_target2 REAL,
            option_lot_size INTEGER,
            status TEXT DEFAULT 'OPEN', exit_price REAL, exit_time TEXT,
            pnl_points REAL, pnl_rupees REAL, result TEXT,
            reasons TEXT, indicators TEXT
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

def save_signal(instrument, signal, option):
    db_exec("""INSERT INTO signals (timestamp,date,instrument,direction,confidence,
        index_price,index_entry,index_sl,index_target1,index_target2,
        option_symbol,option_strike,option_type,option_expiry,
        option_entry,option_sl,option_target1,option_target2,option_lot_size,
        reasons,indicators) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d"),
     instrument, signal["direction"], signal["confidence"],
     signal["price"], signal["entry"], signal["sl"], signal["target1"], signal["target2"],
     option.get("symbol","") if option else "", option.get("strike",0) if option else 0,
     option.get("type","") if option else "", option.get("expiry","") if option else "",
     option.get("entry",0) if option else 0, option.get("sl",0) if option else 0,
     option.get("target1",0) if option else 0, option.get("target2",0) if option else 0,
     option.get("lot_size",0) if option else 0,
     json.dumps(signal.get("reasons",[])), json.dumps(signal.get("indicators",{}))))

def update_result(sig_id, exit_price, result, pnl_pts, pnl_rs):
    db_exec("UPDATE signals SET status='CLOSED',exit_price=?,exit_time=?,pnl_points=?,pnl_rupees=?,result=? WHERE id=?",
            (exit_price, datetime.now().strftime("%H:%M:%S"), pnl_pts, pnl_rs, result, sig_id))

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
class AngelClient:
    def __init__(self):
        self.api = None; self.connected = False; self.last_login = None
    
    def login(self):
        try:
            log.info(f"🔐 Attempting login... client_id={CONFIG['client_id']}")
            self.api = SmartConnect(api_key=CONFIG["api_key"])
            secret = CONFIG["totp_secret"].upper().replace("0","O").replace("1","I").replace("8","B")
            totp = pyotp.TOTP(secret).now()
            data = self.api.generateSession(clientCode=CONFIG["client_id"], password=CONFIG["password"], totp=totp)
            if data and data.get("status"):
                self.connected = True; self.last_login = datetime.now()
                log.info("✅ Angel One login successful"); return True
            log.error(f"❌ Login failed: {data}"); return False
        except Exception as e:
            log.error(f"❌ Login error: {e}"); return False
    
    def ensure(self):
        if not self.connected: return self.login()
        if self.last_login and (datetime.now()-self.last_login).seconds > 18000: return self.login()
        return True
    
    def candles(self, token, exchange, interval="FIVE_MINUTE", days=3):
        try:
            if not self.ensure(): return pd.DataFrame()
            resp = self.api.getCandleData({
                "exchange":exchange,"symboltoken":token,"interval":interval,
                "fromdate":(datetime.now()-timedelta(days=days)).strftime("%Y-%m-%d %H:%M"),
                "todate":datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
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
            strikes = [atm+i*gap for i in range(-3,4)]
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
            try:
                batch_resp = self.api.getMarketData(mode="LTP", exchangeTokens={"NFO": token_list})
                if batch_resp and batch_resp.get("status") and batch_resp.get("data"):
                    fetched = batch_resp["data"].get("fetched", [])
                    unfetched = batch_resp["data"].get("unfetched", [])
                    log.info(f"  Batch: {len(fetched)} fetched, {len(unfetched)} unfetched")
                    
                    for item in fetched:
                        tok = str(item.get("symbolToken", item.get("symboltoken", "")))
                        ltp = item.get("ltp", 0)
                        if tok in token_map and ltp > 0:
                            tk = token_map[tok]
                            opts.append({"strike":tk["strike"],"type":tk["type"],
                                "symbol":tk["symbol"],"ltp":ltp,
                                "token":tok,"expiry":tk.get("expiry","")})
                else:
                    log.error(f"  Batch API failed: {batch_resp}")
            except Exception as be:
                log.error(f"  Batch getMarketData error: {be}")
            
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
            self.load_time = datetime.now()
            return True
        except Exception as e:
            log.error(f"  Master load error: {e}")
            return False
    
    def ensure(self):
        """Ensure master is loaded (reload if stale > 6 hours)."""
        if self.loaded and self.load_time:
            age = (datetime.now() - self.load_time).total_seconds()
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
        today = datetime.now().date()
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
    @staticmethod
    def ema(s,p): return s.ewm(span=p,adjust=False).mean()
    @staticmethod
    def rsi(c,p=14):
        d=c.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0)
        return 100-(100/(1+g.ewm(span=p,adjust=False).mean()/l.ewm(span=p,adjust=False).mean()))
    @staticmethod
    def macd(c):
        ml=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
        return ml, ml.ewm(span=9,adjust=False).mean(), ml-ml.ewm(span=9,adjust=False).mean()
    @staticmethod
    def bb(c,p=20,sd=2):
        m=c.rolling(p).mean();s=c.rolling(p).std();return m+sd*s,m,m-sd*s
    @staticmethod
    def vwap(df):
        tp=(df["high"]+df["low"]+df["close"])/3;return(tp*df["volume"]).cumsum()/df["volume"].cumsum()
    @staticmethod
    def atr(df,p=14):
        tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift(1)).abs(),(df["low"]-df["close"].shift(1)).abs()],axis=1).max(axis=1)
        return tr.ewm(span=p,adjust=False).mean()
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
        atr=TA.atr(df,p);pdi=100*pm.ewm(span=p,adjust=False).mean()/atr;mdi=100*mm.ewm(span=p,adjust=False).mean()/atr
        return(100*((pdi-mdi).abs()/(pdi+mdi))).ewm(span=p,adjust=False).mean(),pdi,mdi

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
        if direction=="LONG":
            entry=round(price+av*0.1,2);stop=round(price-av*1.2,2)
            t1,t2=round(entry+self.tmin,2),round(entry+self.tmax,2)
        else:
            entry=round(price-av*0.1,2);stop=round(price+av*1.2,2)
            t1,t2=round(entry-self.tmin,2),round(entry-self.tmax,2)
        risk=round(abs(entry-stop),2);reward=round(abs(t1-entry),2)
        
        return {"direction":direction,"confidence":conf,"price":round(price,2),
            "entry":entry,"sl":stop,"target1":t1,"target2":t2,
            "risk":risk,"reward":reward,"risk_reward":round(reward/risk,2) if risk>0 else 0,
            "reasons":br if direction=="LONG" else ber,
            "indicators":{"rsi":round(rv,1),"macd":round(mh.iloc[n],3),"ema9":round(e9.iloc[n],2),
                "ema21":round(e21.iloc[n],2),"ema50":round(e50.iloc[n],2),"vwap":round(vwap.iloc[n],2),
                "atr":round(av,2),"bb_upper":round(bbu.iloc[n],2),"bb_lower":round(bbl.iloc[n],2),
                "supertrend":"BULL" if st.iloc[n]==1 else "BEAR","vol_ratio":round(vr,2),
                "stoch":round(skv,0),"adx":round(adxv,0)},
            "timestamp":datetime.now().strftime("%H:%M:%S")}

# ═══════════════════════════════════════════════════════════════════
# AI SIGNAL ANALYSIS (Claude API)
# ═══════════════════════════════════════════════════════════════════
class AIAnalysis:
    API_URL = "https://api.anthropic.com/v1/messages"
    
    @staticmethod
    def analyze(instrument, signal, option):
        """Ask Claude to evaluate a trading signal and give verdict"""
        api_key = CONFIG.get("anthropic_api_key", "")
        if not api_key:
            return None
        
        try:
            ind = signal.get("indicators", {})
            opt_info = ""
            if option:
                opt_info = f"""
Option: {option.get('symbol','')} | LTP: ₹{option.get('ltp',0)} | Delta: {option.get('delta',0)}
Option SL: ₹{option.get('sl',0)} | T1: ₹{option.get('target1',0)} | T2: ₹{option.get('target2',0)}
Capital: ₹{option.get('capital',0)} | Max Loss: ₹{option.get('max_loss',0)}"""

            prompt = f"""You are an expert Indian intraday options trader. Analyze this signal and give a quick verdict.

SIGNAL:
Instrument: {instrument}
Direction: {signal['direction']} | Confidence: {signal['confidence']}%
Entry: {signal['entry']} | SL: {signal['sl']} | T1: {signal['target1']} | T2: {signal['target2']}
R:R: {signal.get('risk_reward',0)}
{opt_info}

INDICATORS:
RSI: {ind.get('rsi',0)} | MACD: {ind.get('macd',0)} | SuperTrend: {ind.get('supertrend','')}
EMA9: {ind.get('ema9',0)} | EMA21: {ind.get('ema21',0)} | VWAP: {ind.get('vwap',0)}
ATR: {ind.get('atr',0)} | Stoch: {ind.get('stoch',0)} | ADX: {ind.get('adx',0)} | Vol: {ind.get('vol_ratio',0)}x

REASONS: {', '.join(signal.get('reasons',[])[:5])}
Current time: {datetime.now().strftime('%H:%M')} IST

Respond in EXACTLY this JSON format (no markdown, no backticks):
{{"verdict": "TAKE" or "SKIP" or "WAIT", "confidence_adj": number between -15 and +15, "reasoning": "1 line why", "risk_note": "1 line risk", "exit_tip": "when to exit if not hitting target"}}"""

            resp = requests.post(
                AIAnalysis.API_URL,
                headers={"Content-Type": "application/json",
                         "x-api-key": api_key,
                         "anthropic-version": "2023-06-01"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 200,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=15
            )
            
            if resp.status_code == 200:
                data = resp.json()
                text = data["content"][0]["text"].strip()
                # Parse JSON response
                import re
                text = re.sub(r'```json\s*|```\s*', '', text).strip()
                result = json.loads(text)
                log.info(f"🤖 AI: {instrument} → {result.get('verdict','?')} ({result.get('reasoning','')[:60]})")
                return result
            else:
                log.warning(f"AI API error: {resp.status_code}")
                return None
        except Exception as e:
            log.warning(f"AI analysis failed: {e}")
            return None


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
    
    now = datetime.now()
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
    """Pick the BEST option from real Angel One chain data.
    Rules: ₹40-80 premium, 1 OTM preferred, max 50% capital per trade."""
    
    def pick(self, sig, info, chain, atm, budget=20000):
        if not sig: return None
        ot="CE" if sig["direction"]=="LONG" else "PE"
        cands=[o for o in chain if o["type"]==ot and o["ltp"]>0]
        if not cands: return None
        
        gap = info["strike_gap"]
        lot = info["lot_size"]
        price = sig["price"]
        max_capital = budget * 0.5  # max 50% per trade
        
        # Score each option
        scored = []
        for o in cands:
            ltp = o["ltp"]
            strike = o["strike"]
            
            if ltp < 5: continue  # too cheap = too far OTM
            
            affordable = (ltp * lot <= max_capital)
            
            # Distance from ATM (in gaps)
            otm_gaps = abs(strike - atm) / gap
            
            # Moneyness for delta estimate
            moneyness = abs(strike - price) / price
            
            # Is it the right side? (CE: strike >= atm, PE: strike <= atm)
            right_side = (ot == "CE" and strike >= atm) or (ot == "PE" and strike <= atm)
            
            # Delta estimate from moneyness
            if moneyness < 0.001: delta = 0.50
            elif moneyness < 0.002: delta = 0.45
            elif moneyness < 0.003: delta = 0.38
            elif moneyness < 0.005: delta = 0.30
            elif moneyness < 0.008: delta = 0.22
            else: delta = 0.12
            
            # For ITM options, delta is higher
            if not right_side: delta = min(0.70, delta + 0.20)
            
            # Score: prefer ₹40-80 premium, 1 OTM, good R:R
            score = 0
            
            # Premium range scoring (₹40-80 is ideal)
            if 40 <= ltp <= 80: score += 30
            elif 30 <= ltp <= 100: score += 20
            elif 20 <= ltp <= 150: score += 10
            
            # OTM distance scoring (1 OTM = best, 0 = ATM ok, 2+ = too far)
            if right_side and otm_gaps == 1: score += 25  # ideal 1 OTM
            elif otm_gaps == 0: score += 20  # ATM is fine
            elif right_side and otm_gaps == 2: score += 15
            elif otm_gaps >= 3: score += 5
            
            # Delta scoring (0.30-0.45 ideal for day trading)
            if 0.30 <= delta <= 0.45: score += 15
            elif 0.20 <= delta <= 0.50: score += 10
            
            # R:R scoring based on real LTP
            idx_move_to_t1 = abs(sig["target1"] - sig["entry"])
            opt_move_to_t1 = idx_move_to_t1 * delta
            idx_move_to_sl = abs(sig["sl"] - sig["entry"])
            opt_move_to_sl = idx_move_to_sl * delta
            rr = opt_move_to_t1 / max(opt_move_to_sl, 1)
            if rr >= 1.5: score += 10
            elif rr >= 1.0: score += 5
            
            # Budget bonus (prefer affordable, but NEVER exclude)
            if affordable: score += 20
            
            scored.append({**o, "delta": round(delta, 2), "score": score, 
                          "otm_gaps": otm_gaps, "right_side": right_side, "rr": round(rr, 2),
                          "affordable": affordable})
        
        if not scored: return None
        scored.sort(key=lambda x: -x["score"])
        b = scored[0]
        
        # Calculate option targets/SL using delta
        e = b["ltp"]
        d = b["delta"]
        idx_to_sl = abs(sig["sl"] - sig["entry"])
        idx_to_t1 = abs(sig["target1"] - sig["entry"])
        idx_to_t2 = abs(sig["target2"] - sig["entry"])
        
        sl = round(max(e - idx_to_sl * d, e * 0.65), 2)
        t1 = round(e + idx_to_t1 * d, 2)
        t2 = round(e + idx_to_t2 * d, 2)
        
        if b.get("affordable", True):
            lots = max(1, min(int(max_capital / (e * lot)), 2))
        else:
            lots = 1  # Show 1 lot even if over budget
        qty = lots * lot
        capital = round(e * qty)
        
        return {
            "action": f"BUY {ot}", "symbol": b["symbol"], "strike": b["strike"], "type": ot,
            "expiry": b.get("expiry", ""), "ltp": round(e, 2), "entry": round(e, 2),
            "sl": sl, "target1": t1, "target2": t2, "delta": d,
            "lot_size": lot, "lots": lots, "qty": qty,
            "capital": capital, "max_loss": round((e - sl) * qty),
            "t1_profit": round((t1 - e) * qty), "t2_profit": round((t2 - e) * qty),
            "rr": b["rr"], "otm_gaps": b["otm_gaps"], "score": b["score"],
            "alternatives": len(scored),  # how many options were considered
            "source": "LIVE"
        }

# ═══════════════════════════════════════════════════════════════════
# P&L TRACKER
# ═══════════════════════════════════════════════════════════════════
class PLTracker:
    def __init__(self, client): self.client = client
    
    def check(self):
        opens = db_exec("SELECT * FROM signals WHERE status='OPEN' AND date=?",
                        (datetime.now().strftime("%Y-%m-%d"),), fetch=True)
        if not opens: return
        for s in opens:
            s=dict(s); inst=INSTRUMENTS.get(s["instrument"])
            if not inst: continue
            ltp=self.client.ltp(inst["exchange"],inst["symbol"],inst["token"])
            if not ltp: continue
            cp=ltp.get("ltp",0)
            if cp==0: continue
            result=None;d=s["direction"];entry=s["index_entry"];sl=s["index_sl"];t1=s["index_target1"]
            if d=="LONG":
                if cp>=t1: result="WIN"
                elif cp<=sl: result="LOSS"
            else:
                if cp<=t1: result="WIN"
                elif cp>=sl: result="LOSS"
            if result:
                pnl_pts=(cp-entry) if d=="LONG" else (entry-cp)
                lot=inst.get("lot_size",25);pnl_rs=round(pnl_pts*lot,0)
                update_result(s["id"],cp,result,round(pnl_pts,2),pnl_rs)
                emoji="✅" if result=="WIN" else "❌"
                log.info(f"{emoji} {s['instrument']} {d} → {result} | ₹{pnl_rs}")
                # WhatsApp close alert
                SlackAlert.send(SlackAlert.format_close(s["instrument"],d,result,pnl_rs))
    
    def close_all(self):
        opens=db_exec("SELECT * FROM signals WHERE status='OPEN' AND date=?",
                      (datetime.now().strftime("%Y-%m-%d"),),fetch=True)
        for s in opens:
            s=dict(s);update_result(s["id"],s["index_price"],"EXPIRED",0,0)
        # Send daily summary
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
        self.running=False;self._prev={}
    
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
    
    def _loop(self):
        while self.running:
            try:
                now=datetime.now()
                if now.hour<9 or(now.hour==9 and now.minute<15)or now.hour>=16:
                    time.sleep(30);continue
                if now.hour==15 and now.minute>=25:
                    self.tracker.close_all();self.running=False
                    log.info("🔔 Market close");break
                
                self.tracker.check()
                
                for name,inst in INSTRUMENTS.items():
                    df=self.client.candles(inst["token"],inst["exchange"])
                    if df.empty or len(df)<30: continue
                    sig=self.sgen.analyze(df)
                    if not sig: continue
                    
                    # Fetch real option chain at 40%+ for dashboard display
                    opt=None
                    if sig["confidence"]>=40 and sig.get("direction") in ("LONG","SHORT"):
                        try:
                            chain,atm=self.client.option_chain(inst,sig["price"])
                            if chain: opt=self.opick.pick(sig,inst,chain,atm,CONFIG.get("budget",20000))
                        except Exception as ce:
                            log.warning(f"  Chain fetch failed for {name}: {ce}")
                    
                    # Estimate exit time
                    timing, _ = estimate_exit_time(sig)
                    
                    result={"instrument":name,"lot_size":inst["lot_size"],"signal":sig,"option":opt,
                            "timing":timing,"updated_at":datetime.now().strftime("%H:%M:%S")}
                    
                    prev=self._prev.get(name,{}).get("signal",{})
                    if sig["confidence"]>=CONFIG["min_confidence"] and(
                        not prev or prev.get("direction")!=sig["direction"]
                        or abs(prev.get("confidence",0)-sig["confidence"])>10):
                        
                        # Run AI analysis (async-safe, has timeout)
                        ai_result = AIAnalysis.analyze(name, sig, opt)
                        
                        self.alerts.insert(0,{"id":int(time.time()*1000),"time":datetime.now().strftime("%H:%M:%S"),
                            "instrument":name,"signal":sig,"option":opt,"timing":timing,"ai":ai_result})
                        self.alerts=self.alerts[:100]
                        save_signal(name,sig,opt)
                        log.info(f"🚨 {name} {sig['direction']} Conf:{sig['confidence']}%")
                        
                        # 📱 SLACK ALERT
                        SlackAlert.send(SlackAlert.format_signal(name, sig, opt, timing, ai_result))
                    
                    self._prev[name]=result;self.latest[name]=result
                
                time.sleep(CONFIG["scan_interval_sec"])
            except Exception as e:
                log.error(f"Loop err: {e}");time.sleep(5)
    
    def get_state(self):
        return{"running":self.running,"signals":self.latest,"alerts":self.alerts[:50],
            "performance":get_perf(),
            "config":{"scan_interval":CONFIG["scan_interval_sec"],"target_min":CONFIG["target_points_min"],
                "target_max":CONFIG["target_points_max"],"min_confidence":CONFIG["min_confidence"]},
            "time":datetime.now().strftime("%H:%M:%S"),
            "market_open":9<=datetime.now().hour<16,
            "slack_enabled":CONFIG["slack_enabled"] and bool(CONFIG["slack_webhook"])}

# ═══════════════════════════════════════════════════════════════════
# FLASK API
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=False)
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
def api_login():
    """Explicitly trigger Angel One login"""
    try:
        ok = engine.client.ensure()
        if ok:
            return jsonify({"status": "ok", "connected": True})
        else:
            return jsonify({"status": "failed", "connected": False, "error": "Login returned false. Check server terminal for details."})
    except Exception as e:
        log.error(f"Login endpoint error: {e}")
        return jsonify({"status": "failed", "connected": False, "error": str(e)})

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
                prices[name] = {"ltp": d["ltp"], "symbol": inst["symbol"], "token": inst["token"]}
        except: pass
    
    return jsonify({"prices": prices, "time": datetime.now().strftime("%H:%M:%S")})

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
        return jsonify({"error": "No data returned. Check Angel One connection."}), 500
    
    candles = []
    IST = timedelta(hours=5, minutes=30)
    for _, row in df.iterrows():
        ts = row["timestamp"]
        # Ensure IST: if timezone-aware (UTC from pd.to_datetime), convert
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts + IST  # UTC → IST
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

@app.route("/api/status")
def status():
    return jsonify(engine.get_state())

@app.route("/api/start", methods=["POST"])
def start():
    return jsonify(engine.start())

@app.route("/api/stop", methods=["POST"])
def stop():
    return jsonify(engine.stop())

@app.route("/api/config", methods=["POST"])
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
    
    # Score all candidates (right type, non-zero LTP)
    candidates = [o for o in chain if o["type"] == ot and o["ltp"] > 0]
    if not candidates:
        return jsonify({"error": "No options of type " + ot + " found", "chain_size": len(chain)}), 500
    
    scored = []
    for o in candidates:
        ltp = o["ltp"]
        strike = o["strike"]
        affordable = (ltp * lot <= max_cap)
        
        # Moneyness
        otm_dist = (strike - atm) / gap if direction == "LONG" else (atm - strike) / gap
        moneyness = abs(strike - spot) / spot
        delta = 0.50 if moneyness < 0.001 else (0.42 if moneyness < 0.002 else (0.35 if moneyness < 0.004 else 0.25))
        
        # Scoring
        score = 0
        # 1. Premium range (30 pts max)
        if 40 <= ltp <= 80: score += 30
        elif 30 <= ltp <= 100: score += 20
        elif 20 <= ltp <= 150: score += 10
        # 2. OTM distance (25 pts max)
        if 0.5 <= otm_dist <= 1.5: score += 25
        elif -0.5 <= otm_dist <= 0.5: score += 20
        elif 1.5 < otm_dist <= 2.5: score += 15
        else: score += 5
        # 3. Delta (15 pts max)
        if 0.30 <= delta <= 0.45: score += 15
        elif 0.20 <= delta <= 0.50: score += 10
        # 4. R:R
        sl_pts = ltp * 0.3
        t1_pts = abs(inst.get("target_min", gap * 0.5)) * delta
        if sl_pts > 0 and t1_pts / sl_pts >= 1.5: score += 10
        elif sl_pts > 0 and t1_pts / sl_pts >= 1.0: score += 5
        # 5. Budget penalty (prefer affordable, but don't exclude)
        if affordable: score += 20
        
        scored.append({**o, "delta": delta, "otm_dist": otm_dist, "score": score, "affordable": affordable})
    
    if not scored:
        return jsonify({"error": "No options scored", "chain_size": len(chain)}), 500
    
    # Pick highest scoring
    scored.sort(key=lambda x: -x["score"])
    best = scored[0]
    ltp = best["ltp"]
    
    # If not affordable, still show it but with 1 lot and over-budget flag
    if best["affordable"]:
        lots = max(1, min(int(max_cap / (ltp * lot)), 2))
        over_budget = False
    else:
        lots = 1
        over_budget = True
        log.info(f"  OptLTP: {name} cheapest option ₹{ltp} × {lot} = ₹{ltp*lot} exceeds ₹{max_cap} cap, showing anyway")
    
    log.info(f"  OptLTP: Picked {best['symbol']} LTP=₹{ltp} score={best['score']} (of {len(scored)} candidates)")
    
    return jsonify({
        "symbol": best["symbol"],
        "strike": best["strike"],
        "type": ot,
        "ltp": ltp,
        "delta": best["delta"],
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
def test_slack():
    ok = SlackAlert.send("✅ *Test Alert*\nSlack notifications are working!\nYou'll receive trading signals here during market hours.")
    return jsonify({"status":"ok" if ok else "failed"})

@app.route("/api/test-chain/<name>")
def test_chain(n):
    """Quick test: /api/test-chain/NIFTY → shows if option chain fetches work"""
    inst = INSTRUMENTS.get(n.upper())
    if not inst: return jsonify({"error":"Unknown"}),400
    if not engine.client.ensure(): return jsonify({"error":"Not logged in"}),401
    spot_data = engine.client.ltp(inst["exchange"], inst["symbol"], inst["token"])
    spot = spot_data.get("ltp",0) if spot_data else 0
    if not spot: return jsonify({"error":"No spot price"}),500
    chain, atm = engine.client.option_chain(inst, spot)
    return jsonify({"instrument":n.upper(),"spot":spot,"atm":atm,
        "master_loaded":_master.loaded,"master_count":len(_master.nfo),"chain_count":len(chain),"expiry":chain[0].get("expiry","") if chain else "NONE",
        "sample":chain[:4] if chain else [],
        "status":"OK" if chain else "FAILED - check server logs"})

# Pre-load instrument master (works with both direct run and gunicorn)
if not _master.loaded:
    log.info("  Loading instrument master...")
    _master.load()

if __name__ == "__main__":
    log.info("="*60)
    log.info("  INTRADAY OPTIONS SIGNAL ENGINE v4.0")
    log.info(f"  Port: {PORT}")
    log.info(f"  Slack Alerts: {'ON' if CONFIG['slack_enabled'] and CONFIG['slack_webhook'] else 'OFF'}")
    log.info(f"  AI Analysis:  {'ON (Sonnet 4)' if CONFIG.get('anthropic_api_key') else 'OFF'}")
    log.info("="*60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
