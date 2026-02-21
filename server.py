"""
╔══════════════════════════════════════════════════════════════════╗
║  INTRADAY OPTIONS SIGNAL ENGINE — Production Server             ║
║  Deploy: Render / Railway / Any Cloud                           ║
║  Features: Live Signals + Option Picks + P&L Tracking           ║
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from SmartApi import SmartConnect
import pyotp

# ═══════════════════════════════════════════════════════════════════
# CONFIG — Uses environment variables for Render deployment
# Set these in Render Dashboard → Environment Variables
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "api_key":      os.environ.get("ANGEL_API_KEY", "zOgZSWbC"),
    "client_id":    os.environ.get("ANGEL_CLIENT_ID", ""),
    "password":     os.environ.get("ANGEL_PASSWORD", ""),
    "totp_secret":  os.environ.get("ANGEL_TOTP_SECRET", ""),

    "scan_interval_sec": int(os.environ.get("SCAN_INTERVAL", "5")),
    "candle_interval":   "FIVE_MINUTE",
    "lookback_days":     3,
    "target_points_min": int(os.environ.get("TARGET_MIN", "10")),
    "target_points_max": int(os.environ.get("TARGET_MAX", "15")),
    "min_confidence":    int(os.environ.get("MIN_CONFIDENCE", "60")),
}

PORT = int(os.environ.get("PORT", "5050"))

# ═══════════════════════════════════════════════════════════════════
# INSTRUMENTS
# ═══════════════════════════════════════════════════════════════════
INSTRUMENTS = {
    "NIFTY": {
        "symbol": "NIFTY", "token": "99926000", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 25, "strike_gap": 50,
        "expiry_prefix": "NIFTY",
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY", "token": "99926009", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 15, "strike_gap": 100,
        "expiry_prefix": "BANKNIFTY",
    },
    "FINNIFTY": {
        "symbol": "NIFTY FIN SERVICE", "token": "99926037", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 25, "strike_gap": 50,
        "expiry_prefix": "FINNIFTY",
    },
}

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("SignalEngine")

# ═══════════════════════════════════════════════════════════════════
# SQLITE DATABASE — Signal Journal & P&L Tracker
# ═══════════════════════════════════════════════════════════════════
DB_PATH = os.environ.get("DB_PATH", "signals.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            instrument TEXT NOT NULL,
            direction TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            
            -- Index levels
            index_price REAL,
            index_entry REAL,
            index_sl REAL,
            index_target1 REAL,
            index_target2 REAL,
            
            -- Option recommendation
            option_symbol TEXT,
            option_strike REAL,
            option_type TEXT,
            option_expiry TEXT,
            option_entry REAL,
            option_sl REAL,
            option_target1 REAL,
            option_target2 REAL,
            option_lot_size INTEGER,
            
            -- Tracking
            status TEXT DEFAULT 'OPEN',
            exit_price REAL,
            exit_time TEXT,
            pnl_points REAL,
            pnl_rupees REAL,
            result TEXT,
            
            -- Analysis snapshot
            reasons TEXT,
            indicators TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            total_signals INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            best_trade REAL DEFAULT 0,
            worst_trade REAL DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    log.info("📊 Database initialized")

init_db()


def db_execute(query, params=(), fetch=False, fetchone=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(query, params)
    result = None
    if fetchone:
        result = c.fetchone()
    elif fetch:
        result = c.fetchall()
    conn.commit()
    conn.close()
    return result


def save_signal(instrument, signal, option):
    """Save a new signal to the database"""
    db_execute("""
        INSERT INTO signals (
            timestamp, date, instrument, direction, confidence,
            index_price, index_entry, index_sl, index_target1, index_target2,
            option_symbol, option_strike, option_type, option_expiry,
            option_entry, option_sl, option_target1, option_target2, option_lot_size,
            reasons, indicators
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        datetime.now().strftime("%Y-%m-%d"),
        instrument,
        signal["direction"],
        signal["confidence"],
        signal["price"],
        signal["entry"],
        signal["sl"],
        signal["target1"],
        signal["target2"],
        option.get("symbol", "") if option else "",
        option.get("strike", 0) if option else 0,
        option.get("type", "") if option else "",
        option.get("expiry", "") if option else "",
        option.get("entry", 0) if option else 0,
        option.get("sl", 0) if option else 0,
        option.get("target1", 0) if option else 0,
        option.get("target2", 0) if option else 0,
        option.get("lot_size", 0) if option else 0,
        json.dumps(signal.get("reasons", [])),
        json.dumps(signal.get("indicators", {})),
    ))
    log.info(f"💾 Signal saved: {instrument} {signal['direction']} @ {signal['entry']}")


def update_signal_result(signal_id, exit_price, result, pnl_points, pnl_rupees):
    """Update a signal with its outcome"""
    db_execute("""
        UPDATE signals SET 
            status = 'CLOSED', exit_price = ?, exit_time = ?,
            pnl_points = ?, pnl_rupees = ?, result = ?
        WHERE id = ?
    """, (exit_price, datetime.now().strftime("%H:%M:%S"),
          pnl_points, pnl_rupees, result, signal_id))


def get_signal_history(limit=100, date=None):
    """Get signal history"""
    if date:
        rows = db_execute(
            "SELECT * FROM signals WHERE date = ? ORDER BY id DESC LIMIT ?",
            (date, limit), fetch=True
        )
    else:
        rows = db_execute(
            "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
            (limit,), fetch=True
        )
    return [dict(r) for r in rows] if rows else []


def get_performance_stats():
    """Get overall performance statistics"""
    all_closed = db_execute(
        "SELECT * FROM signals WHERE status = 'CLOSED'", fetch=True
    )
    if not all_closed:
        return {
            "total": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "best_trade": 0, "worst_trade": 0, "streak": 0,
        }
    
    rows = [dict(r) for r in all_closed]
    wins = [r for r in rows if r["result"] == "WIN"]
    losses = [r for r in rows if r["result"] == "LOSS"]
    
    total_pnl = sum(r["pnl_rupees"] or 0 for r in rows)
    avg_win = sum(r["pnl_rupees"] or 0 for r in wins) / len(wins) if wins else 0
    avg_loss = sum(r["pnl_rupees"] or 0 for r in losses) / len(losses) if losses else 0
    
    pnls = [r["pnl_rupees"] or 0 for r in rows]
    
    # Current streak
    streak = 0
    for r in reversed(rows):
        if r["result"] == "WIN":
            streak += 1
        else:
            break
    
    return {
        "total": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(rows) * 100, 1) if rows else 0,
        "total_pnl": round(total_pnl, 0),
        "avg_win": round(avg_win, 0),
        "avg_loss": round(avg_loss, 0),
        "best_trade": round(max(pnls), 0) if pnls else 0,
        "worst_trade": round(min(pnls), 0) if pnls else 0,
        "streak": streak,
    }


# ═══════════════════════════════════════════════════════════════════
# ANGEL ONE CLIENT
# ═══════════════════════════════════════════════════════════════════
class AngelOneClient:
    def __init__(self):
        self.smart_api = None
        self.connected = False
        self.last_login = None
    
    def login(self):
        try:
            self.smart_api = SmartConnect(api_key=CONFIG["api_key"])
            totp = pyotp.TOTP(CONFIG["totp_secret"]).now()
            
            data = self.smart_api.generateSession(
                clientCode=CONFIG["client_id"],
                password=CONFIG["password"],
                totp=totp
            )
            
            if data and data.get("status"):
                self.connected = True
                self.last_login = datetime.now()
                log.info("✅ Angel One login successful")
                return True
            else:
                log.error(f"❌ Login failed: {data}")
                return False
        except Exception as e:
            log.error(f"❌ Login error: {e}")
            return False
    
    def ensure_connected(self):
        """Re-login if session expired (Angel One sessions expire after ~6 hours)"""
        if not self.connected or not self.last_login:
            return self.login()
        if (datetime.now() - self.last_login).seconds > 18000:  # 5 hours
            log.info("🔄 Re-authenticating (session refresh)")
            return self.login()
        return True
    
    def get_candles(self, token, exchange, interval="FIVE_MINUTE", days=3):
        try:
            if not self.ensure_connected():
                return pd.DataFrame()
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M"),
            }
            
            resp = self.smart_api.getCandleData(params)
            
            if resp and resp.get("status") and resp.get("data"):
                df = pd.DataFrame(
                    resp["data"],
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Candle error ({token}): {e}")
            return pd.DataFrame()
    
    def get_ltp(self, exchange, symbol, token):
        try:
            data = self.smart_api.ltpData(exchange, symbol, token)
            return data["data"] if data and data.get("status") else None
        except Exception as e:
            log.error(f"LTP error: {e}")
            return None
    
    def get_option_chain(self, symbol_info, spot_price):
        """Fetch option chain and find best strikes"""
        try:
            gap = symbol_info["strike_gap"]
            atm = round(spot_price / gap) * gap
            strikes = [atm + (i * gap) for i in range(-3, 4)]
            
            # Find next weekly expiry (Thursday)
            today = datetime.now()
            days_ahead = 3 - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            expiry = today + timedelta(days=days_ahead)
            expiry_str = expiry.strftime("%d%b%Y").upper()
            
            options = []
            for strike in strikes:
                for opt_type in ["CE", "PE"]:
                    try:
                        sym = f"{symbol_info['expiry_prefix']}{expiry_str}{strike}{opt_type}"
                        ltp = self.smart_api.ltpData(
                            symbol_info["option_exchange"], sym, ""
                        )
                        if ltp and ltp.get("status") and ltp.get("data"):
                            d = ltp["data"]
                            options.append({
                                "strike": strike, "type": opt_type, "symbol": sym,
                                "ltp": d.get("ltp", 0), "token": d.get("symboltoken", ""),
                                "expiry": expiry_str,
                                "oi": d.get("opninterest", 0),
                                "volume": d.get("volume", 0),
                            })
                    except:
                        continue
            
            return options, atm
        except Exception as e:
            log.error(f"Option chain error: {e}")
            return [], 0


# ═══════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════
class TA:
    @staticmethod
    def ema(s, p): return s.ewm(span=p, adjust=False).mean()
    
    @staticmethod
    def rsi(c, p=14):
        d = c.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
        ag = g.ewm(span=p, adjust=False).mean(); al = l.ewm(span=p, adjust=False).mean()
        return 100 - (100 / (1 + ag / al))
    
    @staticmethod
    def macd(c, f=12, s=26, sig=9):
        ml = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
        sl = ml.ewm(span=sig, adjust=False).mean()
        return ml, sl, ml - sl
    
    @staticmethod
    def bollinger(c, p=20, sd=2):
        m = c.rolling(p).mean(); s = c.rolling(p).std()
        return m + sd * s, m, m - sd * s
    
    @staticmethod
    def vwap(df):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        return (tp * df["volume"]).cumsum() / df["volume"].cumsum()
    
    @staticmethod
    def atr(df, p=14):
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=p, adjust=False).mean()
    
    @staticmethod
    def supertrend(df, p=10, m=3):
        atr = TA.atr(df, p)
        hl2 = (df["high"] + df["low"]) / 2
        ub, lb = hl2 + m * atr, hl2 - m * atr
        trend = pd.Series(1, index=df.index)
        fu, fl = ub.copy(), lb.copy()
        for i in range(1, len(df)):
            fu.iloc[i] = ub.iloc[i] if ub.iloc[i] < fu.iloc[i-1] or df["close"].iloc[i-1] > fu.iloc[i-1] else fu.iloc[i-1]
            fl.iloc[i] = lb.iloc[i] if lb.iloc[i] > fl.iloc[i-1] or df["close"].iloc[i-1] < fl.iloc[i-1] else fl.iloc[i-1]
            if trend.iloc[i-1] == -1 and df["close"].iloc[i] > fu.iloc[i-1]: trend.iloc[i] = 1
            elif trend.iloc[i-1] == 1 and df["close"].iloc[i] < fl.iloc[i-1]: trend.iloc[i] = -1
            else: trend.iloc[i] = trend.iloc[i-1]
        return trend
    
    @staticmethod
    def stochastic(df, kp=14):
        ll = df["low"].rolling(kp).min(); hh = df["high"].rolling(kp).max()
        k = 100 * (df["close"] - ll) / (hh - ll)
        return k, k.rolling(3).mean()
    
    @staticmethod
    def adx(df, p=14):
        pm = df["high"].diff(); mm = -df["low"].diff()
        pm = pm.where((pm > mm) & (pm > 0), 0)
        mm = mm.where((mm > pm) & (mm > 0), 0)
        atr = TA.atr(df, p)
        pdi = 100 * pm.ewm(span=p, adjust=False).mean() / atr
        mdi = 100 * mm.ewm(span=p, adjust=False).mean() / atr
        dx = 100 * ((pdi - mdi).abs() / (pdi + mdi))
        return dx.ewm(span=p, adjust=False).mean(), pdi, mdi


# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════
class SignalGenerator:
    def __init__(self):
        self.target_min = CONFIG["target_points_min"]
        self.target_max = CONFIG["target_points_max"]
    
    def analyze(self, df):
        if len(df) < 30:
            return None
        
        c = df["close"]; n = len(df) - 1; price = c.iloc[n]
        
        e9 = TA.ema(c, 9); e21 = TA.ema(c, 21); e50 = TA.ema(c, min(50, len(c)))
        rsi = TA.rsi(c); ml, sl, mh = TA.macd(c)
        bbu, bbm, bbl = TA.bollinger(c); vwap = TA.vwap(df)
        atr = TA.atr(df); st = TA.supertrend(df)
        sk, sd = TA.stochastic(df); adx, pdi, mdi = TA.adx(df)
        
        vr_avg = df["volume"].tail(20).mean()
        vol_ratio = df["volume"].iloc[n] / vr_avg if vr_avg > 0 else 1
        
        bs, be = 0, 0
        br, ber = [], []
        
        # EMA Crossover (15)
        if e9.iloc[n] > e21.iloc[n] and e9.iloc[n-1] <= e21.iloc[n-1]:
            bs += 15; br.append("🔥 EMA 9/21 Bullish Crossover")
        elif e9.iloc[n] < e21.iloc[n] and e9.iloc[n-1] >= e21.iloc[n-1]:
            be += 15; ber.append("🔥 EMA 9/21 Bearish Crossover")
        elif e9.iloc[n] > e21.iloc[n]: bs += 8; br.append("EMA 9 > 21 bullish")
        else: be += 8; ber.append("EMA 9 < 21 bearish")
        
        # EMA 50 (5)
        if price > e50.iloc[n]: bs += 5; br.append("Above EMA 50")
        else: be += 5; ber.append("Below EMA 50")
        
        # RSI (12)
        rv = rsi.iloc[-1]
        if rv < 30: bs += 12; br.append(f"RSI Oversold ({rv:.1f})")
        elif rv > 70: be += 12; ber.append(f"RSI Overbought ({rv:.1f})")
        elif 50 < rv < 65: bs += 6; br.append(f"RSI Bullish ({rv:.1f})")
        elif 35 < rv < 50: be += 6; ber.append(f"RSI Bearish ({rv:.1f})")
        
        # MACD (15)
        if mh.iloc[n] > 0 and mh.iloc[n-1] <= 0: bs += 15; br.append("🔥 MACD Bullish Cross")
        elif mh.iloc[n] < 0 and mh.iloc[n-1] >= 0: be += 15; ber.append("🔥 MACD Bearish Cross")
        elif mh.iloc[n] > mh.iloc[n-1] and mh.iloc[n] > 0: bs += 8; br.append("MACD rising")
        elif mh.iloc[n] < mh.iloc[n-1] and mh.iloc[n] < 0: be += 8; ber.append("MACD falling")
        
        # Bollinger (10)
        if price <= bbl.iloc[n] * 1.002: bs += 10; br.append("At Lower BB")
        elif price >= bbu.iloc[n] * 0.998: be += 10; ber.append("At Upper BB")
        
        # VWAP (10)
        if price > vwap.iloc[n] and c.iloc[n-1] <= vwap.iloc[n-1]:
            bs += 10; br.append("🔥 Crossed above VWAP")
        elif price < vwap.iloc[n] and c.iloc[n-1] >= vwap.iloc[n-1]:
            be += 10; ber.append("🔥 Crossed below VWAP")
        elif price > vwap.iloc[n]: bs += 5; br.append("Above VWAP")
        else: be += 5; ber.append("Below VWAP")
        
        # Supertrend (13)
        if st.iloc[n] == 1 and st.iloc[n-1] == -1: bs += 13; br.append("🔥 Supertrend BULL flip")
        elif st.iloc[n] == -1 and st.iloc[n-1] == 1: be += 13; ber.append("🔥 Supertrend BEAR flip")
        elif st.iloc[n] == 1: bs += 7; br.append("Supertrend Bullish")
        else: be += 7; ber.append("Supertrend Bearish")
        
        # Volume (8)
        if vol_ratio > 1.5:
            t = f"Volume Spike ({vol_ratio:.1f}x)"
            if c.iloc[n] > c.iloc[n-1]: bs += 8; br.append(t)
            else: be += 8; ber.append(t)
        
        # Stochastic (7)
        skv = sk.iloc[-1] if not pd.isna(sk.iloc[-1]) else 50
        if skv < 20: bs += 7; br.append(f"Stochastic Oversold ({skv:.0f})")
        elif skv > 80: be += 7; ber.append(f"Stochastic Overbought ({skv:.0f})")
        
        # ADX (7)
        adxv = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        if adxv > 25:
            if pdi.iloc[-1] > mdi.iloc[-1]: bs += 7; br.append(f"ADX {adxv:.0f} + DI+")
            else: be += 7; ber.append(f"ADX {adxv:.0f} + DI-")
        
        # Momentum (3)
        l3 = c.tail(3).values
        if len(l3) == 3 and l3[2] > l3[1] > l3[0]: bs += 3; br.append("3-candle bullish")
        elif len(l3) == 3 and l3[2] < l3[1] < l3[0]: be += 3; ber.append("3-candle bearish")
        
        # Build signal
        conf = min(95, round(max(bs, be)))
        direction = "LONG" if bs > be else "SHORT"
        av = atr.iloc[n]
        
        if direction == "LONG":
            entry = round(price + av * 0.1, 2)
            stop = round(price - av * 1.2, 2)
            t1, t2 = round(entry + self.target_min, 2), round(entry + self.target_max, 2)
        else:
            entry = round(price - av * 0.1, 2)
            stop = round(price + av * 1.2, 2)
            t1, t2 = round(entry - self.target_min, 2), round(entry - self.target_max, 2)
        
        risk = round(abs(entry - stop), 2)
        reward = round(abs(t1 - entry), 2)
        
        return {
            "direction": direction, "confidence": conf,
            "price": round(price, 2), "entry": entry, "sl": stop,
            "target1": t1, "target2": t2,
            "risk": risk, "reward": reward,
            "risk_reward": round(reward / risk, 2) if risk > 0 else 0,
            "reasons": br if direction == "LONG" else ber,
            "indicators": {
                "rsi": round(rv, 1), "macd": round(mh.iloc[n], 3),
                "ema9": round(e9.iloc[n], 2), "ema21": round(e21.iloc[n], 2),
                "ema50": round(e50.iloc[n], 2), "vwap": round(vwap.iloc[n], 2),
                "atr": round(av, 2),
                "bb_upper": round(bbu.iloc[n], 2), "bb_lower": round(bbl.iloc[n], 2),
                "supertrend": "BULL" if st.iloc[n] == 1 else "BEAR",
                "vol_ratio": round(vol_ratio, 2),
                "stoch": round(skv, 0), "adx": round(adxv, 0),
            },
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }


# ═══════════════════════════════════════════════════════════════════
# OPTION RECOMMENDER
# ═══════════════════════════════════════════════════════════════════
class OptionPicker:
    def pick(self, signal, inst_info, chain, atm):
        if not signal or signal["confidence"] < CONFIG["min_confidence"]:
            return None
        
        opt_type = "CE" if signal["direction"] == "LONG" else "PE"
        candidates = [o for o in chain if o["type"] == opt_type and o["ltp"] > 0]
        if not candidates:
            return None
        
        # Sort by proximity to ATM
        candidates.sort(key=lambda x: abs(x["strike"] - atm))
        best = candidates[0]
        
        # Delta estimation
        delta = 0.5 if best["strike"] == atm else (
            0.6 if (signal["direction"] == "LONG" and best["strike"] < atm) or
                   (signal["direction"] == "SHORT" and best["strike"] > atm) else 0.4
        )
        
        entry = best["ltp"]
        t1 = round(entry + abs(signal["target1"] - signal["entry"]) * delta, 2)
        t2 = round(entry + abs(signal["target2"] - signal["entry"]) * delta, 2)
        sl = round(max(entry - abs(signal["sl"] - signal["entry"]) * delta, entry * 0.65), 2)
        lot = inst_info["lot_size"]
        
        return {
            "action": f"BUY {opt_type}",
            "symbol": best["symbol"],
            "strike": best["strike"],
            "type": opt_type,
            "expiry": best.get("expiry", ""),
            "ltp": round(entry, 2),
            "entry": round(entry, 2),
            "sl": sl,
            "target1": t1,
            "target2": t2,
            "delta": delta,
            "lot_size": lot,
            "capital": round(entry * lot, 0),
            "max_loss": round((entry - sl) * lot, 0),
            "t1_profit": round((t1 - entry) * lot, 0),
            "t2_profit": round((t2 - entry) * lot, 0),
            "oi": best.get("oi", 0),
            "volume": best.get("volume", 0),
        }


# ═══════════════════════════════════════════════════════════════════
# AUTO P&L TRACKER
# ═══════════════════════════════════════════════════════════════════
class PLTracker:
    """Monitors open signals and auto-closes them when targets/SL hit"""
    
    def __init__(self, client):
        self.client = client
    
    def check_open_signals(self):
        """Check all OPEN signals and update if target/SL was hit"""
        open_sigs = db_execute(
            "SELECT * FROM signals WHERE status = 'OPEN' AND date = ?",
            (datetime.now().strftime("%Y-%m-%d"),), fetch=True
        )
        
        if not open_sigs:
            return
        
        for sig in open_sigs:
            sig = dict(sig)
            inst_key = sig["instrument"]
            inst = INSTRUMENTS.get(inst_key)
            if not inst:
                continue
            
            # Get current price
            ltp_data = self.client.get_ltp(inst["exchange"], inst["symbol"], inst["token"])
            if not ltp_data:
                continue
            
            current_price = ltp_data.get("ltp", 0)
            if current_price == 0:
                continue
            
            direction = sig["direction"]
            entry = sig["index_entry"]
            sl = sig["index_sl"]
            t1 = sig["index_target1"]
            t2 = sig["index_target2"]
            
            result = None
            exit_price = current_price
            
            if direction == "LONG":
                if current_price >= t1:
                    result = "WIN"
                    pnl_pts = current_price - entry
                elif current_price <= sl:
                    result = "LOSS"
                    pnl_pts = current_price - entry
            else:
                if current_price <= t1:
                    result = "WIN"
                    pnl_pts = entry - current_price
                elif current_price >= sl:
                    result = "LOSS"
                    pnl_pts = entry - current_price
            
            if result:
                lot = inst.get("lot_size", 25)
                pnl_rs = round(pnl_pts * lot, 0)
                update_signal_result(sig["id"], exit_price, result, round(pnl_pts, 2), pnl_rs)
                log.info(f"{'✅' if result == 'WIN' else '❌'} {inst_key} {direction} closed: {result} | P&L: ₹{pnl_rs}")
    
    def force_close_eod(self):
        """Close all open signals at end of day"""
        open_sigs = db_execute(
            "SELECT * FROM signals WHERE status = 'OPEN' AND date = ?",
            (datetime.now().strftime("%Y-%m-%d"),), fetch=True
        )
        for sig in open_sigs:
            sig = dict(sig)
            # Mark as expired
            update_signal_result(sig["id"], sig["index_price"], "EXPIRED", 0, 0)


# ═══════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self):
        self.client = AngelOneClient()
        self.signals = SignalGenerator()
        self.options = OptionPicker()
        self.tracker = PLTracker(self.client)
        self.latest = {}
        self.alerts = []
        self.running = False
        self._prev = {}
    
    def start(self):
        if not self.client.login():
            return {"status": "error", "message": "Login failed"}
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return {"status": "ok", "message": "Engine started"}
    
    def stop(self):
        self.running = False
        self.tracker.force_close_eod()
        return {"status": "ok"}
    
    def _loop(self):
        while self.running:
            try:
                # Check if market hours
                now = datetime.now()
                if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour >= 16:
                    time.sleep(30)
                    continue
                
                # Close at 3:25 PM
                if now.hour == 15 and now.minute >= 25:
                    self.tracker.force_close_eod()
                    self.running = False
                    log.info("🔔 Market closing — all signals closed")
                    break
                
                # Check open P&L
                self.tracker.check_open_signals()
                
                for name, inst in INSTRUMENTS.items():
                    df = self.client.get_candles(inst["token"], inst["exchange"])
                    if df.empty or len(df) < 30:
                        continue
                    
                    sig = self.signals.analyze(df)
                    if not sig:
                        continue
                    
                    opt = None
                    if sig["confidence"] >= CONFIG["min_confidence"]:
                        chain, atm = self.client.get_option_chain(inst, sig["price"])
                        if chain:
                            opt = self.options.pick(sig, inst, chain, atm)
                    
                    result = {
                        "instrument": name, "lot_size": inst["lot_size"],
                        "signal": sig, "option": opt,
                        "updated_at": datetime.now().strftime("%H:%M:%S"),
                    }
                    
                    # New alert?
                    prev = self._prev.get(name, {}).get("signal", {})
                    if sig["confidence"] >= CONFIG["min_confidence"] and (
                        not prev or prev.get("direction") != sig["direction"]
                        or abs(prev.get("confidence", 0) - sig["confidence"]) > 10
                    ):
                        alert = {
                            "id": int(time.time() * 1000),
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "instrument": name, "signal": sig, "option": opt,
                        }
                        self.alerts.insert(0, alert)
                        self.alerts = self.alerts[:100]
                        
                        # Save to DB
                        save_signal(name, sig, opt)
                        log.info(f"🚨 {name} {sig['direction']} Conf:{sig['confidence']}% | Entry:{sig['entry']} SL:{sig['sl']} T1:{sig['target1']}")
                        if opt:
                            log.info(f"   📋 {opt['action']} {opt['symbol']} @ ₹{opt['entry']} | SL:₹{opt['sl']} T1:₹{opt['target1']} T2:₹{opt['target2']}")
                    
                    self._prev[name] = result
                    self.latest[name] = result
                
                time.sleep(CONFIG["scan_interval_sec"])
            except Exception as e:
                log.error(f"Loop error: {e}")
                time.sleep(5)
    
    def get_state(self):
        return {
            "running": self.running,
            "signals": self.latest,
            "alerts": self.alerts[:50],
            "performance": get_performance_stats(),
            "config": {
                "scan_interval": CONFIG["scan_interval_sec"],
                "target_min": CONFIG["target_points_min"],
                "target_max": CONFIG["target_points_max"],
                "min_confidence": CONFIG["min_confidence"],
            },
            "time": datetime.now().strftime("%H:%M:%S"),
            "market_open": 9 <= datetime.now().hour < 16,
        }


# ═══════════════════════════════════════════════════════════════════
# FLASK API
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)
engine = Engine()

@app.route("/")
def home():
    return jsonify({
        "name": "Intraday Options Signal Engine",
        "status": "running" if engine.running else "stopped",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    d = request.json or {}
    for k in ["scan_interval", "target_min", "target_max", "min_confidence"]:
        if k in d:
            CONFIG[f"{'scan_interval_sec' if k == 'scan_interval' else k}"] = int(d[k])
    if "target_min" in d: engine.signals.target_min = int(d["target_min"])
    if "target_max" in d: engine.signals.target_max = int(d["target_max"])
    return jsonify({"status": "ok"})

@app.route("/api/history")
def history():
    date = request.args.get("date")
    limit = int(request.args.get("limit", 100))
    return jsonify(get_signal_history(limit, date))

@app.route("/api/performance")
def performance():
    return jsonify(get_performance_stats())

@app.route("/api/close/<int:signal_id>", methods=["POST"])
def manual_close(signal_id):
    """Manually close a signal with exit price"""
    d = request.json or {}
    exit_price = d.get("exit_price", 0)
    sig = db_execute("SELECT * FROM signals WHERE id = ?", (signal_id,), fetchone=True)
    if not sig:
        return jsonify({"error": "Signal not found"}), 404
    sig = dict(sig)
    pnl = (exit_price - sig["index_entry"]) if sig["direction"] == "LONG" else (sig["index_entry"] - exit_price)
    result = "WIN" if pnl > 0 else "LOSS"
    lot = INSTRUMENTS.get(sig["instrument"], {}).get("lot_size", 25)
    update_signal_result(signal_id, exit_price, result, round(pnl, 2), round(pnl * lot, 0))
    return jsonify({"status": "ok", "result": result, "pnl": round(pnl * lot, 0)})


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  INTRADAY OPTIONS SIGNAL ENGINE v3.0")
    log.info(f"  Port: {PORT}")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=False)
