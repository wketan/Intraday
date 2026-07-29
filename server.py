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

# ── NSE trading-holiday / market-status awareness ─────────────────────────
# The engine's market_open flag used to be naive (9<=hour<16): it had no idea
# about weekends or exchange holidays. On the 2026-06-26 Muharram holiday it
# reported "market open", ran pointless scans, and the dashboard kept showing
# the last (prior-day) signal card with no way to tell it was stale. This
# calendar + the empirical newest-bar-date check close that gap.
#
# Only dates we are CERTAIN about are baked in. Extend via env (safe to add
# the full official NSE circular without a redeploy):
#   NSE_HOLIDAYS="2026-08-15=Independence Day,2026-10-02=Gandhi Jayanti"
# Weekends are always closed regardless of this list.
_NSE_HOLIDAYS_DEFAULT = {
    "2026-06-26": "Muharram",
}

def _nse_holidays():
    hols = dict(_NSE_HOLIDAYS_DEFAULT)
    for tok in os.environ.get("NSE_HOLIDAYS", "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            d, nm = tok.split("=", 1); hols[d.strip()] = nm.strip()
        else:
            hols[tok] = "Holiday"
    return hols

def market_status(now=None):
    """Live market open/closed status with a human reason.

    open=False on weekends and listed NSE holidays. A holiday that is NOT in
    the list still won't produce a bad trade (the strategy's own time-gate and
    the empirical stale-data flag catch it) — this only labels the ones we know.
    """
    now = now or datetime.now(IST)
    d = now.strftime("%Y-%m-%d")
    wd = now.weekday()  # Mon=0 .. Sun=6
    if wd >= 5:
        return {"open": False, "reason": "weekend",
                "label": "Saturday" if wd == 5 else "Sunday", "date": d}
    hols = _nse_holidays()
    if d in hols:
        return {"open": False, "reason": "holiday", "label": hols[d], "date": d}
    hm = now.hour * 60 + now.minute
    is_session = (9 * 60 + 15) <= hm <= (15 * 60 + 30)
    return {"open": is_session,
            "reason": "session" if is_session else "after_hours",
            "label": None, "date": d}

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request as flask_request, send_file

from SmartApi import SmartConnect
import pyotp
from options_intel import OptionsIntel

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "api_key":      os.environ.get("ANGEL_API_KEY", ""),
    "client_id":    os.environ.get("ANGEL_CLIENT_ID", ""),
    "password":     os.environ.get("ANGEL_PASSWORD", ""),
    "totp_secret":  os.environ.get("ANGEL_TOTP_SECRET", ""),

    # Scan loop cadence. Default raised 5s → 30s — 5-minute candles don't update inside
    # a 5s window, and the option-chain cache is 30s anyway. Cuts API calls ~6x with no
    # signal-freshness loss.
    "scan_interval_sec": int(os.environ.get("SCAN_INTERVAL", "30")),
    "candle_interval":   "FIVE_MINUTE",
    "lookback_days":     3,
    # Cache 5-min candles for 270s — candles only update every 5 min so there is no
    # benefit to fetching more often; 270s cuts Angel API calls by 3x vs 90s.
    "candle_cache_ttl":  int(os.environ.get("CANDLE_CACHE_TTL", "270")),
    "target_points_min": int(os.environ.get("TARGET_MIN", "10")),
    "target_points_max": int(os.environ.get("TARGET_MAX", "15")),
    "min_confidence":    int(os.environ.get("MIN_CONFIDENCE", "45")),
    "budget":            int(os.environ.get("BUDGET", "50000")),

    # ── Risk guards (kill-switch) ──
    # OFF by default — user wants every quality signal to flow through.
    # Quality is still gated by:
    #   · the AI veto layer (confidence_adj + TAKE/WAIT/SKIP verdict)
    #   · min_confidence floor (62% by default, raised on RANGING regimes)
    #   · regime filters (VIX, expiry, Monday-block, time-of-day cutoff)
    # Set DAILY_LOSS_LIMIT or MAX_TRADES_PER_DAY env vars to re-enable
    # the kill-switch (non-zero value); 0 = disabled.
    "daily_loss_limit":  int(os.environ.get("DAILY_LOSS_LIMIT", "0")),
    "max_trades_per_day":int(os.environ.get("MAX_TRADES_PER_DAY", "0")),

    # ── Cost model (subtracted from displayed P&L for realism) ──
    # Round-trip brokerage estimate per lot (Zerodha/Angel: ~₹40 entry + ~₹40 exit + STT + GST).
    # Adjust if your broker's structure differs.
    "brokerage_per_lot_roundtrip": float(os.environ.get("BROKERAGE_PER_LOT", "100")),
    # Slippage in basis points (1bp = 0.01%) of premium, applied each side. 50bp = 0.5%
    # per side ≈ realistic for ATM weekly options with spreads in the 0.5-1.5% range.
    "slippage_bps_per_side": float(os.environ.get("SLIPPAGE_BPS_PER_SIDE", "50")),

    # Auto-close cutoff (HH:MM IST). Brought forward 15:25 → 15:15 — last 15 min has 2-3x
    # spreads. Override with env if you want to ride later.
    "auto_close_hour":   int(os.environ.get("AUTO_CLOSE_HOUR", "15")),
    "auto_close_minute": int(os.environ.get("AUTO_CLOSE_MINUTE", "15")),

    # ── Option exit-level mode (step 14) ─────────────────────────────────────
    # "premium_pct"  — SL/T1/T2 are exact percentages of live entry premium.
    #                  Dashboard SL=₹40 means "option exits at ₹40 when hit",
    #                  no extrapolation, no delta model. (DEFAULT — recommended)
    # "delta_scaled" — legacy behaviour: SL/T1/T2 = entry ± (index_distance × delta).
    #                  Estimated and ignores gamma/theta/vega. Kept for backward compat.
    "opt_exit_mode":     os.environ.get("OPT_EXIT_MODE", "premium_pct"),
    # ── SL/T1/T2 as % of option premium ────────────────────────────────
    # Reverted to original 0.35/0.50/1.00 after the 0.30/0.45/0.90 experiment
    # made 30d NIFTY backtest WORSE (-₹17k → -₹92k). Hypothesis was wrong:
    # tighter SL combined with later cutoff meant more late-day trades that
    # didn't have time to reach T1 before EOD, exiting at deep theta-bled
    # losses that blew up avg_loss from ₹1,424 → ₹2,089. v2 is structurally
    # broken; tuning these doesn't help. Building ORB + gamma blast instead.
    "opt_sl_pct":        float(os.environ.get("OPT_SL_PCT", "0.35")),   # 35% premium loss = stop
    "opt_t1_pct":        float(os.environ.get("OPT_T1_PCT", "0.50")),   # 50% premium gain = T1
    "opt_t2_pct":        float(os.environ.get("OPT_T2_PCT", "1.00")),   # 100% premium gain = T2

    # ── Strict greeks (step 15) ──────────────────────────────────────────────
    # When true: signals are REJECTED if Angel's getOptionGreek API does not
    # return a live delta in [0.01, 0.99]. No fallback ladder, no estimation —
    # every signal that fires has real, live, exchange-priced delta.
    # Tradeoff: fewer signals when Angel's greek endpoint misbehaves. Worth it
    # if you've been seeing "off" prices and want to eliminate ALL estimation.
    "strict_greeks":     os.environ.get("STRICT_GREEKS", "false").lower() == "true",

    # ── Slack Alert Config ──
    # Create webhook: Slack → Apps → Incoming Webhooks → Add to Slack → Select your DM
    "slack_webhook":    os.environ.get("SLACK_WEBHOOK", ""),
    "slack_enabled":    os.environ.get("SLACK_ENABLED", "true").lower() == "true",

    # ── AI Analysis (Claude) ──
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    # Model for Claude layers A, B, C. Layer D (in-flight) defaults to Haiku 4.5 — its
    # decisions are tiny structured outputs (HOLD/CLOSE/TRAIL) and Haiku is ~1/12 the
    # cost of Sonnet for nearly identical quality on this task.
    "anthropic_model":   os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    "anthropic_model_inflight": os.environ.get(
        "ANTHROPIC_MODEL_INFLIGHT", "claude-haiku-4-5-20251001"),
    # Master toggle for prompt caching (cache_control: ephemeral). Saves ~80% input tokens
    # on Layer B's repeated rules block. Sonnet 4.5 minimum is 1024 cached tokens — below
    # that the request silently runs uncached, no error. Set to "false" to disable.
    "anthropic_cache_enabled": os.environ.get("ANTHROPIC_CACHE_ENABLED", "true").lower() == "true",

    # ── Strategy selector (Phase 2) ──────────────────────────────────────────
    # "auto"      = regime router: ADX trend day → conductor, range day → reverter
    #               (default — covers both regimes instead of starving one)
    # "conductor" = 5-dim confluence only (the pre-router production setting)
    # "v1" = legacy 13-indicator heuristic scorer (unverified edge)
    # "v2" = trend-momentum 3-of-4 confluence (parked, anti-edge confirmed)
    # Flip via env var on Railway — no code redeploy. NOTE: if STRATEGY is set
    # explicitly on Railway it overrides this default; unset it to get "auto".
    "strategy":          os.environ.get("STRATEGY", "auto").lower(),
    # When STRATEGY=v2 AND DRY_RUN_V2=true, signals fire to Slack (with [DRY RUN] tag)
    # and log lines, but are NOT saved to the signals table and DO NOT trigger
    # OptPicker / kill-switch / Layer B. Use to observe v2 signal quality live
    # without risking capital or polluting trade history.
    "dry_run_v2":        os.environ.get("DRY_RUN_V2", "false").lower() == "true",

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
        # Lot 75 per NSE's Nov-2024 revision (matches conductor.py and
        # SWING_STOCKS). Was 65, which skewed qty sizing and the Rs-profit gate.
        "symbol": "NIFTY", "token": "99926000", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 75, "strike_gap": 50,
        "expiry_prefix": "NIFTY", "expiry_day": 1, "expiry_type": "weekly",  # Tuesday weekly
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY", "token": "99926009", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 30, "strike_gap": 100,
        "expiry_prefix": "BANKNIFTY", "expiry_day": 1, "expiry_type": "monthly",  # Last Tuesday monthly
    },
    "FINNIFTY": {
        # Lot 65 per NSE's Nov-2024 revision (matches conductor.py). Was 60.
        "symbol": "NIFTY FIN SERVICE", "token": "99926037", "exchange": "NSE",
        "option_exchange": "NFO", "lot_size": 65, "strike_gap": 50,
        "expiry_prefix": "FINNIFTY", "expiry_day": 1, "expiry_type": "monthly",  # Last Tuesday monthly
    },
}

# ═══════════════════════════════════════════════════════════════════
# SWING / POSITIONAL STOCK UNIVERSE
# Tokens resolved dynamically from InstrumentMaster at runtime.
# nse_fo_name = the name key in Angel One's NFO instrument master (OPTSTK).
# ═══════════════════════════════════════════════════════════════════
SWING_STOCKS = {
    # ── Indices — swing uses monthly expiry ─────────────────────────
    "NIFTY_SW":     {"nse_sym":"Nifty 50",           "nse_fo":"NIFTY",      "exchange":"NSE","type":"INDEX","fo_eligible":True,"lot_size":75,  "strike_gap":50,  "token":"99926000"},  # NSE revised Jan-2026: 75 (weekly) – monthly still 75; kept as-is
    "BANKNIFTY_SW": {"nse_sym":"Nifty Bank",          "nse_fo":"BANKNIFTY",  "exchange":"NSE","type":"INDEX","fo_eligible":True,"lot_size":30,  "strike_gap":100, "token":"99926009"},
    "FINNIFTY_SW":  {"nse_sym":"Nifty Fin Services",  "nse_fo":"FINNIFTY",   "exchange":"NSE","type":"INDEX","fo_eligible":True,"lot_size":65,  "strike_gap":50,  "token":"99926037"},
    # ── Nifty 50 stocks ─────────────────────────────────────────────
    # lot_size values verified against NSE F&O lot size circular (Jan 2026) via Dhan
    "RELIANCE":    {"nse_sym":"RELIANCE",   "nse_fo":"RELIANCE",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":500, "strike_gap":20},
    "TCS":         {"nse_sym":"TCS",        "nse_fo":"TCS",        "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":175, "strike_gap":50},
    "HDFCBANK":    {"nse_sym":"HDFCBANK",   "nse_fo":"HDFCBANK",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":550, "strike_gap":10},
    "INFY":        {"nse_sym":"INFY",       "nse_fo":"INFY",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":400, "strike_gap":20},
    "ICICIBANK":   {"nse_sym":"ICICIBANK",  "nse_fo":"ICICIBANK",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":700, "strike_gap":10},
    "SBIN":        {"nse_sym":"SBIN",       "nse_fo":"SBIN",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":750, "strike_gap":5},
    "BAJFINANCE":  {"nse_sym":"BAJFINANCE", "nse_fo":"BAJFINANCE", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":750, "strike_gap":50},
    "KOTAKBANK":   {"nse_sym":"KOTAKBANK",  "nse_fo":"KOTAKBANK",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2000,"strike_gap":20},
    "LT":          {"nse_sym":"LT",         "nse_fo":"LT",         "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":175, "strike_gap":50},
    "AXISBANK":    {"nse_sym":"AXISBANK",   "nse_fo":"AXISBANK",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":625, "strike_gap":10},
    "ASIANPAINT":  {"nse_sym":"ASIANPAINT", "nse_fo":"ASIANPAINT", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":250, "strike_gap":25},
    "MARUTI":      {"nse_sym":"MARUTI",     "nse_fo":"MARUTI",     "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":50,  "strike_gap":100},
    "TITAN":       {"nse_sym":"TITAN",      "nse_fo":"TITAN",      "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":175, "strike_gap":20},
    "WIPRO":       {"nse_sym":"WIPRO",      "nse_fo":"WIPRO",      "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":3000,"strike_gap":5},
    "ULTRACEMCO":  {"nse_sym":"ULTRACEMCO", "nse_fo":"ULTRACEMCO", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":50,  "strike_gap":50},
    "NESTLEIND":   {"nse_sym":"NESTLEIND",  "nse_fo":"NESTLEIND",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":500, "strike_gap":100},
    "TECHM":       {"nse_sym":"TECHM",      "nse_fo":"TECHM",      "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":600, "strike_gap":10},
    "SUNPHARMA":   {"nse_sym":"SUNPHARMA",  "nse_fo":"SUNPHARMA",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":350, "strike_gap":10},
    "POWERGRID":   {"nse_sym":"POWERGRID",  "nse_fo":"POWERGRID",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1900,"strike_gap":5},
    "NTPC":        {"nse_sym":"NTPC",       "nse_fo":"NTPC",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1500,"strike_gap":5},
    "ONGC":        {"nse_sym":"ONGC",       "nse_fo":"ONGC",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2250,"strike_gap":5},
    "COALINDIA":   {"nse_sym":"COALINDIA",  "nse_fo":"COALINDIA",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1350,"strike_gap":5},
    "BPCL":        {"nse_sym":"BPCL",       "nse_fo":"BPCL",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1975,"strike_gap":5},
    "CIPLA":       {"nse_sym":"CIPLA",      "nse_fo":"CIPLA",      "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":375, "strike_gap":10},
    "HCLTECH":     {"nse_sym":"HCLTECH",    "nse_fo":"HCLTECH",    "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":350, "strike_gap":10},
    "DIVISLAB":    {"nse_sym":"DIVISLAB",   "nse_fo":"DIVISLAB",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":100, "strike_gap":50},
    "MM":          {"nse_sym":"M&M",        "nse_fo":"M&M",        "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":175, "strike_gap":20},
    "EICHERMOT":   {"nse_sym":"EICHERMOT",  "nse_fo":"EICHERMOT",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":100, "strike_gap":25},
    "TATACONSUM":  {"nse_sym":"TATACONSUM", "nse_fo":"TATACONSUM", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1150,"strike_gap":5},
    "BAJAJFINSV":  {"nse_sym":"BAJAJFINSV", "nse_fo":"BAJAJFINSV", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":250, "strike_gap":5},
    "HEROMOTOCO":  {"nse_sym":"HEROMOTOCO", "nse_fo":"HEROMOTOCO", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":150, "strike_gap":20},
    "DRREDDY":     {"nse_sym":"DRREDDY",    "nse_fo":"DRREDDY",    "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":625, "strike_gap":50},
    "ADANIPORTS":  {"nse_sym":"ADANIPORTS", "nse_fo":"ADANIPORTS", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":475, "strike_gap":10},
    "JSWSTEEL":    {"nse_sym":"JSWSTEEL",   "nse_fo":"JSWSTEEL",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":675, "strike_gap":10},
    "TATASTEEL":   {"nse_sym":"TATASTEEL",  "nse_fo":"TATASTEEL",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2750,"strike_gap":5},
    "GRASIM":      {"nse_sym":"GRASIM",     "nse_fo":"GRASIM",     "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":250, "strike_gap":10},
    "BRITANNIA":   {"nse_sym":"BRITANNIA",  "nse_fo":"BRITANNIA",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":125, "strike_gap":25},
    "HINDALCO":    {"nse_sym":"HINDALCO",   "nse_fo":"HINDALCO",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":700, "strike_gap":5},
    "APOLLOHOSP":  {"nse_sym":"APOLLOHOSP", "nse_fo":"APOLLOHOSP", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":125, "strike_gap":50},
    "INDUSINDBK":  {"nse_sym":"INDUSINDBK", "nse_fo":"INDUSINDBK", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":700, "strike_gap":10},
    "SHRIRAMFIN":  {"nse_sym":"SHRIRAMFIN", "nse_fo":"SHRIRAMFIN", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":375, "strike_gap":20},
    "SBILIFE":     {"nse_sym":"SBILIFE",    "nse_fo":"SBILIFE",    "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":750, "strike_gap":10},
    "HDFCLIFE":    {"nse_sym":"HDFCLIFE",   "nse_fo":"HDFCLIFE",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1100,"strike_gap":5},
    "BAJAJ_AUTO":  {"nse_sym":"BAJAJ-AUTO", "nse_fo":"BAJAJ-AUTO", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":75,  "strike_gap":100},
    "TATAMOTORS":  {"nse_sym":"TATAMOTORS", "nse_fo":"TATAMOTORS", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2000,"strike_gap":5},
    "TATAPOWER":   {"nse_sym":"TATAPOWER",  "nse_fo":"TATAPOWER",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":4275,"strike_gap":2},
    "ITC":         {"nse_sym":"ITC",        "nse_fo":"ITC",        "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":3200,"strike_gap":5},
    # ── High-quality F&O mid/large caps ─────────────────────────────
    "VEDL":        {"nse_sym":"VEDL",       "nse_fo":"VEDL",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2000,"strike_gap":5},
    "ZOMATO":      {"nse_sym":"ZOMATO",     "nse_fo":"ZOMATO",     "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":3750,"strike_gap":2},
    "ADANIENT":    {"nse_sym":"ADANIENT",   "nse_fo":"ADANIENT",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":750, "strike_gap":10},
    "PIDILITIND":  {"nse_sym":"PIDILITIND", "nse_fo":"PIDILITIND", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":375, "strike_gap":20},
    "HAVELLS":     {"nse_sym":"HAVELLS",    "nse_fo":"HAVELLS",    "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":500, "strike_gap":10},
    "HAL":         {"nse_sym":"HAL",        "nse_fo":"HAL",        "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":300, "strike_gap":25},
    "BEL":         {"nse_sym":"BEL",        "nse_fo":"BEL",        "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":4550,"strike_gap":5},
    "IRCTC":       {"nse_sym":"IRCTC",      "nse_fo":"IRCTC",      "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":875, "strike_gap":10},
    "TVSMOTOR":    {"nse_sym":"TVSMOTOR",   "nse_fo":"TVSMOTOR",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":350, "strike_gap":20},
    "CHOLAFIN":    {"nse_sym":"CHOLAFIN",   "nse_fo":"CHOLAFIN",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1000,"strike_gap":10},
    "HDFCAMC":     {"nse_sym":"HDFCAMC",    "nse_fo":"HDFCAMC",    "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":300, "strike_gap":25},
    "ICICIGI":     {"nse_sym":"ICICIGI",    "nse_fo":"ICICIGI",    "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":600, "strike_gap":10},
    "ICICIPRULI":  {"nse_sym":"ICICIPRULI", "nse_fo":"ICICIPRULI", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1500,"strike_gap":5},
    "MUTHOOTFIN":  {"nse_sym":"MUTHOOTFIN", "nse_fo":"MUTHOOTFIN", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":375, "strike_gap":20},
    "MOTHERSON":   {"nse_sym":"MOTHERSON",  "nse_fo":"MOTHERSON",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":3500,"strike_gap":5},
    "BALKRISIND":  {"nse_sym":"BALKRISIND", "nse_fo":"BALKRISIND", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":375, "strike_gap":20},
    "INDHOTEL":    {"nse_sym":"INDHOTEL",   "nse_fo":"INDHOTEL",   "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":1500,"strike_gap":5},
    "CANBK":       {"nse_sym":"CANBK",      "nse_fo":"CANBK",      "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":3000,"strike_gap":2},
    "PNB":         {"nse_sym":"PNB",        "nse_fo":"PNB",        "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":7600,"strike_gap":2},
    "FEDERALBNK":  {"nse_sym":"FEDERALBNK", "nse_fo":"FEDERALBNK", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":6000,"strike_gap":2},
    "SAIL":        {"nse_sym":"SAIL",       "nse_fo":"SAIL",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":6750,"strike_gap":2},
    "NMDC":        {"nse_sym":"NMDC",       "nse_fo":"NMDC",       "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":4500,"strike_gap":5},
    "IDFCFIRSTB":  {"nse_sym":"IDFCFIRSTB", "nse_fo":"IDFCFIRSTB", "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":4500,"strike_gap":2},
    "BANKINDIA":   {"nse_sym":"BANKINDIA",  "nse_fo":"BANKINDIA",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":4500,"strike_gap":2},
    "SBICARD":     {"nse_sym":"SBICARD",    "nse_fo":"SBICARD",    "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2000,"strike_gap":5},
    "NAUKRI":      {"nse_sym":"NAUKRI",     "nse_fo":"NAUKRI",     "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":150, "strike_gap":50},
    "AMBUJACEM":   {"nse_sym":"AMBUJACEM",  "nse_fo":"AMBUJACEM",  "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2000,"strike_gap":5},
    "BIOCON":      {"nse_sym":"BIOCON",     "nse_fo":"BIOCON",     "exchange":"NSE","type":"STOCK","fo_eligible":True,"lot_size":2500,"strike_gap":5},
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
    def format_signal_blocks(instrument, signal, option, timing=None, ai=None):
        """Compact 2-column Slack 'blocks' layout. All info from the legacy
        text format, but rendered side-by-side using Slack `fields` (which
        flow in a 2-col grid) instead of stacked single-column lines.
        Takes the same args as format_signal; callers pass both:
            text   = format_signal(...)        # fallback for push-notif preview
            blocks = format_signal_blocks(...) # rich rendering inside Slack
        """
        def _i(v):
            try: return f"{int(round(float(v))):,}"
            except: return str(v)

        # Delegate to the canonical humanizer (fixed regex for Angel symbols).
        _human_sym = SlackAlert._humanize_symbol

        arrow = "🟢" if signal["direction"] == "LONG" else "🔴"
        entry_time = signal.get("timestamp", datetime.now(IST).strftime("%H:%M"))

        header = f"{arrow}  *{instrument}  {signal['direction']}*"
        if option:
            header += f"\n📋  *{option.get('action','BUY')}  ·  {_human_sym(option.get('symbol',''))}*"

        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": header}},
            {"type": "divider"},
        ]

        if option:
            # Two columns × three rows of trade levels
            levels = [
                {"type": "mrkdwn", "text": f"▶ *Buy*\n`₹{_i(option['entry'])}`  _(Live LTP)_"},
                {"type": "mrkdwn", "text": f"🛑 *SL*\n`₹{_i(option['sl'])}`"},
                {"type": "mrkdwn", "text": f"✅ *T1*  `₹{_i(option['target1'])}`\n→ +₹{_i(option.get('t1_profit', 0))}"},
                {"type": "mrkdwn", "text": f"✅ *T2*  `₹{_i(option['target2'])}`\n→ +₹{_i(option.get('t2_profit', 0))}"},
                {"type": "mrkdwn", "text": f"💼 *Capital*\n`₹{_i(option.get('capital','?'))}` · max loss ₹{_i(option.get('max_loss','?'))}"},
                {"type": "mrkdwn", "text": f"📐 *Greeks*\nΔ {option.get('delta', 0.4)} · R:R {signal.get('risk_reward', '?')}"},
            ]
            blocks.append({"type": "section", "fields": levels})
        else:
            idx_fields = [
                {"type": "mrkdwn", "text": f"▶ *Entry*\n`{signal['entry']}`"},
                {"type": "mrkdwn", "text": f"🛑 *SL*\n`{signal['sl']}`"},
                {"type": "mrkdwn", "text": f"✅ *T1*\n`{signal['target1']}`"},
                {"type": "mrkdwn", "text": f"✅ *T2*\n`{signal['target2']}`"},
            ]
            blocks.append({"type": "section", "fields": idx_fields})

        # Timing + confidence in 2-col layout
        timing_fields = [
            {"type": "mrkdwn", "text": f"⏰ *Entry*\n{entry_time} IST"},
        ]
        if timing:
            timing_fields.append({"type": "mrkdwn", "text": f"🎯 *Target by*\n~{timing['target_by']} IST  _({timing.get('est_duration','')})_"})
            timing_fields.append({"type": "mrkdwn", "text": f"🛑 *SL by*\n~{timing['sl_by']} IST"})
        timing_fields.append({"type": "mrkdwn", "text": f"🎯 *Confidence*\n{signal['confidence']}% · {len(signal.get('reasons',[]))} strategies"})
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "fields": timing_fields})

        # AI analysis — full-width section (the rationale is prose, so 1-col)
        if ai and ai.get("verdict"):
            v = ai["verdict"]
            emoji = "✅" if v == "TAKE" else ("⏸" if v == "WAIT" else "⛔")
            adj = ai.get("confidence_adj", 0)
            adj_str = f"+{adj}" if adj > 0 else str(adj)
            ai_text = f"*🤖 AI: {emoji} {v}*  (Conf {adj_str}%)"
            if ai.get("reasoning"):
                ai_text += f"\n💡 {ai['reasoning']}"
            if ai.get("risk_note"):
                ai_text += f"\n⚠️ {ai['risk_note']}"
            blocks.append({"type": "divider"})
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": ai_text}})

        # Reasons — context block (smaller text)
        reasons = signal.get("reasons", [])[:4]
        if reasons:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "*Why:* " + " · ".join(reasons)}],
            })
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "⚠️ Verify option LTP before trading. Not financial advice."}],
        })
        return blocks

    @staticmethod
    def _humanize_symbol(sym):
        """Parse Angel One option symbol format:
            <SYMBOL><DD><MMM><YY><STRIKE><CE/PE>
        e.g. BANKNIFTY26MAY2554300CE → BANKNIFTY · 26-May-25 · 54300 CE

        Year is EXACTLY 2 digits (the current Angel/NSE format). Earlier
        version used \\d{2,4} which was greedy and chewed up the first
        2 digits of the strike (e.g. parsed "2525700" as year=2525,
        strike=700, breaking FINNIFTY25700 → "700" in the alert).
        """
        if not sym: return ""
        import re as _re
        m = _re.match(r'^([A-Z]+?)(\d{1,2})([A-Z]{3})(\d{2})(\d{3,6})(CE|PE)$', sym.upper())
        if m:
            ix, dd, mon, yy, strike, typ = m.groups()
            return f"{ix} · {dd}-{mon.title()}-{yy} · {strike} {typ}"
        return sym

    @staticmethod
    def format_signal(instrument, signal, option, timing=None, ai=None):
        arrow = "🟢" if signal["direction"] == "LONG" else "🔴"
        entry_time = signal.get("timestamp", datetime.now(IST).strftime("%H:%M"))

        msg = f"""{arrow}  *SIGNAL  ·  {instrument}  {signal["direction"]}*
━━━━━━━━━━━━━━━━━━━━━"""

        if option:
            _sym = SlackAlert._humanize_symbol(option.get("symbol", ""))
            msg += f"""

📋  *{option["action"]}  ·  {_sym}*

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
    def format_close(instrument, direction, result, pnl, option=None, entry_time=None, near_miss=None, journey=None):
        emoji = ("🎯" if result == "T2" else
                 "✅" if result == "WIN" else
                 "❌" if result == "LOSS" else "⊙")
        exit_time = datetime.now(IST).strftime("%H:%M")
        result_label = ("Target 2 hit (+100%)" if result == "T2" else
                        "Target 1 hit (+50%)"  if result == "WIN" else
                        "Stop loss hit"        if result == "LOSS" else result)
        # Sign-guard: LOSS must read negative regardless of how pnl was passed.
        raw = float(pnl or 0)
        if result == "LOSS":                pnl_signed = -abs(raw)
        elif result in ("WIN", "T1", "T2", "TRAIL"): pnl_signed =  abs(raw)
        else:                               pnl_signed =  raw
        sign = "+" if pnl_signed >= 0 else "−"
        amt = int(round(abs(pnl_signed)))
        msg = f"""{emoji} *TRADE CLOSED: {instrument}*
━━━━━━━━━━━━━━━━━━━━━
📊 {direction} → *{result_label}*"""
        if option:
            msg += f"\n📋 {SlackAlert._humanize_symbol(option.get('symbol',''))}"
        if entry_time:
            msg += f"\n⏰ {entry_time} → {exit_time} IST"
        msg += f"""
💰 P&L: *{sign}₹{amt:,}*"""
        if journey and journey.get("entry"):
            msg += "\n" + SlackAlert._journey_line(journey)
        if near_miss:
            msg += f"\n⚠️ Near-miss: {near_miss.get('hint','')}"
        msg += "\n━━━━━━━━━━━━━━━━━━━━━"
        return msg

    @staticmethod
    def _journey_line(j):
        """One-line premium journey: entry → peak (+%) at HH:MM → low at HH:MM → exit."""
        e = float(j["entry"])
        parts = [f"entry ₹{e:.0f}"]
        if j.get("peak") is not None:
            pct = (float(j["peak"]) - e) / e * 100 if e else 0
            parts.append(f"peak ₹{float(j['peak']):.0f} ({'+' if pct >= 0 else ''}{pct:.0f}%)"
                         + (f" at {j['peak_time']}" if j.get("peak_time") else ""))
        if j.get("trough") is not None:
            parts.append(f"low ₹{float(j['trough']):.0f}"
                         + (f" at {j['trough_time']}" if j.get("trough_time") else ""))
        if j.get("exit") is not None:
            parts.append(f"exit ₹{float(j['exit']):.0f}")
        return "📈 Journey: " + " → ".join(parts)

    @staticmethod
    def format_close_blocks(instrument, direction, result, pnl, option=None, entry_time=None, near_miss=None, journey=None):
        emoji = ("🎯" if result == "T2" else
                 "✅" if result == "WIN" else
                 "❌" if result == "LOSS" else "⊙")
        exit_time = datetime.now(IST).strftime("%H:%M")
        result_label = ("Target 2 hit (+100%)" if result == "T2" else
                        "Target 1 hit (+50%)"  if result == "WIN" else
                        "Stop loss hit"       if result == "LOSS" else (result or "—"))
        # Sign-guard: LOSS rows must show a negative number even if the caller
        # accidentally passed a positive value (legacy paths could pass the
        # absolute magnitude). Mirrors the guard in format_daily_summary_blocks.
        raw = float(pnl or 0)
        if result == "LOSS":                       pnl_signed = -abs(raw)
        elif result in ("WIN", "T1", "T2", "TRAIL"):        pnl_signed =  abs(raw)
        else:                                      pnl_signed =  raw
        sign = "+" if pnl_signed >= 0 else "−"
        amt = int(round(abs(pnl_signed)))
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn",
                "text": f"{emoji} *Trade closed: {instrument} {direction}*  ·  *{result_label}*"}},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"📋 *Contract*\n{SlackAlert._humanize_symbol(option.get('symbol','')) if option else '—'}"},
                {"type": "mrkdwn", "text": f"💰 *P&L*\n{sign}₹{amt:,}"},
                {"type": "mrkdwn", "text": f"⏰ *Entry → Exit*\n{entry_time or '—'} → {exit_time} IST"},
                {"type": "mrkdwn", "text": f"🏷 *Outcome*\n{result}"},
            ]},
        ]
        if journey and journey.get("entry"):
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"*Premium journey*\n{SlackAlert._journey_line(journey)}"}})
        if near_miss:
            peak_amt = int(round(float(near_miss.get('peak') or 0)))
            blocks.append({"type": "section", "text": {"type": "mrkdwn",
                "text": f"⚠️ *Near-miss exit*\n"
                        f"Premium peaked at *₹{peak_amt}* at *{near_miss.get('peak_time','')}* — "
                        f"that was *{int(near_miss.get('pct_to_t1', 0))}%* of the way to T1, "
                        f"then reversed before the close trigger."}})
        return blocks

    @staticmethod
    def format_daily_summary(perf, rows=None):
        """Plain-text EOD summary (push-notification fallback)."""
        rows = rows or []
        msg = f"""📊 *DAILY SUMMARY* — {datetime.now(IST).strftime('%a · %d %b %Y')}
━━━━━━━━━━━━━━━━━
Total: {perf["total"]}  ·  ✅ {perf["wins"]} wins  ·  ❌ {perf["losses"]} losses
Win rate: *{perf["win_rate"]}%*  ·  Net P&L: *{'+' if perf["total_pnl"]>=0 else ''}₹{perf["total_pnl"]}*
🏆 Best: ₹{perf["best_trade"]}  ·  📉 Worst: ₹{perf["worst_trade"]}
"""
        if rows:
            msg += "\n*Trade-by-trade:*"
            for r in rows:
                raw_pnl = float(r.get("pnl_rupees", 0) or 0)
                result = r.get("result") or "OPEN"
                pnl_abs = abs(raw_pnl)
                pnl = -pnl_abs if result == "LOSS" else (
                       pnl_abs if result in ("WIN", "T1", "T2", "TRAIL") else raw_pnl)
                sign = "+" if pnl >= 0 else ""
                emoji = "🎯" if result == "T2" else ("✅" if result == "WIN" else
                        ("❌" if result == "LOSS" else "⊙"))
                ts_raw = (r.get('timestamp') or '')
                t_in_text = ts_raw.split(" ")[1][:5] if " " in ts_raw else (ts_raw[:5] or '—')
                line = (f"\n{emoji} {r.get('instrument','')} {r.get('direction','')} · "
                        f"{SlackAlert._humanize_symbol(r.get('option_symbol','') or '')} · "
                        f"{t_in_text}→{r.get('exit_time','—')} · "
                        f"{result} · {sign}₹{int(round(pnl))}")
                # Near-miss: peaked deep into favorable territory before SL
                peak = r.get("peak_premium")
                oent = float(r.get("option_entry") or 0)
                ot1  = float(r.get("option_target1") or 0)
                if (r.get("result") == "LOSS" and peak and oent > 0 and ot1 > oent):
                    favor = (float(peak) - oent) / (ot1 - oent) * 100
                    if favor >= 40:
                        line += f"  (⚠ peaked ₹{int(round(float(peak)))} = {int(round(favor))}% to T1)"
                msg += line
        msg += "\n━━━━━━━━━━━━━━━━━"
        return msg

    @staticmethod
    def format_daily_summary_blocks(perf, rows=None):
        """Rich Slack-blocks version of the EOD summary with a row per trade."""
        def _i(v):
            try: return f"{int(round(float(v))):,}"
            except: return str(v)
        rows = rows or []
        net_sign = "+" if perf["total_pnl"] >= 0 else "−"
        net_color_text = "Net P&L"
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn",
                "text": f"📊 *DAILY SUMMARY* — {datetime.now(IST).strftime('%a · %d %b %Y')}"}},
            {"type": "divider"},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Total signals*\n{perf['total']}"},
                {"type": "mrkdwn", "text": f"*Win rate*\n{perf['win_rate']}%"},
                {"type": "mrkdwn", "text": f"*Wins*\n✅ {perf['wins']}"},
                {"type": "mrkdwn", "text": f"*Losses*\n❌ {perf['losses']}"},
                {"type": "mrkdwn", "text": f"*{net_color_text}*\n{net_sign}₹{_i(abs(perf['total_pnl']))}"},
                {"type": "mrkdwn", "text": f"*Best / worst*\n+₹{_i(perf['best_trade'])} / −₹{_i(abs(perf['worst_trade']))}"},
            ]},
        ]
        if rows:
            blocks.append({"type": "divider"})
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Trade-by-trade*"}})
            # Slack 'context' blocks render small, dense lines — ideal for a list
            for r in rows:
                raw_pnl = float(r.get("pnl_rupees", 0) or 0)
                result = r.get("result") or "OPEN"
                # SIGN GUARD: enforce that LOSS rows display negative P&L
                # and WIN/T1/T2 rows display positive. Some historical rows
                # had a sign-mismatch from earlier code paths (option_entry=0
                # fallback caused positive premium × qty to be stored against
                # a LOSS classification). Coerce to the truth the label says.
                pnl_abs = int(round(abs(raw_pnl)))
                if result == "LOSS":
                    pnl = -pnl_abs
                elif result in ("WIN", "T1", "T2", "TRAIL"):
                    pnl = pnl_abs
                else:
                    pnl = int(round(raw_pnl))  # OPEN / EXPIRED keep their sign
                sign = "+" if pnl >= 0 else "−"
                emoji = ("🎯" if result == "T2" else
                         "✅" if result == "WIN" else
                         "❌" if result == "LOSS" else
                         "⊙")
                inst   = r.get("instrument", "")
                dirn   = r.get("direction", "")
                osym   = SlackAlert._humanize_symbol(r.get("option_symbol", "") or "")
                # timestamp is stored as "YYYY-MM-DD HH:MM:SS" — pull HH:MM
                ts_raw = (r.get("timestamp") or "")
                t_in   = ts_raw.split(" ")[1][:5] if " " in ts_raw else (ts_raw[:5] or "—")
                t_out  = (r.get("exit_time") or "—")[:5]
                # Near-miss tag for losses where premium peaked deep into
                # favorable territory ("could have exited at peak").
                peak  = r.get("peak_premium")
                ptime = r.get("peak_time")
                oent  = float(r.get("option_entry") or 0)
                ot1   = float(r.get("option_target1") or 0)
                near = ""
                if (result == "LOSS" and peak and oent > 0 and ot1 > oent):
                    favor = (float(peak) - oent) / (ot1 - oent) * 100
                    if favor >= 40:
                        near = (f"  ·  ⚠ peaked ₹{int(round(float(peak)))} "
                                f"({int(round(favor))}% to T1) at {ptime}")
                blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
                    "text": f"{emoji}  *{inst} {dirn}* · {osym} · {t_in} → {t_out} · "
                            f"*{result}* · {sign}₹{_i(abs(pnl))}{near}"}]})
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn",
            "text": "_Auto-generated at market close · brokerage + slippage applied_"}]})
        return blocks


# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════
# DB location. Auto-detect a mounted Railway volume: if /data exists and
# DB_PATH isn't set explicitly, everything (signals, swing positions, AI
# spend ledger) lands on the volume and survives redeploys with ZERO
# further configuration. Without a volume the filesystem is ephemeral and
# every deploy wipes state — mount one at /data in the Railway UI.
DB_PATH = os.environ.get("DB_PATH") or (
    "/data/signals.db" if os.path.isdir("/data") else "signals.db")
if DB_PATH.startswith("/data"):
    log.info("💾 Persistent volume detected — DB + AI ledger survive redeploys "
             f"({DB_PATH})")
else:
    log.warning("⚠️ No /data volume — DB and AI spend ledger reset on every "
                "redeploy. Add a Railway volume mounted at /data to fix.")

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
        # Cost-aware P&L: gross is the displayed pnl_rupees; these track the
        # estimated brokerage + slippage and the resulting net.
        ("brokerage_rs", "REAL"),
        ("slippage_rs", "REAL"),
        ("pnl_rupees_net", "REAL"),
        ("option_exit_realistic", "REAL"),
        # High-water / low-water marks during the trade — surface near-miss
        # exits ("could have booked at peak ₹140 before SL hit at ₹70").
        ("peak_premium", "REAL"),   ("peak_time", "TEXT"),
        ("trough_premium", "REAL"), ("trough_time", "TEXT"),
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
    # ── Swing / Positional position tracker ──────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS swing_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument TEXT NOT NULL, instrument_type TEXT,
            direction TEXT NOT NULL,
            entry_date TEXT NOT NULL, entry_time TEXT,
            spot_entry REAL, spot_sl REAL, spot_target1 REAL, spot_target2 REAL,
            option_symbol TEXT, option_strike REAL, option_type TEXT,
            option_expiry TEXT, option_token TEXT, option_dte INTEGER,
            option_entry REAL, option_sl REAL, option_target1 REAL,
            lot_size INTEGER, lots INTEGER, capital REAL,
            status TEXT DEFAULT 'OPEN',
            exit_date TEXT, exit_price REAL, option_exit REAL,
            pnl_pct REAL, pnl_rupees REAL, result TEXT,
            hold_days INTEGER,
            last_ai_decision TEXT, last_ai_reasoning TEXT, last_ai_ts TEXT,
            reasons TEXT, indicators TEXT, source TEXT DEFAULT 'AUTO'
        )
    """)
    # Migrate existing swing_positions table if needed
    sw_cols = {r[1] for r in c.execute("PRAGMA table_info(swing_positions)").fetchall()}
    for col_decl in [("last_ai_decision","TEXT"),("last_ai_reasoning","TEXT"),("last_ai_ts","TEXT"),("source","TEXT"),("hold_days","INTEGER"),("exit_reason","TEXT")]:
        col, typ = col_decl
        if col not in sw_cols:
            try: c.execute(f"ALTER TABLE swing_positions ADD COLUMN {col} {typ}")
            except Exception as e: log.warning(f"  swing migrate add {col}: {e}")

    # ── engine_state (Phase 2.6) — runtime-mutable config from dashboard ──
    # Key-value store for settings the user can toggle from the UI without
    # touching Railway env vars (e.g., strategy, dry_run_v2). Persists across
    # process restarts; survives until next Railway redeploy (filesystem
    # ephemeral on free tier). On boot, server reads this table to override
    # CONFIG defaults.
    c.execute("""
        CREATE TABLE IF NOT EXISTS engine_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    """)

    # ── gate_rejections (shadow log) ──────────────────────────────────
    # Every signal the engine generated but did NOT alert, with the gate
    # that killed it. After a couple of weeks this table answers "which
    # gate is eating profitable signals" with data instead of guesses.
    c.execute("""
        CREATE TABLE IF NOT EXISTS gate_rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, date TEXT NOT NULL,
            instrument TEXT NOT NULL, gate TEXT NOT NULL,
            direction TEXT, confidence INTEGER,
            index_price REAL, detail TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_gate_rej_date ON gate_rejections(date, gate)")

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


# ─── engine_state helpers (Phase 2.6) ─────────────────────────────────

def get_engine_state(key: str, default=None):
    """Read a runtime-mutable setting from `engine_state` table.

    Used to override CONFIG defaults at boot AND at request-time, so the user
    can switch strategy from the dashboard without touching Railway env vars.
    """
    try:
        row = db_exec("SELECT value FROM engine_state WHERE key=?", (key,), fetchone=True)
        return row["value"] if row else default
    except Exception:
        return default


def set_engine_state(key: str, value):
    """Write a runtime setting. Persists across process restarts (Railway's
    filesystem is ephemeral only across redeploys, not restarts)."""
    db_exec(
        "INSERT OR REPLACE INTO engine_state (key, value, updated_at) VALUES (?,?,?)",
        (key, str(value), datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
    )


def hydrate_runtime_config():
    """Apply persisted engine_state values over CONFIG defaults.

    Called once at boot, AND again whenever the dashboard mutates state via
    POST /api/strategy. Precedence (UPDATED 2026-05-27):
        STRATEGY env var (when explicitly set)  >  engine_state DB  >  hardcoded default

    Why env-overrides-DB: a Railway env var change wouldn't take effect
    because SQLite still had a stale 'v1' from a prior /api/strategy POST.
    Confusing for the operator. Now if STRATEGY=X is explicitly set on
    Railway, we force-write it to the DB on boot. Once set, the dashboard
    UI can still mutate via POST /api/strategy.
    """
    # Step 1: hydrate from DB
    for key in ("strategy", "dry_run_v2"):
        v = get_engine_state(key)
        if v is None: continue
        if key == "dry_run_v2":
            CONFIG[key] = (str(v).lower() == "true")
        else:
            CONFIG[key] = v
    # Step 2: env-var override for strategy (if explicitly set, write through)
    env_strategy = os.environ.get("STRATEGY", "").strip().lower()
    if env_strategy and env_strategy in ("auto", "v1", "v2", "conductor", "scalper_v3",
                                          "reverter", "nifty_windows", "deadzone_fade",
                                          "patterns"):
        if CONFIG.get("strategy") != env_strategy:
            log.info(f"🛠️  STRATEGY env var override: {CONFIG.get('strategy')} → {env_strategy}")
            CONFIG["strategy"] = env_strategy
            try:
                set_engine_state("strategy", env_strategy)
            except Exception as e:
                log.warning(f"  engine_state write failed: {e}")
    log.info(f"🛠️  Runtime config hydrated: strategy={CONFIG.get('strategy')} "
             f"dry_run_v2={CONFIG.get('dry_run_v2')}")


hydrate_runtime_config()


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

def estimate_costs(option_entry, option_exit, qty, lots):
    """Estimate brokerage + slippage in rupees for a closed option round-trip.

    Brokerage: flat rupees per lot from CONFIG (Zerodha/Angel typical: ₹100/lot RT).
    Slippage:  bps (basis points) of premium applied each side. 50bp = 0.5% per side.
               Realistic for ATM weekly options where spreads are 0.5-1.5%.
    Returns (brokerage_rs, slippage_rs, realistic_exit_price)."""
    try:
        lots = max(1, int(lots or 1))
        qty = max(1, int(qty or 0))
        bps_side = float(CONFIG.get("slippage_bps_per_side", 50)) / 10000.0  # 50 → 0.005
        per_lot = float(CONFIG.get("brokerage_per_lot_roundtrip", 100))
        brokerage = round(per_lot * lots, 2)
        # Slippage costs you on entry (paying nearer ask) and on exit (receiving nearer bid)
        # Direction-agnostic in absolute rupees: one side of premium each leg.
        entry_slip = float(option_entry or 0) * bps_side * qty
        exit_slip  = float(option_exit  or 0) * bps_side * qty
        slippage = round(entry_slip + exit_slip, 2)
        # Realistic exit price = exit minus one side of slippage (you got hit harder than mid)
        realistic_exit = round(float(option_exit or 0) * (1 - bps_side), 2) if option_exit else None
        return brokerage, slippage, realistic_exit
    except Exception:
        return 0.0, 0.0, option_exit


def update_result(sig_id, exit_price, result, pnl_pts, pnl_rs, option_exit=None,
                  option_entry=None, qty=None, lots=None):
    """Close a signal row.

    If option_entry/qty/lots are supplied, also compute and persist:
      - brokerage_rs / slippage_rs (estimated costs)
      - pnl_rupees_net (gross P&L minus those costs)
      - option_exit_realistic (slippage-adjusted exit premium)
    """
    brokerage_rs = slippage_rs = 0.0
    realistic_exit = option_exit
    pnl_net = pnl_rs
    if option_entry is not None and option_exit is not None and qty:
        brokerage_rs, slippage_rs, realistic_exit = estimate_costs(
            option_entry, option_exit, qty, lots)
        pnl_net = round((pnl_rs or 0) - brokerage_rs - slippage_rs, 0)
    db_exec("""UPDATE signals SET status='CLOSED',exit_price=?,exit_time=?,
               option_exit=?,pnl_points=?,pnl_rupees=?,result=?,
               brokerage_rs=?,slippage_rs=?,pnl_rupees_net=?,option_exit_realistic=?
               WHERE id=?""",
            (exit_price, datetime.now(IST).strftime("%H:%M:%S"),
             option_exit, pnl_pts, pnl_rs, result,
             brokerage_rs, slippage_rs, pnl_net, realistic_exit, sig_id))

def get_history(limit=100, date=None):
    if date:
        rows = db_exec("SELECT * FROM signals WHERE date=? ORDER BY id DESC LIMIT ?", (date,limit), fetch=True)
    else:
        rows = db_exec("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,), fetch=True)
    return [dict(r) for r in rows] if rows else []

def get_perf(date=None):
    """Aggregate performance over closed rows.

    Returns BOTH gross (`total_pnl`, displayed) and net (`total_pnl_net`, after
    estimated brokerage + slippage). The net figure is what your account
    actually grows by.

    `date` — optional 'YYYY-MM-DD' to restrict to a single day (used by
              kill-switch + dashboard daily-risk panel).
    """
    if date:
        rows = db_exec("SELECT * FROM signals WHERE status='CLOSED' AND date=?", (date,), fetch=True)
    else:
        rows = db_exec("SELECT * FROM signals WHERE status='CLOSED'", fetch=True)
    if not rows:
        return {"total":0,"wins":0,"losses":0,"win_rate":0,
                "total_pnl":0,"total_pnl_net":0,
                "total_brokerage":0,"total_slippage":0,
                "avg_win":0,"avg_loss":0,"best_trade":0,"worst_trade":0}
    rows = [dict(r) for r in rows]
    wins = [r for r in rows if r["result"] in ("WIN", "T1", "T2", "TRAIL")]
    losses = [r for r in rows if r["result"] == "LOSS"]
    # Sign-guard: same defensive normalization the formatters use, so the
    # aggregate Net P&L can't show "+₹2,606" when every trade is a LOSS.
    def _signed_pnl(r):
        raw = float(r.get("pnl_rupees") or 0)
        if r.get("result") == "LOSS":               return -abs(raw)
        if r.get("result") in ("WIN", "T1", "T2", "TRAIL"):  return  abs(raw)
        return raw
    def _signed_pnl_net(r):
        raw = r.get("pnl_rupees_net")
        if raw is None: raw = r.get("pnl_rupees") or 0
        raw = float(raw)
        if r.get("result") == "LOSS":               return -abs(raw)
        if r.get("result") in ("WIN", "T1", "T2", "TRAIL"):  return  abs(raw)
        return raw
    pnls     = [_signed_pnl(r)     for r in rows]
    pnls_net = [_signed_pnl_net(r) for r in rows]
    brk      = [r.get("brokerage_rs") or 0 for r in rows]
    slp      = [r.get("slippage_rs")  or 0 for r in rows]
    return {
        "total":len(rows),"wins":len(wins),"losses":len(losses),
        "win_rate":round(len(wins)/len(rows)*100,1) if rows else 0,
        "total_pnl":round(sum(pnls),0),
        "total_pnl_net":round(sum(pnls_net),0),
        "total_brokerage":round(sum(brk),0),
        "total_slippage":round(sum(slp),0),
        "avg_win":round(sum(abs(r["pnl_rupees"] or 0) for r in wins)/len(wins),0) if wins else 0,
        "avg_loss":round(sum(-abs(r["pnl_rupees"] or 0) for r in losses)/len(losses),0) if losses else 0,
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
        # Candle cache: (token, exchange, interval, days) -> {"ts": epoch, "df": DataFrame}
        # 5-min candles only refresh every 5 min, so default 90s TTL is safe and cuts
        # getCandleData calls dramatically. TTL set via CONFIG["candle_cache_ttl"].
        self._candle_cache = {}
        self._candle_cache_hits = 0
        self._candle_cache_misses = 0

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
        # Angel One sessions expire after ~24h. Re-login proactively after 4h to avoid
        # silent data failures mid-session. Also re-login immediately if disconnected.
        if self.last_login and (datetime.now(IST)-self.last_login).total_seconds() > 14400:
            log.info("🔄 Session 4h+ old — proactive re-login")
            return self.login()
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
    
    def candles(self, token, exchange, interval="FIVE_MINUTE", days=3, force_refresh=False,
                from_dt=None, to_dt=None):
        """Fetch OHLCV candles. Cached for `CONFIG['candle_cache_ttl']` seconds (default 90s).

        Two modes:
          - DEFAULT (days=N): fetches the last N days ending at now. Used by the live scanner.
          - HISTORICAL (from_dt + to_dt supplied): fetches the EXACT window. Used by
            the backtest replay + /api/replay-premium so we get the right slice for
            arbitrary historical timestamps. Previously this was silently broken —
            get_spot_bars was passing days=1 and Angel returned "last 24h" instead
            of the requested historical window.

        Pass force_refresh=True to bypass cache."""
        # Cache key incorporates from/to_dt so historical queries don't collide
        # with live "last N days" queries for the same token.
        if from_dt is not None and to_dt is not None:
            cache_key = (str(token), exchange, interval,
                         from_dt.strftime("%Y%m%d%H%M"), to_dt.strftime("%Y%m%d%H%M"))
        else:
            cache_key = (str(token), exchange, interval, int(days))
        ttl = int(CONFIG.get("candle_cache_ttl", 90) or 0)
        now = time.time()
        if not force_refresh and ttl > 0:
            cached = self._candle_cache.get(cache_key)
            if cached and (now - cached["ts"] < ttl):
                self._candle_cache_hits += 1
                # Return a copy so downstream mutations don't poison the cache
                return cached["df"].copy()
        self._candle_cache_misses += 1
        try:
            if not self.ensure(): return pd.DataFrame()
            # ── Use explicit window if provided; otherwise rolling-N-days ──
            if from_dt is not None and to_dt is not None:
                fromdate = from_dt.strftime("%Y-%m-%d %H:%M")
                todate   = to_dt.strftime("%Y-%m-%d %H:%M")
            else:
                fromdate = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
                todate   = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            _params = {"exchange":exchange,"symboltoken":token,"interval":interval,
                "fromdate":fromdate, "todate":todate}
            import concurrent.futures as _cf2
            # Angel's getCandleData intermittently returns an empty/failed payload for a
            # perfectly valid window (transient throttle / hiccup). The old code gave up on
            # the FIRST empty response, which surfaced as random "no candles" gaps — e.g.
            # days=1 FIVE_MINUTE returning [] on one call and 75 on the next, identical params.
            # That also cost the live scanner a scan whenever Angel hiccuped. Retry a few
            # times with a short backoff before treating the window as genuinely empty.
            resp = None
            for _attempt in range(3):
                _rl = False
                try:
                    with _cf2.ThreadPoolExecutor(max_workers=1) as _ex2:
                        resp = _ex2.submit(self.api.getCandleData, _params).result(timeout=12)
                except Exception as _ce:
                    # The SmartAPI client raises here when Angel returns a non-JSON body —
                    # most commonly "Access denied because of exceeding access rate" (rate
                    # limit). Catch it INSIDE the loop so we retry instead of falling through
                    # to the outer handler and returning empty (the original bug).
                    _m = str(_ce)
                    _rl = ("rate" in _m.lower()) or ("access denied" in _m.lower())
                    log.warning(f"⚠️ getCandleData attempt {_attempt+1}/3 failed: {_m[:90]}")
                    resp = None
                if resp and resp.get("status") and resp.get("data"):
                    break
                if _attempt < 2:
                    self._candle_retries = getattr(self, "_candle_retries", 0) + 1
                    # back off longer when rate-limited so Angel's window clears;
                    # 1.3s was too short — rate limit persists for ~10-30s so use 10s steps
                    time.sleep((10 if _rl else 1) * (_attempt + 1))
            if resp and resp.get("status") and resp.get("data"):
                df = pd.DataFrame(resp["data"], columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                if ttl > 0:
                    # Cap cache size: drop oldest entries if it grows beyond ~50 keys
                    if len(self._candle_cache) > 50:
                        oldest = sorted(self._candle_cache.items(), key=lambda kv: kv[1]["ts"])[:10]
                        for k, _ in oldest: self._candle_cache.pop(k, None)
                    self._candle_cache[cache_key] = {"ts": now, "df": df.copy()}
                return df
            log.warning(f"⚠️ getCandleData returned no data after 3 attempts: "
                        f"token={token} {interval} {fromdate}→{todate}")
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Candle err: {e}"); return pd.DataFrame()

    def candle_cache_stats(self):
        """Cache hit/miss telemetry — surfaced via /api/metrics."""
        total = self._candle_cache_hits + self._candle_cache_misses
        rate = round(self._candle_cache_hits / total * 100, 1) if total else 0
        return {"hits": self._candle_cache_hits, "misses": self._candle_cache_misses,
                "hit_rate_pct": rate, "size": len(self._candle_cache)}
    
    def daily_candles(self, token, exchange="NSE", days=90):
        """Fetch daily OHLCV candles — used for swing analysis.
        Returns list of dicts {ts, open, high, low, close, volume} or []."""
        try:
            if not self.ensure(): return []
            from_dt = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d 09:15")
            to_dt   = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            params  = {"exchange": exchange, "symboltoken": str(token),
                       "interval": "ONE_DAY", "fromdate": from_dt, "todate": to_dt}
            import concurrent.futures as _cf3
            with _cf3.ThreadPoolExecutor(max_workers=1) as _ex3:
                try:
                    resp = _ex3.submit(self.api.getCandleData, params).result(timeout=15)
                except _cf3.TimeoutError:
                    log.error(f"[Swing] daily_candles timeout for token {token}")
                    return []
            if resp and resp.get("status") and resp.get("data"):
                raw = resp["data"]  # [[ts, o, h, l, c, v], ...]
                return [{"ts": r[0], "open": float(r[1]), "high": float(r[2]),
                         "low": float(r[3]), "close": float(r[4]),
                         "volume": float(r[5])} for r in raw if len(r) >= 6]
            return []
        except Exception as e:
            log.error(f"[Swing] daily_candles err token={token}: {e}")
            return []

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

                        # Extract OI, volume, and other rich market data from FULL response
                        # Angel One SmartAPI uses "opnInterest" (not "openInterest")
                        # Try all known field variants across different API builds
                        oi    = int(float(
                            item.get("opnInterest") or item.get("openInterest") or
                            item.get("oi") or 0))
                        vol   = int(float(
                            item.get("tradeVolume") or item.get("totalTradedVolume") or
                            item.get("netTradedVolume") or item.get("volume") or 0))
                        gamma  = float(item.get("gamma", 0) or 0)
                        vega   = float(item.get("vega", 0) or 0)
                        # Some SmartAPI builds embed Greeks in the FULL response too
                        delta_raw = float(item.get("delta", 0) or 0)
                        iv_raw    = float(item.get("impliedVolatility", 0)
                                         or item.get("iv", 0) or 0)
                        theta_raw = float(item.get("theta", 0) or 0)

                        opts.append({
                            "strike": tk["strike"], "type": tk["type"],
                            "symbol": tk["symbol"],
                            "ltp": mid,        # price used downstream = mid, not last-trade
                            "last_trade": ltp, # kept for diagnostics
                            "bid": best_bid, "ask": best_ask,
                            "spread": round(spread, 2),
                            "token": tok, "expiry": tk.get("expiry", ""),
                            # Rich market data for AI + scoring
                            "oi": oi, "volume": vol,
                            "gamma": round(gamma, 5), "vega": round(vega, 3),
                            "delta_raw": round(delta_raw, 3),
                            "iv_raw": round(iv_raw, 2), "theta_raw": round(theta_raw, 4),
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
        self.nse_eq = {}   # symbol_upper → {token, lot_size}  (NSE equity)
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

            # Also index NSE equity for swing daily candle lookups
            eq_count = 0
            self.nse_eq = {}
            for item in self.data:
                if item.get("exch_seg") != "NSE": continue
                itype = item.get("instrumenttype", "")
                # EQ = equity, AMXIDX = index
                if itype not in ("EQ", "AMXIDX", ""): continue
                sym = (item.get("symbol") or "").upper()
                name = (item.get("name") or item.get("symbol") or "").upper()
                tok = item.get("token", "")
                if not tok or not sym: continue
                # Store by both symbol (e.g. "RELIANCE-EQ") and name (e.g. "RELIANCE")
                self.nse_eq[sym] = {"token": tok, "symbol": sym, "name": name}
                bare = sym.replace("-EQ", "").replace("-BE", "")
                if bare and bare not in self.nse_eq:
                    self.nse_eq[bare] = {"token": tok, "symbol": sym, "name": name}
                eq_count += 1
            log.info(f"  Master: {eq_count} NSE equity instruments indexed")

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

    def find_equity_token(self, symbol):
        """Return NSE equity token for a stock symbol (e.g. 'RELIANCE' or 'RELIANCE-EQ')."""
        if not self.ensure(): return None
        sym = symbol.upper().replace(" ", "").replace("&", "&")
        hit = self.nse_eq.get(sym) or self.nse_eq.get(sym + "-EQ") or self.nse_eq.get(sym.replace("&", ""))
        if hit: return hit["token"]
        # Fuzzy: try dropping common suffixes/prefixes
        for candidate, info in self.nse_eq.items():
            if candidate.startswith(sym) or sym.startswith(candidate):
                return info["token"]
        return None

    def find_swing_options(self, fo_name, spot, direction, min_dte=15, num_strikes=5):
        """Like find_options but prefers expiry with ≥ min_dte days remaining.
        For swing trades we want time — weekly expiry is too short."""
        if not self.ensure(): return []
        today = datetime.now(IST).date()
        # Collect all future expiries for this name
        expiries_by_date = {}
        for (name, strike, otype, expiry), info in self.nfo.items():
            if name != fo_name: continue
            try:
                d = datetime.strptime(expiry, "%d%b%Y").date()
                if d >= today:
                    expiries_by_date[expiry] = d
            except: continue
        if not expiries_by_date:
            log.debug(f"  [Swing] No future expiries for {fo_name}")
            return []
        # Sort and pick first expiry with ≥ min_dte; fallback to nearest
        sorted_exp = sorted(expiries_by_date.items(), key=lambda x: x[1])
        chosen_exp, chosen_date = None, None
        for exp, d in sorted_exp:
            dte = (d - today).days
            if dte >= min_dte:
                chosen_exp, chosen_date = exp, d
                break
        if not chosen_exp:
            # All expiries are < min_dte — take the furthest available
            chosen_exp, chosen_date = sorted_exp[-1]
        dte = (chosen_date - today).days
        log.info(f"  [Swing] {fo_name} expiry {chosen_exp} ({dte} DTE)")
        # Generate strikes around spot
        gap = 1
        for info in SWING_STOCKS.values():
            if info.get("nse_fo") == fo_name:
                gap = info.get("strike_gap", 50); break
        atm = round(spot / gap) * gap
        strikes = [atm + i * gap for i in range(-num_strikes, num_strikes + 1)]
        right = "CE" if direction == "LONG" else "PE"
        results = []
        for s in strikes:
            key = (fo_name, float(s), right, chosen_exp)
            info = self.nfo.get(key)
            if info:
                results.append({**info, "dte": dte})
        log.info(f"  [Swing] Found {len(results)} {right} option tokens for {fo_name}")
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

    # Class-level tracker so /api/status can surface which strategy actually
    # ran on the most recent scan — the previous silent-fallback hid the fact
    # that Conductor was crashing on every call and v1 was running instead.
    last_dispatch: dict = {"strategy": None, "actually_ran": None, "error": None, "ts": None}

    @staticmethod
    def trend_day_score(df):
        """0-5 trend-day PRE-classifier (Crabel/Raschke family + relative
        volume) — fires at the open, hours before ADX(10) can confirm:
          +1 gap >= 0.4% of prior close
          +1 gap survives (first 5-min bar closes in the gap direction)
          +1 prior day compressed (< 0.6x avg of available prior day ranges)
          +1 open outside the prior day's range
          +1 first-15-min volume >= 1.5x the same window on prior days
        Returns (score, diag) or (None, {}) when there's no prior-day data.
        """
        try:
            ts_col = "ts" if "ts" in df.columns else "timestamp"
            ts = pd.to_datetime(df[ts_col])
            dates = ts.dt.date
            today = dates.iloc[-1]
            tdf = df.loc[dates == today]
            prev_dates = sorted(set(d for d in dates if d != today))
            if len(tdf) < 1 or not prev_dates:
                return None, {}
            yday = df.loc[dates == prev_dates[-1]]
            prev_close = float(yday["close"].iloc[-1])
            prev_high = float(yday["high"].max())
            prev_low = float(yday["low"].min())
            open_ = float(tdf["open"].iloc[0])
            gap = open_ - prev_close
            gap_pct = abs(gap) / prev_close * 100 if prev_close else 0
            score = 0
            d = {"gap_pct": round(gap_pct, 2)}
            if gap_pct >= 0.4:
                score += 1
            b1o, b1c = float(tdf["open"].iloc[0]), float(tdf["close"].iloc[0])
            if gap_pct >= 0.15 and ((gap > 0 and b1c > b1o) or (gap < 0 and b1c < b1o)):
                score += 1   # gap survival
            day_ranges = []
            for dt_ in prev_dates:
                dd = df.loc[dates == dt_]
                day_ranges.append(float(dd["high"].max() - dd["low"].min()))
            if day_ranges and (prev_high - prev_low) < 0.6 * (sum(day_ranges) / len(day_ranges)):
                score += 1   # compression before expansion
            if open_ > prev_high or open_ < prev_low:
                score += 1   # open outside prior range
            if "volume" in df.columns and len(tdf) >= 3:
                v_now = float(tdf["volume"].iloc[:3].sum())
                prior_v = []
                for dt_ in prev_dates:
                    dd = df.loc[dates == dt_]
                    if len(dd) >= 3:
                        prior_v.append(float(dd["volume"].iloc[:3].sum()))
                if prior_v and v_now >= 1.5 * (sum(prior_v) / len(prior_v)):
                    score += 1   # opening relative volume
            d["score"] = score
            return score, d
        except Exception:
            return None, {}

    @staticmethod
    def classify_regime(df):
        """Classify the CURRENT intraday regime as 'trend' or 'range' using
        ADX(14) on the 5-min bars, with the VWAP-cross count as a tiebreak.

        Thresholds (env-tunable, defaults from practitioner literature —
        ADX(14) is too slow on 5-min bars, period 10 is the documented
        compromise; <20 = range regime, >25 = trend regime):
          AUTO_ADX_PERIOD (default 10)
          AUTO_ADX_TREND (default 25) — ADX at/above this = trending
          AUTO_ADX_RANGE (default 20) — ADX at/below this = ranging
          Between the two, price chopping across VWAP (≥4 crosses today)
          resolves to 'range', otherwise 'trend'.

        Returns (label, diag). Fails open to 'trend' (Conductor is the only
        engine with validated live edge, so it keeps priority on errors).
        """
        try:
            # ── Early-session pre-classifier ─────────────────────────────
            # ADX(10) needs ~an hour of bars to see a trend; gap structure,
            # compression, and opening relative volume classify the day at
            # 09:30. Until AUTO_PRECLASS_UNTIL (bar time, default 10:30) the
            # pre-classifier outranks ADX.
            pre_until = os.environ.get("AUTO_PRECLASS_UNTIL", "10:30")
            try:
                _tc = "ts" if "ts" in df.columns else "timestamp"
                bar_hm = pd.to_datetime(df[_tc].iloc[-1]).strftime("%H:%M")
            except Exception:
                bar_hm = "12:00"
            if bar_hm < pre_until:
                score, sdiag = SignalGen.trend_day_score(df)
                if score is not None:
                    if score >= 3:
                        return "trend", {"pre_score": score, **sdiag}
                    if score <= 1 and sdiag.get("gap_pct", 1.0) < 0.28:
                        # small gap + no trend markers: 62% of sub-70pt NIFTY
                        # gaps fill within 90 min — mean-reversion morning
                        return "range", {"pre_score": score, **sdiag}
            adx_period = int(os.environ.get("AUTO_ADX_PERIOD", "10"))
            adx_trend = float(os.environ.get("AUTO_ADX_TREND", "25"))
            adx_range = float(os.environ.get("AUTO_ADX_RANGE", "20"))
            adx_ser, _pdi, _mdi = TA.adx(df, adx_period)
            adx_now = float(adx_ser.iloc[-1])
            adx_is_nan = adx_now != adx_now   # NaN (early window / dead tape)
            if not adx_is_nan:
                if adx_now >= adx_trend:
                    return "trend", {"adx": round(adx_now, 1)}
                if adx_now <= adx_range:
                    return "range", {"adx": round(adx_now, 1)}
            adx_now = None if adx_is_nan else adx_now
            # NaN or grey zone → fall through to the VWAP-cross tiebreak.
            # Grey zone: count today's VWAP crosses — a magnet day is a range day
            ts_col = "ts" if "ts" in df.columns else "timestamp"
            ts_ser = pd.to_datetime(df[ts_col])
            today = df.loc[ts_ser.dt.date == ts_ser.dt.date.iloc[-1]]
            if len(today) >= 6:
                tp = (today["high"] + today["low"] + today["close"]) / 3
                vol = today["volume"].replace(0, 1).fillna(1) if "volume" in today.columns \
                    else pd.Series(1.0, index=today.index)
                vwap = (tp * vol).cumsum() / vol.cumsum()
                above = (today["close"] > vwap)
                crosses = int((above != above.shift(1)).fillna(False).sum())
                label = "range" if crosses >= 4 else "trend"
                return label, {"adx": (round(adx_now, 1) if adx_now is not None else None),
                                "vwap_crosses": crosses}
            return "trend", {"adx": (round(adx_now, 1) if adx_now is not None else None),
                              "note": "few_bars"}
        except Exception as e:
            return "trend", {"adx": None, "note": f"classify_err:{type(e).__name__}"}

    def analyze(self, df, weight_adj=None, blocked_windows=None, symbol=None,
                chain_analytics=None):
        """Score candle data into a directional signal.

        Routes to strategy named by CONFIG['strategy']. Supported:
          • 'auto'      — regime router: ADX trend day → conductor,
                          range day → reverter (recommended default)
          • 'conductor' — 5-dim orthogonal confluence scout (production)
          • 'v2'        — legacy 4-of-6 confluence (parked, anti-edge confirmed)
          • 'v1'        — original heuristic (default fallback)

        weight_adj      — dict with -5..+5 deltas applied to the base contribution
                          for each indicator family. (v1-only)
        blocked_windows — list of "HH:MM-HH:MM" strings. (v1-only)
        symbol          — instrument name (NIFTY/BANKNIFTY/FINNIFTY) — required
                          by conductor & gamma analyzers
        chain_analytics — live option-chain analytics (PCR, IV skew, OI velocity)
                          for conductor's flow dimension. None means flow=0.

        DISPATCH POLICY (post-2026-06-01 bug):
          • Normalize column name 'timestamp' → 'ts' once at the top. Live
            AngelClient.candles produces 'timestamp'; backtest_v2 produces
            'ts'. Conductor/ORB/Gamma/Scalper all read df['ts']. Without
            this normalization, every live conductor call raised KeyError
            and the silent fallback ran v1 instead — 2 weeks of zero alerts.
          • NEVER silently fall back to v1 on strategy crash. Loud-fail
            (log + return None) so the dashboard's signal stream stays
            empty rather than secretly running the wrong code.
          • Record actual run path in SignalGen.last_dispatch for /api/status.
        """
        # Per-instrument routing: INSTRUMENT_STRATEGIES env var lets us run
        # different strategies per index simultaneously. Format:
        #   "BANKNIFTY=conductor,NIFTY=scalper_v3,FINNIFTY=conductor"
        # Falls through to CONFIG["strategy"] if symbol is not in the map.
        # Lets us keep Conductor on trending BANKNIFTY while running a
        # scalper on range-bound NIFTY without changing CONFIG globally.
        strategy_global = CONFIG.get("strategy", "v1").lower()
        strategy = strategy_global
        per_instr_map = {}
        per_env = os.environ.get("INSTRUMENT_STRATEGIES", "").strip()
        if per_env:
            for pair in per_env.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    per_instr_map[k.strip().upper()] = v.strip().lower()
        if symbol and symbol.upper() in per_instr_map:
            strategy = per_instr_map[symbol.upper()]

        SignalGen.last_dispatch = {
            "strategy": strategy, "actually_ran": None,
            "error": None, "ts": datetime.now(IST).strftime("%H:%M:%S"),
            "symbol": symbol, "global_strategy": strategy_global,
        }

        # Column normalization: backtest passes 'ts', live passes 'timestamp'.
        # Strategies were written against 'ts'. One line shields all of them.
        if df is not None and not df.empty and "timestamp" in df.columns and "ts" not in df.columns:
            df = df.rename(columns={"timestamp": "ts"})

        # ── Layer C learned time-window block — ALL strategies ────────────
        # Claude's EOD review writes "avoid 11:30-12:30" style windows after a
        # losing day. These previously gated only the parked v1 path, so the
        # entire learning loop was a no-op for the strategies actually running.
        if blocked_windows:
            now_hm = datetime.now(IST).strftime("%H:%M")
            for _win in blocked_windows:
                try:
                    _a, _b = _win.split("-")
                    if _a.strip() <= now_hm <= _b.strip():
                        SignalGen.last_dispatch["actually_ran"] = "blocked_window"
                        SignalGen.last_dispatch["blocked_window"] = _win
                        return None
                except Exception:
                    continue

        # ── AUTO regime router: trend day → Conductor, range day → Reverter ──
        # The two engines are complements built for opposite regimes; running
        # only Conductor starved range-bound days (most NIFTY days) of any
        # signal source. ADX(14) on the 5-min bars picks the engine per scan.
        if strategy == "auto":
            regime_label, rdiag = SignalGen.classify_regime(df)
            # Midday dead zone (U-shaped intraday volatility: realized vol
            # troughs ~12:00-13:45 IST while theta bleeds continuously) —
            # trend entries there are the documented negative-EV zone, so the
            # dead zone always routes to the mean-reversion engine.
            try:
                _dz_a = os.environ.get("AUTO_DZ_START", "12:00")
                _dz_b = os.environ.get("AUTO_DZ_END", "13:45")
                _hm = datetime.now(IST).strftime("%H:%M")
                if _dz_a <= _hm <= _dz_b:
                    regime_label = "range"
                    rdiag = dict(rdiag); rdiag["note"] = "midday_dead_zone"
            except Exception:
                pass
            SignalGen.last_dispatch["auto_regime"] = regime_label
            SignalGen.last_dispatch["auto_adx"] = rdiag.get("adx")
            engine_mod = "conductor" if regime_label == "trend" else "reverter"
            try:
                if engine_mod == "conductor":
                    from conductor import Conductor
                    result = Conductor.analyze(df, symbol=symbol or "",
                                                chain_analytics=chain_analytics)
                else:
                    from signal_reverter import Reverter
                    result = Reverter.analyze(df, symbol=symbol or "",
                                               chain_analytics=chain_analytics)
                SignalGen.last_dispatch["actually_ran"] = f"auto:{engine_mod}"
                if result is not None:
                    result["regime_route"] = regime_label
                    result.setdefault("reasons", []).append(
                        f"Regime router: {regime_label} day (ADX {rdiag.get('adx', '?')}) → {engine_mod}")
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ auto/{engine_mod} crashed — NO fallback, returning None: {e}", exc_info=True)
                return None

        # ── Conductor dispatch (production for BANKNIFTY) ─────────────────
        if strategy == "conductor":
            try:
                from conductor import Conductor
                result = Conductor.analyze(df, symbol=symbol or "",
                                            chain_analytics=chain_analytics)
                SignalGen.last_dispatch["actually_ran"] = "conductor"
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ conductor crashed — NO fallback, returning None: {e}", exc_info=True)
                return None  # loud-fail: do NOT run v1 silently

        # ── ScalperV3 dispatch (NIFTY scalp candidate) ────────────────────
        if strategy in ("scalper_v3", "scalper3"):
            try:
                from signal_scalper_v3 import ScalperV3
                result = ScalperV3.analyze(df, symbol=symbol or "",
                                            chain_analytics=chain_analytics)
                SignalGen.last_dispatch["actually_ran"] = "scalper_v3"
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ scalper_v3 crashed — NO fallback, returning None: {e}", exc_info=True)
                return None

        # ── Reverter dispatch (NIFTY mean-reversion candidate) ───────────
        if strategy == "reverter":
            try:
                from signal_reverter import Reverter
                result = Reverter.analyze(df, symbol=symbol or "",
                                           chain_analytics=chain_analytics)
                SignalGen.last_dispatch["actually_ran"] = "reverter"
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ reverter crashed — NO fallback, returning None: {e}", exc_info=True)
                return None

        # ── NiftyWindows dispatch (NIFTY: Conductor in vol windows) ──────
        if strategy == "nifty_windows":
            try:
                from signal_nifty_regime import NiftyWindows
                result = NiftyWindows.analyze(df, symbol=symbol or "",
                                              chain_analytics=chain_analytics)
                SignalGen.last_dispatch["actually_ran"] = "nifty_windows"
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ nifty_windows crashed — NO fallback, returning None: {e}", exc_info=True)
                return None

        # ── DeadzoneFade dispatch (NIFTY: VWAP fade in 11:00-13:15) ──────
        if strategy == "deadzone_fade":
            try:
                from signal_nifty_regime import DeadzoneFade
                result = DeadzoneFade.analyze(df, symbol=symbol or "",
                                              chain_analytics=chain_analytics)
                SignalGen.last_dispatch["actually_ran"] = "deadzone_fade"
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ deadzone_fade crashed — NO fallback, returning None: {e}", exc_info=True)
                return None

        # ── PatternScanner dispatch (chart patterns — UNVALIDATED) ───────
        if strategy == "patterns":
            try:
                from signal_patterns import PatternScanner
                result = PatternScanner.analyze(df, symbol=symbol or "",
                                                chain_analytics=chain_analytics)
                SignalGen.last_dispatch["actually_ran"] = "patterns"
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ patterns crashed — NO fallback, returning None: {e}", exc_info=True)
                return None

        # ── v2 dispatch (Phase 2 legacy) ──────────────────────────────────
        if strategy == "v2":
            try:
                from signal_v2 import SignalGenV2
                result = SignalGenV2.analyze(df)
                SignalGen.last_dispatch["actually_ran"] = "v2"
                return result
            except Exception as e:
                SignalGen.last_dispatch["actually_ran"] = None
                SignalGen.last_dispatch["error"] = f"{type(e).__name__}: {e}"
                log.error(f"❌ v2 crashed — NO fallback, returning None: {e}", exc_info=True)
                return None  # loud-fail

        SignalGen.last_dispatch["actually_ran"] = "v1"

        if len(df)<30: return None
        # Time-window block (Layer C): "skip 11:30-12:30 today" type guidance
        if blocked_windows:
            now_hm = datetime.now(IST).strftime("%H:%M")
            for win in blocked_windows:
                try:
                    a, b = win.split("-")
                    if a.strip() <= now_hm <= b.strip():
                        return None
                except Exception:
                    continue
        wa = weight_adj or {}
        w_rsi  = int(wa.get("rsi", 0) or 0)
        w_macd = int(wa.get("macd", 0) or 0)
        w_st   = int(wa.get("supertrend", 0) or 0)
        w_vwap = int(wa.get("vwap", 0) or 0)
        w_ema  = int(wa.get("ema", 0) or 0)
        w_vol  = int(wa.get("volume", 0) or 0)
        c=df["close"];n=len(df)-1;price=c.iloc[n]
        e9=TA.ema(c,9);e21=TA.ema(c,21);e50=TA.ema(c,min(50,len(c)))
        rsi=TA.rsi(c);ml,sl,mh=TA.macd(c);bbu,bbm,bbl=TA.bb(c)
        vwap=TA.vwap(df);atr=TA.atr(df);st=TA.supertrend(df)
        sk=TA.stoch(df);adx,pdi,mdi=TA.adx(df)
        vra=df["volume"].tail(20).mean();vr=df["volume"].iloc[n]/vra if vra>0 else 1

        bs,be=0,0;br,ber=[],[]
        if e9.iloc[n]>e21.iloc[n] and e9.iloc[n-1]<=e21.iloc[n-1]:bs+=15+w_ema;br.append("🔥 EMA 9/21 Bullish Crossover")
        elif e9.iloc[n]<e21.iloc[n] and e9.iloc[n-1]>=e21.iloc[n-1]:be+=15+w_ema;ber.append("🔥 EMA 9/21 Bearish Crossover")
        elif e9.iloc[n]>e21.iloc[n]:bs+=8+max(0,w_ema);br.append("EMA 9>21 bullish")
        else:be+=8+max(0,w_ema);ber.append("EMA 9<21 bearish")
        if price>e50.iloc[n]:bs+=5;br.append("Above EMA 50")
        else:be+=5;ber.append("Below EMA 50")
        rv=rsi.iloc[-1]
        if rv<30:bs+=12+w_rsi;br.append(f"RSI Oversold ({rv:.1f})")
        elif rv>70:be+=12+w_rsi;ber.append(f"RSI Overbought ({rv:.1f})")
        elif 50<rv<65:bs+=6+max(0,w_rsi);br.append(f"RSI Bullish ({rv:.1f})")
        elif 35<rv<50:be+=6+max(0,w_rsi);ber.append(f"RSI Bearish ({rv:.1f})")
        if mh.iloc[n]>0 and mh.iloc[n-1]<=0:bs+=15+w_macd;br.append("🔥 MACD Bull Cross")
        elif mh.iloc[n]<0 and mh.iloc[n-1]>=0:be+=15+w_macd;ber.append("🔥 MACD Bear Cross")
        elif mh.iloc[n]>mh.iloc[n-1] and mh.iloc[n]>0:bs+=8+max(0,w_macd);br.append("MACD rising")
        elif mh.iloc[n]<mh.iloc[n-1] and mh.iloc[n]<0:be+=8+max(0,w_macd);ber.append("MACD falling")
        if price<=bbl.iloc[n]*1.002:bs+=10;br.append("At Lower BB")
        elif price>=bbu.iloc[n]*0.998:be+=10;ber.append("At Upper BB")
        if price>vwap.iloc[n] and c.iloc[n-1]<=vwap.iloc[n-1]:bs+=10+w_vwap;br.append("🔥 Crossed above VWAP")
        elif price<vwap.iloc[n] and c.iloc[n-1]>=vwap.iloc[n-1]:be+=10+w_vwap;ber.append("🔥 Crossed below VWAP")
        elif price>vwap.iloc[n]:bs+=5+max(0,w_vwap);br.append("Above VWAP")
        else:be+=5+max(0,w_vwap);ber.append("Below VWAP")
        if st.iloc[n]==1 and st.iloc[n-1]==-1:bs+=13+w_st;br.append("🔥 Supertrend BULL")
        elif st.iloc[n]==-1 and st.iloc[n-1]==1:be+=13+w_st;ber.append("🔥 Supertrend BEAR")
        elif st.iloc[n]==1:bs+=7+max(0,w_st);br.append("Supertrend Bull")
        else:be+=7+max(0,w_st);ber.append("Supertrend Bear")
        if vr>1.5:
            t=f"Volume {vr:.1f}x"
            if c.iloc[n]>c.iloc[n-1]:bs+=8+w_vol;br.append(t)
            else:be+=8+w_vol;ber.append(t)
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
        
        # Hard time gate: no new entries after 14:50 (not enough time for trade to run)
        # Use a hard block rather than just penalty — any signal here is low quality
        now_hr = datetime.now(IST).hour
        now_min = datetime.now(IST).minute
        if now_hr >= 15 or (now_hr == 14 and now_min >= 50):
            return None   # Hard block — return no signal rather than a useless late one
        if now_hr == 14 and now_min >= 30:
            conf=max(10,conf-5); penalties.append("Late session — theta decay risk")
        
        # Margin too thin: bull-bear spread too narrow = truly ambiguous signal
        # Raised threshold from 8 → 5 (was too aggressive, killed valid trending signals)
        spread = abs(bs - be)
        if spread < 5:
            conf=max(10,conf-6); penalties.append(f"Bull/Bear split close ({bs}B vs {be}S)")

        reasons = br if direction=="LONG" else ber
        if penalties:
            reasons = reasons + [f"⚠️ {p}" for p in penalties]
        
        if direction=="LONG":
            entry=round(price+av*0.1,2);stop=round(price-av*1.2,2)
            risk_dist=round(abs(entry-stop),2)
            t1,t2=round(entry+risk_dist*2.0,2),round(entry+risk_dist*3.0,2)
        else:
            entry=round(price-av*0.1,2);stop=round(price+av*1.2,2)
            risk_dist=round(abs(entry-stop),2)
            t1,t2=round(entry-risk_dist*2.0,2),round(entry-risk_dist*3.0,2)
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

# In-process aggregate cache telemetry — surfaced via /api/metrics so the dashboard
# can show hit/miss/cost. Updated on every _anthropic_call response.
_ANTHROPIC_USAGE = {
    "calls": 0, "errors": 0,
    "input_tokens": 0, "output_tokens": 0,
    "cache_read_tokens": 0, "cache_creation_tokens": 0,
    "by_layer": {},   # layer_name -> dict (same shape, plus calls)
}


# ─── AI monthly spend cap ─────────────────────────────────────────────
# Hard budget lock: once the month's estimated Anthropic spend reaches
# AI_MONTHLY_CAP_INR (default ₹1,000), every further AI call is refused
# until the calendar month rolls over. All layers already handle a None
# return gracefully (Layer B fails open, Layer D/regime/EOD just skip),
# so the ENGINE keeps trading — it only loses the AI garnish.
# Ledger lives in engine_state (SQLite): survives restarts, but NOT a
# Railway redeploy until the /data volume + DB_PATH are set up — until
# then the cap is per deploy-month, which can only under-count, never
# let spend run away silently past a running month's ledger.
_AI_PRICING_USD_PER_M = {   # (input, output, cache_write, cache_read)
    "haiku":  (1.0,  5.0,  1.25,  0.10),
    "sonnet": (3.0, 15.0,  3.75,  0.30),
    "opus":  (15.0, 75.0, 18.75,  1.50),
}
_AI_CAP_STATE = {"warned80": None, "capped_log": 0.0, "capped_slack": None}

def _ai_month_key():
    return "ai_spend_inr_" + datetime.now(IST).strftime("%Y-%m")

# Ledger mirror file: lives next to the DB (i.e. on the /data volume when
# one is mounted, or wherever AI_LEDGER_PATH points). The ledger reads the
# MAX of DB and file, so losing either can only under-count a partial
# month — the cap can never be tricked into overspending a running ledger.
_AI_LEDGER_FILE = os.environ.get("AI_LEDGER_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(DB_PATH)), "ai_ledger.json")

def _ai_ledger_file_read():
    try:
        with open(_AI_LEDGER_FILE) as f:
            d = json.load(f)
        return float(d.get(datetime.now(IST).strftime("%Y-%m")) or 0)
    except Exception:
        return 0.0

def _ai_ledger_file_write(spent):
    try:
        month = datetime.now(IST).strftime("%Y-%m")
        with open(_AI_LEDGER_FILE, "w") as f:
            json.dump({month: round(spent, 4)}, f)
    except Exception:
        pass

def _ai_spent_inr():
    try:
        db_val = float(get_engine_state(_ai_month_key(), default="0") or 0)
    except Exception:
        db_val = 0.0
    return max(db_val, _ai_ledger_file_read())

def _ai_cap_inr():
    try:
        return float(os.environ.get("AI_MONTHLY_CAP_INR", "1000") or 0)
    except Exception:
        return 1000.0

def _ai_record_cost(model, usage):
    """Convert one call's token usage to INR and add to the month ledger.
    Never raises."""
    try:
        price = _AI_PRICING_USD_PER_M["sonnet"]
        for k, p in _AI_PRICING_USD_PER_M.items():
            if k in (model or ""):
                price = p
                break
        usd = ((usage.get("input_tokens") or 0) * price[0]
               + (usage.get("output_tokens") or 0) * price[1]
               + (usage.get("cache_creation_input_tokens") or 0) * price[2]
               + (usage.get("cache_read_input_tokens") or 0) * price[3]) / 1e6
        inr = usd * float(os.environ.get("AI_USD_INR", "88") or 88)
        spent = _ai_spent_inr() + inr
        set_engine_state(_ai_month_key(), f"{spent:.4f}")
        _ai_ledger_file_write(spent)
        cap = _ai_cap_inr()
        month = datetime.now(IST).strftime("%Y-%m")
        if cap > 0 and spent >= 0.8 * cap and _AI_CAP_STATE["warned80"] != month:
            _AI_CAP_STATE["warned80"] = month
            SlackAlert.send(f"⚠️ *AI budget 80% used* — ₹{spent:,.0f} of the "
                            f"₹{cap:,.0f}/month cap. AI layers switch off at the cap; "
                            f"signals keep flowing without the AI check.")
        return spent
    except Exception as e:
        log.warning(f"  AI cost ledger failed: {e}")
        return None

def _ai_budget_blocked():
    """True when this month's spend has hit the cap. Logs (rate-limited)
    and Slacks (once per day) when blocking."""
    cap = _ai_cap_inr()
    if cap <= 0:
        return False
    spent = _ai_spent_inr()
    if spent < cap:
        return False
    now_ts = time.time()
    if now_ts - _AI_CAP_STATE["capped_log"] > 3600:
        _AI_CAP_STATE["capped_log"] = now_ts
        log.warning(f"🔒 AI monthly cap hit: ₹{spent:,.0f} >= ₹{cap:,.0f} — "
                    f"AI calls disabled until next month (engine keeps trading)")
    today = datetime.now(IST).strftime("%Y-%m-%d")
    if _AI_CAP_STATE["capped_slack"] != today:
        _AI_CAP_STATE["capped_slack"] = today
        try:
            SlackAlert.send(f"🔒 *AI monthly budget cap hit* — ₹{spent:,.0f} of "
                            f"₹{cap:,.0f}. All AI layers paused until the month rolls "
                            f"over; the engine keeps generating signals without them. "
                            f"Raise with AI_MONTHLY_CAP_INR if intended.")
        except Exception:
            pass
    return True


def _anthropic_call(prompt, model=None, max_tokens=800, temperature=0.2, timeout=20,
                    system=None, layer=None):
    """Low-level wrapper around Anthropic messages API. Returns parsed JSON dict
    (from Claude's response content) or None if anything failed. Callers decide
    how to handle None — the safe default for validation is SKIP.

    `system`  — optional list of {"type":"text","text":...} blocks. The LAST block
                gets cache_control: ephemeral when CONFIG['anthropic_cache_enabled']
                is True. Sonnet 4.5 minimum for caching is 1024 tokens; below that
                Anthropic silently runs the request uncached (no error).
    `layer`   — short string label ("regime", "validation", "inflight", "eod",
                "swing_exit") used for per-layer telemetry attribution.
    """
    api_key = CONFIG.get("anthropic_api_key", "")
    if not api_key:
        return None
    if _ai_budget_blocked():
        return None
    body = {
        "model": model or CONFIG.get("anthropic_model", "claude-sonnet-4-5"),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        # Normalise to the multi-block format and tag the last block for caching.
        blocks = []
        if isinstance(system, str):
            blocks = [{"type": "text", "text": system}]
        elif isinstance(system, list):
            for b in system:
                if isinstance(b, str):
                    blocks.append({"type": "text", "text": b})
                elif isinstance(b, dict):
                    # Preserve any caller-set cache_control; we'll add to the last block below
                    blocks.append({k: v for k, v in b.items() if k in ("type", "text", "cache_control")})
        if blocks and CONFIG.get("anthropic_cache_enabled", True):
            # Cache breakpoint goes on the last system block — everything before it (incl. tools
            # and earlier system blocks) is cached as one prefix.
            last = blocks[-1]
            if "cache_control" not in last:
                last["cache_control"] = {"type": "ephemeral"}
        if blocks:
            body["system"] = blocks
    text = ""
    try:
        resp = requests.post(
            _ANTHROPIC_URL,
            headers={"Content-Type": "application/json",
                     "x-api-key": api_key,
                     "anthropic-version": "2023-06-01"},
            json=body,
            timeout=timeout,
        )
        if resp.status_code != 200:
            _ANTHROPIC_USAGE["errors"] += 1
            log.warning(f"  Claude API error {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        # Aggregate usage telemetry
        _ANTHROPIC_USAGE["calls"] += 1
        u = data.get("usage") or {}
        _ai_record_cost(body["model"], u)   # monthly ₹ ledger + cap warnings
        _ANTHROPIC_USAGE["input_tokens"]          += int(u.get("input_tokens") or 0)
        _ANTHROPIC_USAGE["output_tokens"]         += int(u.get("output_tokens") or 0)
        _ANTHROPIC_USAGE["cache_read_tokens"]     += int(u.get("cache_read_input_tokens") or 0)
        _ANTHROPIC_USAGE["cache_creation_tokens"] += int(u.get("cache_creation_input_tokens") or 0)
        if layer:
            slot = _ANTHROPIC_USAGE["by_layer"].setdefault(layer, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_creation_tokens": 0,
            })
            slot["calls"]                 += 1
            slot["input_tokens"]          += int(u.get("input_tokens") or 0)
            slot["output_tokens"]         += int(u.get("output_tokens") or 0)
            slot["cache_read_tokens"]     += int(u.get("cache_read_input_tokens") or 0)
            slot["cache_creation_tokens"] += int(u.get("cache_creation_input_tokens") or 0)
        text = (data.get("content", [{}])[0] or {}).get("text", "").strip()
        import re as _re
        text = _re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=_re.M).strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning(f"  Claude JSON parse failed: {e} // raw={text[:200]}")
        return None
    except Exception as e:
        _ANTHROPIC_USAGE["errors"] += 1
        log.warning(f"  Claude call failed: {e}")
        return None


# ─── Layer A: Pre-market Regime Brief (run once at ~08:45 IST) ────────
class RegimeBrief:
    """One call per trading morning. Asks Claude to characterise the day's
    expected regime and return overrides for the scanner (confidence floor,
    R:R floor, instruments to avoid). Persisted to the `regime` table."""

    _SYSTEM_PROMPT = """You are a senior Indian intraday options strategist preparing a ₹20,000 desk for the trading session. Pick a regime, set a directional bias, and define guardrails (confidence floor, R:R floor, instruments to avoid). Be concrete and conservative — these settings gate ALL trades for the day, so getting it wrong costs money.

Regime taxonomy:
- TRENDING_UP / TRENDING_DOWN: clear directional move underway (overnight gap + global cues + macro alignment)
- RANGING: index oscillating in a tight band — favor mean-reversion, smaller targets, lower confidence
- VOLATILE: large moves both directions — widen stops, smaller size, prefer NIFTY (cheapest premium)
- EVENT_RISK: scheduled high-impact event (FOMC, RBI, budget) — avoid the index in the blackout window

Bias & overrides:
- bias = LONG / SHORT / NEUTRAL (suggested directional skew, NOT mandatory)
- confidence_floor — minimum scanner confidence to alert. Higher in volatile/event days
- min_rr — risk:reward floor on the index trigger. 1.5 baseline; 2.0+ on event days
- avoid_instruments — list any of NIFTY/BANKNIFTY/FINNIFTY to skip entirely today

Respond in EXACTLY this JSON (no markdown, no prose):
{"regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "VOLATILE" | "EVENT_RISK",
  "bias": "LONG" | "SHORT" | "NEUTRAL",
  "confidence_floor": integer 55..80,
  "min_rr": number 1.2..2.5,
  "avoid_instruments": [list of NIFTY/BANKNIFTY/FINNIFTY] or [],
  "notes": "1-2 sentence rationale traders can act on"}"""

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

        prompt = f"""Date: {today}   Current IST: {datetime.now(IST).strftime('%H:%M')}
Recent closed trades (most recent first): {recent_txt}
Known events today: {event_txt}

Set today's regime and scanner overrides."""

        raw = _anthropic_call(prompt, max_tokens=300, timeout=20,
                              system=RegimeBrief._SYSTEM_PROMPT, layer="regime")
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

    # Static rules block — cached via cache_control:ephemeral. Must be ≥1024 tokens for
    # Sonnet 4.5 caching to engage. This is the MOST FREQUENTLY-CALLED layer, so the
    # caching savings are highest here (~80% of input tokens on hits).
    _SYSTEM_PROMPT = """You are a ruthlessly disciplined Indian intraday options trader on a ₹20,000 account.
Protect capital first, profit second. Only TAKE with clear edge.
You have full option chain data — use Greeks, OI, PCR, IV skew, volume as PRIMARY signals.
The index signal is a trigger; the option chain confirms or denies it.

HOW TO READ THE CHAIN DATA
- PCR < 0.7 = bullish (more CEs, less put protection needed)
- PCR > 1.0 = bearish (heavy put buying = hedging against fall)
- IV skew > +2 = market pricing fear (PE premium = bears are nervous)
- Max pain is where market makers want price to go — trade WITH it near expiry
- High volume on ATM CE = call buying = bullish momentum
- High volume on ATM PE = put buying = bearish momentum
- Only SKIP if direction conflicts with PCR AND IV skew AND volume — ALL THREE aligned against you

OI VELOCITY SIGNALS (real-time shift detection — highest quality signal):
- OI BUILDING at ATM CE = call writers entering = fresh RESISTANCE forming = bearish for LONG trades
- OI BUILDING at ATM PE = put writers entering = fresh SUPPORT forming = bearish for SHORT trades
- OI UNWINDING at ATM CE = resistance dissolving = BULLISH for LONG trades (ceiling lifting)
- OI UNWINDING at ATM PE = support dissolving = BEARISH for SHORT trades (floor falling)
- CE_ROLL_BULLISH = institutions rolling CE positions to ATM = strong bullish conviction — favor LONG
- PE_ROLL_BEARISH = institutions rolling PE positions to ATM = strong bearish conviction — favor SHORT
- Vol_delta (volume acceleration) is ALWAYS real-time even when oi=0. Use vol_Δ as primary signal.
- PE_BUILD + vol acceleration at ATM = most reliable SHORT signal
- CE_BUILD + vol acceleration at ATM = most reliable LONG signal (call writers = insurance sellers)

OPTION RULES
- IMPORTANT: Angel One's FULL API frequently returns oi=0 and volume=0 even for liquid options.
  Do NOT skip solely because OI=0 or volume=0 — the spread and LTP are more reliable liquidity signals.
- PREFER delta 0.35-0.55 for directional trades (0.25-0.65 is acceptable)
- SKIP if IV > 60% (IV crush risk on event days only)
- SKIP if spread > 10% of LTP (genuinely illiquid — slippage is too high)
- If price is affordable and spread is tight, it's likely tradeable regardless of OI value

SIGNAL RULES
- SKIP if RSI>80 for LONG or RSI<20 for SHORT (extreme, not just elevated)
- Default to TAKE unless there is a strong specific reason to SKIP
- WAIT (not SKIP) if setup is good but one minor concern exists — allow retry
- SKIP after 14:40 IST (insufficient time for trade to run)
- SKIP if SuperTrend strongly disagrees AND RSI also contradicts
- Scale POSITION_PCT down (50%) when setup not perfectly clean — do NOT SKIP unless truly broken
- SL_TIGHTENING = "trailing_atr" on trending, "breakeven_at_half_t1" on ranging
- BANKNIFTY monthly slips harder — use 75% position_pct for BANKNIFTY if in doubt
- BIAS TOWARD TAKING: a missed trade costs 0, but a blocked good trade also costs 0. The engine's
  confidence gates already protect capital. Your job is to confirm clear edge, not find reasons to SKIP.

Respond in EXACTLY this JSON (no markdown, no prose):
{"verdict": "TAKE" | "SKIP" | "WAIT",
  "position_pct": 25 | 50 | 75 | 100,
  "sl_tightening": "none" | "breakeven_at_half_t1" | "trailing_atr",
  "confidence_adj": integer -20..10,
  "reasoning": "one short line mentioning key chain signal that confirmed/denied",
  "risk_note": "one specific risk including PCR/IV/OI concern if any"}"""

    @staticmethod
    def analyze(instrument, signal, option, regime=None, chain_analytics=None):
        api_key = CONFIG.get("anthropic_api_key", "")
        if not api_key:
            # No Claude configured → pass-through so the engine still works
            return {"verdict": "TAKE", "position_pct": 100, "sl_tightening": "none",
                    "reasoning": "AI disabled", "risk_note": "n/a", "confidence_adj": 0}

        ind = signal.get("indicators", {})
        opt_info = ""
        chain_info = ""
        if option:
            opt_info = (f"SELECTED OPTION: {option.get('symbol','')} | Strike {option.get('strike')} {option.get('type')}\n"
                        f"  LTP ₹{option.get('ltp',0)} (bid {option.get('bid')} / ask {option.get('ask')}, spread {option.get('spread')})\n"
                        f"  Greeks: δ={option.get('delta')} ({option.get('delta_source','?')}) "
                        f"γ={option.get('gamma')} θ={option.get('theta')} vega={option.get('vega')} IV={option.get('iv')}%\n"
                        f"  OI={option.get('oi',0):,}  Volume={option.get('volume',0):,}\n"
                        f"  Entry ₹{option.get('entry')} SL ₹{option.get('sl')} T1 ₹{option.get('target1')} T2 ₹{option.get('target2')}\n"
                        f"  {option.get('lots')}×{option.get('lot_size')} lots = ₹{option.get('capital')} capital | "
                        f"MaxLoss ₹{option.get('max_loss')} | T1 profit ₹{option.get('t1_profit')} | T2 profit ₹{option.get('t2_profit')}")
            # Chain snapshot: top 5 candidates for AI to compare
            snap = option.get("chain_snapshot", [])
            if snap:
                chain_info = "\nTOP 5 CHAIN CANDIDATES (ranked by engine score):\n"
                for i, c in enumerate(snap):
                    chain_info += (f"  #{i+1} Strike {c.get('strike')} {c.get('type')}: "
                                   f"₹{c.get('ltp')} δ={c.get('delta')} γ={c.get('gamma')} "
                                   f"θ={c.get('theta')} IV={c.get('iv')}% "
                                   f"OI={c.get('oi',0):,} Vol={c.get('volume',0):,} "
                                   f"RR={c.get('rr')} score={c.get('score')}\n")

        # Build chain-level analytics block for the AI prompt
        chain_anal_txt = ""
        ca = chain_analytics or {}
        if ca:
            atm_ce = ca.get("atm_ce") or {}
            atm_pe = ca.get("atm_pe") or {}
            top_ce = ca.get("top_vol_ce") or []
            top_pe = ca.get("top_vol_pe") or []
            # OI velocity / shift data
            atm_ce_d = ca.get("atm_ce_oi_delta") or {}
            atm_pe_d = ca.get("atm_pe_oi_delta") or {}
            bld_ce = ca.get("building_ce") or []
            bld_pe = ca.get("building_pe") or []
            unw_ce = ca.get("unwinding_ce") or []
            unw_pe = ca.get("unwinding_pe") or []
            oi_shift = ca.get("oi_shift_signal", "NONE")
            has_delta = ca.get("oi_delta_available", False)
            def _vel(d):
                v = d.get("velocity","?")
                return f"{v} Δ{d.get('oi_delta',0):+,} vol_Δ{d.get('vol_delta',0):+,} ({d.get('dt_min','?')}min)"
            vel_txt = ""
            if has_delta:
                vel_txt = f"""
OI & VOLUME VELOCITY (last {atm_ce_d.get('dt_min','?')} min — real-time shift detection):
  OI Shift Signal = {oi_shift}
  ATM CE OI: {_vel(atm_ce_d)} | ATM PE OI: {_vel(atm_pe_d)}
  Building CE strikes (fresh resistance): {', '.join(str(b['strike'])+' +'+str(b['oi_delta']) for b in bld_ce) or 'none'}
  Building PE strikes (fresh support):    {', '.join(str(b['strike'])+' +'+str(b['oi_delta']) for b in bld_pe) or 'none'}
  Unwinding CE strikes (resistance fading): {', '.join(str(u['strike'])+' '+str(u['oi_delta']) for u in unw_ce) or 'none'}
  Unwinding PE strikes (support fading):    {', '.join(str(u['strike'])+' '+str(u['oi_delta']) for u in unw_pe) or 'none'}
  NOTE: OI from Angel One API can be 0 (API limitation). When oi_delta=0, vol_delta is the primary signal.
        Volume delta IS always real-time. Use it as the primary momentum indicator."""
            chain_anal_txt = f"""
OPTION CHAIN MARKET SNAPSHOT (scan from live chain — use this as primary signal):
  ATM = {ca.get('atm')} | PCR (OI) = {ca.get('pcr')} | PCR (Vol) = {ca.get('pcr_vol')}
  PCR interpretation: <0.7 bullish, 0.7-1.0 neutral, >1.0 bearish
  IV Skew (PE-CE) = {ca.get('iv_skew')} | positive = put premium (bearish fear), negative = call premium (bullish)
  Max Pain Strike = {ca.get('max_pain')} (highest combined OI — strong gravitational level)
  CE OI total: {ca.get('total_ce_oi',0):,} | Vol: {ca.get('total_ce_vol',0):,}
  PE OI total: {ca.get('total_pe_oi',0):,} | Vol: {ca.get('total_pe_vol',0):,}
  Max OI CE: {ca.get('max_oi_ce_strike')} ({ca.get('max_oi_ce_oi',0):,} OI) — resistance ceiling
  Max OI PE: {ca.get('max_oi_pe_strike')} ({ca.get('max_oi_pe_oi',0):,} OI) — support floor
  ATM CE ({atm_ce.get('strike')}): ₹{atm_ce.get('ltp')} δ={atm_ce.get('delta')} γ={atm_ce.get('gamma')} IV={atm_ce.get('iv')}% OI={atm_ce.get('oi',0):,} Vol={atm_ce.get('volume',0):,}
  ATM PE ({atm_pe.get('strike')}): ₹{atm_pe.get('ltp')} δ={atm_pe.get('delta')} γ={atm_pe.get('gamma')} IV={atm_pe.get('iv')}% OI={atm_pe.get('oi',0):,} Vol={atm_pe.get('volume',0):,}
  Volume leaders CE: {' | '.join(f"{o.get('strike')} ₹{o.get('ltp')} vol={o.get('volume',0):,}" for o in top_ce)}
  Volume leaders PE: {' | '.join(f"{o.get('strike')} ₹{o.get('ltp')} vol={o.get('volume',0):,}" for o in top_pe)}{vel_txt}"""

        regime_txt = ""
        if regime:
            regime_txt = (f"\nToday's regime: {regime.get('regime')} / bias {regime.get('bias')} "
                          f"/ floor {regime.get('confidence_floor')}%  min RR {regime.get('min_rr')}  "
                          f"avoid {regime.get('avoid_instruments')}  notes: {regime.get('notes','')}")

        # ── Static system prefix (cached) ────────────────────────────────
        # Everything that does NOT change per-signal goes here. Cached as one
        # prefix → 0.1x cost on cache hit. Sonnet 4.5 needs ≥1024 tokens cached;
        # this block + JSON schema is comfortably above that.
        system_prefix = SignalValidation._SYSTEM_PROMPT
        # ── Per-signal user prompt (NOT cached) ──────────────────────────
        prompt = f"""INDEX SIGNAL (trigger)
Instrument: {instrument}   Direction: {signal['direction']}   Engine confidence: {signal['confidence']}%
Index entry {signal['entry']}  SL {signal['sl']}  T1 {signal['target1']}  T2 {signal['target2']}  R:R {signal.get('risk_reward',0)}
{chain_anal_txt}

{opt_info}
{chain_info}

PRICE INDICATORS
RSI {ind.get('rsi','?')}  MACD hist {ind.get('macd','?')}  SuperTrend {ind.get('supertrend','?')}
EMA9/21/50 {ind.get('ema9','?')}/{ind.get('ema21','?')}/{ind.get('ema50','?')}  VWAP {ind.get('vwap','?')}
ATR {ind.get('atr','?')}  Stoch {ind.get('stoch','?')}  ADX {ind.get('adx','?')}  VolRatio {ind.get('vol_ratio','?')}x
Reasons: {', '.join(signal.get('reasons',[])[:6])}
Time: {datetime.now(IST).strftime('%H:%M')} IST{regime_txt}

Respond NOW with the JSON verdict."""

        result = _anthropic_call(prompt, max_tokens=300, timeout=15,
                                 system=system_prefix, layer="validation")
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
    _SYSTEM_PROMPT = """You are a quantitative reviewer for an Indian intraday options desk. Each evening you analyse the day's closed trades and propose SMALL, SPECIFIC tweaks for the scanner to apply tomorrow. Do not propose redesigns — only weight nudges (-5..+5 per indicator) and time-window blocks.

INDICATOR WEIGHTS available to nudge (additive deltas, applied to base contribution):
- rsi: tweak how much RSI extreme contributes to score
- macd: tweak MACD cross/expansion contribution
- supertrend: tweak SuperTrend flip contribution
- vwap: tweak VWAP-cross contribution
- ema: tweak EMA9/21 cross contribution
- volume: tweak volume-surge contribution

EXTRA_FILTERS examples:
- "skip NIFTY signals when ATR < 80"
- "require PCR confirmation on BANKNIFTY SHORTs"
- "block any signal with RSI>78"

Respond in EXACTLY this JSON (no markdown):
{"indicator_weight_adjustments": {"rsi": -5..+5, "macd": -5..+5, "supertrend": -5..+5, "vwap": -5..+5, "ema": -5..+5, "volume": -5..+5},
  "time_windows_to_avoid": ["HH:MM-HH:MM", ...],
  "extra_filters": ["short plain-English filters the scanner should apply tomorrow"],
  "summary": "one line recap of today"}"""

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

        prompt = f"""Date: {today}.
Trades closed today (oldest first):
{chr(10).join(summary)}

Propose tweaks for tomorrow's scanner."""

        raw = _anthropic_call(prompt, max_tokens=500, timeout=25,
                              system=LearningLoop._SYSTEM_PROMPT, layer="eod")
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
    _SYSTEM_PROMPT = """You are a disciplined intraday options position manager. For each OPEN position you receive, decide ONE action and respond in JSON only.

DECISION RULES:
- HOLD: trade is in normal range, thesis intact, time still on side
- TRAIL_SL: position in profit > 1% beyond SL, trail SL toward breakeven or higher (provide new_sl)
- PARTIAL_EXIT_50: hit T1 or close to it, lock 50% profit (provide reasoning)
- CLOSE: thesis broken (price approaching SL with momentum, late session, or move > T2)

CONSTRAINTS:
- new_sl can ONLY tighten (move toward entry from a stop perspective). Never loosen.
- Be biased toward HOLD on normal moves — trades need room to breathe.
- After 14:50 IST, lean toward CLOSE on losing positions to avoid overnight risk.

Respond in EXACTLY this JSON:
{"action": "HOLD" | "PARTIAL_EXIT_50" | "TRAIL_SL" | "CLOSE",
  "new_sl": number or null,
  "reasoning": "one short line"}"""

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

            prompt = f"""Trade: {s['instrument']} {s['direction']}  {s.get('option_symbol','')}
Entry ₹{entry}  Current ₹{ltp}  Move {move_pct}%  Held {held_min}min
SL ₹{s.get('option_sl')}  T1 ₹{s.get('option_target1')}  T2 ₹{s.get('option_target2')}
SL tightening rule in effect: {s.get('sl_tightening','none')}
Time: {datetime.now(IST).strftime('%H:%M')} IST"""

            raw = _anthropic_call(
                prompt,
                model=CONFIG.get("anthropic_model_inflight") or CONFIG.get("anthropic_model"),
                max_tokens=200, timeout=12,
                system=TradeManager._SYSTEM_PROMPT, layer="inflight",
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
                lots_n = int(s.get("option_lots") or 1)
                qty = lots_n * int(s.get("option_lot_size") or 0 or 1)
                update_result(s["id"], s.get("index_price") or 0,
                              "WIN" if pnl_per > 0 else "LOSS",
                              round(pnl_per, 2), round(pnl_per * qty, 0),
                              option_exit=ltp, option_entry=entry,
                              qty=qty, lots=lots_n)
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
    def analyze(instrument, signal, option, regime=None, chain_analytics=None):
        return SignalValidation.analyze(instrument, signal, option, regime=regime,
                                        chain_analytics=chain_analytics)


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
                # STRICT GREEKS (step 15): if env var is set, REJECT this candidate
                # entirely rather than substituting an estimated delta. Means every
                # signal that fires has real exchange-priced delta — eliminates
                # all estimation, at the cost of fewer signals when Angel's
                # greek endpoint is down.
                if CONFIG.get("strict_greeks", False):
                    log.warning(f"  STRICT_GREEKS: rejecting {o.get('symbol','?')} "
                                f"strike {strike} {ot} — no live delta from Angel")
                    continue
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

            # 6. OI quality (5 pts) — high OI = liquid and active
            oi = int(o.get("oi", 0) or 0)
            if oi >= 1_000_000: score += 5
            elif oi >= 500_000:  score += 3
            elif oi >= 100_000:  score += 1

            # 7. Volume activity (3 pts) — relative volume vs peers signals momentum
            vol = int(o.get("volume", 0) or 0)
            if vol >= 50_000: score += 3
            elif vol >= 10_000: score += 2
            elif vol >= 1_000:  score += 1

            # 8. Gamma quality (2 pts) — higher gamma means faster option move near ATM
            gm = float(o.get("gamma", 0) or 0)
            if gm >= 0.0005: score += 2
            elif gm >= 0.0002: score += 1

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

        # ── STEP 14: Premium-percentage exits (no more delta-scaled estimates) ──
        # When OPT_EXIT_MODE=premium_pct (default), SL/T1/T2 are exact percentages of
        # the live entry premium. Dashboard SL=₹40 means "option exits at exactly ₹40
        # when the live mid hits ₹40" — no model, no extrapolation. The legacy
        # delta-scaled mode is still available via env if anyone needs it.
        # Also compute the legacy modelled values for diagnostic visibility.
        exit_mode = str(CONFIG.get("opt_exit_mode", "premium_pct")).lower()
        sl_pct = float(CONFIG.get("opt_sl_pct", 0.35))
        t1_pct = float(CONFIG.get("opt_t1_pct", 0.50))
        t2_pct = float(CONFIG.get("opt_t2_pct", 1.00))

        sl_modelled = round(max(e - idx_to_sl * d, e * 0.65), 2)
        t1_modelled = round(e + idx_to_t1 * d, 2)
        t2_modelled = round(e + idx_to_t2 * d, 2)
        sl_premium  = round(e * (1.0 - sl_pct), 2)
        t1_premium  = round(e * (1.0 + t1_pct), 2)
        t2_premium  = round(e * (1.0 + t2_pct), 2)

        if exit_mode == "premium_pct":
            sl, t1, t2 = sl_premium, t1_premium, t2_premium
        else:
            sl, t1, t2 = sl_modelled, t1_modelled, t2_modelled

        cost_1lot = e * lot
        if cost_1lot <= max_capital:
            lots = max(1, min(int(max_capital / cost_1lot), 3))
        else:
            lots = 1
        # ── Per-trade lot cap (env-tunable) ───────────────────────────
        # During live verification of a new strategy we cap to 1 lot to
        # limit per-trade max loss to ~₹2-3k. Once 3-5 days of live trades
        # match backtest, remove the env var to let the picker scale back
        # up to its computed sizing.
        _max_lots_env = os.environ.get("MAX_LOTS_PER_TRADE", "").strip()
        if _max_lots_env.isdigit():
            lots = min(lots, max(1, int(_max_lots_env)))
        qty = lots * lot
        capital = round(e * qty)

        # Build chain snapshot for AI (top 5 candidates with rich data)
        chain_snapshot = []
        for c in scored[:5]:
            chain_snapshot.append({
                "strike": c["strike"], "type": c["type"],
                "ltp": c["ltp"], "bid": c.get("bid"), "ask": c.get("ask"),
                "delta": c.get("delta"), "gamma": c.get("gamma"), "theta": c.get("theta"),
                "vega": c.get("vega"), "iv": c.get("iv"), "oi": c.get("oi"), "volume": c.get("volume"),
                "score": c["score"], "rr": c.get("rr"), "otm_gaps": c.get("otm_gaps"),
                "affordable": c.get("affordable"), "lots_possible": c.get("lots_possible"),
            })

        return {
            "action": f"BUY {ot}", "symbol": b["symbol"], "strike": b["strike"], "type": ot,
            "expiry": b.get("expiry", ""), "token": b.get("token", ""),
            "ltp": round(e, 2), "entry": round(e, 2),
            "bid": b.get("bid"), "ask": b.get("ask"), "spread": b.get("spread"),
            "sl": sl, "target1": t1, "target2": t2,
            "exit_mode": exit_mode,
            "sl_premium": sl_premium, "t1_premium": t1_premium, "t2_premium": t2_premium,
            "sl_modelled": sl_modelled, "t1_modelled": t1_modelled, "t2_modelled": t2_modelled,
            "delta": d, "delta_source": b.get("delta_source", "fallback"),
            "iv": b.get("iv"), "theta": b.get("theta"),
            "gamma": b.get("gamma"), "vega": b.get("vega"),
            "oi": b.get("oi"), "volume": b.get("volume"),
            "lot_size": lot, "lots": lots, "qty": qty,
            "capital": capital, "max_loss": round((e - sl) * qty),
            "t1_profit": round((t1 - e) * qty), "t2_profit": round((t2 - e) * qty),
            "rr": b["rr"], "otm_gaps": b["otm_gaps"], "score": b["score"],
            "alternatives": len(scored),
            "position_pct": int(pct * 100),
            "chain_snapshot": chain_snapshot,
            "source": "LIVE"
        }

    @staticmethod
    def chain_analytics(chain, atm, oi_delta=None):
        """Compute option-chain-level analytics from the raw chain list.

        Returns a dict with PCR, OI concentration, volume leaders, IV skew,
        max pain strike, ATM CE/PE snapshot, AND OI/volume velocity (shift detection).

        oi_delta — dict from Engine._compute_oi_delta(): {(strike, type): {oi_delta, vol_delta,
                   velocity ("BUILDING"/"UNWINDING"/"STABLE"), vol_trend, dt_min}}
        """
        if not chain: return {}
        try:
            ce_list = [o for o in chain if o.get("type") == "CE" and o.get("ltp", 0) > 0]
            pe_list = [o for o in chain if o.get("type") == "PE" and o.get("ltp", 0) > 0]

            # ── PCR (Put-Call Ratio) by OI ──
            total_ce_oi = sum(o.get("oi", 0) or 0 for o in ce_list)
            total_pe_oi = sum(o.get("oi", 0) or 0 for o in pe_list)
            pcr = round(total_pe_oi / max(total_ce_oi, 1), 3)

            # ── PCR by volume (more reliable than OI for intraday) ──
            total_ce_vol = sum(o.get("volume", 0) or 0 for o in ce_list)
            total_pe_vol = sum(o.get("volume", 0) or 0 for o in pe_list)
            pcr_vol = round(total_pe_vol / max(total_ce_vol, 1), 3)

            # ── Max OI strike (support/resistance where options writers are most exposed) ──
            max_oi_ce = max(ce_list, key=lambda x: x.get("oi", 0) or 0, default=None)
            max_oi_pe = max(pe_list, key=lambda x: x.get("oi", 0) or 0, default=None)

            # ── Volume leaders (active strikes being traded right now) ──
            top_vol_ce = sorted(ce_list, key=lambda x: x.get("volume", 0) or 0, reverse=True)[:3]
            top_vol_pe = sorted(pe_list, key=lambda x: x.get("volume", 0) or 0, reverse=True)[:3]

            # ── ATM CE and PE snapshot ──
            def atm_option(lst):
                if not lst: return None
                return min(lst, key=lambda x: abs(float(x.get("strike", 0)) - atm))

            atm_ce = atm_option(ce_list)
            atm_pe = atm_option(pe_list)

            # ── IV skew: ATM PE IV vs CE IV — positive = market pricing downside ──
            ce_iv = float(atm_ce.get("iv_raw", 0) or atm_ce.get("iv", 0) or 0) if atm_ce else 0
            pe_iv = float(atm_pe.get("iv_raw", 0) or atm_pe.get("iv", 0) or 0) if atm_pe else 0
            iv_skew = round(pe_iv - ce_iv, 2)  # positive = put skew (bearish insurance premium)

            # ── Max pain: strike where total OI (CE+PE) is highest ──
            oi_by_strike = {}
            for o in chain:
                k = float(o.get("strike", 0))
                oi_by_strike[k] = oi_by_strike.get(k, 0) + (o.get("oi", 0) or 0)
            max_pain_strike = max(oi_by_strike, key=oi_by_strike.get) if oi_by_strike else atm

            # ── OI VELOCITY & SHIFT DETECTION (the real intelligence) ──
            # Classify which strikes are seeing fresh OI buildup vs unwinding.
            # OI BUILDING at a CE strike = new call writers = RESISTANCE forming there
            # OI BUILDING at a PE strike = new put writers = SUPPORT forming there
            # OI UNWIND at a strike = positions closing = that level is losing significance
            # OI SHIFT (build at A, unwind at B same type) = roll = directional conviction
            oi_delta = oi_delta or {}
            building_ce = sorted(
                [(k[0], v) for k, v in oi_delta.items() if k[1]=="CE" and v["velocity"]=="BUILDING"],
                key=lambda x: -abs(x[1]["oi_delta"]))[:3]
            building_pe = sorted(
                [(k[0], v) for k, v in oi_delta.items() if k[1]=="PE" and v["velocity"]=="BUILDING"],
                key=lambda x: -abs(x[1]["oi_delta"]))[:3]
            unwinding_ce = sorted(
                [(k[0], v) for k, v in oi_delta.items() if k[1]=="CE" and v["velocity"]=="UNWINDING"],
                key=lambda x: -abs(x[1]["oi_delta"]))[:3]
            unwinding_pe = sorted(
                [(k[0], v) for k, v in oi_delta.items() if k[1]=="PE" and v["velocity"]=="UNWINDING"],
                key=lambda x: -abs(x[1]["oi_delta"]))[:3]

            # ATM-specific delta (most important for signal direction)
            atm_ce_delta = oi_delta.get((int(atm), "CE"), {})
            atm_pe_delta = oi_delta.get((int(atm), "PE"), {})

            # Volume velocity: which strikes had the biggest volume jump since last scan
            # Volume IS truly real-time (cumulative intraday volume updates with every trade)
            vol_accel_ce = sorted(
                [(k[0], v) for k, v in oi_delta.items() if k[1]=="CE" and v.get("vol_delta",0)>500],
                key=lambda x: -x[1]["vol_delta"])[:3]
            vol_accel_pe = sorted(
                [(k[0], v) for k, v in oi_delta.items() if k[1]=="PE" and v.get("vol_delta",0)>500],
                key=lambda x: -x[1]["vol_delta"])[:3]

            # Cross-strike OI shift signal: build at ATM while unwinding at OTM (or vice versa)
            # This suggests a roll — position moving from distant strike to ATM = conviction
            oi_shift_signal = "NONE"
            if building_pe and unwinding_pe:
                oi_shift_signal = "PE_ROLL_BEARISH"  # putting on new puts closer to ATM
            elif building_ce and unwinding_ce:
                oi_shift_signal = "CE_ROLL_BULLISH"  # putting on new calls closer to ATM
            elif building_pe and not building_ce:
                oi_shift_signal = "PE_BUILD"  # fresh put writing = strong support
            elif building_ce and not building_pe:
                oi_shift_signal = "CE_BUILD"  # fresh call writing = strong resistance

            return {
                "pcr": pcr,
                "pcr_vol": pcr_vol,
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "total_ce_vol": total_ce_vol,
                "total_pe_vol": total_pe_vol,
                "max_pain": max_pain_strike,
                "max_oi_ce_strike": max_oi_ce.get("strike") if max_oi_ce else None,
                "max_oi_ce_oi": max_oi_ce.get("oi") if max_oi_ce else 0,
                "max_oi_pe_strike": max_oi_pe.get("strike") if max_oi_pe else None,
                "max_oi_pe_oi": max_oi_pe.get("oi") if max_oi_pe else 0,
                "top_vol_ce": [{"strike": o.get("strike"), "volume": o.get("volume"), "ltp": o.get("ltp")} for o in top_vol_ce],
                "top_vol_pe": [{"strike": o.get("strike"), "volume": o.get("volume"), "ltp": o.get("ltp")} for o in top_vol_pe],
                "atm_ce": {
                    "strike": atm_ce.get("strike"), "ltp": atm_ce.get("ltp"),
                    "delta": atm_ce.get("delta"), "gamma": atm_ce.get("gamma"),
                    "theta": atm_ce.get("theta"), "iv": ce_iv,
                    "oi": atm_ce.get("oi"), "volume": atm_ce.get("volume"),
                } if atm_ce else None,
                "atm_pe": {
                    "strike": atm_pe.get("strike"), "ltp": atm_pe.get("ltp"),
                    "delta": atm_pe.get("delta"), "gamma": atm_pe.get("gamma"),
                    "theta": atm_pe.get("theta"), "iv": pe_iv,
                    "oi": atm_pe.get("oi"), "volume": atm_pe.get("volume"),
                } if atm_pe else None,
                "iv_skew": iv_skew,
                "atm": atm,
                # ── OI & VOLUME VELOCITY DATA ──
                "oi_delta_available": bool(oi_delta),
                "oi_shift_signal": oi_shift_signal,
                "atm_ce_oi_delta": atm_ce_delta,  # {velocity, oi_delta, vol_delta, dt_min}
                "atm_pe_oi_delta": atm_pe_delta,
                "building_ce": [{"strike": s, "oi_delta": v["oi_delta"], "vol_delta": v["vol_delta"], "dt_min": v["dt_min"]} for s, v in building_ce],
                "building_pe": [{"strike": s, "oi_delta": v["oi_delta"], "vol_delta": v["vol_delta"], "dt_min": v["dt_min"]} for s, v in building_pe],
                "unwinding_ce": [{"strike": s, "oi_delta": v["oi_delta"], "vol_delta": v["vol_delta"], "dt_min": v["dt_min"]} for s, v in unwinding_ce],
                "unwinding_pe": [{"strike": s, "oi_delta": v["oi_delta"], "vol_delta": v["vol_delta"], "dt_min": v["dt_min"]} for s, v in unwinding_pe],
                "vol_accel_ce": [{"strike": s, "vol_delta": v["vol_delta"]} for s, v in vol_accel_ce],
                "vol_accel_pe": [{"strike": s, "vol_delta": v["vol_delta"]} for s, v in vol_accel_pe],
            }
        except Exception as e:
            log.warning(f"  chain_analytics error: {e}")
            return {}


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
        # Per-signal peak / trough premium + time. We persist these to the DB
        # on close so the daily summary can flag "near-miss" exits (e.g.
        # premium peaked ₹140 — 80% of the way to T1 — before reversing
        # and hitting SL).
        self._peak     = {}   # {sig_id: (peak_premium, "HH:MM")}
        self._trough   = {}   # {sig_id: (trough_premium, "HH:MM")}
        # Most recent premium per open signal — exposed to the dashboard for
        # the live "multiple compact cards" feed. Updated every check() tick.
        self._last_seen = {}  # {sig_id: current_premium}
        # Cross-instrument loss cooldown: when a SHORT just hit SL, the
        # immediate bearish read was wrong, so block new SHORTs across
        # ALL instruments for LOSS_COOLDOWN_MIN minutes. Same for LONG.
        self._loss_cooldown = {}  # { "LONG" | "SHORT": datetime_until }
        self.LOSS_COOLDOWN_MIN = 60
        # Two-tick exit confirmation (2026-06-12 phantom-SL incident): a
        # single bad quote closed a live winner as a -₹11k SL. An exit
        # condition must hold on TWO consecutive 30s checks before we close.
        # { sig_id: result_str_seen_last_tick }
        self._exit_pending = {}
        # Signals we've already loud-alerted as unpriceable (missing/broken
        # option token) — alert once per signal, never auto-close them.
        self._unpriced_alerted = set()

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

    def _time_stop_due(self, s, opt_entry, cur_opt):
        """True when the trade is older than TIME_STOP_MIN minutes and its
        premium has never advanced TIME_STOP_MIN_GAIN_PCT % over entry (peak
        included, so a trade that ran up and pulled back is NOT time-stopped)."""
        try:
            ts_min = float(os.environ.get("TIME_STOP_MIN", "30") or 0)
            if ts_min <= 0:
                return False
            gain_pct = float(os.environ.get("TIME_STOP_MIN_GAIN_PCT", "15")) / 100.0
            ts_raw = str(s.get("timestamp") or "")
            opened = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            age_min = (datetime.now(IST) - opened).total_seconds() / 60.0
            if age_min < ts_min:
                return False
            hurdle = opt_entry * (1.0 + gain_pct)
            peak = self._best_premium.get(s["id"], cur_opt)
            return cur_opt < hurdle and peak < hurdle
        except Exception:
            return False

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
            opt_t2     = float(s.get("option_target2") or 0)
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
                # Live "current premium" for the dashboard feed
                self._last_seen[s["id"]] = cur_opt
                best = self._best_premium.get(s["id"], opt_entry)
                if cur_opt > best:
                    best = cur_opt
                    self._best_premium[s["id"]] = best

                # Peak / trough tracking — independent of any SL strategy.
                # Surfaces "near-miss" exits in the close alert + EOD summary.
                now_hm = datetime.now(IST).strftime("%H:%M")
                pk = self._peak.get(s["id"])
                if pk is None or cur_opt > pk[0]:
                    self._peak[s["id"]] = (cur_opt, now_hm)
                tr = self._trough.get(s["id"])
                if tr is None or cur_opt < tr[0]:
                    self._trough[s["id"]] = (cur_opt, now_hm)

                # Apply SL tightening rules — they only MOVE SL up (tighter), never loosen.
                # Default is NO tightening: the SL shown on the dashboard is the
                # SL the engine uses. Earlier we silently tightened to breakeven
                # at half-T1, but a single noisy quote could trip it and then a
                # normal pullback to entry would record a LOSS — confusing the
                # user who still sees the original SL on the card.
                # Tightening only runs if the AI / config explicitly opts in.
                new_sl = opt_sl
                if (tighten == "breakeven_at_half_t1" and opt_t1 > opt_entry):
                    half_t1 = opt_entry + (opt_t1 - opt_entry) * 0.5
                    # Require a sustained breach: best AND current both at/above
                    # half-T1 before we lock breakeven. Stops a single noisy tick
                    # from silently shifting SL up to entry.
                    if best >= half_t1 and cur_opt >= half_t1:
                        new_sl = max(new_sl, opt_entry)  # lock breakeven
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

            # Exit detection — prefer option-based levels. T2 wins over T1
            # so a fast surge to T2 is recorded as the bigger result.
            # When SL fires, clamp the recorded exit at the SL level. Between
            # 30s scans the premium can gap below SL; if we just used cur_opt
            # we'd record a loss bigger than the SL itself, contradicting what
            # the user saw on the dashboard. Real fills will differ, but the
            # accounting matches the visible SL.
            result = None
            exit_opt = None
            if cur_opt is not None and opt_entry > 0:
                # ── Exit v2 (INTRADAY_EXIT_V2, default on) ───────────────
                # Fixed +50%/+100% targets amputate the right tail that a
                # ~40%-win momentum profile depends on (Zarattini-style
                # systems use trails, not caps). Ladder becomes:
                #   +25% premium  → SL locks to breakeven
                #   +50% premium  → 25%-below-peak trail arms, T1/T2 caps off
                # SL -35% and the 30-min time stop stay as-is.
                _v2 = os.environ.get("INTRADAY_EXIT_V2", "true").lower() == "true"
                if _v2:
                    _best = self._best_premium.get(s["id"], opt_entry)
                    if _best >= opt_entry * 1.25:
                        opt_sl = max(opt_sl, opt_entry)
                    if _best >= opt_entry * 1.5:
                        opt_sl = max(opt_sl, round(_best * 0.75, 2))
                        opt_t1 = 0; opt_t2 = 0   # uncap: the trail is the exit
                if opt_t2 > 0 and cur_opt >= opt_t2:
                    result = "T2"; exit_opt = cur_opt
                elif opt_t1 > 0 and cur_opt >= opt_t1:
                    result = "WIN"; exit_opt = cur_opt
                elif opt_sl > 0 and cur_opt <= opt_sl:
                    result = "LOSS"; exit_opt = max(cur_opt, opt_sl)
                    # A stop above entry is a profitable trail/breakeven
                    # exit, not a stop-out — label it so, and keep it from
                    # tripping the direction loss-cooldown.
                    if exit_opt >= opt_entry:
                        result = "TRAIL"
                elif self._time_stop_due(s, opt_entry, cur_opt):
                    # Time stop: an option buy that hasn't moved ≥15% in its
                    # favor within 30 min is statistically paying theta for
                    # nothing — documented as the cheapest theta defense for
                    # intraday buyers. Env: TIME_STOP_MIN (0 disables),
                    # TIME_STOP_MIN_GAIN_PCT.
                    result = "TIME_STOP"; exit_opt = cur_opt

                # Expiry-day flat: never carry a weekly-expiry option past
                # 14:00 — ATM decay runs ₹20-40+/hour into the close and
                # premiums collapse regardless of direction.
                if result is None:
                    try:
                        from regime import RegimeFilter as _RF
                        _nw = datetime.now(IST)
                        if _RF._is_expiry_day(_nw, s["instrument"]) and \
                                _nw.strftime("%H:%M") >= os.environ.get("EXPIRY_FLAT_AT", "14:00"):
                            result = "EXPIRY_FLAT"; exit_opt = cur_opt
                    except Exception:
                        pass

                # Two-tick confirmation: never close on a single quote. A lone
                # bad tick (2026-06-12: phantom ₹244 print on an option really
                # trading ₹867) must not book a result. Condition has to hold
                # on two consecutive 30s checks; one clean tick resets it.
                if result:
                    if self._exit_pending.get(s["id"]) != result:
                        self._exit_pending[s["id"]] = result
                        log.warning(f"  ⏳ {s['instrument']} #{s['id']}: {result} condition "
                                    f"at opt ₹{cur_opt} — awaiting confirming tick before close")
                        result = None; exit_opt = None
                else:
                    self._exit_pending.pop(s["id"], None)
            elif idx_px:
                # No option price (missing token or fetch failure). We used to
                # close on INDEX levels with delta-faked option P&L — that
                # fabricated the 2026-06-12 phantom -₹10,982 "SL hit" on a
                # trade that was in profit. NEVER fabricate a close: keep the
                # row OPEN, tell the operator loudly (once), let EOD close_all
                # sweep it if pricing never recovers.
                if s["id"] not in self._unpriced_alerted:
                    self._unpriced_alerted.add(s["id"])
                    log.error(f"  ⚠️ {s['instrument']} #{s['id']}: cannot price option "
                              f"(token={'MISSING' if not token else repr(token)}) — "
                              f"auto-close DISABLED for this signal")
                    SlackAlert.send(
                        f"⚠️ *{s['instrument']} {s['direction']}* (signal #{s['id']}): the engine "
                        f"cannot fetch this option's price ({'no token stored' if not token else 'quote fetch failing'}). "
                        f"It will NOT auto-close or book P&L for this trade — manage it manually "
                        f"from the live chart.")

            if result:
                if cur_opt is not None and opt_entry > 0:
                    # Use the clamped exit_opt (which equals cur_opt for WIN/T1/T2
                    # and max(cur_opt, opt_sl) for LOSS). That way the recorded
                    # loss matches the SL the user saw on the dashboard.
                    use_exit = exit_opt if exit_opt is not None else cur_opt
                    pnl_per_share = (use_exit - opt_entry)
                    pnl_rs = round(pnl_per_share * qty, 0)
                    pnl_pts = round(pnl_per_share, 2)  # points here = rupees per share of premium
                    update_result(s["id"], idx_px or 0, result, pnl_pts, pnl_rs,
                                  option_exit=use_exit, option_entry=opt_entry,
                                  qty=qty, lots=opt_lots)
                else:
                    # fallback: index points × lot_size (old behaviour, flagged inaccurate)
                    d = s["direction"]
                    pnl_pts = (idx_px - s["index_entry"]) if d == "LONG" else (s["index_entry"] - idx_px)
                    pnl_rs = round(pnl_pts * lot_size, 0)
                    update_result(s["id"], idx_px, result, round(pnl_pts, 2), pnl_rs)
                # Persist peak/trough so the daily summary + dashboard can read them
                pk = self._peak.get(s["id"])
                tr = self._trough.get(s["id"])
                try:
                    db_exec(
                        "UPDATE signals SET peak_premium=?, peak_time=?, trough_premium=?, trough_time=? WHERE id=?",
                        ((pk[0] if pk else None), (pk[1] if pk else None),
                         (tr[0] if tr else None), (tr[1] if tr else None), s["id"]),
                    )
                except Exception as _e:
                    log.warning(f"  peak persist failed: {_e}")
                # Compute near-miss summary: how far did the trade go in our
                # favor before exiting? Useful when the result is LOSS but
                # the peak was ≥40% of the way to T1.
                near_miss = None
                if pk and opt_entry > 0 and opt_t1 > opt_entry:
                    favor = (pk[0] - opt_entry) / (opt_t1 - opt_entry) * 100  # %
                    if result == "LOSS" and favor >= 40:
                        near_miss = {"peak": pk[0], "peak_time": pk[1],
                                     "pct_to_t1": round(favor, 0),
                                     "hint": f"Could have exited at ₹{int(round(pk[0]))} "
                                             f"({int(round(favor))}% of the way to T1) at {pk[1]}."}
                emoji = "🎯" if result == "T2" else ("✅" if result == "WIN" else "❌")
                # Cross-instrument LOSS cooldown — same direction blocked for an hour
                if result == "LOSS":
                    until = datetime.now(IST) + timedelta(minutes=self.LOSS_COOLDOWN_MIN)
                    self._loss_cooldown[s["direction"]] = until
                    log.info(f"⏸ {s['direction']} cooldown until {until.strftime('%H:%M')} IST "
                             f"(triggered by {s['instrument']} loss)")
                log.info(f"{emoji} {s['instrument']} {s['direction']} → {result} | ₹{pnl_rs} (opt exit ₹{cur_opt})"
                         + (f" · peak ₹{pk[0]} at {pk[1]}" if pk else ""))
                opt_dict = {"symbol": s.get("option_symbol", "")} if s.get("option_symbol") else None
                # Entry time: timestamp column is "YYYY-MM-DD HH:MM:SS" — the
                # old [:5] slice produced "2026-" in alerts. Take the HH:MM part.
                _ts_raw = str(s.get("timestamp") or "")
                entry_hm = _ts_raw.split(" ")[1][:5] if " " in _ts_raw else _ts_raw[:5]
                # Premium journey: entry → peak (when) → trough (when) → exit,
                # so the operator can see what happened BETWEEN entry and exit
                # and judge whether an earlier manual exit was available.
                journey = {
                    "entry": opt_entry if opt_entry > 0 else None,
                    "exit": exit_opt if exit_opt is not None else cur_opt,
                    "peak": pk[0] if pk else None, "peak_time": pk[1] if pk else None,
                    "trough": tr[0] if tr else None, "trough_time": tr[1] if tr else None,
                }
                SlackAlert.send(
                    SlackAlert.format_close(s["instrument"], s["direction"], result, pnl_rs,
                                            option=opt_dict, entry_time=entry_hm,
                                            near_miss=near_miss, journey=journey),
                    blocks=SlackAlert.format_close_blocks(s["instrument"], s["direction"], result, pnl_rs,
                                                          option=opt_dict, entry_time=entry_hm,
                                                          near_miss=near_miss, journey=journey),
                )
                self._best_premium.pop(s["id"], None)
                self._peak.pop(s["id"], None)
                self._trough.pop(s["id"], None)
                self._last_seen.pop(s["id"], None)

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
                              round(cur_opt - opt_entry, 2), pnl_rs,
                              option_exit=cur_opt, option_entry=opt_entry,
                              qty=qty, lots=lots)
            else:
                update_result(s["id"], s["index_price"], "EXPIRED", 0, 0)
        today = datetime.now(IST).strftime("%Y-%m-%d")
        perf = get_perf(date=today)
        if perf["total"] > 0:
            todays_rows = db_exec(
                "SELECT * FROM signals WHERE date=? ORDER BY id ASC",
                (today,), fetch=True) or []
            todays_rows = [dict(r) for r in todays_rows]
            SlackAlert.send(
                SlackAlert.format_daily_summary(perf, rows=todays_rows),
                blocks=SlackAlert.format_daily_summary_blocks(perf, rows=todays_rows),
            )

# ═══════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self):
        from collections import deque as _deque
        self._deque = _deque
        self.client=AngelClient();self.sgen=SignalGen();self.opick=OptPicker()
        self.tracker=PLTracker(self.client);self.latest={};self.alerts=[]
        self.running=False;self._prev={};self._last_signal={}
        # Set by stop() (operator-initiated) and cleared by start(). The
        # scheduler's catch-up auto-ON respects this so a deliberate stop
        # isn't undone 10 minutes later.
        self._manual_stop=False
        # Shadow log dedupe: last time each (instrument, gate) was written,
        # so persistent conditions don't flood the gate_rejections table.
        self._gate_logged={}
        # Chain cache: 30s TTL per instrument. Now stores raw chain for delta recomputes.
        # { name: {ts, ts_str, chain, opt, atm} }
        self._chain_cache={}
        # Last time we Slack-alerted a chain-fetch failure, per instrument
        # (rate-limits the warning to once per 15 min; the failure itself
        # skips the signal every cycle until the chain comes back).
        self._chain_fail_alerted = {}
        # OI + Volume history: rolling 12 snapshots (~6 min at 30s per fetch) per strike.
        # { name: { (strike, type): deque([{oi, volume, ltp, ts}, ...], maxlen=12) } }
        self._oi_history = {}
        # v2 diagnostic buffer — last 20 SignalGenV2 decisions per instrument.
        # Captures EVERY scan tick (trigger or not) so the user can see exactly
        # what each rule scored and why it didn't fire. Read via /api/v2-diag.
        self._v2_diag = {}   # {name: deque(maxlen=20)}
        # Conductor per-scan diagnostic — the verdict (TRIGGER / AFTER_WINDOW /
        # BELOW_MIN_DIMENSIONS / …) of EVERY scan, even when no signal fires.
        # Exposed via /api/status so the dashboard (and the user) can see WHY a
        # scan produced nothing instead of staring at a silent, stale card.
        self._cond_diag = {}        # {name: last decision dict}
        self._cond_diag_last = {}   # {name: (verdict)} for change-only logging
        # Newest candle date the feed has actually served (any instrument).
        # If this isn't today on a trading day, the data is stale (holiday or
        # feed outage) and the dashboard flags it instead of showing old cards.
        self._newest_bar_date = None
        self._mkt_closed_logged = None  # date we last logged "market closed"
        self._regime=None
        self._last_regime_run=None
        self._last_eod_run=None
        self._last_inflight_run=0.0
        # Layer C feedback applied to today's scanner.  Refreshed at boot and at
        # ~09:10 IST each morning; populated from yesterday's `daily_adjustments`
        # row (Claude's EOD review).
        self._weight_adj = {}
        self._blocked_windows = []
        self._adj_loaded_for = None
        # Per-day metrics — surfaced via /api/metrics for the dashboard.
        self.metrics = {
            "date":               datetime.now(IST).strftime("%Y-%m-%d"),
            "scans_total":        0,
            "signals_generated":  0,
            "signals_alerted":    0,
            "ai_skipped":         0,
            "ai_waited":          0,
            "rr_blocked":         0,
            "time_blocked":       0,
            "spread_rejected":    0,   # populated lazily from logs (best-effort)
            "chain_failures":     0,
            "ai_api_failures":    0,
            "kill_switch_hits":   0,
            "regime_blocked":     0,
            "blocked_window_hits":0,
        }
        # Daily kill-switch latch — once tripped today, stops new alerts until 00:00 next day.
        self._killswitch_tripped = False
        # Premium ORB state — { instrument: {"date", "CE": {...}, "PE": {...}} }.
        # Each side tracks its chosen strike + the premium's own opening range.
        self._porb = {}

    def start(self):
        if not self.client.login(): return{"status":"error","message":"Login failed"}

        # Smoke-test the configured strategy on a synthetic dataframe shaped
        # like AngelClient.candles output (column='timestamp'). If the
        # configured strategy can't even survive a single call, fail LOUD
        # before claiming we're running — the previous silent fallback
        # masked a 'timestamp' vs 'ts' KeyError for 2 weeks.
        smoke_ok, smoke_err = self._smoke_test_strategy()
        if not smoke_ok:
            SlackAlert.send(f"⛔ *Engine refused to start*\nStrategy `{CONFIG.get('strategy','v1')}` failed smoke test: `{smoke_err}`\nFix the bug and redeploy — engine is NOT scanning.")
            log.error(f"❌ Refusing to start: strategy smoke test failed: {smoke_err}")
            return {"status":"error","message":f"Strategy {CONFIG.get('strategy','v1')} smoke test failed: {smoke_err}"}

        self.running=True
        self._manual_stop=False
        threading.Thread(target=self._loop,daemon=True).start()
        SlackAlert.send(f"🚀 *Signal Engine Started*\nStrategy: `{CONFIG.get('strategy','v1')}` (smoke-test passed)\nInstruments: {os.environ.get('ENABLED_INSTRUMENTS','NIFTY,BANKNIFTY,FINNIFTY')}")
        return{"status":"ok","message":"Engine started"}

    def _smoke_test_strategy(self):
        """Invoke EVERY configured strategy on a synthetic dataframe that
        mirrors the live AngelClient.candles shape (column='timestamp').

        Covers both the global CONFIG['strategy'] and each per-instrument
        override in INSTRUMENT_STRATEGIES. Each strategy must return a
        dict or None cleanly — any exception fails the smoke test.

        Returns (True, None) if all configured strategies survive their
        call. Returns (False, error_msg) on first failure — engine
        refuses to start because the same bug would fire on every scan.
        """
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            t0 = datetime.now(IST) - timedelta(minutes=60*5 + 1)
            rows = []
            for i in range(60):
                ts = t0 + timedelta(minutes=5*i)
                o = 23000.0 + i * 0.5
                rows.append({"timestamp": ts, "open": o, "high": o + 5, "low": o - 5,
                              "close": o + 1, "volume": 1_000_000 + i * 1000})
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Build the set of (strategy, symbol) pairs to smoke test:
            # 1. The global CONFIG strategy (BANKNIFTY default symbol).
            # 2. Each instrument-specific override.
            global_strategy = CONFIG.get("strategy", "v1").lower()
            pairs = [(global_strategy, "BANKNIFTY")]
            per_env = os.environ.get("INSTRUMENT_STRATEGIES", "").strip()
            seen = {("BANKNIFTY", global_strategy)}
            if per_env:
                for raw in per_env.split(","):
                    if "=" in raw:
                        sym, strat = raw.split("=", 1)
                        sym = sym.strip().upper()
                        strat = strat.strip().lower()
                        if (sym, strat) not in seen:
                            pairs.append((strat, sym))
                            seen.add((sym, strat))

            failures = []
            for strat, sym in pairs:
                # Temporarily force CONFIG.strategy for this probe so the
                # dispatch routes to `strat` even when no INSTRUMENT_STRATEGIES
                # override applies (e.g. probing the global strategy).
                prev_global = CONFIG.get("strategy")
                CONFIG["strategy"] = strat
                try:
                    out = self.sgen.analyze(df, symbol=sym, chain_analytics=None)
                finally:
                    if prev_global is not None:
                        CONFIG["strategy"] = prev_global
                if out is not None and not isinstance(out, dict):
                    failures.append(f"{strat}/{sym}: returned {type(out).__name__}")
                    continue
                ld = SignalGen.last_dispatch or {}
                if ld.get("error"):
                    failures.append(f"{strat}/{sym}: {ld['error']}")
                    continue
                log.info(f"✅ Smoke test '{strat}' on {sym}: ran cleanly (actually_ran={ld.get('actually_ran')})")
            if failures:
                return (False, "; ".join(failures))
            return (True, None)
        except Exception as e:
            return (False, f"{type(e).__name__}: {e}")

    def stop(self):
        self.running=False;self._manual_stop=True;self.tracker.close_all()
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

    def _maybe_load_adjustments(self, now):
        """Layer C feedback wiring: load yesterday's daily_adjustments and apply to
        today's scanner. Refreshed once per day. Also resets per-day metrics +
        kill-switch latch on date change."""
        today = now.strftime("%Y-%m-%d")
        if self._adj_loaded_for == today: return
        # Find the most recent daily_adjustments row strictly before today
        try:
            row = db_exec(
                "SELECT * FROM daily_adjustments WHERE date < ? ORDER BY date DESC LIMIT 1",
                (today,), fetchone=True)
            if row:
                row = dict(row)
                try:
                    self._weight_adj = json.loads(row.get("indicator_weight_adjustments") or "{}") or {}
                except Exception: self._weight_adj = {}
                try:
                    self._blocked_windows = json.loads(row.get("time_windows_to_avoid") or "[]") or []
                except Exception: self._blocked_windows = []
                log.info(f"🧠 Layer C feedback loaded from {row.get('date')}: "
                         f"weights={self._weight_adj}  blocked={self._blocked_windows}")
            else:
                self._weight_adj = {}
                self._blocked_windows = []
        except Exception as e:
            log.warning(f"  Layer C adjustments load failed: {e}")
            self._weight_adj = {}
            self._blocked_windows = []
        # Reset per-day state on date change
        if self.metrics.get("date") != today:
            self.metrics = {k: (today if k == "date" else 0) for k in self.metrics}
            self._killswitch_tripped = False
        self._adj_loaded_for = today

    def _check_killswitch(self):
        """Return True if engine should stop firing new alerts for the rest of the day.

        Trips when EITHER:
          - cumulative net P&L (after brokerage + slippage) for today <= -DAILY_LOSS_LIMIT
          - count of trades today (open + closed) >= MAX_TRADES_PER_DAY
        Once tripped, stays latched until next day's metrics reset.
        """
        if self._killswitch_tripped: return True
        try:
            today = datetime.now(IST).strftime("%Y-%m-%d")
            row = db_exec(
                "SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_rupees),0) as pnl "
                "FROM signals WHERE date=? AND COALESCE(status,'') != 'VOIDED'",
                (today,), fetchone=True)
            row = dict(row) if row else {"cnt": 0, "pnl": 0}
            cnt = int(row.get("cnt") or 0)
            pnl = float(row.get("pnl") or 0)
            # Apply brokerage + slippage estimate to the displayed gross P&L for closed rows
            closed = db_exec(
                "SELECT option_lots, option_lot_size FROM signals "
                "WHERE date=? AND status='CLOSED'", (today,), fetch=True) or []
            adj_pnl = pnl
            for r in closed:
                r = dict(r)
                lots = int(r.get("option_lots") or 1)
                adj_pnl -= float(CONFIG.get("brokerage_per_lot_roundtrip", 100)) * lots
            limit = float(CONFIG.get("daily_loss_limit", 2000) or 0)
            cap   = int(CONFIG.get("max_trades_per_day", 8) or 0)
            tripped = False
            if limit > 0 and adj_pnl <= -limit:
                tripped = True
                log.warning(f"🛑 KILL-SWITCH: daily loss ₹{adj_pnl:.0f} ≤ -₹{limit:.0f}")
                SlackAlert.send(f"🛑 *Kill-switch tripped — daily loss limit*\n"
                                f"Net P&L (after costs): *₹{adj_pnl:.0f}* ≤ -₹{limit:.0f}\n"
                                f"No new alerts for the rest of today.")
            elif cap > 0 and cnt >= cap:
                tripped = True
                log.warning(f"🛑 KILL-SWITCH: trades today {cnt} ≥ {cap}")
                SlackAlert.send(f"🛑 *Kill-switch tripped — daily trade cap*\n"
                                f"Trades today: *{cnt}* ≥ {cap}\n"
                                f"No new alerts for the rest of today.")
            if tripped:
                self._killswitch_tripped = True
                self.metrics["kill_switch_hits"] += 1
            return tripped
        except Exception as e:
            log.warning(f"  killswitch check err: {e}")
            return False

    def _log_gate(self, name, gate, sig=None, detail=None):
        """Shadow-log a gate rejection to the gate_rejections table.

        Deduped in-memory per (instrument, gate, direction) for 10 minutes so
        a persistent condition (e.g. a cooldown re-checked every 30s) writes
        one row, not 20. Never raises.
        """
        try:
            direction = (sig or {}).get("direction")
            key = (name, gate, direction)
            now_ts = time.time()
            if now_ts - self._gate_logged.get(key, 0) < 600:
                return
            self._gate_logged[key] = now_ts
            now = datetime.now(IST)
            db_exec(
                "INSERT INTO gate_rejections (ts, date, instrument, gate, direction, "
                "confidence, index_price, detail) VALUES (?,?,?,?,?,?,?,?)",
                (now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"),
                 name, gate, direction,
                 (sig or {}).get("confidence"), (sig or {}).get("price"),
                 json.dumps(detail) if detail else None))
        except Exception as e:
            log.warning(f"  gate shadow-log failed ({gate}): {e}")

    def _oi_wall_veto(self, name, chain, spot, direction):
        """Veto an entry aimed into a REINFORCED OI wall within
        OI_WALL_VETO_PCT (default 0.20%) of spot.

        SPX analog: call walls hold intraday in 83% of sessions, put walls
        89% — but a wall whose OI is UNWINDING as price approaches often
        breaks and accelerates, so only building walls veto. Returns
        (veto: bool, note: str). Never raises."""
        try:
            if not chain or not spot:
                return False, ""
            side = "CE" if direction == "LONG" else "PE"
            rows = [o for o in chain if o.get("type") == side and (o.get("oi") or 0) > 0
                    and ((o["strike"] > spot) if direction == "LONG" else (o["strike"] < spot))]
            if not rows:
                return False, ""
            wall = max(rows, key=lambda o: o.get("oi") or 0)
            dist_pct = abs(wall["strike"] - spot) / spot * 100
            if dist_pct > float(os.environ.get("OI_WALL_VETO_PCT", "0.20")):
                return False, ""
            hist = (self._oi_history.get(name) or {}).get((wall["strike"], side))
            if hist and len(hist) >= 2 and (hist[-1].get("oi") or 0) <= (hist[0].get("oi") or 0):
                return False, f"{side} wall {wall['strike']:.0f} unwinding — break allowed"
            return True, (f"{side} OI wall {wall['strike']:.0f} "
                          f"({(wall.get('oi') or 0):,} OI) only {dist_pct:.2f}% away and holding/building")
        except Exception:
            return False, ""

    def _premium_orb_tick(self, name, inst, df):
        """Premium-breakout ORB — the only cost-inclusive, positive-expectancy
        NIFTY option-BUYING backtest found in published research (Zerodha
        'In The Money', Jan 2022 - Feb 2026, slippage + brokerage included,
        ~48% WR with a 20% premium SL).

        Mechanics (all env-tunable):
          • At ~09:16 pick the CE and the PE whose premium is nearest
            PREMIUM_ORB_TARGET_PREM (default ₹200) — a premium-normalized
            proxy that lands near ATM.
          • 09:16 → PREMIUM_ORB_OR_END (default 11:15): record each side's
            premium high/low (sampled from the prefetched chain cache).
          • After the window: BUY a side when its premium breaks above its
            own opening-range high. Max 1 CE + 1 PE per day. No entries
            after PREMIUM_ORB_LAST_ENTRY (default 14:30).
          • SL 20% of entry premium / T1 +50% / T2 +100%; the standard
            PLTracker premium exits + time stop manage the trade.
          • Regime gate (PREMIUM_ORB_REGIME_GATE, default on): skip entries
            while the ADX classifier reads 'range' — the study's losing
            year (2025) was exactly low-vol chop.

        Sampling caveat: premium hi/lo is observed on the chain-prefetch
        cadence (~2 min), not tick data — the recorded range is slightly
        narrower than reality, which makes breakout entries CONSERVATIVE
        (never premature).

        Enabled per-instrument via PREMIUM_ORB_INSTRUMENTS (default NIFTY —
        this module exists to fill the NIFTY signal hole; BANKNIFTY already
        has Conductor edge).
        """
        if os.environ.get("PREMIUM_ORB_ENABLED", "true").lower() != "true":
            return
        allowed = {x.strip().upper() for x in
                   os.environ.get("PREMIUM_ORB_INSTRUMENTS", "NIFTY").split(",") if x.strip()}
        if name.upper() not in allowed:
            return

        now = datetime.now(IST)
        hm = now.strftime("%H:%M")
        today = now.strftime("%Y-%m-%d")
        or_end     = os.environ.get("PREMIUM_ORB_OR_END", "11:15")
        last_entry = os.environ.get("PREMIUM_ORB_LAST_ENTRY", "14:30")
        if hm < "09:16" or hm >= last_entry:
            return
        target_prem = float(os.environ.get("PREMIUM_ORB_TARGET_PREM", "200"))
        sl_pct = float(os.environ.get("PREMIUM_ORB_SL_PCT", "0.20"))
        t1_pct = float(os.environ.get("PREMIUM_ORB_T1_PCT", "0.50"))
        t2_pct = float(os.environ.get("PREMIUM_ORB_T2_PCT", "1.00"))

        st = self._porb.get(name)
        if not st or st.get("date") != today:
            st = {"date": today, "CE": None, "PE": None}
            self._porb[name] = st

        # Warm the chain cache if the scanning strategy didn't already
        # (TTL-guarded — no extra API call when auto/conductor prefetched).
        self._ensure_chain_analytics(name, inst, df)
        chain = (self._chain_cache.get(name) or {}).get("chain") or []
        if not chain:
            return

        # ── Strike selection (once per day per side) ──────────────────────
        for side in ("CE", "PE"):
            if st[side] is None:
                cands = [r for r in chain if r.get("type") == side and (r.get("ltp") or 0) > 0]
                if not cands:
                    continue
                pick = min(cands, key=lambda r: abs(float(r["ltp"]) - target_prem))
                st[side] = {
                    "symbol": pick["symbol"], "token": pick["token"],
                    "strike": pick["strike"], "expiry": pick.get("expiry", ""),
                    "or_high": float(pick["ltp"]), "or_low": float(pick["ltp"]),
                    "fired": False, "picked_at": hm,
                }
                log.info(f"🎯 PremiumORB {name} {side}: tracking {pick['symbol']} "
                         f"@ ₹{pick['ltp']} (target ₹{target_prem:.0f})")

        # ── Update each side's premium range / check breakout ─────────────
        for side in ("CE", "PE"):
            sd = st[side]
            if not sd:
                continue
            row = next((r for r in chain if str(r.get("token")) == str(sd["token"])), None)
            if not row or not (row.get("ltp") or 0) > 0:
                continue
            ltp = float(row["ltp"])

            if hm < or_end:
                sd["or_high"] = max(sd["or_high"], ltp)
                sd["or_low"] = min(sd["or_low"], ltp)
                continue

            if sd["fired"] or sd["or_high"] <= 0:
                continue
            # Entry: premium breaks its own opening-range high (0.1% buffer
            # against the mid-price jitter of the sampled range).
            if ltp < sd["or_high"] * 1.001:
                continue

            # Regime gate — skip while classifier reads 'range'.
            if os.environ.get("PREMIUM_ORB_REGIME_GATE", "true").lower() == "true":
                label, rdiag = SignalGen.classify_regime(df)
                if label == "range":
                    self._log_gate(name, "PORB_RANGE_REGIME", None,
                                   {"side": side, "ltp": ltp, "adx": rdiag.get("adx")})
                    continue
            if self._check_killswitch():
                continue
            _max_open = int(os.environ.get("MAX_OPEN_POSITIONS", "2") or 0)
            if _max_open > 0:
                try:
                    open_cnt_row = db_exec(
                        "SELECT COUNT(*) as cnt FROM signals WHERE status='OPEN' AND date=?",
                        (today,), fetchone=True)
                    if int(dict(open_cnt_row).get("cnt", 0)) >= _max_open:
                        self._log_gate(name, "PORB_MAX_OPEN", None, {"side": side})
                        continue
                except Exception:
                    pass

            sd["fired"] = True
            spot = float(df["close"].iloc[-1]) if df is not None and len(df) else 0.0
            entry_prem = ltp
            sl_prem = round(entry_prem * (1 - sl_pct), 2)
            t1_prem = round(entry_prem * (1 + t1_pct), 2)
            t2_prem = round(entry_prem * (1 + t2_pct), 2)
            direction = "LONG" if side == "CE" else "SHORT"
            # Index-level display approximations at delta ≈ 0.5 (exits are
            # premium-based via PLTracker; these only render on the card).
            idx_sl_dist = (entry_prem * sl_pct) / 0.5
            idx_t1_dist = (entry_prem * t1_pct) / 0.5
            sgn = 1 if direction == "LONG" else -1
            sig = {
                "direction": direction, "confidence": 70,
                "price": round(spot, 2), "entry": round(spot, 2),
                "sl": round(spot - sgn * idx_sl_dist, 2),
                "target1": round(spot + sgn * idx_t1_dist, 2),
                "target2": round(spot + sgn * idx_t1_dist * 2, 2),
                "risk_reward": round(t1_pct / sl_pct, 2),
                "reasons": [
                    f"Premium ORB breakout: {sd['symbol']} broke its 09:16-{or_end} "
                    f"premium range high ₹{sd['or_high']:.1f} (now ₹{ltp:.1f})",
                    f"Premium SL -{sl_pct*100:.0f}% / T1 +{t1_pct*100:.0f}% / T2 +{t2_pct*100:.0f}%",
                    "Basis: Zerodha ITM cost-inclusive NIFTY ORB backtest (2022-2026)",
                ],
                "indicators": {"or_high": sd["or_high"], "or_low": sd["or_low"],
                                "entry_premium": entry_prem},
                "strategy": "premium_orb",
                "timestamp": now.strftime("%H:%M:%S"),
            }
            lots = 1
            opt = {
                "symbol": sd["symbol"], "strike": sd["strike"], "type": side,
                "expiry": sd["expiry"], "token": sd["token"],
                "entry": entry_prem, "ltp": entry_prem,
                "sl": sl_prem, "target1": t1_prem, "target2": t2_prem,
                "lot_size": inst.get("lot_size", 75), "lots": lots,
                "capital": round(entry_prem * inst.get("lot_size", 75) * lots, 0),
            }
            save_signal(name, sig, opt)
            self.metrics["signals_generated"] += 1
            self.metrics["signals_alerted"] += 1
            self._last_signal[name] = now
            self.alerts.insert(0, {
                "id": int(time.time() * 1000),
                "time": now.strftime("%H:%M:%S"), "date": today,
                "iso_ts": now.strftime("%Y-%m-%d %H:%M:%S"),
                "weekday": now.strftime("%a"),
                "instrument": name, "signal": sig, "option": opt,
                "timing": None, "ai": None,
            })
            self.alerts = self.alerts[:100]
            log.info(f"🚨 PremiumORB {name} {side} BUY {sd['symbol']} @ ₹{entry_prem} "
                     f"(OR high ₹{sd['or_high']:.1f}) SL ₹{sl_prem} T1 ₹{t1_prem}")
            SlackAlert.send(
                f"🚨 *Premium ORB — {name} {side} BUY*\n"
                f"{sd['symbol']}  ·  Entry ₹{entry_prem}  (premium broke its "
                f"morning range high ₹{sd['or_high']:.1f})\n"
                f"SL ₹{sl_prem} (-{sl_pct*100:.0f}%)  ·  T1 ₹{t1_prem} (+{t1_pct*100:.0f}%)  ·  "
                f"T2 ₹{t2_prem} (+{t2_pct*100:.0f}%)\n"
                f"1 lot = {opt['lot_size']} qty ≈ ₹{opt['capital']:.0f}  ·  "
                f"Time-stop: exit if premium hasn't gained 15% in 30 min\n"
                f"_Basis: cost-inclusive NIFTY options ORB backtest (Zerodha ITM, 2022-2026)_")

    def _ensure_chain_analytics(self, name, inst, df):
        """Return fresh chain analytics for `name`, fetching the option chain
        proactively when the cache is older than CHAIN_PREFETCH_TTL (default
        120s). This feeds Conductor's flow dimension BEFORE the signal call.

        Fetch cost: one option_chain call per instrument per TTL window —
        modest next to the candle fetches already happening each scan. Never
        raises; on any failure it returns whatever the cache can serve, else
        None (flow dimension simply abstains, exactly as before).
        """
        try:
            ttl = int(os.environ.get("CHAIN_PREFETCH_TTL", "120") or 120)
        except Exception:
            ttl = 120
        now_ts = time.time()
        _cc = self._chain_cache.get(name, {})
        age = now_ts - _cc.get("ts", 0)
        if age >= ttl or not _cc.get("chain"):
            try:
                spot = float(df["close"].iloc[-1])
                chain, atm_val = self.client.option_chain(inst, spot)
                if chain:
                    self._update_oi_history(name, chain, now_ts)
                    self._chain_cache[name] = {
                        "ts": now_ts,
                        "ts_str": datetime.now(IST).strftime("%H:%M:%S"),
                        "chain": chain,
                        # No option picked in the prefetch path; STEP 1 picks
                        # one when a signal actually fires.
                        "opt": _cc.get("opt"),
                        "atm": atm_val,
                    }
                    _cc = self._chain_cache[name]
            except Exception as e:
                _k = ("chain_prefetch", name)
                if self._cond_diag_last.get(_k) != str(e)[:80]:
                    self._cond_diag_last[_k] = str(e)[:80]
                    log.warning(f"  {name} chain prefetch failed (flow dimension abstains): {e}")
        if _cc.get("chain") and _cc.get("atm") is not None:
            try:
                oi_d = self._compute_oi_delta(name)
                return OptPicker.chain_analytics(_cc["chain"], _cc["atm"], oi_delta=oi_d)
            except Exception:
                return None
        return None

    def _update_oi_history(self, name, chain, now_ts):
        """Record OI + volume snapshot for each (strike, type) pair.
        Called after each fresh chain fetch — builds a rolling buffer used
        by _compute_oi_delta() to detect OI shifts between scans.
        """
        if name not in self._oi_history:
            self._oi_history[name] = {}
        for o in chain:
            key = (int(o.get("strike", 0)), str(o.get("type", "")))
            if key not in self._oi_history[name]:
                self._oi_history[name][key] = self._deque(maxlen=12)  # ~6 min at 30s/fetch
            self._oi_history[name][key].append({
                "oi":     int(o.get("oi", 0) or 0),
                "volume": int(o.get("volume", 0) or 0),
                "ltp":    float(o.get("ltp", 0) or 0),
                "ts":     now_ts,
            })

    def _compute_oi_delta(self, name):
        """Compare current OI/volume vs oldest snapshot to find velocity.
        Returns dict { (strike, type): {velocity, oi_delta, vol_delta, dt_min, ...} }
        Velocity: BUILDING (+) / UNWINDING (-) / STABLE.
        NOTE: OI from Angel One FULL mode is often 0 — in that case, volume delta
        (which IS real-time) is used as the primary momentum signal.
        """
        hist = self._oi_history.get(name, {})
        result = {}
        for key, snapshots in hist.items():
            if len(snapshots) < 2:
                continue
            cur = snapshots[-1]
            old = snapshots[0]
            dt_sec = max(1, cur["ts"] - old["ts"])
            dt_min = round(dt_sec / 60.0, 1)
            oi_chg  = cur["oi"]     - old["oi"]
            vol_chg = cur["volume"] - old["volume"]
            # OI-based classification (when OI data is non-zero)
            if cur["oi"] > 0 or old["oi"] > 0:
                oi_pct = round(oi_chg / max(old["oi"], 1) * 100, 1)
                if oi_chg > 10000:   velocity = "BUILDING"
                elif oi_chg < -10000: velocity = "UNWINDING"
                else:                velocity = "STABLE"
            else:
                # OI data unavailable — fall back to volume momentum
                oi_pct = 0
                if vol_chg > 2000:   velocity = "BUILDING"    # heavy volume = fresh positions
                elif vol_chg < -500:  velocity = "UNWINDING"   # volume drying up = positions closing
                else:                velocity = "STABLE"
            # Volume trend (always meaningful — real-time)
            if vol_chg > 1000:   vol_trend = "ACCELERATING"
            elif vol_chg > 200:   vol_trend = "STEADY"
            else:                vol_trend = "SLOWING"
            result[key] = {
                "oi_delta":  oi_chg,
                "vol_delta": vol_chg,
                "oi_pct":    oi_pct,
                "velocity":  velocity,
                "vol_trend": vol_trend,
                "dt_min":    dt_min,
            }
        return result

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
                # Layer C feedback (yesterday's EOD adjustments) + per-day metrics reset.
                self._maybe_load_adjustments(now)

                # Auto-close cutoff is configurable (default 15:15 IST). Check FIRST so
                # close_all() reliably runs before the early-exit branch below.
                close_h = int(CONFIG.get("auto_close_hour", 15))
                close_m = int(CONFIG.get("auto_close_minute", 15))
                if now.hour == close_h and now.minute >= close_m:
                    self.tracker.close_all()
                    # Run Layer C EOD learning RIGHT BEFORE we stop — the previous
                    # 15:45 trigger never fired because the loop exited at 15:25.
                    try:
                        LearningLoop.run()
                        self._last_eod_run = now.strftime("%Y-%m-%d")
                    except Exception as e:
                        log.warning(f"  EOD learning at close failed: {e}")
                    self.running=False
                    log.info(f"🔔 Auto-close at {close_h:02d}:{close_m:02d}");break
                if now.hour<9 or(now.hour==9 and now.minute<20) or \
                   now.hour>close_h or(now.hour==close_h and now.minute>=close_m):
                    time.sleep(30);continue
                self.metrics["scans_total"] += 1

                # ── Market-closed short-circuit (weekend / NSE holiday) ──
                # Skip the whole scan when the exchange is definitively closed.
                # Saves API calls and lets the dashboard say "Market closed —
                # <reason>" instead of pretending to scan a dead market. We only
                # short-circuit on weekends (certain) and LISTED holidays
                # (conservative, env-extendable); an unknown holiday still falls
                # through harmlessly — the strategy time-gate won't trade on it.
                _ms = market_status(now)
                if _ms["reason"] in ("weekend", "holiday"):
                    _td = now.strftime("%Y-%m-%d")
                    if self._mkt_closed_logged != _td:
                        self._mkt_closed_logged = _td
                        log.info(f"🛑 Market closed today ({_ms['reason']}: "
                                 f"{_ms.get('label')}) — scanning paused")
                    time.sleep(60); continue

                # Mid/late-session learning + in-flight management
                self._maybe_eod(now)
                self._maybe_inflight()

                # P&L check
                self.tracker.check()

                # Regime-level overrides (Layer A)
                regime = self._regime or RegimeBrief.today()
                avoid = set((regime or {}).get("avoid_instruments") or [])
                # BUG FIX #6: Do NOT use regime.confidence_floor to override MIN_CONFIDENCE.
                # RegimeBrief asks Claude for a floor in [55,80] every morning, which would
                # immediately undo our lowered 45% threshold and block ALL signals.
                # Regime can control avoid_instruments, bias, and min_rr — but confidence
                # gating is the engine's job.
                conf_floor = CONFIG["min_confidence"]  # always 45% (env MIN_CONFIDENCE)
                # Cap regime min_rr at 1.9 — Claude sometimes returns 2.0-2.5 on normal
                # days which blocks every signal since T1 is set at 2.0× risk.
                # The hard floor is 1.2; the hard ceiling is 1.9 to always leave headroom.
                min_rr_floor = min(max(1.2, float((regime or {}).get("min_rr") or 1.2)), 1.9)

                # Event blackout (Layer E) — short-circuit the whole scan
                blackout, ev = EventCalendar.in_blackout()
                if blackout:
                    log.info(f"🚫 Event blackout active: {ev.get('name')} ({ev.get('blackout',{})}) — skipping scan")
                    time.sleep(30); continue

                # Filter to only instruments enabled by env flag.
                # ENABLED_INSTRUMENTS="BANKNIFTY" → scan only BANKNIFTY.
                # Used during the Conductor live-verification phase since
                # backtest showed NIFTY/FINNIFTY lose on Conductor; only
                # BANKNIFTY has confirmed edge. Default = all 3 (no filter).
                _enabled_env = os.environ.get("ENABLED_INSTRUMENTS", "").strip()
                _enabled = {x.strip().upper() for x in _enabled_env.split(",") if x.strip()} if _enabled_env else None

                for name,inst in INSTRUMENTS.items():
                    if _enabled and name not in _enabled:
                        # Silent skip — only log once when the engine starts up
                        continue
                    # BUG FIX #7: regime avoid_instruments must NOT override an explicit
                    # ENABLED_INSTRUMENTS allow-list. When the engine is pinned to a single
                    # confirmed-edge instrument (BANKNIFTY+conductor), one morning RegimeBrief
                    # opinion that drops it onto the avoid list silently kills the ENTIRE day —
                    # and self-perpetuates, since the recent losses it reacts to never get
                    # replaced by new trades. Same anti-pattern as the confidence_floor (#6) and
                    # min_rr guards above. Only honor regime avoid when NOT running a pinned list.
                    if name in avoid:
                        if _enabled:
                            log.info(f"  ⚠️ regime flagged {name} on avoid list — OVERRIDDEN (operator-enabled via ENABLED_INSTRUMENTS)")
                        else:
                            self.metrics["regime_blocked"] += 1
                            log.info(f"  {name} skipped — regime avoid list")
                            continue

                    df=self.client.candles(inst["token"],inst["exchange"])
                    if df.empty or len(df)<30:
                        # BUG FIX: this used to be a SILENT continue. When the
                        # candle feed returns <30 bars the scan produced nothing
                        # with zero trace — exactly the kind of quiet failure that
                        # hides for days. Log it (change-only) so it's visible.
                        _k = (name, "INSUFFICIENT_BARS", 0 if df.empty else len(df))
                        if self._cond_diag_last.get(("bars", name)) != _k:
                            self._cond_diag_last[("bars", name)] = _k
                            log.warning(f"📭 {name} no scan — candle feed returned "
                                        f"{0 if df.empty else len(df)} bars (<30 needed)")
                        continue
                    # Track the newest candle date the feed has actually served.
                    # If this lags behind 'today' on a trading day the data is
                    # stale (holiday or feed outage) and the dashboard flags it.
                    try:
                        _tcol = "ts" if "ts" in df.columns else "timestamp"
                        _nbd = pd.to_datetime(df[_tcol].iloc[-1]).strftime("%Y-%m-%d")
                        if not self._newest_bar_date or _nbd > self._newest_bar_date:
                            self._newest_bar_date = _nbd
                    except Exception:
                        pass

                    # ── v2 regime gate (Phase 2): check VIX, day-of-week, expiry-window ──
                    # Only active when STRATEGY=v2; v1 path keeps the legacy time gate inside analyze().
                    strategy = CONFIG.get("strategy", "v1").lower()
                    if strategy == "v2":
                        try:
                            from regime import RegimeFilter
                            ok, reason = RegimeFilter.should_trade(
                                angel_client=self.client, symbol=name)
                            if not ok:
                                # Don't even bother computing the signal — bail early.
                                if name not in (self._prev or {}):
                                    log.info(f"  {name} v2 regime BLOCK: {reason}")
                                self.metrics.setdefault("regime_blocked", 0)
                                self.metrics["regime_blocked"] += 1
                                continue
                        except Exception as e:
                            log.warning(f"  regime filter crashed (failing open): {e}")

                    # Fetch chain analytics BEFORE the signal call so the flow
                    # dimension (PCR / IV skew / OI velocity) can actually vote.
                    # The old code only read a cache that was populated AFTER a
                    # signal fired — a chicken-and-egg that starved the flow
                    # dimension on nearly every scan, silently turning the
                    # "3-of-5" confluence gate into 3-of-4. Now we proactively
                    # fetch the chain on a CHAIN_PREFETCH_TTL cadence (default
                    # 120s) for flow-aware strategies.
                    _chain_anal_for_sig = None
                    if strategy in ("conductor", "auto"):
                        _chain_anal_for_sig = self._ensure_chain_analytics(name, inst, df)

                    # ── Premium ORB module (independent of strategy dispatch) ──
                    # Evidence-backed premium-breakout buyer for NIFTY; manages
                    # its own once-per-side-per-day state and delivery.
                    try:
                        self._premium_orb_tick(name, inst, df)
                    except Exception as _pe:
                        log.warning(f"  PremiumORB tick error {name}: {_pe}")

                    sig=self.sgen.analyze(df, weight_adj=self._weight_adj,
                                          blocked_windows=self._blocked_windows,
                                          symbol=name,
                                          chain_analytics=_chain_anal_for_sig)

                    # ── Capture v2 diagnostic EVERY scan (signal or not) ──
                    # The v2 analyzer always populates SignalGenV2.last_decision
                    # so we can see why each bar didn't fire — no more black-box silence.
                    if strategy == "v2":
                        try:
                            from signal_v2 import SignalGenV2
                            diag = dict(SignalGenV2.last_decision or {})
                            if diag:
                                diag["scan_at"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                                diag["instrument"] = name
                                buf = self._v2_diag.setdefault(name, self._deque(maxlen=20))
                                buf.append(diag)
                                # Log if close-to-trigger (within 1 of threshold) so we can
                                # spot when v2 is "almost firing" — useful for tuning.
                                near_trigger = max(diag.get("long_score",0), diag.get("short_score",0))
                                trig = diag.get("trigger", 3)
                                if near_trigger >= trig - 1:
                                    log.info(f"[v2 diag] {name} {diag.get('verdict','?')} "
                                             f"L={diag.get('long_score','?')} S={diag.get('short_score','?')} "
                                             f"RSI={diag.get('rsi','?')} VWAP_dev%={diag.get('vwap_dev_pct','?')} "
                                             f"range={diag.get('range_ratio','?')}×")
                        except Exception as e:
                            log.warning(f"  v2 diag capture failed: {e}")

                    # ── Capture conductor diagnostic EVERY scan (signal or not) ──
                    # Conductor.last_decision holds the real verdict for this bar
                    # (TRIGGER / AFTER_WINDOW / BELOW_MIN_DIMENSIONS / RR_BELOW_MIN
                    # / …). It used to live only in-process — invisible. Surface it
                    # via /api/status and loud-log the no-signal reason (change-only,
                    # so a 14:30+ AFTER_WINDOW doesn't spam 120×/hour).
                    if strategy in ("conductor", "auto"):
                        try:
                            _ran = (SignalGen.last_dispatch or {}).get("actually_ran") or ""
                            if _ran.endswith("reverter"):
                                from signal_reverter import Reverter as _DiagEngine
                            else:
                                from conductor import Conductor as _DiagEngine
                            cdiag = dict(_DiagEngine.last_decision or {})
                            if cdiag:
                                cdiag["scan_at"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
                                cdiag["instrument"] = name
                                cdiag["engine"] = _ran or strategy
                                if strategy == "auto":
                                    cdiag["auto_regime"] = (SignalGen.last_dispatch or {}).get("auto_regime")
                                    cdiag["auto_adx"] = (SignalGen.last_dispatch or {}).get("auto_adx")
                                self._cond_diag[name] = cdiag
                                if not sig:
                                    _ck = (cdiag.get("verdict"),)
                                    if self._cond_diag_last.get(name) != _ck:
                                        self._cond_diag_last[name] = _ck
                                        log.info(f"📭 {name} no signal — "
                                                 f"{cdiag.get('verdict')} "
                                                 f"{cdiag.get('votes', cdiag.get('time',''))}")
                        except Exception as e:
                            log.warning(f"  conductor diag capture failed: {e}")

                    # ── MACD Scalper runs alongside Conductor — no strategy switch needed ──
                    # Fires independently on MACD histogram zero-line crossover + EMA9/EMA21.
                    # Only activates when Conductor is quiet (avoids duplicate signals on same
                    # bar). Under 'auto' it piggybacks only on trend-routed scans — MACD
                    # crossovers are a trend tool; firing them on range days is the exact
                    # whipsaw the 90d backtests already killed.
                    _cond_ran = strategy == "conductor" or (
                        strategy == "auto"
                        and ((SignalGen.last_dispatch or {}).get("actually_ran") or "").endswith("conductor"))
                    if not sig and _cond_ran:
                        try:
                            from macd_scalper import MACDScalper
                            _ms = MACDScalper.analyze(df, symbol=name)
                            if _ms:
                                sig = _ms
                                _ind = sig.get("indicators", {})
                                log.info(f"⚡ {name} MACD Scalper: {sig['direction']} "
                                         f"hist {_ind.get('macd_hist_prev'):+.1f}→{_ind.get('macd_hist'):+.1f}  "
                                         f"EMA9={_ind.get('ema9'):.0f}/{_ind.get('ema21'):.0f}  "
                                         f"RSI={_ind.get('rsi'):.0f}")
                        except Exception as _mse:
                            log.warning(f"  MACD Scalper error: {_mse}")

                    # ── Expiry-day gamma-blast piggyback (14:15-14:45) ──
                    # Compressed expiry afternoons attacking a strike can 2-3x
                    # premium in minutes; most attempts die — HALF SIZE only.
                    if not sig and strategy in ("conductor", "auto"):
                        try:
                            from regime import RegimeFilter as _RF
                            _nw = datetime.now(IST)
                            if _RF._is_expiry_day(_nw, name) and \
                                    "14:15" <= _nw.strftime("%H:%M") <= "14:45":
                                from signal_gamma import SignalGenGamma
                                _g = SignalGenGamma.analyze(df, symbol=name)
                                if _g:
                                    _g["priority"] = "half"
                                    _g.setdefault("reasons", []).append(
                                        "Expiry gamma window — HALF SIZE, lottery-risk profile")
                                    sig = _g
                        except Exception as _ge:
                            log.warning(f"  gamma piggyback error: {_ge}")

                    if not sig:
                        # Either too few bars OR analyzer returned None due to a blocked
                        # time window or hard time-gate (post-14:50). Track the window hits.
                        if self._blocked_windows:
                            now_hm = datetime.now(IST).strftime("%H:%M")
                            for win in self._blocked_windows:
                                try:
                                    a, b = win.split("-")
                                    if a.strip() <= now_hm <= b.strip():
                                        self.metrics["blocked_window_hits"] += 1
                                        break
                                except Exception: continue
                        continue
                    self.metrics["signals_generated"] += 1

                    # ── Signal-bar volume check (soft — confidence, not gate) ──
                    try:
                        if "volume" in df.columns and len(df) >= 21:
                            _vr = float(df["volume"].iloc[-1]) / \
                                max(1.0, float(df["volume"].iloc[-21:-1].mean()))
                            if _vr < 1.2:
                                sig["confidence"] = max(10, int(sig["confidence"]) - 8)
                                sig.setdefault("reasons", []).append(
                                    f"Thin signal-bar volume {_vr:.2f}× (<1.2×): confidence -8")
                    except Exception:
                        pass

                    # ── Weekly-expiry-day structure ──
                    # 09:15-10:30 = gamma zone (buys OK, tight rules) ·
                    # 10:30-14:00 = theta zone (no new buys) · 14:00+ flat
                    # (tracker enforces); gamma piggyback re-enters 14:15+.
                    try:
                        from regime import RegimeFilter as _RF
                        _nw = datetime.now(IST)
                        if _RF._is_expiry_day(_nw, name):
                            _hm = _nw.strftime("%H:%M")
                            if "10:30" <= _hm < "14:00":
                                self._log_gate(name, "EXPIRY_MIDDAY_BLOCK", sig,
                                               {"window": "10:30-14:00 theta zone"})
                                continue
                            if _hm < "10:30":
                                sig.setdefault("reasons", []).append(
                                    "Expiry morning gamma window: ATM only · 30% SL · "
                                    "30-min cut · flat by 14:00")
                    except Exception:
                        pass

                    # ── DRY_RUN_V2 (Phase 2): observe v2 signals live without trading ──
                    # When STRATEGY=v2 AND DRY_RUN_V2=true: fire to Slack + log only.
                    # Skip OptPicker, kill-switch, Layer B, save_signal. Comes BEFORE all
                    # those gates so dry-run signals are never blocked by them.
                    if strategy == "v2" and CONFIG.get("dry_run_v2", False):
                        log.info(f"📋 [DRY RUN v2] {name} {sig['direction']} "
                                 f"conf={sig['confidence']}%  score={sig.get('v2_score','?')}/4  "
                                 f"price={sig.get('price','?')}  rr={sig.get('risk_reward','?')}  "
                                 f"reasons={'; '.join(sig.get('reasons',[])[:3])}")
                        try:
                            SlackAlert.send(
                                f"📋 *[DRY RUN v2]* {name} {sig['direction']}\n"
                                f"  Confidence: {sig['confidence']}%  ·  Score: {sig.get('v2_score','?')}/4\n"
                                f"  Spot ₹{sig.get('price')}  ·  Entry ₹{sig.get('entry')}  ·  SL ₹{sig.get('sl')}  ·  T1 ₹{sig.get('target1')}\n"
                                f"  R:R {sig.get('risk_reward')}\n"
                                f"  Reasons: {'; '.join(sig.get('reasons',[])[:3])}\n"
                                f"  _(observation only — no trade taken)_"
                            )
                        except Exception:
                            pass
                        # Update self.latest so dashboard sees v2 signals
                        self.latest[name] = {
                            "instrument": name, "lot_size": inst["lot_size"],
                            "signal": sig, "option": None, "timing": None,
                            "chain_analytics": {}, "dry_run": True,
                            "updated_at": datetime.now(IST).strftime("%H:%M:%S"),
                        }
                        # Don't continue to OptPicker / save_signal / kill-switch.
                        # Just track in metrics and move on.
                        self.metrics.setdefault("dry_run_v2_fires", 0)
                        self.metrics["dry_run_v2_fires"] += 1
                        continue

                    # ════════════════════════════════════════════════════════
                    # STEP 1: Fetch option chain + analytics ALWAYS
                    # This must happen OUTSIDE the alert cooldown so that
                    # self.latest always carries real option data (strike,
                    # premium, Greeks, PCR) — not null.
                    # Rate-limited: re-fetch at most every 30s per instrument.
                    # ════════════════════════════════════════════════════════
                    now_ts = time.time()
                    opt         = None
                    chain       = None
                    atm_val     = None
                    greeks      = None
                    chain_anal  = {}

                    # Use cached chain if fetched in the last 30 s
                    _cc = self._chain_cache.get(name, {})
                    _cc_age = now_ts - _cc.get("ts", 0)

                    # Compute OI delta BEFORE chain_analytics (uses rolling history)
                    # This is always recomputed from history regardless of cache state
                    oi_delta = self._compute_oi_delta(name)

                    if _cc_age < 30 and _cc.get("opt") is not None:
                        opt     = _cc["opt"]
                        atm_val = _cc["atm"]
                        chain   = _cc.get("chain")
                        # Recompute analytics with FRESH oi_delta even on cache hit —
                        # volume velocity changes every few seconds even when chain prices don't
                        cached_chain = _cc.get("chain") or []
                        chain_anal = OptPicker.chain_analytics(cached_chain, atm_val, oi_delta=oi_delta)
                        log.info(f"  {name}: using cached chain (age {_cc_age:.0f}s) | "
                                 f"OI shift={chain_anal.get('oi_shift_signal','?')}")
                    elif _cc_age < 30 and _cc.get("chain") and _cc.get("atm") is not None:
                        # Prefetch path cached a raw chain (no option picked yet).
                        # Pick from the cached chain instead of re-fetching.
                        chain   = _cc["chain"]
                        atm_val = _cc["atm"]
                        try:
                            expiry = chain[0].get("expiry", "") if chain else ""
                            greeks = self.client.option_greeks(
                                inst.get("expiry_prefix", name), expiry) if expiry else None
                            opt = self.opick.pick(sig, inst, chain, atm_val,
                                                  CONFIG.get("budget", 20000), greeks=greeks)
                            self._chain_cache[name]["opt"] = opt
                        except Exception as e:
                            log.warning(f"  {name}: pick from prefetched chain failed: {e}")
                        chain_anal = OptPicker.chain_analytics(chain, atm_val, oi_delta=oi_delta)
                    elif sig.get("direction") in ("LONG", "SHORT"):
                        try:
                            chain, atm_val = self.client.option_chain(inst, sig["price"])
                            if chain:
                                # Update OI history with fresh data BEFORE computing analytics
                                self._update_oi_history(name, chain, now_ts)
                                oi_delta = self._compute_oi_delta(name)  # recompute with new snapshot

                                expiry = chain[0].get("expiry", "") if chain else ""
                                greeks = self.client.option_greeks(
                                    inst.get("expiry_prefix", name), expiry) if expiry else None
                                opt = self.opick.pick(sig, inst, chain, atm_val,
                                                      CONFIG.get("budget", 20000), greeks=greeks)
                                chain_anal = OptPicker.chain_analytics(chain, atm_val, oi_delta=oi_delta)
                                # Cache raw chain + opt for 30s (so analytics can be recomputed fresh)
                                self._chain_cache[name] = {
                                    "ts": now_ts,
                                    "ts_str": datetime.now(IST).strftime("%H:%M:%S"),
                                    "chain": chain,   # raw chain for delta recompute on cache hit
                                    "opt": opt,
                                    "atm": atm_val
                                }
                                log.info(f"  {name}: chain fetched — {opt.get('strike') if opt else '?'} "
                                         f"{opt.get('type','?') if opt else ''} ₹{opt.get('ltp','?') if opt else '?'} | "
                                         f"PCR={chain_anal.get('pcr')} skew={chain_anal.get('iv_skew')} "
                                         f"OI_shift={chain_anal.get('oi_shift_signal','?')}")
                        except Exception as ce:
                            self.metrics["chain_failures"] += 1
                            log.error(f"  Chain fetch failed for {name}: {ce} — "
                                      f"holding back {sig.get('direction')} signal this cycle, "
                                      f"will retry next scan")
                            # An alert without a tradeable option (no strike/token)
                            # can't be acted on — but skipping silently all day is
                            # the 2-week-Conductor-bug pattern. Slack the failure,
                            # rate-limited to once per 15 min per instrument.
                            self._log_gate(name, "CHAIN_FETCH_FAIL", sig, {"error": str(ce)[:200]})
                            _last_warn = self._chain_fail_alerted.get(name, 0)
                            if now_ts - _last_warn > 900:
                                self._chain_fail_alerted[name] = now_ts
                                SlackAlert.send(
                                    f"⚠️ {name}: option chain fetch FAILED — a live "
                                    f"{sig.get('direction')} signal (conf {sig.get('confidence','?')}) "
                                    f"is being held back. Retrying every scan; this warning "
                                    f"repeats at most every 15 min. Error: {ce}")
                            continue

                    # ════════════════════════════════════════════════════════
                    # STEP 1b: Option chain confidence boost
                    # The index signal gives ~30-50% confidence. Chain data
                    # (PCR, IV skew, ATM volume) can confirm or deny direction
                    # and push a signal over the alert threshold.
                    # ════════════════════════════════════════════════════════
                    if chain_anal:
                        ca_boost = 0
                        ca_notes = []
                        pcr    = chain_anal.get("pcr", 1.0) or 1.0
                        iv_sk  = chain_anal.get("iv_skew", 0) or 0
                        ce_vol = chain_anal.get("total_ce_vol", 0) or 0
                        pe_vol = chain_anal.get("total_pe_vol", 0) or 0
                        dir_   = sig.get("direction")

                        # PCR confirms direction
                        if dir_ == "SHORT" and pcr > 1.1:
                            ca_boost += 10; ca_notes.append(f"PCR {pcr:.2f} confirms bearish")
                        elif dir_ == "SHORT" and pcr > 0.9:
                            ca_boost += 5; ca_notes.append(f"PCR {pcr:.2f} neutral")
                        elif dir_ == "SHORT" and pcr < 0.7:
                            ca_boost -= 8; ca_notes.append(f"⚠️ PCR {pcr:.2f} bullish — contradicts SHORT")
                        if dir_ == "LONG" and pcr < 0.7:
                            ca_boost += 10; ca_notes.append(f"PCR {pcr:.2f} confirms bullish")
                        elif dir_ == "LONG" and pcr < 0.9:
                            ca_boost += 5; ca_notes.append(f"PCR {pcr:.2f} neutral")
                        elif dir_ == "LONG" and pcr > 1.1:
                            ca_boost -= 8; ca_notes.append(f"⚠️ PCR {pcr:.2f} bearish — contradicts LONG")

                        # IV skew confirms direction (put skew = bearish pressure)
                        if dir_ == "SHORT" and iv_sk > 2:
                            ca_boost += 7; ca_notes.append(f"IV skew +{iv_sk:.1f} put premium (bearish fear)")
                        elif dir_ == "LONG" and iv_sk < -2:
                            ca_boost += 7; ca_notes.append(f"IV skew {iv_sk:.1f} call premium (bullish)")

                        # Volume leadership confirms momentum (truly real-time from NSE)
                        if dir_ == "SHORT" and pe_vol > ce_vol * 1.3:
                            ca_boost += 8; ca_notes.append(f"PE vol {pe_vol:,} > CE {ce_vol:,} (bearish flow)")
                        elif dir_ == "LONG" and ce_vol > pe_vol * 1.3:
                            ca_boost += 8; ca_notes.append(f"CE vol {ce_vol:,} > PE {pe_vol:,} (bullish flow)")

                        # ── OI VELOCITY BOOST (shift detection — highest signal quality) ──
                        # OI building = fresh positions = institutional conviction at that strike
                        # OI unwinding = positions closing = level losing significance
                        # OI roll = shift from far strike to near strike = directional conviction
                        oi_shift   = chain_anal.get("oi_shift_signal", "NONE")
                        atm_ce_d   = chain_anal.get("atm_ce_oi_delta", {})
                        atm_pe_d   = chain_anal.get("atm_pe_oi_delta", {})
                        vol_accel_ce = chain_anal.get("vol_accel_ce", [])
                        vol_accel_pe = chain_anal.get("vol_accel_pe", [])

                        if dir_ == "LONG":
                            if atm_pe_d.get("velocity") == "BUILDING":
                                ca_boost += 10
                                ca_notes.append(f"🔥 ATM PE OI building (+{atm_pe_d.get('oi_delta',0):,} in {atm_pe_d.get('dt_min','?')}min) — fresh support")
                            if atm_ce_d.get("velocity") == "UNWINDING":
                                ca_boost += 7
                                ca_notes.append("📉 ATM CE OI unwinding — resistance dissolving")
                            if oi_shift == "CE_ROLL_BULLISH":
                                ca_boost += 8; ca_notes.append("🔄 CE OI roll — bullish conviction")
                            if atm_ce_d.get("velocity") == "BUILDING":
                                ca_boost -= 6; ca_notes.append(f"⚠️ ATM CE OI building — fresh resistance above")
                            if vol_accel_ce:
                                ca_boost += 5
                                ca_notes.append(f"⚡ CE vol surge +{vol_accel_ce[0]['vol_delta']:,} at {vol_accel_ce[0]['strike']}")
                        elif dir_ == "SHORT":
                            if atm_ce_d.get("velocity") == "BUILDING":
                                ca_boost += 10
                                ca_notes.append(f"🔥 ATM CE OI building (+{atm_ce_d.get('oi_delta',0):,} in {atm_ce_d.get('dt_min','?')}min) — fresh resistance")
                            if atm_pe_d.get("velocity") == "UNWINDING":
                                ca_boost += 7
                                ca_notes.append("📉 ATM PE OI unwinding — support dissolving")
                            if oi_shift == "PE_ROLL_BEARISH":
                                ca_boost += 8; ca_notes.append("🔄 PE OI roll — bearish conviction")
                            if atm_pe_d.get("velocity") == "BUILDING":
                                ca_boost -= 6; ca_notes.append(f"⚠️ ATM PE OI building — fresh support below")
                            if vol_accel_pe:
                                ca_boost += 5
                                ca_notes.append(f"⚡ PE vol surge +{vol_accel_pe[0]['vol_delta']:,} at {vol_accel_pe[0]['strike']}")

                        # Apply boost (raised cap to 35 — OI shift is a high-quality signal)
                        ca_boost = max(-20, min(35, ca_boost))
                        if ca_boost != 0:
                            sig["confidence"] = min(95, max(10, sig["confidence"] + ca_boost))
                            sig.setdefault("reasons", []).extend(ca_notes)
                            log.info(f"  {name}: chain boost {ca_boost:+d}pts → conf={sig['confidence']}% "
                                     f"[OI_shift={oi_shift} atm_ce={atm_ce_d.get('velocity','?')} atm_pe={atm_pe_d.get('velocity','?')}]")

                    # ════════════════════════════════════════════════════════
                    # STEP 1c: OptionsIntel composite gate
                    # Six signals (PCR, OI velocity, IV skew, OI wall, GEX,
                    # max pain) scored against the proposed direction.
                    # BLOCK = options market strongly contradicts direction;
                    # skip saving the signal this cycle.
                    # ════════════════════════════════════════════════════════
                    # ── OI-wall veto: don't buy INTO a reinforced wall ──
                    if chain and sig.get("direction") in ("LONG", "SHORT"):
                        _wv, _wnote = self._oi_wall_veto(name, chain, sig.get("price"),
                                                          sig["direction"])
                        if _wv:
                            log.info(f"  ⛔ {name} OI-wall veto: {_wnote}")
                            self._log_gate(name, "OI_WALL_VETO", sig, {"note": _wnote})
                            self._prev[name] = {"instrument": name, "signal": sig,
                                                 "option": opt, "chain_analytics": chain_anal}
                            continue

                    _oi_intel = None
                    if chain_anal:
                        try:
                            _expiry = (opt or {}).get("expiry", "")
                            _oi_intel = OptionsIntel.score(
                                chain_analytics=chain_anal,
                                chain_raw=chain,
                                spot=sig.get("price", 0),
                                direction=sig.get("direction", "LONG"),
                                expiry=_expiry,
                                instrument=name,
                            )
                            sig["intel"] = _oi_intel
                            log.info(
                                f"  {name} OptionsIntel: {_oi_intel['summary']}"
                            )
                            if _oi_intel["gate"] == "BLOCK":
                                log.warning(
                                    f"  OptionsIntel BLOCK {name} {sig.get('direction')} — "
                                    f"composite={_oi_intel['composite']:+.3f} contradicts direction. "
                                    f"Signal suppressed this cycle."
                                )
                                self._log_gate(name, "OPTIONS_INTEL_BLOCK", sig,
                                               {"composite": _oi_intel.get("composite")})
                                self._prev[name] = {
                                    "instrument": name, "signal": sig,
                                    "option": opt, "chain_analytics": chain_anal,
                                }
                                continue
                            elif _oi_intel["confidence_delta"] != 0:
                                sig["confidence"] = min(95, max(10,
                                    sig["confidence"] + _oi_intel["confidence_delta"]
                                ))
                                sig.setdefault("reasons", []).append(
                                    f"OptionsIntel {_oi_intel['gate']} "
                                    f"({_oi_intel['confidence_delta']:+d}pts, "
                                    f"composite={_oi_intel['composite']:+.2f})"
                                )
                        except Exception as _ie:
                            log.warning(f"  OptionsIntel error for {name}: {_ie}")

                    # ════════════════════════════════════════════════════════
                    # STEP 2: Always update self.latest with real option data
                    # (dashboard polls this; must reflect current scan state)
                    # ════════════════════════════════════════════════════════
                    timing, _ = estimate_exit_time(sig)
                    now_ist_str = datetime.now(IST).strftime("%H:%M:%S")
                    self.latest[name] = {
                        "instrument": name, "lot_size": inst["lot_size"],
                        "signal": sig, "option": opt,
                        "timing": timing, "chain_analytics": chain_anal,
                        "options_intel": _oi_intel,
                        "updated_at": now_ist_str,
                        # Separate timestamp for when the option chain was last fetched
                        # (option prices in 'opt' correspond to this time, not now)
                        "chain_fetched_at": _cc.get("ts_str") or now_ist_str,
                    }
                    # Also store human-readable chain fetch time in cache
                    if name in self._chain_cache:
                        self._chain_cache[name]["ts_str"] = now_ist_str

                    # ════════════════════════════════════════════════════════
                    # STEP 3: 15-min cooldown gates ALERT ONLY — not chain fetch
                    # ════════════════════════════════════════════════════════
                    _last_t = self._last_signal.get(name)
                    if _last_t and (datetime.now(IST) - _last_t).total_seconds() < 900:
                        self._log_gate(name, "ALERT_COOLDOWN", sig,
                                       {"since_last_s": int((datetime.now(IST) - _last_t).total_seconds())})
                        continue

                    result = {"instrument": name, "lot_size": inst["lot_size"],
                              "signal": sig, "option": opt,
                              "timing": timing, "chain_analytics": chain_anal,
                              # signal_date = the date of the bar this was computed
                              # on, so the dashboard can flag prior-day (stale) cards.
                              "signal_date": (self._newest_bar_date
                                              or datetime.now(IST).strftime("%Y-%m-%d")),
                              "updated_at": datetime.now(IST).strftime("%H:%M:%S")}

                    prev = self._prev.get(name, {}).get("signal", {})
                    # Hard R:R gate — honour regime min_rr override (≥1.5)
                    if sig.get("risk_reward", 0) < min_rr_floor:
                        self.metrics["rr_blocked"] += 1
                        log.info(f"⛔ R:R gate blocked {name} {sig['direction']} "
                                 f"RR={sig.get('risk_reward',0)} (need ≥{min_rr_floor})")
                        self._log_gate(name, "RR_BELOW_FLOOR", sig,
                                       {"rr": sig.get("risk_reward"), "floor": min_rr_floor})
                        self._prev[name] = result
                        continue
                    # Daily kill-switch — stops new alerts after loss limit or trade cap.
                    if self._check_killswitch():
                        self._log_gate(name, "KILLSWITCH", sig)
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

                        # ── Re-pick option on fresh spot ONLY if cached chain is stale ──
                        # Previously this always re-fetched, doubling API calls per alert.
                        # Skip the re-fetch when our cached chain is < 30s old; the price
                        # delta on cached chain prices vs fresh-LTP is negligible inside
                        # that window. (The premium-mid was already captured in `chain`.)
                        cc_age_now = time.time() - (self._chain_cache.get(name, {}).get("ts", 0) or 0)
                        if chain and opt is not None and cc_age_now > 30:
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

                        # ── Cross-instrument LOSS cooldown ──
                        # If a same-direction trade just hit SL, suppress
                        # new fires of that direction for 60 min — the read
                        # was likely wrong, don't double down.
                        cd_until = self.tracker._loss_cooldown.get(sig["direction"])
                        if cd_until and datetime.now(IST) < cd_until:
                            mins_left = int((cd_until - datetime.now(IST)).total_seconds() / 60) + 1
                            log.info(f"⏸ {name} {sig['direction']} skipped — "
                                     f"loss cooldown active ({mins_left}m remaining)")
                            self._log_gate(name, "LOSS_COOLDOWN", sig,
                                           {"mins_left": mins_left})
                            # NOTE: deliberately NOT setting _last_signal here.
                            # Burning the 15-min alert cooldown on a skip used to
                            # suppress a real signal that appeared right after the
                            # loss cooldown expired. This check is cheap (no API
                            # call), so re-evaluating every scan is fine.
                            continue

                        # ── Max open positions cap (env-tunable) ──────
                        # Default 2: one trending-index trade plus one
                        # counter-setup can coexist; a cap of 1 was starving
                        # the alert stream whenever any position sat open.
                        # Override with MAX_OPEN_POSITIONS env var (0 = off).
                        _max_open = int(os.environ.get("MAX_OPEN_POSITIONS", "2") or 0)
                        if _max_open > 0:
                            try:
                                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                                open_cnt_row = db_exec(
                                    "SELECT COUNT(*) as cnt FROM signals "
                                    "WHERE status='OPEN' AND date=?",
                                    (today_str,), fetchone=True)
                                open_cnt = int(dict(open_cnt_row).get("cnt", 0)) if open_cnt_row else 0
                                if open_cnt >= _max_open:
                                    log.info(f"⏸ {name} {sig['direction']} skipped — "
                                             f"max open positions cap ({open_cnt}/{_max_open})")
                                    self._log_gate(name, "MAX_OPEN_CAP", sig,
                                                   {"open": open_cnt, "cap": _max_open})
                                    # No _last_signal here — see loss-cooldown note.
                                    # The cap check is a cheap DB count; the moment a
                                    # position closes, the next scan can alert again.
                                    continue
                            except Exception as e:
                                log.warning(f"  max_open check failed: {e}")

                        # ── Layer B: validation + sizing + SL rule ──
                        ai_result = SignalValidation.analyze(name, sig, opt, regime=regime,
                                                               chain_analytics=chain_anal)
                        if ai_result is None:
                            self.metrics["ai_api_failures"] += 1

                        if ai_result and ai_result.get("verdict") in ("SKIP", "WAIT"):
                            v = ai_result.get("verdict")
                            if v == "SKIP": self.metrics["ai_skipped"] += 1
                            else: self.metrics["ai_waited"] += 1
                            log.info(f"🤖 AI {v} {name} {sig['direction']} — "
                                     f"{ai_result.get('reasoning','')[:80]}")
                            self._log_gate(name, f"AI_{v}", sig,
                                           {"reasoning": (ai_result.get("reasoning") or "")[:200]})
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
                                # Was `atm` (undefined) — the NameError was swallowed
                                # by this except, so the AI position-pct re-pick
                                # silently never ran since the day it shipped.
                                opt = self.opick.pick(sig, inst, chain, atm_val,
                                                      CONFIG.get("budget", 20000),
                                                      greeks=greeks,
                                                      position_pct=ai_result["position_pct"]) or opt
                                result["option"] = opt
                            except Exception as e:
                                log.warning(f"  {name}: AI position re-pick failed: {e}")

                        # Apply adj to confidence for display (don't re-gate)
                        adj = int((ai_result or {}).get("confidence_adj") or 0)
                        sig["confidence_ai_adj"] = max(0, min(100, sig["confidence"] + adj))

                        _now_ist = datetime.now(IST)
                        self.alerts.insert(0,{
                            "id": int(time.time()*1000),
                            "time": _now_ist.strftime("%H:%M:%S"),
                            "date": _now_ist.strftime("%Y-%m-%d"),
                            "iso_ts": _now_ist.strftime("%Y-%m-%d %H:%M:%S"),
                            "weekday": _now_ist.strftime("%a"),       # "Mon", "Tue", …
                            "instrument": name, "signal": sig, "option": opt,
                            "timing": timing, "ai": ai_result,
                        })
                        self.alerts=self.alerts[:100]
                        save_signal(name,sig,opt,ai=ai_result)
                        self.metrics["signals_alerted"] += 1
                        log.info(f"🚨 {name} {sig['direction']} Conf:{sig['confidence']}% "
                                 f"pos={(ai_result or {}).get('position_pct',100)}% "
                                 f"tighten={(ai_result or {}).get('sl_tightening','none')}")

                        # 📱 SLACK ALERT — rich 2-column blocks + plain-text fallback
                        SlackAlert.send(
                            SlackAlert.format_signal(name, sig, opt, timing, ai_result),
                            blocks=SlackAlert.format_signal_blocks(name, sig, opt, timing, ai_result),
                        )

                        # BUG FIX #7: Only start the 15-min cooldown AFTER an alert fires.
                        # Previously this was set unconditionally at the bottom of the loop,
                        # wasting a cooldown window on every sub-threshold scan.
                        self._last_signal[name] = datetime.now(IST)
                    else:
                        self._log_gate(name, "CONF_BELOW_FLOOR", sig,
                                       {"confidence": sig.get("confidence"), "floor": conf_floor})

                    self._prev[name]=result;self.latest[name]=result
                    # _last_signal is now ONLY set when an alert fires (TAKE) or when
                    # AI SKIPs (line above in the SKIP/WAIT branch). Sub-threshold signals
                    # (conf < conf_floor) no longer consume a cooldown window.

                time.sleep(CONFIG["scan_interval_sec"])
            except Exception as e:
                log.error(f"Loop err: {e}");time.sleep(5)
    
    def get_state(self):
        # Live-tracking feed: every OPEN signal with current premium, peak,
        # and live P&L. Drives the dashboard's "multiple compact cards"
        # showing each trade's status in real time.
        open_signals = []
        try:
            today = datetime.now(IST).strftime("%Y-%m-%d")
            opens = db_exec(
                "SELECT * FROM signals WHERE status='OPEN' AND date=? ORDER BY id DESC",
                (today,), fetch=True) or []
            for r in opens:
                r = dict(r)
                pk = self.tracker._peak.get(r["id"])
                cur = self.tracker._last_seen.get(r["id"]) if hasattr(self.tracker, "_last_seen") else None
                opt_entry = float(r.get("option_entry") or 0)
                opt_sl    = float(r.get("option_sl") or 0)
                opt_t1    = float(r.get("option_target1") or 0)
                opt_t2    = float(r.get("option_target2") or 0)
                lot_size  = int(r.get("option_lot_size") or 0)
                lots      = max(1, int(r.get("option_lots") or 1))
                qty       = lot_size * lots
                pnl_now = None; pnl_peak = None; pct_to_t1 = None
                if cur is not None and opt_entry > 0 and qty > 0:
                    pnl_now = round((cur - opt_entry) * qty, 0)
                if pk and opt_entry > 0 and qty > 0:
                    pnl_peak = round((pk[0] - opt_entry) * qty, 0)
                    if opt_t1 > opt_entry:
                        pct_to_t1 = round((pk[0] - opt_entry) / (opt_t1 - opt_entry) * 100, 0)
                # Build a clean dict for the client — no raw DB columns
                ts_raw = r.get("timestamp") or ""
                t_in = ts_raw.split(" ")[1][:5] if " " in ts_raw else ts_raw[:5]
                open_signals.append({
                    "id": r["id"],
                    "instrument": r.get("instrument"),
                    "direction":  r.get("direction"),
                    "option_symbol": r.get("option_symbol"),
                    "option_type":   r.get("option_type"),
                    "option_strike": r.get("option_strike"),
                    "entry":  opt_entry,
                    "sl":     opt_sl,
                    "t1":     opt_t1,
                    "t2":     opt_t2,
                    "lots":   lots,
                    "lot_size": lot_size,
                    "qty":    qty,
                    "t_in":   t_in,
                    "current_premium": cur,
                    "peak_premium":    pk[0] if pk else None,
                    "peak_time":       pk[1] if pk else None,
                    "pnl_now":  pnl_now,
                    "pnl_peak": pnl_peak,
                    "pct_to_t1": pct_to_t1,
                    "confidence": r.get("confidence"),
                })
        except Exception as _e:
            log.warning(f"  open-signals build failed: {_e}")
        _rm = os.getenv("INTRADAY_RESEARCH_MODE", "1").strip()
        _now = datetime.now(IST)
        _today = _now.strftime("%Y-%m-%d")
        _ms = market_status(_now)
        # Stamp each live signal with a stale flag (its bar date != today) so the
        # dashboard can show the date and badge prior-day cards instead of
        # silently presenting an old signal as if it were live.
        _sig_out = {}
        for _k, _v in (self.latest or {}).items():
            if isinstance(_v, dict):
                _v = dict(_v)
                _bd = _v.get("signal_date")
                _v["signal_date"] = _bd
                _v["stale"] = bool(_bd and _bd != _today)
            _sig_out[_k] = _v
        # Data freshness: True only when the feed has actually served today's
        # bars. False on a trading day = stale feed (holiday or outage).
        _data_fresh = (self._newest_bar_date == _today)
        return{"running":self.running,"signals":_sig_out,"alerts":self.alerts[:50],
            "open_signals": open_signals,
            "performance":get_perf(),
            "config":{"scan_interval":CONFIG["scan_interval_sec"],"target_min":CONFIG["target_points_min"],
                "target_max":CONFIG["target_points_max"],"min_confidence":CONFIG["min_confidence"],
                "strategy":CONFIG.get("strategy","v1"),
                "enabled_instruments":os.environ.get("ENABLED_INSTRUMENTS","")},
            "research_mode": (_rm != "0"),
            "dispatch": dict(getattr(SignalGen, "last_dispatch", {}) or {}),
            "signal_diag": dict(getattr(self, "_cond_diag", {}) or {}),
            "time":_now.strftime("%H:%M:%S"),
            "server_date": _today,
            "market": _ms,
            "market_open": _ms["open"],
            "newest_bar_date": self._newest_bar_date,
            "data_fresh": _data_fresh,
            "slack_enabled":CONFIG["slack_enabled"] and bool(CONFIG["slack_webhook"])}

# ═══════════════════════════════════════════════════════════════════
# SWING POSITION DB HELPERS
# ═══════════════════════════════════════════════════════════════════
def swing_pos_save(data):
    """Insert a new swing position row. Returns new row id."""
    q = """INSERT INTO swing_positions
           (instrument,instrument_type,direction,entry_date,entry_time,
            spot_entry,spot_sl,spot_target1,spot_target2,
            option_symbol,option_strike,option_type,option_expiry,option_token,option_dte,
            option_entry,option_sl,option_target1,lot_size,lots,capital,
            source,reasons,indicators)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
    ind = json.dumps(data.get("indicators") or {})
    reas = json.dumps(data.get("reasons") or [])
    params = (
        data["instrument"], data.get("instrument_type","STOCK"),
        data["direction"],
        data.get("entry_date", datetime.now(IST).strftime("%Y-%m-%d")),
        data.get("entry_time", datetime.now(IST).strftime("%H:%M")),
        data.get("spot_entry"), data.get("spot_sl"),
        data.get("spot_target1"), data.get("spot_target2"),
        data.get("option_symbol"), data.get("option_strike"),
        data.get("option_type"), data.get("option_expiry"),
        data.get("option_token"), data.get("option_dte"),
        data.get("option_entry"), data.get("option_sl"), data.get("option_target1"),
        data.get("lot_size"), data.get("lots",1), data.get("capital"),
        data.get("source","AUTO"), reas, ind,
    )
    # db_exec returns None for INSERTs — open a connection directly so the
    # promised row id is actually returned (mirrors save_signal).
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(q, params)
    row_id = c.lastrowid
    conn.commit(); conn.close()
    return row_id

def swing_pos_list(status=None):
    if status:
        rows = db_exec("SELECT * FROM swing_positions WHERE status=? ORDER BY id DESC", (status,), fetch=True)
    else:
        rows = db_exec("SELECT * FROM swing_positions ORDER BY id DESC LIMIT 200", fetch=True)
    return [dict(r) for r in rows] if rows else []

def swing_pos_update(pos_id, **kwargs):
    if not kwargs: return
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [pos_id]
    db_exec(f"UPDATE swing_positions SET {sets} WHERE id=?", vals)

def swing_pos_close(pos_id, exit_price, option_exit=None):
    pos = db_exec("SELECT * FROM swing_positions WHERE id=?", (pos_id,), fetchone=True)
    if not pos: return None
    pos = dict(pos)
    entry_date = pos.get("entry_date","")
    today = datetime.now(IST).strftime("%Y-%m-%d")
    hold_days = 0
    try:
        from datetime import date as _date
        d0 = _date.fromisoformat(entry_date)
        d1 = _date.fromisoformat(today)
        hold_days = (d1 - d0).days
    except: pass
    opt_entry = pos.get("option_entry") or 0
    lots      = pos.get("lots") or 1
    lot_size  = pos.get("lot_size") or 1
    direction = pos.get("direction","LONG")
    qty       = lots * lot_size
    if option_exit is not None and opt_entry > 0:
        pnl_per = (option_exit - opt_entry) if direction == "LONG" else (opt_entry - option_exit)
        pnl_rs  = round(pnl_per * qty, 0)
        pnl_pct = round(pnl_per / max(opt_entry, 0.01) * 100, 1) if direction == "LONG" else round(-pnl_per / max(opt_entry, 0.01) * 100, 1)
    elif pos.get("spot_entry") and exit_price:
        spot_entry = pos.get("spot_entry")
        pnl_per = (exit_price - spot_entry) if direction == "LONG" else (spot_entry - exit_price)
        pnl_rs  = round(pnl_per * qty, 0)
        pnl_pct = round(pnl_per / max(spot_entry, 0.01) * 100, 1)
    else:
        pnl_rs = 0; pnl_pct = 0
    result = "WIN" if pnl_rs >= 0 else "LOSS"
    swing_pos_update(pos_id, status="CLOSED", exit_date=today,
                     exit_price=exit_price, option_exit=option_exit,
                     pnl_rupees=pnl_rs, pnl_pct=pnl_pct,
                     result=result, hold_days=hold_days)
    return {"pnl_rupees": pnl_rs, "pnl_pct": pnl_pct, "result": result}

def swing_perf():
    rows = db_exec("SELECT * FROM swing_positions WHERE status='CLOSED'", fetch=True)
    if not rows: return {"total":0,"wins":0,"losses":0,"win_rate":0,"total_pnl":0,"avg_hold_days":0}
    rows = [dict(r) for r in rows]
    wins = [r for r in rows if r.get("result")=="WIN"]
    losses = [r for r in rows if r.get("result")=="LOSS"]
    pnls = [r.get("pnl_rupees") or 0 for r in rows]
    hold = [r.get("hold_days") or 0 for r in rows if r.get("hold_days")]
    return {
        "total": len(rows), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins)/len(rows)*100,1) if rows else 0,
        "total_pnl": round(sum(pnls),0),
        "avg_win": round(sum(r.get("pnl_rupees",0) or 0 for r in wins)/max(len(wins),1),0),
        "avg_loss": round(sum(r.get("pnl_rupees",0) or 0 for r in losses)/max(len(losses),1),0),
        "avg_hold_days": round(sum(hold)/max(len(hold),1),1),
    }


def restore_swing_backfill():
    """One-shot restore of paper swing positions from swing_backfill.json.

    Railway's filesystem is EPHEMERAL: signals.db is wiped on every deploy
    until a volume is mounted (set DB_PATH to the volume, e.g.
    DB_PATH=/data/signals.db). This backfill re-seeds paper positions
    (recovered from the Slack alert history) whenever the table has no
    AUTO_PAPER rows at boot, so a wipe doesn't erase live tracking.

    Guards:
      • No-op when any AUTO_PAPER row exists (open OR closed) — a live DB
        is never touched.
      • Rows older than SWING_RESTORE_MAX_AGE_DAYS (default 7) are skipped:
        restoring them as OPEN would make the tracker book bogus MAX_HOLD
        outcomes at whatever today's price happens to be.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swing_backfill.json")
    if not os.path.exists(path):
        return
    try:
        rows = json.load(open(path))
    except Exception as e:
        log.warning(f"📼 swing backfill unreadable: {e}")
        return
    try:
        existing = db_exec("SELECT COUNT(*) as c FROM swing_positions "
                           "WHERE source='AUTO_PAPER'", fetchone=True)
        if existing and int(dict(existing).get("c") or 0) > 0:
            return
    except Exception as e:
        log.warning(f"📼 swing backfill precheck failed: {e}")
        return
    try:
        max_age = int(os.environ.get("SWING_RESTORE_MAX_AGE_DAYS", "7"))
    except Exception:
        max_age = 7
    today = datetime.now(IST).date()
    n = skipped = 0
    for r in rows:
        try:
            d0 = datetime.strptime(r.get("entry_date", ""), "%Y-%m-%d").date()
            if (today - d0).days > max_age:
                skipped += 1
                continue
            swing_pos_save(r)
            n += 1
        except Exception as e:
            log.warning(f"📼 backfill row failed ({r.get('instrument')}): {e}")
    if n or skipped:
        log.info(f"📼 Swing backfill: restored {n} paper positions"
                 + (f", skipped {skipped} older than {max_age}d" if skipped else ""))

restore_swing_backfill()


# ═══════════════════════════════════════════════════════════════════
# SWING ANALYSIS  — daily-timeframe technical signal engine
# ═══════════════════════════════════════════════════════════════════
class SwingAnalysis:
    """
    Scores a daily candle series for swing trade setups.
    Returns a signal dict (direction, confidence, levels, reasons) or None.
    """

    @staticmethod
    def _ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def analyze(name, candles, info):
        """
        candles: list of dicts {ts, open, high, low, close, volume} — oldest first.
        Returns signal dict or None if no clear setup.
        """
        try:
            if len(candles) < 35: return None
            closes  = pd.Series([c["close"]  for c in candles], dtype=float)
            highs   = pd.Series([c["high"]   for c in candles], dtype=float)
            lows    = pd.Series([c["low"]    for c in candles], dtype=float)
            volumes = pd.Series([c["volume"] for c in candles], dtype=float)

            # ── Indicators ───────────────────────────────────────────
            ema9   = SwingAnalysis._ema(closes, 9)
            ema21  = SwingAnalysis._ema(closes, 21)
            ema50  = SwingAnalysis._ema(closes, 50)
            ema200 = SwingAnalysis._ema(closes, 200) if len(closes) >= 60 else ema50.copy()

            # RSI-14
            delta    = closes.diff()
            avg_g    = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            avg_l    = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
            rsi      = 100 - 100 / (1 + avg_g / (avg_l + 1e-9))

            # MACD (12/26/9)
            macd_line   = SwingAnalysis._ema(closes, 12) - SwingAnalysis._ema(closes, 26)
            signal_line = SwingAnalysis._ema(macd_line, 9)
            macd_hist   = macd_line - signal_line

            # ATR-14
            tr = pd.concat([
                highs - lows,
                (highs - closes.shift()).abs(),
                (lows  - closes.shift()).abs(),
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean()

            vol_avg20 = volumes.rolling(20).mean()

            # ── Current (last bar) values ─────────────────────────────
            c       = closes.iloc[-1]
            p_close = closes.iloc[-2]
            c9      = ema9.iloc[-1];  c21 = ema21.iloc[-1]
            c50     = ema50.iloc[-1]; c200 = ema200.iloc[-1]
            c_rsi   = rsi.iloc[-1]
            c_hist  = macd_hist.iloc[-1]; p_hist = macd_hist.iloc[-2]
            c_macd  = macd_line.iloc[-1]; c_sig = signal_line.iloc[-1]
            c_atr   = atr14.iloc[-1]
            c_vol   = volumes.iloc[-1]; c_vol_avg = vol_avg20.iloc[-1]
            high20  = highs.rolling(20).max().iloc[-2]   # excl today
            low20   = lows.rolling(20).min().iloc[-2]

            # ── Scoring ───────────────────────────────────────────────
            ls = 0; ss = 0; reasons = []   # long_score, short_score

            # 1. EMA trend stack (most important — 25 pts)
            if c9 > c21 > c50:
                ls += 25; reasons.append("EMA bullish stack 9>21>50")
            if c9 < c21 < c50:
                ss += 25; reasons.append("EMA bearish stack 9<21<50")
            if c > c200:
                ls += 10; reasons.append("Above 200 EMA — major uptrend")
            else:
                ss += 8;  reasons.append("Below 200 EMA — major downtrend")

            # 2. RSI momentum zone (20 pts)
            if 45 <= c_rsi <= 65:
                ls += 20; reasons.append(f"RSI {c_rsi:.0f} — momentum zone (not overbought)")
            elif c_rsi > 70:
                ls -= 10; reasons.append(f"RSI {c_rsi:.0f} — overbought risk")
            if 35 <= c_rsi <= 55:
                ss += 15; reasons.append(f"RSI {c_rsi:.0f} — bearish momentum zone")
            elif c_rsi < 30:
                ss -= 10; reasons.append(f"RSI {c_rsi:.0f} — oversold (fade risk)")

            # 3. MACD (20 pts for fresh cross, 10 for expanding)
            if c_hist > 0 and p_hist <= 0:
                ls += 20; reasons.append("MACD fresh bullish crossover")
            elif c_hist > 0 and c_hist > p_hist:
                ls += 10; reasons.append("MACD histogram expanding bullish")
            if c_hist < 0 and p_hist >= 0:
                ss += 20; reasons.append("MACD fresh bearish crossover")
            elif c_hist < 0 and c_hist < p_hist:
                ss += 10; reasons.append("MACD histogram expanding bearish")

            # 4. Breakout / breakdown (20 pts)
            if c > high20:
                ls += 20; reasons.append(f"20-day breakout above {high20:.0f}")
            elif c > c21 and 40 <= c_rsi <= 55:
                ls += 10; reasons.append("Pullback to 21 EMA — bounce entry")
            if c < low20:
                ss += 20; reasons.append(f"20-day breakdown below {low20:.0f}")

            # 5. Volume confirmation (15 pts)
            vol_ratio = c_vol / max(c_vol_avg, 1)
            if vol_ratio >= 1.5:
                boost = 15 if ls > ss else 15
                if ls > ss: ls += boost
                else:       ss += boost
                reasons.append(f"Volume surge {vol_ratio:.1f}× avg — conviction move")
            elif vol_ratio >= 1.2:
                if ls > ss: ls += 8
                else:       ss += 8
                reasons.append(f"Above-avg volume {vol_ratio:.1f}×")

            # ── Determine direction ───────────────────────────────────
            gap = 15  # min margin between long/short score
            if ls >= 45 and ls >= ss + gap:
                direction = "LONG"; confidence = min(95, ls)
            elif ss >= 45 and ss >= ls + gap:
                direction = "SHORT"; confidence = min(95, ss)
            else:
                return None   # no clear setup

            # ── Risk levels (2 × ATR SL, 3-5 × ATR targets) ──────────
            sl_mult = 2.0; t1_mult = 3.0; t2_mult = 5.0
            if direction == "LONG":
                entry = c
                sl     = round(c - sl_mult * c_atr, 2)
                t1     = round(c + t1_mult * c_atr, 2)
                t2     = round(c + t2_mult * c_atr, 2)
            else:
                entry = c
                sl     = round(c + sl_mult * c_atr, 2)
                t1     = round(c - t1_mult * c_atr, 2)
                t2     = round(c - t2_mult * c_atr, 2)

            rr = round(abs(t1 - entry) / max(abs(sl - entry), 0.01), 2)

            return {
                "direction": direction, "confidence": confidence,
                "price": round(c, 2), "entry": round(entry, 2),
                "sl": sl, "target1": t1, "target2": t2,
                "risk_reward": rr, "atr": round(c_atr, 2),
                "rsi": round(c_rsi, 1),
                "ema9": round(c9, 2), "ema21": round(c21, 2),
                "ema50": round(c50, 2), "ema200": round(c200, 2),
                "macd_hist": round(c_hist, 4), "macd_line": round(c_macd, 4),
                "vol_ratio": round(vol_ratio, 2),
                "reasons": reasons,
                "timeframe": "DAILY", "hold_days_est": "3-15 days",
            }

        except Exception as e:
            log.warning(f"[SwingAnalysis] {name} error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════
# SWING PULLBACK v1 — momentum-filtered oversold pullback (LONG-ONLY)
#
# Built 2026-07-29 from a 4-track deep-research pass (~90 primary
# sources). Full spec + evidence in SWING_STRATEGY.md. The one-line
# thesis: momentum picks the stock, fear times the entry.
#
#   UNIVERSE  stock beats NIFTY over 126 days AND closes within 15% of
#             its 252-day high (Indian momentum + 52wk-high effect —
#             both academically validated on NSE).
#   TRIGGER   uptrend intact (close > 200SMA and > 50SMA) AND a sharp
#             2-5 day pullback: RSI(2) < 10, OR close at a 7-day
#             closing low, OR 3 consecutive lower closes
#             (Connors/Alvarez dip-buy family: 65-83% win rates,
#             2-5 day holds, stops HURT this entry type).
#   LONG-ONLY the short leg of momentum is where crashes live and
#             India's market structure starves the short side; PE
#             signals remain only in the legacy analyzer.
#   EXITS     handled by the engine: first-strength exit (close > SMA5
#             or RSI(2) > 65), stagnation stop, 10-session max hold,
#             -50% premium backstop; the bought premium itself is the
#             hard floor (why this entry suits option buying).
# ═══════════════════════════════════════════════════════════════════
class SwingPullback:
    """Stateless analyze() per instrument. Returns a candidate dict or
    None. Candidates are RANKED by the engine and only the top few are
    alerted — selectivity is part of the documented edge."""

    MIN_BARS = 220   # needs a real 200SMA + most of a 252d high window

    @staticmethod
    def analyze(name, candles, info, ctx):
        """ctx: market context dict from SwingEngine._swing_market_ctx()
        with nifty_ret126 (float) at minimum."""
        try:
            if len(candles) < SwingPullback.MIN_BARS:
                return None
            closes = pd.Series([c["close"] for c in candles], dtype=float)
            highs  = pd.Series([c["high"]  for c in candles], dtype=float)
            lows   = pd.Series([c["low"]   for c in candles], dtype=float)

            c = float(closes.iloc[-1])
            sma200 = float(closes.rolling(200).mean().iloc[-1])
            sma100 = float(closes.rolling(100).mean().iloc[-1])
            sma5   = float(closes.rolling(5).mean().iloc[-1])
            # Trend gate uses the 100SMA (Alvarez's tested variant), NOT the
            # 50SMA — a genuine 2-5 day dump often pierces the 50 while the
            # uptrend stays perfectly intact; gating on it starves the entry.
            if not (c > sma200 and c > sma100):
                return None   # pullback must live inside an intact uptrend

            # ── Momentum universe filters ─────────────────────────────
            ret126 = c / float(closes.iloc[-127]) - 1.0
            nifty_ret126 = float(ctx.get("nifty_ret126") or 0.0)
            rs_excess = ret126 - nifty_ret126
            if rs_excess <= 0:
                return None   # must beat the index over ~6 months
            hi252 = float(highs.iloc[-252:].max())
            prox = c / hi252 if hi252 > 0 else 0
            min_prox = float(os.environ.get("PULLBACK_MIN_52W_PROX", "0.85"))
            if prox < min_prox:
                return None   # too far below the 52-week high

            # ── Pullback trigger (any one) ────────────────────────────
            rsi2 = float(TA.rsi(closes, 2).iloc[-1])
            seven_low = c <= float(closes.iloc[-8:-1].min())
            three_down = all(closes.iloc[-i] < closes.iloc[-i - 1] for i in (1, 2, 3))
            rsi2_max = float(os.environ.get("PULLBACK_RSI2_MAX", "10"))
            triggers = []
            if rsi2 < rsi2_max: triggers.append(f"RSI(2) {rsi2:.0f} < {rsi2_max:.0f}")
            if seven_low:       triggers.append("7-day closing low")
            if three_down:      triggers.append("3 consecutive lower closes")
            if not triggers:
                return None

            # ── Levels ────────────────────────────────────────────────
            # The SL here is a DISASTER floor (2x ATR), not a trading
            # stop — Connors/Alvarez found stops hurt this entry; the
            # real exits are strength / stagnation / time / premium.
            tr = pd.concat([highs - lows, (highs - closes.shift()).abs(),
                             (lows - closes.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            rsi14 = float(TA.rsi(closes, 14).iloc[-1])
            sl = round(c - 2.0 * atr, 2)
            t1 = round(c + 3.0 * atr, 2)
            t2 = round(c + 5.0 * atr, 2)

            # Rank score: how much it beats NIFTY + how close to highs.
            score = round(rs_excess * 100 + (prox - min_prox) * 40, 2)
            confidence = int(max(65, min(90, 70 + score / 2)))

            return {
                "direction": "LONG", "confidence": confidence,
                "price": round(c, 2), "entry": round(c, 2),
                "sl": sl, "target1": t1, "target2": t2,
                "risk_reward": round((t1 - c) / max(c - sl, 0.01), 2),
                "atr": round(atr, 2), "rsi": round(rsi14, 1),
                "rsi2": round(rsi2, 1),
                "score": score,
                "reasons": [
                    f"Momentum: +{ret126*100:.0f}% / 126d (NIFTY {nifty_ret126*100:+.0f}%) — beats index by {rs_excess*100:.0f}pts",
                    f"Within {100-prox*100:.0f}% of 52-week high {hi252:.0f}",
                    "Pullback trigger: " + " + ".join(triggers),
                    f"Uptrend intact: close > 100SMA {sma100:.0f} and > 200SMA {sma200:.0f}",
                    "Exits: first close > 5SMA / RSI(2)>65 · stagnation d5 · max 10 sessions · -50% premium floor",
                ],
                "sma5": round(sma5, 2),
                "timeframe": "DAILY", "hold_days_est": "2-10 days",
                "strategy": "pullback_v1",
            }
        except Exception as e:
            log.warning(f"[Pullback] {name} error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════
# SWING ENGINE — multi-day scanner + AI exit analysis
# ═══════════════════════════════════════════════════════════════════
class SwingEngine:
    """
    Scans SWING_STOCKS on daily candles every SCAN_MIN minutes.
    Picks swing-expiry options (monthly, ≥15 DTE).
    Tracks open positions in SQLite + runs Claude AI exit analysis.
    """
    SCAN_MIN = 30          # re-scan every 30 min during market hours
    MIN_CONF = 55          # minimum confidence to surface signal
    ALERT_CONF = 65        # minimum confidence to send Slack alert
    EXIT_RECHECK_MIN = 120 # AI exit recheck interval (2 h)

    def __init__(self, client):
        self.client  = client
        self.signals = {}    # name → latest signal snapshot
        self.running = False
        self._last_scan   = {}    # name → timestamp of last scan
        self._last_exit_check = datetime.now(IST)

    def start(self):
        if self.running: return
        self.running = True
        threading.Thread(target=self._loop, daemon=True, name="SwingLoop").start()
        log.info("[Swing] Engine started — scanning on 30-min cycle")

    def _loop(self):
        # Stagger swing start by 150s — intraday scanner needs 2-3 candle fetches
        # to warm its cache before the swing scan fires 20+ daily candle calls.
        time.sleep(150)
        while self.running:
            try:
                now = datetime.now(IST)
                if 9 <= now.hour < 15 or (now.hour == 15 and now.minute < 30):
                    self._scan_all()
                    # Paper-position outcome tracking (SL/T1/T2/max-hold) —
                    # also refreshes live P&L for the app's Swing tab.
                    try:
                        self._track_paper_outcomes()
                    except Exception as e:
                        log.warning(f"[Swing] paper outcome pass failed: {e}")
                    # Periodic AI exit check on open positions
                    mins_since = (now - self._last_exit_check).total_seconds() / 60
                    if mins_since >= self.EXIT_RECHECK_MIN:
                        self._ai_exit_all_open()
                        self._last_exit_check = now
                time.sleep(self.SCAN_MIN * 60)
            except Exception as e:
                log.error(f"[Swing] Loop error: {e}")
                time.sleep(60)

    def _scan_all(self):
        strategy = os.environ.get("SWING_STRATEGY", "pullback").lower()
        if strategy != "pullback":
            # Legacy multi-signal scorer — alerts every qualifying setup.
            log.info(f"[Swing] Scanning {len(SWING_STOCKS)} instruments (legacy)...")
            for name, info in SWING_STOCKS.items():
                try:
                    self._scan_instrument(name, info)
                    time.sleep(0.5)   # gentle throttle
                except Exception as e:
                    log.warning(f"[Swing] {name} error: {e}")
            return

        # ── Pullback v1: gate → collect → rank → admit top slots ──────
        ctx = self._swing_market_ctx()
        log.info(f"[Swing] Pullback scan: regime={'OK' if ctx['long_ok'] else 'BLOCKED'} "
                 f"({ctx['note']}) · {len(SWING_STOCKS)} instruments")
        candidates = []
        for name, info in SWING_STOCKS.items():
            try:
                cand = self._scan_pullback(name, info, ctx)
                if cand:
                    candidates.append(cand)
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"[Swing] {name} error: {e}")
        try:
            self._admit_pullback_candidates(candidates, ctx)
        except Exception as e:
            log.warning(f"[Swing] candidate admission failed: {e}")

    # ── Pullback v1 machinery ─────────────────────────────────────────

    def _swing_market_ctx(self):
        """Market regime context, cached ~30 min. Gates NEW entries only:
        NIFTY > 200SMA with a rising 50SMA (Faber-class trend filter),
        India VIX band, and the macro event blackout. Exits always run."""
        now_ts = time.time()
        cached = getattr(self, "_mctx", None)
        if cached and now_ts - cached.get("_ts", 0) < 1800:
            return cached
        ctx = {"_ts": now_ts, "long_ok": False, "nifty_ret126": 0.0,
               "vix": None, "vix_band": "full", "note": ""}
        notes = []
        try:
            candles = self.client.daily_candles("99926000", "NSE", days=400)
            if len(candles) >= 220:
                closes = pd.Series([c["close"] for c in candles], dtype=float)
                c = float(closes.iloc[-1])
                sma200 = float(closes.rolling(200).mean().iloc[-1])
                sma50s = closes.rolling(50).mean()
                slope_up = float(sma50s.iloc[-1]) > float(sma50s.iloc[-21])
                ctx["nifty_ret126"] = c / float(closes.iloc[-127]) - 1.0
                above200 = c > sma200
                notes.append(f"NIFTY {'>' if above200 else '<'}200SMA, 50SMA {'rising' if slope_up else 'falling'}")
                trend_ok = above200 and slope_up
            else:
                trend_ok = False
                notes.append("NIFTY history short — trend gate failed closed")
        except Exception as e:
            trend_ok = False
            notes.append(f"NIFTY fetch failed ({e}) — gate closed")
        try:
            vd = self.client.ltp("NSE", "India VIX", "26017")
            vix = float((vd or {}).get("ltp") or 0)
            if vix > 0:
                ctx["vix"] = round(vix, 2)
                vix_full = float(os.environ.get("PULLBACK_VIX_FULL", "20"))
                vix_max = float(os.environ.get("PULLBACK_VIX_MAX", "25"))
                ctx["vix_band"] = "full" if vix <= vix_full else ("half" if vix <= vix_max else "block")
                notes.append(f"VIX {vix:.1f} ({ctx['vix_band']})")
        except Exception:
            notes.append("VIX unavailable (fails open)")
        blackout = False
        try:
            blackout, ev = EventCalendar.in_blackout()
            if blackout:
                notes.append(f"event blackout: {(ev or {}).get('name', '?')}")
        except Exception:
            pass
        ctx["long_ok"] = trend_ok and ctx["vix_band"] != "block" and not blackout
        ctx["note"] = " · ".join(notes)
        self._mctx = ctx
        return ctx

    def _scan_pullback(self, name, info, ctx):
        """One instrument: run strength-exits on its open pullback rows
        (always), then return a ranked entry candidate (only when the
        regime allows new entries)."""
        token, exch, sym = self._resolve_token(name, info)
        if not token:
            return None
        candles = self.client.daily_candles(token, exch, days=400)
        if len(candles) < 35:
            return None

        # Strength exit on open positions for this name (regime-independent)
        try:
            self._pullback_strength_exit(name, candles)
        except Exception as e:
            log.warning(f"[Swing] strength-exit check {name}: {e}")

        if not ctx["long_ok"]:
            return None
        sig = SwingPullback.analyze(name, candles, info, ctx)
        if not sig:
            return None
        ltp_data = self.client.ltp(exch, sym, token)
        spot = (ltp_data or {}).get("ltp") or sig["price"]
        sig["price"] = round(float(spot), 2)
        self.signals[name] = {
            "instrument": name, "type": info.get("type", "STOCK"),
            "fo_name": info.get("nse_fo", ""), "signal": sig, "option": None,
            "updated_at": datetime.now(IST).strftime("%H:%M:%S"),
            "updated_date": datetime.now(IST).strftime("%Y-%m-%d"),
        }
        return {"name": name, "info": info, "sig": sig, "spot": float(spot)}

    def _pullback_strength_exit(self, name, candles):
        """Exit an open pullback position on the FIRST sign of strength —
        close > 5SMA or RSI(2) > 65 (the Connors/Alvarez exit). This is
        the profit-taking half of the mean-reversion trade."""
        opens = [p for p in swing_pos_list(status="OPEN")
                 if p.get("instrument") == name and (p.get("source") or "") == "AUTO_PAPER"]
        if not opens:
            return
        closes = pd.Series([c["close"] for c in candles], dtype=float)
        if len(closes) < 10:
            return
        c = float(closes.iloc[-1])
        sma5 = float(closes.rolling(5).mean().iloc[-1])
        rsi2 = float(TA.rsi(closes, 2).iloc[-1])
        if not (c > sma5 or rsi2 > 65):
            return
        for pos in opens:
            try:
                if (json.loads(pos.get("indicators") or "{}") or {}).get("strategy") != "pullback_v1":
                    continue   # legacy rows keep their SL/T1/T2 exits
            except Exception:
                continue
            ltp = self._paper_live_quote(pos) or c
            self._finalize_paper_close(
                pos, float(ltp), "STRENGTH_EXIT",
                note=f"close {c:.1f} > 5SMA {sma5:.1f}" if c > sma5 else f"RSI(2) {rsi2:.0f} > 65")

    def _admit_pullback_candidates(self, candidates, ctx):
        """Rank candidates by score and admit only into free slots —
        selectivity is part of the edge (Zarattini; Turtle caps). Never
        queue stale signals: whatever misses a slot today is discarded."""
        if not candidates:
            return
        max_open = int(os.environ.get("SWING_MAX_OPEN", "4"))
        account = float(os.environ.get("SWING_ACCOUNT_CAPITAL", "150000"))
        open_rows = [p for p in swing_pos_list(status="OPEN")
                     if (p.get("source") or "") == "AUTO_PAPER"]
        open_names = {(p["instrument"], p.get("direction")) for p in open_rows}
        slots = max_open - len(open_rows)
        candidates.sort(key=lambda x: x["sig"].get("score", 0), reverse=True)
        log.info(f"[Swing] {len(candidates)} pullback candidates, {max(0, slots)} free slots "
                 f"(top: {', '.join(c['name'] + ' ' + str(c['sig'].get('score')) for c in candidates[:5])})")
        if slots <= 0:
            return
        admitted = 0
        for cand in candidates:
            if admitted >= slots:
                break
            name, info, sig, spot = cand["name"], cand["info"], cand["sig"], cand["spot"]
            if (name, "LONG") in open_names:
                continue
            opt = None
            if info.get("fo_eligible"):
                opt = self._pick_option(name, info, sig, spot, style="itm")
            # Affordability gate: 1-lot premium must fit the account.
            cap_frac = float(os.environ.get("PULLBACK_MAX_PREMIUM_FRAC", "0.10"))
            if opt and account > 0 and (opt.get("capital") or 0) > account * cap_frac:
                log.info(f"[Swing] {name} skipped — 1 lot ≈ ₹{opt.get('capital'):,.0f} "
                         f"> {cap_frac*100:.0f}% of ₹{account:,.0f} account")
                continue
            self.signals.setdefault(name, {})["option"] = opt
            half = ctx.get("vix_band") == "half"
            if os.environ.get("SWING_SLACK_ENABLED", "true").lower() == "true":
                _key = f"{name}:LONG:{datetime.now(IST).strftime('%Y-%m-%d')}"
                if not hasattr(self, "_slack_sent"): self._slack_sent = set()
                if _key not in self._slack_sent:
                    self._slack_sent.add(_key)
                    msg = self._format_slack(name, sig, opt)
                    msg = f"🎯 *Pullback v1 · rank #{admitted+1} · score {sig.get('score')}*\n" + msg
                    if half:
                        msg += "\n⚠️ VIX 20-25: elevated-vol regime — half-size territory"
                    SlackAlert.send(msg)
            try:
                sig_for_open = dict(sig)
                self._open_paper_position(name, info, sig_for_open, opt)
            except Exception as e:
                log.warning(f"[Swing] paper-open failed {name}: {e}")
            admitted += 1

    def _resolve_token(self, name, info):
        """Return (equity_token, ltp_exchange, ltp_symbol) for candle + LTP calls."""
        # INDEX types have hardcoded tokens
        if info.get("type") == "INDEX":
            tok = info.get("token")
            return tok, info["exchange"], info["nse_sym"]
        # STOCK — look up from instrument master
        sym = info.get("nse_sym", name)
        tok = _master.find_equity_token(sym)
        if not tok:
            log.debug(f"[Swing] No equity token for {name} ({sym})")
        return tok, info["exchange"], sym

    def _scan_instrument(self, name, info):
        token, exch, sym = self._resolve_token(name, info)
        if not token: return

        candles = self.client.daily_candles(token, exch, days=90)
        if len(candles) < 35:
            log.debug(f"[Swing] {name}: only {len(candles)} daily candles — skipping")
            return

        sig = SwingAnalysis.analyze(name, candles, info)
        if not sig: return

        # Enrich with live LTP
        ltp_data = self.client.ltp(exch, sym, token)
        spot = (ltp_data or {}).get("ltp") or sig["price"]
        sig["price"] = round(float(spot), 2)

        # Option picking for F&O eligible instruments (≥ MIN_CONF)
        opt = None
        if info.get("fo_eligible") and sig["confidence"] >= self.MIN_CONF:
            opt = self._pick_option(name, info, sig, spot)

        snapshot = {
            "instrument": name, "type": info.get("type","STOCK"),
            "fo_name": info.get("nse_fo",""), "signal": sig, "option": opt,
            "updated_at": datetime.now(IST).strftime("%H:%M:%S"),
            "updated_date": datetime.now(IST).strftime("%Y-%m-%d"),
        }
        self.signals[name] = snapshot

        if sig["confidence"] >= self.ALERT_CONF:
            log.info(f"[Swing] 🎯 {name} {sig['direction']} conf={sig['confidence']}% "
                     f"price={sig['price']} rr={sig['risk_reward']}")
            # Swing Slack alerts re-enabled (2026-07-28): the engine was
            # producing signals with no delivery channel at all. Mute again
            # with SWING_SLACK_ENABLED=false if the pings get noisy.
            if os.environ.get("SWING_SLACK_ENABLED", "true").lower() == "true":
                # Dedupe: one Slack ping per instrument+direction per day —
                # the 30-min swing cycle re-detects the same setup all day.
                _key = f"{name}:{sig['direction']}:{datetime.now(IST).strftime('%Y-%m-%d')}"
                if not hasattr(self, "_slack_sent"): self._slack_sent = set()
                if _key not in self._slack_sent:
                    self._slack_sent.add(_key)
                    SlackAlert.send(self._format_slack(name, sig, opt))
            # ── Paper-track every alerted swing signal ─────────────────
            # A signal that is never recorded can never be judged. Each
            # alert opens ONE paper position (deduped against an existing
            # OPEN one for the same instrument+direction); the outcome
            # tracker closes it on SL / T1 / T2 / max-hold and reports
            # what actually happened.
            try:
                self._open_paper_position(name, info, sig, opt)
            except Exception as e:
                log.warning(f"[Swing] paper-open failed {name}: {e}")

    # ── Paper tracking of alerted signals ────────────────────────────
    # Every Slack-alerted swing signal becomes ONE paper position so the
    # outcome is verifiable: after N days you can see whether it actually
    # made money instead of wondering. Closed on SL / T1 / T2 / max-hold
    # by _track_paper_outcomes(), with a Slack outcome ping.

    @staticmethod
    def _paper_max_hold_days():
        try:
            return max(1, int(os.environ.get("SWING_PAPER_MAX_HOLD_DAYS", "5")))
        except Exception:
            return 5

    def _open_paper_position(self, name, info, sig, opt):
        """Insert an AUTO_PAPER swing position for an alerted signal, unless
        an OPEN one for the same instrument+direction already exists."""
        existing = db_exec(
            "SELECT id FROM swing_positions WHERE status='OPEN' AND instrument=? "
            "AND direction=? AND source='AUTO_PAPER' LIMIT 1",
            (name, sig["direction"]), fetchone=True)
        if existing:
            return None
        pos = {
            "instrument": name, "instrument_type": info.get("type", "STOCK"),
            "direction": sig["direction"],
            "spot_entry": sig.get("price"), "spot_sl": sig.get("sl"),
            "spot_target1": sig.get("target1"), "spot_target2": sig.get("target2"),
            "source": "AUTO_PAPER",
            "reasons": sig.get("reasons", []),
            "indicators": {
                "confidence": sig.get("confidence"),
                "rsi": sig.get("rsi"),
                "risk_reward": sig.get("risk_reward"),
                "hold_days_est": sig.get("hold_days_est"),
                "strategy": sig.get("strategy"),
                "score": sig.get("score"),
                # Pullback winners need room to run (right-tail evidence);
                # legacy rows keep the shorter default.
                "max_hold_days": (int(os.environ.get("PULLBACK_MAX_HOLD_DAYS", "10"))
                                   if sig.get("strategy") == "pullback_v1"
                                   else SwingEngine._paper_max_hold_days()),
            },
        }
        if opt:
            pos.update({
                "option_symbol": opt.get("symbol"), "option_strike": opt.get("strike"),
                "option_type": opt.get("type"), "option_expiry": opt.get("expiry"),
                "option_token": opt.get("token"), "option_dte": opt.get("dte"),
                "option_entry": opt.get("entry"), "option_sl": opt.get("sl"),
                "option_target1": opt.get("target1"),
                "lot_size": opt.get("lot_size"), "lots": 1,
                "capital": opt.get("capital"),
            })
        row_id = swing_pos_save(pos)
        log.info(f"[Swing] 📓 paper position #{row_id} opened: {name} {sig['direction']} "
                 f"@ {sig.get('price')} (SL {sig.get('sl')} / T1 {sig.get('target1')}, "
                 f"max hold {SwingEngine._paper_max_hold_days()}d)")
        return row_id

    def _track_paper_outcomes(self):
        """Close AUTO_PAPER positions on SL / T1 / T2 / max-hold and report
        the outcome to Slack. Runs every swing cycle (~30 min)."""
        opens = [p for p in swing_pos_list(status="OPEN")
                 if (p.get("source") or "") == "AUTO_PAPER"]
        for pos in opens:
            try:
                self._track_one_paper(pos)
                time.sleep(0.3)
            except Exception as e:
                log.warning(f"[Swing] paper-track err #{pos.get('id')}: {e}")

    def _paper_live_quote(self, pos):
        """Fetch spot LTP for a position's instrument. Returns float or None.
        Also caches into self._paper_live for the /api/swing/results feed."""
        name = pos["instrument"]
        info = SWING_STOCKS.get(name)
        if not info:
            return None
        token, exch, sym = self._resolve_token(name, info)
        if not token:
            return None
        ltp_data = self.client.ltp(exch, sym, token)
        ltp = (ltp_data or {}).get("ltp")
        if not ltp:
            return None
        ltp = float(ltp)
        if not hasattr(self, "_paper_live"):
            self._paper_live = {}
        favorable = (ltp - (pos.get("spot_entry") or ltp)) if pos.get("direction") == "LONG" \
            else ((pos.get("spot_entry") or ltp) - ltp)
        self._paper_live[pos["id"]] = {
            "ltp": ltp,
            "pnl_pct": round(favorable / max(float(pos.get("spot_entry") or 1), 0.01) * 100, 2),
            "est_pnl_rs": self._paper_est_option_pnl(pos, favorable),
            "ts": datetime.now(IST).strftime("%H:%M:%S"),
            "epoch": time.time(),
        }
        return ltp

    @staticmethod
    def _paper_est_option_pnl(pos, favorable_spot_move):
        """Estimated rupee P&L of the BOUGHT option (CE for LONG / PE for
        SHORT) at delta ≈ 0.5. Estimate only — flagged as such everywhere
        it's shown. None when the position has no option leg."""
        opt_entry = float(pos.get("option_entry") or 0)
        if opt_entry <= 0:
            return None
        lot_size = int(pos.get("lot_size") or 0) or 1
        lots = int(pos.get("lots") or 1)
        est_exit = max(0.05 * opt_entry, opt_entry + favorable_spot_move * 0.5)
        return round((est_exit - opt_entry) * lot_size * lots, 0)

    def _track_one_paper(self, pos):
        direction = pos.get("direction", "LONG")
        entry = float(pos.get("spot_entry") or 0)
        sl = float(pos.get("spot_sl") or 0)
        t1 = float(pos.get("spot_target1") or 0)
        t2 = float(pos.get("spot_target2") or 0)
        if entry <= 0:
            return
        ltp = self._paper_live_quote(pos)
        if ltp is None:
            return

        # Days held (calendar) + max-hold from the row's own snapshot
        try:
            d0 = datetime.strptime(pos.get("entry_date", ""), "%Y-%m-%d").date()
            held = (datetime.now(IST).date() - d0).days
        except Exception:
            held = 0
        try:
            max_hold = int((json.loads(pos.get("indicators") or "{}") or {})
                           .get("max_hold_days") or SwingEngine._paper_max_hold_days())
        except Exception:
            max_hold = SwingEngine._paper_max_hold_days()

        try:
            ind = json.loads(pos.get("indicators") or "{}") or {}
        except Exception:
            ind = {}
        is_pullback = ind.get("strategy") == "pullback_v1"
        opt_entry = float(pos.get("option_entry") or 0)
        favorable = (ltp - entry) if direction == "LONG" else (entry - ltp)

        exit_reason = None
        if direction == "LONG":
            if sl > 0 and ltp <= sl:      exit_reason = "SL_HIT"
            elif t2 > 0 and ltp >= t2:    exit_reason = "T2_HIT"
            elif t1 > 0 and ltp >= t1:    exit_reason = "T1_HIT"
        else:
            if sl > 0 and ltp >= sl:      exit_reason = "SL_HIT"
            elif t2 > 0 and ltp <= t2:    exit_reason = "T2_HIT"
            elif t1 > 0 and ltp <= t1:    exit_reason = "T1_HIT"
        # Premium backstop: est option premium down ~50% from entry
        # (delta-0.5 model → spot has moved a full premium against us).
        if exit_reason is None and opt_entry > 0 and favorable * 0.5 <= -0.5 * opt_entry:
            exit_reason = "PREMIUM_BACKSTOP"
        # Stagnation stop (pullback only): an option that hasn't earned
        # ~+25% premium after 5 sessions is just feeding theta.
        if exit_reason is None and is_pullback:
            stag_days = int(os.environ.get("PULLBACK_STAG_DAYS", "5"))
            stag_gain = float(os.environ.get("PULLBACK_STAG_MIN_GAIN_PCT", "25"))
            if held >= stag_days:
                gain_pct = (favorable * 0.5 / opt_entry * 100) if opt_entry > 0 \
                    else (favorable / entry * 100 / 0.03)   # spot proxy
                if gain_pct < stag_gain:
                    exit_reason = "STAGNATION"
        if exit_reason is None and held >= max_hold:
            exit_reason = "MAX_HOLD"
        if exit_reason is None:
            return
        self._finalize_paper_close(pos, ltp, exit_reason, held=held, max_hold=max_hold)

    def _finalize_paper_close(self, pos, ltp, exit_reason, note=None,
                              held=None, max_hold=None):
        """Close a paper position at `ltp` with `exit_reason`, persist the
        outcome, and Slack the result. Shared by the price/time tracker and
        the pullback strength-exit path."""
        direction = pos.get("direction", "LONG")
        entry = float(pos.get("spot_entry") or 0)
        if entry <= 0:
            return
        if held is None:
            try:
                d0 = datetime.strptime(pos.get("entry_date", ""), "%Y-%m-%d").date()
                held = (datetime.now(IST).date() - d0).days
            except Exception:
                held = 0
        favorable = (ltp - entry) if direction == "LONG" else (entry - ltp)
        pnl_pct = round(favorable / entry * 100, 2)
        est_rs = self._paper_est_option_pnl(pos, favorable)
        result = "WIN" if favorable > 0 else "LOSS"
        opt_entry = float(pos.get("option_entry") or 0)
        est_opt_exit = (max(0.05 * opt_entry, opt_entry + favorable * 0.5)
                        if opt_entry > 0 else None)
        swing_pos_update(pos["id"], status="CLOSED",
                         exit_date=datetime.now(IST).strftime("%Y-%m-%d"),
                         exit_price=ltp,
                         option_exit=(round(est_opt_exit, 2) if est_opt_exit else None),
                         pnl_pct=pnl_pct,
                         pnl_rupees=(est_rs if est_rs is not None else round(favorable, 0)),
                         result=result, hold_days=held, exit_reason=exit_reason)
        if hasattr(self, "_paper_live"):
            self._paper_live.pop(pos["id"], None)

        story = {
            "SL_HIT": "stop-loss hit",
            "T1_HIT": "target 1 hit",
            "T2_HIT": "target 2 hit",
            "MAX_HOLD": f"max hold ({max_hold or '?'}d) reached",
            "STRENGTH_EXIT": "strength exit (mean-reversion target)",
            "STAGNATION": "stagnation stop (no progress, theta bleeding)",
            "PREMIUM_BACKSTOP": "premium backstop (-50% est)",
        }.get(exit_reason, exit_reason)
        if note:
            story += f" — {note}"
        emoji = "🎯" if exit_reason in ("T2_HIT", "STRENGTH_EXIT") and result == "WIN" \
            else ("✅" if result == "WIN" else "❌")
        log.info(f"[Swing] {emoji} paper #{pos['id']} {pos['instrument']} {direction} "
                 f"CLOSED ({story}) after {held}d: {pnl_pct:+.1f}% spot"
                 + (f", est ₹{est_rs:+,.0f}" if est_rs is not None else ""))
        if os.environ.get("SWING_SLACK_ENABLED", "true").lower() == "true":
            opt_line = ""
            if opt_entry > 0 and est_opt_exit:
                opt_line = (f"\n{pos.get('option_symbol')} est ₹{opt_entry:.1f} → "
                            f"₹{est_opt_exit:.1f}  ·  est P&L ₹{est_rs:+,.0f} "
                            f"_(delta-0.5 model, not a real fill)_")
            SlackAlert.send(
                f"{emoji} *Swing outcome: {pos['instrument']} {direction}* — "
                f"{result} ({story})\n"
                f"Held {held} day{'s' if held != 1 else ''}: "
                f"₹{entry:,.1f} → ₹{ltp:,.1f} ({pnl_pct:+.1f}% spot)"
                f"{opt_line}")

    def _pick_option(self, name, info, sig, spot, style=None):
        """Select swing option (monthly expiry, ≥15 DTE).

        style=None  → ATM (legacy behavior).
        style='itm' → one strike in-the-money (~0.60-0.70 delta). For
        multi-day holds ITM cuts flat-market theta bleed ~30% vs ATM
        while still capturing ~0.7x of the spot move — the documented
        sweet spot for 3-10 day option buys.
        """
        fo = info.get("nse_fo", name)
        direction = sig["direction"]
        options = _master.find_swing_options(fo, spot, direction, min_dte=15)
        if not options: return None
        if style == "itm":
            # ITM = strike below spot for CE, above spot for PE.
            itm = [o for o in options if (o["strike"] < spot if direction == "LONG"
                                           else o["strike"] > spot)]
            pool = itm or options
            # Nearest ITM strike to spot = roughly 0.6-0.7 delta.
            pool.sort(key=lambda o: abs(o["strike"] - spot))
            chosen = pool[0]
            intrinsic = max(0.0, (spot - chosen["strike"]) if direction == "LONG"
                            else (chosen["strike"] - spot))
            # Premium ≈ intrinsic + ~1.5% of spot extrinsic (monthly, slight ITM)
            est_prem = round(intrinsic + float(spot) * 0.015, 1)
        else:
            # Pick closest strike to ATM
            options.sort(key=lambda o: abs(o["strike"] - spot))
            chosen = options[0]
            # Rough premium estimate: ~2.5% of spot for ATM monthly
            est_prem = round(float(spot) * 0.025, 1)
        # Lot size: prefer the exchange's own value from Angel's instrument
        # master (revised quarterly per stock) over the hardcoded SWING_STOCKS
        # table, which goes stale — the NIFTY 65-vs-75 bug all over again.
        lot = int(chosen.get("lotsize") or 0) or info.get("lot_size", 50)
        capital = round(est_prem * lot, 0)
        sl_prem  = round(est_prem * 0.4, 1)   # 60% premium SL (2 ATR index move)
        t1_prem  = round(est_prem * 2.0, 1)   # 100% gain at T1
        return {
            "symbol":   chosen["symbol"],
            "strike":   chosen["strike"],
            "type":     chosen.get("type","CE") if direction=="LONG" else chosen.get("type","PE"),
            "expiry":   chosen["expiry"],
            "token":    chosen["token"],
            "dte":      chosen.get("dte",20),
            "lot_size": lot,
            "entry":    est_prem,
            "sl":       sl_prem,
            "target1":  t1_prem,
            "capital":  capital,
        }

    # ── AI Exit Analysis ─────────────────────────────────────────────
    def _ai_exit_all_open(self):
        """Run AI exit analysis on all open swing positions."""
        opens = swing_pos_list(status="OPEN")
        if not opens:
            return
        log.info(f"[Swing] AI exit check: {len(opens)} open positions")
        for pos in opens:
            try:
                self._ai_exit_one(pos)
                time.sleep(2)
            except Exception as e:
                log.warning(f"[Swing] AI exit err pos #{pos['id']}: {e}")

    _EXIT_SYSTEM_PROMPT = """You are a professional swing trader reviewing OPEN equity-or-options positions on a multi-day timeframe. Decide ONE of EXIT / HOLD / PARTIAL_EXIT and return JSON only.

DECISION RULES:
- EXIT if: original thesis broken (EMA stack reversed), RSI hit extremes against position, price touched SL zone, < 5 DTE on a profitable option (theta will erode quickly)
- PARTIAL_EXIT if: hit T1 and signals weakening — lock 50% profit
- HOLD if: thesis still intact, price in middle of range, setup improving

URGENCY:
- IMMEDIATE = act today
- SOON = act within 1-2 sessions
- MONITORING = no action needed, recheck next cycle

Respond in JSON only:
{"decision":"EXIT|HOLD|PARTIAL_EXIT","urgency":"IMMEDIATE|SOON|MONITORING","reasoning":"2-3 sentences","risk_note":"specific risk if any (or null)"}"""

    def _ai_exit_one(self, pos):
        """Claude AI analyses one open position and returns EXIT/HOLD/PARTIAL_EXIT."""
        api_key = CONFIG.get("anthropic_api_key","")
        if not api_key: return None

        name = pos["instrument"]
        direction = pos["direction"]
        info = SWING_STOCKS.get(name, {})
        token, exch, sym = self._resolve_token(name, info)

        # Get latest daily candles for technical freshness
        candles = self.client.daily_candles(token, exch, days=30) if token else []
        fresh_sig = SwingAnalysis.analyze(name, candles, info) if len(candles) >= 5 else {}
        fresh_sig = fresh_sig or {}

        # Current price
        ltp_data = self.client.ltp(exch, sym, token) if token else None
        cur_price = (ltp_data or {}).get("ltp") or pos.get("spot_entry") or 0

        # P&L calc
        entry = pos.get("option_entry") or pos.get("spot_entry") or 0
        pnl_pct = round((cur_price - entry) / max(entry, 0.01) * 100, 1) if entry else 0

        # Entry date → days held
        today = datetime.now(IST).date()
        try:
            d0 = datetime.strptime(pos.get("entry_date",""), "%Y-%m-%d").date()
            days_held = (today - d0).days
        except: days_held = 0

        dte = pos.get("option_dte") or 0
        if dte > 0 and pos.get("entry_date"):
            try:
                dte = max(0, dte - days_held)
            except: pass

        reasons = []
        try: reasons = json.loads(pos.get("reasons") or "[]")
        except: pass

        prompt = f"""OPEN POSITION:
Instrument: {name} ({direction})
{'Option: ' + (pos.get('option_symbol') or 'equity trade')}
Entry date: {pos.get('entry_date')} (held {days_held} days)
Spot entry: ₹{pos.get('spot_entry') or 'N/A'} | SL: ₹{pos.get('spot_sl') or 'N/A'} | T1: ₹{pos.get('spot_target1') or 'N/A'} | T2: ₹{pos.get('spot_target2') or 'N/A'}
Option entry: ₹{pos.get('option_entry') or 'N/A'} | Option SL: ₹{pos.get('option_sl') or 'N/A'}
Days to expiry (approx): {dte}

CURRENT STATUS:
Current price: ₹{round(cur_price, 2)}
P&L (approx): {pnl_pct:+.1f}%

FRESH DAILY TECHNICALS:
RSI: {fresh_sig.get('rsi', 'N/A')}
EMA stack: {('BULLISH' if fresh_sig.get('direction')=='LONG' else 'BEARISH') if fresh_sig.get('direction') else 'NEUTRAL'}
MACD hist: {fresh_sig.get('macd_hist', 'N/A')} ({'EXPANDING' if (fresh_sig.get('macd_hist',0) or 0) > 0 else 'CONTRACTING'})
Signal direction: {fresh_sig.get('direction') or 'NEUTRAL'}
Signal confidence: {fresh_sig.get('confidence') or 'N/A'}%

ORIGINAL ENTRY REASONS:
{'; '.join(reasons[:5]) or 'N/A'}"""

        ai = _anthropic_call(
            prompt,
            model=CONFIG.get("anthropic_model","claude-sonnet-4-5"),
            max_tokens=300, temperature=0, timeout=20,
            system=SwingEngine._EXIT_SYSTEM_PROMPT, layer="swing_exit",
        )
        if ai is None:
            return None
        try:
            decision = ai.get("decision","HOLD")
            reasoning = ai.get("reasoning","")
            now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            swing_pos_update(pos["id"],
                last_ai_decision=decision,
                last_ai_reasoning=reasoning,
                last_ai_ts=now_str)
            log.info(f"[Swing] AI exit #{pos['id']} {name}: {decision} — {reasoning[:60]}")
            # Swing exit Slack pings re-enabled (2026-07-28) for actionable
            # decisions only. Mute with SWING_SLACK_ENABLED=false.
            if decision in ("EXIT","PARTIAL_EXIT") and \
                    os.environ.get("SWING_SLACK_ENABLED", "true").lower() == "true":
                urgency = ai.get("urgency", "SOON")
                SlackAlert.send(
                    f"📤 *Swing {decision.replace('_', ' ').title()}: {name} "
                    f"{pos.get('direction','')}* ({urgency})\n"
                    f"{pos.get('option_symbol') or 'equity'} · entry ₹{entry} · "
                    f"now ₹{round(cur_price, 2)} ({pnl_pct:+.1f}%)\n"
                    f"_{reasoning[:200]}_")
            return ai
        except Exception as e:
            log.warning(f"[Swing] AI exit post-processing error: {e}")
            return None

    def _format_slack(self, name, sig, opt):
        arrow = "🟢" if sig["direction"]=="LONG" else "🔴"
        msg = (f"{arrow} *Swing: {name} {sig['direction']}*  conf={sig['confidence']}%\n"
               f"Spot: ₹{sig['price']} · SL: ₹{sig['sl']} · T1: ₹{sig['target1']} · T2: ₹{sig['target2']}\n"
               f"R:R {sig['risk_reward']} · RSI {sig['rsi']} · Est hold {sig.get('hold_days_est','3-15 days')}")
        if opt:
            msg += (f"\n📋 *Option:* {opt['symbol']} {opt['dte']}DTE"
                    f" | Entry ~₹{opt['entry']} · SL ₹{opt['sl']} · T1 ₹{opt['target1']}"
                    f" | 1 lot = {opt.get('lot_size','?')} qty · Capital ~₹{opt.get('capital','?')}")
        reasons = sig.get("reasons",[])[:3]
        if reasons:
            msg += "\nReasons: " + " · ".join(reasons)
        return msg

    def get_state(self):
        signals_list = sorted(self.signals.values(),
                              key=lambda x: x["signal"]["confidence"], reverse=True)
        return {
            "signals": signals_list,
            "positions": swing_pos_list(),
            "performance": swing_perf(),
            "instruments_count": len(SWING_STOCKS),
            "last_scan_time": max((s["updated_at"] for s in self.signals.values()), default="—"),
        }


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
@app.route("/dashboard")
def home():
    """Serve the trading dashboard UI (index.html).
    Both / and /dashboard work so Railway's health check + any bookmarked
    /dashboard URL both load the React app correctly."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        return send_file(html_path)
    # Fallback status JSON so the server at least responds meaningfully
    return jsonify({"name": "Intraday Signal Engine",
                    "status": "running" if engine.running else "stopped",
                    "error": "index.html not found — place it in the same folder as server.py"})

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

# ── Async backtest job store ────────────────────────────────────────
# In-memory only; persistent disk would be overkill since the backtest
# is interactive and the user always sees the result before refreshing.
_BACKTEST_JOBS = {}  # { job_id: { status, progress, result, started_at, error } }
_BACKTEST_JOBS_LOCK = threading.Lock()


def _run_backtest_job(job_id, days, budget, symbols, strategy="v2"):
    """Long-running backtest body. Updates _BACKTEST_JOBS[job_id] as it goes
    so the GET /api/backtest/jobs/<id> endpoint can stream progress.
    Runs in a daemon thread spawned by the POST handler.
    """
    def _set(**kw):
        with _BACKTEST_JOBS_LOCK:
            _BACKTEST_JOBS[job_id].update(kw)

    try:
        from datetime import date, timedelta
        try:
            from backtest_v2 import run_backtest
            import data_layer as _dl
            import dataclasses
        except Exception as e:
            _set(status="error", error=f"backtest import failed: {e}")
            return

        # Clear the option-day cache at the start of every backtest run so we
        # don't serve stale data from a previous job with different dates.
        try:
            _dl.reset_option_day_cache()
        except Exception:
            pass

        to_date   = datetime.now(IST).date() - timedelta(days=1)
        from_date = to_date - timedelta(days=days)

        all_trades = []
        by_symbol  = {}
        diag_log   = {}
        import io, contextlib
        for sym_i, sym in enumerate(symbols):
            _set(progress=f"Processing {sym} ({sym_i+1}/{len(symbols)}) [{strategy}]...")
            buf = io.StringIO()
            err_msg = None
            try:
                with contextlib.redirect_stdout(buf):
                    ts = run_backtest(sym, from_date, to_date, budget=budget,
                                       verbose=False, strategy=strategy)
            except Exception as e:
                log.warning(f"  backtest {sym} crashed: {e}")
                err_msg = f"{type(e).__name__}: {e}"
                ts = []
            captured = buf.getvalue()
            diag_lines = []
            for ln in captured.splitlines():
                if any(k in ln for k in ("✗", "✓", "No spot bars", "Angel login",
                                          "Got ", "Skip", "Backtest:")):
                    diag_lines.append(ln.strip())
            diag_log[sym] = {
                "lines":  diag_lines[-12:],
                "raw_signals": len(ts),
                "error":  err_msg,
            }
            taken_w = [t for t in ts if t.bucket == "TAKEN_WIN"]
            taken_l = [t for t in ts if t.bucket == "TAKEN_LOSS"]
            taken_n = len(taken_w) + len(taken_l)
            net = sum(t.net_pnl for t in taken_w + taken_l)
            wr  = round(len(taken_w) / taken_n * 100, 1) if taken_n else 0.0
            by_symbol[sym] = {
                "trades": taken_n,
                "wins":   len(taken_w),
                "losses": len(taken_l),
                "win_rate": wr,
                "net_pnl":  round(net, 0),
                "avg_win":  round(sum(t.net_pnl for t in taken_w) / len(taken_w), 0) if taken_w else 0,
                "avg_loss": round(sum(t.net_pnl for t in taken_l) / len(taken_l), 0) if taken_l else 0,
                "filtered": len([t for t in ts if t.bucket.startswith("FILTERED")]),
            }
            all_trades.extend(ts)

        _set(progress="Aggregating results...")
        taken = [t for t in all_trades if t.bucket in ("TAKEN_WIN", "TAKEN_LOSS")]
        wins  = [t for t in taken if t.bucket == "TAKEN_WIN"]
        losses = [t for t in taken if t.bucket == "TAKEN_LOSS"]
        cum = 0.0; peak = 0.0; dd = 0.0
        for t in sorted(taken, key=lambda x: (x.date, x.time)):
            cum += t.net_pnl
            peak = max(peak, cum)
            dd = min(dd, cum - peak)
        win_rate = round(len(wins) / len(taken) * 100, 1) if taken else 0.0
        net_pnl = sum(t.net_pnl for t in taken)
        avg_win  = (sum(t.net_pnl for t in wins) / len(wins))  if wins  else 0
        avg_loss = (sum(t.net_pnl for t in losses) / len(losses)) if losses else 0
        expectancy = round((win_rate/100) * avg_win + ((100-win_rate)/100) * avg_loss, 0)
        ordered = sorted(taken, key=lambda x: (x.date, x.time))
        curve = []; c = 0.0
        for t in ordered:
            c += t.net_pnl
            curve.append({"date": t.date, "time": t.time, "pnl": round(c, 0)})
        trades_out = [dataclasses.asdict(t) for t in all_trades[:500]]

        _set(status="done", progress="Complete", result={
            "ok": True,
            "params": {"days": days, "from": str(from_date), "to": str(to_date),
                       "symbols": symbols, "budget": budget, "strategy": strategy},
            "summary": {
                "total_signals": len(all_trades),
                "taken_count":   len(taken),
                "filtered_count": len(all_trades) - len(taken),
                "wins":          len(wins),
                "losses":        len(losses),
                "win_rate":      win_rate,
                "net_pnl":       round(net_pnl, 0),
                "avg_win":       round(avg_win, 0),
                "avg_loss":      round(avg_loss, 0),
                "expectancy":    expectancy,
                "max_drawdown":  round(dd, 0),
                "by_symbol":     by_symbol,
                "equity_curve":  curve,
                "diag":          diag_log,
            },
            "trades": trades_out,
        })
    except Exception as e:
        log.error(f"backtest job {job_id} crashed: {e}", exc_info=True)
        _set(status="error", error=f"{type(e).__name__}: {e}")


@app.route("/api/backtest", methods=["POST"])
@require_auth
def api_backtest():
    """Start an async backtest job. Returns a job_id immediately; the client
    polls /api/backtest/jobs/<job_id> for progress + final result.

    Body:
      { "days": int, "symbols": [...], "budget": int }
    Response:
      { "ok": true, "job_id": "<uuid>", "status": "pending" }

    Backtest is run in a daemon thread so the HTTP response returns in
    under a second, avoiding the Railway proxy's ~100s connection timeout
    that was breaking 30+ day windows.
    """
    try:
        body = flask_request.json or {}
        days     = max(1, min(int(body.get("days", 30)), 90))
        budget   = int(body.get("budget", CONFIG.get("budget", 50000)))
        symbols  = body.get("symbols") or ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        symbols  = [s for s in symbols if s in INSTRUMENTS]
        if not symbols:
            return jsonify({"ok": False, "error": "No valid symbols"}), 400
        # Strategy selector
        strategy = str(body.get("strategy", "v2")).lower()
        _allowed = ("v2", "orb", "gamma", "conductor", "scalper", "scalper_v3",
                    "reverter", "nifty_windows", "deadzone_fade", "patterns")
        if strategy not in _allowed:
            return jsonify({"ok": False,
                            "error": f"Unknown strategy '{strategy}' (expected {'|'.join(_allowed)})"}), 400

        import uuid
        job_id = uuid.uuid4().hex[:12]
        with _BACKTEST_JOBS_LOCK:
            _BACKTEST_JOBS[job_id] = {
                "status": "pending", "progress": "Starting...",
                "result": None, "error": None,
                "started_at": datetime.now(IST).isoformat(),
                "params": {"days": days, "symbols": symbols, "budget": budget,
                            "strategy": strategy},
            }
            # Garbage-collect jobs older than 30 minutes
            cutoff = datetime.now(IST) - timedelta(minutes=30)
            for jid in list(_BACKTEST_JOBS.keys()):
                try:
                    started = datetime.fromisoformat(_BACKTEST_JOBS[jid]["started_at"])
                    if started < cutoff:
                        _BACKTEST_JOBS.pop(jid, None)
                except Exception:
                    pass

        threading.Thread(target=_run_backtest_job,
                         args=(job_id, days, budget, symbols, strategy),
                         daemon=True,
                         name=f"Backtest-{job_id}-{strategy}").start()

        return jsonify({"ok": True, "job_id": job_id, "status": "pending"})
    except Exception as e:
        log.error(f"  /api/backtest spawn error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/backtest/jobs/<job_id>", methods=["GET"])
@require_auth
def api_backtest_job(job_id):
    """Poll a running backtest job. Returns status / progress, plus the
    final result once status='done'. Public-ish — same auth as POST."""
    with _BACKTEST_JOBS_LOCK:
        job = _BACKTEST_JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "Unknown job_id"}), 404
        return jsonify({
            "ok": True,
            "job_id": job_id,
            "status": job["status"],
            "progress": job.get("progress"),
            "error":   job.get("error"),
            "result":  job.get("result"),
            "started_at": job.get("started_at"),
        })


# Legacy synchronous path retained as a no-op stub — UI no longer calls this.
def _api_backtest_legacy_unused():
    """Run a multi-day backtest of the v2 strategy across NIFTY/BANKNIFTY/
    FINNIFTY and return JSON results. (Replaced by async job pattern above.)
    """
    try:
        body = flask_request.json or {}
        days     = max(1, min(int(body.get("days", 30)), 90))
        budget   = int(body.get("budget", CONFIG.get("budget", 50000)))
        symbols  = body.get("symbols") or ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        symbols  = [s for s in symbols if s in INSTRUMENTS]
        if not symbols:
            return jsonify({"ok": False, "error": "No valid symbols"}), 400

        from datetime import date, timedelta
        try:
            from backtest_v2 import run_backtest, summarise
            import dataclasses
        except Exception as e:
            return jsonify({"ok": False, "error": f"backtest import failed: {e}"}), 500

        to_date   = datetime.now(IST).date() - timedelta(days=1)  # yesterday
        from_date = to_date - timedelta(days=days)

        all_trades = []
        by_symbol  = {}
        diag_log   = {}   # { symbol: "captured stdout text" } — surfaced in API response
        import io, contextlib
        for sym in symbols:
            # Capture stdout so the user sees WHY a symbol returned 0 trades
            # (Angel login failed, no spot bars, etc.) instead of an empty card.
            buf = io.StringIO()
            err_msg = None
            try:
                with contextlib.redirect_stdout(buf):
                    ts = run_backtest(sym, from_date, to_date, budget=budget, verbose=False)
            except Exception as e:
                log.warning(f"  backtest {sym} crashed: {e}")
                err_msg = f"{type(e).__name__}: {e}"
                ts = []
            captured = buf.getvalue()
            # Pull out the key diagnostic lines from the captured output
            diag_lines = []
            for ln in captured.splitlines():
                if any(k in ln for k in ("✗", "✓", "No spot bars", "Angel login",
                                          "Got ", "Skip", "Backtest:")):
                    diag_lines.append(ln.strip())
            diag_log[sym] = {
                "lines":  diag_lines[-20:],  # last 20 diagnostic lines
                "raw_signals": len(ts),
                "error":  err_msg,
            }
            # Compute per-symbol summary
            taken_w = [t for t in ts if t.bucket == "TAKEN_WIN"]
            taken_l = [t for t in ts if t.bucket == "TAKEN_LOSS"]
            taken_n = len(taken_w) + len(taken_l)
            net = sum(t.net_pnl for t in taken_w + taken_l)
            wr  = round(len(taken_w) / taken_n * 100, 1) if taken_n else 0.0
            by_symbol[sym] = {
                "trades": taken_n,
                "wins":   len(taken_w),
                "losses": len(taken_l),
                "win_rate": wr,
                "net_pnl":  round(net, 0),
                "avg_win":  round(sum(t.net_pnl for t in taken_w) / len(taken_w), 0) if taken_w else 0,
                "avg_loss": round(sum(t.net_pnl for t in taken_l) / len(taken_l), 0) if taken_l else 0,
                "filtered": len([t for t in ts if t.bucket.startswith("FILTERED")]),
            }
            all_trades.extend(ts)

        # Aggregate stats across all symbols (TAKEN only — those are what the engine would have fired)
        taken = [t for t in all_trades if t.bucket in ("TAKEN_WIN", "TAKEN_LOSS")]
        wins  = [t for t in taken if t.bucket == "TAKEN_WIN"]
        losses = [t for t in taken if t.bucket == "TAKEN_LOSS"]
        cum = 0.0; peak = 0.0; dd = 0.0
        for t in sorted(taken, key=lambda x: (x.date, x.time)):
            cum += t.net_pnl
            peak = max(peak, cum)
            dd = min(dd, cum - peak)
        win_rate = round(len(wins) / len(taken) * 100, 1) if taken else 0.0
        net_pnl = sum(t.net_pnl for t in taken)
        avg_win  = (sum(t.net_pnl for t in wins) / len(wins))  if wins  else 0
        avg_loss = (sum(t.net_pnl for t in losses) / len(losses)) if losses else 0
        expectancy = round((win_rate/100) * avg_win + ((100-win_rate)/100) * avg_loss, 0)

        # Equity curve (cumulative net P&L per closed trade, ordered by time)
        ordered = sorted(taken, key=lambda x: (x.date, x.time))
        curve = []; c = 0.0
        for t in ordered:
            c += t.net_pnl
            curve.append({"date": t.date, "time": t.time, "pnl": round(c, 0)})

        # Cap returned per-trade rows at 500 for payload size
        trades_out = [dataclasses.asdict(t) for t in all_trades[:500]]

        return jsonify({
            "ok": True,
            "params": {
                "days": days, "from": str(from_date), "to": str(to_date),
                "symbols": symbols, "budget": budget,
            },
            "summary": {
                "total_signals": len(all_trades),
                "taken_count":   len(taken),
                "filtered_count": len(all_trades) - len(taken),
                "wins":          len(wins),
                "losses":        len(losses),
                "win_rate":      win_rate,
                "net_pnl":       round(net_pnl, 0),
                "avg_win":       round(avg_win, 0),
                "avg_loss":      round(avg_loss, 0),
                "expectancy":    expectancy,
                "max_drawdown":  round(dd, 0),
                "by_symbol":     by_symbol,
                "equity_curve":  curve,
                "diag":          diag_log,
            },
            "trades": trades_out,
        })
    except Exception as e:
        log.error(f"  /api/backtest error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/config", methods=["POST"])
@require_auth
def config():
    d=flask_request.json or{}
    if"target_min"in d:CONFIG["target_points_min"]=int(d["target_min"]);engine.sgen.tmin=int(d["target_min"])
    if"target_max"in d:CONFIG["target_points_max"]=int(d["target_max"]);engine.sgen.tmax=int(d["target_max"])
    # Trading capital — drives OptPicker's max_capital cap. The user can
    # now bump or lower this from the dashboard profile modal and the
    # NEXT signal will size its option position to fit the new budget.
    if "budget" in d:
        try:
            new_budget = int(d["budget"])
            if new_budget >= 5000:
                CONFIG["budget"] = new_budget
                log.info(f"💰 Capital updated → ₹{new_budget:,}")
        except Exception:
            pass
    return jsonify({"status":"ok", "budget": CONFIG.get("budget")})

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
            # STRICT GREEKS (step 15): reject if no live delta — mirrors OptPicker.pick
            if CONFIG.get("strict_greeks", False):
                log.warning(f"  STRICT_GREEKS (option_ltp): rejecting {o.get('symbol','?')} — no live delta")
                continue
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
    if not CONFIG["slack_enabled"] or not CONFIG["slack_webhook"]:
        return jsonify({"status": "failed", "reason": "SLACK_WEBHOOK env var is not set on the server"}), 200
    # Render a realistic sample so the user sees exactly what live alerts look like.
    sample_args = dict(
        instrument="BANKNIFTY",
        signal={
            "direction": "LONG", "confidence": 72,
            "entry": 53742, "sl": 53590, "target1": 53896, "target2": 54050,
            "risk_reward": "1 : 1.8", "rsi": 61,
            "reasons": ["EMA21 > EMA50 (stacked bullish)",
                        "VWAP reclaim with rising volume",
                        "Break of 09:25 ORB high (+18 pts)",
                        "RSI 61 (trending, not overbought)"],
            "timestamp": datetime.now(IST).strftime("%H:%M"),
        },
        option={
            "symbol": "BANKNIFTY 53800 CE", "action": "BUY",
            "entry": 235, "sl": 153, "target1": 352, "target2": 470,
            "t1_profit": 3510, "t2_profit": 7050,
            "capital": 7050, "max_loss": 2460,
            "delta": 0.46, "dte": 5,
        },
        timing={"target_by": "10:45", "est_duration": "~75 min", "sl_by": "11:30"},
        ai={
            "verdict": "TAKE", "confidence_adj": 4,
            "reasoning": "Strong morning trend with stacked EMAs and VIX below mean. "
                         "Setup respects 09:25 ORB and has clear invalidation.",
            "risk_note": "Watch 53,700 on retest — close below it nullifies the setup.",
        },
    )
    text = "🧪 SAMPLE — this is what a real signal looks like\n\n" + SlackAlert.format_signal(**sample_args)
    blocks = ([{"type": "section", "text": {"type": "mrkdwn",
                                            "text": "*🧪 SAMPLE — this is what a real signal looks like*"}}]
              + SlackAlert.format_signal_blocks(**sample_args))
    ok = SlackAlert.send(text, blocks=blocks)
    return jsonify({"status":"ok" if ok else "failed",
                    "reason": "delivered to Slack" if ok else "Slack rejected the webhook (URL or perms)"})

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


@app.route("/api/metrics")
def api_metrics():
    """Per-day engine telemetry — for the dashboard's risk + cost panels.

    Combines:
    - engine.metrics counters (scans, generated, alerted, AI skipped, blocks)
    - today's get_perf() snapshot (gross P&L, net after costs, totals)
    - candle-cache hit rate
    - Anthropic API usage (calls + tokens + cache hit/creation)
    - kill-switch + risk thresholds so the UI can show "₹X of Y limit"
    Public read endpoint — no secrets, no auth required.
    """
    today = datetime.now(IST).strftime("%Y-%m-%d")
    perf  = get_perf(date=today)
    cache_stats = engine.client.candle_cache_stats()
    # Trades today (open + closed) — used for trade-cap progress bar
    cnt_row = db_exec("SELECT COUNT(*) as cnt FROM signals WHERE date=?",
                      (today,), fetchone=True)
    trades_today = int(dict(cnt_row).get("cnt", 0)) if cnt_row else 0
    # Anthropic usage snapshot
    ant = dict(_ANTHROPIC_USAGE)
    ant["by_layer"] = dict(_ANTHROPIC_USAGE.get("by_layer") or {})
    return jsonify({
        "date": today,
        "engine": {
            "running":  bool(engine.running),
            "killswitch_tripped": bool(engine._killswitch_tripped),
            "scan_interval_sec": int(CONFIG.get("scan_interval_sec", 30)),
            "weight_adjustments": engine._weight_adj,
            "blocked_windows":    engine._blocked_windows,
            "auto_close": f"{int(CONFIG.get('auto_close_hour',15)):02d}:{int(CONFIG.get('auto_close_minute',15)):02d}",
            "strategy":   CONFIG.get("strategy", "v1"),
            "dry_run_v2": bool(CONFIG.get("dry_run_v2", False)),
        },
        "metrics_today": engine.metrics,
        "perf_today":    perf,
        "trades_today":  trades_today,
        "risk": {
            "daily_loss_limit":    int(CONFIG.get("daily_loss_limit", 2000)),
            "max_trades_per_day":  int(CONFIG.get("max_trades_per_day", 8)),
            "brokerage_per_lot_roundtrip": float(CONFIG.get("brokerage_per_lot_roundtrip", 100)),
            "slippage_bps_per_side":       float(CONFIG.get("slippage_bps_per_side", 50)),
        },
        "cache": {
            "candles":   cache_stats,
            "anthropic": ant,
            "anthropic_caching_enabled": bool(CONFIG.get("anthropic_cache_enabled", True)),
        },
        # Live AI spend ledger — computed per call from the API's own usage
        # numbers at official pricing, converted at AI_USD_INR (default 88).
        "ai_budget": {
            "month": datetime.now(IST).strftime("%Y-%m"),
            "spent_inr": round(_ai_spent_inr(), 2),
            "cap_inr": _ai_cap_inr(),
            "blocked": _ai_spent_inr() >= _ai_cap_inr() > 0,
        },
        "time": datetime.now(IST).strftime("%H:%M:%S"),
    })


@app.route("/api/gate-stats")
def api_gate_stats():
    """Shadow-log analytics: which gate is eating signals, and the most
    recent rejections. ?days=N (default 7) restricts the window."""
    try:
        days = max(1, min(int(flask_request.args.get("days", 7)), 90))
    except Exception:
        days = 7
    since = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
    counts = db_exec(
        "SELECT gate, instrument, COUNT(*) as cnt FROM gate_rejections "
        "WHERE date >= ? GROUP BY gate, instrument ORDER BY cnt DESC",
        (since,), fetch=True) or []
    recent = db_exec(
        "SELECT * FROM gate_rejections WHERE date >= ? ORDER BY id DESC LIMIT 50",
        (since,), fetch=True) or []
    return jsonify({
        "since": since, "days": days,
        "by_gate": [dict(r) for r in counts],
        "recent": [dict(r) for r in recent],
    })


@app.route("/api/signals/<int:sig_id>/void", methods=["POST"])
@require_auth
def api_void_signal(sig_id):
    """Void a phantom/erroneous trade record (e.g. a close booked off a bad
    quote). Sets status=VOIDED and zeroes its P&L so it no longer counts
    toward the kill-switch, stats, or daily summary — then unlatches the
    kill-switch and clears loss cooldowns so the engine can resume.
    Does NOT touch any real broker position (the engine never holds one)."""
    row = db_exec("SELECT * FROM signals WHERE id=?", (sig_id,), fetchone=True)
    if not row:
        return jsonify({"error": f"signal {sig_id} not found"}), 404
    prev = dict(row)
    db_exec("UPDATE signals SET status='VOIDED', result='VOIDED', pnl_rupees=0, "
            "pnl_rupees_net=0, pnl_points=0 WHERE id=?", (sig_id,))
    engine._killswitch_tripped = False
    try:
        engine.tracker._loss_cooldown.clear()
        engine.tracker._exit_pending.pop(sig_id, None)
    except Exception:
        pass
    log.info(f"🩹 Signal #{sig_id} VOIDED by operator (was {prev.get('result')} "
             f"₹{prev.get('pnl_rupees')}) — killswitch unlatched, cooldowns cleared")
    SlackAlert.send(f"🩹 Trade record #{sig_id} ({prev.get('instrument')} {prev.get('direction')}, "
                    f"booked {prev.get('result')} ₹{prev.get('pnl_rupees')}) was VOIDED — "
                    f"phantom close removed from today's stats. Kill-switch re-armed and unblocked.")
    return jsonify({"ok": True, "voided": sig_id,
                    "was": {"result": prev.get("result"), "pnl_rupees": prev.get("pnl_rupees")}})


@app.route("/api/killswitch", methods=["POST"])
@require_auth
def api_killswitch():
    """Manual kill-switch toggle. Body: {"action": "trip" | "reset"}.

    'trip'  — immediately stop new alerts (latches until next day)
    'reset' — clear the latch (use with caution — only if you know why it tripped)
    """
    d = flask_request.json or {}
    action = (d.get("action") or "").lower()
    if action == "trip":
        engine._killswitch_tripped = True
        engine.metrics["kill_switch_hits"] += 1
        SlackAlert.send("🛑 *Kill-switch tripped manually* — no new alerts today.")
        return jsonify({"ok": True, "tripped": True})
    elif action == "reset":
        engine._killswitch_tripped = False
        return jsonify({"ok": True, "tripped": False})
    return jsonify({"error": "action must be 'trip' or 'reset'"}), 400


@app.route("/api/strategy", methods=["GET"])
def api_strategy_get():
    """Read the current strategy + dry-run state.

    Used by the dashboard's StrategyToggle to render the current selection
    without polling /api/metrics (which is bigger).
    """
    return jsonify({
        "strategy":   CONFIG.get("strategy", "auto"),
        "dry_run_v2": bool(CONFIG.get("dry_run_v2", False)),
        "available":  ["auto", "conductor", "reverter", "scalper_v3",
                        "nifty_windows", "deadzone_fade", "patterns", "v1", "v2"],
        "updated_at": get_engine_state("strategy_updated_at", default=None),
    })


@app.route("/api/strategy", methods=["POST"])
@require_auth
def api_strategy_set():
    """Switch strategy + dry-run mode from the dashboard. Persists in SQLite
    so the choice survives process restarts. Engine's next scan tick picks
    up the new mode automatically — no redeploy, no env vars.

    Body:
        {"strategy": "v1" | "v2", "dry_run_v2": true | false}

    Either field is optional; missing fields stay at current value.
    """
    d = flask_request.json or {}
    new_strategy = d.get("strategy")
    new_dry_run  = d.get("dry_run_v2")

    if new_strategy is not None:
        s = str(new_strategy).lower()
        if s not in ("auto", "v1", "v2", "conductor", "scalper_v3", "reverter",
                     "nifty_windows", "deadzone_fade", "patterns"):
            return jsonify({"error": "strategy must be 'auto', 'v1', 'v2', 'conductor', 'scalper_v3', 'reverter', 'nifty_windows', 'deadzone_fade', or 'patterns'"}), 400
        CONFIG["strategy"] = s
        set_engine_state("strategy", s)

    if new_dry_run is not None:
        b = bool(new_dry_run)
        CONFIG["dry_run_v2"] = b
        set_engine_state("dry_run_v2", "true" if b else "false")

    set_engine_state("strategy_updated_at", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))

    # Slack-notify the change so it's auditable
    try:
        SlackAlert.send(
            f"⚙️ *Engine strategy changed*\n"
            f"strategy = `{CONFIG.get('strategy','v1')}`  ·  "
            f"dry_run_v2 = `{CONFIG.get('dry_run_v2', False)}`"
        )
    except Exception:
        pass

    log.info(f"⚙️  Strategy set via API: strategy={CONFIG.get('strategy')} "
             f"dry_run_v2={CONFIG.get('dry_run_v2')}")
    return jsonify({
        "ok": True,
        "strategy":   CONFIG.get("strategy", "v1"),
        "dry_run_v2": bool(CONFIG.get("dry_run_v2", False)),
    })


@app.route("/api/replay-premium")
def api_replay_premium():
    """Real-data premium lookup for the dashboard's Replay panel.

    Replaces the old client-side linear formula (indexEntry × 0.0034) with
    the same NSE-bridge cascade used by backtest_v2.py:
      1. Cache hit (instant)
      2. Real Angel One historical (for active contracts)
      3. NSE daily bhavcopy + back-solved real IV + hard clamp to day range
      4. BS-with-default-IV last resort

    Returns the SAME extended dict the backtest sees, so the dashboard can
    show: NSE day low/high/settle, real IV, clamp flag, bs_raw.

    Query params:
      symbol     — NIFTY / BANKNIFTY / FINNIFTY
      strike     — option strike (integer)
      opt_type   — CE or PE
      expiry     — YYYY-MM-DD (option's expiry date)
      ts         — YYYY-MM-DD HH:MM:SS (the moment we're querying)

    Returns 200 with the dict, or 200 with {"price": null, "source": "missing"}
    when no path could resolve. Never 500s — the dashboard handles missing
    gracefully and falls back to its old estimate.
    """
    try:
        symbol   = (flask_request.args.get("symbol") or "").upper().strip()
        strike   = float(flask_request.args.get("strike") or 0)
        opt_type = (flask_request.args.get("opt_type") or "").upper().strip()
        expiry_s = flask_request.args.get("expiry") or ""
        ts_s     = flask_request.args.get("ts") or ""
        if not (symbol and strike and opt_type in ("CE", "PE") and ts_s):
            return jsonify({"error": "missing params",
                            "needed": "symbol,strike,opt_type,ts (expiry optional)"}), 400

        # Accept both "YYYY-MM-DD HH:MM:SS" and ISO "YYYY-MM-DDTHH:MM:SS"
        ts_s_clean = ts_s.replace("T", " ").split(".")[0].split("+")[0]
        ts = datetime.strptime(ts_s_clean[:19], "%Y-%m-%d %H:%M:%S")

        # If expiry not supplied, pick an expiry that ACTUALLY EXISTS for the
        # symbol. NIFTY has weekly Tuesday expiries; BANKNIFTY and FINNIFTY
        # are MONTHLY last-Tuesday only (post-Nov-2024). The old logic picked
        # "next Tuesday" universally — fine for NIFTY but a non-existent
        # contract for BANKNIFTY/FINNIFTY, which is why the NSE bhavcopy
        # returned nothing for those two.
        if expiry_s:
            expiry_d = datetime.strptime(expiry_s, "%Y-%m-%d").date()
        else:
            d = ts.date()
            min_dte = int(flask_request.args.get("min_dte", 5))
            if symbol == "NIFTY":
                # Walk weekly Tuesdays forward until we find one >= min_dte
                days_ahead = (1 - d.weekday()) % 7
                if days_ahead == 0: days_ahead = 7
                while days_ahead < min_dte:
                    days_ahead += 7
                expiry_d = d + timedelta(days=days_ahead)
            else:
                # BANKNIFTY / FINNIFTY — monthly last-Tuesday only
                try:
                    cands = data_layer._candidate_expiries_for_date(d, symbol)
                    cands = [e for e in cands if (e - d).days >= min_dte]
                    expiry_d = cands[0] if cands else (d + timedelta(days=21))
                except Exception:
                    expiry_d = d + timedelta(days=21)
            expiry_s = expiry_d.strftime("%Y-%m-%d")

        # ── Diagnostic trace — every step records its outcome so the dashboard
        # can show EXACTLY which dependency failed (Angel auth? NSE block? etc) ──
        debug = {
            "expiry_used":          expiry_s,
            "expiry_d_diff_days":   (expiry_d - ts.date()).days,
            "angel_present":        bool(engine.client),
            "angel_connected":      bool(engine.client and engine.client.connected),
            "ensure_login_tried":   False,
            "ensure_login_ok":      False,
            "instrument_master":    bool(_master.loaded),
            "option_token_found":   None,
            "jugaad_present":       None,
            "nse_day_fetched":      None,
            "nse_day_settle":       None,
            "spot_bars_count":      None,
        }

        # Force Angel login if not connected — the bridge needs it for spot + (sometimes) the option token
        if engine.client and not engine.client.connected:
            debug["ensure_login_tried"] = True
            try:
                debug["ensure_login_ok"] = bool(engine.client.ensure())
            except Exception as _e:
                debug["ensure_login_err"] = str(_e)[:200]

        ac = engine.client if engine.client and engine.client.connected else None

        # Check if the option token exists in current instrument master (will fail for expired contracts)
        try:
            if _master.loaded:
                key = (symbol, float(strike), opt_type,
                       expiry_d.strftime("%d%b%Y").upper())
                debug["option_token_found"] = key in _master.nfo
        except Exception:
            pass

        # Pre-check: does jugaad-data even load?
        try:
            import data_layer
            debug["jugaad_present"] = bool(data_layer._HAS_JUGAAD)
        except Exception as _e:
            debug["data_layer_import_err"] = str(_e)[:200]

        # Spot probe — independent of option-chain, validates Angel historical works for this date
        try:
            spot_probe = data_layer.get_spot_bars(
                symbol, ts - timedelta(minutes=10), ts + timedelta(minutes=10),
                "5min", angel_client=ac
            )
            debug["spot_bars_count"] = int(len(spot_probe))
            if not spot_probe.empty and "close" in spot_probe.columns:
                debug["spot_at_ts"] = float(spot_probe["close"].iloc[-1])
        except Exception as _e:
            debug["spot_probe_err"] = str(_e)[:200]

        # NSE day probe — does jugaad-data return this contract's daily OHLC?
        nse_probe = None
        try:
            nse_probe = data_layer.get_nse_contract_day(symbol, float(strike), opt_type,
                                                        expiry_d, ts.date())
            debug["nse_day_fetched"] = bool(nse_probe)
            if nse_probe:
                debug["nse_day_settle"] = nse_probe.get("settle")
                debug["nse_day_high"]   = nse_probe.get("high")
                debug["nse_day_low"]    = nse_probe.get("low")
        except Exception as _e:
            debug["nse_day_err"] = str(_e)[:200]

        # Now the actual cascade
        try:
            result = data_layer.get_option_premium_at(
                symbol, strike, opt_type, expiry_d, ts, angel_client=ac,
            )
        except Exception as _e:
            log.warning(f"  /api/replay-premium cascade crash: {_e}")
            return jsonify({"price": None, "source": "error",
                            "error": str(_e), "debug": debug,
                            "symbol": symbol, "strike": strike, "opt_type": opt_type,
                            "expiry": expiry_s, "ts": ts_s_clean}), 200

        if result is None:
            return jsonify({
                "price": None, "source": "missing",
                "reason": _diagnose_missing(debug),
                "debug": debug,
                "symbol": symbol, "strike": strike, "opt_type": opt_type,
                "expiry": expiry_s, "ts": ts_s_clean,
            })
        out = dict(result)
        out["debug"] = debug
        out.update({
            "symbol": symbol, "strike": strike, "opt_type": opt_type,
            "expiry": expiry_s, "ts": ts_s_clean,
        })

        # If the cascade resolved via Angel-live or cache (path that doesn't
        # carry NSE day OHLC), promote the data from the nse_probe we already
        # fetched above so the outcome block can classify against the day range.
        try:
            if nse_probe:
                if out.get("nse_day_low")    is None: out["nse_day_low"]    = nse_probe.get("low")
                if out.get("nse_day_high")   is None: out["nse_day_high"]   = nse_probe.get("high")
                if out.get("nse_day_settle") is None: out["nse_day_settle"] = nse_probe.get("settle")
                if out.get("nse_day_close")  is None: out["nse_day_close"]  = nse_probe.get("close")
        except Exception:
            pass

        # ─── Outcome estimate (real backtest the dashboard can show) ───
        # We BUY the option at `entry`. SL/T1/T2 use the same premium-pct
        # ladder as the live engine (35% loss / 50% gain / 100% gain).
        # Then classify against the NSE day OHLC we already fetched.
        try:
            entry = float(out.get("price") or 0)
            day_low  = out.get("nse_day_low")
            day_high = out.get("nse_day_high")
            day_settle = out.get("nse_day_settle") or out.get("nse_day_close")
            if entry > 0:
                sl_pct = float(os.getenv("OPT_SL_PCT", 0.35))
                t1_pct = float(os.getenv("OPT_T1_PCT", 0.50))
                t2_pct = float(os.getenv("OPT_T2_PCT", 1.00))
                sl = max(round(entry * (1.0 - sl_pct)), 5)
                t1 = round(entry * (1.0 + t1_pct))
                t2 = round(entry * (1.0 + t2_pct))
                LOT_SIZES = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 65}
                lot_size = LOT_SIZES.get(symbol, 75)
                budget = 20000
                est_lots = max(1, min(3, int((budget * 0.5) // max(entry * lot_size, 1))))
                contracts = est_lots * lot_size

                outcome = {
                    "entry": entry, "sl": sl, "t1": t1, "t2": t2,
                    "lot_size": lot_size, "lots": est_lots, "contracts": contracts,
                }
                if day_low is not None and day_high is not None:
                    day_low_f  = float(day_low)
                    day_high_f = float(day_high)
                    sl_touched = day_low_f  <= sl
                    t1_touched = day_high_f >= t1
                    t2_touched = day_high_f >= t2

                    outcome["max_profit_pts"] = round(day_high_f - entry, 2)
                    outcome["max_loss_pts"]   = round(day_low_f  - entry, 2)
                    outcome["max_profit_rs"]  = int(round((day_high_f - entry) * contracts))
                    outcome["max_loss_rs"]    = int(round((day_low_f  - entry) * contracts))
                    if day_settle is not None:
                        outcome["close_pnl_pts"] = round(float(day_settle) - entry, 2)
                        outcome["close_pnl_rs"]  = int(round((float(day_settle) - entry) * contracts))

                    if sl_touched and t1_touched:
                        outcome["result"] = "WHIPSAW"
                        outcome["result_pnl_rs"] = outcome.get("close_pnl_rs", 0)
                        outcome["result_note"]   = "Both SL and T1 touched same day — order unknown without intraday bars. Net = held-to-close."
                    elif t2_touched:
                        outcome["result"] = "T2_HIT"
                        outcome["result_pnl_rs"] = int(round((t2 - entry) * contracts))
                        outcome["result_note"]   = "Day's high reached Target 2 (+100%)."
                    elif t1_touched:
                        outcome["result"] = "T1_HIT"
                        outcome["result_pnl_rs"] = int(round((t1 - entry) * contracts))
                        outcome["result_note"]   = "Day's high reached Target 1 (+50%)."
                    elif sl_touched:
                        outcome["result"] = "SL_HIT"
                        outcome["result_pnl_rs"] = int(round((sl - entry) * contracts))
                        outcome["result_note"]   = "Day's low reached SL (−35%)."
                    else:
                        outcome["result"] = "HELD_TO_CLOSE"
                        outcome["result_pnl_rs"] = outcome.get("close_pnl_rs", 0)
                        outcome["result_note"]   = "Neither SL nor T1 touched — closed at settle."

                    # Full-day OHLC includes pre-signal moves. Honest disclaimer.
                    outcome["caveat"] = ("Based on the option's full-day OHLC. "
                                         "For exact post-signal outcome, intraday option bars (Dhan) are needed.")
                else:
                    outcome["result"] = "UNKNOWN"
                    outcome["result_note"] = "NSE day range not available — cannot estimate outcome."

                out["outcome"] = outcome
        except Exception as _e:
            log.warning(f"  /api/replay-premium outcome calc failed: {_e}")

        return jsonify(out)
    except Exception as e:
        log.warning(f"  /api/replay-premium top-level error: {e}")
        return jsonify({"error": str(e), "price": None, "source": "error"}), 200


def _diagnose_missing(debug: dict) -> str:
    """Translate a debug snapshot into a human-readable failure reason so the
    dashboard's user can act on it (e.g., 'log in to Angel' or 'NSE blocked')."""
    if not debug.get("angel_connected"):
        if debug.get("ensure_login_tried") and not debug.get("ensure_login_ok"):
            return ("Angel One re-login failed during the request. "
                    "Hit /api/login on the server, or set the ANGEL_* env vars on Railway and redeploy.")
        return "Angel One client is not connected. The bridge needs Angel for spot data."
    if debug.get("spot_bars_count") == 0:
        return ("Angel returned no historical spot bars for this timestamp. "
                "May be too far in the past (Angel keeps ~30 days of intraday).")
    if not debug.get("jugaad_present"):
        return ("jugaad-data not installed in the Railway image. "
                "Add `jugaad-data>=0.30` to requirements.txt and redeploy.")
    if debug.get("nse_day_fetched") is False:
        return ("NSE blocked the bhavcopy request (common from cloud IPs). "
                "Try locally; or wait for the Dhan account.")
    return "Unknown — see debug field for details."


@app.route("/api/v2-diag")
def api_v2_diag():
    """Last 20 v2 score-card decisions per instrument.

    The v2 strategy ALWAYS records its score-card on every scan tick (even
    when no signal fires). This endpoint exposes the buffer so the user can
    see exactly which conditions failed on every bar.

    Returns:
        {
          "NIFTY":     [{ts, price, long_score, short_score, long_checks, ...}],
          "BANKNIFTY": [...],
          "FINNIFTY":  [...],
        }
    """
    out = {}
    for name, buf in (engine._v2_diag or {}).items():
        out[name] = list(buf)
    return jsonify({
        "strategy_active": CONFIG.get("strategy", "v1"),
        "dry_run_v2":      bool(CONFIG.get("dry_run_v2", False)),
        "decisions":       out,
        "time":            datetime.now(IST).strftime("%H:%M:%S"),
    })


# ── Swing engine singleton ──────────────────────────────────────────
swing_engine = SwingEngine(engine.client)

# ═══════════════════════════════════════════════════════════════════
# SWING FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════
@app.route("/api/swing/results")
def api_swing_results():
    """Swing paper-tracking feed for the app's Swing tab.

    open[]   — OPEN AUTO_PAPER positions with live LTP / P&L (refreshed each
               ~30-min swing cycle), days held, max hold, days remaining.
    closed[] — most recent closed positions with result, exit_reason,
               hold_days, pnl. ?days=N (default 30) window.
    summary  — win/loss aggregate over the closed set.
    """
    try:
        days = max(1, min(int(flask_request.args.get("days", 30)), 180))
    except Exception:
        days = 30
    since = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
    today = datetime.now(IST).date()

    # On-request quote refresh: the swing cycle only updates every ~30 min,
    # but the app wants a live "how much would I be making" read. Refresh
    # stale quotes here (per-position TTL, default 45s; capped per request
    # so a pile of open positions can't stall the endpoint or hit Angel
    # rate limits).
    try:
        _ttl = float(os.environ.get("SWING_LIVE_TTL", "45"))
    except Exception:
        _ttl = 45.0
    _fetched = 0
    _now_ts = time.time()
    open_src = [p for p in swing_pos_list(status="OPEN")
                if (p.get("source") or "") == "AUTO_PAPER"]
    for p in open_src:
        lv = (getattr(swing_engine, "_paper_live", {}) or {}).get(p["id"], {})
        if _fetched < 8 and (_now_ts - (lv.get("epoch") or 0)) >= _ttl:
            try:
                swing_engine._paper_live_quote(p)
                _fetched += 1
            except Exception:
                pass
    live = getattr(swing_engine, "_paper_live", {}) or {}

    open_rows = []
    for p in open_src:
        try:
            d0 = datetime.strptime(p.get("entry_date", ""), "%Y-%m-%d").date()
            held = (today - d0).days
        except Exception:
            held = 0
        try:
            max_hold = int((json.loads(p.get("indicators") or "{}") or {})
                           .get("max_hold_days") or SwingEngine._paper_max_hold_days())
        except Exception:
            max_hold = SwingEngine._paper_max_hold_days()
        lv = live.get(p["id"], {})
        p["held_days"] = held
        p["max_hold_days"] = max_hold
        p["days_remaining"] = max(0, max_hold - held)
        p["live_ltp"] = lv.get("ltp")
        p["live_pnl_pct"] = lv.get("pnl_pct")
        p["live_est_pnl_rs"] = lv.get("est_pnl_rs")
        p["live_ts"] = lv.get("ts")
        open_rows.append(p)

    closed_raw = db_exec(
        "SELECT * FROM swing_positions WHERE status='CLOSED' AND source='AUTO_PAPER' "
        "AND COALESCE(exit_date, entry_date) >= ? ORDER BY id DESC LIMIT 100",
        (since,), fetch=True) or []
    closed = [dict(r) for r in closed_raw]

    wins = [r for r in closed if r.get("result") == "WIN"]
    pnls = [r.get("pnl_rupees") or 0 for r in closed]
    summary = {
        "total": len(closed), "wins": len(wins), "losses": len(closed) - len(wins),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "est_total_pnl": round(sum(pnls), 0),
        "open_count": len(open_rows),
        "window_days": days,
    }
    return jsonify({"open": open_rows, "closed": closed, "summary": summary,
                    "time": datetime.now(IST).strftime("%H:%M:%S")})


@app.route("/api/swing/status")
def swing_status():
    """Current swing signals + open positions + performance."""
    return jsonify(swing_engine.get_state())

@app.route("/api/swing/positions", methods=["GET"])
def swing_positions_get():
    status_filter = flask_request.args.get("status")
    return jsonify({"positions": swing_pos_list(status=status_filter)})

@app.route("/api/swing/positions", methods=["POST"])
@require_auth
def swing_positions_post():
    """Manually add a swing position."""
    data = flask_request.get_json(force=True) or {}
    if not data.get("instrument") or not data.get("direction"):
        return jsonify({"error":"instrument and direction required"}), 400
    data["source"] = "MANUAL"
    row_id = swing_pos_save(data)
    return jsonify({"ok": True, "id": row_id})

@app.route("/api/swing/positions/<int:pos_id>", methods=["POST"])
@require_auth
def swing_positions_update(pos_id):
    """Update a position (partial update or close)."""
    data = flask_request.get_json(force=True) or {}
    action = data.pop("action", None)
    if action == "close":
        exit_price  = data.get("exit_price") or data.get("spot_exit")
        option_exit = data.get("option_exit")
        result = swing_pos_close(pos_id, exit_price, option_exit)
        return jsonify({"ok": True, "result": result})
    # Generic field update
    allowed = {"status","lots","option_entry","spot_entry","spot_sl","spot_target1",
               "spot_target2","option_sl","option_target1","last_ai_decision"}
    filtered = {k:v for k,v in data.items() if k in allowed}
    if filtered:
        swing_pos_update(pos_id, **filtered)
    return jsonify({"ok": True})

@app.route("/api/swing/exit-analysis", methods=["POST"])
@require_auth
def swing_exit_analysis_all():
    """Trigger AI exit analysis for all open positions immediately."""
    try:
        threading.Thread(target=swing_engine._ai_exit_all_open, daemon=True).start()
        return jsonify({"ok": True, "msg": "AI exit analysis triggered for all open positions"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/swing/exit-analysis/<int:pos_id>", methods=["POST"])
@require_auth
def swing_exit_analysis_one(pos_id):
    """Trigger AI exit analysis for one position."""
    pos = db_exec("SELECT * FROM swing_positions WHERE id=?", (pos_id,), fetchone=True)
    if not pos:
        return jsonify({"error":"position not found"}), 404
    pos = dict(pos)
    try:
        result = swing_engine._ai_exit_one(pos)
        return jsonify({"ok": True, "ai": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/swing/save-signal", methods=["POST"])
@require_auth
def swing_save_signal():
    """Save a swing signal as a position (when user decides to take the trade)."""
    data = flask_request.get_json(force=True) or {}
    name = data.get("instrument")
    snap = swing_engine.signals.get(name)
    if not snap:
        return jsonify({"error": f"No active swing signal for {name}"}), 404
    sig = snap["signal"]; opt = snap.get("option")
    pos_data = {
        "instrument": name, "instrument_type": snap.get("type","STOCK"),
        "direction": sig["direction"],
        "spot_entry": sig["price"], "spot_sl": sig["sl"],
        "spot_target1": sig["target1"], "spot_target2": sig["target2"],
        "reasons": sig.get("reasons",[]),
        "indicators": {k:sig.get(k) for k in ["rsi","atr","ema9","ema21","ema50","vol_ratio","macd_hist"]},
    }
    if opt:
        pos_data.update({
            "option_symbol": opt.get("symbol"), "option_strike": opt.get("strike"),
            "option_type": opt.get("type"), "option_expiry": opt.get("expiry"),
            "option_token": opt.get("token"), "option_dte": opt.get("dte"),
            "option_entry": opt.get("entry"), "option_sl": opt.get("sl"),
            "option_target1": opt.get("target1"),
            "lot_size": opt.get("lot_size"), "lots": data.get("lots",1),
            "capital": opt.get("capital"),
        })
    row_id = swing_pos_save(pos_data)
    return jsonify({"ok": True, "id": row_id})


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
                # Run the SAME strategy smoke-test as engine.start() so the
                # auto-startup path can't silently bypass it. Without this,
                # commit c489960's loud-fail guarantee wouldn't apply to the
                # boot-time engine start (only to manual /api/start calls).
                smoke_ok, smoke_err = engine._smoke_test_strategy()
                if not smoke_ok:
                    log.error(f"▶ Auto-startup: REFUSING TO START — strategy `{CONFIG.get('strategy','v1')}` smoke test failed: {smoke_err}")
                    SlackAlert.send(f"⛔ *Auto-startup refused*\nStrategy `{CONFIG.get('strategy','v1')}` failed smoke test: `{smoke_err}`\nFix the bug and redeploy — engine is NOT scanning.")
                    return
                log.info("▶ Auto-startup: starting signal scan engine...")
                engine.running = True
                threading.Thread(target=engine._loop, daemon=True, name="ScanLoop").start()
                log.info("▶ Auto-startup: scan engine running ✅")
            # Start swing engine after intraday engine is up
            if not swing_engine.running:
                log.info("▶ Auto-startup: starting swing engine...")
                swing_engine.start()
                log.info("▶ Auto-startup: swing engine running ✅")
            SlackAlert.send(f"🚀 *Intraday Engine Started*\n"
                            f"Scanning NIFTY · BANKNIFTY · FINNIFTY\n"
                            f"Alerts will arrive here when confidence ≥ threshold")
            return
        err = engine.client.last_login_error or "unknown"
        log.warning(f"▶ Auto-startup: login attempt {attempt}/3 failed — {err}")
        if attempt < 3:
            _t.sleep(35)  # next TOTP window
    log.error("▶ Auto-startup: all 3 login attempts failed. Dashboard /api/diag shows the reason.")

threading.Thread(target=_startup, daemon=True, name="Startup").start()


def _scheduler():
    """Server-side auto on/off so the engine doesn't depend on a browser tab.

    Mon-Fri 08:45 IST → engine.start() if not already running.
    Mon-Fri 15:30 IST → engine.stop()  if currently running.

    Wakes every 30 seconds, fires once per slot (dedup'd by date+slot key so
    a restart inside the same minute doesn't double-fire).

    RESEARCH-MODE OVERRIDE: when env var INTRADAY_RESEARCH_MODE=1 is set
    on Railway, the scheduler refuses to auto-start the engine (auto-off
    still works as a safety net). Used while we're rebuilding the strategy
    — stops the engine from quietly turning itself on tomorrow morning and
    burning more capital on a strategy we've proven loses money.
    """
    import time as _t
    _t.sleep(20)  # give _startup a head-start to log in
    last_fired = {}  # { "YYYY-MM-DD-on" | "YYYY-MM-DD-off": True }
    while True:
        try:
            now = datetime.now(IST)
            day = now.weekday()       # Mon=0 ... Sun=6
            day_key = now.strftime("%Y-%m-%d")
            hh, mm = now.hour, now.minute
            # Default OFF — the engine auto-starts every trading morning. The old
            # default ("1") combined with the loop's daily 15:15 self-stop meant the
            # engine silently never scanned again after day one unless the container
            # restarted. Set INTRADAY_RESEARCH_MODE=1 on Railway to pause auto-start
            # during a strategy rebuild.
            _rm = os.getenv("INTRADAY_RESEARCH_MODE", "0").strip()
            research_mode = (_rm == "1")

            # ── Auto-ON: weekdays at 08:45 IST ────────────────────────────
            if day <= 4 and hh == 8 and mm == 45:
                key = f"{day_key}-on"
                if not last_fired.get(key):
                    last_fired[key] = True
                    if research_mode:
                        log.info("⏰ Scheduler: 08:45 IST — RESEARCH MODE active, "
                                 "refusing to auto-start. Set INTRADAY_RESEARCH_MODE=0 "
                                 "on Railway to re-enable auto-start.")
                        try:
                            SlackAlert.send("🔬 *Engine auto-start blocked* — research mode on. "
                                            "Strategy rebuild in progress; clear the flag on Railway when ready.")
                        except Exception:
                            pass
                    elif not engine.running:
                        log.info("⏰ Scheduler: auto-ON at 08:45 IST")
                        try:
                            engine.start()
                            SlackAlert.send("⏰ *Engine auto-started* — 08:45 IST")
                        except Exception as e:
                            log.warning(f"⏰ auto-ON failed: {e}")
                    else:
                        log.info("⏰ Scheduler: 08:45 IST hit, engine already running")

            # ── Catch-up auto-ON: engine down during market hours ─────────
            # The 08:45 slot fires exactly once; if the container was asleep,
            # restarting, or login failed at that minute, the engine stayed
            # dead all day. Retry every ~10 min during market hours unless the
            # operator stopped it manually or research mode is on.
            if (day <= 4 and not research_mode and not engine.running
                    and not getattr(engine, "_manual_stop", False)
                    and (915 <= hh * 100 + mm < 1500)):
                key = f"{day_key}-catchup-{hh:02d}{mm // 10}"
                if not last_fired.get(key):
                    last_fired[key] = True
                    log.info("⏰ Scheduler: engine down during market hours — catch-up start")
                    try:
                        res = engine.start()
                        if (res or {}).get("status") == "ok":
                            SlackAlert.send("⏰ *Engine catch-up start* — was down during market hours")
                    except Exception as e:
                        log.warning(f"⏰ catch-up start failed: {e}")

            # ── Auto-OFF: weekdays at 15:30 IST ───────────────────────────
            if day <= 4 and hh == 15 and mm == 30:
                key = f"{day_key}-off"
                if not last_fired.get(key):
                    last_fired[key] = True
                    if engine.running:
                        log.info("⏰ Scheduler: auto-OFF at 15:30 IST")
                        try:
                            engine.stop()
                            SlackAlert.send("⏰ *Engine auto-stopped* — 15:30 IST · session closed")
                        except Exception as e:
                            log.warning(f"⏰ auto-OFF failed: {e}")
                    else:
                        log.info("⏰ Scheduler: 15:30 IST hit, engine already stopped")

            # Garbage-collect old keys (yesterday and earlier) to keep dict bounded
            cutoff = (now - timedelta(days=2)).strftime("%Y-%m-%d")
            for k in list(last_fired.keys()):
                if k[:10] < cutoff:
                    last_fired.pop(k, None)
        except Exception as e:
            log.warning(f"⏰ Scheduler tick error: {e}")
        _t.sleep(30)

threading.Thread(target=_scheduler, daemon=True, name="Scheduler").start()


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
