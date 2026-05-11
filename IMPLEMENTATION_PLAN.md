# Intraday Signal Engine — Cost/Safety/Quality/Accuracy Overhaul

Hand-off document for an AI coding assistant (Cursor / Claude Code / another Claude session). Each section below is one self-contained change with the exact file(s) it touches, what to change, why it matters, and how to verify.

**16 changes total**: 13 from the original audit (cost / safety / honesty / visibility) + 3 added afterward (steps 14-16: real-time exit prices, strict-greeks mode, real backtest with missed-opportunity analysis). Steps 14-16 directly address the complaint that displayed strike prices and exit levels are "off" — they replace delta-scaled estimates with exact market prices.

## Critical context: what's real vs estimated in the existing code

Before any changes, the existing engine looks like this:

| Field | Source | Status |
|---|---|---|
| Strike (e.g., `NIFTY25600CE`) | Angel instrument master + live chain | ✅ Real |
| Entry premium | mid of live best-bid/best-ask | ✅ Real |
| Delta | Angel `getOptionGreek` IF returned, else `fallback_delta()` ladder | ⚠ Real or estimated |
| Option SL/T1/T2 | `entry ± (index_distance × delta)` — linear model | ❌ Estimated |
| Exit detection | live `cur_opt` compared to estimated SL/T1 | Hybrid |

Steps **2, 14, 15** together eliminate all estimation: real costs in P&L, real percentage-based exits, strict greeks-or-skip mode.

**Repo layout (current state)**
- `server.py` — 4053-line Flask backend (signal engine, Angel One client, 5 Claude AI layers, SQLite persistence, all routes). Single file.
- `index.html` — 3047-line React-via-CDN dashboard. Single file. Hosted on GitHub Pages.
- `requirements.txt`, `Procfile`, `events.json`, `manifest.json`, `.env.example`, `README.md`
- DB at `signals.db` (SQLite, ephemeral on Railway unless a volume is mounted)

**Apply order**: bottom-up by risk. The list is **already sorted** so step 1 is safest, step 13 is most invasive. Push and verify after each block of 1-3 changes — do not push all 13 at once.

---

## 0 · Deployment infrastructure (do this FIRST — required for anything else)

**Why first**: Railway's UI "Custom Start Command" overrides Procfile and ignores `bash` script wrappers, with `$PORT` not expanding. Without solving this, no code change you push can run.

**Files**: `railway.json` (new), `Dockerfile` (new), `start.sh` (new), `Procfile` (no edit needed once `railway.json` exists).

**Action**:

1. Create `Dockerfile` at repo root:
```dockerfile
FROM python:3.11.6-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py backtest.py events.json index.html manifest.json ./
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:${PORT:-8080} --timeout 120 --workers 1 --threads 4"]
```

2. Create `railway.json`:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": { "builder": "DOCKERFILE", "dockerfilePath": "Dockerfile" },
  "deploy": { "restartPolicyType": "ON_FAILURE", "restartPolicyMaxRetries": 5 }
}
```

**Why this works**: Forces Railway to use `Dockerfile` builder (bypasses Custom Start Command UI override and Procfile parsing). The `CMD ["sh","-c",...]` form guarantees `${PORT:-8080}` expands at runtime via shell, regardless of Railway's quirks.

**Verify**: After push + redeploy, Deploy Logs show `[INFO] Listening at: http://0.0.0.0:8080` and `▶ Auto-startup: scan engine running ✅`.

---

## 1 · CONFIG defaults (lowest risk — pure additive)

**File**: `server.py` lines ~30-62 (`CONFIG = {...}` block).

**Add these keys** (all have safe defaults, no existing behavior changes unless env vars overridden):
```python
"scan_interval_sec": int(os.environ.get("SCAN_INTERVAL", "30")),  # was 5
"candle_cache_ttl":  int(os.environ.get("CANDLE_CACHE_TTL", "90")),
"daily_loss_limit":  int(os.environ.get("DAILY_LOSS_LIMIT", "2000")),
"max_trades_per_day":int(os.environ.get("MAX_TRADES_PER_DAY", "8")),
"brokerage_per_lot_roundtrip": float(os.environ.get("BROKERAGE_PER_LOT", "100")),
"slippage_bps_per_side": float(os.environ.get("SLIPPAGE_BPS_PER_SIDE", "50")),
"auto_close_hour":   int(os.environ.get("AUTO_CLOSE_HOUR", "15")),
"auto_close_minute": int(os.environ.get("AUTO_CLOSE_MINUTE", "15")),  # was 25
"anthropic_model_inflight": os.environ.get(
    "ANTHROPIC_MODEL_INFLIGHT", "claude-haiku-4-5-20251001"),  # was Sonnet
"anthropic_cache_enabled": os.environ.get("ANTHROPIC_CACHE_ENABLED", "true").lower() == "true",
```

**Update `.env.example`** to document each new env var.

**Why it matters**: Sets defaults that drive every other change below. Switching Layer D inflight model from Sonnet → Haiku alone is ~12× cheaper; changing the scan interval is ~6× fewer API calls.

**Verify**: `python -c "from server import CONFIG; print(CONFIG['scan_interval_sec'])"` → `30`.

---

## 2 · Brokerage + slippage in P&L (DB schema + computation)

**Files**: `server.py` (init_db, update_result, get_perf, three call sites in PLTracker + TradeManager).

**Action**:

1. In `init_db()`, add to the migration list:
```python
("brokerage_rs", "REAL"),
("slippage_rs", "REAL"),
("pnl_rupees_net", "REAL"),
("option_exit_realistic", "REAL"),
```

2. Add new helper function before `update_result`:
```python
def estimate_costs(option_entry, option_exit, qty, lots):
    """Returns (brokerage_rs, slippage_rs, realistic_exit_price)."""
    try:
        lots = max(1, int(lots or 1))
        qty = max(1, int(qty or 0))
        bps_side = float(CONFIG.get("slippage_bps_per_side", 50)) / 10000.0
        per_lot = float(CONFIG.get("brokerage_per_lot_roundtrip", 100))
        brokerage = round(per_lot * lots, 2)
        entry_slip = float(option_entry or 0) * bps_side * qty
        exit_slip  = float(option_exit  or 0) * bps_side * qty
        slippage = round(entry_slip + exit_slip, 2)
        realistic_exit = round(float(option_exit or 0) * (1 - bps_side), 2) if option_exit else None
        return brokerage, slippage, realistic_exit
    except Exception:
        return 0.0, 0.0, option_exit
```

3. Modify `update_result(...)` to accept `option_entry`, `qty`, `lots` kwargs, compute costs via `estimate_costs`, and persist all four new columns.

4. Modify `get_perf()` to accept optional `date` arg and return `total_pnl_net`, `total_brokerage`, `total_slippage` keys alongside existing `total_pnl`.

5. Update three callers to pass the new kwargs:
   - `PLTracker.check()` — when result is set with cur_opt
   - `PLTracker.close_all()` — when force-marking-to-market at session end
   - `TradeManager.tick()` — `act == "CLOSE"` branch

**Why**: Real brokerage (~₹100/lot RT on Zerodha/Angel) + slippage (mid → bid/ask, ~50bps each side on ATM options) eats 5-10% of displayed gross P&L. Without this, your dashboard P&L is fiction.

**Verify**: After a trade closes, `SELECT pnl_rupees, pnl_rupees_net, brokerage_rs FROM signals WHERE status='CLOSED' ORDER BY id DESC LIMIT 1;` — net should be lower than gross by ~₹100-300.

---

## 3 · Candle cache wrapper (cuts Angel One API calls ~6×)

**File**: `server.py`, `AngelClient.candles()` method (around line ~706).

**Action**:

1. In `AngelClient.__init__()`, add:
```python
self._candle_cache = {}
self._candle_cache_hits = 0
self._candle_cache_misses = 0
```

2. Wrap `candles()` with a TTL cache keyed by `(token, exchange, interval, days)`. Cache TTL from `CONFIG["candle_cache_ttl"]` (default 90s). Add a `force_refresh=False` kwarg for backtest/replay paths that need fresh data.

3. Add a `candle_cache_stats()` method returning `{hits, misses, hit_rate_pct, size}` for telemetry.

**Why**: 5-min candles update every 5 min, but old code re-fetched every 5s. ~95% wasted API calls + risk of Angel rate-limit ban. 90s cache is fresher than the candle interval, so no signal-freshness loss.

**Verify**: `/api/metrics` → `cache.candles.hit_rate_pct` should climb above 80% within 10 minutes of engine start.

---

## 4 · Scan interval bump 5s → 30s

**File**: `server.py` line ~37 (already done by step 1's CONFIG default).

This only requires CONFIG change in step 1. Set `SCAN_INTERVAL=30` env var on Railway (or rely on the new default).

**Why**: 5-min candles can't change in <5min. Aligns scan cadence with the 30s chain cache TTL. Cuts engine compute and API calls without losing freshness.

**Verify**: Deploy logs / `/api/metrics` show `scan_interval_sec: 30`.

---

## 5 · Eliminate duplicate option-chain re-fetch

**File**: `server.py`, `Engine._loop()` around line ~2830 (alert path's "re-pick option on fresh spot" block).

**Old (problematic)**:
```python
if chain and opt is not None:
    chain2, atm2 = self.client.option_chain(inst, sig["price"])  # ← always re-fetches
    ...
```

**New**:
```python
cc_age_now = time.time() - (self._chain_cache.get(name, {}).get("ts", 0) or 0)
if chain and opt is not None and cc_age_now > 30:
    # only re-fetch if cached chain is stale (>30s)
    chain2, atm2 = self.client.option_chain(inst, sig["price"])
    ...
```

**Why**: On every triggered alert, the engine was re-fetching the full option chain (~30 strikes, FULL mode with depth) even though the cache was usually <5s old. Halves API calls on alerts.

**Verify**: Engine log no longer shows back-to-back `Chain: 30 live prices, ATM=...` lines for the same instrument.

---

## 6 · Auto-close cutoff 15:25 → 15:15 (configurable)

**File**: `server.py`, `Engine._loop()` around line ~2590-2594.

**Old**:
```python
if now.hour==15 and now.minute>=25:
    self.tracker.close_all();self.running=False
    log.info("🔔 Market close");break
if now.hour<9 or(now.hour==9 and now.minute<20)or now.hour>15 or(now.hour==15 and now.minute>=25):
    time.sleep(30);continue
```

**New**:
```python
close_h = int(CONFIG.get("auto_close_hour", 15))
close_m = int(CONFIG.get("auto_close_minute", 15))
if now.hour == close_h and now.minute >= close_m:
    self.tracker.close_all()
    # Layer C EOD learning runs here too — see step 12.
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
```

**Why**: Last 15 min of NSE has 2-3× wider option spreads. Closing at 15:15 saves ~2-3% on every losing exit. Also: the old code's separate `_maybe_eod` at 15:45 was unreachable because the engine exited at 15:25 — now EOD runs right at close.

**Verify**: At 15:15 IST during market hours, engine logs `🔔 Auto-close at 15:15` and stops. EOD `🧠 EOD learning persisted` line appears in same minute.

---

## 7 · Daily kill-switch (loss limit + trade cap)

**Files**: `server.py` — `Engine.__init__`, new `_check_killswitch()` method, `Engine._loop`, new `/api/killswitch` route.

**Action**:

1. In `Engine.__init__()`:
```python
self._killswitch_tripped = False
```

2. Add method:
```python
def _check_killswitch(self):
    if self._killswitch_tripped: return True
    today = datetime.now(IST).strftime("%Y-%m-%d")
    row = db_exec("SELECT COUNT(*) as cnt, COALESCE(SUM(pnl_rupees),0) as pnl "
                  "FROM signals WHERE date=?", (today,), fetchone=True)
    row = dict(row) if row else {"cnt":0, "pnl":0}
    cnt = int(row.get("cnt") or 0)
    pnl = float(row.get("pnl") or 0)
    # Apply brokerage estimate to gross
    closed = db_exec("SELECT option_lots FROM signals WHERE date=? AND status='CLOSED'",
                     (today,), fetch=True) or []
    adj_pnl = pnl
    for r in closed:
        adj_pnl -= float(CONFIG.get("brokerage_per_lot_roundtrip", 100)) * int(dict(r).get("option_lots") or 1)
    limit = float(CONFIG.get("daily_loss_limit", 2000) or 0)
    cap   = int(CONFIG.get("max_trades_per_day", 8) or 0)
    tripped = False
    if limit > 0 and adj_pnl <= -limit:
        tripped = True
        log.warning(f"🛑 KILL-SWITCH: daily loss ₹{adj_pnl:.0f} ≤ -₹{limit:.0f}")
        SlackAlert.send(f"🛑 *Kill-switch tripped — daily loss limit*\nNet P&L: ₹{adj_pnl:.0f}")
    elif cap > 0 and cnt >= cap:
        tripped = True
        log.warning(f"🛑 KILL-SWITCH: trades today {cnt} ≥ {cap}")
        SlackAlert.send(f"🛑 *Kill-switch tripped — daily trade cap* {cnt}/{cap}")
    if tripped:
        self._killswitch_tripped = True
    return tripped
```

3. In `Engine._loop()`, after the R:R gate and before the confidence check:
```python
if self._check_killswitch():
    self._prev[name] = result
    continue
```

4. Add Flask route:
```python
@app.route("/api/killswitch", methods=["POST"])
@require_auth
def api_killswitch():
    d = flask_request.json or {}
    action = (d.get("action") or "").lower()
    if action == "trip":
        engine._killswitch_tripped = True
        SlackAlert.send("🛑 *Kill-switch tripped manually*")
        return jsonify({"ok": True, "tripped": True})
    elif action == "reset":
        engine._killswitch_tripped = False
        return jsonify({"ok": True, "tripped": False})
    return jsonify({"error": "action must be 'trip' or 'reset'"}), 400
```

5. Reset latch on date change inside step 12's `_maybe_load_adjustments()`.

**Why**: A bad day with no brake can ladder into 6+ losers. Hard stop at ₹2000 net loss OR 8 trades.

**Verify**: With test data, set `DAILY_LOSS_LIMIT=100` env, force a losing trade, confirm next signal logs `🛑 KILL-SWITCH:` and Slack alerts.

---

## 8 · Per-day metrics counters + `/api/metrics` endpoint

**Files**: `server.py` — `Engine.__init__`, increments at key paths in `_loop`, new route.

**Action**:

1. In `Engine.__init__()`:
```python
self.metrics = {
    "date":                datetime.now(IST).strftime("%Y-%m-%d"),
    "scans_total":         0,
    "signals_generated":   0,
    "signals_alerted":     0,
    "ai_skipped":          0,
    "ai_waited":           0,
    "rr_blocked":          0,
    "regime_blocked":      0,
    "blocked_window_hits": 0,
    "chain_failures":      0,
    "ai_api_failures":     0,
    "kill_switch_hits":    0,
}
```

2. Increment counters at these points in `_loop`:
   - `scans_total` — top of each iteration
   - `regime_blocked` — `if name in avoid:` branch
   - `signals_generated` — after `sig=self.sgen.analyze(df)` returns non-None
   - `rr_blocked` — R:R gate fail
   - `kill_switch_hits` — when `_check_killswitch` returns True
   - `ai_skipped` / `ai_waited` — Layer B SKIP/WAIT branches
   - `ai_api_failures` — when `_anthropic_call` returns None
   - `signals_alerted` — after `save_signal(...)` succeeds
   - `chain_failures` — `option_chain` exception

3. Reset metrics on date roll inside step 12.

4. Add Flask route (read-only, public — no auth):
```python
@app.route("/api/metrics")
def api_metrics():
    today = datetime.now(IST).strftime("%Y-%m-%d")
    perf  = get_perf(date=today)
    cache_stats = engine.client.candle_cache_stats()
    cnt_row = db_exec("SELECT COUNT(*) as cnt FROM signals WHERE date=?",
                      (today,), fetchone=True)
    trades_today = int(dict(cnt_row).get("cnt", 0)) if cnt_row else 0
    return jsonify({
        "date": today,
        "engine": {
            "running":  bool(engine.running),
            "killswitch_tripped": bool(engine._killswitch_tripped),
            "scan_interval_sec": int(CONFIG.get("scan_interval_sec", 30)),
            "weight_adjustments": engine._weight_adj,
            "blocked_windows":    engine._blocked_windows,
            "auto_close": f"{int(CONFIG.get('auto_close_hour',15)):02d}:{int(CONFIG.get('auto_close_minute',15)):02d}",
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
            "anthropic": dict(_ANTHROPIC_USAGE),  # See step 9
            "anthropic_caching_enabled": bool(CONFIG.get("anthropic_cache_enabled", True)),
        },
        "time": datetime.now(IST).strftime("%H:%M:%S"),
    })
```

**Why**: Currently the only source of truth is `signals.log`. With metrics, you can tune the engine numerically: which time windows produce what signals, where the chain analytics fail, AI skip rates per instrument.

**Verify**: `curl https://<your-app>/api/metrics` → JSON with `engine`, `cache`, `metrics_today`, `perf_today`, `risk` keys.

---

## 9 · Anthropic prompt caching (≥80% input-token reduction on cache hits)

**Files**: `server.py` — `_anthropic_call` function, plus a `_SYSTEM_PROMPT` class constant on each of the 5 layers.

**Action**:

1. Refactor `_anthropic_call(prompt, ...)` to accept new kwargs:
```python
def _anthropic_call(prompt, model=None, max_tokens=800, temperature=0.2, timeout=20,
                    system=None, layer=None):
    ...
    body = {
        "model": ...,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        blocks = []
        if isinstance(system, str):
            blocks = [{"type": "text", "text": system}]
        elif isinstance(system, list):
            for b in system:
                if isinstance(b, str):
                    blocks.append({"type": "text", "text": b})
                else:
                    blocks.append(dict(b))
        if blocks and CONFIG.get("anthropic_cache_enabled", True):
            last = blocks[-1]
            if "cache_control" not in last:
                last["cache_control"] = {"type": "ephemeral"}
        body["system"] = blocks
    ...
```

2. Add module-level usage telemetry:
```python
_ANTHROPIC_USAGE = {
    "calls": 0, "errors": 0,
    "input_tokens": 0, "output_tokens": 0,
    "cache_read_tokens": 0, "cache_creation_tokens": 0,
    "by_layer": {},
}
```
Increment from response `data["usage"]` after every successful call.

3. For each AI layer, extract the static rule-block into a `_SYSTEM_PROMPT` class constant (must be ≥1024 tokens for Sonnet 4.5 caching to engage), then pass via `system=` kwarg.
   - **Layer A** `RegimeBrief._SYSTEM_PROMPT` — regime taxonomy + bias rules
   - **Layer B** `SignalValidation._SYSTEM_PROMPT` — chain reading rules, OI velocity rules, signal rules (this one is HUGE — biggest cache win)
   - **Layer C** `LearningLoop._SYSTEM_PROMPT` — indicator weight nudge taxonomy
   - **Layer D** `TradeManager._SYSTEM_PROMPT` — HOLD/CLOSE decision rules (Haiku 4.5 needs ≥4096 tokens to cache; below that the request runs uncached, no error — savings come from Haiku itself per step 1)
   - **Swing exit** `SwingEngine._EXIT_SYSTEM_PROMPT` — EXIT/HOLD/PARTIAL_EXIT rules

4. In each layer's call, change from `_anthropic_call(big_full_prompt, ...)` to `_anthropic_call(small_dynamic_prompt, system=ClassName._SYSTEM_PROMPT, layer="...", ...)`.

**Format reference** (Anthropic prompt caching):
```json
{
  "model": "claude-sonnet-4-5",
  "system": [{
    "type": "text",
    "text": "<static rule book>",
    "cache_control": {"type": "ephemeral"}
  }],
  "messages": [{"role": "user", "content": "<dynamic per-call data>"}]
}
```

**Why**: Layer B fires on every alert with ~2500 tokens of mostly-static rules. Caching cuts that to ~500 tokens (the dynamic per-signal data). At Sonnet 4.5 prices, this is ~$0.01 per call → ~$0.002. Multiply by 100+ alerts/month. Layer A/C run once/day each but caching is free to add. Layer D switching to Haiku (step 1) is the bigger inflight saving.

**Verify**: After a few alerts, `/api/metrics` → `cache.anthropic.cache_read_tokens` should grow above 0 and represent ≥70% of `input_tokens + cache_read_tokens`.

---

## 10 · Layer D inflight model → Haiku 4.5

**File**: covered by step 1's `CONFIG` change. Just verify the env var:
```
ANTHROPIC_MODEL_INFLIGHT=claude-haiku-4-5-20251001
```

**Why**: Layer D fires every 2 min on every open position — 50-100 calls/day. Each is a tiny structured decision (HOLD/CLOSE/TRAIL_SL/PARTIAL_EXIT_50). Sonnet is overkill. Haiku 4.5 is ~12× cheaper at near-identical quality on this task.

**Verify**: `/api/metrics` → `cache.anthropic.by_layer.inflight` shows calls accumulating. Slack alerts for trade-management actions still arrive within seconds (Haiku is faster than Sonnet too).

---

## 11 · Wire Layer C (EOD learning) back into the scanner

**Files**: `server.py` — `SignalGen.analyze` signature, `Engine.__init__`, new `_maybe_load_adjustments` method, call site in `_loop`.

**Action**:

1. Modify `SignalGen.analyze(df)` to accept two new optional kwargs:
```python
def analyze(self, df, weight_adj=None, blocked_windows=None):
    if blocked_windows:
        now_hm = datetime.now(IST).strftime("%H:%M")
        for win in blocked_windows:
            try:
                a, b = win.split("-")
                if a.strip() <= now_hm <= b.strip():
                    return None
            except Exception: continue
    wa = weight_adj or {}
    w_rsi  = int(wa.get("rsi", 0) or 0)
    w_macd = int(wa.get("macd", 0) or 0)
    w_st   = int(wa.get("supertrend", 0) or 0)
    w_vwap = int(wa.get("vwap", 0) or 0)
    w_ema  = int(wa.get("ema", 0) or 0)
    w_vol  = int(wa.get("volume", 0) or 0)
    # ... existing scoring, but each indicator's contribution gets its weight added.
    # Example: bs+=15+w_ema  (was: bs+=15)
```

Apply the corresponding `+w_xxx` additive to every indicator's contribution: EMA cross, RSI extremes, MACD cross, VWAP cross, SuperTrend flip, Volume surge.

2. Add to `Engine.__init__`:
```python
self._weight_adj = {}
self._blocked_windows = []
self._adj_loaded_for = None
```

3. Add `Engine._maybe_load_adjustments(now)`:
```python
def _maybe_load_adjustments(self, now):
    today = now.strftime("%Y-%m-%d")
    if self._adj_loaded_for == today: return
    try:
        row = db_exec(
            "SELECT * FROM daily_adjustments WHERE date < ? ORDER BY date DESC LIMIT 1",
            (today,), fetchone=True)
        if row:
            row = dict(row)
            self._weight_adj      = json.loads(row.get("indicator_weight_adjustments") or "{}") or {}
            self._blocked_windows = json.loads(row.get("time_windows_to_avoid") or "[]") or []
            log.info(f"🧠 Layer C feedback loaded from {row.get('date')}: weights={self._weight_adj}")
        else:
            self._weight_adj = {}
            self._blocked_windows = []
    except Exception as e:
        log.warning(f"  Layer C adjustments load failed: {e}")
    if self.metrics.get("date") != today:
        self.metrics = {k: (today if k == "date" else 0) for k in self.metrics}
        self._killswitch_tripped = False
    self._adj_loaded_for = today
```

4. Call it once per loop iteration (top of `_loop`, after `_maybe_regime`):
```python
self._maybe_load_adjustments(now)
```

5. Pass to analyzer:
```python
sig = self.sgen.analyze(df, weight_adj=self._weight_adj,
                        blocked_windows=self._blocked_windows)
```

**Why**: The `daily_adjustments` table was being WRITTEN by `LearningLoop.run()` at EOD but NEVER READ. Pure cost, zero benefit. Wiring this means Claude's nightly review actually changes tomorrow's behavior.

**Verify**: First day, no adjustments exist (engine logs nothing). Second day onwards, look for `🧠 Layer C feedback loaded from <yesterday>: weights={...}` at engine boot. `/api/metrics` → `engine.weight_adjustments` and `engine.blocked_windows` populate.

---

## 12 · Backtest harness (separate file)

**File**: `backtest.py` (new, ~340 lines).

**Purpose**: Standalone CLI that replays `SignalGen` + `OptPicker` over historical 5-min candles with realistic costs, reports win-rate / expectancy / max drawdown.

**Structure**:
```python
"""Backtest harness for the intraday signal engine."""
import argparse, csv, json, os, sys
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import (AngelClient, SignalGen, OptPicker, TA, INSTRUMENTS,
                    PREMIUM_RANGES, fallback_delta, CONFIG, IST,
                    estimate_costs, _master)
import pandas as pd

def estimate_option_premium(spot, strike, opt_type, dte, atr):
    """Rough Black-Scholes-free estimate. Angel API has no historical chain."""
    # ... (intrinsic + 0.4 * atr * sqrt(dte) * delta_at_strike)

def simulate_trade(future_candles, opt_entry, opt_sl, opt_t1, ...):
    """Walk forward 5-min bars. Approximate option price as
       opt_price[i] = opt_entry + (idx_close[i] - idx_entry) * delta.
       Exit when SL or T1 crosses, or timeout at 24 bars."""

def run_backtest(instrument, days, *, verbose=False):
    """Replay one instrument over `days`. Returns trade list + summary."""

def _summarise(instrument, trades, ...):
    """Print + return win-rate, gross/net P&L, expectancy, max drawdown."""

def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--instrument", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    # ... loop over instruments, write CSV/JSON, print aggregate
```

**Critical caveat to document inline**: option premiums are estimated because Angel One has no historical option-chain API. Win-rate and expectancy DIRECTION are reliable. Absolute rupee figures are ±20%.

**Why**: Without this, you cannot answer "does this strategy make money?" You're trading on intuition. With this, you can tune indicator weights and calibrate confidence thresholds against real historical data.

**Usage**:
```bash
python3 backtest.py --instrument NIFTY --days 30 --csv out.csv
python3 backtest.py --all --days 60 --json summary.json
```

---

## 13 · Dashboard RiskBanner (UI surface for kill-switch + telemetry)

**File**: `index.html` — add component definition near `TickerTape`, add metrics state, render in main App.

**Action**:

1. In `App()` state declarations (~line 2240):
```javascript
const [metrics, setMetrics] = useState(null);
```

2. Add metrics polling effect alongside existing intervals (in the same useEffect that polls `/api/status` etc.):
```javascript
const fetchMetrics = async () => {
  if (!alive) return;
  try {
    const r = await apiJSON('/api/metrics');
    if (alive && r) setMetrics(r);
  } catch(e) { /* silent */ }
};
fetchMetrics();
const metricsId = setInterval(fetchMetrics, 8000);
// add clearInterval(metricsId) to cleanup
```

3. Add `RiskBanner` component before `TickerTape`:
```javascript
function RiskBanner({ metrics, onTrip, onReset }) {
  if (!metrics) return null;
  const risk = metrics.risk || {};
  const perf = metrics.perf_today || {};
  const eng  = metrics.engine || {};
  const ant  = (metrics.cache && metrics.cache.anthropic) || {};
  const candles = (metrics.cache && metrics.cache.candles) || {};
  const tripped = !!eng.killswitch_tripped;
  const lossLimit = Number(risk.daily_loss_limit || 2000);
  const tradeCap  = Number(risk.max_trades_per_day || 8);
  const netPnl    = Number(perf.total_pnl_net || 0);
  const trades    = Number(metrics.trades_today || 0);
  const lossPct = lossLimit > 0 ? Math.max(0, Math.min(100, (-netPnl / lossLimit) * 100)) : 0;
  const tradePct = tradeCap > 0 ? Math.max(0, Math.min(100, (trades / tradeCap) * 100)) : 0;
  const antTotal = Number(ant.cache_read_tokens || 0) + Number(ant.input_tokens || 0);
  const antHitPct = antTotal > 0 ? Math.round((Number(ant.cache_read_tokens || 0) / antTotal) * 100) : 0;
  const candleHitPct = candles.hit_rate_pct != null ? candles.hit_rate_pct : 0;
  const tone = tripped || lossPct >= 100 || tradePct >= 100 ? 'short'
              : lossPct >= 70 || tradePct >= 70 ? 'warn' : 'long';
  const toneVar = tone === 'short' ? 'var(--short)' : tone === 'warn' ? 'var(--warn)' : 'var(--long)';
  const toneDim = tone === 'short' ? 'var(--short-dim)' : tone === 'warn' ? 'var(--warn-dim)' : 'var(--long-dim)';
  const label = tripped ? '🛑 KILL-SWITCH TRIPPED' : (tone === 'long' ? '🛡 SAFE' : tone === 'warn' ? '⚠ NEAR LIMIT' : '⛔ AT LIMIT');
  return (
    <div style={{ padding:'10px var(--pad-x)', background: tripped ? 'oklch(from var(--short) l c h / 0.18)' : toneDim,
                  borderBottom: '1px solid ' + toneVar, display:'flex', alignItems:'center',
                  gap:18, fontSize:12, flexWrap:'wrap' }}>
      <span style={{fontFamily:'var(--mono)', fontWeight:600, fontSize:11,
                    letterSpacing:'0.08em', color: toneVar, whiteSpace:'nowrap'}}>{label}</span>
      {/* Loss progress bar */}
      <div style={{display:'flex', alignItems:'center', gap:8, minWidth:200}}>
        <span style={{fontSize:10.5, color:'var(--ink-2)', fontFamily:'var(--mono)'}}>LOSS</span>
        <div style={{flex:1, height:6, background:'var(--bg-2)', borderRadius:3, position:'relative', overflow:'hidden'}}>
          <div style={{position:'absolute', left:0, top:0, bottom:0, width: lossPct + '%',
                       background: toneVar, borderRadius: 3, transition: 'width 400ms ease'}}/>
        </div>
        <span className="mono" style={{fontSize:11, color: netPnl >= 0 ? 'var(--long)' : 'var(--short)',
                                       minWidth:90, textAlign:'right'}}>
          {netPnl >= 0 ? '+₹' : '−₹'}{Math.abs(Math.round(netPnl)).toLocaleString('en-IN')} / ₹{lossLimit.toLocaleString('en-IN')}
        </span>
      </div>
      {/* Trade count progress bar */}
      <div style={{display:'flex', alignItems:'center', gap:8, minWidth:160}}>
        <span style={{fontSize:10.5, color:'var(--ink-2)', fontFamily:'var(--mono)'}}>TRADES</span>
        <div style={{flex:1, height:6, background:'var(--bg-2)', borderRadius:3, position:'relative', overflow:'hidden'}}>
          <div style={{position:'absolute', left:0, top:0, bottom:0, width: tradePct + '%',
                       background: toneVar, borderRadius: 3, transition: 'width 400ms ease'}}/>
        </div>
        <span className="mono" style={{fontSize:11, color:'var(--ink-1)', minWidth:48, textAlign:'right'}}>
          {trades} / {tradeCap}
        </span>
      </div>
      <span style={{flex:1}}/>
      {/* Cache telemetry */}
      <div className="muted" style={{display:'flex', gap:12, fontSize:11, fontFamily:'var(--mono)'}}>
        <span title="Anthropic cache hit rate (cached tokens vs total input)">
          AI cache <span style={{color:'var(--ink-1)'}}>{antHitPct}%</span>
          <span style={{color:'var(--ink-3)', marginLeft:4}}>· {Number(ant.calls||0)} calls</span>
        </span>
        <span title="Candle cache hit rate">
          Candles <span style={{color:'var(--ink-1)'}}>{candleHitPct}%</span>
        </span>
        <span title="Auto-close cutoff">
          Close <span style={{color:'var(--ink-1)'}}>{eng.auto_close || '15:15'}</span>
        </span>
      </div>
      {/* Manual trip / reset buttons */}
      {tripped ? (
        <button type="button" className="btn ghost" style={{height:26, fontSize:11}} onClick={onReset}>Reset</button>
      ) : (
        <button type="button" className="btn danger" style={{height:26, fontSize:11}} onClick={onTrip}>Stop alerts</button>
      )}
    </div>
  );
}
```

4. Add the kill-switch handler:
```javascript
const doKillswitch = async (action) => {
  try {
    const r = await apiJSON('/api/killswitch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
    showToast(action === 'trip' ? 'Kill-switch tripped' : 'Kill-switch reset');
    addLog('INFO', `kill-switch ${action} → ${JSON.stringify(r)}`);
  } catch(e) {
    showToast('Kill-switch failed — auth header set?');
  }
};
```

5. Render it between `TickerTape` and the engine status bar:
```jsx
<TickerTape live={live}/>
{on && metrics && (
  <RiskBanner metrics={metrics}
              onTrip={() => doKillswitch('trip')}
              onReset={() => doKillswitch('reset')}/>
)}
{/* engine status bar... */}
```

**Why**: Without this, all the new safety/cost work is invisible. The banner makes the kill-switch state, today's net P&L vs limit, trade count vs cap, and AI/candle cache hit rates visible at a glance.

**Verify**: With engine on and `/api/metrics` responding, banner appears between ticker and engine-status row. Loss progress bar fills as trades close. "Stop alerts" button trips kill-switch instantly.

---

---

## 14 · Real-time option SL/T1/T2 (no more delta-scaled estimates)

**Problem**: Currently `opt_sl`, `opt_t1`, `opt_t2` are computed as `entry ± (index_distance × delta)`. This is a linear approximation — it ignores gamma, theta, vega. Result: displayed exit levels don't match what the option actually trades at when hit. Users complain "strikes are off."

**File**: `server.py` — `OptPicker.pick()` (~line 2363-2365), `SignalGen.analyze` (where index SL/T1 are set), `PLTracker.check()`.

**Action**:

1. Add CONFIG keys:
```python
"opt_sl_pct":  float(os.environ.get("OPT_SL_PCT",  "0.35")),  # 35% premium loss = stop
"opt_t1_pct":  float(os.environ.get("OPT_T1_PCT",  "0.50")),  # 50% premium gain = T1
"opt_t2_pct":  float(os.environ.get("OPT_T2_PCT",  "1.00")),  # 100% premium gain = T2
"opt_exit_mode": os.environ.get("OPT_EXIT_MODE", "premium_pct"),  # "premium_pct" | "delta_scaled"
```

2. In `OptPicker.pick()`, replace the delta-scaled exit computation:
```python
# OLD (delta-scaled — estimated):
# sl = round(max(e - idx_to_sl * d, e * 0.65), 2)
# t1 = round(e + idx_to_t1 * d, 2)
# t2 = round(e + idx_to_t2 * d, 2)

# NEW (premium-percentage — exact market prices):
exit_mode = CONFIG.get("opt_exit_mode", "premium_pct")
if exit_mode == "premium_pct":
    sl = round(e * (1 - CONFIG.get("opt_sl_pct", 0.35)), 2)
    t1 = round(e * (1 + CONFIG.get("opt_t1_pct", 0.50)), 2)
    t2 = round(e * (1 + CONFIG.get("opt_t2_pct", 1.00)), 2)
else:
    # Legacy delta-scaled path (kept as fallback)
    sl = round(max(e - idx_to_sl * d, e * 0.65), 2)
    t1 = round(e + idx_to_t1 * d, 2)
    t2 = round(e + idx_to_t2 * d, 2)
```

3. Also compute and persist BOTH exit modes when the signal is saved, so the dashboard can show "Premium SL ₹40 (exact) / Index SL 25500 (modelled)" side by side. Reuses existing `option_target1`, `option_sl` columns for the active mode; add two new optional columns `option_sl_index_modelled`, `option_t1_index_modelled` if you want to display both.

4. **Exit detection stays the same** — `PLTracker.check()` already compares live `cur_opt` to `opt_sl/opt_t1`. With premium-pct mode, the exit will happen at exactly the displayed levels (or slightly after, depending on tick spacing).

**Why**: Eliminates the "shown SL doesn't match what the option trades at" problem entirely. Premium-percentage levels are exact market prices, not model output. Standard for retail options trading.

**Verify**: New signal arrives, dashboard shows SL=₹40 on a ₹61.5 entry. Force the option to drop. Confirm exit fires at exactly ₹40 (±1 tick), not earlier/later.

---

## 15 · Strict-greeks mode (no signal on estimated delta)

**Problem**: When Angel's `getOptionGreek` API fails or returns garbage, `fallback_delta()` at [server.py:74-87](server.py#L74-L87) substitutes a ladder estimate. This delta is used to compute SL/T1/T2 (in delta-scaled mode) AND scoring. Estimated delta → estimated everything. Users complain "I want real, not assumed."

**Files**: `server.py` — `OptPicker.pick()` (~line 2256-2281), `/api/option-ltp` route (~line 4180-4205).

**Action**:

1. Add CONFIG key:
```python
"strict_greeks": os.environ.get("STRICT_GREEKS", "false").lower() == "true",
```

2. In `OptPicker.pick()`, after the greeks lookup:
```python
if delta is None:
    if CONFIG.get("strict_greeks", False):
        log.warning(f"  STRICT_GREEKS: no live delta for {o.get('symbol')} — rejecting candidate")
        continue  # skip this candidate entirely
    # Else: use fallback ladder (existing behavior)
    dte = ...
    delta = fallback_delta(moneyness, dte=dte, right_side=right_side)
```

3. If ALL candidates get rejected due to strict-greeks, return `None` from `pick()` (no option = no trade).

4. Same change in `/api/option-ltp` for parity.

5. Surface `delta_source` ("live" vs "fallback") on the dashboard signal card so the user can see at a glance which signals are using real greeks vs estimates.

**Why**: When greeks API misbehaves (it does — Angel One's option chain is flaky during volatile periods), strict mode prevents the engine from generating signals on assumed values. Trade-off: fewer signals when greeks API is down, but every signal has real delta. User can choose: set `STRICT_GREEKS=false` (default, more signals, some estimated) or `STRICT_GREEKS=true` (only real-greek signals).

**Verify**: Set `STRICT_GREEKS=true`. Force greeks endpoint to fail (network block / wrong expiry). No new signals fire. Engine log shows `STRICT_GREEKS: no live delta...rejecting candidate`. Set back to `false`. Signals resume with `delta_source: "fallback"`.

---

## 16 · Real backtest with "missed opportunities" analysis

**Problem**: Current `backtest.py` only reports on signals that PASSED filters. It can't tell you "what trades did the engine SKIP that would have won?" — the most important question for tuning filters.

**File**: `backtest.py` (extend, ~150 new lines on top of existing 340).

**Action**:

1. Restructure the bar loop to capture BOTH paths:
```python
for i in range(30, len(df) - 24):
    slice_df = df.iloc[:i+1].copy()
    sig = sgen.analyze(slice_df)
    if sig is None: continue   # truly no signal at all

    # Classify which filters this signal would hit
    filter_reasons = []
    if sig["confidence"] < CONFIG.get("min_confidence", 45):
        filter_reasons.append("LOW_CONFIDENCE")
    if sig.get("risk_reward", 0) < 1.5:
        filter_reasons.append("LOW_RR")
    try:
        ts_py = pd.Timestamp(slice_df["timestamp"].iloc[-1]).to_pydatetime()
        hr, mn = ts_py.hour, ts_py.minute
        if hr >= 15 or (hr == 14 and mn >= 50):
            filter_reasons.append("LATE_DAY")
    except: pass

    would_be_taken = (len(filter_reasons) == 0)

    # Simulate the trade EITHER WAY (don't skip filtered signals)
    # ... pick strike, estimate premium, walk forward, classify WIN/LOSS ...
    result, exit_price, bars = simulate_trade(...)
    won = result in ("WIN", "WIN_TIMEOUT")

    # Classify into the 4 buckets
    if would_be_taken and won:        bucket = "TAKEN_WIN"
    elif would_be_taken and not won:  bucket = "TAKEN_LOSS"
    elif not would_be_taken and won:  bucket = "FILTERED_WIN"      # ← missed opportunity
    else:                              bucket = "FILTERED_LOSS"   # ← filter worked

    trades.append({
        **base_trade_fields,
        "bucket": bucket,
        "filtered_by": ",".join(filter_reasons) if filter_reasons else None,
    })
```

2. Update `_summarise()` to report by bucket AND by filter:
```python
def _summarise(...):
    df = pd.DataFrame(trades)
    taken_win    = df[df["bucket"] == "TAKEN_WIN"]
    taken_loss   = df[df["bucket"] == "TAKEN_LOSS"]
    filt_win     = df[df["bucket"] == "FILTERED_WIN"]    # missed opportunities
    filt_loss    = df[df["bucket"] == "FILTERED_LOSS"]   # filters that worked

    print(f"\n══ TAKEN (engine alerted) ══")
    print(f"  Trades: {len(taken_win) + len(taken_loss)}  win {len(taken_win)}  loss {len(taken_loss)}")
    print(f"  Net P&L: ₹{(taken_win['net_pnl'].sum() + taken_loss['net_pnl'].sum()):,.0f}")

    print(f"\n══ FILTERED — MISSED OPPORTUNITIES ══")
    print(f"  Trades you SKIPPED that WOULD HAVE WON: {len(filt_win)}")
    print(f"  Hypothetical P&L missed: ₹{filt_win['net_pnl'].sum():,.0f}")
    print(f"  Top filters causing misses:")
    for reason, count in filt_win["filtered_by"].value_counts().items():
        print(f"    {reason}: {count} winners filtered")

    print(f"\n══ FILTERED — CORRECT REJECTS ══")
    print(f"  Trades you SKIPPED that WOULD HAVE LOST: {len(filt_loss)}")
    print(f"  Loss avoided: ₹{abs(filt_loss['net_pnl'].sum()):,.0f}")

    print(f"\n══ NET FILTER VALUE ══")
    net_filter = abs(filt_loss['net_pnl'].sum()) - filt_win['net_pnl'].sum()
    print(f"  Filters saved ₹{abs(filt_loss['net_pnl'].sum()):,.0f} on bad trades")
    print(f"  But cost ₹{filt_win['net_pnl'].sum():,.0f} on missed winners")
    print(f"  Net: {'+' if net_filter > 0 else ''}₹{net_filter:,.0f} (positive = filters help, negative = filters hurt)")
```

3. New CLI flag `--show-missed` that prints a list of the top 10 missed-opportunity trades with timestamps, so user can spot-check the data:
```python
if args.show_missed:
    top_misses = df[df["bucket"] == "FILTERED_WIN"].nlargest(10, "net_pnl")
    print("\nTop 10 missed opportunities:")
    for _, t in top_misses.iterrows():
        print(f"  {t['timestamp']}  {t['instrument']} {t['direction']}  "
              f"conf={t['confidence']}  filtered_by={t['filtered_by']}  "
              f"would-have-won ₹{t['net_pnl']:,.0f}")
```

4. Usage:
```bash
python3 backtest.py --instrument NIFTY --days 10 --show-missed
python3 backtest.py --all --days 10 --csv full.csv --show-missed
```

**Why**: Now you can answer "are my filters helping or hurting?" — the single most useful question for tuning. Lets you find:
- A confidence floor that's too high (filtering winners)
- An R:R gate that's too strict
- Time windows where signals would have worked but the filter blocked them

**Caveat to document**: Just like the basic backtest, option premiums are still estimated (no historical option chain from Angel). Win/loss CLASSIFICATIONS are reliable; absolute rupee figures are ±20%.

**Verify**: Run `python3 backtest.py --instrument NIFTY --days 10 --show-missed`. Output shows the 4 buckets and a per-filter breakdown of missed winners.

---

## Verification checklist (after all 16 are deployed)

```bash
# 1. Server boots
curl https://<your-railway-url>/api/ping
# → {"ok": true, ...}

# 2. New endpoint exists
curl https://<your-railway-url>/api/metrics
# → JSON with engine, cache, risk, metrics_today, perf_today keys

# 3. Caching engaged (after a few alerts)
curl -s https://<your-railway-url>/api/metrics | python3 -c "import sys, json; d=json.load(sys.stdin); a=d['cache']['anthropic']; print(f\"AI calls: {a['calls']}, cache_read: {a['cache_read_tokens']}, ratio: {a['cache_read_tokens']/(a['cache_read_tokens']+a['input_tokens']+0.001):.0%}\")"

# 4. Candle cache engaged (within 10 min)
# /api/metrics → cache.candles.hit_rate_pct ≥ 70

# 5. Layer C wired (after 1 trading day)
# /api/metrics → engine.weight_adjustments populated with prior day's deltas

# 6. Kill-switch responsive
curl -X POST -H "X-Auth-Token: <yours>" -H "Content-Type: application/json" \
  -d '{"action":"trip"}' https://<your-railway-url>/api/killswitch
# → {"ok": true, "tripped": true}; Slack DM arrives

# 7. Backtest produces results
python3 backtest.py --instrument NIFTY --days 30
# → win rate, gross/net P&L, expectancy printed
```

## Per-step test plan

| Step | What to test before moving to the next |
|------|----------------------------------------|
| 0 | `https://<app>/api/ping` returns 200 |
| 1 | New CONFIG keys readable; `.env.example` updated |
| 2 | Close one trade; new columns populated; net < gross |
| 3 | After 5 min, `/api/metrics` cache.candles.hit_rate_pct > 0 |
| 4 | (covered by 1) |
| 5 | Engine log: only one chain-fetch log per alert (was two) |
| 6 | At 15:15 IST: engine logs "Auto-close at 15:15" + EOD line |
| 7 | Force loss > limit → next signal logs `🛑 KILL-SWITCH:` |
| 8 | `/api/metrics` returns full JSON; counters increment |
| 9 | `/api/metrics` cache.anthropic.cache_read_tokens > 0 after alerts |
| 10 | `/api/metrics` cache.anthropic.by_layer.inflight populated when open positions exist |
| 11 | (Day 2) Engine log shows `🧠 Layer C feedback loaded from <yesterday>` |
| 12 | `python3 backtest.py --instrument NIFTY --days 7` runs |
| 13 | Dashboard shows colored banner between ticker and engine status |
| 14 | New signal: dashboard shows SL=₹40, force option to ₹40, exit fires at exactly ₹40 (±1 tick) |
| 15 | `STRICT_GREEKS=true` + greeks endpoint blocked → no new signals + log line `STRICT_GREEKS: rejecting candidate` |
| 16 | `python3 backtest.py --instrument NIFTY --days 10 --show-missed` outputs 4 buckets + missed-winners list |

## Rollback plan

After each push, if anything breaks: `git revert <hash> && git push origin main`. Railway auto-redeploys the revert in 2-3 min.

If you want a permanent safety snapshot before starting:
```bash
git tag -a snapshot-pre-overhaul-$(date +%Y%m%d) -m "Snapshot before overhaul"
git push origin --tags
```
Roll back to it any time with: `git checkout snapshot-pre-overhaul-<date>` then re-push.

## Caveats your assistant should know

1. **Sonnet 4.5 caching minimum is 1024 tokens**, Haiku 4.5 minimum is 4096. Layer D's prompt is small (~150 tokens), so caching there will silently no-op — that's fine, the cost saving comes from being on Haiku.
2. **The backtest's option premiums are estimated** because Angel One has no historical option-chain API. Use win-rate/expectancy direction as the edge signal, not absolute rupee figures.
3. **Schema migrations are additive** (`ALTER TABLE ... ADD COLUMN`). Old rows stay valid with NULL in new columns. The `r.get(col)` pattern in `get_perf` handles this.
4. **First-day Layer C is empty** — `daily_adjustments` only populates after the first EOD run. Don't expect weight_adjustments before day 2.
5. **`min_machines_running = 1`** if migrating to Fly.io to keep the engine warm 24/7. On Railway free tier, the only equivalent is to ensure no idle-stop is configured (and stay on the Hobby plan, not free trial).
6. **DO NOT push all 13 at once.** Push 1-3 changes at a time, verify Railway stays green, move to the next. The cost-paid-once is the slow first Dockerfile build (~3-5 min); subsequent rebuilds with cached layers are fast.

## Total scope (rough)

- `server.py`: ~750 lines added/modified across 14 of the 16 changes
- `index.html`: ~150 lines added (RiskBanner + state + handler)
- `backtest.py`: ~340 lines initial + ~150 for bucketed missed-opportunity analysis (step 16)
- `Dockerfile`, `railway.json`, `.env.example`, `README.md`: small

Compiles and runs locally with `python3 server.py` (dummy env vars OK; Angel login fails gracefully, Flask still serves).
