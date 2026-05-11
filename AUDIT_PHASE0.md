# Phase 0 Audit — v1 → v2 Inventory

Date: 2026-05-11
Live commit: c43547e → 9ded3af (docs only)

## KEEP as-is (no v2 change needed)

| File | Why keep |
|---|---|
| `AngelClient` class in `server.py` | Live data feed works; candles + option chain + greeks API integrations are solid |
| `OptPicker.pick()` | Strike selection logic is sound; just feeds will change |
| `OptPicker.chain_analytics()` | PCR/OI/IV computation reusable |
| `PLTracker` | Premium-pct exit detection works |
| `SlackAlert` | Notification layer is independent |
| Database schema (`signals`, `regime`, `daily_adjustments`, `inflight_actions`, `swing_positions`) | All useful; v2 just adds new columns |
| Kill-switch + `/api/killswitch` + `/api/metrics` | Safety + telemetry layer is reusable |
| Dockerfile + railway.json + start.sh | Deployment infra is now stable |
| `index.html` (dashboard) | UI is independent of strategy — same alerts flow through |
| Auth (X-Auth-Token), CORS, `/api/ping`, `/api/status` | All independent of strategy |
| Layer A (RegimeBrief), Layer C (LearningLoop), Layer E (EventCalendar) | Operate on results, agnostic to which strategy generated them |

## MODIFY (v2 changes specific parts)

| File | Change | Why |
|---|---|---|
| `server.py` `SignalGen.analyze()` | Wrap with strategy selector: if `STRATEGY=v2`, route to `SignalGenV2`. v1 stays callable. | A/B switchable via env var, easy rollback |
| `server.py` CONFIG | Add `STRATEGY` env, plus v2-specific knobs (`V2_CONFLUENCE_MIN`, `V2_VIX_MAX`, `V2_DAY_OF_WEEK_BLOCK`) | Env-driven, no redeploy to tune |
| `server.py` Layer B (SignalValidation) | Optional — for v2 we may bypass AI validation since rules are mechanical | Reduces noise + cost; AI verdict can still log but not gate |
| `requirements.txt` | Add `jugaad-data`, `matplotlib` (for backtest reports) | Required for Phase 1 data layer + Phase 3 report |
| `events.json` | Extend with 2025-2026 RBI/FOMC/Budget dates | Drives the regime filter in v2 |
| `index.html` | Add tiny "Strategy: v1 / v2" tag in engine status bar | Visibility |

## DELETE (or deprecate)

| File | Why |
|---|---|
| `backtest.py` (the one I wrote that you didn't like) | Estimated premiums, no dates per trade. Replaced by `backtest_v2.py` |

I'll keep the file in git history (don't actually `rm`) — just rename to `backtest_v1_deprecated.py` so the new one can take the canonical name.

## ADD (new files in v2)

| File | Purpose | Phase |
|---|---|---|
| `data_layer.py` | jugaad-data + Angel One historical + Black-Scholes fallback. Returns real prices for any timestamp. | 1 |
| `verify_data_layer.py` | Standalone script that proves the data layer can recover real prices. Run this BEFORE trusting backtest. | 1 |
| `signal_v2.py` | `SignalGenV2.analyze()` — the 3-of-4 confluence rule | 2 |
| `regime.py` | `RegimeFilter.should_trade(now, symbol)` — VIX/day-of-week/event gate | 2 |
| `backtest_v2.py` | The real backtest: walks history, gets real option prices from data_layer, outputs per-trade CSV with full date+time+symbol+price detail | 3 |
| `data/` directory | Cached spot bars + bhavcopy + option chain pulls (gitignored except a small sample) | 1 |
| `reports/` directory | HTML + CSV outputs from backtest runs (gitignored) | 3 |

## Strategy switching mechanism

```python
# In server.py SignalGen.analyze()
def analyze(self, df, weight_adj=None, blocked_windows=None):
    strategy = CONFIG.get("strategy", "v1").lower()
    if strategy == "v2":
        from signal_v2 import SignalGenV2
        return SignalGenV2.analyze(df, regime=self._regime_filter)
    # v1 (existing) path unchanged
    return self._analyze_v1(df, weight_adj, blocked_windows)
```

Single env var flips strategies. Zero risk to live v1.

## What doesn't change in production until Phase 5 gate passes

- Live engine remains on v1 (commit c43547e behaviour) through all of Phase 1-4.
- v2 code is added to the codebase progressively but only invoked by `backtest_v2.py` and explicit DRY_RUN flag.
- Phase 5 deployment is a single env var flip (`STRATEGY=v2`) on Railway when backtest gates pass.

## Estimated work

| Phase | Files touched | Lines added (approx) | Risk to live |
|---|---|---|---|
| 0 (audit) | — | 0 | 0 |
| 1 (data layer) | data_layer.py, verify_data_layer.py, requirements.txt | ~500 | 0 — additive only |
| 2 (signal v2) | signal_v2.py, regime.py, events.json, server.py (small) | ~400 | Low — DRY_RUN flag prevents firing |
| 3 (backtest v2) | backtest_v2.py, reports/ | ~600 | 0 — runs offline |
| 4 (validation) | (no code, just runs) | 0 | 0 |
| 5 (deploy) | server.py (env switch) | ~20 | Medium — but rollback is one env var |

Phase 0 done.
