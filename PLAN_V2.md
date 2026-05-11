# Intraday Signal Engine — Foolproof Plan v2

**Authored after researching 5 strategy families in parallel and reviewing the strongest public Indian-market reference implementations.**

This plan replaces the indicator-soup signal generation in v1 with a single, mechanically defined, backtest-verifiable strategy — and bolts on a real backtest harness that produces a per-trade CSV with exact dates, exact option symbols, and exact prices from historical chain data.

---

## The honest priors (read first)

- SEBI Sept-2024 study: **93% of individual F&O traders lost money FY22-FY24** ([source](https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html)). Aggregate loss ₹1.8 lakh crore.
- FY25 update: **91% lost** (~₹1.06 lakh crore).
- Post-SEBI Nov-2024 changes:
  - Only NIFTY has weekly expiry (Tuesday). BANKNIFTY/FINNIFTY weeklies discontinued.
  - Calendar-spread margin benefit removed.
  - 2% additional ELM (Extreme Loss Margin) on expiry-day shorts.
  - Many pre-2024 backtests on these instruments are partially invalidated.
- **The plan is built around being in the 7-9%, not the 91-93%.** That means: strict rules, real backtests, real costs, real kill-switches, and a willingness to NOT trade when conditions don't match.

---

## The strategy: Trend-Momentum Option Buying

Selected after parallel research on 5 candidates. Full scorecard:

| Strategy | Score | Why ruled out |
|---|---|---|
| Short Straddle/Strangle (theta) | 3/10 | ₹1.5L+ margin/lot. Not viable for ₹20-50K. |
| **Trend-Momentum Option Buying** | **6.5/10** ⭐ | Only candidate with reproducible Indian-market tick-level backtest. |
| Opening Range Breakout | 5/10 | Edge halved by option-buying overlay. 48% WR best public 4yr backtest. |
| IV-Percentile Vol Plays | 6/10 | Needs ₹2-3L+ account for hedged version. Complex. |
| VWAP+RSI Confluence | 4/10 | No credible long-horizon backtest. Framework, not edge. |

### Why trend-momentum (specifically) wins

1. **Capital fits ₹20-50K account** — option buying needs only premium (₹4-10K/lot for NIFTY ATM weekly), not margin.
2. **Best public reference**: [`aaryansinha16/AI-trader`](https://github.com/aaryansinha16/AI-trader) — tick-level replay backtest with real bid/ask slippage. 18-day MEDIUM-profile run: 49 trades, 71% WR, +₹53,715 net, max DD ₹5,411. Sample is small but it's the cleanest public proof of concept on Indian options.
3. **Rules are mechanical**, not discretionary.
4. **Compatible with existing infrastructure** — Angel One feed, 5-min candles, dashboard. We rewrite the SignalGen layer, keep the OptPicker / Slack / PLTracker.

### Entry rule (exact)

On 5-min candle close of the spot (NIFTY or BANKNIFTY), score 4 binary checks:

**For CALL buy (vwap_momentum_breakout)**:
1. Close > session VWAP
2. RSI(14) > 55
3. EMA(20) > EMA(50)
4. Volume on the bar > 1.5× the 20-bar volume average

**For PUT buy (bearish_momentum)**:
1. Close < session VWAP
2. RSI(14) < 45
3. EMA(20) < EMA(50)
4. Volume > 1.5× 20-bar average

**Trigger**: at least 3 of 4 conditions met (`score ≥ 0.75`). All 4 = higher conviction = size up.

### Strike selection

- ATM only for current weekly expiry (NIFTY: Tuesday weekly; BANKNIFTY: monthly Tuesday post Nov 2024).
- Delta target: 0.40-0.60 (this is what ATM gives on weekly expiry).
- Skip strikes where spread > 5% of LTP (illiquid).

### Exit rules (exact)

- **Stop-loss**: 30% of entry premium (i.e., exit at `entry × 0.70`).
- **Target 1**: 50% gain on premium (`entry × 1.50`), exit 50% of position.
- **Target 2**: 100% gain on premium (`entry × 2.00`), exit remainder.
- **Trailing stop**: once price hits +12%, trail SL to lock +8%. Once at +50%, trail to lock +25%.
- **Time stop**: hard exit at 12:45 IST. No new entries after 12:30 IST.
- **Daily limit**: max 3 trades/day; max 2 losses/day = stop trading that day.

### Day-of-week / regime filters

- **Skip Monday** (weekend gap risk).
- **Skip days with India VIX > 22** (event days — RBI, Fed, Budget, elections).
- **Skip expiry day after 11:00 IST** (gamma + theta = unpredictable).
- Engine logs "REGIME_BLOCKED" with reason for any day skipped — visible on dashboard.

---

## The backtest (this is what the user asked for)

### What it must produce

For every signal generated (whether taken or filtered), a CSV row with:
- **Exact date** (YYYY-MM-DD)
- **Exact entry time** (HH:MM:SS IST)
- **Spot price at entry**
- **Exact option symbol** (e.g. `NIFTY07JAN26 25600 CE`)
- **Option entry price** — from real historical chain data, not estimated
- **Option exit price** — from real historical chain data at exit time
- **Exact exit time**
- **Exit reason** (TARGET / SL / TRAIL / TIME / END_OF_DAY)
- **Gross P&L** (option_exit - option_entry) × qty
- **Brokerage** (Zerodha/Angel typical: ₹40/leg)
- **Slippage** (modelled from real bid-ask spread at that timestamp)
- **Net P&L**
- **Filter bucket** (TAKEN_WIN / TAKEN_LOSS / FILTERED_WIN / FILTERED_LOSS — same as v1 step 16)
- **If filtered**: which filter (LOW_VIX_DAY / MONDAY / LATE_DAY / EXPIRY_AFTERNOON / etc.)

### Data sources (all free, real)

| Source | What it provides | URL |
|---|---|---|
| `aeron7/nifty-banknifty-intraday-data` | 1-min OHLCV NIFTY+BANKNIFTY spot from 2019 | [github.com/aeron7/nifty-banknifty-intraday-data](https://github.com/aeron7/nifty-banknifty-intraday-data) |
| `jugaad-data` (Python lib) | NSE historical option chain — OHLCV per strike per expiry | [github.com/jugaad-py/jugaad-data](https://github.com/jugaad-py/jugaad-data) |
| NSE F&O bhavcopy archives | EOD option settlement prices, OI, volume | [nseindia.com/all-reports](https://www.nseindia.com/all-reports) |
| Angel One `getCandleData` | 1-min candles for any NFO option token (live + historical) | (already in engine) |

### Backtest architecture (new `backtest_v2.py`)

```
PHASE 1: Data layer
  - Pull spot 1-min bars for backtest period from aeron7 dataset (one-time download, cached locally)
  - For each historical trading day:
    - Determine the active weekly expiry contract (e.g. NIFTY07JAN26)
    - Pull that day's option chain bhavcopy → cache
    - For each 5-min spot bar where a signal fires:
      - Look up the actual option premium at that timestamp from cached option chain
      - If chain data missing (rare), fall back to Black-Scholes from spot + IV (clearly marked in CSV)

PHASE 2: Replay loop
  - Walk every 5-min spot bar
  - Compute VWAP/RSI/EMA/volume on the rolling window
  - For each signal:
    - Apply filter chain → mark TAKEN or FILTERED + reason
    - Run forward simulation through next 5-min bars until SL/T1/T2/TIME
    - Record exact times, prices, fills

PHASE 3: Report
  - Per-trade CSV (the user's specific ask)
  - Aggregate report: total trades, win rate, expectancy net of costs,
    max DD, Sharpe, profit factor
  - Per-bucket breakdown (4 buckets from step 16)
  - Per-filter breakdown: "If you removed the LOW_VIX filter, you'd have
    +N more trades, ₹X more P&L, but Y% lower win rate"
  - HTML report with equity curve + trade markers on spot chart
```

### Go/no-go gates

The strategy goes live ONLY if the backtest shows:

- **Sample**: ≥ 6 months of trading days
- **Trades**: ≥ 50 (statistical significance floor)
- **Win rate**: ≥ 55% (over a 0DTE option-buying baseline of ~35%)
- **Expectancy net of costs**: ≥ +₹200/trade
- **Max drawdown**: ≤ 30% of capital
- **Profit factor**: ≥ 1.5
- **Sharpe** (annualized): ≥ 1.0

If any gate fails: the strategy doesn't go live. We iterate or pick another one.

---

## Implementation phases (with explicit go/no-go gates)

### Phase 0 — Audit current state (1 hour)

- Inventory what's already live (v c43547e) that we keep, modify, or delete.
- **KEEP**: AngelClient, OptPicker, PLTracker, Slack, kill-switch, /api/metrics, Dockerfile/Railway deploy
- **REWRITE**: SignalGen (13-indicator soup → 3-of-4 confluence)
- **REWRITE**: backtest.py (estimated → real chain data)
- **ADD**: regime filter (VIX, day-of-week, event calendar)
- **ADD**: data ingestion layer (jugaad-data, aeron7 cache)

**Gate**: confirm we're not breaking anything live before Phase 1.

### Phase 1 — Data layer (1 day of work, mostly offline)

1. Download `aeron7/nifty-banknifty-intraday-data` to a local `data/` directory.
2. `pip install jugaad-data`
3. Build `data_layer.py`:
   - `get_spot_bars(symbol, date_from, date_to, interval="1min")` — reads from cached files
   - `get_option_chain_eod(symbol, date)` — uses jugaad-data
   - `get_option_premium_at(symbol, strike, opt_type, expiry, timestamp)` — interpolates from EOD chain + spot intraday + Black-Scholes IV smile (when intraday option data unavailable)
4. Verify against a known day (e.g. last Tuesday's NIFTY expiry) that we can recover the exact ATM premium at 09:30, 10:00, 11:00 etc.

**Gate**: must be able to retrieve a real historical option premium for at least 90% of 5-min timestamps in a sample week. Output a `data_layer_verify.csv` with side-by-side: requested time, retrieved price, source (chain/interpolated).

### Phase 2 — Strategy v2 (`signal_v2.py`) (2 days)

1. Write `SignalGenV2.analyze(df_5min_spot)` implementing the 3-of-4 confluence rule.
2. Wire regime filter (`regime.py`):
   - VIX check (read from NSE daily history or cache from `IndiaVIX` token)
   - Day-of-week
   - Event calendar (`events.json` already exists)
   - Expiry-day window check
3. Unit-test against synthetic inputs (e.g., a known momentum bar → should fire CALL).
4. Run on the live engine in **DRY_RUN** mode for 3 trading days: signals fire to Slack + DB but no Layer-B (AI), no automatic trade. Observe quality of signals manually.

**Gate**: dry-run produces ≥ 5 signals over 3 days that look qualitatively reasonable. If it fires too much (>15/day) or too little (<2/day), tighten or loosen the confluence threshold.

### Phase 3 — Backtest v2 (`backtest_v2.py`) (2 days)

1. Build the bar-replay loop using the new SignalGen + data layer.
2. Output: per-trade CSV with the columns listed above.
3. Output: aggregate JSON report.
4. Output: HTML report with equity curve + trade markers (matplotlib + simple HTML wrapper).

**Run it**: 6 months of NIFTY weekly expiry data (e.g. Nov 2024 - Apr 2025 — post the SEBI regime change so results are forward-applicable).

**Gate**: ALL six metrics from "Go/no-go gates" section above must pass. If any fail, don't deploy. Iterate on filters or pick another strategy.

### Phase 4 — Walk-forward validation (1 day)

1. Take the Phase 3 results and split into 3 non-overlapping 2-month windows.
2. Compute metrics per window. They should all be positive (or at least 2 of 3).
3. If one window is a disaster, identify what regime it was (high vol? specific expiry? earnings season?) and add it as a filter.

**Gate**: at least 2 of 3 walk-forward windows show positive expectancy.

### Phase 5 — Live deployment (1 day)

1. Add a `STRATEGY=v2` env var on Railway. When set, engine uses `SignalGenV2` instead of v1.
2. Run for 5 trading days in **live but small** mode (₹10K capital, 1 lot max).
3. Compare live results to backtest expectations.

**Gate**: live results within ±50% of backtest expectations on win rate and average winner.

### Phase 6 — Scale up (ongoing)

Once Phase 5 passes:
- Gradually increase capital allocation
- Add the second strategy (e.g. iron-fly when account grows past ₹2L)
- Daily monitoring via `/api/metrics`
- Monthly review against backtest performance

---

## What we ship (file inventory for the implementer)

| File | New/Modify | Purpose |
|---|---|---|
| `data_layer.py` | new | jugaad-data + aeron7 cache + Black-Scholes fallback |
| `data/` | new directory | cached spot 1-min bars + option chain bhavcopy |
| `signal_v2.py` | new | The 3-of-4 confluence rule, wraps existing SignalGen |
| `regime.py` | new | VIX / day-of-week / event filters |
| `backtest_v2.py` | new | Real-chain backtest with per-trade CSV + HTML report |
| `server.py` | modify | Add `STRATEGY` env var, route to SignalGenV2 when v2 |
| `requirements.txt` | modify | Add jugaad-data, matplotlib |
| `events.json` | extend | Add 2025-2026 RBI/FOMC/Budget dates for regime filter |
| `index.html` | small modify | Add a "Strategy: v1/v2" indicator to the engine status bar |

Estimated total work: **~1 week** of focused implementation (Phases 1-5).

---

## What about the existing v1 (commit c43547e)?

It stays live. The `STRATEGY` env var defaults to `v1`. When you're confident in v2's backtest, you flip it to `v2` on Railway. If v2 misbehaves live, flip back. No code redeploy, just an env var change.

This gives us:
- A working baseline that doesn't get destroyed
- A clean A/B comparison once v2 is live
- A trivial rollback path

---

## Honest expectations

If the backtest passes all gates and live trades within 50% of backtest expectations, **realistic monthly returns on a ₹50K account**: -10% to +15%. Worst-case monthly drawdown: ~30%. Annual expectation if everything works: ~30-60% on capital, but with months that lose money.

**This is not "make 5% per day."** Anyone promising that is selling courses. The realistic path is small consistent edge compounded across many trades — IF the backtest is real and the strategy is followed mechanically.

---

## Sources (research backing this plan)

- [SEBI Sep-2024 study — 93% retail F&O lose money](https://www.sebi.gov.in/media-and-notifications/press-releases/sep-2024/updated-sebi-study-reveals-93-of-individual-traders-incurred-losses-in-equity-fando-between-fy22-and-fy24-aggregate-losses-exceed-1-8-lakh-crores-over-three-years_86906.html)
- [aaryansinha16/AI-trader — the closest public reference implementation](https://github.com/aaryansinha16/AI-trader)
- [jugaad-data — free Python lib for NSE historical](https://github.com/jugaad-py/jugaad-data)
- [aeron7/nifty-banknifty-intraday-data — 1-min historical bars](https://github.com/aeron7/nifty-banknifty-intraday-data)
- [Nomad Trader — 9:20 short straddle backtest](https://medium.com/@amit179.iitk2/optimized-920-straddle-strategy-to-get-more-than-80-return-annually-e977453dca80)
- [NSE Working Paper 9/2013 — India VIX vs realised vol](https://nsearchives.nseindia.com/research/content/res_WorkingPaper9.pdf)
- [Concretum Group — ORB clean implementation](https://concretumgroup.com/backtesting-the-opening-range-breakout-orb-strategy-using-polygon-io/)
- [tradingstats.net — ORB 2014-2025 study, no decay](https://tradingstats.net/orb-strategy-research/)
- [Zerodha In The Money — NIFTY ORB option-buying backtest](https://inthemoneybyzerodha.substack.com/p/how-to-trade-opening-range-breakout)
- [Sahi.com — VWAP scalping rules (most-cited retail spec)](https://www.sahi.com/blogs/vwap-scalping-strategy-for-nifty-and-bank-nifty-3-setups-that-work)
- [umeshpalai BANKNIFTY straddle backtest notebook](https://github.com/umeshpalai/Algorithmic-Trading---Backtesting---Banknifty-Straddle-using-Python)
