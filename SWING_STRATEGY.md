# Swing Pullback v1 — "buy fear inside strength"

Built 2026-07-29 from a four-track deep-research pass (~90 primary sources:
SSRN/academic papers, NSE index factsheets, multi-decade published backtests,
exchange rules). This document is the strategy of record; the implementation
lives in `server.py` (`SwingPullback`, `SwingEngine._scan_pullback`,
`_admit_pullback_candidates`, `_pullback_strength_exit`).

## The one-line thesis

**Momentum picks the stock, fear times the entry, strength takes the exit.**
Long-only, executed with slightly-ITM monthly options, at most 4 positions,
only when the market regime allows.

## Why this exact architecture (evidence summary)

| Layer | Rule | Evidence |
|---|---|---|
| Universe | Stock beats NIFTY over 126 days | Indian momentum validated academically (Sehgal & Jain 2011; SSRN studies on the NSE F&O universe) and in 18.5-year survivorship-free backtests (14-18% CAGR). **STRONG** |
| Universe | Close within 15% of 52-week high | The 52-week-high effect is separately validated on Indian data 2004-2023 (Raju 2023, SSRN 4587697) and is more stable than plain momentum. **STRONG** |
| Trigger | Uptrend intact (close > 200SMA and > 100SMA) + sharp 2-5 day pullback: RSI(2) < 10 OR 7-day closing low OR 3 lower closes | Connors/Alvarez dip-buy family: 65-83% win rates, 2-5 day holds, hundreds of thousands of trades; Alvarez's Russell-1000 variant 22.35% CAGR. Mean reversion dominates at <1 month horizons; momentum at 3-12 months — so momentum filters, reversion triggers. **STRONG (US mechanism), MODERATE (India)** |
| Long-only | No PE side in v1 | Momentum crashes live in the short leg (Daniel-Moskowitz); Indian momentum profit is documented almost entirely long-side; India's market structure (SLB ~7% of names) starves shorts. **STRONG** |
| Regime | NIFTY > 200SMA with rising 50SMA, else no new entries | Faber-class trend filter: similar returns at roughly half the volatility; ~91% of extreme days occur below the 200DMA. **STRONG** |
| Regime | India VIX ≤ 20 full size · 20-25 flagged half-size · >25 no entries | Nifty variance risk premium is persistently positive (option buyers systematically overpay, most when IV is high); momentum crashes cluster in high-vol states. **MODERATE** |
| Regime | Macro event blackout (Budget/RBI/elections via events.json) | Budget day averages a 2.65% intraday range over 25 years; election day 2024 was -5.93% with VIX +96% into it. **STRONG** |
| Selectivity | Rank all candidates by score, admit only into ≤4 free slots, never queue stale signals | Zarattini et al: restricting to top-ranked candidates was the source of the edge (Sharpe 2.81); Turtle correlation caps; NSE F&O longs are one correlated NIFTY bet. **STRONG** |
| Sizing | Skip signals whose 1-lot premium > 10% of account (SWING_ACCOUNT_CAPITAL) | 1-2% risk per trade is the documented standard; 1-lot minimums make ~5% realized risk (after the -50% premium floor) the achievable floor — cap it, don't pretend it's smaller. **STRONG (risk-of-ruin math)** |
| Vehicle | Slightly ITM (~0.6-0.7 delta) monthly, ≥15 DTE at entry | Cuts flat-week theta bleed ~30% vs ATM while capturing ~0.7x of the spot move; OTM dies on a slow win. **MODERATE + options math** |
| Exit | First strength: close > 5SMA or RSI(2) > 65 | The documented profit-taking exit for this entry family. **STRONG** |
| Exit | Stagnation stop: < +25% est premium after 5 sessions | Flat = losing to theta for an option buyer; stagnation stops are the cheapest theta defense. **MODERATE** |
| Exit | -50% premium backstop; 2×ATR spot disaster floor; 10-session max hold; T2 windfall at +5 ATR | Premium floor is structural to the vehicle; hard 5-day time exits amputate the right tail (Davey, 567k backtests), hence 10 sessions. **STRONG/MODERATE** |

## The full rule set (as implemented)

```
REGIME (checked once per cycle, ~30 min cache)
  NIFTY close > 200SMA  AND  50SMA higher than 21 sessions ago
  India VIX <= 25 (<= 20 for unflagged size)
  No macro event blackout (events.json)
  → else: NO new entries. Exits always run.

PER STOCK (daily candles, 400 days)
  126d return > NIFTY 126d return          (relative strength)
  close >= 0.85 x 252-day high             (52-week-high proximity)
  close > 200SMA AND close > 100SMA        (uptrend intact)
  AND at least one pullback trigger:
      RSI(2) < 10 · 7-day closing low · 3 consecutive lower closes

RANK & ADMIT
  score = (stock_126d - nifty_126d) x 100 + (proximity - 0.85) x 40
  slots = SWING_MAX_OPEN(4) - open paper positions
  admit top-scored candidates only; discard the rest (no queueing)
  skip if 1-lot premium > 10% x SWING_ACCOUNT_CAPITAL(1.5L)
  dedupe: one open position per instrument+direction

VEHICLE
  monthly option, >= 15 DTE, one strike ITM (~0.6-0.7 delta)
  lot size from the exchange instrument master

EXITS (first to trigger wins)
  STRENGTH_EXIT      close > 5SMA or RSI(2) > 65      ← the intended profit exit
  T2_HIT             spot +5 ATR (windfall)
  T1_HIT             spot +3 ATR
  SL_HIT             spot -2 ATR (disaster floor, not a trading stop)
  PREMIUM_BACKSTOP   est premium <= 50% of entry
  STAGNATION         < +25% est premium after 5 sessions
  MAX_HOLD           10 sessions
```

## What to expect (honest numbers)

- The entry family is a HIGH win rate / SMALL average win profile: expect
  55-75% winners with most exits in 2-6 sessions, not lottery tickets.
- The regime gate will keep the system flat for weeks at a time in
  downtrends. A quiet system in a falling market is the strategy working.
- Paper-track ≥ 30 closed trades before trusting it with money; ≥ 100
  before touching any parameter. The gate stats to watch: win rate by exit
  reason (STRENGTH_EXIT should dominate wins) and the stagnation rate.
- Known estimate weaknesses: option P&L falls back to a delta-0.5 model
  when a live quote is unavailable (real quotes are primary since v1.0.1).

## v1.1 additions (2026-08-04)

| Layer | Rule | Evidence |
|---|---|---|
| Regime | Breadth gate: no entries when < 30% of the universe holds its 50DMA (computed free from the scan's own fetches) | Market-internals confirmation. **MODERATE** |
| Risk | Sector cap: max 2 open positions per sector (static NSE-sector map) | Turtle correlation caps — same-sector longs are one doubled bet. **STRONG** |
| Risk | Loser re-entry cooldown: 5 sessions per name after a stopped-out loss | Churn control on broken charts. **MODERATE** |
| Entry | IV-percentile gate: skip when the picked strike's IV sits above the 70th pctile of the stock's own recorded history (self-calibrating, activates at 20 observations) | Variance-risk-premium literature: buyers overpay most when IV is rich. **MODERATE** |
| Entry+Exit | Earnings blackout: BSE Corpforthresults calendar (NSE's is geo-blocked from cloud IPs), refreshed daily — no entries within T-3 of results, open positions force-close at T-1 (EARNINGS_EXIT) | IV crush + unhedgeable gap risk around results. **STRONG** |
| Telemetry | Per-exit-reason win/P&L breakdown in /api/swing/results and the app | The tuning dashboard this doc calls for. |

## Environment knobs

`SWING_STRATEGY` (pullback|legacy) · `SWING_MAX_OPEN` · `SWING_ACCOUNT_CAPITAL`
· `PULLBACK_MAX_PREMIUM_FRAC` · `PULLBACK_MIN_52W_PROX` · `PULLBACK_RSI2_MAX`
· `PULLBACK_VIX_FULL` · `PULLBACK_VIX_MAX` · `PULLBACK_MAX_HOLD_DAYS`
· `PULLBACK_STAG_DAYS` · `PULLBACK_STAG_MIN_GAIN_PCT`

## v1.2 changes (2026-08-12) — backtest-verified against 3y of real NSE bars

A deep-research pass (Alvarez stop/exit/entry studies, Quantitativo cumulative
RSI2, Chui et al. 2023 Pacific-Basin on Indian momentum-vs-reversal) produced
candidate changes; each was tested on a 746-trading-day walk-forward over the
same 76-symbol universe before adoption. Baseline (v1.1 rules, spot logic):
285 trades, 60.7% win, payoff 0.62, SUM -14.7%.

| Change | Rule | Own-backtest evidence |
|---|---|---|
| Entry | Trigger = 2-day cumulative RSI(2) < 10; the OR-triggers (7-day low, 3 lower closes) dropped | Alone: 96 trades, 66.7% win, payoff 0.88, SUM +62.1%. **The single biggest fix.** |
| Entry | Limit entry at signal close - 0.5x ATR(10), valid through the next session, unfilled = no trade (PENDING/CANCELLED rows) | Combo evidence below; Alvarez published avg P/L per trade doubling. |
| Exit | Spot SL disabled for pullback rows (premium backstop, strength, stagnation, time, earnings exits all stay) | Baseline SL bucket was -171%; combo without it: 46 trades, 73.9% win, payoff 0.99, SUM +57.2%. |
| Regime | 3-consecutive-close confirmation on the 200SMA; rising-50SMA condition dropped | 352 regime-ON days vs 259 (+36% tradeable days); combo still 71% win, SUM +48.4%. |
| A/B control | 20-day-high breakout entry (momentum continuation) tested against the improved pullback | +19.1% over 150 trades — positive but far weaker; the pullback frame stays. |

New knobs: `PULLBACK_ENTRY_MODE` (cumrsi|legacy) · `PULLBACK_CUMRSI2_MAX` (10)
· `PULLBACK_LIMIT_ENTRY` (on|off) · `PULLBACK_SPOT_SL` (off|on)
· `SWING_REGIME_MODE` (confirm3|legacy)

Known gap the backtest cannot see (research finding, not yet acted on): NSE
single-stock option spreads + theta on a 5-day hold may consume much of a
+1.66%-avg-win edge; recommendation on live capital is options only for
indices (delta >= 0.75) and cash equity for stock signals.

## v1.3 (2026-09-04) — "can we trade below the 200SMA?" answered with data

Owner question after a month of zero entries with NIFTY ~3% under its 200SMA.
Same 746-day walk-forward, same v1.2 rules, only the regime gate varied:

| Gate | Trades | Win | Payoff | Exp/trade | SUM | Max DD |
|---|---|---|---|---|---|---|
| v1.2 confirm-3 200SMA (shipped) | 51 | 70.6% | 0.73 | +0.56% | +28.5% | **-5.8%** |
| NO index gate | 74 | 73.0% | 0.53 | +0.37% | +27.2% | **-25.4%** |
| ONLY the gate-OFF periods | 23 | 78.3% | 0.26 | -0.06% | -1.4% | -20.6% |
| Gate-OFF periods at half size | 74 | 73.0% | 0.57 | +0.38% | +27.8% | -15.4% |
| confirm-3 + breadth-thrust override | 56 | 71.4% | 0.73 | +0.58% | **+32.2%** | **-5.8%** |

Reading: the 23 extra trades taken below the 200SMA win 78% of the time and
still lose money — avg loss -5.1% vs avg win +1.3%. Trading through the gate
gives the same total for 4x the drawdown. Half-sizing doesn't fix it. The
one thing that adds trades WITHOUT adding drawdown is the Zweig-style
breadth thrust: when universe breadth (% above own 50DMA) snaps from <30% to
>55% within 10 sessions, entries are allowed for the next 20 sessions
regardless of the 200SMA. Adopted.

Implementation: `swing_breadth` table (one reading per session, seeded from
`swing_breadth_backfill.json` on a cold start), `SwingEngine._breadth_thrust_active()`,
override applied inside `_swing_market_ctx()`; surfaced as `breadth_thrust`
in `/api/swing/results.regime` and in the regime note. Knobs:
`SWING_BREADTH_THRUST` (on|off) · `BREADTH_THRUST_LOW` (30) ·
`BREADTH_THRUST_HIGH` (55) · `BREADTH_THRUST_HOLD_DAYS` (20).

Practitioner consensus the data agrees with: Connors/Alvarez require price
above the 200SMA for dip-buys; ~90% of extreme days occur below it; the
sanctioned early re-entry is a breadth thrust, not "buy dips anyway".

### v1.3 addendum — tiered exposure (2026-09-04, two research passes: 30+ quant sources, 30+ practitioner sources)

Both passes converged on the same verdict: nobody credible runs a long-only
stock system on a binary index-200DMA switch; the split is breadth-based bias
with pilot positions (Nitin R / Stockbee / Minervini), stock-level rules only
(Weekend Investing, Capitalmind), or RS names at reduced size (TradingQnA
regulars). War stories: the 200DMA reclaim arrives ~11% off the low in 2022
and 2025, and never arrived at all in 2026 (100+ sessions) while ~50% of
stocks sat above their own 200DMA and printed 52-week highs.

Own backtest of the tier design (same 746 days, v1.2 rules):

| Regime | Trades | Win | Exp/trade | SUM | Max DD |
|---|---|---|---|---|---|
| Hard gate (v1.2) | 51 | 70.6% | +0.56% | +28.5% | -5.8% |
| Gate + breadth thrust | 56 | 71.4% | +0.58% | +32.2% | -5.8% |
| Tiered A/B, half size in B | 56 | 71.4% | +0.55% | +31.0% | -6.4% |
| Tiered A/B, full size in B | 56 | 71.4% | +0.60% | +33.6% | -7.3% |
| Tier-B trades alone (half size) | 5 | 80% | +0.32% | +1.6% | -1.6% |

Reading: with the per-stock filters (own 100/200SMA, beats NIFTY, near 52w
high, 2-day washout) already emptying the screen in a weak tape, opening the
index gate adds only a handful of trades, and they are small positives. The
gate was mostly redundant with the stock filters; the tier design keeps the
drawdown profile while letting the system participate in bear-market rallies
led by relative-strength names.

Shipped rules (`_swing_market_ctx`, env `SWING_TIERED_REGIME` on|off):
- **Tier A** (4 slots, full): NIFTY > 200SMA confirm-3, OR breadth thrust, OR
  B200 >= 60% (`SWING_TIER_A_B200`).
- **Tier B** (2 slots, tagged HALF SIZE, `SWING_TIER_B_MAX_OPEN`): B200 >= 40%
  (`SWING_TIER_B_B200`) AND NIFTY above its 20EMA two sessions; or capitulation
  (B200 < 20% within 20 sessions) with NIFTY back above the 20EMA once.
- **Tier C**: flat. B200 = % of the universe above its own 200DMA, one reading
  per session in `swing_breadth.pct200` (seeded from the backfill).
- Tier is stored on each position (`indicators.tier`), shown in the regime
  pill and the Slack alert; `/api/swing/results.regime` exposes `tier`, `b200`.

State on 2026-09-04: B200 46.7% (Tier-B eligible), NIFTY 23,873 vs 20EMA
24,165 → Tier C. Tier B opens on two closes above the 20EMA (~+1.2%), versus
~+3.1% to the 200SMA.

Research items deliberately NOT shipped (no backtest support or out of scope
for a paper tracker): ATR-based position sizing (1-lot floor), futures/cash
instead of stock options, VIX-above-10dMA entry qualifier for Tier B (India
VIX history not fetched), sector-index regime, F&O ban / MWPL check, gap-cancel
on resting limits. The last three are execution hygiene worth adding before
real capital.

Intraday side, same session: EOD learning-loop time blocks clamped (one
window, <= 30 min, never before 10:30; it had blocked 10:00-11:00 +
13:45-14:15 off a 3-trade day). MAX_OPEN_POSITIONS=1 in the Railway env
blocked 40 signals in 21 days, many at 95% confidence — operator decision.
