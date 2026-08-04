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
