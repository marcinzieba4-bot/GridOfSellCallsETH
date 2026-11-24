# Grid of Sold Calls: ETH daily ATM calls with a break-even short-perp grid

This repo simulates the following strategy on real ETH data (Apr 2021 to Nov 2026) and in Monte Carlo.

> **Correction (study 3, below): studies 1 and 2 overstated option-selling profits.** They priced 1-day
> options from the 30-day DVOL index. Real Deribit 1-day ATM options traded at an average **60% IV vs DVOL 73%**,
> which is about the fair value implied by realised moves (62%). At real prices, selling daily ATM options lost about
> **6 bps of spot per option at mid** (7 bps at the bid), and every option-selling variant in studies 1–2 loses money.
> The re-run table is in [Study 3](#study-3-is-sell-straddles-after-up-candles-reliable). The comparisons *between*
> rules in studies 1–2 still hold. The absolute profit numbers do not.

## The strategy (as simulated)

Each day runs from 08:00 UTC to 08:00 UTC, which is when Deribit daily options expire.

1. **Every day, sell 1 ATM ETH call** (1-day expiry, strike = spot at 08:00).
2. **If yesterday's call lost money**, meaning settle > strike + premium, the loss is realised and
   a **short perp of 1 ETH** is opened at the settle price. Its **break-even (BE) = strike + premium**
   (less fees). That is the price where the perp's gain exactly repays the call loss.
   *Your example: strike 1700, settle 1800, premium 15, so loss 85, short at 1800, BE = 1715.*
3. **If a call made money while perps are open**, that profit is spent on **lifting a perp's BE**
   (1715 → 1730 after another 15 premium). Funding received on the shorts lifts the BE too.
   If the BE reaches the market price, the perp is closed right away.
4. **Each perp has a resting limit buy at its BE.** When price trades down to it, the perp closes and
   that loss cycle ends at exactly 0.
5. Over time this leaves a **grid of short perps whose break-evens keep rising**
   (see `results/fig2_grid_zoom.png`).

The perp to lift (step 3) is chosen by one of these rules: `nearest` (highest BE, the one closest to
closing; this is the default), `farthest`, `prorata` or `fifo`.

Real example from the backtest (perp #34): the call with strike 1867.3 and premium 43.6 settled at
2049. A short was opened at 2049 with BE 1909.3. Later call profits and funding lifted the BE
1909 → 1925 → 1974 → 2021, and it closed at 2020.9 on 2021-07-02 (cycle PnL 0).

## Assumptions

| Item | Value |
|---|---|
| Price | Binance ETHUSDT 1h candles, aggregated to 08:00→08:00 UTC days; settle = 08:00 price |
| Option price | Black-Scholes ATM with IV = **Deribit DVOL × 0.95** (selling a bit below mid); sensitivity 0.85 to 1.05 below |
| Option fee | Deribit 0.03% of underlying, capped at 12.5% of premium; no delivery fee on dailies |
| Perp | open: taker 0.05%; close at BE: maker limit at 0%; Binance ETHUSDT funding history (shorts receive positive funding) |
| BE touch | daily high/low (limit order at BE filled when the day's low ≤ BE) |
| Size | **1 ETH call per day**. All $ figures are per that unit and scale linearly |

## Headline results (Apr 2021 to Nov 2026, per 1 ETH call/day)

| | A: plain call selling (take every loss) | **B: your grid** | J: grid + stop at BE+10% |
|---|---:|---:|---:|
| Total PnL | +$7,223 | **+$10,902** | +$5,352 |
| PnL / year | +$1,280 | +$1,932 | +$948 |
| **Max drawdown** | **−$1,746** | **−$16,497** (Mar 2024) | −$3,354 |
| Max open short perps | 0 | 12 | 3 |
| Sharpe (daily, annualised) | **1.18** | 0.23 | 0.51 |
| 1-year windows that lost money | 7% | 29% | 30% |
| Worst 1-year window | −$568 | −$11,849 | −$1,829 |
| Bootstrap MC: **mean** 1-year PnL | **+$845** | **−$2,247** | +$432 |
| Bootstrap MC: median 1-year PnL | +$974 | +$1,979 | +$848 |
| Bootstrap MC: 1% worst year | −$2,800 | −$55,670 | −$6,618 |

Grid B facts: **612 perps opened and 611 closed at break-even (99.8% "win rate")**.
322 closed the same day and 534 within a week, but 34 were held more than 30 days and the longest
was held 381 days. The grid was non-empty on 71% of days.

Where grid B's extra profit came from (vs plain):
`funding +$5,668 − perp price PnL $1,206 − perp fees $840 = +$3,622`.
**On price alone, shorting ETH after each losing call lost money. All of the extra profit was funding.**

All tables are in [`results/RESULTS.md`](results/RESULTS.md). Charts:
`fig1_backtest.png` (price, grid, equity, open perps), `fig2_grid_zoom.png` (the 2023-24 grid
build-up), `fig3_rolling_windows.png`, `fig4_monte_carlo.png`.

## Conclusions: what you can take from this

1. **The break-even grid does not recover losses. It moves them into the tail.**
   Calls are sold the same way every day, so the call PnL is *identical* with or without the grid.
   The grid only adds a separate bet: "short ETH after an up-day, until it comes back".
   Premium spent on lifting a BE is given back when that perp closes earlier at a higher price, because every
   cycle ends at exactly 0. The result is the classic martingale profile: many small wins (99.8% of
   cycles), a higher *median*, and rare huge losses. In a strong rally (the Oct 2023 to Mar 2024 rally, the 2025 rally)
   the grid piled up 10 to 12 shorts and equity fell about $16k. In the bootstrap the grid's *mean* is negative
   (−$2.2k/yr) while plain selling stays positive (+$0.8k/yr).

2. **Your real edge is the price you sell the call at.** Plain PnL goes from **−$842** (selling at
   0.85×DVOL) to **+$15,289** (1.05×DVOL). Each 5% of IV is worth about ±$4k over the period (about ±$700/yr per
   ETH/day). Use limit orders at or above mid and avoid crossing the spread. That matters far more than the grid.

3. **The grid's profit was funding, and you can collect funding without the risk.** Long spot plus short
   perp (cash-and-carry) earns the same funding (about 8%/yr on notional historically) with no directional
   exposure.

4. **The grid only helps in falling or range-bound markets.** In the −40%/yr Monte Carlo it beats plain
   (+$1,141 vs −$174 mean). In the +60%/yr trend it averages −$7.7k, with a 1% worst case of −$84k. If you
   want it, run it only when you are bearish or expect a range (a trend filter is a sensible next test;
   it is not tested here).

5. **If you run the grid anyway, size it for the tail and cap it.**
   - Capital: at least **$20k per 1 ETH/day** on history (max DD $16.5k plus margin). The bootstrap's 5% worst
     drawdown is about $33k. Plain selling needs about $3–4k per unit. **Return on required capital is
     several times better for plain selling.**
   - The best grid variant tested is a **stop at BE+10%** (J): drawdown −$3.4k instead of −$16.5k. It still
     earns less than plain selling and has a lower Sharpe.
   - The lift rule matters: `nearest` (default) keeps the grid smallest. `farthest`, `prorata` and `fifo`
     earned more in this sample (+$15–17k) but had drawdowns of −$21k to −$25k.
   - "Keep 50% of premium, lift with the rest" (K) made the most in the backtest (+$25.6k) because it is
     simply a bigger short. Its drawdown was −$22.6k.

**Bottom line:** sell the daily ATM calls with good execution and take the losses (plain A).
If you want the perp grid, treat it as an explicit, capped short-ETH trade (stop at about BE+10%, max 3–5
perps) that you turn on only when you are bearish. Do not treat it as a way to repair losses.

## Study 2: sell a call or a put depending on the previous candle (take the whole loss)

Rule: **if the previous daily candle (08:00 → 08:00 UTC) was DOWN, sell an ATM call. If it was UP, sell an ATM put.**
There is no hedge and no perps; every loss is taken in full. Run it with `python3 run_directional.py`; full tables are in
[`results/directional/RESULTS.md`](results/directional/RESULTS.md).

| Per ~1 ETH of options sold per day | PnL 2021–26 | Max DD | Sharpe | Bootstrap: losing years | Bootstrap: 1% worst year |
|---|---:|---:|---:|---:|---:|
| **Your rule** (call after down, put after up) | **+$7,958** | −$2,178 | 1.31 | 11.9% | −$1,545 |
| Always sell call | +$7,083 | −$1,746 | 1.15 | 13.6% | −$1,499 |
| Always sell put | +$7,055 | −$1,751 | 1.05 | 17.3% | −$1,964 |
| Reverse (call after up, put after down) | +$6,182 | −$2,011 | 0.91 | 19.7% | −$1,992 |
| Half straddle every day (0.5 call + 0.5 put) | +$7,069 | **−$788** | 1.95 | 5.5% | −$616 |
| **Straddle only after an UP candle** (skip after down) | **+$10,860** | −$1,063 | **2.25** | **1.7%** | −$158 |

What it shows:

1. **Your rule beats always-call or always-put by about $875 over 5.6 years, but this is not statistically real.**
   Shuffling the same up/down signals at random gives +$3.3k to +$10.7k (5–95% range), and
   **p = 0.33**. Year by year it is mixed: it beat always-call in 3 of 6 years, lost in 2 and tied in 1.
   ETH's day-to-day return autocorrelation is about 0.004, so yesterday's direction says almost nothing about tomorrow's.
2. **Picking one side is the risky choice.** A call and a put sold together can't both lose on the same day, so a
   half straddle earns the same profit per option as your rule with **a third of the drawdown** (−$788 vs −$2,178).
3. **What mattered was *whether* to sell, not *which side*.** Selling either side after an UP candle earned
   **19.2 bps of spot per option**. After a DOWN candle it earned only **5.6 bps** (t = 2.46, permutation p = 0.01;
   better in 4 of 6 years). After down days ETH moves about 12% more the next day (274 vs 245 bps), but the 30-day
   DVOL used for pricing barely changes (73.5% vs 72.9%), so options sold after down days were too cheap.
   *Caveat: real 1-day options often reprice after a drop more than the 30-day index does. This edge may be smaller
   with real quotes, and it was found after several tests (data-mining risk). Check it against real Deribit daily
   quotes before sizing up.*
4. **Selling price still dominates.** Every 5% of IV changes the PnL by about $4k over the period. That is
   much more than the call/put choice (+$875).

**What to do:** don't pick the side from the previous candle. Sell **both sides (a straddle or strangle)** and
use the candle for **timing/size**: full size after up days, reduced or no size after down days. Verify that on real
daily-option IVs first.

## Study 3: is "sell straddles after UP candles" reliable?

Run it with `python3 run_reliability.py`. Tables are in [`results/reliability/RESULTS.md`](results/reliability/RESULTS.md), and the summary chart is
`results/reliability/fig1_forest.png`. Real option prices come from Deribit's public trade history
(`gridsim/option_iv.py`): every day, the near-ATM trades in the next-day expiry between 08:00 and 11:00 UTC.

**Verdict: not reliable. Don't trade it.**

| Check | Result | Pass? |
|---|---|:-:|
| ETH 2021–26, DVOL pricing (the original finding) | +13.6 bps, 90% CI 3.7 to 23.5 | ✅ |
| **Real Deribit 1-day option prices** (ETH) | **+7.7 bps, CI −1.7 to 16.8**. Both sides lose: −3.6 bps after up, −11.3 after down | ❌ |
| Real prices, BTC | −1.6 bps | ❌ |
| Other asset: BTC 2021–26, DVOL pricing | +1.5 bps, CI −6.4 to 9.1 | ❌ |
| Out of sample: ETH Sep 2017 – Mar 2021 | +8.6 bps, CI −8.0 to 25.7 | ❌ |
| Out of sample: BTC Sep 2017 – Mar 2021 | −3.1 bps | ❌ |
| Stable over time | 1st half +5.9 (noise), 2nd half +21.8. Years: 4 up, 2 flat or negative | ⚠️ |
| Neighbouring signal definitions | calendar day ✅, 72h ✅, 48h ⚠️, **12h and 4h negative** ❌ | ⚠️ |
| Driven by a few big days? | median difference only 4.8 bps vs a mean of 13.6 | ⚠️ |

Why it fails:

- **The market already reprices after down days.** Real 1-day IV / DVOL averaged 0.83 after down candles vs
  0.80 after up candles (BTC: 0.81 vs 0.79). The 30-day DVOL doesn't show this, so pricing with DVOL created
  most of the "edge".
- **Daily options carry no volatility premium.** Real 1-day ATM IV averaged 60.6%. Fair value from realised
  moves was 62.1%. Selling them at mid lost about 6 bps per option, both after up and after down days. Timing
  can only make the losses smaller.

Earlier strategies at **real 1-day option prices** (ETH, USD per 1 ETH/day, 2021–26):

| Strategy | PnL, DVOL pricing | **PnL, real prices** | Max DD, real prices |
|---|---:|---:|---:|
| Study 1 A: plain call selling | +7,223 | **−3,490** | −4,184 |
| Study 1 B: perp grid | +10,902 | +7,762 (of which funding +6,643) | −17,867 |
| Study 1 J: grid + stop at BE+10% | +5,352 | −4,202 | −6,108 |
| Study 2: your candle rule | +7,958 | −2,753 | −5,109 |
| Study 2: half straddle daily | +7,069 | −3,610 | −4,008 |
| Study 2: straddle after UP only | +10,860 | −1,604 | −2,126 |

Only the perp grid stays positive, and that comes from perp funding and the short-ETH bet, not from selling
options. Study 1's Monte Carlo already showed that bet has a negative mean and a very large tail.

## Study 4: BUY weekly options in the direction of the last weekly candle, with martingale sizing

Rule: every Friday 08:00 UTC (Deribit weekly expiry), look at the previous Friday→Friday weekly candle.
**If it was up, buy an ATM weekly call. If it was down, buy an ATM weekly put.** Hold to expiry.
Martingale sizing: size = base + (losses since the last equity high) / (8% × spot − premium). That way an
8% move in your favour repays all earlier losses and earns the base trade's profit. Options are priced at
**real Deribit weekly IV paid by buyers** (Apr 2019 – Nov 2026, 395 weeks, average IV 69%, premium about 3.8% of spot).
Run it with `python3 run_weekly.py`. Tables are in [`results/weekly/RESULTS.md`](results/weekly/RESULTS.md).

**The signal (flat size, 1 ETH per week)**

| | PnL 2019–26 | Max DD | Winning weeks | Longest losing streak |
|---|---:|---:|---:|---:|
| **Your rule** (up → call, down → put) | **+$2,407** | −$1,765 | 34% | 14 weeks |
| Reverse (up → put, down → call) | −$4,501 | −$5,598 | 27% | 15 |
| Always buy call | −$280 | −$2,834 | 34% | 14 |
| Always buy put | −$1,815 | −$3,095 | 27% | 20 |
| Buy straddle | −$1,105 | −$2,270 | 38% | 9 |

- Buying options loses on average (random signals: −$979 mean). Your rule beats random with **p = 0.057**.
  It averaged **+0.46% of spot per week in both halves** of the sample. That is promising but not proven (t = 1.3).
- ETH and BTC weekly momentum has the **same sign in every sample**, including 2017–19, which this test didn't
  use. Next week moved in last week's direction 52–54% of the time. But it is weak: only ETH 2019–26 reaches t ≈ 2.

**The martingale**

- With unlimited capital it ends at +$1.06M, but it needed **$2.9M** of capital and a **26,388 ETH** position
  after a 14-week losing streak (2022). The same martingale on the *reverse* signal ends at +$20M. A martingale
  "wins" with any signal if the money never runs out, so the profit says nothing about the signal.
- **Real $10,000 account:** the 8% martingale was **wiped out on 2022-09-09 at every base size** ($1k–$10k
  base trade). Flat sizing survived every time and ended at $11.8k–$28.2k.
- **Monte Carlo (3 years, 3,000 bootstrap paths, $10k account):** the martingale loses 90%+ of the account in
  **73–93% of paths**, and the median ending value is **$0**. Flat sizing at $1k–$5k base had a 0–0.2% ruin chance and a
  median ending value of $10.6k–$12.9k.
- A 5% target is far worse (positions grow beyond any account). Even a 15% target needed $58k of capital per
  1 ETH base.

**What to do:** drop the martingale. If you trade the signal, use **flat size**, with a base trade
of about 10–25% of the account in notional (premium about 0.4–1% of the account per week). Paper-trade it first, because the momentum
edge is not yet statistically proven.

## Limitations

- Studies 1–2 take IV from the 30-day DVOL index (see the correction above; study 3 uses real 1-day trades), not real daily-option quotes. Daily IV can differ, especially at weekends.
- Daily bars only. Intraday drawdowns, margin calls and liquidations are not modelled, and BE fills
  assume the limit order fills at the BE price.
- Binance funding is used as a stand-in for Deribit funding. Funding for Nov 2026 is not published yet, so it is set to 0.
- The settle price is the 08:00 spot price, not Deribit's 30-minute TWAP index.

## Run it

```bash
pip install numpy pandas matplotlib tabulate
```

```bash
python3 run_analysis.py
```

`run_analysis.py` (study 1) and `run_directional.py` (study 2) download and cache the data into `data/`, then write `results/`. Study 1 rules are set in
`gridsim/engine.py` → `Config` (`alloc`, `iv_mult`, `max_perps`, `stop_pct`, `profit_share`,
`trigger="itm"`, fees, funding). For a single run:

```python
from gridsim.data import build_daily
from gridsim.engine import run, Config, stats
res = run(build_daily(), Config(alloc="nearest", stop_pct=0.10))
print(stats(res))
```
