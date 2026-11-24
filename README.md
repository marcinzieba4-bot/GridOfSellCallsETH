# Grid of Sold Calls: ETH daily ATM calls with a break-even short-perp grid

This repo simulates the following strategy on real ETH data (Apr 2021 to Nov 2026) and in Monte Carlo.

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

## Limitations

- IV comes from the 30-day DVOL index, not real daily-option quotes. Daily IV can differ, especially at weekends.
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

`run_analysis.py` downloads and caches the data into `data/`, then writes `results/`. Rules are set in
`gridsim/engine.py` → `Config` (`alloc`, `iv_mult`, `max_perps`, `stop_pct`, `profit_share`,
`trigger="itm"`, fees, funding). For a single run:

```python
from gridsim.data import build_daily
from gridsim.engine import run, Config, stats
res = run(build_daily(), Config(alloc="nearest", stop_pct=0.10))
print(stats(res))
```
