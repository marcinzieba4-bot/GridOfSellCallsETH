"""Study: sell daily ATM CALL after a DOWN candle, ATM PUT after an UP candle,
take the whole loss. Compared with always-call / always-put / contrarian / straddle,
plus a permutation test, previous-move threshold sweep, IV sensitivity, per-year
breakdown and block-bootstrap 1-year distribution. Output: results/directional/."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridsim.data import build_daily
from gridsim.directional import DCfg, simulate, summary

OUT = Path(__file__).parent / "results" / "directional"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 1.4,
})

RULES = {
    "YOUR RULE: call after down, put after up": DCfg(rule="momentum"),
    "always sell call": DCfg(rule="always_call"),
    "always sell put": DCfg(rule="always_put"),
    "reverse: call after up, put after down": DCfg(rule="contrarian"),
    "half straddle: 0.5 call + 0.5 put every day": DCfg(rule="half_straddle"),
    "straddle only after UP candle (skip after down)": DCfg(rule="straddle_after_up"),
    "straddle: call + put every day (2 options)": DCfg(rule="straddle"),
}
GREEN = "#008300"
COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, INK2]


def md(df, fmt_map=None, idx_fmt=""):
    fmt_map = fmt_map or {}
    cols = [fmt_map.get(c, ",.0f") for c in df.columns]
    return df.to_markdown(floatfmt=[idx_fmt] + cols)


FMT = {"edge bps/option": ".1f", "win %": ".1f", "Sharpe": ".2f", "PnL $/option": ".2f",
       "options sold": ",.0f"}


def main():
    d = build_daily()
    years = (d.index[-1] - d.index[0]).days / 365.25
    sims = {k: simulate(d, c) for k, c in RULES.items()}
    main_t = pd.DataFrame({k: summary(v) for k, v in sims.items()}).T
    main_t.to_csv(OUT / "rules.csv")

    # ---- per-year PnL
    py = pd.DataFrame({k: v.pnl.groupby(v.index.year).sum() for k, v in sims.items()})
    eth = d.groupby(d.index.year).apply(lambda x: 100 * (x.close.iloc[-1] / x.open.iloc[0] - 1))
    py.insert(0, "ETH %", eth)
    py.columns = ["ETH %", "your rule", "always call", "always put", "reverse", "half straddle",
                  "straddle after up", "straddle x2"]
    py.index.name = "year"

    # ---- call leg vs put leg of your rule
    m = sims["YOUR RULE: call after down, put after up"]
    legs_t = pd.DataFrame({
        "calls after DOWN candle": {"trades": int((m.call_pnl != 0).sum()), "PnL $": m.call_pnl.sum()},
        "puts after UP candle": {"trades": int((m.put_pnl != 0).sum()), "PnL $": m.put_pnl.sum()},
    }).T
    c_all, p_all = sims["always sell call"], sims["always sell put"]
    prev_up = (d.close > d.open).shift(1).fillna(False).to_numpy()
    prev_dn = (d.close < d.open).shift(1).fillna(False).to_numpy()
    cond = pd.DataFrame({
        "after UP candle": {"sell call $": c_all.pnl[prev_up].sum(), "sell put $": p_all.pnl[prev_up].sum(),
                            "days": int(prev_up.sum())},
        "after DOWN candle": {"sell call $": c_all.pnl[prev_dn].sum(), "sell put $": p_all.pnl[prev_dn].sum(),
                              "days": int(prev_dn.sum())},
    }).T

    # ---- timing: is selling after UP candles better than after DOWN candles (any side)?
    st = sims["straddle: call + put every day (2 options)"]
    bps = 1e4 * st.pnl / st.S / 2
    pv = np.sign((d.close / d.open - 1).shift(1)).to_numpy()
    up_b, dn_b = bps[pv > 0], bps[pv < 0]
    t_stat = (up_b.mean() - dn_b.mean()) / math.sqrt(up_b.var() / len(up_b) + dn_b.var() / len(dn_b))
    prng = np.random.default_rng(3)
    bb = bps.to_numpy()
    diffs = []
    for _ in range(5000):
        pp = prng.permutation(np.nan_to_num(pv))
        diffs.append(bb[pp > 0].mean() - bb[pp < 0].mean())
    p_timing = (np.array(diffs) >= up_b.mean() - dn_b.mean()).mean()
    mv = (d.close / d.open - 1).abs() * 1e4
    timing_t = pd.DataFrame({
        "after UP candle": {"days": len(up_b), "edge bps/option": up_b.mean(),
                            "next-day |move| bps": mv[pv > 0].mean(), "DVOL %": 100 * d.dvol[pv > 0].mean()},
        "after DOWN candle": {"days": len(dn_b), "edge bps/option": dn_b.mean(),
                              "next-day |move| bps": mv[pv < 0].mean(), "DVOL %": 100 * d.dvol[pv < 0].mean()},
    }).T
    timing_y = pd.DataFrame({"after UP": up_b.groupby(up_b.index.year).mean(),
                             "after DOWN": dn_b.groupby(dn_b.index.year).mean()})
    timing_y.index.name = "year"

    # ---- permutation test: does the ORDER of up/down signals matter?
    rng = np.random.default_rng(11)
    sig = np.sign(np.nan_to_num((d.close / d.open - 1).shift(1).to_numpy()))
    sig[0] = 0
    actual = m.pnl.sum()
    perm = np.array([simulate(d, DCfg(rule="momentum"), rng.permutation(sig)).pnl.sum() for _ in range(2000)])
    p_val = (perm >= actual).mean()

    # ---- threshold sweep on |previous candle|
    thr_rows = []
    for thr in (0.0, 0.01, 0.02, 0.03, 0.05):
        for rule, lab in (("momentum", "your rule"), ("contrarian", "reverse")):
            s = summary(simulate(d, DCfg(rule=rule, thr=thr)))
            thr_rows.append({"min |prev candle|": f"{thr:.0%}", "rule": lab, "options sold": s["options sold"],
                             "PnL $": s["PnL $"], "PnL $/option": s["PnL $/option"],
                             "edge bps/option": s["edge bps/option"], "max DD $": s["max DD $"],
                             "Sharpe": s["Sharpe"]})
    thr_t = pd.DataFrame(thr_rows).set_index(["min |prev candle|", "rule"])

    # ---- IV sensitivity
    iv_rows = []
    for mlt in (0.85, 0.90, 0.95, 1.00, 1.05):
        row = {"IV = DVOL x": mlt}
        for rule, lab in (("momentum", "your rule"), ("always_call", "always call"), ("always_put", "always put")):
            row[f"{lab} PnL $"] = simulate(d, DCfg(rule=rule, iv_mult=mlt)).pnl.sum()
        iv_rows.append(row)
    iv_t = pd.DataFrame(iv_rows).set_index("IV = DVOL x")

    # ---- block bootstrap of daily PnL (same blocks for every rule), 1-year horizon
    n, block, H = 4000, 20, 365
    P = np.column_stack([v.pnl.to_numpy() for v in sims.values()])
    starts = rng.integers(0, len(P) - block, size=(n, math.ceil(H / block)))
    res = {k: [] for k in sims}
    for st in starts:
        idx = (st[:, None] + np.arange(block)).ravel()[:H]
        path = P[idx]
        eq = path.cumsum(0)
        dd = (eq - np.maximum.accumulate(eq, 0)).min(0)
        for j, k in enumerate(sims):
            res[k].append((eq[-1, j], dd[j]))
    boot = pd.DataFrame({k: {
        "mean 1y $": np.mean([x[0] for x in v]), "median 1y $": np.median([x[0] for x in v]),
        "5% worst 1y $": np.quantile([x[0] for x in v], 0.05), "1% worst 1y $": np.quantile([x[0] for x in v], 0.01),
        "P(losing year) %": 100 * np.mean([x[0] < 0 for x in v]),
        "median max DD $": np.median([x[1] for x in v]), "5% worst max DD $": np.quantile([x[1] for x in v], 0.05),
    } for k, v in res.items()}).T

    # ---- charts
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    ax[0].plot(d.index, d.close, color=INK2, lw=1.0)
    ax[0].set_title("ETH price (USD)")
    for (k, v), col in list(zip(sims.items(), COLORS))[:6]:
        lw = 2.0 if k.startswith("YOUR") else 1.2
        ax[1].plot(v.index, v.equity, color=col, lw=lw, label=f"{k}  ({v.equity.iloc[-1]:+,.0f})")
    ax[1].axhline(0, color=INK2, lw=0.6)
    ax[1].set_title("Cumulative PnL, USD - every rule sells about 1 ETH of options per day")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set_ylabel("USD")
    ax[1].xaxis.set_major_locator(mdates.YearLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_equity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ax[0].hist(perm, bins=50, color=INK2, alpha=0.35, label="same rule, signals randomly shuffled (2000x)")
    ax[0].axvline(actual, color=BLUE, lw=2, label=f"your rule, real signals: {actual:+,.0f}")
    ax[0].set_title(f"Is the candle signal real?  p = {p_val:.2f}")
    ax[0].set_xlabel("total PnL 2021-26, USD")
    ax[0].set_ylabel("shuffles")
    ax[0].legend(fontsize=8, loc="upper left")
    short = {"YOUR": "your rule", "always sell call": "always call", "half": "half straddle daily",
             "straddle only": "straddle after UP only"}
    for (k, v), col in zip(res.items(), COLORS):
        lab = next((nm for pre, nm in short.items() if k.startswith(pre)), None)
        if lab is None:
            continue
        ax[1].hist(np.clip([x[0] for x in v], -6000, 6000), bins=60, histtype="step", color=col, lw=1.4,
                   label=lab)
    ax[1].axvline(0, color=INK2, lw=0.6)
    ax[1].set_title("1-year PnL, block bootstrap of real days (4000 paths)")
    ax[1].set_xlabel("1-year PnL, USD (clipped at +-6000)")
    ax[1].legend(fontsize=8, loc="upper left", frameon=True, facecolor=SURF, edgecolor=GRID, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_permutation_bootstrap.png", dpi=160)
    plt.close(fig)

    text = [
        "# Directional daily option selling (auto-generated by `run_directional.py`)\n",
        f"Data {d.index[0].date()} -> {d.index[-1].date()} ({len(d)} days, {years:.1f} yrs). "
        "Sell 1 ATM daily option at 08:00 UTC, strike = spot, take the whole loss. IV = DVOL x 0.95, "
        "Deribit fee 0.03% (cap 12.5% of premium). USD per 1 ETH option.\n",
        "## 1. Rules compared\n", md(main_t, FMT), "\n",
        "## 2. Where the rule's PnL comes from: sell call vs sell put, split by previous candle\n",
        md(cond, {"days": ",.0f"}), "\n", md(legs_t, {"trades": ",.0f"}), "\n",
        f"## 3. Permutation test\nReal signals: {actual:+,.0f} USD. Random shuffles of the same up/down "
        f"signals: mean {perm.mean():+,.0f}, 5-95% range {np.quantile(perm, .05):+,.0f} .. "
        f"{np.quantile(perm, .95):+,.0f}. **p-value = {p_val:.3f}** (share of shuffles that did at least as well).\n",
        "## 3b. Timing: selling (either side) after UP vs after DOWN candles\n",
        md(timing_t, {"days": ",.0f", "edge bps/option": ".1f", "next-day |move| bps": ".0f", "DVOL %": ".1f"}), "\n",
        f"Difference t-stat {t_stat:.2f}, permutation p = {p_timing:.3f}. By year (edge bps/option):\n",
        md(timing_y, {"after UP": ".1f", "after DOWN": ".1f"}), "\n",
        "## 4. Only trade after a big candle (|prev candle| >= threshold, else no trade)\n", md(thr_t, FMT), "\n",
        "## 5. Selling-price sensitivity\n", md(iv_t, idx_fmt=".2f"), "\n",
        "## 6. PnL by calendar year\n", md(py, {"ETH %": "+.0f"}), "\n",
        "## 7. Block bootstrap (20-day blocks), 1-year horizon, 4000 paths\n", md(boot, {"P(losing year) %": ".1f"}), "\n",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
