"""Run the full study: historical backtest, rolling 1-year windows, Monte Carlo,
risk-control sweep. Writes CSV/markdown tables and PNG charts into results/."""
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
from gridsim.engine import Config, run, stats
from gridsim.montecarlo import block_bootstrap, gbm_days

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# palette (validated default instance: slot1 blue, slot2 orange, slot3 aqua) + inks
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 1.4,
})

STRATS = {
    "A plain: sell calls, take losses": Config(mode="plain"),
    "B grid (your rules, nearest-BE lift)": Config(),
    "C grid, lift farthest BE first": Config(alloc="farthest"),
    "D grid, pro-rata lift": Config(alloc="prorata"),
    "E grid, FIFO lift": Config(alloc="fifo"),
    "F grid, no funding": Config(use_funding=False),
    "G grid, max 5 perps": Config(max_perps=5),
    "H grid, max 3 perps": Config(max_perps=3),
    "I grid, stop perp at BE+20%": Config(stop_pct=0.20),
    "J grid, stop perp at BE+10%": Config(stop_pct=0.10),
    "K grid, 50% of premium kept": Config(profit_share=0.5),
}
SHORT = {"A": "plain", "B": "grid", "J": "grid+stop10%", "G": "grid max5"}


def fmt(df: pd.DataFrame) -> str:
    idx_fmt = ".2f" if df.index.dtype.kind == "f" else ""
    cols = [".2f" if c in ("PnL/|maxDD|", "Sharpe") else ",.0f" for c in df.columns]
    return df.to_markdown(floatfmt=[idx_fmt] + cols)


def full_backtest(d: pd.DataFrame):
    rows, res = [], {}
    for name, cfg in STRATS.items():
        r = run(d, cfg)
        res[name[0]] = r
        s = stats(r)
        rows.append({
            "strategy": name, "PnL $": s["final_pnl"], "PnL $/yr": s["pnl_per_year"],
            "max DD $": s["max_drawdown"], "worst equity $": s["worst_equity"],
            "max open perps": s["max_perps"], "longest perp (days)": s["max_days_to_close"],
            "funding $": s["funding_total"], "fees $": s["fees_total"],
            "PnL/|maxDD|": s["final_pnl"] / abs(s["max_drawdown"]),
            "Sharpe": s["sharpe_like"],
        })
    t = pd.DataFrame(rows).set_index("strategy")
    t.to_csv(OUT / "backtest_variants.csv")
    return t, res


def iv_sensitivity(d):
    rows = []
    for m in (0.85, 0.90, 0.95, 1.00, 1.05):
        a = stats(run(d, Config(mode="plain", iv_mult=m)))
        b = stats(run(d, Config(iv_mult=m)))
        rows.append({"IV = DVOL x": m, "plain PnL $": a["final_pnl"], "plain max DD $": a["max_drawdown"],
                     "grid PnL $": b["final_pnl"], "grid max DD $": b["max_drawdown"]})
    return pd.DataFrame(rows).set_index("IV = DVOL x")


def rolling(d, horizon=365):
    starts = pd.date_range(d.index[0], d.index[-1] - pd.Timedelta(days=horizon), freq="14D")
    rows = []
    for st in starts:
        w = d.loc[st: st + pd.Timedelta(days=horizon - 1)]
        row = {"start": st, "eth_chg_%": 100 * (w.close.iloc[-1] / w.open.iloc[0] - 1)}
        for k in SHORT:
            s = stats(run(w, STRATS[[n for n in STRATS if n[0] == k][0]]))
            row[f"{SHORT[k]} pnl"] = s["final_pnl"]
            row[f"{SHORT[k]} dd"] = s["max_drawdown"]
            row[f"{SHORT[k]} perps"] = s["max_perps"]
            row[f"{SHORT[k]} end_open"] = s["perps_open_end"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("start")


def mc(gen, n=600, seed=7):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in SHORT}
    for _ in range(n):
        p = gen(rng)
        for k in SHORT:
            s = stats(run(p, STRATS[[n_ for n_ in STRATS if n_[0] == k][0]]))
            out[k].append((s["final_pnl"], s["max_drawdown"], s["max_perps"], s["perps_open_end"],
                           100 * (p.close.iloc[-1] / p.open.iloc[0] - 1)))
    return {k: pd.DataFrame(v, columns=["pnl", "dd", "max_perps", "end_open", "eth_chg"]) for k, v in out.items()}


def mc_summary(r: dict, label: str) -> pd.DataFrame:
    rows = []
    for k, v in r.items():
        rows.append({"scenario": label, "strategy": SHORT[k], "mean PnL $": v.pnl.mean(),
                     "median PnL $": v.pnl.median(), "5% worst PnL $": v.pnl.quantile(0.05),
                     "1% worst PnL $": v.pnl.quantile(0.01), "P(loss) %": 100 * (v.pnl < 0).mean(),
                     "median max DD $": v.dd.median(), "5% worst max DD $": v.dd.quantile(0.05),
                     "avg max perps": v.max_perps.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts
def chart_backtest(d, res):
    rb, ra = res["B"], res["A"]
    pdf = rb.perps_df()
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True,
                           gridspec_kw={"height_ratios": [2.3, 1.6, 0.9]})
    ax[0].plot(d.index, d.close, color=INK2, lw=1.0, label="ETH price")
    longp = pdf[((pdf.t_close.fillna(d.index[-1]) - pdf.t_open).dt.days >= 3)]
    for _, p in longp.iterrows():
        ts, bes = zip(*p.path)
        ts, bes = list(ts), list(bes)
        if pd.isna(p.t_close):
            ts.append(d.index[-1]); bes.append(bes[-1])
        ax[0].step(ts, bes, where="post", color=BLUE, lw=0.8, alpha=0.6)
        ax[0].plot(p.t_open, p.entry, marker="v", ms=3, color=ORANGE, lw=0)
    ax[0].plot([], [], color=BLUE, label="perp break-even (steps up as premium/funding lift it)")
    ax[0].plot([], [], marker="v", color=ORANGE, lw=0, label="short perp entry")
    ax[0].set_title(f"ETH price and the short-perp grid  ({len(pdf)} perps opened; {len(longp)} lived 3+ days, shown)")
    ax[0].set_ylabel("USD")
    ax[0].legend(loc="upper left", fontsize=8, frameon=True, facecolor=SURF, edgecolor=GRID, framealpha=0.95)
    ax[1].plot(ra.daily.index, ra.daily.equity, color=ORANGE, label="A plain call selling")
    ax[1].plot(rb.daily.index, rb.daily.equity, color=BLUE, label="B grid (your rules)")
    ax[1].plot(res["J"].daily.index, res["J"].daily.equity, color=AQUA, label="J grid + stop at BE+10%")
    ax[1].axhline(0, color=INK2, lw=0.6)
    ax[1].set_title("Cumulative PnL, USD per 1 ETH call sold per day (marked to market)")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].set_ylabel("USD")
    ax[2].fill_between(rb.daily.index, rb.daily.n_perps, step="post", color=BLUE, alpha=0.35, lw=0)
    ax[2].plot(rb.daily.index, rb.daily.n_perps, color=BLUE, lw=0.8, drawstyle="steps-post")
    ax[2].set_title("Open short perps in grid B (= ETH short, on top of the daily short call)")
    ax[2].set_ylabel("perps")
    ax[2].xaxis.set_major_locator(mdates.YearLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_backtest.png", dpi=160)
    plt.close(fig)


def chart_zoom(d, res, a="2023-09-15", b="2024-06-30"):
    rb = res["B"]
    pdf = rb.perps_df()
    w = d.loc[a:b]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(w.index, w.close, color=INK2, lw=1.0, label="ETH price")
    sel = pdf[(pdf.t_open <= pd.Timestamp(b, tz="UTC")) &
              (pdf.t_close.fillna(d.index[-1]) >= pd.Timestamp(a, tz="UTC"))]
    for _, p in sel.iterrows():
        end = min(p.t_close if pd.notna(p.t_close) else d.index[-1], pd.Timestamp(b, tz="UTC"))
        pts = [(t_, b_) for t_, b_ in p.path if t_ <= end]
        if pd.isna(p.t_close) or p.t_close > end:
            pts.append((end, pts[-1][1]))
        ts, bes = zip(*pts)
        ax[0].step(ts, bes, where="post", color=BLUE, lw=0.9, alpha=0.7)
        ax[0].plot(p.t_open, p.entry, marker="v", ms=4, color=ORANGE, lw=0)
    ax[0].plot([], [], color=BLUE, label="perp break-even (steps up each time premium/funding is spent on it)")
    ax[0].plot([], [], marker="v", color=ORANGE, lw=0, label="perp short entry (day after losing call)")
    ax[0].set_title("Zoom: the grid building up in the Oct 2023 - Mar 2024 rally")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_ylabel("USD")
    dd = rb.daily.loc[a:b]
    ax[1].plot(dd.index, dd.equity, color=BLUE, label="grid B equity")
    ax[1].plot(res["A"].daily.loc[a:b].index, res["A"].daily.loc[a:b].equity, color=ORANGE, label="plain A equity")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].set_ylabel("USD")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig2_grid_zoom.png", dpi=160)
    plt.close(fig)


def chart_rolling(rl):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, col in (("plain", ORANGE), ("grid", BLUE), ("grid+stop10%", AQUA)):
        ax[0].scatter(rl["eth_chg_%"], rl[f"{k} pnl"], s=14, color=col, label=k, alpha=0.8, edgecolor=SURF, lw=0.4)
        ax[1].scatter(rl[f"{k} dd"], rl[f"{k} pnl"], s=14, color=col, label=k, alpha=0.8, edgecolor=SURF, lw=0.4)
    ax[0].set_xlabel("ETH change over the 1-year window, %")
    ax[0].set_ylabel("1-year PnL, USD")
    ax[0].set_title("1-year windows: PnL vs ETH trend")
    ax[1].set_xlabel("max drawdown in the window, USD")
    ax[1].set_title("1-year windows: PnL vs max drawdown")
    for a in ax:
        a.axhline(0, color=INK2, lw=0.6)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_rolling_windows.png", dpi=160)
    plt.close(fig)


def chart_mc(mcs: dict):
    fig, ax = plt.subplots(1, len(mcs), figsize=(4.0 * len(mcs), 3.8), sharey=False)
    for a, (label, r) in zip(ax, mcs.items()):
        lo = min(r["B"].pnl.quantile(0.01), r["A"].pnl.quantile(0.01))
        hi = max(r["B"].pnl.quantile(0.99), r["A"].pnl.quantile(0.99))
        bins = np.linspace(lo, hi, 45)
        a.hist(r["A"].pnl.clip(lo, hi), bins=bins, color=ORANGE, alpha=0.55, label="plain")
        a.hist(r["B"].pnl.clip(lo, hi), bins=bins, color=BLUE, alpha=0.55, label="grid")
        a.axvline(r["A"].pnl.mean(), color=ORANGE, lw=1.2)
        a.axvline(r["B"].pnl.mean(), color=BLUE, lw=1.2)
        a.set_title(label, fontsize=9)
        a.set_xlabel("1-year PnL, USD (lines = mean; tails clipped at 1%)")
        a.legend(fontsize=8)
    ax[0].set_ylabel("paths")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_monte_carlo.png", dpi=160)
    plt.close(fig)


def main():
    d = build_daily()
    print(f"data: {d.index[0].date()} -> {d.index[-1].date()}  ({len(d)} days)")
    t, res = full_backtest(d)
    ivs = iv_sensitivity(d)
    rl = rolling(d)
    rl.to_csv(OUT / "rolling_1y_windows.csv")
    n = 600
    gens = {
        "GBM flat, no vol premium": lambda g: gbm_days(365, 0.70, 0.0, rng=g),
        "GBM, ETH +60%/yr trend": lambda g: gbm_days(365, 0.70, math.log(1.6), rng=g),
        "GBM, ETH -40%/yr trend": lambda g: gbm_days(365, 0.70, math.log(0.6), rng=g),
        "Bootstrap of real 2021-26 days": lambda g: block_bootstrap(d, 365, 20, rng=g),
    }
    mcs = {k: mc(g, n) for k, g in gens.items()}
    mct = pd.concat([mc_summary(v, k) for k, v in mcs.items()]).set_index(["scenario", "strategy"])
    mct.to_csv(OUT / "monte_carlo.csv")

    chart_backtest(d, res)
    chart_zoom(d, res)
    chart_rolling(rl)
    chart_mc(mcs)

    # trade-level facts for grid B
    pdf = res["B"].perps_df()
    closed = pdf[pdf.t_close.notna()]
    life = (closed.t_close - closed.t_open).dt.days
    facts = {
        "perps opened": len(pdf), "closed at break-even or better": len(closed),
        "closed same day": int((life == 0).sum()), "closed within 7 days": int((life <= 7).sum()),
        "held > 30 days": int((life > 30).sum()), "held > 180 days": int((life > 180).sum()),
        "median life (days)": float(life.median()), "mean life (days)": float(life.mean()),
        "max life (days)": int(life.max()),
    }
    # how much of the call PnL the grid "consumed"
    bd = res["B"].daily
    dec = {
        "call premium collected": bd.tot_prem.iloc[-1], "call payouts": -bd.tot_payout.iloc[-1],
        "option fees": -bd.tot_opt_fee.iloc[-1], "perp fees": -bd.tot_perp_fee.iloc[-1],
        "funding received": bd.tot_fund.iloc[-1],
        "perp price PnL": bd.equity.iloc[-1] - (bd.tot_prem.iloc[-1] - bd.tot_payout.iloc[-1]
                                                - bd.tot_opt_fee.iloc[-1] - bd.tot_perp_fee.iloc[-1]
                                                + bd.tot_fund.iloc[-1]),
        "TOTAL": bd.equity.iloc[-1],
    }
    roll_sum = pd.DataFrame({k: {
        "median 1y PnL $": rl[f"{k} pnl"].median(), "worst 1y PnL $": rl[f"{k} pnl"].min(),
        "% windows losing": 100 * (rl[f"{k} pnl"] < 0).mean(),
        "median max DD $": rl[f"{k} dd"].median(), "worst max DD $": rl[f"{k} dd"].min(),
        "max perps": rl[f"{k} perps"].max(),
    } for k in ("plain", "grid", "grid+stop10%", "grid max5")}).T

    md = [
        "# Results (auto-generated by `run_analysis.py`)\n",
        f"Data: ETHUSDT {d.index[0].date()} -> {d.index[-1].date()} ({len(d)} daily expiries, 08:00 UTC). "
        f"Realized vol {np.log(d.close / d.open).std() * math.sqrt(365):.0%}, mean DVOL {d.dvol.mean():.0%}, "
        f"mean funding {d.funding.mean() * 365:.1%}/yr. All PnL in USD for **1 ETH call sold per day** "
        "(scale linearly).\n",
        "## 1. Full-period backtest\n", fmt(t), "\n",
        "## 2. PnL decomposition of grid B\n",
        pd.Series(dec).to_frame("USD").to_markdown(floatfmt=",.0f"), "\n",
        "## 3. Grid B perp lifecycle\n", pd.Series(facts).to_frame("value").to_markdown(), "\n",
        "## 4. Selling-price sensitivity (IV used = DVOL x multiplier)\n", fmt(ivs), "\n",
        "## 5. Rolling 1-year windows (start every 14 days)\n", fmt(roll_sum), "\n",
        "## 6. Monte Carlo, 1-year horizon, 600 paths per scenario\n", fmt(mct), "\n",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
