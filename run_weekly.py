"""Study 4: BUY weekly ATM ETH options in the direction of the previous weekly
candle (up -> call, down -> put), flat vs martingale sizing ("an 8% move must
cover earlier losses"). Real Deribit weekly IVs (taker buys). Output: results/weekly/."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridsim.data import spot_daily
from gridsim.weekly import WCfg, run, summary, weekly_table, premium

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "weekly"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID, SURF = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.4,
})
F = {"win weeks %": ".0f", "longest losing streak": ".0f", "max size ETH": ",.1f", "weeks": ".0f",
     "P(ruin) %": ".1f", "P(end below start) %": ".1f", "p-value": ".3f",
     "avg PnL % of spot/week": ".3f", "avg signed next-week move %": ".2f", "t-stat": ".2f",
     "next week same direction %": ".1f"}


def md(df):
    return df.to_markdown(floatfmt=[""] + [F.get(c, ",.0f") for c in df.columns])


def boot_paths(w: pd.DataFrame, n_paths: int, weeks: int, block: int = 8, s0: float = 2000.0) -> list[pd.DataFrame]:
    """Block-bootstrap weekly paths (keeps short momentum within blocks + the real IV of each week)."""
    ret = (w.S1 / w.S0).to_numpy()
    iv = w.iv.to_numpy()
    prev = w.prev_ret.to_numpy()
    out = []
    for _ in range(n_paths):
        idx = []
        while len(idx) < weeks:
            st = RNG.integers(0, len(w) - block)
            idx.extend(range(st, st + block))
        idx = np.array(idx[:weeks])
        r = ret[idx]
        S0 = s0 * np.concatenate([[1.0], np.cumprod(r[:-1])])
        p = np.concatenate([[prev[idx[0]]], r[:-1] - 1])
        t = pd.date_range("2030-01-04 08:00", periods=weeks, freq="7D", tz="UTC")
        out.append(pd.DataFrame({"S0": S0, "S1": S0 * r, "prev_ret": p, "iv": iv[idx]}, index=t))
    return out


def main():
    sp = spot_daily("ETH")
    iv = pd.read_csv(ROOT / "data" / "eth_7d_option_iv.csv", parse_dates=["t"], index_col="t")
    w = weekly_table(sp, iv)
    yrs = (w.index[-1] - w.index[0]).days / 365.25

    # ---- 1. flat sizing (1 ETH per trade): is the signal any good?
    flat = {
        "YOUR RULE: up week -> buy call, down week -> buy put": WCfg(),
        "reverse: up week -> buy put, down week -> buy call": WCfg(rule="contrarian"),
        "always buy call": WCfg(rule="always_call"),
        "always buy put": WCfg(rule="always_put"),
        "buy straddle (0.5 call + 0.5 put)": WCfg(rule="straddle"),
    }
    fr = {k: run(w, c) for k, c in flat.items()}
    flat_t = pd.DataFrame({k: summary(v) for k, v in fr.items()}).T.drop(columns=["ruined"])

    # permutation test of the signal (flat sizing)
    sig = np.sign(w.prev_ret.to_numpy())
    actual = fr[next(iter(fr))].equity.iloc[-1]
    perm = np.array([run(w, WCfg(), RNG.permutation(sig)).equity.iloc[-1] for _ in range(3000)])
    p_val = (perm >= actual).mean()
    # in % of spot (price-level neutral), by half
    mom = fr[next(iter(fr))]
    bps = 1e4 * mom.pnl / mom.S0
    half = w.index[len(w) // 2]
    halves = pd.DataFrame({
        "weeks": [int((bps.index < half).sum()), int((bps.index >= half).sum())],
        "avg PnL, % of spot per week": [bps[bps.index < half].mean() / 100, bps[bps.index >= half].mean() / 100],
        "win weeks %": [100 * (mom.pnl[mom.index < half] > 0).mean(), 100 * (mom.pnl[mom.index >= half] > 0).mean()],
    }, index=[f"{w.index[0]:%Y-%m} to {half:%Y-%m}", f"{half:%Y-%m} to {w.index[-1]:%Y-%m}"])
    t_stat = bps.mean() / (bps.std() / math.sqrt(len(bps)))
    yr = pd.DataFrame({"ETH %": w.groupby(w.index.year).apply(lambda x: 100 * (x.S1.iloc[-1] / x.S0.iloc[0] - 1)),
                       "your rule $": mom.pnl.groupby(mom.index.year).sum(),
                       "reverse $": fr["reverse: up week -> buy put, down week -> buy call"].pnl.groupby(w.index.year).sum()})
    yr.index.name = "year"

    # ---- 1b. model-free momentum check on other data (no options): does next week follow last week?
    mm = []
    for asset in ("ETH", "BTC"):
        spa = spot_daily(asset)
        f = spa.open[spa.index.dayofweek == 4]
        r = f.shift(-1) / f - 1
        pr = f / f.shift(1) - 1
        x = pd.DataFrame({"r": r, "p": pr}).dropna()
        x = x[(x.index.to_series().diff().dt.days == 7)]
        for lab, m in ((f"{asset} Sep 2017 - Mar 2019 (before options data, out of sample)", x.index < pd.Timestamp("2019-04-01", tz="UTC")),
                       (f"{asset} Apr 2019 - Nov 2026", x.index >= pd.Timestamp("2019-04-01", tz="UTC"))):
            y = x[m]
            sgn = np.sign(y.p) * y.r * 100
            mm.append({"sample": lab, "weeks": len(y), "avg signed next-week move %": sgn.mean(),
                       "t-stat": sgn.mean() / (sgn.std() / math.sqrt(len(y))),
                       "next week same direction %": 100 * (np.sign(y.p) == np.sign(y.r)).mean()})
    mom_t = pd.DataFrame(mm).set_index("sample")

    # ---- 2. martingale on history, unlimited capital (what it would take)
    mart = {
        "martingale 8%, your rule": WCfg(sizing="recover"),
        "martingale 8% ('cover losses only'), your rule": WCfg(sizing="cover"),
        "martingale 5%, your rule": WCfg(sizing="recover", move=0.05),
        "martingale 10%, your rule": WCfg(sizing="recover", move=0.10),
        "martingale 15%, your rule": WCfg(sizing="recover", move=0.15),
        "martingale 8%, REVERSE rule (control)": WCfg(sizing="recover", rule="contrarian"),
        "martingale 8%, always buy call (control)": WCfg(sizing="recover", rule="always_call"),
    }
    mr = {k: run(w, c) for k, c in mart.items()}
    mart_t = pd.DataFrame({k: summary(v) for k, v in mr.items()}).T.drop(columns=["ruined"])

    # ---- 3. realistic account: $10,000, base trade notional $1k..$10k
    cap = 10_000.0
    acct_rows, mc_rows = [], []
    paths = boot_paths(w, 3000, 156)
    for base_usd in (1_000, 2_500, 5_000, 10_000):
        for sizing in ("flat", "recover"):
            cfg = WCfg(sizing=sizing, base_usd=base_usd, capital=cap)
            h = run(w, cfg)
            s = summary(h)
            acct_rows.append({"base trade notional $": base_usd, "sizing": "flat" if sizing == "flat" else "martingale 8%",
                              "end equity $": max(cap + s["PnL $"], 0.0), "worst equity $": max(cap + h.equity.min(), 0.0),
                              "ruined": "YES" if s["ruined"] else "no",
                              "ruined on": (h.index[h.ruined.argmax()].strftime("%Y-%m-%d") if s["ruined"] else ""),
                              "weeks capped by cash": int(h.capped.sum())})
            ends, ruin = [], []
            for pth in paths:
                r = run(pth, cfg)
                e = cap + r.equity.to_numpy()
                ends.append(e[-1])
                ruin.append((e <= 0.1 * cap).any())
            ends = np.array(ends)
            mc_rows.append({"base trade notional $": base_usd, "sizing": "flat" if sizing == "flat" else "martingale 8%",
                            "P(ruin) %": 100 * np.mean(ruin), "P(end below start) %": 100 * (ends < cap).mean(),
                            "median end $": np.median(ends), "mean end $": ends.mean(),
                            "5% worst end $": np.quantile(ends, 0.05), "95% best end $": np.quantile(ends, 0.95)})
    acct_t = pd.DataFrame(acct_rows).set_index(["base trade notional $", "sizing"])
    mc_t = pd.DataFrame(mc_rows).set_index(["base trade notional $", "sizing"])

    # ---- charts
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4, 1.2]})
    ax[0].plot(w.index, w.S0, color=INK2, lw=1.0)
    ax[0].set_title("ETH price at Friday 08:00 UTC (USD)")
    for (k, v), col in zip(fr.items(), (BLUE, ORANGE, AQUA, YELLOW, INK2)):
        if k.startswith(("always", "buy straddle")) and k != "buy straddle (0.5 call + 0.5 put)":
            continue
        ax[1].plot(v.index, v.equity, color=col, lw=2.0 if k.startswith("YOUR") else 1.2,
                   label=f"{k.split(':')[0] if k.startswith('YOUR') else k}  ({v.equity.iloc[-1]:+,.0f})")
    ax[1].axhline(0, color=INK2, lw=0.6)
    ax[1].set_title("Flat size (1 ETH of options per week): cumulative PnL, USD")
    ax[1].legend(fontsize=8, loc="lower left")
    m8 = mr["martingale 8%, your rule"]
    ax[2].semilogy(m8.index, m8["size"].clip(lower=1e-3), color=BLUE, lw=1.2, drawstyle="steps-post",
                   label="martingale 8%: ETH of options bought each week")
    ax[2].semilogy(m8.index, m8.outlay.clip(lower=1), color=ORANGE, lw=1.0, drawstyle="steps-post",
                   label="premium paid that week, USD")
    ax[2].set_title("Martingale 8%, unlimited capital: position size explodes in losing streaks (log scale)")
    ax[2].legend(fontsize=8, loc="upper left")
    ax[2].xaxis.set_major_locator(mdates.YearLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    xs = np.arange(4)
    fl = mc_t.xs("flat", level="sizing")
    mg = mc_t.xs("martingale 8%", level="sizing")
    ax[0].bar(xs - 0.2, fl["P(ruin) %"], width=0.38, color=AQUA, label="flat size")
    ax[0].bar(xs + 0.2, mg["P(ruin) %"], width=0.38, color=ORANGE, label="martingale 8%")
    for x, v in zip(xs + 0.2, mg["P(ruin) %"]):
        ax[0].text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    for x, v in zip(xs - 0.2, fl["P(ruin) %"]):
        ax[0].text(x, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"${b/1000:g}k" for b in fl.index])
    ax[0].set_xlabel("base trade notional (account $10,000)")
    ax[0].set_ylabel("% of 3-year paths")
    ax[0].set_title("Chance of losing 90%+ of the account within 3 years")
    ax[0].set_ylim(0, 110)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[1].plot(xs, fl["median end $"], "o-", color=AQUA, label="flat: median end")
    ax[1].plot(xs, mg["median end $"], "o-", color=ORANGE, label="martingale: median end")
    ax[1].axhline(cap, color=INK2, lw=0.8)
    ax[1].text(3, cap * 1.02, "start $10k", ha="right", fontsize=8, color=INK2)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"${b/1000:g}k" for b in fl.index])
    ax[1].set_xlabel("base trade notional")
    ax[1].set_ylabel("USD")
    ax[1].set_title("Median account value after 3 years (bootstrap)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_account_ruin.png", dpi=160)
    plt.close(fig)

    prem_pct = 100 * (premium(w.S0, w.iv) / w.S0).mean()
    move8 = 100 * ((np.abs(w.S1 / w.S0 - 1) >= 0.08)).mean()
    dir8 = 100 * (np.sign(w.S1 / w.S0 - 1) == np.sign(w.prev_ret)) [np.abs(w.S1 / w.S0 - 1) >= 0.08].mean()
    txt = [
        "# Study 4: buy weekly options in the direction of the last weekly candle (auto-generated by `run_weekly.py`)\n",
        f"Weeks {w.index[0].date()} -> {w.index[-1].date()} ({len(w)} weeks, {yrs:.1f} yrs), Friday 08:00 UTC entry, "
        f"ATM strike, priced at REAL Deribit weekly IV paid by buyers (avg {100 * w.iv.mean():.0f}%), "
        f"fees 0.03% in + 0.015% delivery. Avg premium {prem_pct:.1f}% of spot. ETH moved >= 8% in "
        f"{move8:.0f}% of weeks; when it did, it went in the signal's direction {dir8:.0f}% of the time. "
        f"Weekly return autocorrelation {(w.S1 / w.S0 - 1).autocorr(1):+.3f}.\n",
        "## 1. Flat size, 1 ETH of options per week (USD)\n", md(flat_t), "\n",
        f"Permutation test (signals shuffled 3000x): real {actual:+,.0f} vs shuffled mean {perm.mean():+,.0f}, "
        f"5-95% {np.quantile(perm, .05):+,.0f}..{np.quantile(perm, .95):+,.0f}, **p = {p_val:.3f}**. "
        f"t-stat of weekly PnL in % of spot: {t_stat:.2f}.\n",
        md(halves.rename(columns={"avg PnL, % of spot per week": "avg PnL % of spot/week"})), "\n",
        md(yr.rename(columns={"ETH %": "ETH %"})), "\n",
        "### Model-free check: does next week move in last week's direction? (spot only)\n", md(mom_t), "\n",
        "## 2. Martingale on history with UNLIMITED capital (USD, base 1 ETH)\n", md(mart_t), "\n",
        "## 3. Real account: $10,000, base trade notional $1k-$10k, history 2019-2026\n",
        acct_t.to_markdown(floatfmt=",.0f"), "\n",
        "## 4. Block bootstrap of real weeks (8-week blocks), 3 years, 3000 paths, $10,000 account\n",
        "Ruin = account falls to 10% of start or less.\n", md(mc_t), "\n",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
