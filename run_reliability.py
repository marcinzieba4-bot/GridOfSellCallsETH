"""Is "sell a straddle after UP candles, less/none after DOWN candles" reliable?

Core quantity, per day: the edge of selling one ATM 1-day option (straddle leg),
in bps of spot = premium - fee - |move|/2  (a straddle pays |move|; per option = /2).
We compare the mean edge on days after an UP candle vs after a DOWN candle and
run it through:
  1. statistics that respect volatility clustering (block bootstrap CI, circular-shift test)
  2. stability over time (years, halves, rolling) and weekday/weekend
  3. neighbouring signal definitions (other candle windows, thresholds)
  4. out-of-sample: BTC (other asset) and ETH/BTC 2017-2021 (period not used to find it)
  5. REAL Deribit 1-day option IVs (does the market already reprice after down days?)
Writes results/reliability/.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridsim.data import build_daily, fetch_hourly, spot_daily
from gridsim.directional import DCfg, simulate, summary
from gridsim.engine import Config, run, stats

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "reliability"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)
T = 1 / 365
FEE_BPS, FEE_CAP = 3.0, 0.125

BLUE, ORANGE, AQUA, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.4,
})


# ------------------------------------------------------------------ building blocks
def prem_bps(iv) -> np.ndarray:
    x = np.asarray(iv, float) * math.sqrt(T) / 2.0
    return 1e4 * np.vectorize(lambda v: math.erf(v / math.sqrt(2.0)))(x)


def edge_bps(d: pd.DataFrame, iv: pd.Series) -> pd.Series:
    """Per-option edge (bps of spot) of selling an ATM straddle at 08:00, held to expiry."""
    p = prem_bps(iv.reindex(d.index))
    fee = np.minimum(FEE_BPS, FEE_CAP * p)
    move = np.abs(d.close / d.open - 1).to_numpy() * 1e4
    return pd.Series(p - fee - move / 2.0, index=d.index)


def prev_candle(d: pd.DataFrame) -> pd.Series:
    """Base signal: previous 08:00->08:00 candle, +1 up / -1 down."""
    return np.sign((d.close / d.open - 1).shift(1))


def split(e: pd.Series, sig: pd.Series):
    sig = sig.reindex(e.index)
    m = e.notna() & sig.notna()
    return e[m & (sig > 0)], e[m & (sig < 0)]


def stationary_boot_idx(n: int, reps: int, mean_block: int = 10) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (reps x n)."""
    idx = np.empty((reps, n), dtype=np.int64)
    p = 1.0 / mean_block
    start = RNG.integers(0, n, size=(reps,))
    idx[:, 0] = start
    jumps = RNG.random((reps, n)) < p
    newpos = RNG.integers(0, n, size=(reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jumps[:, t], newpos[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def test(e: pd.Series, sig: pd.Series, reps: int = 2000) -> dict:
    """Difference (after-up minus after-down) with dependence-robust inference."""
    sig = sig.reindex(e.index)
    m = e.notna() & sig.notna() & (sig != 0)
    ev, sv = e[m].to_numpy(), sig[m].to_numpy()
    up, dn = ev[sv > 0], ev[sv < 0]
    diff = up.mean() - dn.mean()
    se = math.sqrt(up.var(ddof=1) / len(up) + dn.var(ddof=1) / len(dn))
    # block bootstrap of the joint (signal, edge) series -> CI of the difference
    bi = stationary_boot_idx(len(ev), reps)
    E, S = ev[bi], sv[bi]
    bd = (np.nanmean(np.where(S > 0, E, np.nan), 1) - np.nanmean(np.where(S < 0, E, np.nan), 1))
    lo, hi = np.quantile(bd, [0.05, 0.95])
    # circular-shift test: misalign signal and edge by a random lag (keeps each series' own clustering)
    n = len(ev)
    shifts = np.arange(30, n - 30)
    null = np.array([ev[np.roll(sv, k) > 0].mean() - ev[np.roll(sv, k) < 0].mean() for k in shifts])
    p_shift = (null >= diff).mean()
    # robust versions
    q1, q99 = np.quantile(ev, [0.01, 0.99])
    tr = np.clip(ev, q1, q99)
    return {"n up": len(up), "n down": len(dn), "edge after UP": up.mean(), "edge after DOWN": dn.mean(),
            "diff": diff, "t": diff / se, "CI90 lo": lo, "CI90 hi": hi, "p shift": p_shift,
            "diff winsorized 1%": tr[sv > 0].mean() - tr[sv < 0].mean(),
            "diff of medians": np.median(up) - np.median(dn)}


def strat_pnl(e_bps: pd.Series, S: pd.Series, sig: pd.Series, size_up: float, size_dn: float) -> pd.Series:
    """USD PnL per day selling `size` straddles (2 options each) on 1 ETH notional."""
    sig = sig.reindex(e_bps.index)
    size = np.where(sig > 0, size_up, np.where(sig < 0, size_dn, (size_up + size_dn) / 2))
    return pd.Series(2 * size * e_bps.to_numpy() / 1e4 * S.to_numpy(), index=e_bps.index).fillna(0.0)


# ------------------------------------------------------------------ signal variants (need hourly prices)
def signal_variants(d: pd.DataFrame, asset: str) -> dict[str, pd.Series]:
    h = fetch_hourly(date(2021, 3, 25), date(2026, 11, 23), f"{asset}USDT")
    px = h["open"]

    def at(offset_h):
        return px.reindex(d.index + pd.Timedelta(hours=offset_h)).to_numpy()

    now = at(0)
    r24 = (d.close / d.open - 1).shift(1)
    out = {
        "base: prev 24h candle (08->08)": np.sign(r24),
        "prev UTC calendar day (00->00)": pd.Series(np.sign(at(-8) / at(-32) - 1), d.index),
        "last 12h (20->08)": pd.Series(np.sign(now / at(-12) - 1), d.index),
        "last 4h (04->08)": pd.Series(np.sign(now / at(-4) - 1), d.index),
        "last 48h": pd.Series(np.sign(now / at(-48) - 1), d.index),
        "last 72h": pd.Series(np.sign(now / at(-72) - 1), d.index),
        "prev 24h, only moves > 1%": pd.Series(np.where(r24 > 0.01, 1, np.where(r24 < -0.01, -1, np.nan)), d.index),
        "prev 24h, only moves > 3%": pd.Series(np.where(r24 > 0.03, 1, np.where(r24 < -0.03, -1, np.nan)), d.index),
    }
    return out


# ------------------------------------------------------------------ main
def main():
    rows, notes = [], {}
    eth, btc = build_daily(), build_daily(asset="BTC")
    sig_e, sig_b = prev_candle(eth), prev_candle(btc)
    e_eth = edge_bps(eth, eth.dvol * 0.95)
    e_btc = edge_bps(btc, btc.dvol * 0.95)

    def add(group, label, e, sig, **kw):
        r = test(e, sig, **kw)
        r.update({"group": group, "check": label})
        rows.append(r)
        return r

    base = add("A. base", "ETH 2021-26, DVOL pricing", e_eth, sig_e)

    # ---- time stability
    for y in sorted(set(eth.index.year)):
        m = eth.index.year == y
        add("B. by year", f"ETH {y}", e_eth[m], sig_e[m], reps=500)
    half = eth.index[len(eth) // 2]
    add("B. halves", f"ETH 1st half (to {half:%Y-%m})", e_eth[:half], sig_e[:half])
    add("B. halves", f"ETH 2nd half (from {half:%Y-%m})", e_eth[half:], sig_e[half:])
    wk = eth.index.dayofweek >= 5
    add("B. weekday/weekend", "ETH weekdays (Mon-Fri)", e_eth[~wk], sig_e[~wk])
    add("B. weekday/weekend", "ETH weekends (Sat-Sun)", e_eth[wk], sig_e[wk], reps=500)

    # ---- neighbouring definitions
    for lab, s in signal_variants(eth, "ETH").items():
        if lab.startswith("base"):
            continue
        add("C. signal variants", lab, e_eth, s)

    # ---- other asset
    add("D. other asset", "BTC 2021-26, BTC DVOL pricing", e_btc, sig_b)

    # ---- out-of-sample period, model-free & proxy priced
    rv_ratio = {}
    for asset, dd in (("ETH", eth), ("BTC", btc)):
        sp = spot_daily(asset)
        r = np.log(sp.close / sp.open)
        rv30 = r.rolling(30).std().shift(1) * math.sqrt(365)            # trailing realised vol (known at 08:00)
        k = (dd.dvol / rv30.reindex(dd.index)).median()                  # typical DVOL / RV ratio in 2021-26
        rv_ratio[asset] = k
        proxy_iv = rv30 * k * 0.95
        e_proxy = edge_bps(sp, proxy_iv)
        s_sp = prev_candle(sp)
        pre = sp.index < pd.Timestamp("2021-04-01", tz="UTC")
        add("E. out-of-sample 2017-21", f"{asset} Sep 2017 - Mar 2021, IV = trailing RV x {k:.2f}",
            e_proxy[pre], s_sp[pre])
        # pure price test: next-day |move| relative to trailing vol (no option pricing at all)
        surprise = (np.abs(r) / (rv30 / math.sqrt(365)))
        u, dn_ = split(surprise[pre], s_sp[pre])
        notes[f"{asset} 2017-21 next-day |move| / trailing daily vol: after UP {u.mean():.3f}, after DOWN {dn_.mean():.3f}"] = None

    # ---- REAL 1-day option IVs from Deribit trades
    real = {}
    for asset, dd, sig in (("ETH", eth, sig_e), ("BTC", btc, sig_b)):
        f = ROOT / "data" / f"{asset.lower()}_1d_option_iv.csv"
        if not f.exists():
            continue
        iv = pd.read_csv(f, parse_dates=["t"], index_col="t")
        iv = iv[iv.n >= 5]
        sell_iv = iv.iv_sell.fillna(iv.iv_mid * 0.97)
        e_real = edge_bps(dd, sell_iv)
        real[asset] = (iv, e_real)
        add("F. REAL 1-day option IV", f"{asset}: priced at real Deribit 1-day IV (seller fills)", e_real, sig)
        ratio = (iv.iv_mid / dd.dvol.reindex(iv.index))
        u, dn_ = split(ratio, sig)
        notes[f"{asset} real 1-day IV / DVOL: after UP {u.mean():.3f}, after DOWN {dn_.mean():.3f} "
              f"(days with data {len(iv)})"] = None
        e_dvol_same_days = e_eth if asset == "ETH" else e_btc
        add("F. REAL 1-day option IV", f"{asset}: DVOL pricing on the same days (comparison)",
            e_dvol_same_days.reindex(iv.index), sig)

    # ---- G. how wrong was DVOL pricing? edge by pricing source, and earlier strategies re-run
    price_t, rerun_t = None, None
    if "ETH" in real:
        iv, _ = real["ETH"]
        sell_iv = iv.iv_sell.fillna(iv.iv_mid * 0.97)
        wk = iv.index.dayofweek >= 5
        sg = sig_e.reindex(iv.index)
        r_abs = np.abs(eth.close / eth.open - 1).reindex(iv.index)
        pr = {}
        for lab, ivs in (("DVOL x0.95 (used in studies 1-2)", eth.dvol * 0.95), ("real 1-day IV, mid", iv.iv_mid),
                         ("real 1-day IV, seller fills", sell_iv)):
            e = edge_bps(eth, ivs).reindex(iv.index)
            pr[lab] = {"IV used %": 100 * ivs.reindex(iv.index).mean(), "edge all days": e.mean(),
                       "after UP": e[sg > 0].mean(), "after DOWN": e[sg < 0].mean(),
                       "weekdays": e[~wk].mean(), "weekends": e[wk].mean()}
        pr["fair IV (from realised |moves|)"] = {"IV used %": 100 * (r_abs * math.sqrt(math.pi / 2) * math.sqrt(365)).mean()}
        price_t = pd.DataFrame(pr).T
        rp = eth.copy()
        rp["dvol"] = sell_iv.reindex(eth.index)
        rp = rp.dropna(subset=["dvol"])
        rr = []
        for lab, cfg in (("Study 1 A: plain call selling", Config(mode="plain")),
                         ("Study 1 B: your perp grid", Config()),
                         ("Study 1 J: grid + stop at BE+10%", Config(stop_pct=0.10))):
            a, b = run(eth, cfg), run(rp, Config(**{**cfg.__dict__, "iv_mult": 1.0}))
            sa, sb = stats(a), stats(b)
            rr.append({"strategy": lab, "PnL $ (DVOL x0.95)": sa["final_pnl"], "max DD $ (DVOL)": sa["max_drawdown"],
                       "PnL $ (real IV)": sb["final_pnl"], "max DD $ (real IV)": sb["max_drawdown"],
                       "of which funding $ (real IV)": sb["funding_total"]})
        for lab, rule in (("Study 2: your candle rule", "momentum"), ("Study 2: always sell call", "always_call"),
                          ("Study 2: always sell put", "always_put"), ("Study 2: half straddle daily", "half_straddle"),
                          ("Study 2: straddle after UP only", "straddle_after_up")):
            sa, sb = summary(simulate(eth, DCfg(rule=rule))), summary(simulate(rp, DCfg(rule=rule, iv_mult=1.0)))
            rr.append({"strategy": lab, "PnL $ (DVOL x0.95)": sa["PnL $"], "max DD $ (DVOL)": sa["max DD $"],
                       "PnL $ (real IV)": sb["PnL $"], "max DD $ (real IV)": sb["max DD $"],
                       "of which funding $ (real IV)": 0.0})
        rerun_t = pd.DataFrame(rr).set_index("strategy")
        rerun_t.to_csv(OUT / "rerun_real_iv.csv")
        price_t.to_csv(OUT / "edge_by_pricing.csv")

    res = pd.DataFrame(rows).set_index(["group", "check"])
    res.to_csv(OUT / "checks.csv")

    # ---- strategy-level comparison (USD per 1 ETH notional), DVOL and real IV
    strat_rows = []
    sizing = {"straddle every day (1x)": (1, 1), "after UP 1x, after DOWN 0.5x": (1, 0.5),
              "after UP 1x, after DOWN 0x": (1, 0)}
    pricings = [("ETH, DVOL x0.95", e_eth, eth, sig_e)]
    if "ETH" in real:
        iv, e_real = real["ETH"]
        pricings.append(("ETH, real 1-day IV", e_real, eth, sig_e))
    curves = {}
    for plab, e, dd, sg in pricings:
        idx = e.dropna().index
        for slab, (su, sd) in sizing.items():
            p = strat_pnl(e.loc[idx], dd.open.reindex(idx), sg, su, sd)
            eq = p.cumsum()
            opt = (2 * np.where(sg.reindex(idx) > 0, su, sd)).sum()
            strat_rows.append({"pricing": plab, "sizing": slab, "days": len(idx), "PnL $": eq.iloc[-1],
                               "PnL $ per option": eq.iloc[-1] / opt, "max DD $": (eq - eq.cummax()).min(),
                               "Sharpe": p.mean() / p.std() * math.sqrt(365)})
            curves[(plab, slab)] = eq
    strat = pd.DataFrame(strat_rows).set_index(["pricing", "sizing"])
    strat.to_csv(OUT / "strategy.csv")

    # ---------------------------------------------------------------- charts
    # forest plot of the difference (after-UP edge minus after-DOWN edge)
    plot = res[~res.index.get_level_values(0).str.startswith("B. by year")]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(plot) + 1.4))
    y = np.arange(len(plot))[::-1]
    for yi, ((g, c), r) in zip(y, plot.iterrows()):
        col = BLUE if r["CI90 lo"] > 0 else (ORANGE if r["CI90 hi"] < 0 else INK2)
        ax.plot([r["CI90 lo"], r["CI90 hi"]], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r["diff"], yi, "o", color=col, ms=6, mec=SURF, mew=1.2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}" for g, c in plot.index], fontsize=8)
    ax.set_xlabel("edge after UP minus after DOWN candle, bps of spot per option\n(dot = estimate, bar = 90% block-bootstrap CI)")
    ax.set_title("Is selling after UP candles really better?  Blue = CI excludes 0, gray = could be noise")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_forest.png", dpi=160)
    plt.close(fig)

    if "ETH" in real:
        iv, e_real = real["ETH"]
        ratio = (iv.iv_mid / eth.dvol.reindex(iv.index)).dropna()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        u, dn_ = split(ratio, sig_e)
        bins = np.linspace(0.4, 1.6, 50)
        ax[0].hist(u.clip(0.4, 1.6), bins=bins, color=BLUE, alpha=0.55, label=f"after UP candle (mean {u.mean():.2f})")
        ax[0].hist(dn_.clip(0.4, 1.6), bins=bins, color=ORANGE, alpha=0.55, label=f"after DOWN candle (mean {dn_.mean():.2f})")
        ax[0].set_title("ETH: real 1-day ATM IV at 08-11 UTC  /  30-day DVOL")
        ax[0].set_xlabel("ratio")
        ax[0].set_ylabel("days")
        ax[0].legend(fontsize=8)
        for (plab, slab), eq in curves.items():
            if slab == "straddle every day (1x)":
                continue
            col = BLUE if "UP 1x, after DOWN 0x" in slab else AQUA
            ls = "-" if "real" in plab else (0, (1, 0))
            alpha = 1.0 if "real" in plab else 0.45
            ax[1].plot(eq.index, eq, color=col, alpha=alpha, lw=1.6 if "real" in plab else 1.1,
                       label=f"{slab} - {plab}")
        ax[1].axhline(0, color=INK2, lw=0.6)
        ax[1].set_title("Cumulative PnL, USD per 1 ETH straddle")
        ax[1].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / "fig2_real_iv.png", dpi=160)
        plt.close(fig)

    def fmt(df):
        f = {"n up": ",.0f", "n down": ",.0f", "p shift": ".3f", "t": ".2f", "days": ",.0f", "Sharpe": ".2f",
             "PnL $": ",.0f", "max DD $": ",.0f", "PnL $ per option": ".2f"}
        return df.to_markdown(floatfmt=[""] + [f.get(c, ".1f") for c in df.columns])

    txt = ["# Reliability of 'sell straddles after UP candles' (auto-generated by `run_reliability.py`)\n",
           "Edge = premium - fee - |move|/2 per ATM 1-day option sold at 08:00 UTC, in **bps of spot**. "
           "`diff` = edge after UP candle minus after DOWN candle. CI90 = 90% stationary block bootstrap "
           "(mean block 10 days). `p shift` = circular-shift test (signal misaligned by every lag 30..n-30).\n",
           "## Checks\n", fmt(res.reset_index(level=0, drop=True)), "\n",
           "## Notes\n", "\n".join(f"- {k}" for k in notes), "\n",
           "## Strategy level\n", fmt(strat), "\n"]
    if price_t is not None:
        txt += ["## Edge per option by pricing source (ETH, bps of spot)\n", price_t.to_markdown(floatfmt=".1f"), "\n",
                "## Earlier strategies re-run at REAL 1-day option prices (ETH, USD per 1 ETH/day)\n",
                rerun_t.to_markdown(floatfmt=",.0f"), "\n"]
    (OUT / "RESULTS.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
