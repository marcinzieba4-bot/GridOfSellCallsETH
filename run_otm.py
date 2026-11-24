"""Study 5: weekly momentum with OTM options (flat size, no martingale).

Every Friday 08:00 UTC: previous Fri->Fri week UP -> buy a weekly CALL with strike S*(1+x);
DOWN -> buy a weekly PUT with strike S*(1-x). Hold to next Friday 08:00 expiry.
x in {0 (ATM), 2.5, 5, 7.5, 10, 15}%.

Pricing: REAL Deribit smile. For each Friday we take all trades (08:00-16:00 UTC) in the
next-Friday expiry, bucket them by moneyness (2.5% buckets) and take the median IV per
bucket, then add the week's buyer premium (taker-buy IV minus bucket median). Thin weeks
fall back to the average smile shape x that week's ATM buyer IV.
Fees: 0.03% of spot per option (capped at 12.5% of premium) + 0.015% delivery (cap 12.5% of value).

Sizing: a fixed premium budget per trade, so results read as "return on premium spent" (R).
Output: results/weekly_otm/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridsim.data import spot_daily
from gridsim.weekly import weekly_table

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "weekly_otm"
OUT.mkdir(parents=True, exist_ok=True)
SMILE = ROOT / "data" / "raw" / "deribit_trades" / "ETH_7d_smile"
RNG = np.random.default_rng(7)
T = 7 / 365
LEVELS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]
BUCKETS = np.round(np.arange(-0.30, 0.3001, 0.025), 3)

BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # ordinal: ATM -> 15% OTM
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.4,
})


# ------------------------------------------------------------------ pricing
def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, iv, cp):
    s = iv * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s) / s
    d2 = d1 - s
    if cp == "C":
        return S * ncdf(d1) - K * ncdf(d2)
    return K * ncdf(-d2) - S * ncdf(-d1)


def week_smile(day: pd.Timestamp) -> tuple[dict, float, int]:
    """Bucket medians of IV (fraction) by moneyness, buyer premium (vol pts), trade count."""
    f = SMILE / f"{day:%Y-%m-%d}.json"
    if not f.exists():
        return {}, np.nan, 0
    tr = pd.DataFrame(json.loads(f.read_text()), columns=["ts", "K", "S", "cp", "dir", "iv", "amt"])
    if tr.empty:
        return {}, np.nan, 0
    tr["m"] = tr.K / tr.S - 1
    tr["b"] = (tr.m / 0.025).round() * 0.025
    tr["b"] = tr["b"].round(3)
    tr = tr[(tr.iv > 5) & (tr.iv < 400)]
    g = tr.groupby("b").iv
    med = g.median()[g.size() >= 2] / 100
    buys = tr[tr.dir == "buy"]
    prem = (buys.iv / 100 - buys.b.map(med)).dropna()
    return med.to_dict(), (prem.median() if len(prem) >= 5 else np.nan), len(tr)


def build_iv_table(w: pd.DataFrame, atm_buy: pd.Series) -> pd.DataFrame:
    smiles = {d: week_smile(d) for d in w.index}
    # pooled smile shape (ratio to ATM bucket) for fallback, and typical buyer premium
    ratios = {b: [] for b in BUCKETS}
    prems = []
    for med, pr, n in smiles.values():
        if 0.0 in med and n >= 30:
            for b, v in med.items():
                if b in ratios:
                    ratios[b].append(v / med[0.0])
        if not np.isnan(pr):
            prems.append(pr)
    pooled = {b: (np.median(v) if len(v) >= 20 else np.nan) for b, v in ratios.items()}
    pooled_prem = float(np.median(prems))
    rows = []
    for d in w.index:
        med, pr, n = smiles[d]
        pr = pooled_prem if np.isnan(pr) else pr
        row = {"t": d, "n_trades": n}
        for x in LEVELS:
            for cp, m in (("C", round(x, 3)), ("P", round(-x, 3))):
                src = "own"
                if m in med:
                    iv = med[m] + pr
                else:
                    # interpolate between neighbouring buckets of this week (max 5% apart)
                    lo = [b for b in med if b < m and m - b <= 0.05]
                    hi = [b for b in med if b > m and b - m <= 0.05]
                    if lo and hi:
                        b0, b1 = max(lo), min(hi)
                        iv = med[b0] + (med[b1] - med[b0]) * (m - b0) / (b1 - b0) + pr
                    else:
                        iv = atm_buy.get(d, np.nan) * pooled.get(m, np.nan)
                        src = "pooled"
                row[f"iv_{cp}{x:.3f}"] = iv
                row[f"src_{cp}{x:.3f}"] = src
        rows.append(row)
    return pd.DataFrame(rows).set_index("t"), pooled, pooled_prem


# ------------------------------------------------------------------ trades
def returns_on_premium(w: pd.DataFrame, ivt: pd.DataFrame, x: float, direction: np.ndarray) -> np.ndarray:
    """R per week = (payoff - delivery fee) / (premium + entry fee) - 1, for the given +1/-1 directions."""
    out = np.full(len(w), np.nan)
    for i, (S0, S1) in enumerate(zip(w.S0.to_numpy(), w.S1.to_numpy())):
        cp = "C" if direction[i] > 0 else "P"
        iv = ivt[f"iv_{cp}{x:.3f}"].iloc[i]
        if not np.isfinite(iv):
            continue
        K = S0 * (1 + x) if cp == "C" else S0 * (1 - x)
        prem = bs(S0, K, iv, cp)
        cost = prem + min(0.0003 * S0, 0.125 * prem)
        pay = max(S1 - K, 0.0) if cp == "C" else max(K - S1, 0.0)
        pay -= min(0.00015 * S1, 0.125 * pay)
        out[i] = pay / cost - 1
    return out


def streak(r):
    best = cur = 0
    for v in r:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def block_ci(r, reps=3000, block=8):
    n = len(r)
    nb = math.ceil(n / block)
    means = []
    for _ in range(reps):
        st = RNG.integers(0, n - block, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        means.append(r[idx].mean())
    return np.quantile(means, [0.05, 0.95])


def main():
    sp = spot_daily("ETH")
    ivw = pd.read_csv(ROOT / "data" / "eth_7d_option_iv.csv", parse_dates=["t"], index_col="t")
    w = weekly_table(sp, ivw)
    atm_buy = w.iv
    ivt, pooled, pooled_prem = build_iv_table(w, atm_buy)
    ok = ivt.filter(like="iv_").notna().all(axis=1)
    w, ivt = w[ok], ivt[ok]
    sig = np.sign(w.prev_ret.to_numpy())
    half = w.index[len(w) // 2]
    first = (w.index < half)

    rows, curves, perms = [], {}, {}
    for x in LEVELS:
        lab = "ATM" if x == 0 else f"{x:.1%} OTM"
        r = returns_on_premium(w, ivt, x, sig)
        r_rev = returns_on_premium(w, ivt, x, -sig)
        r_call = returns_on_premium(w, ivt, x, np.ones(len(w)))
        r_put = returns_on_premium(w, ivt, x, -np.ones(len(w)))
        perm = np.array([np.nanmean(returns_on_premium(w, ivt, x, RNG.permutation(sig))) for _ in range(400)])
        lo, hi = block_ci(r)
        top5 = np.sort(r)[-5:].sum()
        own = (ivt[f"src_C{x:.3f}"].eq("own") | ivt[f"src_P{x:.3f}"].eq("own")).mean()
        rows.append({
            "strike": lab, "weeks": len(r), "avg R / trade %": 100 * r.mean(), "CI90 lo %": 100 * lo,
            "CI90 hi %": 100 * hi, "median R %": 100 * np.median(r), "win weeks %": 100 * (r > 0).mean(),
            "longest losing streak": streak(r), "best week R x": r.max(),
            "total R without best 5 weeks": r.sum() - top5, "total R": r.sum(),
            "p vs random direction": (perm >= r.mean()).mean(),
            "reverse avg R %": 100 * r_rev.mean(), "always call avg R %": 100 * r_call.mean(),
            "always put avg R %": 100 * r_put.mean(),
            "1st half avg R %": 100 * r[first].mean(), "2nd half avg R %": 100 * r[~first].mean(),
            "priced from own-week trades %": 100 * own,
        })
        curves[lab] = pd.Series(np.cumsum(r), index=w.index)
        perms[lab] = (r, r_rev)
    res = pd.DataFrame(rows).set_index("strike")
    res.to_csv(OUT / "by_strike.csv")

    # ---- by period, and the signal's value over a random direction (avg of always-call and always-put)
    periods = [("2019-20", "2019", "2020"), ("2021-22", "2021", "2022"), ("2023-24", "2023", "2024"),
               ("2025-26", "2025", "2026")]
    yrs = w.index.year.astype(str)
    per_rows, base_rows = [], []
    for x in LEVELS:
        lab = "ATM" if x == 0 else f"{x:.1%} OTM"
        r, _ = perms[lab]
        rc = returns_on_premium(w, ivt, x, np.ones(len(w)))
        rp = returns_on_premium(w, ivt, x, -np.ones(len(w)))
        rnd = (rc + rp) / 2
        pr, br = {"strike": lab}, {"strike": lab}
        for name, a, b in periods:
            m = (yrs >= a) & (yrs <= b)
            pr[name] = 100 * r[m].mean()
            br[name] = 100 * (r[m].mean() - rnd[m].mean())
        per_rows.append(pr)
        base_rows.append(br)
    per_t = pd.DataFrame(per_rows).set_index("strike")
    sigval_t = pd.DataFrame(base_rows).set_index("strike")

    # ---- cost stress: pay more than the typical fill (+2 / +5 vol points on every purchase)
    stress = []
    for x in LEVELS:
        lab = "ATM" if x == 0 else f"{x:.1%} OTM"
        row = {"strike": lab, "as measured": 100 * perms[lab][0].mean()}
        for add in (0.02, 0.05):
            iv2 = ivt.copy()
            cols = [c for c in iv2.columns if c.startswith("iv_")]
            iv2[cols] = iv2[cols] + add
            row[f"+{add * 100:.0f} vol pts"] = 100 * np.nanmean(returns_on_premium(w, iv2, x, sig))
        stress.append(row)
    stress_t = pd.DataFrame(stress).set_index("strike")

    # ---- $10k account, flat weekly premium budget (no compounding, no martingale)
    acct = []
    for x in LEVELS:
        lab = "ATM" if x == 0 else f"{x:.1%} OTM"
        r = perms[lab][0]
        for budget in (100, 200, 500):
            eq = 10_000 + np.cumsum(budget * r)
            acct.append({"strike": lab, "weekly premium $": budget, "end $": eq[-1], "worst $": eq.min(),
                         "max DD $": (eq - np.maximum.accumulate(eq)).min()})
    acct_t = pd.DataFrame(acct).set_index(["strike", "weekly premium $"])

    # ---- 3-year block bootstrap of the account at $200/week
    boot = []
    for x in LEVELS:
        lab = "ATM" if x == 0 else f"{x:.1%} OTM"
        r = perms[lab][0]
        ends, dds = [], []
        for _ in range(4000):
            st = RNG.integers(0, len(r) - 8, math.ceil(156 / 8))
            idx = (st[:, None] + np.arange(8)).ravel()[:156]
            eq = 10_000 + np.cumsum(200 * r[idx])
            ends.append(eq[-1])
            dds.append((eq - np.maximum.accumulate(eq)).min())
        ends = np.array(ends)
        boot.append({"strike": lab, "P(end below $10k) %": 100 * (ends < 10_000).mean(),
                     "P(end below $5k) %": 100 * (ends < 5_000).mean(), "median end $": np.median(ends),
                     "5% worst end $": np.quantile(ends, 0.05), "95% best end $": np.quantile(ends, 0.95),
                     "median max DD $": np.median(dds)})
    boot_t = pd.DataFrame(boot).set_index("strike")

    # ---- smile actually used (avg buyer IV / ATM buyer IV)
    sm = pd.DataFrame({"moneyness": [f"{'+' if cp == 'C' else '-'}{x:.1%}" for x in LEVELS for cp in ("C", "P")],
                       "type": [cp for x in LEVELS for cp in ("C", "P")],
                       "avg buyer IV %": [100 * ivt[f"iv_{cp}{x:.3f}"].mean() for x in LEVELS for cp in ("C", "P")]})

    # ---------------------------------------------------------------- charts
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    xs = np.arange(len(LEVELS))
    a = res["avg R / trade %"].to_numpy()
    ax[0].errorbar(xs - 0.08, a, yerr=[a - res["CI90 lo %"], res["CI90 hi %"] - a], fmt="o", color=BLUE,
                   ecolor=BLUE, elinewidth=2, capsize=0, ms=6, label="your rule (90% block-bootstrap CI)")
    ax[0].plot(xs + 0.08, res["reverse avg R %"], "o", color=ORANGE, ms=6, label="reverse rule")
    ax[0].axhline(0, color=INK, lw=0.8)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(res.index, fontsize=8)
    ax[0].set_ylabel("% of premium spent")
    ax[0].set_title("Average return per trade, by strike")
    ax[0].legend(fontsize=8, loc="upper left")
    for (lab, c), col in zip(curves.items(), RAMP):
        ax[1].plot(c.index, c, color=col, lw=1.4, label=f"{lab} ({c.iloc[-1]:+.0f}x)")
    ax[1].axhline(0, color=INK2, lw=0.6)
    ax[1].set_title("Your rule: cumulative return, in units of the weekly premium budget")
    ax[1].set_ylabel("x weekly budget")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_by_strike.png", dpi=160)
    plt.close(fig)

    def md(df, fm=None):
        fm = fm or {}
        return df.to_markdown(floatfmt=[""] + [fm.get(c, ",.1f") for c in df.columns])

    txt = [
        "# Study 5: weekly momentum with OTM options (auto-generated by `run_otm.py`)\n",
        f"{len(w)} weeks {w.index[0].date()} -> {w.index[-1].date()}. Up week -> buy call K=S(1+x), down week -> "
        "buy put K=S(1-x), weekly expiry, held to Friday 08:00 UTC. Priced at REAL Deribit smile (buyer side; "
        f"pooled buyer premium {100 * pooled_prem:.1f} vol pts). R = return on premium spent per trade. "
        "Flat size, no martingale.\n",
        "## 1. By strike\n",
        md(res, {"weeks": ".0f", "longest losing streak": ".0f", "best week R x": ".1f",
                 "total R without best 5 weeks": ".1f", "total R": ".1f", "p vs random direction": ".3f"}), "\n",
        "## 1b. Your rule, avg R per trade % by period\n", md(per_t), "\n",
        "## 1c. Value of the momentum signal: your rule minus random direction (avg of always-call & always-put), % of premium\n",
        md(sigval_t), "\n",
        "## 1d. Cost stress: avg R per trade % if every purchase costs more IV than the typical fill\n", md(stress_t), "\n",
        "## 2. $10,000 account, flat weekly premium budget, history\n", md(acct_t, {c: ",.0f" for c in acct_t.columns}), "\n",
        "## 3. Block bootstrap (8-week blocks), 3 years, $10k account, $200/week premium\n",
        md(boot_t, {"median end $": ",.0f", "5% worst end $": ",.0f", "95% best end $": ",.0f", "median max DD $": ",.0f"}), "\n",
        "## 4. Smile used (average buyer IV)\n", sm.set_index("moneyness").to_markdown(floatfmt=".1f"), "\n",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
