"""BUY weekly ATM ETH options in the direction of the previous weekly candle,
optionally with martingale sizing.

Week = Friday 08:00 UTC -> next Friday 08:00 UTC (Deribit weekly expiry).
Previous W1 candle = the previous Friday->Friday week (known at entry).
  prev week UP   -> buy ATM weekly CALL
  prev week DOWN -> buy ATM weekly PUT
Premium = Black-Scholes ATM at the REAL Deribit weekly IV paid by buyers
(median IV of taker-buy trades in the next-Friday expiry, Fri 08:00-11:00).

Martingale sizing ("an X% move must cover earlier losses"):
  L    = losses since the last equity high  (high-water mark - equity)
  g    = net gain per 1 ETH of options if price moves X% our way = X%*S - premium
  size = base + L / g          ("recover" mode: an X% move repays L and earns the base trade's profit)
  size = max(base, L / g)      ("cover" mode: an X% move just repays L)
With a finite account, a week's premium can't exceed the cash left; cash <= 0 means ruin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WCfg:
    rule: str = "momentum"     # momentum | contrarian | always_call | always_put | straddle
    sizing: str = "flat"       # flat | recover | cover
    move: float = 0.08         # martingale target move
    base: float = 1.0          # ETH per base trade
    base_usd: float = 0.0      # if > 0: base trade = this USD notional (base ETH = base_usd / S0)
    capital: float = math.inf  # USD account; inf = unconstrained
    max_size: float = math.inf
    opt_fee: float = 0.0003
    deliv_fee: float = 0.00015
    fee_cap: float = 0.125


def weekly_table(spot: pd.DataFrame, iv: pd.DataFrame) -> pd.DataFrame:
    """One row per Friday: entry S0, settle S1, previous-week return, buyer's IV."""
    fr = spot.open[spot.index.dayofweek == 4]
    w = pd.DataFrame({"S0": fr})
    w["S1"] = fr.shift(-1)
    w["prev_ret"] = fr / fr.shift(1) - 1
    gap = (w.index.to_series().diff().dt.days != 7) | (w.index.to_series().diff(-1).dt.days != -7)
    w.loc[gap, "S1"] = np.nan  # missing data around this Friday
    ivb = iv.iv_buy.fillna(iv.iv_mid * 1.03)
    w["iv"] = ivb.reindex(w.index).ffill()
    w["iv_mid"] = iv.iv_mid.reindex(w.index).ffill()
    return w.dropna(subset=["S1", "prev_ret", "iv"])


def premium(S, iv, T=7 / 365):
    x = np.asarray(iv, float) * math.sqrt(T) / 2.0
    return np.asarray(S, float) * np.vectorize(lambda v: math.erf(v / math.sqrt(2.0)))(x)


def run(w: pd.DataFrame, cfg: WCfg, signal: np.ndarray | None = None) -> pd.DataFrame:
    S0, S1 = w.S0.to_numpy(), w.S1.to_numpy()
    prem = premium(S0, w.iv.to_numpy())
    sig = np.sign(w.prev_ret.to_numpy()) if signal is None else signal
    n = len(w)
    if cfg.rule == "momentum":
        qc, qp = (sig > 0) * 1.0, (sig < 0) * 1.0
    elif cfg.rule == "contrarian":
        qc, qp = (sig < 0) * 1.0, (sig > 0) * 1.0
    elif cfg.rule == "always_call":
        qc, qp = np.ones(n), np.zeros(n)
    elif cfg.rule == "always_put":
        qc, qp = np.zeros(n), np.ones(n)
    elif cfg.rule == "straddle":
        qc, qp = np.full(n, 0.5), np.full(n, 0.5)
    else:
        raise ValueError(cfg.rule)
    payoff = qc * np.maximum(S1 - S0, 0) + qp * np.maximum(S0 - S1, 0)
    fee_in = np.minimum(cfg.opt_fee * S0, cfg.fee_cap * prem)
    fee_out = np.minimum(cfg.deliv_fee * S1, cfg.fee_cap * payoff)
    cost_u = prem + fee_in                      # cash paid per 1 ETH of options
    unit = payoff - fee_out - cost_u            # PnL per 1 ETH of options
    gain_u = cfg.move * S0 - prem               # net gain per unit at the target move

    eq, hwm = 0.0, 0.0
    rows = []
    ruined = False
    for i in range(n):
        L = hwm - eq
        b = cfg.base_usd / S0[i] if cfg.base_usd > 0 else cfg.base
        if cfg.sizing == "flat" or gain_u[i] <= 0:
            # flat, or premium >= target move (an X% move can't even repay the premium): base size
            size = b
        elif cfg.sizing == "recover":
            size = b + L / gain_u[i]
        else:
            size = max(b, L / gain_u[i])
        size = min(size, cfg.max_size)
        cash = cfg.capital + eq
        capped = False
        if size * cost_u[i] > cash:
            size, capped = max(cash, 0.0) / cost_u[i], True
        if cash <= 1e-6 or ruined:
            size, ruined = 0.0, True
        pnl = size * unit[i]
        eq += pnl
        hwm = max(hwm, eq)
        rows.append((w.index[i], S0[i], S1[i], qc[i] > 0, size, size * cost_u[i], pnl, eq, L, capped, ruined))
    return pd.DataFrame(rows, columns=["t", "S0", "S1", "call", "size", "outlay", "pnl", "equity",
                                       "loss_to_recover", "capped", "ruined"]).set_index("t")


def summary(r: pd.DataFrame, capital: float = math.inf) -> dict:
    eq = r.equity
    dd = eq - eq.cummax()
    traded = r[r["size"] > 0]
    lose = (traded.pnl < 0).astype(int).to_numpy()
    streak = cur = 0
    for x in lose:
        cur = cur + 1 if x else 0
        streak = max(streak, cur)
    # capital needed so that no week's premium ever had to be cut
    need = (r.outlay - (eq.shift(1).fillna(0.0))).max()
    return {"PnL $": eq.iloc[-1], "max DD $": dd.min(), "win weeks %": 100 * (traded.pnl > 0).mean(),
            "longest losing streak": streak, "max size ETH": r["size"].max(),
            "max weekly premium $": r.outlay.max(), "capital needed $": max(need, 0.0),
            "ruined": bool(r.ruined.any()), "weeks": len(r)}
