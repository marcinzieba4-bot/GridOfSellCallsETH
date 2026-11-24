"""Sell one daily ATM option per day, chosen by the previous daily candle, and
take the whole loss (no hedge, no perps).

Day i: option sold at open[i] (08:00 UTC), strike K = open[i], settles at close[i]
(next 08:00). Previous candle = day i-1 (08:00 -> 08:00), known at sale time.

Rules ("signal" for each day is +1 up / -1 down / 0 flat for the previous candle):
  momentum  (your rule): prev DOWN -> sell CALL,   prev UP -> sell PUT
  contrarian           : prev UP   -> sell CALL,   prev DOWN -> sell PUT
  always_call / always_put / straddle (both, 2 options) as baselines
  half_straddle        : 0.5 call + 0.5 put every day (same option count as one-per-day)
  straddle_after_up    : 1 call + 1 put after an UP candle, nothing after a DOWN candle
Optional |prev move| threshold: trade only if the previous candle moved >= thr.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from math import erf, sqrt


@dataclass
class DCfg:
    rule: str = "momentum"   # momentum | contrarian | always_call | always_put | straddle
    iv_mult: float = 0.95
    opt_fee: float = 0.0003
    opt_fee_cap: float = 0.125
    thr: float = 0.0         # min |previous candle return| to trade (e.g. 0.02 = 2%)
    T_days: float = 1.0


def atm_premium(S: np.ndarray, iv: np.ndarray, T: float) -> np.ndarray:
    """ATM call == ATM put for r=0 (put-call parity at K=S)."""
    x = iv * math.sqrt(T) / 2.0
    return S * np.vectorize(lambda v: erf(v / sqrt(2.0)))(x)


def legs(d: pd.DataFrame, cfg: DCfg, signal: np.ndarray | None = None) -> pd.DataFrame:
    """Per-day call/put quantities sold (0/1) for the rule."""
    if signal is None:
        prev_ret = (d.close / d.open - 1).shift(1).to_numpy()
        signal = np.sign(np.nan_to_num(prev_ret))
        active = np.abs(np.nan_to_num(prev_ret)) >= cfg.thr
        signal = np.where(active, signal, 0)
        signal[0] = 0
    up, dn = signal > 0, signal < 0
    if cfg.rule == "momentum":
        call, put = dn, up
    elif cfg.rule == "contrarian":
        call, put = up, dn
    elif cfg.rule == "always_call":
        call, put = np.ones(len(d), bool), np.zeros(len(d), bool)
    elif cfg.rule == "always_put":
        call, put = np.zeros(len(d), bool), np.ones(len(d), bool)
    elif cfg.rule == "straddle":
        call = put = np.ones(len(d), bool)
    elif cfg.rule == "half_straddle":
        call = put = np.full(len(d), 0.5)
    elif cfg.rule == "straddle_after_up":
        call = put = up
    else:
        raise ValueError(cfg.rule)
    return pd.DataFrame({"call": np.asarray(call, float), "put": np.asarray(put, float)}, index=d.index)


def simulate(d: pd.DataFrame, cfg: DCfg, signal: np.ndarray | None = None) -> pd.DataFrame:
    S, C = d.open.to_numpy(), d.close.to_numpy()
    prem = atm_premium(S, d.dvol.to_numpy() * cfg.iv_mult, cfg.T_days / 365.0)
    fee = np.minimum(cfg.opt_fee * S, cfg.opt_fee_cap * prem)
    q = legs(d, cfg, signal)
    call_pnl = q.call.to_numpy() * (prem - fee - np.maximum(C - S, 0.0))
    put_pnl = q.put.to_numpy() * (prem - fee - np.maximum(S - C, 0.0))
    out = pd.DataFrame({
        "S": S, "settle": C, "premium": prem, "n_opt": q.call + q.put,
        "call_pnl": call_pnl, "put_pnl": put_pnl, "pnl": call_pnl + put_pnl,
        "prem_in": (q.call + q.put).to_numpy() * prem, "fees": (q.call + q.put).to_numpy() * fee,
    }, index=d.index)
    out["equity"] = out.pnl.cumsum()
    return out


def summary(o: pd.DataFrame) -> dict:
    eq = o.equity
    dd = eq - eq.cummax()
    traded = o[o.n_opt > 0]
    years = (o.index[-1] - o.index[0]).days / 365.25
    # bps of spot per option sold (price-level neutral)
    bps = 1e4 * (traded.pnl / traded.S / traded.n_opt)
    return {
        "PnL $": eq.iloc[-1], "PnL $/yr": eq.iloc[-1] / years,
        "options sold": int(o.n_opt.sum()), "PnL $/option": eq.iloc[-1] / max(o.n_opt.sum(), 1),
        "edge bps/option": bps.mean(), "win %": 100 * (traded.pnl > 0).mean(),
        "max DD $": dd.min(), "worst day $": o.pnl.min(),
        "Sharpe": (o.pnl.mean() / o.pnl.std() * math.sqrt(365)) if o.pnl.std() > 0 else np.nan,
        "premium $": o.prem_in.sum(), "fees $": o.fees.sum(),
    }
