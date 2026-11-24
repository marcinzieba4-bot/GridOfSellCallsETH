"""Simulation engine: sell 1 daily ATM ETH call every day; when a call loses,
open a short perp whose break-even (BE) is the call's strike + premium.
Later call profits (and funding) are spent on lifting the BE of open perps.
A perp is closed with a limit buy exactly at its BE (cycle PnL = 0).

Day loop (one day = 08:00 UTC -> 08:00 UTC, Deribit daily expiry):
  08:00  previous call expires at settle price S  (= today's open)
         - net loss  -> open short perp at S, deficit D = loss, BE = S - D
         - net gain  -> spend gain lifting BEs (allocation rule); a perp whose
                        BE reaches the market is closed at once; leftover = profit
         sell new ATM call, strike K = S, premium = Black-Scholes(IV)
  intraday  every perp whose BE >= day's low gets closed at BE (limit order)
  08:00+1  funding credited/debited on open perps (also lifts/lowers BE)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

SQRT2 = math.sqrt(2.0)


def bs_atm_call(S: float, iv: float, T: float) -> float:
    """ATM (K=S) call, r=0:  C = S * (2N(sigma*sqrt(T)/2) - 1)."""
    x = iv * math.sqrt(T) / 2.0
    return S * math.erf(x / SQRT2)  # 2N(x)-1 == erf(x/sqrt2)


@dataclass
class Config:
    mode: str = "grid"            # "grid" (user strategy) | "plain" (just sell calls, eat losses)
    alloc: str = "nearest"        # nearest | farthest | prorata | fifo
    iv_mult: float = 0.95         # sell price = BS(DVOL * iv_mult); <1 = selling below mid
    T_days: float = 1.0
    size: float = 1.0             # ETH per call and per perp
    opt_fee: float = 0.0003       # Deribit: 0.03% of underlying ...
    opt_fee_cap: float = 0.125    # ... capped at 12.5% of option price
    perp_taker: float = 0.0005    # open perp with market order
    perp_maker: float = 0.0000    # close at BE with resting limit order (Deribit maker 0%)
    use_funding: bool = True
    funding_to_be: bool = True    # funding changes the BE (vs. booked as profit)
    trigger: str = "net_loss"     # net_loss (S > K+prem) | itm (S > K)
    max_perps: int = 0            # 0 = unlimited; else when full, the loss is just realized
    stop_pct: float = 0.0         # 0 = none; close perp if price >= BE*(1+stop_pct)
    profit_share: float = 1.0     # share of call gains used for BE lifting (rest booked)


@dataclass
class Perp:
    pid: int
    t_open: pd.Timestamp
    entry: float
    size: float
    deficit: float               # loss this perp still has to win back
    origin_strike: float
    origin_be: float
    t_close: pd.Timestamp | None = None
    exit: float | None = None
    funding: float = 0.0
    lifted: float = 0.0          # call profits spent on this perp
    reason: str = ""
    path: list = field(default_factory=list)  # [(t, BE)] - how the break-even was lifted

    @property
    def be(self) -> float:
        return self.entry - self.deficit / self.size

    def upnl(self, px: float) -> float:
        return (self.entry - px) * self.size


@dataclass
class Result:
    cfg: Config
    daily: pd.DataFrame
    perps: list[Perp] = field(default_factory=list)

    def perps_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(p) | {"be": p.be} for p in self.perps])


def run(df: pd.DataFrame, cfg: Config | None = None, start_equity: float = 0.0) -> Result:
    cfg = cfg or Config()
    T = cfg.T_days / 365.0
    o, hi, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    iv = df["dvol"].to_numpy(float) * cfg.iv_mult
    fund = df["funding"].to_numpy(float) if "funding" in df else np.zeros(len(df))
    idx = df.index

    cash = start_equity
    open_p: list[Perp] = []
    closed: list[Perp] = []
    pid = 0
    call = None  # (K, premium, fee)
    rec = []
    # running tallies
    tot_prem = tot_opt_fee = tot_payout = tot_perp_fee = tot_fund = 0.0
    booked = 0.0  # PnL that is "free" (not used for BE lifting)

    def close(p: Perp, px: float, t, fee_rate: float, why: str):
        nonlocal cash, tot_perp_fee
        fee = fee_rate * px * p.size
        cash += p.upnl(px) - fee
        tot_perp_fee += fee
        p.deficit += fee
        p.exit, p.t_close, p.reason = px, t, why
        p.path.append((t, px))
        open_p.remove(p)
        closed.append(p)

    def order(ps: list[Perp]) -> list[Perp]:
        if cfg.alloc == "nearest":   # highest BE = closest below price = closes first
            return sorted(ps, key=lambda p: -p.be)
        if cfg.alloc == "farthest":
            return sorted(ps, key=lambda p: p.be)
        if cfg.alloc == "fifo":
            return sorted(ps, key=lambda p: p.pid)
        return list(ps)

    def spend(amount: float, S: float, t) -> float:
        """Lift BEs with `amount`; return what's left over (=free profit)."""
        if amount <= 0 or not open_p:
            return amount
        if cfg.alloc == "prorata":
            share = amount / len(open_p)
            left = 0.0
            for p in list(open_p):
                need = max(p.deficit - p.upnl(S), 0.0)  # amount that would put BE at S
                use = min(share, need)
                p.deficit -= use
                p.lifted += use
                left += share - use
                if p.be >= S - 1e-9:
                    close(p, S, t, cfg.perp_taker, "lifted")
            return spend(left, S, t) if left > 1e-9 and open_p else left
        for p in order(open_p):
            need = max(p.deficit - p.upnl(S), 0.0)
            use = min(amount, need)
            p.deficit -= use
            p.lifted += use
            amount -= use
            if p.be >= S - 1e-9:
                close(p, S, t, cfg.perp_taker, "lifted")
            if amount <= 1e-12:
                break
        return amount

    for i in range(len(df)):
        t, S = idx[i], o[i]
        # ---- 1. expiry of yesterday's call --------------------------------
        if call is not None:
            K, prem, ofee = call
            payout = max(S - K, 0.0) * cfg.size
            pnl = prem - ofee - payout
            cash -= payout
            tot_payout += payout
            lose = pnl < 0 if cfg.trigger == "net_loss" else S > K
            if cfg.mode == "grid" and lose and (cfg.max_perps == 0 or len(open_p) < cfg.max_perps):
                fee = cfg.perp_taker * S * cfg.size
                cash -= fee
                tot_perp_fee += fee
                p = Perp(pid, t, S, cfg.size, deficit=-pnl + fee, origin_strike=K, origin_be=0.0)
                p.origin_be = p.be
                p.path.append((t, p.be))
                pid += 1
                open_p.append(p)
            elif pnl > 0 and cfg.mode == "grid":
                use = pnl * cfg.profit_share
                booked += pnl - use + spend(use, S, t)
            else:
                booked += pnl
        # ---- 2. sell today's ATM call --------------------------------------
        prem = bs_atm_call(S, iv[i], T) * cfg.size
        ofee = min(cfg.opt_fee * S * cfg.size, cfg.opt_fee_cap * prem)
        cash += prem - ofee
        tot_prem += prem
        tot_opt_fee += ofee
        call = (S, prem, ofee)
        # ---- 3. intraday: limit buys at BE, optional stop -------------------
        for p in list(open_p):
            stop = p.be * (1 + cfg.stop_pct) if cfg.stop_pct > 0 else math.inf
            if hi[i] >= stop:  # if both stop and BE are hit the same day, assume the worse (stop)
                close(p, max(stop, o[i]), t, cfg.perp_taker, "stop")
            elif lo[i] <= p.be:
                close(p, min(p.be, o[i]), t, cfg.perp_maker, "be")
        # ---- 4. funding (shorts receive positive funding) -------------------
        if cfg.use_funding and open_p and fund[i] != 0.0:
            for p in open_p:
                f = fund[i] * c[i] * p.size
                cash += f
                tot_fund += f
                p.funding += f
                if cfg.funding_to_be:
                    p.deficit -= f
            # funding may push BE above market -> close at market
            for p in list(open_p):
                if p.be >= c[i]:
                    close(p, c[i], t, cfg.perp_taker, "funding")
        # ---- record (end of day; open call marked at intrinsic+premium decay ignored)
        # equity at next 08:00 = cash + perp uPnL - payoff of today's call
        for p in open_p:
            if abs(p.path[-1][1] - p.be) > 0.01:
                p.path.append((t, p.be))
        payoff = max(c[i] - S, 0.0) * cfg.size
        upnl = sum(p.upnl(c[i]) for p in open_p)
        rec.append(dict(
            t=t, open=S, close=c[i], strike=S, premium=prem, iv=iv[i],
            n_perps=len(open_p), short_eth=len(open_p) * cfg.size,
            upnl=upnl, cash=cash, equity=cash + upnl - payoff,
            grid_deficit=sum(p.deficit for p in open_p),
            avg_be=(np.mean([p.be for p in open_p]) if open_p else np.nan),
            min_be=(min(p.be for p in open_p) if open_p else np.nan),
            max_be=(max(p.be for p in open_p) if open_p else np.nan),
            tot_prem=tot_prem, tot_payout=tot_payout, tot_opt_fee=tot_opt_fee,
            tot_perp_fee=tot_perp_fee, tot_fund=tot_fund, booked=booked,
        ))
    daily = pd.DataFrame(rec).set_index("t")
    return Result(cfg, daily, closed + open_p)


def stats(res: Result, capital: float | None = None) -> dict:
    d = res.daily
    eq = d["equity"]
    dd = eq - eq.cummax()
    years = (d.index[-1] - d.index[0]).days / 365.25
    pdf = res.perps_df()
    closed = pdf[pdf["t_close"].notna()] if len(pdf) else pdf
    still = pdf[pdf["t_close"].isna()] if len(pdf) else pdf
    days_open = ((closed["t_close"] - closed["t_open"]).dt.days if len(closed) else pd.Series(dtype=float))
    out = {
        "final_pnl": eq.iloc[-1],
        "pnl_per_year": eq.iloc[-1] / years,
        "max_drawdown": dd.min(),
        "max_dd_date": dd.idxmin(),
        "worst_equity": eq.min(),
        "max_perps": int(d["n_perps"].max()),
        "perps_open_end": int(d["n_perps"].iloc[-1]),
        "perps_opened": len(pdf),
        "perps_closed": len(closed),
        "median_days_to_close": float(days_open.median()) if len(days_open) else np.nan,
        "max_days_to_close": float(days_open.max()) if len(days_open) else np.nan,
        "days_with_grid_%": 100 * (d["n_perps"] > 0).mean(),
        "premium_total": d["tot_prem"].iloc[-1],
        "payout_total": d["tot_payout"].iloc[-1],
        "funding_total": d["tot_fund"].iloc[-1],
        "fees_total": d["tot_opt_fee"].iloc[-1] + d["tot_perp_fee"].iloc[-1],
        "end_upnl": d["upnl"].iloc[-1],
        "end_price": d["close"].iloc[-1],
        "end_grid_avg_be": d["avg_be"].iloc[-1],
    }
    if capital:
        out["return_on_capital_%"] = 100 * out["final_pnl"] / capital
        out["max_dd_%cap"] = 100 * out["max_drawdown"] / capital
    daily_ret = eq.diff().dropna()
    out["sharpe_like"] = (daily_ret.mean() / daily_ret.std() * math.sqrt(365)) if daily_ret.std() > 0 else np.nan
    return out
