"""Synthetic price paths for Monte Carlo runs of the engine."""
from __future__ import annotations

import numpy as np
import pandas as pd


def gbm_days(n_days: int, sigma: float = 0.70, mu: float = 0.0, s0: float = 2000.0,
             steps: int = 24, rng: np.random.Generator | None = None,
             iv: float | None = None, funding_day: float = 0.0) -> pd.DataFrame:
    """Daily OHLC from an hourly GBM. mu/sigma are annual (log) drift and vol.
    iv defaults to sigma -> options are fairly priced (no volatility premium)."""
    rng = rng or np.random.default_rng()
    dt = 1.0 / (365 * steps)
    r = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=(n_days, steps))
    lp = np.log(s0) + np.concatenate([[0.0], np.cumsum(r.ravel())])
    px = np.exp(lp)
    o = px[:-1:steps]
    path = px[1:].reshape(n_days, steps)
    hi = np.maximum(path.max(1), o)
    lo = np.minimum(path.min(1), o)
    c = path[:, -1]
    idx = pd.date_range("2030-01-01 08:00", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "dvol": iv if iv is not None else sigma, "funding": funding_day}, index=idx)


def block_bootstrap(hist: pd.DataFrame, n_days: int, block: int = 20, s0: float = 2000.0,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Re-shuffle blocks of real ETH days (keeps vol clustering, short trends,
    the real DVOL and funding that went with each day)."""
    rng = rng or np.random.default_rng()
    h = hist
    rel = pd.DataFrame({"h": h.high / h.open, "l": h.low / h.open, "c": h.close / h.open,
                        "dvol": h.dvol, "funding": h.funding}).to_numpy()
    rows = []
    while len(rows) < n_days:
        st = rng.integers(0, len(rel) - block)
        rows.extend(rel[st:st + block])
    rel = np.array(rows[:n_days])
    o = np.empty(n_days)
    o[0] = s0
    o[1:] = s0 * np.cumprod(rel[:-1, 2])
    idx = pd.date_range("2030-01-01 08:00", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame({"open": o, "high": o * rel[:, 0], "low": o * rel[:, 1], "close": o * rel[:, 2],
                         "dvol": rel[:, 3], "funding": rel[:, 4]}, index=idx)
