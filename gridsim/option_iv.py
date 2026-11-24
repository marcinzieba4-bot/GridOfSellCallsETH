"""Real Deribit 1-day option implied vols, from historical trades.

For every option day D (08:00 UTC) we download the first <=1000 option trades in
[D 08:00, D 11:00) from history.deribit.com and keep the ones in the option that
expires at D+1 08:00 (the one our strategy sells) with strike within +-3% of the
index. Per day we store the median trade IV (all trades ~ mid), the median IV of
trades where the taker SOLD (filled at the bid = what a seller actually gets),
and trade counts.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "deribit_trades"
URL = ("https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
       "?currency={cur}&kind=option&start_timestamp={st}&end_timestamp={en}&count=1000"
       "&include_old=true&sorting=asc")
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _exp_code(d: datetime) -> str:
    return f"{d.day}{MON[d.month - 1]}{d.strftime('%y')}"


def _fetch_day(cur: str, day: datetime, tenor: int = 1) -> dict:
    f = (CACHE / cur if tenor == 1 else CACHE / f"{cur}_{tenor}d") / f"{day:%Y-%m-%d}.json"
    if f.exists():
        return json.loads(f.read_text())
    st = int(day.timestamp() * 1000)
    en = st + 3 * 3600 * 1000
    for i in range(6):
        try:
            with urllib.request.urlopen(URL.format(cur=cur, st=st, en=en), timeout=60) as r:
                trades = json.loads(r.read())["result"]["trades"]
            break
        except Exception:
            time.sleep(2 ** i)
    else:
        return {}
    code = _exp_code(day + timedelta(days=tenor))
    keep = []
    for t in trades:
        parts = t["instrument_name"].split("-")
        if len(parts) != 4 or parts[1] != code:
            continue
        K, S = float(parts[2]), t["index_price"]
        if abs(K / S - 1) <= 0.03 and t.get("iv"):
            keep.append([t["timestamp"], K, S, parts[3], t["direction"], t["iv"], t["amount"]])
    out = {"n_all": len(trades), "trades": keep}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def daily_iv(cur: str, days: pd.DatetimeIndex, workers: int = 3, tenor: int = 1) -> pd.DataFrame:
    dts = [d.to_pydatetime() for d in days]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda d: _fetch_day(cur, d, tenor), dts))
    rows = []
    for d, r in zip(days, res):
        tr = pd.DataFrame(r.get("trades", []), columns=["ts", "K", "S", "cp", "dir", "iv", "amt"])
        if tr.empty:
            rows.append({"t": d, "n": 0})
            continue
        tr["m"] = np.abs(np.log(tr.K / tr.S))
        near = tr[tr.m <= 0.015]
        use = near if len(near) >= 3 else tr
        sells = use[use.dir == "sell"]
        buys = use[use.dir == "buy"]
        rows.append({"t": d, "n": len(use), "iv_mid": use.iv.median() / 100,
                     "iv_sell": (sells.iv.median() / 100) if len(sells) >= 2 else np.nan,
                     "iv_buy": (buys.iv.median() / 100) if len(buys) >= 2 else np.nan,
                     "iv_wavg": np.average(use.iv, weights=use.amt) / 100})
    return pd.DataFrame(rows).set_index("t")


if __name__ == "__main__":
    import sys
    cur = sys.argv[1] if len(sys.argv) > 1 else "ETH"
    tenor = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t0 = time.time()
    if tenor == 7:   # weeklies: every Friday 08:00 -> next Friday 08:00
        days = pd.date_range("2019-04-05 08:00", "2026-11-20 08:00", freq="W-FRI", tz="UTC")
        df = daily_iv(cur, days, tenor=7)
        df.to_csv(ROOT / "data" / f"{cur.lower()}_7d_option_iv.csv")
    else:
        days = pd.date_range("2021-04-01 08:00", "2026-11-22 08:00", freq="D", tz="UTC")
        df = daily_iv(cur, days)
        df.to_csv(ROOT / "data" / f"{cur.lower()}_1d_option_iv.csv")
    print(cur, "days", len(df), "with data", int((df.n > 0).sum()), f"{time.time() - t0:.0f}s")
    print(df.describe())
