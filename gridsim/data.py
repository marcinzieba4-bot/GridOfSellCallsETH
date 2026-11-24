"""Download and cache market data.

Sources (all public, no key):
  * ETHUSDT 1h spot candles   - data.binance.vision (monthly + daily zips)
  * ETHUSDT perp funding rate  - data.binance.vision (futures/um monthly zips)
  * ETH DVOL (30d implied vol) - Deribit public API

Everything is aggregated into one *daily* table whose day runs from
08:00 UTC to 08:00 UTC next day - the Deribit daily-option expiry time.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DAILY_CSV = ROOT / "data" / "eth_daily_0800utc.csv"
BV = "https://data.binance.vision/data"


def _get(url: str, retries: int = 4) -> bytes | None:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** i)
        except Exception:
            time.sleep(2 ** i)
    return None


def _zip_csv(url: str, cache: Path) -> pd.DataFrame | None:
    if not cache.exists():
        blob = _get(url)
        if blob is None:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(blob)
    with zipfile.ZipFile(cache) as z:
        name = z.namelist()[0]
        raw = z.read(name).decode()
    first = raw.split("\n", 1)[0]
    header = 0 if any(c.isalpha() for c in first) else None
    return pd.read_csv(io.StringIO(raw), header=header)


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def _to_utc(ms_or_us: pd.Series) -> pd.Series:
    v = ms_or_us.astype("int64")
    v = np.where(v > 10**14, v // 1000, v)  # Binance switched to microseconds in 2025
    return pd.to_datetime(v, unit="ms", utc=True)


def fetch_hourly(start: date, end: date, symbol: str = "ETHUSDT") -> pd.DataFrame:
    frames = []
    today = datetime.now(timezone.utc).date()
    for y, m in _months(start, end):
        tag = f"{y}-{m:02d}"
        url = f"{BV}/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{tag}.zip"
        df = _zip_csv(url, RAW / "klines" / f"{symbol}-1h-{tag}.zip")
        if df is None:  # current month: fall back to daily files
            d = date(y, m, 1)
            while d.month == m and d < today:
                t = d.isoformat()
                url = f"{BV}/spot/daily/klines/{symbol}/1h/{symbol}-1h-{t}.zip"
                dd = _zip_csv(url, RAW / "klines" / f"{symbol}-1h-{t}.zip")
                if dd is not None:
                    frames.append(dd.iloc[:, :6])
                d += timedelta(days=1)
            continue
        frames.append(df.iloc[:, :6])
    h = pd.concat(frames, ignore_index=True)
    h.columns = ["t", "open", "high", "low", "close", "volume"]
    h["t"] = _to_utc(h["t"])
    return h.drop_duplicates("t").sort_values("t").set_index("t")


def fetch_funding(start: date, end: date, symbol: str = "ETHUSDT") -> pd.Series:
    frames = []
    for y, m in _months(start, end):
        tag = f"{y}-{m:02d}"
        url = f"{BV}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{tag}.zip"
        df = _zip_csv(url, RAW / "funding" / f"{symbol}-fundingRate-{tag}.zip")
        if df is not None:
            frames.append(df)
    f = pd.concat(frames, ignore_index=True)
    tcol = f.columns[0]
    rcol = [c for c in f.columns if "rate" in str(c).lower()][-1]
    s = pd.Series(f[rcol].astype(float).values, index=_to_utc(f[tcol]))
    return s[~s.index.duplicated()].sort_index()


def fetch_dvol(start: date, end: date, currency: str = "ETH") -> pd.Series:
    cache = RAW / f"dvol_{currency.lower()}_1d.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        rows = []
        t0 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp() * 1000)
        t1 = int(datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp() * 1000)
        cur_end = t1
        while True:
            url = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={currency}"
                   f"&start_timestamp={t0}&end_timestamp={cur_end}&resolution=1D")
            res = json.loads(_get(url))["result"]
            rows += res["data"]
            if not res.get("continuation") or not res["data"]:
                break
            cur_end = res["continuation"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).drop_duplicates("t")
    return pd.Series(d["c"].values / 100.0, index=pd.to_datetime(d["t"], unit="ms", utc=True)).sort_index()


def to_0800_days(h: pd.DataFrame) -> pd.DataFrame:
    """Hourly candles -> 08:00->08:00 UTC days (Deribit expiry), complete days only."""
    g = h.groupby((h.index - pd.Timedelta(hours=8)).floor("D"))
    daily = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "n": g["open"].size(),
    })
    daily = daily[daily["n"] >= 23].drop(columns="n")
    daily.index = daily.index + pd.Timedelta(hours=8)
    daily.index.name = "t"
    return daily


def spot_daily(asset: str = "ETH", start: date = date(2017, 9, 1), end: date | None = None) -> pd.DataFrame:
    """Price-only 08:00 days (no implied vol needed) - used for model-free tests incl. pre-2021."""
    f = ROOT / "data" / f"{asset.lower()}_spot_daily_0800utc.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["t"], index_col="t")
    end = end or datetime.now(timezone.utc).date()
    d = to_0800_days(fetch_hourly(start, end, f"{asset}USDT"))
    d.to_csv(f)
    return d


def build_daily(start: date = date(2021, 4, 1), end: date | None = None, refresh: bool = False,
                asset: str = "ETH") -> pd.DataFrame:
    """One row per option day: [open 08:00, high, low, close 08:00 next day], DVOL, funding."""
    out = DAILY_CSV if asset == "ETH" else ROOT / "data" / f"{asset.lower()}_daily_0800utc.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, parse_dates=["t"], index_col="t")
    end = end or datetime.now(timezone.utc).date()
    sym = f"{asset}USDT"
    daily = to_0800_days(fetch_hourly(start, end, sym))
    fr = fetch_funding(start, end, sym)
    fr_day = fr.groupby((fr.index - pd.Timedelta(hours=8, minutes=1)).floor("D")).sum()
    fr_day.index = fr_day.index + pd.Timedelta(hours=8)
    daily["funding"] = fr_day.reindex(daily.index).fillna(0.0)
    dv = fetch_dvol(start, end, asset)
    dv.index = dv.index.floor("D") + pd.Timedelta(hours=8)
    daily["dvol"] = dv.reindex(daily.index).ffill()
    daily = daily.dropna(subset=["dvol"])
    daily.index.name = "t"
    daily.to_csv(out)
    return daily


if __name__ == "__main__":
    d = build_daily(refresh=True)
    print(d.head(), d.tail(), len(d), sep="\n")
