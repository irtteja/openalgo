"""Turn the daily NSE files from rsi/market_data.py into model inputs.

  var      - NSE's VaR margin per stock (its own risk estimate), log
  fii_fut  - FII net position in index futures: (long - short) / (long + short), same for every stock
  cli_fut  - same for retail clients (often the opposite side of FIIs)
  fii_stk  - FII net position in stock futures
  bulk     - bulk-deal net buying that day, as a fraction of the stock's volume (0 when none)

Each value is known at the close of its own day (NSE publishes these files that evening),
so a model value at close t uses only data up to day t.
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

from .market_data import DAILY


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _read_days(sub):
    out = {}
    for p in sorted(glob.glob(os.path.join(DAILY, sub, "*.csv"))):
        if os.path.getsize(p) < 10:
            continue
        try:
            out[pd.Timestamp(os.path.basename(p)[:-4])] = pd.read_csv(p)
        except Exception:
            pass
    return out


def add_market_features(panel: dict) -> dict:
    idx, cols = panel["close"].index, panel["close"].columns
    bcast = lambda s: pd.DataFrame({c: s.reindex(idx) for c in cols})

    var = {}
    for d, df in _read_days("var").items():
        var[d] = pd.Series(_num(df["SecurityVaR"]).values, index=df["Symbol"].astype(str)).groupby(level=0).last()
    if var:
        v = pd.DataFrame(var).T.reindex(index=idx, columns=cols)
        panel["var"] = np.log(v.clip(lower=1))

    rows = {}
    for d, df in _read_days("part_oi").items():
        df.columns = [c.strip() for c in df.columns]
        df = df.set_index(df.columns[0])
        df.index = df.index.astype(str).str.strip()
        g = lambda who, a, b: (_num(df.loc[who, [a]]).iloc[0], _num(df.loc[who, [b]]).iloc[0])
        r = {}
        for who, key in [("FII", "fii_fut"), ("Client", "cli_fut")]:
            if who in df.index:
                L, S = g(who, "Future Index Long", "Future Index Short")
                r[key] = (L - S) / (L + S) if L + S > 0 else np.nan
        if "FII" in df.index:
            L, S = g("FII", "Future Stock Long", "Future Stock Short")
            r["fii_stk"] = (L - S) / (L + S) if L + S > 0 else np.nan
        rows[d] = r
    if rows:
        po = pd.DataFrame(rows).T.sort_index().reindex(idx).ffill(limit=3)
        for k in po.columns:
            panel[k] = bcast(po[k])

    parts = [pd.read_csv(p) for p in glob.glob(os.path.join(DAILY, "bulk", "*.csv")) if os.path.getsize(p) > 10]
    if parts:
        b = pd.concat(parts)
        b["Date"] = pd.to_datetime(b["Date"], format="%d-%b-%Y", errors="coerce")
        q = _num(b["QuantityTraded"])
        b["net"] = np.where(b["Buy/Sell"].astype(str).str.upper().str.startswith("B"), q, -q)
        net = b.pivot_table(index="Date", columns="Symbol", values="net", aggfunc="sum")
        net = net.reindex(index=idx, columns=cols)
        volraw = panel.get("volume_raw", panel["volume"])
        panel["bulk"] = (net / volraw).clip(-1, 1).fillna(0.0)
    return panel
