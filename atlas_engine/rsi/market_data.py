"""Daily NSE datasets found by the nselib audit (audit/probe.py), cached in data/extra/daily/.

Market-wide (one row per day):
  cash_flows   - FII/FPI and DII net buying in the cash market (category_turnover_cash)
  part_oi      - futures/options positions by participant: Client, DII, FII, Pro
                 (participant_wise_open_interest)
Per stock (one file per day covering all stocks):
  var          - NSE's own daily risk estimate per stock: VaR margin (var_end_of_day)
Per stock, date ranges (events):
  short        - disclosed short sales (short_selling_data)
  bulk         - bulk deals: big trades with client names (bulk_deal_data)

    python -m rsi.market_data
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd

from .data import DATA_DIR

DAILY = os.path.join(DATA_DIR, "extra", "daily")


def _days(start="2021-01-01"):
    """NSE trading days we have whole-market price files for (scripts/fetch_bhav.py)."""
    import glob
    fs = [f for f in glob.glob(os.path.join(DATA_DIR, "extra", "bhav", "*.csv")) if os.path.getsize(f) > 50]
    d = pd.to_datetime(sorted(os.path.basename(f)[:-4] for f in fs))
    return d[d >= start]


def _get(fn, *a, tries=3, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except Exception as e:
            if "not found" in str(e).lower() or "no data" in str(e).lower():
                return None
            if i == tries - 1:
                return None
            time.sleep(3 * (i + 1))


def download():
    from nselib import capital_market as cm, derivatives as dv
    for sub in ["cash", "part_oi", "var"]:
        os.makedirs(os.path.join(DAILY, sub), exist_ok=True)
    days = _days()
    for i, d in enumerate(days):
        ds = d.strftime("%d-%m-%Y")
        for sub, fn in [("cash", cm.category_turnover_cash), ("part_oi", dv.participant_wise_open_interest),
                        ("var", cm.var_end_of_day)]:
            path = os.path.join(DAILY, sub, f"{d.date()}.csv")
            if os.path.exists(path):
                continue
            df = _get(fn, ds)
            if df is None:
                pd.DataFrame().to_csv(path, index=False)
                continue
            if sub == "var":  # keep equities only, key columns
                df = df[df["Series"].astype(str).isin(["EQ", "BE"])][["Symbol", "SecurityVaR", "ApplicableMarginRate"]]
            df.to_csv(path, index=False)
            time.sleep(0.3)
        if i % 25 == 0:
            print(f"[{i+1}/{len(days)}] {d.date()}", flush=True)
    # Event-style data by month.
    for name, fn in [("short", cm.short_selling_data), ("bulk", cm.bulk_deal_data)]:
        os.makedirs(os.path.join(DAILY, name), exist_ok=True)
        for a in pd.date_range("2021-01-01", pd.Timestamp.today(), freq="MS"):
            path = os.path.join(DAILY, name, f"{a.strftime('%Y-%m')}.csv")
            recent = a >= pd.Timestamp.today().normalize() - pd.DateOffset(months=1) - pd.offsets.MonthBegin(1)
            if os.path.exists(path) and not recent:   # this month and last month are fetched again
                continue
            b = min(a + pd.offsets.MonthEnd(0), pd.Timestamp.today())
            df = _get(fn, from_date=a.strftime("%d-%m-%Y"), to_date=b.strftime("%d-%m-%Y"))
            (df if df is not None else pd.DataFrame()).to_csv(path, index=False)
            time.sleep(0.5)
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    download()
