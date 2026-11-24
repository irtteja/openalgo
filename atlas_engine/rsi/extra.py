"""Non-price NSE data via nselib, cached under data/extra/.

  delivery  - daily % of traded shares taken into delivery (per stock)
  futures   - daily stock-futures open interest summed across expiries, and the
              front-month basis (future / spot - 1)   (F&O stocks only)
  vix       - India VIX close (market-wide)

    python -m rsi.extra            # download everything for the Nifty 100
"""
from __future__ import annotations
import os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from .data import DATA_DIR, nifty100_symbols

EXTRA = os.path.join(DATA_DIR, "extra")
YEARS = range(2021, pd.Timestamp.today().year + 1)


def _retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def fetch_delivery(symbol):
    path = os.path.join(EXTRA, f"dlv_{symbol}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    from nselib import capital_market as cm
    end = pd.Timestamp.today().strftime("%d-%m-%Y")
    df = _retry(lambda: cm.price_volume_and_deliverable_position_data(symbol=symbol, from_date="01-01-2021", to_date=end))
    df = df.rename(columns=lambda c: c.replace("ï»¿", "").strip('"'))
    df = df[df["Series"].isin(["EQ", "BE"])]
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"], format="%d-%b-%Y"),
        "dlv_qty": pd.to_numeric(df["DeliverableQty"].astype(str).str.replace(",", ""), errors="coerce"),
        "qty": pd.to_numeric(df["TotalTradedQuantity"].astype(str).str.replace(",", ""), errors="coerce"),
    }).drop_duplicates("Date").sort_values("Date")
    out.to_csv(path, index=False)
    return out


def fetch_futures(symbol):
    path = os.path.join(EXTRA, f"fut_{symbol}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    from nselib import derivatives as d
    parts = []
    for y in YEARS:
        end = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp.today()).strftime("%d-%m-%Y")
        try:
            parts.append(_retry(lambda: d.future_price_volume_data(
                symbol=symbol, instrument="FUTSTK", from_date=f"01-01-{y}", to_date=end)))
        except Exception:
            pass  # not in F&O that year
    if not parts:
        out = pd.DataFrame(columns=["Date", "oi", "basis"])
    else:
        f = pd.concat(parts)
        f["Date"] = pd.to_datetime(f["TIMESTAMP"], format="%d-%b-%Y")
        f["exp"] = pd.to_datetime(f["EXPIRY_DT"], format="%d-%b-%Y")
        for c in ["OPEN_INT", "CLOSING_PRICE", "UNDERLYING_VALUE"]:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        f = f.drop_duplicates(["Date", "exp"])
        oi = f.groupby("Date")["OPEN_INT"].sum()
        front = f.sort_values("exp").groupby("Date").first()
        out = pd.DataFrame({"oi": oi, "basis": front["CLOSING_PRICE"] / front["UNDERLYING_VALUE"] - 1}).reset_index()
    out.to_csv(path, index=False)
    return out


def fetch_vix(update=False):
    path = os.path.join(EXTRA, "vix.csv")
    old = pd.read_csv(path, parse_dates=["Date"]) if os.path.exists(path) else None
    if old is not None and not update:
        return old
    from nselib import capital_market as cm
    parts = []
    if old is not None and len(old):   # only the days after the last one we have
        a = old["Date"].max() + pd.Timedelta(days=1)
        if a > pd.Timestamp.today():
            return old
        a = min(a.normalize(), pd.Timestamp.today().normalize() - pd.Timedelta(days=1))   # the source needs end > start
        v = _retry(lambda: cm.india_vix_data(from_date=a.strftime("%d-%m-%Y"), to_date=pd.Timestamp.today().strftime("%d-%m-%Y")))
        if v is None or len(v) == 0:
            return old
        new = pd.DataFrame({"Date": pd.to_datetime(v["TIMESTAMP"], format="%d-%b-%Y"), "vix": pd.to_numeric(v["CLOSE_INDEX_VAL"], errors="coerce")})
        out = pd.concat([old, new]).drop_duplicates("Date", keep="last").sort_values("Date")
        out.to_csv(path, index=False)
        return out
    for y in YEARS:
        end = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp.today()).strftime("%d-%m-%Y")
        parts.append(_retry(lambda: cm.india_vix_data(from_date=f"01-01-{y}", to_date=end)))
    v = pd.concat(parts)
    out = pd.DataFrame({"Date": pd.to_datetime(v["TIMESTAMP"], format="%d-%b-%Y"),
                        "vix": pd.to_numeric(v["CLOSE_INDEX_VAL"], errors="coerce")})
    out = out.drop_duplicates("Date").sort_values("Date")
    out.to_csv(path, index=False)
    return out


def add_extra_features(panel: dict) -> dict:
    """Add wide (date x symbol) frames: dlv (delivery fraction), oi, basis, vix.
    OI is converted to split-consistent units like volume. Missing data stays NaN."""
    idx, cols = panel["close"].index, panel["close"].columns
    dlv, oi, basis = {}, {}, {}
    for s in cols:
        p = os.path.join(EXTRA, f"dlv_{s}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
            dlv[s] = d["dlv_qty"] / d["qty"]
        p = os.path.join(EXTRA, f"fut_{s}.csv")
        if os.path.exists(p):
            f = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
            if len(f):
                oi[s], basis[s] = f["oi"], f["basis"]
    W = lambda d: pd.DataFrame(d).reindex(index=idx, columns=cols)
    if "dlv" not in panel:  # the whole-market panel already carries delivery for every stock
        panel["dlv"] = W(dlv).clip(0, 1)
    raw_oi = W(oi)
    panel["oi"] = raw_oi * panel["volume"] / panel["volume_raw"] if "volume_raw" in panel else raw_oi
    panel["basis"] = W(basis).clip(-0.2, 0.2)
    vp = os.path.join(EXTRA, "vix.csv")
    if os.path.exists(vp):
        v = pd.read_csv(vp, parse_dates=["Date"]).set_index("Date")["vix"].reindex(idx).ffill()
        panel["vix"] = pd.DataFrame({s: v for s in cols})
    return panel


if __name__ == "__main__":
    os.makedirs(EXTRA, exist_ok=True)
    fetch_vix(); print("vix ok", flush=True)
    syms = nifty100_symbols()

    def job(s):
        msg = []
        for name, fn in [("delivery", fetch_delivery), ("futures", fetch_futures)]:
            try:
                n = len(fn(s)); msg.append(f"{name}={n}")
            except Exception as e:
                msg.append(f"{name}=FAILED({e!r:.60})")
        return f"{s}: {' '.join(msg)}"

    with ThreadPoolExecutor(2) as ex:
        for i, line in enumerate(ex.map(job, syms)):
            print(f"[{i+1}/{len(syms)}] {line}", flush=True)
