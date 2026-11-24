"""Market data loading with on-disk cache.

Source: nselib (NSE). The price feed is NOT adjusted for corporate actions, so daily
returns are rebuilt as TOTAL returns using NSE's official corporate-actions list
(data/_corporate_actions.csv, fetched with cm.corporate_actions_for_equity):

  ex-date return = (Close + dividend) / (PrevClose * f) - 1
  f = product of price factors for that day:
        Bonus a:b   (a new per b held)   -> b / (a + b)
        Split FV X -> Y                  -> Y / X
        Rights a:b @ premium P            -> theoretical ex-rights price / PrevClose
Actions the list gives no ratio for (demergers, bonus debentures/preference shares) fall
back to a gap rule: if the open is still >20% from the adjusted previous close, the day's
return is taken as open->close only. The first trading day (listing) is dropped.
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NUM_COLS = ["PrevClose", "OpenPrice", "HighPrice", "LowPrice", "ClosePrice",
            "AveragePrice", "TotalTradedQuantity", "No.ofTrades"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.replace("ï»¿", "").strip('"'))
    df = df[df["Series"].isin(["EQ", "BE"])].copy()
    turnover = [c for c in df.columns if c.startswith("Turnover")]
    for c in NUM_COLS + turnover:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    if turnover:
        df = df.rename(columns={turnover[0]: "Turnover"})
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
    return df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def fetch_symbol(symbol: str, start="01-01-2021", end=None, retries=3) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    from nselib import capital_market as cm
    end = end or pd.Timestamp.today().strftime("%d-%m-%Y")
    for i in range(retries):
        try:
            df = _clean(cm.price_volume_and_deliverable_position_data(symbol=symbol, from_date=start, to_date=end))
            _save_delivery(symbol, df)
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(path, index=False)
            return df
        except Exception as e:  # network / throttling
            print(f"  {symbol}: attempt {i+1} failed: {e!r}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"could not fetch {symbol}")


def _save_delivery(symbol, df):
    if "DeliverableQty" not in df.columns:
        return
    d = os.path.join(DATA_DIR, "extra"); os.makedirs(d, exist_ok=True)
    pd.DataFrame({"Date": df["Date"],
                  "dlv_qty": pd.to_numeric(df["DeliverableQty"].astype(str).str.replace(",", ""), errors="coerce"),
                  "qty": df["TotalTradedQuantity"]}).to_csv(os.path.join(d, f"dlv_{symbol}.csv"), index=False)


_RS = r"(?:Rs|Re)\.?\s*(\d+(?:\.\d+)?)"


def parse_action(subject: str, prev_close: float):
    """Return (price_factor, dividend_per_share) implied by one corporate-action subject."""
    import re
    f, div = 1.0, 0.0
    subject = str(subject)
    m = re.search(r"from\s*" + _RS + r".*?to\s*" + _RS, subject, re.I)
    if m and re.search(r"split|sub-division", subject, re.I) and float(m.group(2)) > 0:
        f *= float(m.group(2)) / float(m.group(1))
        subject = subject[:m.start()] + subject[m.end():]
    for part in re.split(r"/(?!-)|\band\b", subject):
        low = part.lower()
        if "debenture" in low or "ncrps" in low or "preference" in low:
            continue  # bonus debentures / preference shares: no share-count change
        m = re.search(r"bonus\s*(\d+)\s*:\s*(\d+)", low)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            f *= b / (a + b)
            continue
        if "split" in low or "sub-division" in low:
            continue
        m = re.search(r"rights\s*(\d+)\s*:\s*(\d+)\s*@\s*premium\s*" + _RS, part, re.I)
        if m and prev_close > 0:
            a, b, price = int(m.group(1)), int(m.group(2)), float(m.group(3))
            terp = (b * prev_close + a * price) / (a + b)  # face value ignored (small)
            f *= min(1.0, terp / prev_close)
            continue
        if "dividend" in low:
            div += sum(float(x) for x in re.findall(_RS, part))
    return f, div


def update_corporate_actions(days=120, start_year=2015):
    """Fetch the last `days` of corporate actions (or everything since start_year if none yet) and merge."""
    from nselib import capital_market as cm
    path = os.path.join(DATA_DIR, "_corporate_actions.csv")
    parts = [pd.read_csv(path)] if os.path.exists(path) else []
    today = pd.Timestamp.today()
    starts = [today - pd.Timedelta(days=days)] if parts else list(pd.date_range(f"{start_year}-01-01", today, freq="6MS"))
    for a in starts:
        b = min(a + pd.DateOffset(months=6) - pd.Timedelta(days=1), today)
        for i in range(3):
            try:
                parts.append(cm.corporate_actions_for_equity(from_date=a.strftime("%d-%m-%Y"), to_date=b.strftime("%d-%m-%Y")))
                break
            except Exception:
                time.sleep(5 * (i + 1))
    if parts:
        pd.concat(parts).drop_duplicates().to_csv(path, index=False)


def corporate_actions() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "_corporate_actions.csv")
    if not os.path.exists(path):
        from nselib import capital_market as cm
        parts = []
        for y in range(2021, pd.Timestamp.today().year + 1):
            for a, b in [("01-01", "30-06"), ("01-07", "31-12")]:
                parts.append(cm.corporate_actions_for_equity(from_date=f"{a}-{y}", to_date=f"{b}-{y}"))
        pd.concat(parts).drop_duplicates().to_csv(path, index=False)
    ca = pd.read_csv(path)
    if "symbol" in ca:
        ca["symbol"] = ca["symbol"].replace(symbol_renames())
    ca["exDate"] = pd.to_datetime(ca["exDate"], format="%d-%b-%Y", errors="coerce")
    return ca.dropna(subset=["exDate"])


def _adjusted_returns(d: pd.DataFrame, actions: pd.DataFrame | None = None, log=None) -> pd.Series:
    d = d.set_index("Date")
    factor = pd.Series(1.0, index=d.index)
    div = pd.Series(0.0, index=d.index)
    if actions is not None:
        for _, a in actions.iterrows():
            if a["exDate"] in d.index:
                f, dv = parse_action(a["subject"], d.at[a["exDate"], "PrevClose"])
                factor[a["exDate"]] *= f
                div[a["exDate"]] += dv
    base = d["PrevClose"] * factor
    r = (d["ClosePrice"] + div) / base - 1
    # Fallback for actions without a usable ratio (e.g. demergers).
    gap = d["OpenPrice"] / base - 1
    fb = gap.abs() > 0.20
    r[fb] = (d["ClosePrice"] / d["OpenPrice"] - 1)[fb]
    if log is not None:
        for dt in d.index[(factor != 1) | fb]:
            log.append((dt.date(), round(float(factor[dt]), 4), bool(fb[dt]), round(float(r[dt]), 4)))
    r.iloc[0] = np.nan  # listing day / first row: PrevClose may be an IPO price
    return r


def nifty50_symbols() -> list[str]:
    path = os.path.join(DATA_DIR, "_nifty50.csv")
    if not os.path.exists(path):
        from nselib import capital_market as cm
        os.makedirs(DATA_DIR, exist_ok=True)
        cm.nifty50_equity_list().to_csv(path, index=False)
    return pd.read_csv(path)["Symbol"].tolist()


def nifty100_symbols() -> list[str]:
    path = os.path.join(DATA_DIR, "_niftynext50.csv")
    if not os.path.exists(path):
        from nselib import capital_market as cm
        cm.niftynext50_equity_list().to_csv(path, index=False)
    nxt = pd.read_csv(path)["Symbol"].tolist()
    return nifty50_symbols() + [s for s in nxt if s not in nifty50_symbols()]


def nifty500_symbols() -> list[str]:
    """Nifty 100 + Midcap 150 + Smallcap 250 (current members)."""
    path = os.path.join(DATA_DIR, "_nifty500.csv")
    if not os.path.exists(path):
        from nselib import capital_market as cm
        extra = list(cm.niftymidcap150_equity_list()["Symbol"]) + list(cm.niftysmallcap250_equity_list()["Symbol"])
        pd.DataFrame({"Symbol": extra}).to_csv(path, index=False)
    out = nifty100_symbols()
    return out + [s for s in pd.read_csv(path)["Symbol"] if s not in out]


def load_panel(symbols=None, min_coverage=0.9) -> dict[str, pd.DataFrame]:
    """Return wide (date x symbol) frames: close (adjusted), open, high, low, volume, vwap, ret."""
    symbols = symbols or nifty50_symbols()
    frames = {}
    for s in symbols:
        try:
            d = fetch_symbol(s)
            if len(d) >= 30:  # skip empty files (REITs / placeholder index entries)
                frames[s] = d
        except RuntimeError as e:
            print(e)
    ca = corporate_actions()
    logs = {s: [] for s in frames}
    ret = pd.DataFrame({s: _adjusted_returns(d, ca[ca["symbol"] == s], logs[s]) for s, d in frames.items()}).sort_index()
    keep = ret.columns[ret.notna().mean() >= min_coverage]
    ret = ret[keep]
    adj_close = (1 + ret.fillna(0)).cumprod()
    traded = pd.DataFrame({s: frames[s].set_index("Date")["ClosePrice"] for s in keep}).reindex(ret.index).notna()
    adj_close = adj_close.where(traded)  # blank before listing / on untraded days, not a fake flat price
    out = {"ret": ret, "close": adj_close}
    # Scale open/high/low/vwap by the same adjustment factor as close.
    for name, col in [("open", "OpenPrice"), ("high", "HighPrice"), ("low", "LowPrice"), ("vwap", "AveragePrice")]:
        raw = pd.DataFrame({s: frames[s].set_index("Date")[col] for s in keep}).reindex(ret.index)
        raw_close = pd.DataFrame({s: frames[s].set_index("Date")["ClosePrice"] for s in keep}).reindex(ret.index)
        out[name] = raw / raw_close * adj_close
    vol = pd.DataFrame({s: frames[s].set_index("Date")["TotalTradedQuantity"] for s in keep}).reindex(ret.index)
    raw_close = pd.DataFrame({s: frames[s].set_index("Date")["ClosePrice"] for s in keep}).reindex(ret.index)
    # Volume in split-consistent units: shares * (raw price / adjusted price).
    out["volume"] = vol * raw_close / adj_close
    out["volume_raw"] = vol
    out["close_raw"] = raw_close
    out["events"] = {s: logs[s] for s in keep if logs[s]}
    return out


_RENAMES = None


def symbol_renames() -> dict:
    """{old symbol: current symbol} for companies that changed their NSE symbol (e.g. ICICIPRULI -> ICICILIFE,
    Oct 2026). Found from the daily whole-market files: same ISIN, different symbol over time; the symbol used
    on the most recent day wins. Only main-board series are used, and an old symbol later reused by another
    company is not remapped."""
    global _RENAMES
    if _RENAMES is not None:
        return _RENAMES
    import glob
    path = os.path.join(DATA_DIR, "extra", "symbol_renames.json")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "extra", "bhav", "*.csv")))
    if os.path.exists(path) and files and os.path.getmtime(path) >= os.path.getmtime(files[-1]):
        import json
        _RENAMES = json.load(open(path))
        return _RENAMES
    last = {}   # (isin, symbol) -> last date seen
    for p in files:
        if os.path.getsize(p) > 50:
            d = pd.read_csv(p, usecols=["symbol", "series", "isin"])
            d = d[d["series"].isin(["EQ", "BE", "BZ"])]
            day = os.path.basename(p)[:-4]
            for k in zip(d["isin"], d["symbol"]):
                last[k] = day
    by_isin = {}
    for (isin, sym), day in last.items():
        by_isin.setdefault(isin, []).append((day, sym))
    sym_isins = {}
    for (isin, sym) in last:
        sym_isins.setdefault(sym, set()).add(isin)
    out = {}
    for isin, v in by_isin.items():
        if len(v) < 2:
            continue
        v.sort()
        cur = v[-1][1]
        for day, old in v[:-1]:
            if old != cur and len(sym_isins[old]) == 1:   # skip symbols that point to more than one company
                out[old] = cur
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=0, sort_keys=True)
    _RENAMES = out
    return out


def load_market_panel(min_days=60, series=("EQ", "BE", "BZ")) -> dict[str, pd.DataFrame]:
    """Like load_panel, but for EVERY main-board NSE stock, built from the daily whole-market files
    (scripts/fetch_bhav.py). A stock counts only on days it was listed and traded, so later delistings
    and new listings are both included (no survivorship). Also returns 'dlv' (delivery fraction)."""
    import glob
    parts = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "extra", "bhav", "*.csv"))):
        if os.path.getsize(p) > 50:
            d = pd.read_csv(p, usecols=["symbol", "series", "open", "high", "low", "close", "volume", "value", "dlv_qty"])
            d["Date"] = pd.Timestamp(os.path.basename(p)[:-4])
            parts.append(d)
    b = pd.concat(parts, ignore_index=True)
    b = b[b["series"].isin(series) & (b["volume"] > 0) & (b["close"] > 0)]
    ren = symbol_renames()
    if ren:   # one continuous history per company across symbol changes
        b["symbol"] = b["symbol"].replace(ren)
    b["pri"] = b["series"].map({s: i for i, s in enumerate(series)})
    b = b.sort_values(["symbol", "Date", "pri"]).drop_duplicates(["symbol", "Date"])
    b = b[b.groupby("symbol")["Date"].transform("size") >= min_days]
    b["PrevClose"] = b.groupby("symbol")["close"].shift(1)
    b = b.rename(columns={"open": "OpenPrice", "high": "HighPrice", "low": "LowPrice", "close": "ClosePrice",
                          "volume": "TotalTradedQuantity"})
    b["AveragePrice"] = b["value"] / b["TotalTradedQuantity"]
    b["dlv"] = (b["dlv_qty"] / b["TotalTradedQuantity"]).clip(0, 1)
    frames = {s: d.reset_index(drop=True) for s, d in b.groupby("symbol", sort=True)}
    ca = corporate_actions()
    ca_by = dict(tuple(ca.groupby("symbol")))
    logs = {s: [] for s in frames}
    ret = pd.DataFrame({s: _adjusted_returns(d, ca_by.get(s), logs[s]) for s, d in frames.items()}).sort_index()
    adj_close = (1 + ret.fillna(0)).cumprod()
    W = lambda col: b.pivot(index="Date", columns="symbol", values=col).reindex(index=ret.index, columns=ret.columns)
    raw_close = W("ClosePrice")
    adj_close = adj_close.where(raw_close.notna())
    out = {"ret": ret, "close": adj_close}
    for name, col in [("open", "OpenPrice"), ("high", "HighPrice"), ("low", "LowPrice"), ("vwap", "AveragePrice")]:
        out[name] = W(col) / raw_close * adj_close
    vol = W("TotalTradedQuantity")
    out["volume"] = vol * raw_close / adj_close
    out["volume_raw"] = vol
    out["close_raw"] = raw_close
    out["dlv"] = W("dlv")
    out["events"] = {s: logs[s] for s in frames if logs[s]}
    return out


if __name__ == "__main__":
    import sys
    syms = (nifty500_symbols() if "--nifty500" in sys.argv else
            nifty100_symbols() if "--nifty100" in sys.argv else nifty50_symbols())
    for i, s in enumerate(syms):
        fetch_symbol(s)
        print(f"[{i+1}/{len(syms)}] {s} ok", flush=True)
