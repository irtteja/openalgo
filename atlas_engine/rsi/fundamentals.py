"""Quarterly financial results (NSE XBRL filings) via nselib, cached in data/extra/results/.

Each filing is usable by the model only from the trading day AFTER its board-meeting
date (the day results are announced), so nothing is known before it was public.

    python -m rsi.fundamentals         # download weekly filing batches 2021 -> today
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd

from .data import DATA_DIR

RES = os.path.join(DATA_DIR, "extra", "results")
KEEP = ["Symbol", "DateOfBoardMeetingWhenFinancialResultsWereApproved", "DateOfEndOfReportingPeriod",
        "DateOfStartOfReportingPeriod", "NatureOfReportStandaloneConsolidated", "LevelOfRoundingUsedInFinancialStatements",
        "RevenueFromOperations", "ProfitLossForPeriod", "ProfitBeforeTax",
        "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations", "PaidUpValueOfEquityShareCapital",
        "FaceValueOfEquityShareCapital"]


def download(start="2021-01-01", workers=1, passes=3):
    """Use NSE's filings index (fast), keep only our universe, one filing per company per
    quarter (consolidated preferred, latest revision), then fetch just those XBRL files."""
    import xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from nselib.capital_market.get_func import get_financial_results_master
    from .data import nifty500_symbols
    os.makedirs(RES, exist_ok=True)
    universe = set(nifty500_symbols())
    windows = pd.date_range(start, pd.Timestamp.today(), freq="MS")
    for _pass in range(passes):
        _download_pass(windows, universe, workers, _pass)


def _download_pass(windows, universe, workers, _pass=0):
    import xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from nselib.capital_market.get_func import get_financial_results_master
    for i, a in enumerate(windows):
        path = os.path.join(RES, f"m_{a.strftime('%Y-%m')}.csv")
        recent = a >= pd.Timestamp.today().normalize() - pd.DateOffset(months=1) - pd.offsets.MonthBegin(1)
        if os.path.exists(path) and not (recent and _pass == 0):   # this month and last month are fetched again
            continue
        b = min(a + pd.offsets.MonthEnd(0), pd.Timestamp.today())
        m, headers, ns, keys = _retry(lambda: get_financial_results_master(
            a.strftime("%d-%m-%Y"), b.strftime("%d-%m-%Y"), None, False, "Quarterly"))
        if m is None or "symbol" not in getattr(m, "columns", []):  # NSE returned nothing / error page
            print(f"[{i+1}/{len(windows)}] {a.strftime('%Y-%m')}: empty index, retrying later", flush=True)
            time.sleep(30)
            continue
        m = m[m["symbol"].isin(universe) & m["xbrl"].astype(str).str.endswith(".xml")].copy()
        m["cons"] = (m["consolidated"].astype(str) == "Consolidated").astype(int)
        m["bc"] = pd.to_datetime(m["broadCastDate"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
        m = m.sort_values(["symbol", "toDate", "cons", "bc"]).drop_duplicates(["symbol", "toDate"], keep="last")

        def fetch(row):
            try:
                r = requests.get(row.xbrl, headers=headers, timeout=30)
                root = ET.fromstring(r.content)
                rec = {k: (lambda e: e.text if e is not None else None)(root.find(f".//in-bse-fin:{k}", ns)) for k in keys}
            except Exception:
                return None
            rec.update(Symbol=row.symbol, broadcast=str(row.bc), nature=row.consolidated)
            return rec

        with ThreadPoolExecutor(workers) as ex:
            recs = [r for r in ex.map(fetch, m.itertuples()) if r]
        if len(m) and len(recs) < 0.9 * len(m):  # throttled: don't cache a partial month
            print(f"[{i+1}/{len(windows)}] {a.strftime('%Y-%m')}: only {len(recs)}/{len(m)} parsed, retrying later", flush=True)
            time.sleep(60)
            continue
        pd.DataFrame(recs).to_csv(path, index=False)
        print(f"[{i+1}/{len(windows)}] {a.strftime('%Y-%m')}: {len(m)} filings for our stocks, {len(recs)} parsed", flush=True)


def _retry(fn, tries=3):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def load_results() -> pd.DataFrame:
    files = [os.path.join(RES, f) for f in os.listdir(RES) if f.endswith(".csv")] if os.path.isdir(RES) else []
    if not files:
        return pd.DataFrame(columns=KEEP)
    df = pd.concat([pd.read_csv(f) for f in files if os.path.getsize(f) > 5], ignore_index=True)
    if "nature" in df.columns:  # index flag from the newer format; merge into the XBRL field
        col = "NatureOfReportStandaloneConsolidated"
        df[col] = df[col].fillna(df["nature"]) if col in df.columns else df["nature"]
        df = df.drop(columns=["nature"])
    if "broadcast" in df.columns:  # exact publication time (newer download format)
        bc = pd.to_datetime(df["broadcast"], errors="coerce")
        bm = pd.to_datetime(df["DateOfBoardMeetingWhenFinancialResultsWereApproved"], errors="coerce")
        df["DateOfBoardMeetingWhenFinancialResultsWereApproved"] = bc.fillna(bm)
    df = df.rename(columns={"DateOfBoardMeetingWhenFinancialResultsWereApproved": "announced",
                            "DateOfEndOfReportingPeriod": "period_end", "DateOfStartOfReportingPeriod": "period_start",
                            "NatureOfReportStandaloneConsolidated": "nature",
                            "RevenueFromOperations": "revenue", "ProfitLossForPeriod": "profit",
                            "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "eps"})
    for c in ["announced", "period_end", "period_start"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["revenue", "profit", "eps", "ProfitBeforeTax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Quarterly reports only (period ~3 months); prefer consolidated over standalone.
    span = (df["period_end"] - df["period_start"]).dt.days
    df = df[(span > 80) & (span < 100)].dropna(subset=["announced", "period_end"])
    df["rank"] = (df["nature"].astype(str).str.lower() != "consolidated").astype(int)
    df = df.sort_values(["Symbol", "period_end", "rank", "announced"])
    return df.drop_duplicates(["Symbol", "period_end"], keep="first")


def add_fundamental_features(panel: dict) -> dict:
    """Wide (date x symbol) frames, forward-filled from the day after each announcement:
       ey      - earnings yield: 4 x quarterly EPS / price on that day (raw, unadjusted price,
                 because EPS is in the same per-share units as the day it was reported)
       rev_g   - revenue growth vs the same quarter a year earlier
       prof_g  - profit growth vs a year earlier (signed, scaled by |last year's profit|)
       fresh   - days since the latest result (recency of information)"""
    res = load_results()
    idx, cols = panel["close"].index, panel["close"].columns
    raw = panel["close_raw"] if "close_raw" in panel else None
    ey, rg, pg, fr = ({} for _ in range(4))
    for s, g in res[res["Symbol"].isin(cols)].groupby("Symbol"):
        g = g.sort_values("period_end").copy()
        g["q"] = g["period_end"].dt.to_period("Q")
        prev = g.set_index("q")
        ly = [prev.loc[q - 4] if (q - 4) in prev.index else None for q in g["q"]]
        g["rev_g"] = [r.revenue / ly_.revenue - 1 if ly_ is not None and ly_.revenue > 0 else np.nan for r, ly_ in zip(g.itertuples(), ly)]
        g["prof_g"] = [(r.profit - ly_.profit) / abs(ly_.profit) if ly_ is not None and ly_.profit else np.nan for r, ly_ in zip(g.itertuples(), ly)]
        # Available from the next trading day after the announcement.
        pos = idx.searchsorted(g["announced"].values, side="right")
        ok = pos < len(idx)
        g, pos = g[ok], pos[ok]
        avail = idx[pos]
        g = g.assign(avail=avail).drop_duplicates("avail", keep="last").set_index("avail")
        eps = g["eps"].reindex(idx).ffill(limit=130)
        if raw is not None:
            ey[s] = 4 * eps / raw[s]
        rg[s] = g["rev_g"].reindex(idx).ffill(limit=130)
        pg[s] = g["prof_g"].reindex(idx).ffill(limit=130)
        last = pd.Series(np.where(idx.isin(g.index), np.arange(len(idx)), np.nan), index=idx).ffill()
        fr[s] = np.arange(len(idx)) - last
    W = lambda d: pd.DataFrame(d).reindex(index=idx, columns=cols)
    panel["ey"] = W(ey).clip(-0.5, 0.5)
    panel["rev_g"] = W(rg).clip(-1, 3)
    panel["prof_g"] = W(pg).clip(-5, 5)
    panel["fresh"] = W(fr).clip(0, 130)
    return panel


if __name__ == "__main__":
    download()
