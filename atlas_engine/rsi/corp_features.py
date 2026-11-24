"""Company-event inputs from scripts/fetch_corporate.py (insider trades, pledges, shareholding, announcements).

  ins_net   - senior insiders' (promoters, directors, KMP, relatives) open-market buying minus selling
              over the last 60 trading days, as a fraction of the stock's traded value in that window
  pledge    - promoter pledge events over the last 120 trading days: +1 created, +2 invoked, -1 released
  prom_chg  - change in promoter holding % in the latest shareholding filing vs the one before
  (announcement inputs below only with announcements=True; off by default)
  ann_surp  - number of announcements in the last 5 days minus the stock's usual 5-day count (past year)
  ann_call  - analyst / investor meetings and con-call notices in the last 20 days
  ann_deal  - acquisition / order-win / press-release style announcements in the last 20 days

No look-ahead: every event counts from the first trading day AFTER NSE broadcast it.
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

from .data import DATA_DIR

CORP = os.path.join(DATA_DIR, "extra", "corp")
SENIOR = {"promoters", "promoter group", "director", "key managerial personnel", "immediate relative"}


def _read(sub):
    # r31_*.csv = SAST Regulation 31 pledge disclosures (only from May 2026): kept for display, left out of the
    # features so recent pledge scores stay comparable with the history the patterns learned from.
    ps = [p for p in glob.glob(os.path.join(CORP, sub, "*.csv")) if os.path.getsize(p) > 60 and not os.path.basename(p).startswith("r31_")]
    if not ps:
        return pd.DataFrame()
    t = pd.concat([pd.read_csv(p, dtype=str) for p in ps], ignore_index=True)
    if "symbol" in t:
        from rsi.data import symbol_renames
        t["symbol"] = t["symbol"].replace(symbol_renames())
    return t


def _next_day(ts: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Map broadcast timestamps to the first trading day strictly after the broadcast date."""
    pos = idx.searchsorted(ts.dt.normalize() + pd.Timedelta(days=1))
    out = pd.Series(pd.NaT, index=ts.index)
    ok = pos < len(idx)
    out[ok] = idx[pos[ok]]
    return out


def _daily(df, idx, cols, value):
    t = df.assign(v=value).dropna(subset=["day"])
    t = t[t["symbol"].isin(cols)]
    return t.pivot_table(index="day", columns="symbol", values="v", aggfunc="sum").reindex(index=idx, columns=cols).fillna(0.0)


def add_corp_features(panel: dict, announcements: bool = False) -> dict:
    idx, cols = panel["close"].index, panel["close"].columns

    ins = _read("insider")
    if len(ins):
        ins = ins.drop_duplicates()
        ins["day"] = _next_day(pd.to_datetime(ins["date"], format="%d-%b-%Y %H:%M", errors="coerce"), idx)
        cat = ins["personCategory"].astype(str).str.strip().str.lower()
        mode = ins["acqMode"].astype(str).str.strip().str.lower()
        val = pd.to_numeric(ins["secVal"], errors="coerce").fillna(0)
        senior = cat.isin(SENIOR)
        sign = np.where(mode == "market purchase", 1.0, np.where(mode == "market sale", -1.0, 0.0))
        net = _daily(ins, idx, cols, np.where(senior, sign * val, 0.0))
        price = panel.get("close_raw", panel["close"])
        traded = (price * panel.get("volume_raw", panel["volume"])).rolling(60, min_periods=20).sum()
        panel["ins_net"] = (net.rolling(60, min_periods=1).sum() / traded).clip(-1, 1).fillna(0.0)

        promo = cat.isin({"promoters", "promoter group"})
        pw = np.select([mode == "pledge creation", mode == "invocation of pledge", mode == "revokation of pledge"],
                       [1.0, 2.0, -1.0], 0.0)
        panel["pledge"] = _daily(ins, idx, cols, np.where(promo, pw, 0.0)).rolling(120, min_periods=1).sum()

    shp = _read("shp")
    if len(shp):
        shp["prom"] = pd.to_numeric(shp["pr_and_prgrp"], errors="coerce")
        shp["qtr"] = pd.to_datetime(shp["date"], format="%d-%b-%Y", errors="coerce")
        shp = shp.dropna(subset=["qtr"]).sort_values(["symbol", "qtr"]).drop_duplicates(["symbol", "qtr"], keep="last")
        shp["chg"] = shp.groupby("symbol")["prom"].diff()
        b = pd.to_datetime(shp["broadcastDate"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
        shp["day"] = _next_day(b.fillna(shp["qtr"] + pd.Timedelta(days=30)), idx)  # SEBI deadline: 21 days
        t = shp.dropna(subset=["day", "chg"])
        t = t[t["symbol"].isin(cols)]
        if len(t):
            pc = t.pivot_table(index="day", columns="symbol", values="chg", aggfunc="last")
            panel["prom_chg"] = pc.reindex(index=idx, columns=cols).ffill(limit=130)

    ann = _read("ann") if announcements else pd.DataFrame()
    if len(ann):
        ann["day"] = _next_day(pd.to_datetime(ann["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce"), idx)
        desc = (ann["desc"].fillna("") + " " + ann["attchmntText"].fillna("")).str.lower()
        n = _daily(ann, idx, cols, 1.0)
        n5 = n.rolling(5, min_periods=1).sum()
        panel["ann_surp"] = n5 - n.rolling(250, min_periods=20).sum() * 5 / 250
        call = desc.str.contains("analyst|institutional investor|con. call|conference call|earnings call|investor meet")
        panel["ann_call"] = _daily(ann, idx, cols, call.astype(float)).rolling(20, min_periods=1).sum()
        deal = desc.str.contains("acquisition|award|order|contract|bagging|press release|launch|agreement")
        panel["ann_deal"] = _daily(ann, idx, cols, deal.astype(float)).rolling(20, min_periods=1).sum()
    return panel
