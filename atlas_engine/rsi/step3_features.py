"""Step-3 inputs from scripts/fetch_step3.py: sector, market regime, margin trading, options.

  sec_rel   - stock's 20-day return minus the median 20-day return of its industry (in our universe)
  sec_rel5  - same over 5 days
  sec_mom   - its industry's median 60-day return (same for every stock in the industry)
  mkt20     - Nifty 500 20-day return (same for every stock)
  small_big - Smallcap 250 minus Nifty 50, 60-day return (same for every stock): small-cap mood
  sty_*     - style mood: 60-day return of the momentum / low-vol / quality / value / equal-weight index
              minus the Nifty 500 (same for every stock)
  grp_rel5  - stock's 5-day return minus that of its closest index (out of all NSE indices, chosen monthly
              from the past 250 days of moves beyond the market); grp_rel20 same over 20 days
  grp_mom   - that index's 60-day return minus the Nifty 500's
  mem_rel5  - stock's 5-day return minus the average of the sector/theme indices it belongs to TODAY
  mem_mom   - those indices' 60-day return minus the Nifty 500's
  mtf       - shares bought on margin (borrowed money) / the stock's average daily volume, log
  mtf_chg   - change in mtf over 5 days
  pcr       - put/call open-interest ratio of the stock's options, log, minus the day's median (0 if no options)
  opt_act   - options volume today vs its 60-day average, log (0 if no options)

Timing: index, options and price data are known at that day's close. The margin-trading file named for
day t is published on day t+1, so it is used from t+1.
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

from .data import DATA_DIR

S3 = os.path.join(DATA_DIR, "extra", "step3")


def _days(sub):
    out = {}
    for p in sorted(glob.glob(os.path.join(S3, sub, "*.csv"))):
        if os.path.getsize(p) > 20:
            out[pd.Timestamp(os.path.basename(p)[:-4])] = pd.read_csv(p)
    return out


def index_file(name):
    """File name for an index: '/' -> '~', ':' -> '=' (not allowed on Windows), spaces -> '_'."""
    return name.replace("/", "~").replace(":", "=").replace(" ", "_") + ".csv"


def index_name(path):
    return os.path.basename(path)[:-4].replace("_", " ").replace("~", "/").replace("=", ":")


def _index(name, idx):
    p = os.path.join(S3, "indices", index_file(name))
    if not os.path.exists(p) or os.path.getsize(p) < 100:
        return None
    d = pd.read_csv(p)
    d["Date"] = pd.to_datetime(d["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
    s = d.dropna(subset=["Date"]).drop_duplicates("Date").set_index("Date")["EOD_CLOSE_INDEX_VAL"].sort_index()
    return np.log(s.astype(float)).reindex(idx).ffill(limit=3)


BROAD = ("VIX", "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500", "LEV", "INVERSE", "DIVIDEND POINTS",
         "G-SEC", "GSEC", "BOND", "SDL", "T-BILL", "TBILL", "1D RATE", "REPO", "GOLD", "SILVER")


def _closest_index_features(panel, logc, n500, idx, cols, look=250):
    """grp_rel5 / grp_rel20: stock return minus its closest index's return over 5 / 20 days.
    grp_mom: closest index's 60-day return minus the Nifty 500's."""
    lv = {}
    for p in glob.glob(os.path.join(S3, "indices", "*.csv")):
        name = index_name(p)
        if any(b in name.upper() for b in BROAD) or os.path.getsize(p) < 1000:
            continue
        s = _index(name, idx)
        if s is not None and s.notna().mean() > 0.5:
            lv[name] = s
    if not lv:
        return
    L = pd.DataFrame(lv)
    m = n500.diff()
    ex_i = L.diff().sub(m, axis=0).values            # T x K index excess returns
    ex_s = logc.diff().sub(m, axis=0).values          # T x N stock excess returns
    T, N = ex_s.shape
    pick = np.full((T, N), -1)
    months = np.flatnonzero(pd.Series(idx.to_period("M")).ne(pd.Series(idx.to_period("M")).shift()).values)
    for k, t0 in enumerate(months):
        if t0 < look:
            continue
        a, b = ex_i[t0 - look:t0], ex_s[t0 - look:t0]     # past only: rows before t0
        okk = np.isfinite(a).all(0)
        if not okk.any():
            continue
        a = a[:, okk]; a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        bb = np.nan_to_num((b - np.nanmean(b, 0)) / (np.nanstd(b, 0) + 1e-12))
        c = bb.T @ a / look                                # N x K correlations
        best = np.flatnonzero(okk)[c.argmax(1)]
        best[np.isfinite(b).sum(0) < look // 2] = -1       # too little history for this stock
        t1 = months[k + 1] if k + 1 < len(months) else T
        pick[t0:t1] = best
    Lv = L.values
    def grp(diff):
        d = np.vstack([np.full((diff, Lv.shape[1]), np.nan), Lv[diff:] - Lv[:-diff]])
        out = np.take_along_axis(d, np.clip(pick, 0, None), axis=1)
        out[pick < 0] = np.nan
        return pd.DataFrame(out, index=idx, columns=cols)
    panel["grp_rel5"] = logc.diff(5) - grp(5)
    panel["grp_rel20"] = logc.diff(20) - grp(20)
    panel["grp_mom"] = grp(60).sub(n500.diff(60), axis=0)
    panel["_closest_index"] = pd.DataFrame(np.where(pick >= 0, np.array(L.columns, dtype=object)[np.clip(pick, 0, None)], None),
                                           index=idx, columns=cols)


STYLE_SIZE = ("NIFTY 50", "NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500", "MIDCAP", "SMALLCAP", "MICROCAP",
              "LARGEMIDCAP", "MIDSMALL", "TOTAL MARKET", "MOMENTUM", "QUALITY", "VALUE", "ALPHA", "LOW VOL", "LOW-VOL",
              "EQUAL", "DIVIDEND", "GROWTH", "HIGH BETA", "BETA", "FLEXICAP", "MULTICAP", "TOP ", "LEADERS", "SHARIAH",
              "SHARIAH", "IPO", "ESG", "NON-CYCLICAL")


def _member_features(panel, logc, n500, idx, cols):
    """mem_rel5 / mem_mom from TODAY's member lists of sector & theme indices (size/style lists skipped:
    stocks join those because they already rose, which would leak the future)."""
    p = os.path.join(S3, "constituents.csv")
    if not os.path.exists(p):
        return
    c = pd.read_csv(p)
    c = c[~c["index"].str.upper().apply(lambda n: any(b in n for b in STYLE_SIZE + BROAD))]
    c = c[c["symbol"].isin(cols)]
    lv = {n: _index(n, idx) for n in c["index"].unique()}
    lv = {n: s for n, s in lv.items() if s is not None and s.notna().mean() > 0.5}
    c = c[c["index"].isin(lv)]
    if c.empty:
        return
    L = pd.DataFrame(lv)
    W = pd.crosstab(c["symbol"], c["index"]).reindex(index=cols, columns=L.columns).fillna(0.0)
    W = W.div(W.sum(1).replace(0, np.nan), axis=0)   # equal weight over a stock's indices; NaN if none
    Wv = W.fillna(0).values.T                          # K x N

    def avg(d):  # average return of the stock's indices that have data that day
        r = L.diff(d).values
        num = np.nan_to_num(r) @ Wv
        den = np.isfinite(r).astype(float) @ Wv
        return pd.DataFrame(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=idx, columns=cols)
    panel["mem_rel5"] = logc.diff(5) - avg(5)
    panel["mem_mom"] = avg(60).sub(n500.diff(60), axis=0)


def add_step3_features(panel: dict) -> dict:
    idx, cols = panel["close"].index, panel["close"].columns
    bcast = lambda s: pd.DataFrame({c: s.reindex(idx) for c in cols})
    logc = np.log(panel["close"])

    p = os.path.join(S3, "sectors.csv")
    if os.path.exists(p):
        ind = pd.read_csv(p).set_index("Symbol")["Industry"].reindex(cols)  # unknown industry -> no sector input
        r20, r5, r60 = logc.diff(20), logc.diff(5), logc.diff(60)
        med = lambda r: r.T.groupby(ind).transform("median").T
        panel["sec_rel"] = r20 - med(r20)
        panel["sec_rel5"] = r5 - med(r5)
        panel["sec_mom"] = med(r60)

    n500, n50, sml = (_index(n, idx) for n in ("NIFTY 500", "NIFTY 50", "NIFTY SMALLCAP 250"))

    # Closest index: for every stock, the index (of all ~235 NSE indices) whose moves beyond the
    # market best match the stock's, judged on the PAST 250 days only and re-chosen each month.
    if n500 is not None:
        _closest_index_features(panel, logc, n500, idx, cols)
        _member_features(panel, logc, n500, idx, cols)
    if n500 is not None:
        panel["mkt20"] = bcast(n500.diff(20))
    if n50 is not None and sml is not None:
        panel["small_big"] = bcast(sml.diff(60) - n50.diff(60))

    # Style mood: which kind of stock has been winning over 60 days (each vs the Nifty 500), same for all stocks.
    if n500 is not None:
        for key, name in [("sty_mom", "NIFTY200 MOMENTUM 30"), ("sty_lowvol", "NIFTY100 LOW VOLATILITY 30"),
                          ("sty_qual", "NIFTY100 QUALITY 30"), ("sty_val", "NIFTY50 VALUE 20"),
                          ("sty_eqw", "NIFTY50 EQUAL WEIGHT")]:
            s = _index(name, idx)
            if s is not None and s.notna().mean() > 0.5:
                panel[key] = bcast(s.diff(60) - n500.diff(60))

    m = _days("mtf")
    if m:
        q = pd.DataFrame({d: df.groupby("symbol")["qty"].sum() for d, df in m.items()}).T.sort_index()
        q.index = q.index.map(lambda d: idx[min(idx.searchsorted(d + pd.Timedelta(days=1)), len(idx) - 1)])
        q = q[~q.index.duplicated(keep="last")].reindex(index=idx, columns=cols).ffill(limit=3)
        vol = panel.get("volume_raw", panel["volume"]).rolling(20, min_periods=5).mean()
        panel["mtf"] = np.log1p(q / vol)
        panel["mtf_chg"] = panel["mtf"].diff(5)

    o = _days("options")
    if o:
        get = lambda k: pd.DataFrame({d: df.set_index("symbol")[k] for d, df in o.items() if k in df}).T.sort_index().reindex(index=idx, columns=cols)
        ce, pe = get("oi_ce"), get("oi_pe")
        pcr = np.log((pe + 1) / (ce + 1))
        panel["pcr"] = pcr.sub(pcr.median(axis=1), axis=0).fillna(0.0)
        act = np.log1p(get("vol_ce").fillna(0) + get("vol_pe").fillna(0))
        panel["opt_act"] = (act - act.rolling(60, min_periods=10).mean()).where(ce.notna()).fillna(0.0)
    return panel


def index_universe():
    """Stocks that belong to at least one NSE index today, excluding the SME (Emerge) index."""
    c = pd.read_csv(os.path.join(S3, "constituents.csv"))
    c = c[~c["index"].str.upper().str.contains("SME|EMERGE")]
    return set(c["symbol"])


def is_sme(group_key):
    return "SME" in group_key.upper() or "EMERGE" in group_key.upper()


NONSTOCK = ("VIX", "LEVERAGE", "INVERSE", "G-SEC", "GSEC", "BOND", "SDL", "T-BILL", "TBILL", "1D RATE", "REPO", "GOLD",
            "SILVER", "DIVIDEND POINTS", "USD", "ARBITRAGE", "FUTURES")
MIN_DAYS, SHORT_DAYS = 120, 500        # shown if >= 120 days of prices; "short history" if under 500


def skip_index(name):
    """True for indices that are not stock groups (bonds, VIX, leverage...), the yardsticks, and the SME index."""
    u = name.upper()
    return u in {"NIFTY 500", "NIFTY CONSUMPTION"} or any(s in u for s in NONSTOCK) or is_sme(name)


_BROAD = {"NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY LARGEMIDCAP 250", "NIFTY MIDCAP 50", "NIFTY MIDCAP 100",
          "NIFTY MIDCAP 150", "NIFTY MIDSMALLCAP 400", "NIFTY SMALLCAP 50", "NIFTY SMALLCAP 100", "NIFTY SMALLCAP 250",
          "NIFTY SMALLCAP 500", "NIFTY MICROCAP 250", "NIFTY TOTAL MARKET"}
_SECTOR = {"NIFTY BANK", "NIFTY PRIVATE BANK", "NIFTY PSU BANK", "NIFTY FINANCIAL SERVICES", "NIFTY FINANCIAL SERVICES 25/50",
           "NIFTY FINANCIAL SERVICES EX-BANK", "NIFTY MIDSMALL FINANCIAL SERVICES", "NIFTY CAPITAL MARKETS", "NIFTY IT",
           "NIFTY MIDSMALL IT & TELECOM", "NIFTY PHARMA", "NIFTY HEALTHCARE INDEX", "NIFTY MIDSMALL HEALTHCARE", "NIFTY500 HEALTHCARE",
           "NIFTY AUTO", "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY", "NIFTY ENERGY", "NIFTY OIL & GAS", "NIFTY MEDIA",
           "NIFTY CONSUMER DURABLES", "NIFTY CHEMICALS", "NIFTY CEMENT"}
_THEME = {"NIFTY CPSE", "NIFTY PSE", "NIFTY MNC", "NIFTY COMMODITIES", "NIFTY INFRASTRUCTURE", "NIFTY HOUSING", "NIFTY CORE HOUSING",
          "NIFTY INDIA CONSUMPTION", "NIFTY INDIA NEW AGE CONSUMPTION", "NIFTY MIDSMALL INDIA CONSUMPTION", "NIFTY NON-CYCLICAL CONSUMER",
          "NIFTY RURAL", "NIFTY INDIA DEFENCE", "NIFTY INDIA DIGITAL", "NIFTY INDIA INTERNET", "NIFTY INDIA MANUFACTURING",
          "NIFTY INDIA TOURISM", "NIFTY MOBILITY", "NIFTY SERVICES SECTOR", "NIFTY CONGLOMERATE 50", "NIFTY IPO", "NIFTY GROWTH SECTORS 15",
          "NIFTY500 MULTICAP INDIA MANUFACTURING 50:30:20", "NIFTY500 MULTICAP INFRASTRUCTURE 50:30:20", "NIFTY500 MULTICAP 50:25:25",
          "NIFTY EV & NEW AGE AUTOMOTIVE", "NIFTY INDIA FPI 150", "NIFTY INDIA INFRASTRUCTURE & LOGISTICS", "NIFTY INDIA RAILWAYS PSU",
          "NIFTY INDIA SELECT 5 CORPORATE GROUPS (MAATR)", "NIFTY REITS & REALTY", "NIFTY TRANSPORTATION & LOGISTICS", "NIFTY WAVES"}


def index_category(key):
    """Broad market / Sectors / Themes / Strategy & style / Industries, for a group key like 'index: NIFTY IT'."""
    if key.startswith("industry: "):
        return "Industries"
    n = key.replace("index: ", "").upper()
    if n in _BROAD:
        return "Broad market"
    if n in _SECTOR:
        return "Sectors"
    if n in _THEME or "GROUP" in n:
        return "Themes"
    return "Strategy & style"
