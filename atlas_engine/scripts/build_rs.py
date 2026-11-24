"""Relative strength: sectors vs the Nifty 500, stocks vs the Nifty 500 and vs their own sector.

    python3 scripts/build_rs.py      -> explorer/data/deep/rs.json

RS over a period = (1 + own return) / (1 + benchmark return) - 1.
RS rating (1-99): percentile of a weighted RS score (40% last 3 months, 20% each of 6, 9 and 12 months),
the way chart services rate leadership. Stocks are rated against all index stocks; sectors against all groups.
A stock's sector = its NSE industry, measured as the equal-weight average of every listed stock in that industry.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel
from rsi.step3_features import _index, S3, skip_index, MIN_DAYS, index_universe, index_category, index_name

OUT = "explorer/data/deep"


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


def stat(v):
    v = pd.Series(v).dropna()
    return None if len(v) < 30 else {"n": len(v), "mean": float(v.mean()), "median": float(v.median()), "pos": float((v > 0).mean())}


p = load_market_panel()
close, ret = p["close"], p["ret"]
idx, cols = close.index, close.columns
T = idx[-1]
listed = close.notna()
logc = np.log(close)
n500 = _index("NIFTY 500", idx).ffill()
UNI = index_universe()
P = {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 250}
W = idx[::-1][::5][::-1]
Wy = W[-53:]

# groups
G = {}
for f in glob.glob(os.path.join(S3, "indices", "*.csv")):
    n = index_name(f)
    if skip_index(n):
        continue
    s = _index(n, idx)
    if s is not None and s.notna().sum() >= MIN_DAYS:
        G["index: " + n] = s
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
IND = {}
for name, syms in ind.groupby(ind).groups.items():
    syms = [s for s in syms if s in cols]
    if len(syms) >= 3:
        IND[name] = np.log1p(ret[syms].where(listed[syms]).mean(1).fillna(0)).cumsum().where(listed[syms].sum(1) >= 2)
        if len(syms) >= 8:
            G["industry: " + name] = IND[name]
L = pd.DataFrame(G)
RSg = L.sub(n500, axis=0)                                   # log relative strength vs Nifty 500


def rs_score(R):
    """Weighted RS (log): 40% 3m + 20% 6m + 20% 9m + 20% 12m."""
    return 0.4 * R.diff(63) + 0.2 * R.diff(126) + 0.2 * R.diff(189) + 0.2 * R.diff(250)


def rating(score):
    return (score.rank(axis=1, pct=True) * 98 + 1).round()


def rs_trend(R):
    ma = R.rolling(50, min_periods=30).mean()
    return pd.DataFrame(np.select([(R > ma) & (ma.diff(10) > 0), (R < ma) & (ma.diff(10) < 0)], ["rising", "falling"], "flat"),
                        index=R.index, columns=R.columns).where(ma.notna(), "")


# ---------------------------------------------------------------- sectors
g_score = rs_score(RSg); g_rate = rating(g_score)
g_hi = RSg >= RSg.rolling(250, min_periods=120).max() - 1e-9
g_tr = rs_trend(RSg)
sectors = []
for k in L.columns:
    if not np.isfinite(RSg[k].loc[T]):
        continue
    sectors.append({"g": k, "cat": index_category(k), **{f"vs{h}": np.expm1(RSg[k].diff(n).loc[T]) for h, n in P.items()},
                    "abs1m": np.expm1(L[k].diff(21).loc[T]), "abs3m": np.expm1(L[k].diff(63).loc[T]),
                    "rating": g_rate[k].loc[T], "rating4w": g_rate[k].loc[W[-5]], "rsHigh": bool(g_hi[k].iloc[-5:].any()),
                    "trend": g_tr[k].loc[T], "line": np.exp(RSg[k].reindex(Wy) - RSg[k].reindex(Wy).dropna().iloc[0]).round(4).tolist()
                    if RSg[k].reindex(Wy).notna().any() else []})

# ---------------------------------------------------------------- stocks
RSm = logc.sub(n500, axis=0)                                # vs Nifty 500
sec_of = pd.Series({s: ind.get(s) for s in cols})
INDf = pd.DataFrame(IND)
SEC = pd.DataFrame({s: INDf[sec_of[s]] if sec_of[s] in INDf.columns else np.nan for s in cols}, index=idx)
RSs = logc - SEC                                            # vs own industry
uni_cols = [s for s in cols if s in UNI]
m_score = rs_score(RSm)
m_rate_all = (m_score.rank(axis=1, pct=True) * 98 + 1).round()            # vs all listed stocks (for history tests)
m_rate = (m_score[uni_cols].rank(axis=1, pct=True) * 98 + 1).round()      # vs index stocks (shown)
s_score = rs_score(RSs)
ind_rank = s_score[uni_cols].loc[T].groupby(sec_of[uni_cols]).rank(ascending=False)
ind_n = sec_of[uni_cols].map(sec_of[uni_cols].value_counts())
m_hi = RSm >= RSm.rolling(250, min_periods=120).max() - 1e-9
m_tr = rs_trend(RSm)
s_tr = rs_trend(RSs)
ind_phase = {}
rr = RSg - RSg.rolling(126, min_periods=100).mean(); mm = rr.diff(21)
for name in IND:
    k = "industry: " + name
    if k in rr.columns:
        x, y = rr[k].loc[T], mm[k].loc[T]
        ind_phase[name] = "" if not (np.isfinite(x) and np.isfinite(y)) else ("Leading" if x > 0 and y > 0 else "Weakening" if x > 0 else "Lagging" if y <= 0 else "Improving")
sec_rate = {name: g_rate["industry: " + name].loc[T] for name in IND if "industry: " + name in g_rate.columns}
stocks = {}
for s in uni_cols:
    if not np.isfinite(logc[s].loc[T]):
        continue
    v3m, s3m = RSm[s].diff(63).loc[T], RSs[s].diff(63).loc[T]
    quad = None
    if np.isfinite(v3m) and np.isfinite(s3m):
        quad = "Beats both" if v3m > 0 and s3m > 0 else "Beats sector only" if s3m > 0 else "Beats market only" if v3m > 0 else "Lags both"
    sec = sec_of[s]
    stocks[s] = {"sec": sec, **{f"m{h}": np.expm1(RSm[s].diff(n).loc[T]) for h, n in P.items()},
                 **{f"s{h}": np.expm1(RSs[s].diff(n).loc[T]) for h, n in P.items()},
                 "rating": m_rate[s].loc[T], "rating4w": m_rate[s].loc[W[-5]], "secRank": ind_rank.get(s), "secN": ind_n.get(s),
                 "rsHigh": bool(m_hi[s].iloc[-5:].any()), "trendM": m_tr[s].loc[T], "trendS": s_tr[s].loc[T], "quad": quad,
                 "secPhase": ind_phase.get(sec), "secRating": sec_rate.get(sec),
                 "lineM": np.exp(RSm[s].reindex(Wy) - RSm[s].reindex(Wy).dropna().iloc[0]).round(4).tolist() if RSm[s].reindex(Wy).notna().any() else [],
                 "lineS": np.exp(RSs[s].reindex(Wy) - RSs[s].reindex(Wy).dropna().iloc[0]).round(4).tolist() if RSs[s].reindex(Wy).notna().any() else []}

# ---------------------------------------------------------------- does RS predict? (all listed stocks, weekly since 2016)
print("evidence", flush=True)
Wp = W[(W >= "2016-06-01") & (W <= T - pd.Timedelta(days=200))]
fwd = {h: (RSm.shift(-n) - RSm).loc[Wp] for h, n in [("1m", 21), ("3m", 63), ("6m", 126)]}
R = m_rate_all.loc[Wp]
ev = []
for lo, hi, lab in [(90, 100, "RS rating 90-99 (top tenth)"), (80, 89, "RS rating 80-89"), (50, 79, "RS rating 50-79"),
                    (21, 49, "RS rating 21-49"), (1, 20, "RS rating 1-20 (bottom fifth)")]:
    m = (R >= lo) & (R <= hi)
    ev.append({"case": lab, **{h: stat(fwd[h][m].stack()) for h in fwd}})
Sw = s_score.loc[Wp]
secw = pd.DataFrame({s: g_rate["industry: " + sec_of[s]].reindex(Wp) if isinstance(sec_of[s], str) and "industry: " + sec_of[s] in g_rate.columns else np.nan for s in cols}, index=Wp)
strong_s = Sw.rank(axis=1, pct=True) > 0.8
for lab, m in [("Top-fifth RS stock in a top-third sector", (R >= 80) & (secw >= 67)),
               ("Top-fifth RS stock in a bottom-third sector", (R >= 80) & (secw <= 33)),
               ("Bottom-fifth RS stock in a top-third sector", (R <= 20) & (secw >= 67)),
               ("Beats both its sector and the market over 3m", (RSm.diff(63).loc[Wp] > 0) & (RSs.diff(63).loc[Wp] > 0)),
               ("Lags both its sector and the market over 3m", (RSm.diff(63).loc[Wp] < 0) & (RSs.diff(63).loc[Wp] < 0)),
               ("RS line (vs Nifty 500) at a 52-week high", m_hi.loc[Wp]),
               ("Any stock", listed.loc[Wp])]:
    ev.append({"case": lab, **{h: stat(fwd[h][m].stack()) for h in fwd}})
fwdg = {h: (RSg.shift(-n) - RSg).loc[Wp] for h, n in [("1m", 21), ("3m", 63), ("6m", 126)]}
Rg = g_rate.loc[Wp]
evg = []
for lo, hi, lab in [(80, 100, "Sector RS rating 80-99"), (40, 79, "Sector RS rating 40-79"), (1, 39, "Sector RS rating 1-39")]:
    m = (Rg >= lo) & (Rg <= hi)
    evg.append({"case": lab, **{h: stat(fwdg[h][m].stack()) for h in fwdg}})
evg.append({"case": "Sector RS line at a 52-week high", **{h: stat(fwdg[h][g_hi.loc[Wp]].stack()) for h in fwdg}})
evg.append({"case": "Any sector", **{h: stat(fwdg[h].stack()) for h in fwdg}})

json.dump(clean({"date": T, "weeks": [str(w.date()) for w in Wy], "sectors": sectors, "stocks": stocks, "evidence": ev, "evidenceGroups": evg}),
          open(os.path.join(OUT, "rs.json"), "w"), separators=(",", ":"))
print("DONE rs", os.path.getsize(os.path.join(OUT, "rs.json")) // 1024, "KB", flush=True)
