"""Stage analysis, accumulation/distribution and screener data -> explorer/data/deep/screen.json

    python3 scripts/build_screen.py      (after build_rs.py and build_watch.py)

Stage (Weinstein-style, using the 150-day / 30-week average):
  2 Uptrend   - price above a RISING 150-day average, and 50-day average above the 150-day
  4 Downtrend - price below a FALLING 150-day average
  1 Base      - 150-day average flat (moved < 2% in a month) after a decline
  3 Top       - 150-day average flat after an advance
  (anything else keeps the stage it had the week before)
Accumulation / distribution (last 50 days), three parts ranked across stocks and averaged:
  up-day volume / down-day volume, up-day delivered shares / down-day delivered shares, and money flow
  (where the close sits in each day's range, weighted by volume). Grades A (top fifth, buying) .. E (bottom fifth, selling).
History checks use every listed stock, weekly, 2016 -> 2026, returns measured against the Nifty 500.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel
from rsi.fundamentals import add_fundamental_features
from rsi.corp_features import add_corp_features
from rsi.step3_features import _index, S3, index_universe

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
    return None if len(v) < 30 else {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()), "pos": float((v > 0).mean())}


p = add_corp_features(add_fundamental_features(load_market_panel()))
close, ret = p["close"], p["ret"]
idx, cols = close.index, close.columns
T = idx[-1]
listed = close.notna()
logc = np.log(close)
n500 = _index("NIFTY 500", idx).ffill()
UNI = index_universe()
W = idx[::-1][::5][::-1]
Wp = W[(W >= "2016-06-01") & (W <= T - pd.Timedelta(days=200))]

# ---------------------------------------------------------------- stages
print("stages", flush=True)
ma50, ma150, ma200 = (close.rolling(n, min_periods=int(n * 0.8)).mean() for n in (50, 150, 200))
slope = ma150 / ma150.shift(21) - 1
trend_before = ma150 / ma150.shift(126) - 1
raw = np.select([(close > ma150) & (slope > 0.005) & (ma50 > ma150),
                 (close < ma150) & (slope < -0.005),
                 (slope.abs() <= 0.02) & (trend_before < 0),
                 (slope.abs() <= 0.02) & (trend_before >= 0)], [2, 4, 1, 3], 0)
raw = pd.DataFrame(raw, index=idx, columns=cols).where(ma150.notna(), np.nan).replace(0, np.nan)
stage_d = raw.ffill(limit=60).where(listed)
stage = stage_d.loc[W]
# weeks in current stage
def weeks_in_stage(s):
    s = s.dropna()
    if s.empty:
        return None, None
    v = s.iloc[-1]; n = 0
    for x in s.iloc[::-1]:
        if x != v:
            break
        n += 1
    prev = s.iloc[-n - 1] if len(s) > n else None
    return n, prev

# ---------------------------------------------------------------- accumulation / distribution
print("acc/dist", flush=True)
vol = p["volume"]
up = ret > 0
dn = ret < 0
uv = vol.where(up, 0).rolling(50, min_periods=30).sum() / vol.where(dn, 0).rolling(50, min_periods=30).sum().replace(0, np.nan)
dq = p["dlv"] * vol
ud = dq.where(up, 0).rolling(50, min_periods=30).sum() / dq.where(dn, 0).rolling(50, min_periods=30).sum().replace(0, np.nan)
rng = (p["high"] - p["low"]).replace(0, np.nan)
mfm = ((close - p["low"]) - (p["high"] - close)) / rng
cmf = (mfm * vol).rolling(50, min_periods=30).sum() / vol.rolling(50, min_periods=30).sum()
ad_score = (np.log(uv).rank(axis=1, pct=True) + np.log(ud).rank(axis=1, pct=True) + cmf.rank(axis=1, pct=True)) / 3
ad_pct = ad_score.rank(axis=1, pct=True)
grade = pd.DataFrame(np.select([ad_pct > 0.8, ad_pct > 0.6, ad_pct > 0.4, ad_pct > 0.2], ["A", "B", "C", "D"], "E"),
                     index=idx, columns=cols).where(ad_pct.notna())

# ---------------------------------------------------------------- RS rating and sector phase (history, all listed stocks)
print("rs history", flush=True)
RSm = logc.sub(n500, axis=0)
score = 0.4 * RSm.diff(63) + 0.2 * RSm.diff(126) + 0.2 * RSm.diff(189) + 0.2 * RSm.diff(250)
rsr = score.rank(axis=1, pct=True) * 98 + 1
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
IND = {}
for name, syms in ind.groupby(ind).groups.items():
    syms = [s for s in syms if s in cols]
    if len(syms) >= 3:
        IND[name] = np.log1p(ret[syms].where(listed[syms]).mean(1).fillna(0)).cumsum()
INDf = pd.DataFrame(IND)
IRS = INDf.sub(n500, axis=0)
ir = IRS - IRS.rolling(126, min_periods=100).mean(); im = ir.diff(21)
IPH = pd.DataFrame(np.select([(ir > 0) & (im > 0), (ir > 0) & (im <= 0), (ir <= 0) & (im <= 0), (ir <= 0) & (im > 0)],
                             ["Leading", "Weakening", "Lagging", "Improving"], ""), index=idx, columns=IRS.columns)
sec_of = pd.Series({s: ind.get(s) for s in cols})
def per_stock(dfg):
    return pd.DataFrame({s: dfg[sec_of[s]] if isinstance(sec_of[s], str) and sec_of[s] in dfg.columns else pd.Series(np.nan, index=idx) for s in cols}, index=idx)
SPH = per_stock(IPH)
SEC = per_stock(INDf)
vs_sec3 = (logc - SEC).diff(63)

# ---------------------------------------------------------------- other flags
hi250 = logc.rolling(250, min_periods=120).max()
from_hi = np.exp(logc - hi250) - 1
d = logc.diff()
tight = d.rolling(21, min_periods=15).std() / d.rolling(250, min_periods=150).std()
rs_hi = RSm - RSm.rolling(250, min_periods=120).max()
base = (from_hi > -0.08) & (rs_hi > -0.08) & (tight < 0.75)
brk = (logc >= hi250 - 1e-9) & base.shift(1).rolling(21, min_periods=1).max().fillna(0).astype(bool)
brk10 = brk.rolling(10, min_periods=1).max().fillna(0).astype(bool)
vr = p["volume"].rolling(5, min_periods=3).mean() / p["volume"].rolling(60, min_periods=20).mean()
pledge = (p["pledge"] > 0) if "pledge" in p else pd.DataFrame(False, index=idx, columns=cols)
ins_sell = (p["ins_net"] < -0.001) if "ins_net" in p else pd.DataFrame(False, index=idx, columns=cols)
stage2_new = (stage_d == 2) & (stage_d.shift(40) != 2)
rs_rising = rsr - rsr.shift(21) > 5

# ---------------------------------------------------------------- presets (rule + history)
fwd = {h: (RSm.shift(-n) - RSm).loc[Wp] for h, n in [("1m", 21), ("3m", 63), ("6m", 126)]}
PRESETS = [
    ("leaders", "Leaders in leading sectors", "RS rating 80+, in Stage 2 (uptrend), and its sector is Leading or Improving.",
     lambda: (rsr >= 80) & (stage_d == 2) & SPH.isin(["Leading", "Improving"])),
    ("stage2new", "Fresh Stage 2 uptrends", "Moved into Stage 2 within the last 8 weeks, with RS rating 60+.",
     lambda: stage2_new & (rsr >= 60)),
    ("bases", "Quiet bases near highs", "Within 8% of the 52-week high with unusually small swings (a base), RS rating 60+.",
     lambda: base & (rsr >= 60)),
    ("breakouts", "Breakouts with volume", "Broke out of a base to a new 52-week high in the last 10 days, on volume 1.5x usual or more.",
     lambda: brk10 & (vr >= 1.5)),
    ("accum", "Accumulation in an uptrend", "Accumulation grade A (heavy buying in volume and delivery) and in Stage 2.",
     lambda: (ad_pct > 0.8) & (stage_d == 2)),
    ("secstrong", "Beating a weak sector", "Beat its own sector over 3 months while the sector is Lagging or Weakening (stand-out stocks).",
     lambda: (vs_sec3 > 0.10) & SPH.isin(["Lagging", "Weakening"]) & (stage_d == 2)),
    ("bottoming", "Bottoming: Stage 1 with improving strength", "In Stage 1 (base after a decline) and RS rating up 5+ points in a month.",
     lambda: (stage_d == 1) & rs_rising),
    ("distrib", "Warning: distribution", "Distribution grade E (heavy selling) while in Stage 3 (top) or Stage 4 (downtrend).",
     lambda: (ad_pct <= 0.2) & stage_d.isin([3, 4])),
    ("risk", "Warning: pledges or insider selling in a downtrend", "Promoter pledged shares or insiders sold in the last few months, and in Stage 4.",
     lambda: (pledge | ins_sell) & (stage_d == 4)),
]
print("presets", flush=True)
pres = []
masks_now = {}
for key, name, rule, fn in PRESETS:
    m = fn().fillna(False).astype(bool)
    masks_now[key] = m.loc[T]
    mh = m.loc[Wp] & listed.loc[Wp]
    yr = {}
    for y in sorted(set(Wp.year)):
        sel = mh.loc[Wp.year == y]
        v = fwd["3m"].loc[sel.index][sel].stack().dropna()
        if len(v) >= 10:
            yr[int(y)] = {"mean": float(v.mean()), "pos": float((v > 0).mean()), "n": int(len(v))}
    pres.append({"key": key, "name": name, "rule": rule, **{h: stat(fwd[h][mh].stack()) for h in fwd},
                 "perWeek": float(mh.sum(1).mean()), "years": yr})
anyst = {h: stat(fwd[h][listed.loc[Wp]].stack()) for h in fwd}

# stage and grade evidence
stg_ev = [{"stage": int(s), **{h: stat(fwd[h][stage.loc[Wp] == s].stack()) for h in fwd}} for s in (1, 2, 3, 4)]
grd_ev = [{"grade": g, **{h: stat(fwd[h][grade.loc[Wp] == g].stack()) for h in fwd}} for g in "ABCDE"]
trans = []
st_prev = stage.shift(4)
for a, b in [(1, 2), (2, 3), (3, 4), (4, 1), (3, 2), (1, 4)]:
    m = (st_prev.loc[Wp] == a) & (stage.loc[Wp] == b)
    trans.append({"from": a, "to": b, **{h: stat(fwd[h][m].stack()) for h in fwd}})
# market by stage over time (all listed stocks) and for index stocks today
mk = {str(s): (stage == s).sum(1).div(stage.notna().sum(1)).round(4).tolist() for s in (1, 2, 3, 4)}

# ---------------------------------------------------------------- current rows (index stocks)
print("rows", flush=True)
rsj = json.load(open(os.path.join(OUT, "rs.json")))["stocks"]
det = json.load(open("results/patterns/detail_stocks.json"))["rows"]
mcap = {}
try:
    mc = pd.read_csv(os.path.join("data", "extra", "step3", "snapshot", "mcap.csv")); mc.columns = [c.strip() for c in mc.columns]
    mc["Symbol"] = mc["Symbol"].str.strip(); mcap = (pd.to_numeric(mc.drop_duplicates("Symbol").set_index("Symbol")["Market Cap(Rs.)"], errors="coerce") / 1e7).to_dict()
except Exception:
    pass
tv = ((p["close_raw"] * p["volume_raw"]).rolling(60, min_periods=20).mean() / 1e7).loc[T]
rows = []
for s in cols:
    if s not in UNI or not np.isfinite(close[s].loc[T]):
        continue
    n_in, prev = weeks_in_stage(stage[s])
    rj = rsj.get(s, {}); pr = det.get(s, {}).get("prob", {})
    rows.append({"s": s, "ind": ind.get(s), "price": p["close_raw"][s].loc[T], "mcap": mcap.get(s), "tv": tv.get(s),
                 "stage": stage_d[s].loc[T], "stageWeeks": n_in, "stagePrev": prev,
                 "ad": grade[s].loc[T], "adScore": ad_pct[s].loc[T], "upDownVol": uv[s].loc[T], "upDownDlv": ud[s].loc[T], "cmf": cmf[s].loc[T],
                 "rs": rj.get("rating"), "rs4w": rj.get("rating4w"), "vsSec3": rj.get("s3m"), "vsMkt3": rj.get("m3m"), "secPhase": rj.get("secPhase"),
                 "secRating": rj.get("secRating"), "fromHigh": from_hi[s].loc[T],
                 "above50": bool(close[s].loc[T] > ma50[s].loc[T]) if np.isfinite(ma50[s].loc[T]) else None,
                 "above200": bool(close[s].loc[T] > ma200[s].loc[T]) if np.isfinite(ma200[s].loc[T]) else None,
                 "base": bool(base[s].loc[T]), "breakout": bool(brk10[s].loc[T]), "volRatio": vr[s].loc[T],
                 "dlv": p["dlv"][s].rolling(20, min_periods=10).mean().loc[T],
                 "pledge": bool(pledge[s].loc[T]), "insSell": bool(ins_sell[s].loc[T]),
                 "pe": (1 / p["ey"][s].loc[T]) if "ey" in p and p["ey"][s].loc[T] > 0.002 else None,
                 "r1m": np.expm1(logc[s].diff(21).loc[T]), "r3m": np.expm1(logc[s].diff(63).loc[T]), "r1y": np.expm1(logc[s].diff(250).loc[T]),
                 "pBeat3": pr.get("3m|beat"), "pFall3": pr.get("3m|big_down"),
                 "presets": [k for k in masks_now if bool(masks_now[k].get(s, False))]})
uni_stage = pd.Series([r["stage"] for r in rows]).value_counts(normalize=True).to_dict()

json.dump(clean({"date": T, "rows": rows, "presets": pres, "anyStock": anyst, "stageEvidence": stg_ev, "gradeEvidence": grd_ev,
                 "transitions": trans, "marketStages": {"d": [str(w.date()) for w in W], **mk}, "indexStockStages": uni_stage}),
          open(os.path.join(OUT, "screen.json"), "w"), separators=(",", ":"))
print("DONE screen", len(rows), os.path.getsize(os.path.join(OUT, "screen.json")) // 1024, "KB", flush=True)
