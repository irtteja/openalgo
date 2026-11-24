"""Rotation tracker, market breadth and breakout watch -> explorer/data/deep/watch.json

    python3 scripts/build_watch.py
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel
from rsi.step3_features import _index, S3, skip_index, MIN_DAYS, SHORT_DAYS, index_universe, index_category, index_name

OUT = "explorer/data/deep"
os.makedirs(OUT, exist_ok=True)


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
    if len(v) < 15:
        return None
    return {"n": len(v), "mean": v.mean(), "median": v.median(), "pos": (v > 0).mean()}


p = load_market_panel()
close, ret = p["close"], p["ret"]
idx, cols = close.index, close.columns
T = idx[-1]
listed = close.notna()
logc = np.log(close)
mkt = ret.where(listed).mean(1)
X = ret.sub(mkt, axis=0).fillna(0).cumsum().where(listed)
n500, n50, sml = _index("NIFTY 500", idx), _index("NIFTY 50", idx), _index("NIFTY SMALLCAP 250", idx)
W = idx[::-1][::5][::-1]
Wp = W[W <= T - pd.Timedelta(days=200)]
FWD = {"1m": 21, "3m": 63, "6m": 126}
UNI = index_universe()
detail_s = json.load(open("results/patterns/detail_stocks.json"))["rows"]
detail_g = json.load(open("results/patterns/detail_groups.json"))["rows"]

# ============================================================ groups (same definitions as the rest of the page)
G, members = {}, {}
for f in glob.glob(os.path.join(S3, "indices", "*.csv")):
    n = index_name(f)
    if skip_index(n):
        continue
    s = _index(n, idx)
    if s is not None and s.notna().sum() >= MIN_DAYS:
        G["index: " + n] = s
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
for name, syms in ind.groupby(ind).groups.items():
    syms = [s for s in syms if s in cols]
    if len(syms) >= 8:
        G["industry: " + name] = np.log1p(ret[syms].where(listed[syms]).mean(1).fillna(0)).cumsum().where(listed[syms].sum(1) >= 5)
        members["industry: " + name] = syms
cons = pd.read_csv(os.path.join(S3, "constituents.csv"))
for k in G:
    if k.startswith("index: "):
        members[k] = [s for s in cons.loc[cons["index"] == k[7:], "symbol"] if s in cols]
L = pd.DataFrame(G)
RS = L.sub(n500, axis=0)
ratio = RS - RS.rolling(126, min_periods=100).mean()
mom = ratio.diff(21)
PH = pd.DataFrame(np.select([(ratio > 0) & (mom > 0), (ratio > 0) & (mom <= 0), (ratio <= 0) & (mom <= 0), (ratio <= 0) & (mom > 0)],
                            ["Leading", "Weakening", "Lagging", "Improving"], ""), index=idx, columns=L.columns).where(ratio.notna() & mom.notna(), "")

# ============================================================ 1. ROTATION
print("rotation", flush=True)
sx = ratio.loc[W].std()
sy = mom.loc[W].std()
zx, zy = ratio.loc[W] / sx, mom.loc[W] / sy
quad = lambda x, y: np.select([(x > 0) & (y > 0), (x > 0) & (y <= 0), (x <= 0) & (y <= 0), (x <= 0) & (y > 0)],
                              ["Leading", "Weakening", "Lagging", "Improving"], "")
# early warning: extend the last 4 weeks' path 4 weeks ahead; flag when it lands in another phase
px = zx + (zx - zx.shift(4))
py = zy + (zy - zy.shift(4))
proj = pd.DataFrame(quad(px.values, py.values), index=W, columns=L.columns)
nowq = pd.DataFrame(quad(zx.values, zy.values), index=W, columns=L.columns)
later = nowq.shift(-4)
flag = (proj != nowq) & (proj != "") & (nowq != "")
hist_flags = pd.DataFrame({"now": nowq[flag].stack(), "proj": proj[flag].stack(), "later": later[flag].stack()}).dropna()
hist_flags = hist_flags[hist_flags["later"] != ""]
acc = hist_flags.assign(hit=hist_flags["proj"] == hist_flags["later"]).groupby(["now", "proj"])["hit"].agg(["mean", "count"])
base_move = pd.DataFrame({"now": nowq.stack(), "later": later.stack()}).dropna()
base_move = base_move[(base_move["now"] != "") & (base_move["later"] != "")]
base_rate = base_move.groupby("now")["later"].value_counts(normalize=True)
warn_acc = [{"from": a, "to": b, "hit": r["mean"], "n": int(r["count"]), "base": base_rate.get((a, b), np.nan)} for (a, b), r in acc.iterrows()]
rot = []
last = W[-1]
for k in L.columns:
    if not np.isfinite(zx[k].loc[last]) or not np.isfinite(zy[k].loc[last]):
        continue
    tail = [[zx[k].loc[w], zy[k].loc[w]] for w in W[-9:]]
    ph = PH[k].loc[T]
    ph4 = PH[k].loc[W[-5]] if len(W) > 5 else ""
    nxt = proj[k].loc[last] if flag[k].loc[last] else None
    a = next((x for x in warn_acc if x["from"] == nowq[k].loc[last] and x["to"] == nxt), None) if nxt else None
    rot.append({"g": k, "cat": index_category(k), "x": zx[k].loc[last], "y": zy[k].loc[last], "tail": tail, "phase": ph,
                "phase4wAgo": ph4, "changed": ph != ph4 and ph4 != "", "next": nxt, "nextHit": a["hit"] if a else None,
                "rs1m": RS[k].diff(21).loc[T], "rs3m": RS[k].diff(63).loc[T], "short": int(L[k].notna().sum()) < SHORT_DAYS})

# ============================================================ 2. BREADTH (all listed main-board stocks, so history is not survivor-only)
print("breadth", flush=True)
ma50, ma200 = close.rolling(50, min_periods=40).mean(), close.rolling(200, min_periods=150).mean()
hi = logc >= logc.rolling(250, min_periods=200).max()
lo = logc <= logc.rolling(250, min_periods=200).min()
cnt = listed.sum(1)
B = pd.DataFrame({
    "above50": (close > ma50).where(listed & ma50.notna()).mean(1),
    "above200": (close > ma200).where(listed & ma200.notna()).mean(1),
    "newHighs": hi.where(listed & logc.rolling(250, min_periods=200).max().notna()).rolling(5).max().mean(1),
    "newLows": lo.where(listed & logc.rolling(250, min_periods=200).min().notna()).rolling(5).max().mean(1),
    "rising": (ret > 0).where(listed).mean(1).rolling(10).mean(),
})
B["hiMinusLo"] = B["newHighs"] - B["newLows"]
B.loc[B.index < pd.Timestamp("2016-01-01")] = np.nan       # rolling 1-year measures need a year of history first
NAMES = {"above50": "Stocks above their 50-day average", "above200": "Stocks above their 200-day average",
         "newHighs": "Stocks at a 52-week high (this week)", "newLows": "Stocks at a 52-week low (this week)",
         "rising": "Stocks rising on an average day (last 10 days)", "hiMinusLo": "New highs minus new lows"}
fN = {h: n500.shift(-n) - n500 for h, n in FWD.items()}
fSB = {h: (sml.shift(-n) - sml) - (n50.shift(-n) - n50) for h, n in FWD.items()}
breadth = {"hist": {"d": [str(w.date()) for w in W], **{k: B[k].reindex(W).round(4).tolist() for k in B}, "n500": np.exp(n500.reindex(W)).round(1).tolist()},
           "now": {}, "levels": {}, "names": NAMES}
for k in B:
    s = B[k]
    cur = s.loc[T]
    pct_rank = (s.dropna() < cur).mean()
    q = s.loc[Wp].rank(pct=True)
    rows = []
    for lo_, hi_, lab in [(0, .2, "Lowest fifth"), (.2, .4, "Low"), (.4, .6, "Middle"), (.6, .8, "High"), (.8, 1.01, "Highest fifth")]:
        m = (q > lo_) & (q <= hi_)
        lv = s.loc[Wp][m]
        rows.append({"level": lab, "range": [lv.min(), lv.max()], **{f"n500_{h}": stat(fN[h].loc[Wp][m]) for h in FWD},
                     "smallBeat3m": (fSB["3m"].loc[Wp][m].dropna() > 0).mean(), "weeks": int(m.sum()),
                     "isNow": bool(lv.min() <= cur <= lv.max()) if len(lv) else False})
    breadth["now"][k] = {"value": cur, "pctRank": pct_rank, "chg4w": cur - s.loc[W[-5]], "chg13w": cur - s.loc[W[-14]]}
    breadth["levels"][k] = rows
# divergence: index within 3% of its high while breadth (above200) fell by 10+ pts in 3 months
nh500 = n500 - n500.rolling(250, min_periods=60).max()
div = (nh500 > -0.03) & (B["above200"] - B["above200"].shift(63) < -0.10)
dv = div.loc[Wp]
breadth["divergence"] = {"now": bool(div.loc[T]), "weeks": int(dv.sum()), **{f"n500_{h}": stat(fN[h].loc[Wp][dv]) for h in FWD},
                         "normal": {h: stat(fN[h].loc[Wp]) for h in FWD}}
# breadth inside each group (members today; history of members = survivor view, so only now and 4 weeks ago)
gb = []
a200 = (close > ma200).where(listed)
a50 = (close > ma50).where(listed)
for k, syms in members.items():
    syms = [s for s in syms if s in UNI or k.startswith("industry")]
    if len(syms) < 5 or k not in L.columns:
        continue
    gb.append({"g": k, "cat": index_category(k), "n": len(syms), "above200": a200[syms].loc[T].mean(), "above200_4w": a200[syms].loc[W[-5]].mean(),
               "above50": a50[syms].loc[T].mean(), "above50_4w": a50[syms].loc[W[-5]].mean(),
               "newHigh": hi[syms].iloc[-5:].max().mean(), "newLow": lo[syms].iloc[-5:].max().mean(), "phase": PH[k].loc[T]})
breadth["groups"] = gb

# ============================================================ 3. BREAKOUTS
print("breakouts", flush=True)
def bases(Lv, RSv):
    d = Lv.diff()
    tight = d.rolling(21, min_periods=15).std() / d.rolling(250, min_periods=150).std()
    nh = Lv - Lv.rolling(250, min_periods=150).max()
    rsnh = RSv - RSv.rolling(250, min_periods=150).max()
    base = (nh > -0.08) & (rsnh > -0.08) & (tight < 0.75)
    brk = (Lv >= Lv.rolling(250, min_periods=150).max()) & base.shift(1).rolling(21, min_periods=1).max().fillna(0).astype(bool)
    return tight, nh, base, brk
# groups
tg, nhg, bg, kg = bases(L, RS)
fRS = {h: RS.shift(-n) - RS for h, n in FWD.items()}
first_g = kg & ~kg.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
gstats = {h: stat(fRS[h][first_g].stack()) for h in FWD}
gnorm = {h: stat(fRS[h].loc[Wp].stack()) for h in FWD}
recent_g = []
for k in L.columns:
    ev = first_g[k][first_g[k]].index
    ev = ev[ev >= idx[-30]]
    if len(ev):
        d0 = ev[-1]
        recent_g.append({"g": k, "cat": index_category(k), "date": d0, "days": int((idx > d0).sum()),
                         "since": L[k].loc[T] - L[k].loc[d0], "vsMktSince": RS[k].loc[T] - RS[k].loc[d0],
                         "beat3m": detail_g.get(k, {}).get("prob", {}).get("3m|beat")})
base_g = [{"g": k, "cat": index_category(k), "fromHigh": nhg[k].loc[T], "tight": tg[k].loc[T], "phase": PH[k].loc[T],
           "weeksInBase": int(bg[k].iloc[-60:][::-1].cumprod().sum() // 5), "beat3m": detail_g.get(k, {}).get("prob", {}).get("3m|beat")}
          for k in L.columns if bool(bg[k].loc[T])]
# stocks
RSs = X                                                   # stock vs equal-weight market
ts_, nhs, bs, ks = bases(logc, RSs)
vr = np.log(p["volume"].rolling(5, min_periods=3).mean() / p["volume"].rolling(60, min_periods=20).mean())
dlv20 = p["dlv"].rolling(20, min_periods=10).mean()
first_s = ks & ~ks.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
fX = {h: X.shift(-n) - X for h, n in FWD.items()}
ev = first_s.loc[:T - pd.Timedelta(days=200)]
sstat = {h: stat(fX[h].loc[ev.index][ev].stack()) for h in FWD}
volc = (vr > 0.3)
sstat_vol = {h: stat(fX[h].loc[ev.index][ev & volc.loc[ev.index]].stack()) for h in FWD}
sstat_novol = {h: stat(fX[h].loc[ev.index][ev & ~volc.loc[ev.index]].stack()) for h in FWD}
plain_hi = (logc >= logc.rolling(250, min_periods=150).max()) & ~bs.shift(1).rolling(21, min_periods=1).max().fillna(0).astype(bool)
ph_ev = plain_hi & ~plain_hi.shift(1).rolling(20, min_periods=1).max().fillna(0).astype(bool)
ph_ev = ph_ev.loc[:T - pd.Timedelta(days=200)]
sstat_plain = {h: stat(fX[h].loc[ph_ev.index][ph_ev].stack()) for h in FWD}
snorm = {h: stat(fX[h].loc[Wp].stack()) for h in FWD}
# industry phase lookup for stocks
ind_phase = {s: PH["industry: " + ind[s]].loc[T] for s in cols if s in ind.index and ("industry: " + ind[s]) in PH.columns}
liq = (p["close_raw"] * p["volume_raw"]).rolling(60, min_periods=20).mean().loc[T] / 1e7
recent_s, base_s = [], []
uni = [s for s in cols if s in UNI]
for s in uni:
    e = first_s[s].iloc[-15:]
    if e.any():
        d0 = e[e].index[-1]
        recent_s.append({"s": s, "ind": ind.get(s), "date": d0, "days": int((idx > d0).sum()), "since": logc[s].loc[T] - logc[s].loc[d0],
                         "vol": vr[s].loc[d0], "dlv": dlv20[s].loc[T], "indPhase": ind_phase.get(s), "tvCr": liq.get(s),
                         "beat3m": detail_s.get(s, {}).get("prob", {}).get("3m|beat"), "bigDown3m": detail_s.get(s, {}).get("prob", {}).get("3m|big_down")})
    elif bool(bs[s].loc[T]):
        base_s.append({"s": s, "ind": ind.get(s), "fromHigh": nhs[s].loc[T], "tight": ts_[s].loc[T], "dlv": dlv20[s].loc[T],
                       "weeksInBase": int(bs[s].iloc[-60:][::-1].cumprod().sum() // 5), "indPhase": ind_phase.get(s), "tvCr": liq.get(s),
                       "beat3m": detail_s.get(s, {}).get("prob", {}).get("3m|beat"), "bigDown3m": detail_s.get(s, {}).get("prob", {}).get("3m|big_down")})
breakouts = {"groups": {"recent": recent_g, "base": base_g, "stats": gstats, "normal": gnorm},
             "stocks": {"recent": sorted(recent_s, key=lambda x: x["days"]), "base": sorted(base_s, key=lambda x: x["fromHigh"], reverse=True),
                        "stats": sstat, "statsVol": sstat_vol, "statsNoVol": sstat_novol, "statsPlainHigh": sstat_plain, "normal": snorm}}

json.dump(clean({"date": T, "rotation": {"groups": rot, "warnAccuracy": warn_acc}, "breadth": breadth, "breakouts": breakouts}),
          open(os.path.join(OUT, "watch.json"), "w"), separators=(",", ":"))
print("DONE watch", os.path.getsize(os.path.join(OUT, "watch.json")) // 1024, "KB", flush=True)
