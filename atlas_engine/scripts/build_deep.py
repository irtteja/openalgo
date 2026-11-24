"""Deep data for the explorer: per-sector and per-stock histories, events, analogs, pattern examples, forecasts.

    python3 scripts/build_deep.py     (after pattern_loop.py live and build_explorer.py)

Writes explorer/data/deep/{groups.json, stocks_NN.json, patterns_stocks.json, patterns_groups.json, forecast.json}.
Everything about the past uses only data up to each date; "what followed" uses the dates after.
"""
import glob, json, os, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel, DATA_DIR
from rsi.extra import add_extra_features
from rsi.fundamentals import add_fundamental_features
from rsi.market_features import add_market_features
from rsi.corp_features import add_corp_features, _read, _next_day, SENIOR
from rsi.step3_features import add_step3_features, _index, S3, index_universe, is_sme, skip_index, MIN_DAYS, index_name
from rsi.patterns import build_stock_table, build_group_table, HORIZONS, BIG

OUT = "explorer/data/deep"
os.makedirs(OUT, exist_ok=True)
NSH = 48
UNI = index_universe()                       # page shows only stocks in NSE indices (no SME index)                                       # stock shards


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


def dump(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(clean(obj), f, separators=(",", ":"))


def shard(sym):
    return zlib.crc32(sym.encode()) % NSH


def dist(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 20:
        return None
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"n": len(v), "p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4], "pos": (v > 0).mean(), "mean": v.mean()}


print("loading", flush=True)
p = load_market_panel()
for f in (add_extra_features, add_fundamental_features, add_market_features, add_corp_features, add_step3_features):
    p = f(p)
close, ret = p["close"], p["ret"]
idx, cols = close.index, close.columns
listed = close.notna()
logc = np.log(close)
mkt = ret.where(listed).mean(1)
X = ret.sub(mkt, axis=0).fillna(0).cumsum().where(listed)          # cumulative excess vs equal-weight market
n500, n50 = _index("NIFTY 500", idx), _index("NIFTY 50", idx)
sml = _index("NIFTY SMALLCAP 250", idx)
W = idx[::-1][::5][::-1]                                            # weekly, anchored on the last day
T = idx[-1]
vix = pd.read_csv(os.path.join(DATA_DIR, "extra", "vix.csv"), parse_dates=["Date"]).set_index("Date")["vix"].reindex(idx).ffill()
vpct = vix.rolling(750, min_periods=120).rank(pct=True)
br200 = (close > close.rolling(200, min_periods=150).mean()).where(listed).astype(float).mean(1)
REG = {
    "VIX high (fear)": vpct > 0.67, "VIX low (calm)": vpct < 0.33,
    "Market below its 200-day avg": n500 < n500.rolling(200).mean(), "Market above its 200-day avg": n500 > n500.rolling(200).mean(),
    "Market 10%+ below its high": (n500 - n500.rolling(250, min_periods=60).max()) < -0.10,
    "Small caps beating large (3m)": (sml.diff(63) - n50.diff(63)) > 0.03, "Large caps beating small (3m)": (sml.diff(63) - n50.diff(63)) < -0.03,
    "Few stocks above 200-day avg (<35%)": br200 < 0.35, "Most stocks above 200-day avg (>60%)": br200 > 0.6,
}
if "fii_fut" in p:
    f = p["fii_fut"].iloc[:, 0]
    REG["FIIs net long index futures"] = f > 0.1
    REG["FIIs net short index futures"] = f < -0.3
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FWD = {"1m": 21, "3m": 63, "6m": 126}

# ===================================================================== GROUPS
print("groups", flush=True)
G, members = {}, {}
for fpath in glob.glob(os.path.join(S3, "indices", "*.csv")):
    n = index_name(fpath)
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
groups_members_shown = {k: [s for s in v if s in UNI] for k, v in members.items()}
L = pd.DataFrame(G)
RS = L.sub(n500, axis=0)
ratio = RS - RS.rolling(126, min_periods=100).mean()
mom = ratio.diff(21)
PH = pd.DataFrame(np.select([(ratio > 0) & (mom > 0), (ratio > 0) & (mom <= 0), (ratio <= 0) & (mom <= 0), (ratio <= 0) & (mom > 0)],
                            ["Leading", "Weakening", "Lagging", "Improving"], ""), index=idx, columns=L.columns).where(ratio.notna() & mom.notna(), "")
d = L.diff()
tight = d.rolling(21, min_periods=15).std() / d.rolling(250, min_periods=150).std()
nh = L - L.rolling(250, min_periods=150).max()
rsnh = RS - RS.rolling(250, min_periods=150).max()
base = (nh > -0.08) & (rsnh > -0.08) & (tight < 0.75)
brk = (L >= L.rolling(250, min_periods=150).max()) & base.shift(1).rolling(21, min_periods=1).max().astype(bool)
above50 = (close > close.rolling(50, min_periods=40).mean()).where(listed).astype(float)
B = pd.DataFrame({k: above50[v].mean(1) for k, v in members.items() if len(v) >= 5}).reindex(columns=L.columns)
fRS = {h: RS.shift(-n) - RS for h, n in FWD.items()}
fL = {h: L.shift(-n) - L for h, n in FWD.items()}

# state buckets for analogs (all groups pooled)
rq3 = RS.diff(63).rank(axis=1, pct=True)
rq6 = RS.diff(126).rank(axis=1, pct=True)
hb = pd.DataFrame(np.select([nh > -0.03, nh > -0.12], ["at/near high", "a bit below high"], "well below high"), index=idx, columns=L.columns)
bucket = lambda x: pd.DataFrame(np.select([x > 0.8, x > 0.4], ["top", "middle"], "bottom"), index=x.index, columns=x.columns).where(x.notna(), "")
Wp = W[W <= T - pd.Timedelta(days=200)]                          # past weeks with all outcomes known
state = pd.DataFrame({"phase": PH.loc[Wp].stack(), "r3": bucket(rq3).loc[Wp].stack(), "r6": bucket(rq6).loc[Wp].stack(),
                      "hb": hb.loc[Wp].stack(), **{f"f{h}": fRS[h].loc[Wp].stack() for h in FWD}})
state = state[(state["phase"] != "") & (state["r3"] != "")]

# phase transitions (4, 13, 26 weeks ahead), conditioned on phase now and momentum strength
mq = mom.rank(axis=1, pct=True)
trans_rows = []
for k, steps in [("1m", 4), ("3m", 13), ("6m", 26)]:
    a = PH.loc[W]; b = PH.loc[W].shift(-steps); m = bucket(mq).loc[W]
    s = pd.DataFrame({"now": a.stack(), "mom": m.stack(), "then": b.stack()})
    s = s[(s["now"] != "") & (s["then"].notna()) & (s["then"] != "")]
    s["h"] = k
    trans_rows.append(s)
TR = pd.concat(trans_rows)
TRp = TR.groupby(["h", "now", "mom"])["then"].value_counts(normalize=True).unstack(fill_value=0)

corrW = L.loc[W].diff()
cw = corrW.corr()
n50w = n50.loc[W].diff()
groups_out = {}
for k in L.columns:
    s = L[k]; r = RS[k]
    first = s.first_valid_index()
    wk = W[W >= first]
    ph = PH[k].loc[wk]
    seg, cur = [], None
    for dte, v in ph.items():
        if v == "":
            continue
        if cur and cur["phase"] == v:
            cur["end"] = dte
        else:
            if cur:
                seg.append(cur)
            cur = {"phase": v, "start": dte, "end": dte}
    if cur:
        seg.append(cur)
    dur = pd.DataFrame([{"phase": x["phase"], "weeks": (x["end"] - x["start"]).days / 7 + 1} for x in seg])
    durs = dur.groupby("phase")["weeks"].agg(["mean", "count"]).to_dict("index") if len(dur) else {}
    bdates = brk[k][brk[k] & ~brk[k].shift(1, fill_value=False).rolling(20, min_periods=1).max().astype(bool)].index
    breaks = [{"date": b_, **{h: fRS[h][k].get(b_) for h in FWD}} for b_ in bdates]
    me = s.resample("ME").last(); rme = r.resample("ME").last()
    mr = rme.diff().dropna(); ma = me.diff().dropna()
    season = [{"m": MONTHS[i], "vsMkt": mr[mr.index.month == i + 1].mean(), "beatPct": (mr[mr.index.month == i + 1] > 0).mean(),
               "abs": ma[ma.index.month == i + 1].mean(), "n": int((mr.index.month == i + 1).sum())} for i in range(12)]
    ye = s.resample("YE").last(); rye = r.resample("YE").last()
    years = [{"y": int(dte.year), "ret": v, "vsMkt": rye.diff().get(dte)} for dte, v in ye.diff().dropna().items()]
    if s.index[-1].year == T.year:
        years.append({"y": T.year, "ret": s.loc[T] - ye.iloc[-2] if len(ye) > 1 else np.nan, "vsMkt": r.loc[T] - rye.iloc[-2] if len(rye) > 1 else np.nan, "ytd": True})
    regimes = []
    for nm, msk in REG.items():
        mm = msk.reindex(Wp).fillna(False).astype(bool).values
        a1, a3 = fRS["1m"][k].loc[Wp][mm], fRS["3m"][k].loc[Wp][mm]
        if a1.notna().sum() >= 8:
            regimes.append({"cond": nm, "next1m": a1.mean(), "next3m": a3.mean(), "beat3m": (a3.dropna() > 0).mean(), "weeks": int(a1.notna().sum())})
    gw = corrW[k]
    beta = gw.cov(n50w) / n50w.var()
    cc = cw[k].drop(k).dropna().sort_values()
    leads = {o: gw.corr(corrW[o].shift(-1)) for o in L.columns if o != k}
    follows = {o: corrW[o].corr(gw.shift(-1)) for o in L.columns if o != k}
    lead = sorted(leads.items(), key=lambda x: -abs(x[1] if np.isfinite(x[1]) else 0))[:4]
    foll = sorted(follows.items(), key=lambda x: -abs(x[1] if np.isfinite(x[1]) else 0))[:4]
    # analogs: past weeks (any group) in the same state as this group today
    now = {"phase": PH[k].loc[T], "r3": bucket(rq3).loc[T, k], "r6": bucket(rq6).loc[T, k], "hb": hb.loc[T, k]}
    an = state
    for c_ in ["phase", "r3", "hb"]:
        an = an[an[c_] == now[c_]]
    own = an[an.index.get_level_values(1) == k]
    analog = {"state": now, "all": {h: dist(an[f"f{h}"]) for h in FWD}, "own": {h: dist(own[f"f{h}"]) for h in FWD},
              "ownExamples": [{"date": i[0], **{h: row[f"f{h}"] for h in FWD}} for i, row in own.iloc[::max(1, len(own) // 12)].tail(12).iterrows()]}
    mn = "strong" if mq.loc[T, k] > 0.8 else "middle" if mq.loc[T, k] > 0.4 else "weak"
    pt = {}
    for h in ["1m", "3m", "6m"]:
        key = (h, now["phase"], {"strong": "top", "middle": "middle", "weak": "bottom"}[mn])
        pt[h] = TRp.loc[key].to_dict() if key in TRp.index else None
    groups_out[k] = {
        "hist": {"d": [str(x.date()) for x in wk], "lvl": np.exp(s.loc[wk] - s.loc[wk].dropna().iloc[0]).round(4).tolist(),
                 "rs": np.exp(r.loc[wk] - r.loc[wk].dropna().iloc[0]).round(4).tolist(), "ph": ph.tolist(),
                 "breadth": (B[k].loc[wk].round(3).tolist() if k in B else [])},
        "segments": seg[-40:], "phaseDur": durs, "breakouts": breaks[-25:], "nBreakouts": len(breaks),
        "breakoutAvg": {h: np.nanmean([b_[h] for b_ in breaks if b_[h] is not None]) if breaks else None for h in FWD},
        "season": season, "years": years, "regimes": regimes, "beta": beta, "corrN50": gw.corr(n50w),
        "together": [[o, v] for o, v in cc.tail(5)[::-1].items()], "apart": [[o, v] for o, v in cc.head(5).items()],
        "leads": lead, "follows": foll, "analog": analog, "phaseNext": pt, "momStrength": mn,
        "maxDD": float((s - s.cummax()).min()),
    }
dump(groups_out, "groups.json")
print("groups done", len(groups_out), flush=True)

# ===================================================================== FORECAST
print("forecast", flush=True)
cur = {nm: bool(m.get(T, False)) for nm, m in REG.items()}
active = [nm for nm, v in cur.items() if v]
fN = {h: n500.shift(-n) - n500 for h, n in FWD.items()}
fSB = {h: (sml.shift(-n) - sml) - (n50.shift(-n) - n50) for h, n in FWD.items()}
scen = []
for combo in [active] + [[a] for a in active]:
    m = pd.Series(True, index=Wp)
    for c_ in combo:
        m &= REG[c_].reindex(Wp).fillna(False).astype(bool)
    rows = {h: dist(fN[h].loc[Wp][m]) for h in FWD}
    sb = {h: (fSB[h].loc[Wp][m].dropna() > 0).mean() if m.sum() else None for h in FWD}
    scen.append({"conds": combo, "weeks": int(m.sum()), "n500": rows, "smallBeatsLarge": sb,
                 "periods": sorted({str(x.year) for x in Wp[m.values]})})
allrows = {h: dist(fN[h].loc[Wp]) for h in FWD}
ds = json.load(open("results/patterns/detail_groups.json"))
lead_fc = []
for k in L.columns:
    g = groups_out[k]
    pn = g["phaseNext"]
    lead_fc.append({"g": k, "phase": PH[k].loc[T], "mom": g["momStrength"],
                    "pLead1m": (pn["1m"] or {}).get("Leading"), "pLead3m": (pn["3m"] or {}).get("Leading"), "pLead6m": (pn["6m"] or {}).get("Leading"),
                    "pLag3m": (pn["3m"] or {}).get("Lagging"),
                    "beat3m": ds["rows"].get(k, {}).get("prob", {}).get("3m|beat"),
                    "analog3m": (g["analog"]["all"]["3m"] or {}).get("p50"), "analogPos3m": (g["analog"]["all"]["3m"] or {}).get("pos"),
                    "breadth": B[k].loc[T] if k in B else None, "breadthChg": (B[k].loc[T] - B[k].shift(21).loc[T]) if k in B else None,
                    "base": bool(base[k].loc[T]), "tight": tight[k].loc[T], "fromHigh": nh[k].loc[T], "vs3m": RS[k].diff(63).loc[T]})
building = [x for x in lead_fc if x["phase"] in ("Improving", "Lagging") and (x["breadthChg"] or 0) > 0.05]
dump({"date": T, "active": active, "scenarios": scen, "allWeeks": allrows, "leaders": lead_fc,
      "building": sorted(building, key=lambda x: -(x["pLead3m"] or 0)),
      "transitions": {f"{h}|{a}|{b_}": TRp.loc[(h, a, b_)].to_dict() for (h, a, b_) in TRp.index}}, "forecast.json")
print("forecast done", flush=True)

# ===================================================================== STOCKS
print("stocks", flush=True)
ins = _read("insider").drop_duplicates()
ins["day"] = _next_day(pd.to_datetime(ins["date"], format="%d-%b-%Y %H:%M", errors="coerce"), idx)
cat = ins["personCategory"].astype(str).str.strip().str.lower()
mode = ins["acqMode"].astype(str).str.strip().str.lower()
ins["val"] = pd.to_numeric(ins["secVal"], errors="coerce").fillna(0)
ins["kind"] = np.select([cat.isin(SENIOR) & (mode == "market purchase"), cat.isin(SENIOR) & (mode == "market sale"),
                         cat.isin({"promoters", "promoter group"}) & (mode == "pledge creation"),
                         cat.isin({"promoters", "promoter group"}) & (mode == "invocation of pledge"),
                         cat.isin({"promoters", "promoter group"}) & (mode == "revokation of pledge")],
                        ["Insider buy", "Insider sell", "Pledge created", "Lender sold pledged shares", "Pledge released"], "")
ins = ins[(ins["kind"] != "") & ins["day"].notna()]
ev_ins = ins.groupby(["symbol", "day", "kind"])["val"].sum().reset_index()
ev_ins = ev_ins[~((ev_ins["kind"].str.startswith("Insider")) & (ev_ins["val"] < 1e6))]

ma200 = close.rolling(200, min_periods=150).mean()
hi = logc >= logc.rolling(250, min_periods=200).max()
lo = logc <= logc.rolling(250, min_periods=200).min()
first_hi = hi & ~hi.shift(1).rolling(60, min_periods=1).max().fillna(0).astype(bool)
first_lo = lo & ~lo.shift(1).rolling(60, min_periods=1).max().fillna(0).astype(bool)
vspike = p["volume"] > 5 * p["volume"].rolling(60, min_periods=20).mean().shift(1)
res = (p["fresh"] < p["fresh"].shift(1)) if "fresh" in p else None
fX = {h: X.shift(-n) - X for h, n in FWD.items()}
fr_raw = {h: np.log1p(ret.fillna(0)).cumsum().shift(-n) - np.log1p(ret.fillna(0)).cumsum() for h, n in FWD.items()}

# analog cells: 3 buckets on 6 traits, all stocks pooled, past weeks only
traits = {"r3m": logc.diff(63), "r1y": logc.shift(21) - logc.shift(250), "vol": ret.rolling(60, min_periods=20).std(),
          "hi": logc - logc.rolling(250, min_periods=120).max(), "dlv": p["dlv"].rolling(20, min_periods=10).mean(),
          "size": (p["close_raw"] * p["volume_raw"]).rolling(60, min_periods=20).mean()}
TB = {k: np.ceil(v.rank(axis=1, pct=True) * 3).fillna(0).astype(int) for k, v in traits.items()}
code = sum(TB[k] * (4 ** i) for i, k in enumerate(traits))
Wp2 = W[W <= T - pd.Timedelta(days=200)]
cell = pd.DataFrame({"c": code.loc[Wp2].stack(), **{h: fX[h].loc[Wp2].stack() for h in FWD}})
cell = cell[cell["c"] > 0]
CELL = {}
for c_, g in cell.groupby("c"):
    CELL[c_] = {h: dist(g[h]) for h in FWD}
TRN = {"r3m": "last 3 months", "r1y": "last year", "vol": "volatility", "hi": "distance from 52w high", "dlv": "delivery %", "size": "size"}
LV = {1: "low", 2: "middle", 3: "high"}

grp_of = p["_closest_index"].iloc[-1] if "_closest_index" in p else pd.Series(dtype=object)
ds_s = json.load(open("results/patterns/detail_stocks.json"))
shards = {i: {} for i in range(NSH)}
evs_by = {}
for r_ in ev_ins.itertuples():
    evs_by.setdefault(r_.symbol, []).append((r_.day, r_.kind, r_.val))
WX = X.loc[W]
Wc = close.loc[W]
for n_, k in enumerate(ds_s["rows"]):
    if k not in cols or k not in UNI:
        continue
    lst = listed[k]
    first = lst.idxmax()
    wk = W[W >= first]
    c = Wc[k].loc[wk]
    if c.notna().sum() < 5:
        continue
    gname = ind.get(k)
    gk = ("industry: " + gname) if gname is not None and ("industry: " + gname) in L.columns else None
    if gk is None and isinstance(grp_of.get(k), str) and ("index: " + grp_of.get(k)) in L.columns:
        gk = "index: " + grp_of.get(k)
    rsg = (logc[k] - L[gk]).loc[wk] if gk else None
    hist = {"d": [str(x.date()) for x in wk], "px": (c / c.dropna().iloc[0]).round(4).tolist(),
            "rsMkt": np.exp(WX[k].loc[wk] - WX[k].loc[wk].dropna().iloc[0]).round(4).tolist(),
            "rsGrp": (np.exp(rsg - rsg.dropna().iloc[0]).round(4).tolist() if rsg is not None and rsg.notna().any() else None), "grp": gk}
    ev = []
    for dte, kind, val in evs_by.get(k, []):
        ev.append((dte, kind, val))
    for kind, m in [("Jumped 10%+ in a day", ret[k] > 0.10), ("Fell 10%+ in a day", ret[k] < -0.10), ("New 52-week high", first_hi[k]),
                    ("New 52-week low", first_lo[k]), ("Volume 5x usual", vspike[k])] + ([("Results announced", res[k])] if res is not None else []):
        for dte in m[m.fillna(False).astype(bool)].index:
            ev.append((dte, kind, None))
    if "bulk" in p:
        b_ = p["bulk"][k]
        for dte in b_[b_ > 0.05].index:
            ev.append((dte, "Bulk deal: net buy", None))
        for dte in b_[b_ < -0.05].index:
            ev.append((dte, "Bulk deal: net sell", None))
    ev.sort(key=lambda x: x[0])
    events, summ = [], {}
    for dte, kind, val in ev:
        a = {h: fX[h][k].get(dte) for h in FWD}
        extra = None
        if kind == "Results announced" and "prof_g" in p:
            extra = p["prof_g"][k].shift(-3).get(dte)
        events.append({"d": dte, "k": kind, "v": val, "pg": extra, **a})
        sm = summ.setdefault(kind, {h: [] for h in FWD})
        for h in FWD:
            if a[h] is not None and np.isfinite(a[h]):
                sm[h].append(a[h])
    summ = {kind: {"n": max(len(v) for v in hh.values()), **{h: (np.mean(hh[h]) if hh[h] else None) for h in FWD},
                   "beat3m": (np.mean(np.array(hh["3m"]) > 0) if hh["3m"] else None)} for kind, hh in summ.items()}
    # behaviour
    xr = ret[k].dropna()
    wr = Wc[k].loc[wk].pipe(np.log).diff()
    beta = wr.cov(n500.loc[wk].diff()) / n500.loc[wk].diff().var()
    gcorr = wr.corr(L[gk].loc[wk].diff()) if gk else None
    lv = logc[k].dropna()
    dd = float((lv - lv.cummax()).min())
    me = X[k].resample("ME").last().diff().dropna()
    season = [{"m": MONTHS[i], "vsMkt": me[me.index.month == i + 1].mean(), "beatPct": (me[me.index.month == i + 1] > 0).mean(),
               "n": int((me.index.month == i + 1).sum())} for i in range(12)]
    ye = logc[k].resample("YE").last(); xe = X[k].resample("YE").last()
    years = [{"y": int(dte.year), "ret": v, "vsMkt": xe.diff().get(dte)} for dte, v in ye.diff().dropna().items()]
    # analogs
    cc_ = int(code[k].loc[T]) if np.isfinite(code[k].loc[T]) else 0
    an = CELL.get(cc_)
    desc = [f"{TRN[t]}: {LV.get(int(TB[t][k].loc[T]), '?')}" for t in traits if TB[t][k].loc[T] > 0]
    own_m = cell.xs(k, level=1) if k in cell.index.get_level_values(1) else None
    own = own_m[own_m["c"] == cc_] if own_m is not None else None
    shards[shard(k)][k] = {
        "hist": hist, "events": events[-150:], "nEvents": len(events), "eventSummary": summ,
        "beta": beta, "grpCorr": gcorr, "maxDD": dd, "vol": xr.std() * np.sqrt(250), "season": season, "years": years,
        "analog": {"desc": desc, "all": an, "own": {h: dist(own[h]) for h in FWD} if own is not None and len(own) else None,
                   "ownWhen": [str(x.date()) for x in own.index[-8:]] if own is not None else []},
        "listed": str(first.date()),
    }
    if n_ % 300 == 0:
        print("stock", n_, flush=True)
for i, v in shards.items():
    dump(v, f"stocks_{i:02d}.json")
print("stocks done", flush=True)

# ===================================================================== PATTERNS: year-by-year and examples
print("patterns", flush=True)
for name, build in [("stocks", build_stock_table), ("groups", build_group_table)]:
    t = build(p)
    det = json.load(open(f"results/patterns/detail_{name}.json"))
    yrs = np.arange(t.years.min(), t.years.max() + 1)
    now_rows = np.flatnonzero(t.dates == t.dates.max())
    out = []
    base_cache = {}
    for s in det["patterns"]:
        h, o = s["h"], s["o"]
        if (h, o) not in base_cache:
            y, ok = t.outcome(h, o)
            cnt = np.bincount(t.years[ok] - yrs[0], minlength=len(yrs))
            base_cache[(h, o)] = (y, ok, np.bincount(t.years[ok] - yrs[0], weights=y[ok], minlength=len(yrs)) / np.maximum(cnt, 1))
        y, ok, by = base_cache[(h, o)]
        m = np.ones(len(t.keys), bool)
        for c_ in s["conds"]:
            m &= t.conds[c_]
        mk = m & ok
        cnt = np.bincount(t.years[mk] - yrs[0], minlength=len(yrs))
        rate = np.bincount(t.years[mk] - yrs[0], weights=y[mk], minlength=len(yrs)) / np.maximum(cnt, 1)
        yearly = [{"y": int(yy), "n": int(cnt[i]), "rate": rate[i] if cnt[i] else None, "base": by[i]} for i, yy in enumerate(yrs) if cnt[i]]
        rows = np.flatnonzero(mk)
        ex = []
        if len(rows):
            # spread examples over time: take the last occurrence per key per year, then sample 16
            df = pd.DataFrame({"r": rows, "k": t.keys[rows], "y": t.years[rows]}).drop_duplicates(["k", "y"], keep="last")
            if name == "stocks" and df["k"].isin(UNI).sum() >= 8:
                df = df[df["k"].isin(UNI)]
            if name == "groups":
                df = df[~df["k"].map(is_sme)]
            pick = df.sample(min(16, len(df)), random_state=1).sort_values("r")
            fx = t.fwd[h] if o != "up" else t.fwd_raw[h]
            ex = [{"k": str(t.keys[r_]), "d": str(t.dates[r_].date()), "out": fx[r_], "hit": bool(y[r_])} for r_ in pick["r"]]
        nowm = [str(t.keys[r_]) for r_ in now_rows if m[r_] and (str(t.keys[r_]) in UNI if name == "stocks" else not is_sme(str(t.keys[r_])))]
        out.append({"id": s["id"], "yearly": yearly, "examples": ex, "now": nowm[:200], "nNow": len(nowm)})
    dump(out, f"patterns_{name}.json")
    print("patterns", name, len(out), flush=True)
print("DONE deep", flush=True)
