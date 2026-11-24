"""Market structure for the Atlas: regime, hidden groups and driving forces, move breakdown and breadth of moves.

    python3 scripts/build_structure.py      -> explorer/data/deep/structure.json

1. Market regime (unsupervised learning): a Gaussian mixture model groups every week since 2016 by Nifty 500 trend,
   volatility, breadth, new highs/lows and India VIX into 4 states, named by their averages (calm uptrend,
   steady/mixed, correction, stress). Shows today's state, its history, and what followed each state.
2. Hidden groups and driving forces (unsupervised): over the last year of daily returns, principal components find
   the few forces that move the market; stocks are clustered by their moves after removing the market, so groups
   follow how stocks actually trade, not NSE's labels.
3. Is the move real and broad: each stock's 1- and 3-month move split into market, industry and own parts; each
   sector/theme index's rise checked for how many members joined in and how much came from its 3 biggest movers.
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from rsi.data import load_market_panel, DATA_DIR
from rsi.step3_features import _index, S3, index_universe, _SECTOR, _THEME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "explorer", "data", "deep", "structure.json")
r4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)

p = load_market_panel()
close = p["close"]; idx = close.index
UNI = sorted(index_universe() & set(close.columns))
c = close[UNI]
logc = np.log(c)
n500 = _index("NIFTY 500", idx)
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
out = {"date": str(idx[-1].date())}

# ------------------------------------------------------------------ 1. regime
print("regime", flush=True)
above200 = (c > c.rolling(200, min_periods=150).mean()).where(c.notna()).mean(axis=1)
above50 = (c > c.rolling(50, min_periods=40).mean()).where(c.notna()).mean(axis=1)
hi = (c >= c.rolling(250, min_periods=200).max()).where(c.notna()).mean(axis=1)
lo = (c <= c.rolling(250, min_periods=200).min()).where(c.notna()).mean(axis=1)
vix = pd.read_csv(os.path.join(DATA_DIR, "extra", "vix.csv"), parse_dates=["Date"]).set_index("Date")["vix"].reindex(idx).ffill()
mret = n500.diff()
F = pd.DataFrame({"trend_3m": n500 - n500.shift(63), "trend_1m": n500 - n500.shift(21), "vol_1m": mret.rolling(21).std() * np.sqrt(250),
                  "above200": above200, "above50": above50, "hi_minus_lo": (hi - lo).rolling(10).mean(), "vix": vix})
wk = idx[(idx.to_series().shift(-1).dt.weekday < idx.to_series().dt.weekday) | (idx == idx[-1])]
W = F.reindex(wk)
W = W[W.index >= "2016-03-01"].dropna()
Z = (W - W.mean()) / W.std()
gm = GaussianMixture(4, covariance_type="full", n_init=10, random_state=0).fit(Z.values)
lab = pd.Series(gm.predict(Z.values), index=W.index)
prob_now = gm.predict_proba(Z.values[-1:])[0]
# smooth single-week flips: a state must hold 2 weeks unless it is today's
sm = lab.copy()
for i in range(1, len(sm) - 1):
    if sm.iloc[i] != sm.iloc[i - 1] and sm.iloc[i] != sm.iloc[i + 1]:
        sm.iloc[i] = sm.iloc[i - 1]
means = W.groupby(sm).mean()
# name states from their averages: the most volatile = stress; the rest by 3-month trend
order = list(means.index)
stress = means["vol_1m"].idxmax()
rest = sorted([s for s in order if s != stress], key=lambda s: -means.loc[s, "trend_3m"])
NAMES = {rest[0]: "Strong uptrend", rest[1]: "Steady / mixed", rest[2]: "Weak market (correction or early recovery)", stress: "Stress / panic"}
DESC = {"Strong uptrend": "Market rising over 3 months with most stocks above their 200-day average; swings a little above normal.",
        "Steady / mixed": "The usual state: mild trend, calm swings, low VIX, most stocks above their 200-day average.",
        "Weak market (correction or early recovery)": "Fewer than half the stocks above their 200-day average after a fall; includes the first weeks of a rebound before most stocks recover.",
        "Stress / panic": "Very large daily swings and VIX far above normal, as in March 2020."}
fwd4 = (n500.shift(-20) - n500).reindex(sm.index); fwd13 = (n500.shift(-63) - n500).reindex(sm.index)
states = []
for s in order:
    m = sm == s
    runs = (sm != sm.shift()).cumsum()[m]
    dur = runs.value_counts().median() if m.any() else None
    nxt = sm.shift(-4)[m].dropna()
    states.append({"name": NAMES[s], "desc": DESC[NAMES[s]], "share_of_weeks": r4(m.mean()), "typical_weeks": None if dur is None else float(dur),
                   "avg": {k: r4(means.loc[s, k]) for k in means.columns},
                   "nifty500_next_4w": {"mean": r4(fwd4[m].mean()), "pos": r4((fwd4[m] > 0).mean()), "n": int(fwd4[m].notna().sum())},
                   "nifty500_next_13w": {"mean": r4(fwd13[m].mean()), "pos": r4((fwd13[m] > 0).mean()), "n": int(fwd13[m].notna().sum())},
                   "in_4w": {NAMES[k]: r4((nxt == k).mean()) for k in order}})
cur = sm.iloc[-1]
streak = int((sm[::-1] == cur).cumprod().sum())
out["regime"] = {"now": NAMES[cur], "weeks_in_state": streak, "probabilities": {NAMES[k]: r4(prob_now[k]) for k in order},
                 "today": {k: r4(W.iloc[-1][k]) for k in W.columns}, "states": states,
                 "history": {"d": [str(d.date()) for d in sm.index], "s": [NAMES[x] for x in sm], "n500": [r4(np.exp(n500.get(d))) for d in sm.index]},
                 "method": "Gaussian mixture model (4 states) on weekly Nifty 500 1- and 3-month trend, 1-month volatility, % of index stocks above 50/200-day averages, new highs minus new lows and India VIX, 2016 onwards. States are named from their averages; one-week flips are smoothed."}

# ------------------------------------------------------------------ 2. hidden groups and forces
print("groups", flush=True)
R = logc.diff().iloc[-250:]
ok = R.notna().mean() >= 0.95
R = R.loc[:, ok].fillna(0.0)
m = mret.reindex(R.index).fillna(0.0)
beta = R.apply(lambda col: np.cov(col, m)[0, 1] / m.var())
resid = R - np.outer(m.values, beta.values)
Zr = (resid - resid.mean()) / resid.std().replace(0, np.nan)
Zr = Zr.loc[:, Zr.notna().all()]
# forces: PCA on raw standardized returns (component 1 is the market as a whole)
Zf = ((R - R.mean()) / R.std()).loc[:, Zr.columns]
pca = PCA(n_components=6, random_state=0).fit(Zf.values)
L = pd.DataFrame(pca.components_.T, index=Zf.columns)
fac = pd.DataFrame(Zf.values @ pca.components_.T, index=Zf.index)
forces = []
for k in range(6):
    l = L[k]
    if l.mean() < 0:
        l = -l; fac[k] = -fac[k]
    li = ind.reindex(l.index).fillna("Other")
    big = li.value_counts(); big = big[big >= 5].index
    bi = l[li.isin(big)].groupby(li[li.isin(big)]).mean().sort_values()
    top, bot = l.sort_values(ascending=False).index[:8], l.sort_values().index[:8]
    cum = fac[k].cumsum() / np.sqrt(len(Zf.columns))
    forces.append({"k": k + 1, "share": r4(pca.explained_variance_ratio_[k]),
                   "name": "The whole market moving together" if k == 0 else f"{bi.index[-1]} vs {bi.index[0]}",
                   "plus_industries": list(bi.index[::-1][:3]), "minus_industries": list(bi.index[:3]),
                   "plus_stocks": list(top), "minus_stocks": list(bot) if k else [],
                   "move_1m": r4(cum.iloc[-1] - cum.iloc[-22]), "move_3m": r4(cum.iloc[-1] - cum.iloc[-64]),
                   "path": [r4(x) for x in cum.iloc[::5]]})
# hidden groups: k-means on each stock's market-removed moves (in PCA space for stability)
emb = PCA(n_components=15, random_state=0).fit_transform(Zr.T.values)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
K = 24
km = KMeans(K, n_init=20, random_state=0).fit(emb)
lbl = pd.Series(km.labels_, index=Zr.columns)
groups = []
for g in range(K):
    mem = list(lbl[lbl == g].index)
    if len(mem) < 4:
        continue
    cnt = Counter(ind.reindex(mem).fillna("Other"))
    main, n_main = cnt.most_common(1)[0]
    ew = logc[mem].iloc[-260:].diff().mean(axis=1).cumsum()
    rel = ew - (n500.iloc[-260:] - n500.iloc[-260])
    corr = resid[mem].corr().values; within = float((corr.sum() - len(mem)) / (len(mem) * (len(mem) - 1)))
    up = (c[mem].iloc[-1] > c[mem].rolling(50).mean().iloc[-1]).mean()
    groups.append({"name": main if n_main / len(mem) >= 0.6 else f"Mixed: {main} + {cnt.most_common(2)[1][0] if len(cnt) > 1 else 'others'}",
                   "n": len(mem), "label_share": r4(n_main / len(mem)), "industries": [[k, v] for k, v in cnt.most_common(4)],
                   "members": sorted(mem, key=lambda s: -float(c[s].iloc[-1] * p["volume_raw"][s].iloc[-60:].mean())), "togetherness": r4(within),
                   "vs_n500_1m": r4(rel.iloc[-1] - rel.iloc[-22]), "vs_n500_3m": r4(rel.iloc[-1] - rel.iloc[-64]), "above50": r4(up),
                   "path": [r4(x) for x in rel.iloc[::5]]})
for g in groups:
    a, b = g["vs_n500_1m"] or 0, g["vs_n500_3m"] or 0
    g["state"] = ("Gaining strength" if a > 0.01 else "Losing strength" if a < -0.01 else "Steady") + (" (also ahead over 3 months)" if a > 0.01 and b > 0 else " (also behind over 3 months)" if a < -0.01 and b < 0 else "")
groups.sort(key=lambda g: -(g["vs_n500_1m"] or 0))
out["forces"] = {"items": forces, "method": "Principal components of the last 250 days of daily returns of index stocks (standardized). Force 1 is the market as a whole; others are named by the industries most on each side."}
out["groups"] = {"items": groups, "k": K, "stocks": int(len(lbl)),
                 "method": "Each stock's last 250 days of daily moves with the market's part removed (by its beta), compressed to 15 components and clustered into 24 groups (k-means). 'Togetherness' = average correlation of members' own moves; 'Mixed' groups cut across NSE industries."}

# ------------------------------------------------------------------ 3. move breakdown and breadth of moves
print("breakdown", flush=True)
rows = []
for h, n in (("1m", 21), ("3m", 63)):
    tot = logc.iloc[-1] - logc.iloc[-1 - n]
    mk = n500.iloc[-1] - n500.iloc[-1 - n]
    ig = ind.reindex(tot.index)
    imed = tot.groupby(ig).median()
    ipart = ig.map(imed) - mk
    rows.append(pd.DataFrame({f"tot_{h}": tot, f"mkt_{h}": mk, f"ind_{h}": ipart, f"own_{h}": tot - mk - ipart}))
B = pd.concat(rows, axis=1).dropna(subset=["tot_3m"])
B["ind"] = ind.reindex(B.index)
out["breakdown"] = {"rows": [{"s": s, "ind": r["ind"] if isinstance(r["ind"], str) else None, **{k: r4(r[k]) for k in B.columns if k != "ind"}} for s, r in B.iterrows()],
                    "method": "Log returns. Market part = Nifty 500's move. Industry part = the median move of the stock's NSE industry minus the market. Own part = the rest."}
cons = pd.read_csv(os.path.join(S3, "constituents.csv"))
I = {x["name"]: x for x in json.load(open(os.path.join(ROOT, "explorer", "data", "deep", "indices.json")))["indices"]}
conc = []
for name in sorted((_SECTOR | _THEME) & set(I)):
    x = I[name]; mem = [mm for mm in x["members"] if mm.get("w") is not None and mm.get("m1") is not None]
    if len(mem) < 5:
        continue
    for h in ("m1", "m3"):
        contrib = sorted(((mm["w"] * mm[h], mm["s"]) for mm in mem if mm.get(h) is not None), reverse=True)
        tot = sum(v for v, _ in contrib)
        up = sum(1 for mm in mem if (mm.get(h) or 0) > 0) / len(mem)
        pick = contrib[:3] if tot > 0 else contrib[-3:]
        top3 = sum(v for v, _ in pick)
        share = top3 / tot if tot else None
        wsum = sum(mm["w"] for mm in mem); wt = {mm["s"]: mm["w"] for mm in mem}
        w3 = sum(wt[s_] for _, s_ in pick) / wsum if wsum else None
        if h == "m1":
            row = {"name": name, "g": x.get("g"), "n": len(mem)}
        row.update({f"ret_{h}": r4(x.get(h)), f"up_{h}": r4(up), f"top3_share_{h}": r4(share), f"top3_weight_{h}": r4(w3),
                    f"top3_{h}": [s for _, s in (contrib[:3] if tot > 0 else contrib[-3:])]})
    r = row["ret_m1"] or 0; u = row["up_m1"]; ex = (row["top3_share_m1"] or 0) - (row["top3_weight_m1"] or 0)   # top 3's share of the move beyond their weight
    row["excess_m1"] = r4(ex)
    if abs(r) < 0.01:
        row["verdict"] = "Flat"
    elif r > 0:
        row["verdict"] = "Broad rise" if u >= 0.6 and ex < 0.25 else "Narrow rise (few stocks)" if u < 0.5 or ex >= 0.4 else "Mixed rise"
    else:
        row["verdict"] = "Broad fall" if u <= 0.4 and ex < 0.25 else "Narrow fall (few stocks)" if u > 0.5 or ex >= 0.4 else "Mixed fall"
    conc.append(row)
out["breadth_of_moves"] = {"rows": conc, "method": "Sector and theme indices, last 1 and 3 months. Members up = share of members with a positive return. Top-3 share = how much of the index's weighted move came from its 3 biggest contributors, compared with their weight in the index (weights approximate). Broad = most members joined and the top 3 gave no more than about their weight; narrow = few members joined or the top 3 gave far more than their weight. Moves under 1% count as flat."}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), separators=(",", ":"), allow_nan=False, default=lambda o: None)
print("structure.json", os.path.getsize(OUT) // 1024, "KB; regime now:", out["regime"]["now"], "for", streak, "weeks; groups", len(groups), flush=True)
