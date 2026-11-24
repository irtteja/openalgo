"""Collect everything the explorer page shows into explorer/data/*.json.

    python3 scripts/build_explorer.py      (after scripts/pattern_loop.py live and scripts/rotation.py)
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel, DATA_DIR
from rsi.fundamentals import add_fundamental_features
from rsi.corp_features import add_corp_features
from rsi.step3_features import add_step3_features, _index, S3, index_universe, is_sme, index_category, SHORT_DAYS

R = "results/patterns"
OUT = "explorer/data"
os.makedirs(OUT, exist_ok=True)


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def dump(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(clean(obj), f, separators=(",", ":"))
    print(name, os.path.getsize(os.path.join(OUT, name)) // 1024, "KB")


ds, dg = json.load(open(f"{R}/detail_stocks.json")), json.load(open(f"{R}/detail_groups.json"))
UNI = index_universe()                       # page shows only stocks in NSE indices (no SME index)
ds["rows"] = {k: v for k, v in ds["rows"].items() if k in UNI}
dg["rows"] = {k: v for k, v in dg["rows"].items() if not is_sme(k)}
dg["members"] = {k: [s for s in v if s in UNI] for k, v in dg.get("members", {}).items() if not is_sme(k)}
p = load_market_panel()
for f in (add_fundamental_features, add_corp_features, add_step3_features):
    p = f(p)
close, ret, idx = p["close"], p["ret"], p["close"].index
t = idx[-1]
logc = np.log(close)
n500 = _index("NIFTY 500", idx)
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
tv = (p["close_raw"] * p["volume_raw"]).rolling(60, min_periods=20).mean()
wk = idx[::-1][::5][::-1][-53:]                       # last year, weekly


def last(df):
    return df.loc[t] if t in df.index else df.iloc[-1]


tr = pd.DataFrame({
    "price": last(p["close_raw"]), "r1w": last(logc.diff(5)), "r1m": last(logc.diff(21)), "r3m": last(logc.diff(63)),
    "r1y": last(logc.diff(250)), "vs3m": last(logc.diff(63)) - n500.diff(63).loc[t],
    "fromHigh": last(logc - logc.rolling(250, min_periods=60).max()), "fromLow": last(logc - logc.rolling(250, min_periods=60).min()),
    "dlv": last(p["dlv"].rolling(20, min_periods=10).mean()),
    "dlvChg": last(p["dlv"].rolling(5, min_periods=3).mean() - p["dlv"].rolling(60, min_periods=20).mean()),
    "vol": last(ret.rolling(60, min_periods=20).std()) * np.sqrt(250), "tvCr": last(tv) / 1e7,
    "ey": last(p["ey"]) if "ey" in p else np.nan, "profG": last(p["prof_g"]) if "prof_g" in p else np.nan,
    "pledge": last(p["pledge"]) if "pledge" in p else np.nan, "ins": last(p["ins_net"]) if "ins_net" in p else np.nan,
    "grp": last(p["_closest_index"]) if "_closest_index" in p else None,
})
spark = np.exp(logc.reindex(wk)).div(np.exp(logc.reindex(wk)).bfill().iloc[0])

stocks = []
for k, r in ds["rows"].items():
    if k not in tr.index:
        continue
    a = tr.loc[k]
    s = spark[k].round(3).tolist() if k in spark else []
    stocks.append({"s": k, "ind": ind.get(k), "grp": a["grp"], **{c: a[c] for c in tr.columns if c != "grp"},
                   "pr": r["prob"], "m": r["match"], "f": r["facts"], "sp": s})
dump(stocks, "stocks.json")

rot = pd.read_csv("results/rotation_now.csv", index_col=0)
rkey = lambda k: ("idx: " + k[7:].replace("NIFTY ", "").replace("NIFTY", "")) if k.startswith("index: ") else "ind: " + k[10:]
G = {}
for k in dg["rows"]:
    if k.startswith("index: "):
        G[k] = _index(k[7:], idx)
    else:
        syms = [s for s in dg.get("members", {}).get(k, []) if s in close.columns]
        if syms:
            G[k] = np.log1p(ret[syms].where(close[syms].notna()).mean(1).fillna(0)).cumsum()
groups = []
for k, r in dg["rows"].items():
    rr = rot.loc[rkey(k)] if rkey(k) in rot.index else None
    L = G.get(k)
    sp = rs = []
    if L is not None:
        sp = np.exp(L.reindex(wk) - L.reindex(wk).bfill().iloc[0]).round(4).tolist()
        rs = np.exp((L - n500).reindex(wk) - (L - n500).reindex(wk).bfill().iloc[0]).round(4).tolist()
    mem = dg.get("members", {}).get(k, [])
    days = int(L.notna().sum()) if L is not None else 0
    groups.append({"g": k, "cat": index_category(k), "days": days, "short": days < SHORT_DAYS,
                   "since": str(L.first_valid_index().date()) if L is not None and L.notna().any() else None, "pr": r["prob"], "m": r["match"], "f": r["facts"], "sp": sp, "rs": rs, "members": mem,
                   **({} if rr is None else {"phase": rr["phase"], "vs1m": rr["1m vs mkt %"], "vs3m": rr["3m vs mkt %"],
                                             "vs6m": rr["6m vs mkt %"], "vs1y": rr["1y vs mkt %"], "fromHigh": rr["from 52w high %"],
                                             "breadth": rr["% members >50d avg"], "breadthChg": rr["breadth chg 1m"],
                                             "dlvTrend": rr["delivery trend"], "base": bool(rr["base"]), "breakout": bool(rr["breakout"])})})
dump(groups, "groups.json")

pats = {"stocks": ds["patterns"], "groups": dg["patterns"]}
dump(pats, "patterns.json")


def md_tables(path):
    out, cur, name = {}, None, None
    for line in open(path):
        if line.startswith("## "):
            name = line[3:].strip()
        elif line.startswith("|") and not set(line.strip()) <= set("|:- "):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = {"cols": cells, "rows": []}
            else:
                cur["rows"].append(cells)
        elif cur is not None:
            out[name] = cur; cur = None
    if cur is not None:
        out[name] = cur
    return out


vix = pd.read_csv(os.path.join(DATA_DIR, "extra", "vix.csv"), parse_dates=["Date"]).set_index("Date")["vix"]
br = (close > close.rolling(200, min_periods=150).mean()).where(close.notna()).astype(float).mean(1)
br50 = (close > close.rolling(50, min_periods=40).mean()).where(close.notna()).astype(float).mean(1)
adv = (ret > 0).where(close.notna()).astype(float).mean(1)
w2 = idx[::-1][::5][::-1][-157:]                       # 3 years weekly
lib_hist = json.load(open(f"{R}/library_stocks.json")).get("history", [])
market = {
    "date": str(t.date()), "regime": ds["market_now"], "normal": {"stocks": ds["normal"], "groups": dg["normal"]},
    "n500": float(np.exp(n500.loc[t])), "n500_1m": float(n500.diff(21).loc[t]), "n500_3m": float(n500.diff(63).loc[t]),
    "n500_1y": float(n500.diff(250).loc[t]), "n500_fromHigh": float((n500 - n500.rolling(250).max()).loc[t]),
    "vix": float(vix.iloc[-1]), "vixPct": float((vix.rank(pct=True)).iloc[-1]),
    "breadth200": float(br.loc[t]), "breadth50": float(br50.loc[t]), "adv10": float(adv.rolling(10).mean().loc[t]),
    "listed": len(stocks), "allListed": int(close.loc[t].notna().sum()),
    "hist": {"d": [str(d.date()) for d in w2], "n500": np.exp(n500.reindex(w2)).round(1).tolist(),
             "breadth200": br.reindex(w2).round(3).tolist(), "vix": vix.reindex(w2, method="ffill").round(2).tolist()},
    "backtest": md_tables(f"{R}/backtest.md"), "loop": lib_hist,
    "groupLoop": json.load(open(f"{R}/library_groups.json")).get("history", []),
    "explore": md_tables("results/explore.md"), "rotationHistory": md_tables("results/rotation.md"),
}
dump(market, "market.json")
