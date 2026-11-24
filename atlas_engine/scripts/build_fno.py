"""F&O build-up -> explorer/data/deep/fno.json

    python3 scripts/build_fno.py      (after fetch_step3.py futures and options)

Each day, for every stock and index with futures (all expiries added together):
  Long build-up   - price up,   open interest up    (fresh buying)
  Short build-up  - price down, open interest up    (fresh selling)
  Short covering  - price up,   open interest down  (sellers exiting)
  Long unwinding  - price down, open interest down  (buyers exiting)
  No clear signal - open interest changed by less than 1%
Price = the stock's own (split-adjusted) close; for indices, the index close. Also done over 5 days.
Expiry days drop open interest as contracts close, so they are flagged.
History checks: every F&O stock 2018 -> 2026, return against the Nifty 500 afterwards.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel
from rsi.step3_features import _index, S3, index_universe

OUT = "explorer/data/deep"
CL = ["Long build-up", "Short build-up", "Short covering", "Long unwinding", "No clear signal"]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
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


def classify(pr, oi, thr=0.01):
    return pd.DataFrame(np.select([(oi >= thr) & (pr > 0), (oi >= thr) & (pr < 0), (oi <= -thr) & (pr > 0), (oi <= -thr) & (pr < 0)],
                                  CL[:4], CL[4]), index=pr.index, columns=pr.columns).where(pr.notna() & oi.notna())


p = load_market_panel()
close, ret = p["close"], p["ret"]
idx = close.index
T = idx[-1]
logc = np.log(close)
n500 = _index("NIFTY 500", idx).ffill()

# ---------------------------------------------------------------- read futures files
frames = {}
for f in sorted(glob.glob(os.path.join(S3, "fut", "*.csv"))):
    if os.path.getsize(f) > 50:
        frames[pd.Timestamp(os.path.basename(f)[:-4])] = pd.read_csv(f)
F = pd.concat(frames, names=["Date"]).reset_index(level=0)
from rsi.data import symbol_renames
F["symbol"] = F["symbol"].replace(symbol_renames())   # e.g. ICICIPRULI -> ICICILIFE
F = F[F["Date"].isin(idx)]
piv = lambda k, col: F[F["kind"] == k].pivot_table(index="Date", columns="symbol", values=col, aggfunc="sum").reindex(idx)
OI, OIC, CTR, VAL, FCL = (piv("STK", c) for c in ("oi", "oi_chg", "contracts", "value", "close"))
syms = [s for s in OI.columns if s in close.columns]
OI, OIC, CTR, VAL, FCL = (x[syms] for x in (OI, OIC, CTR, VAL, FCL))
on = OI.notna()
oi_pct = (OIC / (OI - OIC).replace(0, np.nan)).clip(-0.9, 3)
pr1 = ret[syms].where(on)
# expiry days: OI falls for most stocks at once
exp_day = (oi_pct.lt(-0.05).mean(1) > 0.5)
cls1 = classify(pr1, oi_pct)
oi5 = (OI / OI.shift(5) - 1).clip(-0.9, 3)
pr5 = (logc[syms].diff(5)).where(on)
cls5 = classify(pr5, oi5, 0.03)

# ---------------------------------------------------------------- history checks (vs Nifty 500)
print("evidence", flush=True)
RSm = logc[syms].sub(n500, axis=0)
fwd = {h: RSm.shift(-n) - RSm for h, n in [("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63)]}
okd = ~exp_day & (idx <= T - pd.Timedelta(days=120)) & (idx >= "2018-02-01")
ev1 = []
for c in CL:
    m = (cls1 == c).loc[okd]
    ev1.append({"cls": c, **{h: stat(fwd[h].loc[okd][m].stack()) for h in fwd}})
big = (oi_pct.abs() >= oi_pct.abs().rolling(250, min_periods=60).quantile(0.9)) & (pr1.abs() >= 0.02)
for c in CL[:4]:
    m = ((cls1 == c) & big).loc[okd]
    ev1.append({"cls": c + " (strong: big price move + top-10% OI change)", **{h: stat(fwd[h].loc[okd][m].stack()) for h in fwd}})
ev1.append({"cls": "Any F&O stock, any day", **{h: stat(fwd[h].loc[okd][cls1.loc[okd].notna()].stack()) for h in fwd}})
W = idx[::-1][::5][::-1]
Wp = W[(W >= "2018-03-01") & (W <= T - pd.Timedelta(days=120))]
ev5 = [{"cls": c, **{h: stat(fwd[h].loc[Wp][(cls5 == c).loc[Wp]].stack()) for h in ("1w", "1m", "3m")}} for c in CL]
ev5.append({"cls": "Any F&O stock, any week", **{h: stat(fwd[h].loc[Wp][cls5.loc[Wp].notna()].stack()) for h in ("1w", "1m", "3m")}})

# ---------------------------------------------------------------- options PCR
pcr = {}
for f in sorted(glob.glob(os.path.join(S3, "options", "*.csv")))[-80:]:
    if os.path.getsize(f) > 50:
        d = pd.read_csv(f).set_index("symbol")
        if "oi_pe" in d and "oi_ce" in d:
            pcr[pd.Timestamp(os.path.basename(f)[:-4])] = d["oi_pe"] / d["oi_ce"].replace(0, np.nan)
PCR = pd.DataFrame(pcr).T.sort_index() if pcr else pd.DataFrame()

# ---------------------------------------------------------------- today
print("today", flush=True)
UNI = index_universe()
rsj = json.load(open(os.path.join(OUT, "rs.json")))["stocks"]
scr = {r["s"]: r for r in json.load(open(os.path.join(OUT, "screen.json")))["rows"]}
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
lastd = OI.dropna(how="all").index[-1]
rows = []
L60 = idx[idx <= lastd][-60:]
for s in syms:
    if not np.isfinite(OI[s].loc[lastd]):
        continue
    c1 = cls1[s].loc[L60].tolist()
    streak = 0
    for x in reversed(c1):
        if x == c1[-1]:
            streak += 1
        else:
            break
    spot = p["close_raw"][s].loc[lastd]
    pc = PCR[s] if (len(PCR) and s in PCR.columns) else None
    rows.append({"s": s, "ind": ind.get(s), "price": spot, "chg": pr1[s].loc[lastd], "oiChg": oi_pct[s].loc[lastd], "oi": OI[s].loc[lastd],
                 "oiVal": OI[s].loc[lastd] * spot / 1e7, "oiVs20": OI[s].loc[lastd] / OI[s].loc[:lastd].iloc[-20:].mean() - 1,
                 "chg5": np.expm1(pr5[s].loc[lastd]), "oi5": oi5[s].loc[lastd], "cls": cls1[s].loc[lastd], "cls5": cls5[s].loc[lastd], "streak": streak,
                 "basis": FCL[s].loc[lastd] / spot - 1 if np.isfinite(FCL[s].loc[lastd]) and spot else None,
                 "pcr": pc.iloc[-1] if pc is not None else None, "pcrChg": (pc.iloc[-1] - pc.iloc[-6]) if pc is not None and len(pc) > 6 else None,
                 "contracts": CTR[s].loc[lastd], "ctrVs20": CTR[s].loc[lastd] / CTR[s].loc[:lastd].iloc[-20:].mean() - 1,
                 "rs": rsj.get(s, {}).get("rating"), "stage": scr.get(s, {}).get("stage"), "inIndex": s in UNI,
                 "hist": {"cls": c1, "oi": OI[s].loc[L60].round(0).tolist(), "px": p["close_raw"][s].loc[L60].round(2).tolist()}})
dist = [{"d": str(d.date()), **{c: int((cls1.loc[d] == c).sum()) for c in CL}, "expiry": bool(exp_day.loc[d])} for d in L60]

# ---------------------------------------------------------------- index futures
IDXF = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FINANCIAL SERVICES", "MIDCPNIFTY": "NIFTY MIDCAP SELECT", "NIFTYNXT50": "NIFTY NEXT 50"}
IOI, IOIC, ICL = piv("IDX", "oi"), piv("IDX", "oi_chg"), piv("IDX", "close")
idxrows = []
L250 = idx[idx <= lastd][-250:]
for sym, iname in IDXF.items():
    if sym not in IOI.columns:
        continue
    lvl = _index(iname, idx)
    if lvl is None:
        continue
    lv = np.exp(lvl)
    o, oc = IOI[sym], IOIC[sym]
    op = (oc / (o - oc).replace(0, np.nan)).clip(-0.9, 3)
    pr = lv.pct_change()
    c = classify(pd.DataFrame({sym: pr}), pd.DataFrame({sym: op}))[sym]
    idxrows.append({"sym": sym, "name": iname, "level": lv.loc[lastd], "chg": pr.loc[lastd], "oiChg": op.loc[lastd], "oi": o.loc[lastd],
                    "basis": ICL[sym].loc[lastd] / lv.loc[lastd] - 1 if np.isfinite(ICL[sym].loc[lastd]) else None, "cls": c.loc[lastd],
                    "last20": [{"d": str(d.date()), "chg": pr.loc[d], "oiChg": op.loc[d], "cls": c.loc[d], "expiry": bool(exp_day.loc[d])} for d in L60[-20:]],
                    "chart": {"d": [str(d.date()) for d in L250], "lvl": lv.loc[L250].round(2).tolist(), "oi": o.loc[L250].round(0).tolist()}})
part = {}
try:
    from rsi.market_features import add_market_features
    sub = {k: p[k][syms[:2]] for k in ("close", "volume", "volume_raw", "close_raw") if k in p}
    q = add_market_features(sub)
    for k in ("fii_fut", "cli_fut"):
        if k in q:
            part[k] = q[k].iloc[:, 0].reindex(L250).round(4).tolist()
except Exception as e:
    print("participants skipped:", repr(e)[:120])
part["d"] = [str(d.date()) for d in L250]
tot = (OI * p["close_raw"][syms]).sum(1) / 1e7
mkt = {"d": [str(d.date()) for d in L250], "oiVal": tot.loc[L250].round(0).tolist(), "n500": np.exp(n500.loc[L250]).round(1).tolist()}

json.dump(clean({"date": lastd, "classes": CL, "rows": rows, "dist": dist, "evidence1": ev1, "evidence5": ev5, "index": idxrows,
                 "participants": part, "market": mkt, "expiryToday": bool(exp_day.loc[lastd])}),
          open(os.path.join(OUT, "fno.json"), "w"), separators=(",", ":"))
print("DONE fno", len(rows), os.path.getsize(os.path.join(OUT, "fno.json")) // 1024, "KB", flush=True)
