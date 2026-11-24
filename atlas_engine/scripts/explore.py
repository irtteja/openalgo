"""Broad look at the data: patterns and relations at many time scales, events, calendar, sectors, market.

    python3 scripts/explore.py      -> results/explore.md

Everything uses all NSE main-board stocks 2015 -> 2026 (each stock only while listed).
"Excess return" = stock return minus the average stock's return the same day (equal-weighted market).
"IC" = rank correlation across stocks between a trait today and the return over the next period.
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel, DATA_DIR
from rsi.extra import add_extra_features
from rsi.fundamentals import add_fundamental_features
from rsi.market_features import add_market_features
from rsi.corp_features import add_corp_features, _read, _next_day, SENIOR
from rsi.step3_features import add_step3_features, _index

OUT = []
def say(s=""): OUT.append(s); print(s, flush=True)
def table(df, nd=3): say(df.round(nd).to_markdown()); say()

p = load_market_panel()
for f in (add_extra_features, add_fundamental_features, add_market_features, add_corp_features, add_step3_features):
    p = f(p)
close, ret = p["close"], p["ret"]
idx, cols = close.index, close.columns
logc = np.log(close)
listed = close.notna()
mkt = ret.where(listed).mean(1)                       # equal-weight market daily return
ex = ret.sub(mkt, axis=0).where(listed)
X = ex.fillna(0).cumsum().where(listed).ffill()        # cumulative excess return index (log-ish)
tv = (p["close_raw"] * p["volume_raw"]).rolling(60, min_periods=20).mean()
size_rank = tv.rank(axis=1, ascending=False)
top500 = size_rank <= 500
vol20 = ret.rolling(20, min_periods=10).std()
weekly = idx[::5]

say(f"# Data exploration: all NSE main-board stocks\n\n{cols.size} stocks, {idx[0].date()} -> {idx[-1].date()}, "
    f"{int(listed.sum(1).mean())} listed on an average day.\n")

# ------------------------------------------------------------------ 1. time scales
say("## 1. Which traits matter, and over what time span\n")
say("Rank correlation (IC) between each trait and the stock's return over the NEXT 1 day ... 1 year. "
    "Positive = higher trait, higher later return. All stocks / biggest 500.\n")
traits = {
    "last week's return": logc.diff(5), "last month's return": logc.diff(20), "last 3 months' return": logc.diff(60),
    "last year's return": logc.diff(250), "volatility (wildness)": vol20, "size": np.log(tv),
    "distance from 52-week high": logc - logc.rolling(250, min_periods=60).max(),
    "delivery % (20-day avg)": p["dlv"].rolling(20, min_periods=10).mean(),
    "delivery % rising": p["dlv"].rolling(5, min_periods=3).mean() - p["dlv"].rolling(60, min_periods=20).mean(),
    "volume vs usual": np.log(p["volume"].rolling(20, min_periods=10).mean() / p["volume"].rolling(250, min_periods=60).mean()),
    "price level (rupees)": np.log(p["close_raw"]),
}
for k, lab in [("ey", "earnings yield (cheapness)"), ("prof_g", "profit growth"), ("rev_g", "sales growth"),
               ("pledge", "promoter pledging"), ("ins_net", "insider net buying"), ("prom_chg", "promoter stake change"),
               ("sec_mom", "sector's 3-month trend"), ("var", "NSE risk margin"), ("mtf", "margin-funded buying")]:
    if k in p:
        traits[lab] = p[k]
H = {"1 day": 1, "1 week": 5, "1 month": 21, "3 months": 63, "6 months": 126, "1 year": 250}
fwd = {h: (logc.shift(-n) - logc).loc[weekly] for h, n in H.items()}


def ic(x, y, m):
    x, y = x.where(m), y.where(m)
    ok = x.notna() & y.notna()
    rx, ry = x.where(ok).rank(axis=1), y.where(ok).rank(axis=1)
    rx, ry = rx.sub(rx.mean(1), axis=0), ry.sub(ry.mean(1), axis=0)
    r = (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    return r[ok.sum(1) >= 30]


rows = {}
for lab, x in traits.items():
    xs = x.reindex_like(close).replace([np.inf, -np.inf], np.nan).loc[weekly]
    for g, m in [("all", listed.loc[weekly]), ("big 500", top500.loc[weekly])]:
        rows[(lab, g)] = {h: ic(xs, fwd[h], m).mean() for h in H}
table(pd.DataFrame(rows).T)

# ------------------------------------------------------------------ 2. events
say("## 2. What happens around events\n")
say("Average excess return (%) from the event day to N days later (negative N: the run-up BEFORE the event). "
    "'% beat mkt 3m' = share of events where the stock beat the market over the next 3 months.\n")
K = [-60, -20, -5, 5, 20, 60, 120, 250]
Xv, pos_of = X.to_numpy(), {s: i for i, s in enumerate(cols)}


def study(ev):  # ev: DataFrame [day, symbol]
    ev = ev.dropna().drop_duplicates()
    ev = ev[ev["symbol"].isin(pos_of)]
    t = idx.get_indexer(ev["day"]); c = ev["symbol"].map(pos_of).to_numpy()
    ok = t >= 0; t, c = t[ok], c[ok]
    base = Xv[t, c]
    r = {}
    for k in K:
        tt = t + k; m = (tt >= 0) & (tt < len(idx))
        v = np.full(len(t), np.nan); v[m] = Xv[tt[m], c[m]] - base[m]
        if k < 0: v = -v
        r[f"{k:+d}d"] = np.nanmean(v) * 100
    tt = t + 60; m = tt < len(idx)
    v = Xv[tt[m], c[m]] - base[m]
    r["% beat mkt 3m"] = np.mean(v[np.isfinite(v)] > 0) * 100
    r["events"] = len(t)
    return r


def from_mask(m):
    s = m.stack(); s = s[s]
    return pd.DataFrame({"day": s.index.get_level_values(0), "symbol": s.index.get_level_values(1)})


E = {}
ins = _read("insider")
if len(ins):
    ins = ins.drop_duplicates()
    ins["day"] = _next_day(pd.to_datetime(ins["date"], format="%d-%b-%Y %H:%M", errors="coerce"), idx)
    cat = ins["personCategory"].astype(str).str.strip().str.lower()
    mode = ins["acqMode"].astype(str).str.strip().str.lower()
    val = pd.to_numeric(ins["secVal"], errors="coerce").fillna(0)
    sen, prom = cat.isin(SENIOR), cat.isin({"promoters", "promoter group"})
    pick = lambda m: ins.loc[m, ["day", "symbol"]]
    E["insider BUYS (promoter/director, > Rs 10 lakh)"] = pick(sen & (mode == "market purchase") & (val > 1e6))
    E["insider SELLS (promoter/director, > Rs 10 lakh)"] = pick(sen & (mode == "market sale") & (val > 1e6))
    E["promoter PLEDGES shares"] = pick(prom & (mode == "pledge creation"))
    E["lender SELLS pledged shares (invocation)"] = pick(prom & (mode == "invocation of pledge"))
    E["promoter RELEASES pledge"] = pick(prom & (mode == "revokation of pledge"))
if "prom_chg" in p:
    pc = p["prom_chg"]; new = pc.notna() & pc.shift(1).isna()
    E["promoter stake UP >= 1%"] = from_mask(new & (pc >= 1))
    E["promoter stake DOWN >= 1%"] = from_mask(new & (pc <= -1))
r1 = ret
E["stock jumps > +10% in a day"] = from_mask(r1 > 0.10)
E["stock falls > -10% in a day"] = from_mask(r1 < -0.10)
hi = logc >= logc.rolling(250, min_periods=200).max()
E["new 52-week high (first in 3 months)"] = from_mask(hi & ~hi.shift(1).rolling(60, min_periods=1).max().astype(bool))
lo = logc <= logc.rolling(250, min_periods=200).min()
E["new 52-week low (first in 3 months)"] = from_mask(lo & ~lo.shift(1).rolling(60, min_periods=1).max().astype(bool))
E["volume 5x its usual"] = from_mask(p["volume"] > 5 * p["volume"].rolling(60, min_periods=20).mean().shift(1))
dl = p["dlv"]
E["delivery % jumps 30+ points above usual"] = from_mask(dl - dl.rolling(60, min_periods=20).mean().shift(1) > 0.30)
if "bulk" in p:
    E["bulk deal: net BUY"] = from_mask(p["bulk"] > 0.05)
    E["bulk deal: net SELL"] = from_mask(p["bulk"] < -0.05)
first = listed & ~listed.shift(1, fill_value=False)
first.iloc[:5] = False
E["new listing (IPO), from day 1 close"] = from_mask(first)
table(pd.DataFrame({k: study(v) for k, v in E.items()}).T, 1)

say("### Stocks that stopped trading (delisted / suspended / merged) before the end\n")
last = listed & ~listed.shift(-1, fill_value=False)
last.iloc[-30:] = False
gone = from_mask(last)
g = gone.groupby("symbol")["day"].max().reset_index()
g = g[g["day"] < idx[-30]]
r = {}
for k in [250, 120, 60, 20]:
    t = idx.get_indexer(g["day"]); c = g["symbol"].map(pos_of).to_numpy(); tt = t - k; m = tt >= 0
    r[f"last {k}d"] = np.nanmean(Xv[t[m], c[m]] - Xv[tt[m], c[m]]) * 100
r["stocks"] = len(g)
table(pd.DataFrame([r], index=["excess return before disappearing (%)"]), 1)

# ------------------------------------------------------------------ 3. calendar
say("## 3. Calendar\n")
n500 = _index("NIFTY 500", idx)
small = _index("NIFTY SMALLCAP 250", idx)
big = _index("NIFTY 50", idx)
mr = n500.diff()
say("Nifty 500 average daily return (%) and share of up days, by weekday and by month; small minus large caps by month.\n")
wd = pd.DataFrame({"avg %": mr.groupby(idx.dayofweek).mean() * 100, "% up days": (mr > 0).groupby(idx.dayofweek).mean() * 100})
wd.index = ["Mon", "Tue", "Wed", "Thu", "Fri"][: len(wd)]
table(wd, 3)
mon = n500.resample("ME").last().diff().dropna()
sm = (small.resample("ME").last().diff() - big.resample("ME").last().diff()).dropna()
cal = pd.DataFrame({"Nifty500 avg month %": mon.groupby(mon.index.month).mean() * 100,
                    "% up years": (mon > 0).groupby(mon.index.month).mean() * 100,
                    "small minus large %": sm.groupby(sm.index.month).mean() * 100})
cal.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
table(cal, 2)
dom = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).transform(lambda s: s - s.iloc[0])
dto = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).transform(lambda s: s.iloc[-1] - s)
tom = (dom < 3) | (dto < 2)
table(pd.DataFrame({"avg daily %": [mr[tom].mean() * 100, mr[~tom].mean() * 100]}, index=["turn of month (last 2 + first 3 days)", "rest of month"]), 3)
say("Average size of daily moves (all stocks, %) by month — when are stocks most active?\n")
am = ret.abs().where(listed).mean(1)
table(pd.DataFrame({"avg absolute daily move %": am.groupby(idx.month).mean().values * 100}, index=cal.index).T, 2)

# ------------------------------------------------------------------ 4. relations
say("## 4. Relations between sectors, stocks and the market\n")
SECT = ["NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY AUTO", "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY",
        "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY PSU BANK", "NIFTY PRIVATE BANK", "NIFTY OIL & GAS", "NIFTY HEALTHCARE INDEX",
        "NIFTY CONSUMER DURABLES", "NIFTY INFRASTRUCTURE", "NIFTY PSE", "NIFTY MNC", "NIFTY COMMODITIES", "NIFTY MIDCAP 150",
        "NIFTY SMALLCAP 250", "NIFTY 50"]
S = pd.DataFrame({s.replace("NIFTY ", ""): _index(s, idx) for s in SECT}).dropna(axis=1, thresh=1000)
Sw = S.resample("W-FRI").last().diff().dropna(how="all")
cm = Sw.drop(columns=["50", "MIDCAP 150", "SMALLCAP 250"], errors="ignore").corr()
pairs = cm.where(np.triu(np.ones(cm.shape, bool), 1)).stack().sort_values()
say("Weekly return correlation between sectors (1 = always move together, 0 = unrelated).\n")
table(pd.DataFrame({"most separate": pairs.head(8).index.map(" vs ".join), "corr": pairs.head(8).values}).set_index("most separate"), 2)
table(pd.DataFrame({"most together (excluding same family)": pairs.tail(8).index.map(" vs ".join), "corr": pairs.tail(8).values}).set_index("most together (excluding same family)"), 2)
say("Correlation of each sector with the Nifty 50 (how much it just follows the market), and its beta (moves x% per 1% of Nifty 50).\n")
b = pd.DataFrame({"corr with Nifty 50": Sw.corr()["50"], "beta": Sw.cov()["50"] / Sw["50"].var()}).drop("50").sort_values("beta")
table(b, 2)

say("How closely do stocks move together, by year (average pairwise correlation of daily returns, biggest 300 stocks). "
    "High = everything moves as one (usually in falls).\n")
yr = {}
for y in range(idx[0].year, idx[-1].year + 1):
    m = idx.year == y
    big300 = size_rank[m].mean().nsmallest(300).index
    c = ret.loc[m, big300].corr().to_numpy()
    yr[y] = np.nanmean(c[np.triu_indices_from(c, 1)])
y5 = n500.resample("YE").last().diff()
table(pd.DataFrame({"avg stock correlation": pd.Series(yr), "Nifty 500 return %": (y5.values * 100)[: len(yr)]}).T, 2)

say("Does one sector's week predict another sector's NEXT week? (top lead-lag links, correlation)\n")
L = {}
for a in Sw.columns:
    for b_ in Sw.columns:
        if a != b_:
            L[(a, b_)] = Sw[a].corr(Sw[b_].shift(-1))
L = pd.Series(L).sort_values(key=abs, ascending=False).head(8)
table(pd.DataFrame({"leader -> follower next week": [f"{a} -> {b_}" for a, b_ in L.index], "corr": L.values}).set_index("leader -> follower next week"), 3)
say("(With ~200 pairs some links appear by chance; values under ~0.1 are weak.)\n")

say("Market-wide signals vs the Nifty 500 later (correlation, weekly samples).\n")
vix = pd.read_csv(os.path.join(DATA_DIR, "extra", "vix.csv"), parse_dates=["Date"]).set_index("Date")["vix"].reindex(idx).ffill()
breadth200 = (close > close.rolling(200, min_periods=150).mean()).where(listed).mean(1)
adv = (ret > 0).where(listed).mean(1).rolling(10).mean()
mkt_sig = {"India VIX (fear)": np.log(vix), "% of stocks above 200-day avg": breadth200,
           "% of stocks rising (10-day avg)": adv, "Nifty 500 last 3 months": n500.diff(63),
           "small caps vs large caps, last 3 months": small.diff(63) - big.diff(63)}
for k in ["fii_fut", "cli_fut"]:
    if k in p:
        mkt_sig[{"fii_fut": "FII index-futures position", "cli_fut": "retail index-futures position"}[k]] = p[k].iloc[:, 0]
rows = {}
for lab, s in mkt_sig.items():
    rows[lab] = {h: s.loc[weekly].corr((n500.shift(-n) - n500).loc[weekly], method="spearman") for h, n in list(H.items())[1:]}
table(pd.DataFrame(rows).T, 3)
fv = mr.rolling(21).std().shift(-21) * np.sqrt(250) * 100
say(f"VIX vs the market's ACTUAL swings over the next month: correlation {np.log(vix).corr(np.log(fv)):.2f} "
    f"(VIX is a good forecast of how bumpy the next month will be, even though it doesn't tell direction).\n")
say("Big stocks vs small stocks: does Nifty 50's week predict the Smallcap 250's next week? "
    f"corr = {Sw['50'].corr(Sw['SMALLCAP 250'].shift(-1)):.3f}; reverse = {Sw['SMALLCAP 250'].corr(Sw['50'].shift(-1)):.3f}\n")

say("Delivery % and volatility: average of the next month's volatility (daily %, annualised) by delivery group.\n")
dq = p["dlv"].rolling(20, min_periods=10).mean().loc[weekly]
fvol = ret.rolling(21).std().shift(-21).loc[weekly] * np.sqrt(250) * 100
qq = dq.rank(axis=1, pct=True)
table(pd.DataFrame({f"delivery group {i + 1} (low->high)": [fvol.where((qq > i / 5) & (qq <= (i + 1) / 5)).stack().mean()] for i in range(5)},
                   index=["next-month volatility %"]), 1)

# ------------------------------------------------------------------ 5. big winners / losers
say("## 5. What did big winners and big losers look like BEFORE their move?\n")
say("Stocks whose next-year return was > +100% (doubled) or < -50% (halved), vs all stocks; median of each trait at the start.\n")
f1 = (logc.shift(-250) - logc).loc[weekly]
dbl, hlf = f1 > np.log(2), f1 < np.log(0.5)
prof = {}
for lab in ["last year's return", "last 3 months' return", "volatility (wildness)", "size", "distance from 52-week high",
            "delivery % (20-day avg)", "price level (rupees)"] + [k for k in ["earnings yield (cheapness)", "profit growth", "promoter pledging"] if k in traits]:
    x = traits[lab].reindex_like(close).loc[weekly].replace([np.inf, -np.inf], np.nan)
    pr = x.rank(axis=1, pct=True) * 100
    prof[lab] = {"doubled": pr.where(dbl).stack().median(), "halved": pr.where(hlf).stack().median(), "all": 50}
say("(Shown as percentile among stocks that day: 50 = typical, 90 = near the top, 10 = near the bottom.)\n")
table(pd.DataFrame(prof).T, 0)
cnt = pd.DataFrame({"% that doubled next year": dbl.where(f1.notna()).mean(1) * 100, "% that halved": hlf.where(f1.notna()).mean(1) * 100})
table(cnt.groupby(cnt.index.year).mean().T, 1)
if "sec_mom" in p:
    ind = pd.read_csv(os.path.join(DATA_DIR, "extra", "step3", "sectors.csv")).set_index("Symbol")["Industry"]
    d = dbl.stack(); d = d[d].index.get_level_values(1).map(ind).value_counts()
    a = f1.notna().stack(); a = a[a].index.get_level_values(1).map(ind).value_counts()
    sh = (d / a * 100).dropna().sort_values(ascending=False)
    say("Chance (%) that a stock doubled within a year, by industry (stocks with a known industry).\n")
    table(sh.to_frame("% doubled").T, 1)

# ------------------------------------------------------------------ 6. regimes
say("## 6. Do patterns change with market mood?\n")
say("1-month IC of key traits when the market is calm vs fearful (VIX below/above its median), and in rising vs falling markets (Nifty 500 above/below its 200-day average).\n")
calm = (vix <= vix.median()).loc[weekly]
bull = (n500 > n500.rolling(200).mean()).loc[weekly]
rows = {}
for lab in ["last week's return", "last month's return", "last year's return", "volatility (wildness)", "size",
            "distance from 52-week high", "delivery % (20-day avg)"] + [k for k in ["earnings yield (cheapness)"] if k in traits]:
    x = traits[lab].reindex_like(close).loc[weekly].replace([np.inf, -np.inf], np.nan)
    s = ic(x, fwd["1 month"], listed.loc[weekly])
    rows[lab] = {"calm": s[calm.reindex(s.index)].mean(), "fearful": s[~calm.reindex(s.index)].mean(),
                 "rising mkt": s[bull.reindex(s.index)].mean(), "falling mkt": s[~bull.reindex(s.index)].mean()}
table(pd.DataFrame(rows).T)

open("results/explore.md", "w").write("\n".join(OUT) + "\n")
print("DONE explore")
