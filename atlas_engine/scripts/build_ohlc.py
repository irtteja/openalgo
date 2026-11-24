"""Daily OHLC + volume for candlestick charts (last ~2 years, split/bonus adjusted).

    python3 scripts/build_ohlc.py   -> explorer/data/deep/ohlc_NN.json (stocks, same shards as stocks_NN) and ohlc_index.json
"""
import glob, json, os, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel
from rsi.step3_features import S3, skip_index, MIN_DAYS, index_universe, index_name

OUT = "explorer/data/deep"
NSH, DAYS = 48, 500


def r(v, nd):
    return [None if not np.isfinite(x) else round(float(x), nd) for x in v]


p = load_market_panel()
close = p["close"]
idx = close.index[-DAYS:]
dates = [d.strftime("%Y-%m-%d") for d in idx]
# panel prices are adjusted to an arbitrary base; rescale so the latest close equals the real latest price
scale = (p["close_raw"].iloc[-1] / close.iloc[-1])
UNI = index_universe()
shards = {i: {} for i in range(NSH)}
for s in close.columns:
    if s not in UNI:
        continue
    c = close[s].reindex(idx)
    if c.notna().sum() < 20:
        continue
    k = scale.get(s)
    if not np.isfinite(k):
        continue
    nd = 2 if c.iloc[-1] * k < 1000 else 1
    shards[zlib.crc32(s.encode()) % NSH][s] = {
        "o": r(p["open"][s].reindex(idx) * k, nd), "h": r(p["high"][s].reindex(idx) * k, nd), "l": r(p["low"][s].reindex(idx) * k, nd),
        "c": r(c * k, nd), "v": r(p["volume"][s].reindex(idx) / k, 0)}
for i, v in shards.items():
    json.dump({"d": dates, "s": v}, open(os.path.join(OUT, f"ohlc_{i:02d}.json"), "w"), separators=(",", ":"))

ix = {}
for f in glob.glob(os.path.join(S3, "indices", "*.csv")):
    n = index_name(f)
    if skip_index(n) and n != "NIFTY 500":
        continue
    d = pd.read_csv(f)
    if len(d) < MIN_DAYS:
        continue
    d["Date"] = pd.to_datetime(d["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
    d = d.dropna(subset=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
    d = d.reindex(idx)
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=idx)
    cl = num("EOD_CLOSE_INDEX_VAL")
    if cl.notna().sum() < 20:
        continue
    o, h, l = num("EOD_OPEN_INDEX_VAL"), num("EOD_HIGH_INDEX_VAL"), num("EOD_LOW_INDEX_VAL")
    o, h, l = o.where(o > 0, cl), h.where(h > 0, cl), l.where(l > 0, cl)
    ix["index: " + n] = {"o": r(o, 2), "h": r(h, 2), "l": r(l, 2), "c": r(cl, 2)}
json.dump({"d": dates, "s": ix}, open(os.path.join(OUT, "ohlc_index.json"), "w"), separators=(",", ":"))
print("DONE ohlc", sum(len(v) for v in shards.values()), "stocks", len(ix), "indices")

# small weekly closes for mini-charts (last 2 years), all index stocks in one file
wk = list(range(len(idx) - 1, -1, -5))[::-1]
sp = {}
for i in range(NSH):
    for s, v in shards[i].items():
        c = [v["c"][j] for j in wk]
        sp[s] = c
# industry levels (equal weight of every listed stock in the industry, as on the rest of the page), at the same weeks
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
ret, listed = p["ret"], p["close"].notna()
INDS = {}
for name, syms in ind.groupby(ind).groups.items():
    syms = [x for x in syms if x in close.columns]
    if len(syms) >= 3:
        lvl = np.exp(np.log1p(ret[syms].where(listed[syms]).mean(1).fillna(0)).cumsum())
        INDS[name] = r(lvl.reindex(idx).values[wk], 4)
json.dump({"d": [dates[j] for j in wk], "s": sp, "ind": INDS}, open(os.path.join(OUT, "spark.json"), "w"), separators=(",", ":"))
print("spark", len(sp))
