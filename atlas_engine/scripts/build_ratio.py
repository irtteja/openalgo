"""Weekly closes since 2015 for the ratio-chart tool -> explorer/data/deep/ratio_w.json

    python3 scripts/build_ratio.py

Indices: NSE levels. Stocks: split/bonus-adjusted closes scaled so the latest value is the real price.
Industries: equal-weight average of every listed stock in the industry (as elsewhere in the Atlas), base 1000.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel
from rsi.step3_features import _index, S3, skip_index, MIN_DAYS, index_universe, index_name, is_sme

OUT = "explorer/data/deep"
p = load_market_panel()
close, ret = p["close"], p["ret"]
idx = close.index
W = idx[::-1][::5][::-1]
rd = lambda s: [None if not np.isfinite(x) else (round(float(x), 2) if abs(x) >= 10 else round(float(x), 4)) for x in s]
ix = {}
for f in glob.glob(os.path.join(S3, "indices", "*.csv")):
    n = index_name(f)
    if (skip_index(n) and n not in ("NIFTY 500",)) or is_sme(n):
        continue
    s = _index(n, idx)
    if s is not None and s.notna().sum() >= MIN_DAYS:
        ix[n] = rd(np.exp(s.reindex(W)))
scale = p["close_raw"].ffill().iloc[-1] / close.ffill().iloc[-1]
UNI = index_universe()
st = {s: rd(close[s].reindex(W) * scale[s]) for s in close.columns if s in UNI and np.isfinite(scale.get(s, np.nan))}
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
listed = close.notna()
inds = {}
for name, syms in ind.groupby(ind).groups.items():
    syms = [x for x in syms if x in close.columns]
    if len(syms) >= 3:
        inds[name] = rd(1000 * np.exp(np.log1p(ret[syms].where(listed[syms]).mean(1).fillna(0)).cumsum()).reindex(W))
json.dump({"d": [str(d.date()) for d in W], "idx": ix, "stk": st, "ind": inds}, open(os.path.join(OUT, "ratio_w.json"), "w"), separators=(",", ":"))
print("DONE ratio", len(ix), len(st), len(inds), os.path.getsize(os.path.join(OUT, "ratio_w.json")) // 1024, "KB")
