"""Pattern discovery with probabilities, in a learn -> test -> relearn loop.

Rows are (stock, week) or (group, week) samples. Each row has
  * conditions  - yes/no facts known at that week's close, e.g. "near 52-week high (top fifth)",
                  "delivery % rising", "VIX high", "sector improving"
  * outcomes    - what happened next over 1 week / 1 month / 3 months / 6 months, relative to the market
A pattern is 1-3 conditions joined by AND. The miner evolves patterns: it scores every single condition,
keeps the useful ones, grows them by adding a condition, mutates them by swapping one, and keeps those
that beat the base rate consistently year after year. A logistic model turns the matching patterns into a
calibrated probability for each outcome.
"""
from __future__ import annotations
import itertools, json, os
import numpy as np
import pandas as pd

HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}
BIG = {"1w": 0.05, "1m": 0.10, "3m": 0.20, "6m": 0.30}       # "big" move vs market, per horizon (log)
OUTCOMES = ("up", "beat", "big_up", "big_down")              # rose / beat market / big gain vs mkt / big fall vs mkt
LABEL = {"up": "rises", "beat": "beats the market", "big_up": "big gain vs market", "big_down": "big fall vs market"}


# ------------------------------------------------------------------------------------------- tables
def _q(df, k=5):
    """Cross-sectional bucket 1..k per date (0 = unknown)."""
    r = df.rank(axis=1, pct=True)
    return np.ceil(r * k).fillna(0).astype(int)


class Table:
    def __init__(self, dates, keys, conds, labels, fwd, fwd_raw, years):
        self.dates, self.keys, self.conds, self.labels = dates, keys, conds, labels
        self.fwd, self.fwd_raw, self.years = fwd, fwd_raw, years
        self.names = list(conds)

    def outcome(self, h, o):
        x = self.fwd[h] if o != "up" else self.fwd_raw[h]
        y = {"up": x > 0, "beat": x > 0, "big_up": x > BIG[h], "big_down": x < -BIG[h]}[o]
        return y.astype(float), np.isfinite(x)


def _stack(frames: dict, dates, cols):
    """dict name -> DataFrame(date x key) sampled on `dates` -> flat arrays in (date, key) order."""
    return {k: v.reindex(index=dates, columns=cols).to_numpy().ravel() for k, v in frames.items()}


def build_stock_table(p: dict, step=5) -> Table:
    close, ret = p["close"], p["ret"]
    idx, cols = close.index, close.columns
    listed = close.notna()
    logc = np.log(close)
    mkt = ret.where(listed).mean(1)
    X = ret.sub(mkt, axis=0).fillna(0).cumsum()
    LR = np.log1p(ret.fillna(0)).cumsum()
    dates = idx[::-1][::step][::-1]                   # anchored on the latest day
    tv = (p["close_raw"] * p["volume_raw"]).rolling(60, min_periods=20).mean()
    vol = p["volume"]
    feats = {
        "last week's return": logc.diff(5), "last month's return": logc.diff(21), "last 3 months' return": logc.diff(63),
        "last year's return": logc.shift(21) - logc.shift(250), "volatility": ret.rolling(20, min_periods=10).std(),
        "size": tv, "distance from 52w high": logc - logc.rolling(250, min_periods=120).max(),
        "distance from 52w low": logc - logc.rolling(250, min_periods=120).min(),
        "delivery %": p["dlv"].rolling(20, min_periods=10).mean(),
        "delivery % change": p["dlv"].rolling(5, min_periods=3).mean() - p["dlv"].rolling(60, min_periods=20).mean(),
        "volume vs usual": np.log(vol.rolling(5, min_periods=3).mean() / vol.rolling(60, min_periods=20).mean()),
        "day range": ((p["high"] - p["low"]) / close).rolling(5, min_periods=3).mean(),
        "share price": p["close_raw"],
    }
    for k, lab in [("ey", "earnings yield (cheapness)"), ("prof_g", "profit growth"), ("rev_g", "sales growth"),
                   ("var", "NSE risk margin"), ("grp_rel20", "return vs its group (month)"),
                   ("grp_rel5", "return vs its group (week)"), ("grp_mom", "its group's 3-month trend"),
                   ("sec_mom", "its sector's 3-month trend"), ("mtf", "margin-funded buying")]:
        if k in p:
            feats[lab] = p[k]
    conds = {}
    for name, f in feats.items():
        q = _q(f.reindex(index=dates)).reindex(columns=cols).to_numpy().ravel()
        conds[f"{name}: top fifth"] = q == 5
        conds[f"{name}: bottom fifth"] = q == 1
        conds[f"{name}: high (top 40%)"] = q >= 4
        conds[f"{name}: low (bottom 40%)"] = (q >= 1) & (q <= 2)
    flags = {}
    if "pledge" in p:
        flags["promoter pledging recently"] = p["pledge"] > 0
        flags["promoter released pledges recently"] = p["pledge"] < 0
    if "ins_net" in p:
        flags["insiders net buying (60d)"] = p["ins_net"] > 0.001
        flags["insiders net selling (60d)"] = p["ins_net"] < -0.001
    if "prom_chg" in p:
        flags["promoter stake rose last filing"] = p["prom_chg"] > 0.25
        flags["promoter stake fell last filing"] = p["prom_chg"] < -0.25
    hi = logc >= logc.rolling(250, min_periods=120).max()
    flags["made a new 52w high this week"] = hi.rolling(5, min_periods=1).max().astype(bool)
    lo = logc <= logc.rolling(250, min_periods=120).min()
    flags["made a new 52w low this week"] = lo.rolling(5, min_periods=1).max().astype(bool)
    flags["above 200-day average"] = close > close.rolling(200, min_periods=150).mean()
    flags["below 200-day average"] = close < close.rolling(200, min_periods=150).mean()
    ma50, ma200 = close.rolling(50, min_periods=40).mean(), close.rolling(200, min_periods=150).mean()
    flags["50-day avg crossed above 200-day (last month)"] = ((ma50 > ma200) & (ma50.shift(21) <= ma200.shift(21)))
    flags["jumped 10%+ in a day this week"] = (ret > 0.10).rolling(5, min_periods=1).max().astype(bool)
    flags["fell 10%+ in a day this week"] = (ret < -0.10).rolling(5, min_periods=1).max().astype(bool)
    flags["listed less than a year"] = listed.cumsum() < 250
    for k, v in flags.items():
        conds[k] = v.reindex(index=dates, columns=cols).fillna(False).to_numpy(dtype=bool).ravel()
    conds.update(_regime(p, idx, dates, len(cols)))
    lst = listed.reindex(dates).to_numpy().ravel()
    fwd, fwd_raw = {}, {}
    for h, n in HORIZONS.items():
        ok = listed.shift(-n, fill_value=False)
        fwd[h] = (X.shift(-n) - X).where(ok & listed).reindex(dates).to_numpy().ravel()
        fwd_raw[h] = (LR.shift(-n) - LR).where(ok & listed).reindex(dates).to_numpy().ravel()
    keep = lst.astype(bool)
    D = np.repeat(dates.values, len(cols))[keep]
    K = np.tile(np.asarray(cols), len(dates))[keep]
    return Table(pd.DatetimeIndex(D), K, {k: v[keep] for k, v in conds.items()}, None,
                 {h: v[keep] for h, v in fwd.items()}, {h: v[keep] for h, v in fwd_raw.items()},
                 pd.DatetimeIndex(D).year.to_numpy())


def _regime(p, idx, dates, ncols):
    """Market-wide conditions (same for every row on a date)."""
    from .step3_features import _index
    from .data import DATA_DIR
    out = {}
    n500 = _index("NIFTY 500", idx)
    vixp = os.path.join(DATA_DIR, "extra", "vix.csv")
    if os.path.exists(vixp):
        vix = pd.read_csv(vixp, parse_dates=["Date"]).set_index("Date")["vix"].reindex(idx).ffill()
        pct = vix.rolling(750, min_periods=120).rank(pct=True)            # vs its own past 3 years
        out["market: VIX high (fear)"] = pct > 0.67
        out["market: VIX low (calm)"] = pct < 0.33
    if n500 is not None:
        out["market: Nifty 500 above 200-day avg"] = n500 > n500.rolling(200, min_periods=150).mean()
        out["market: Nifty 500 below 200-day avg"] = n500 < n500.rolling(200, min_periods=150).mean()
        out["market: fell 10%+ from its high"] = (n500 - n500.rolling(250, min_periods=60).max()) < -0.10
        sml, big = _index("NIFTY SMALLCAP 250", idx), _index("NIFTY 50", idx)
        if sml is not None and big is not None:
            sb = sml.diff(63) - big.diff(63)
            out["market: small caps beating large (3m)"] = sb > 0.03
            out["market: large caps beating small (3m)"] = sb < -0.03
    close = p["close"]
    br = (close > close.rolling(200, min_periods=150).mean()).where(close.notna()).astype(float).mean(1)
    out["market: most stocks above 200-day avg (>60%)"] = br > 0.6
    out["market: few stocks above 200-day avg (<35%)"] = br < 0.35
    m = pd.Series(idx.month, index=idx)
    out["calendar: Jan-Mar"] = m.isin([1, 2, 3])
    out["calendar: Apr-Jul"] = m.isin([4, 5, 6, 7])
    out["calendar: Oct-Dec"] = m.isin([10, 11, 12])
    return {k: np.repeat(v.reindex(dates).fillna(False).to_numpy(dtype=bool), ncols) for k, v in out.items()}


def build_group_table(p: dict, step=5) -> Table:
    """Rows = (sector/theme index or industry, week). Outcomes relative to the Nifty 500."""
    import glob
    from .step3_features import _index, S3, skip_index, MIN_DAYS, index_name
    close, ret = p["close"], p["ret"]
    idx = close.index
    listed = close.notna()
    n500 = _index("NIFTY 500", idx)
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
        syms = [s for s in syms if s in close.columns]
        if len(syms) >= 8:
            r = ret[syms].where(listed[syms]).mean(1).fillna(0)
            G["industry: " + name] = np.log1p(r).cumsum().where(listed[syms].sum(1) >= 5)
            members["industry: " + name] = syms
    cons = pd.read_csv(os.path.join(S3, "constituents.csv"))
    for k in G:
        if k.startswith("index: "):
            members[k] = [s for s in cons.loc[cons["index"] == k[7:], "symbol"] if s in close.columns]
    L = pd.DataFrame(G)
    RS = L.sub(n500, axis=0)
    cols = L.columns
    dates = idx[::-1][::step][::-1]
    ratio = RS - RS.rolling(126, min_periods=100).mean()
    mom = ratio.diff(21)
    d = L.diff()
    tight = d.rolling(21, min_periods=15).std() / d.rolling(250, min_periods=150).std()
    nh = L - L.rolling(250, min_periods=150).max()
    rsnh = RS - RS.rolling(250, min_periods=150).max()
    base = (nh > -0.08) & (rsnh > -0.08) & (tight < 0.75)
    above50 = (close > close.rolling(50, min_periods=40).mean()).where(listed)
    dlvu = p["dlv"].rolling(20, min_periods=10).mean() - p["dlv"].rolling(120, min_periods=60).mean()
    B, DL = {}, {}
    for k, syms in members.items():
        if len(syms) >= 5:
            B[k], DL[k] = above50[syms].mean(1), dlvu[syms].mean(1)
    B, DL = pd.DataFrame(B).reindex(columns=cols), pd.DataFrame(DL).reindex(columns=cols)
    feats = {"group vs market, last month": RS.diff(21), "group vs market, last 3 months": RS.diff(63),
             "group vs market, last 6 months": RS.diff(126), "group vs market, last year": RS.diff(250),
             "group distance from 52w high": nh, "group swings vs usual (tightness)": tight,
             "share of members above 50-day avg": B, "members' delivery trend": DL}
    conds = {}
    for name, f in feats.items():
        q = _q(f.reindex(index=dates)).reindex(columns=cols).to_numpy().ravel()
        conds[f"{name}: top fifth"] = q == 5
        conds[f"{name}: bottom fifth"] = q == 1
        conds[f"{name}: high (top 40%)"] = q >= 4
        conds[f"{name}: low (bottom 40%)"] = (q >= 1) & (q <= 2)
    flags = {"phase: Leading": (ratio > 0) & (mom > 0), "phase: Weakening": (ratio > 0) & (mom <= 0),
             "phase: Lagging": (ratio <= 0) & (mom <= 0), "phase: Improving": (ratio <= 0) & (mom > 0),
             "building a base near its high": base,
             "broke out of a base to a new high": (L >= L.rolling(250, min_periods=150).max()) & base.shift(1).rolling(21, min_periods=1).max().astype(bool),
             "group at a new 52w high": nh >= 0, "group 20%+ below its high": nh < -0.20,
             "breadth rose 20+ pts in a month": (B - B.shift(21)) > 0.20}
    for k, v in flags.items():
        conds[k] = v.reindex(index=dates, columns=cols).fillna(False).to_numpy(dtype=bool).ravel()
    conds.update(_regime(p, idx, dates, len(cols)))
    have = RS.notna().reindex(dates).to_numpy().ravel()
    fwd, fwd_raw = {}, {}
    for h, n in HORIZONS.items():
        fwd[h] = (RS.shift(-n) - RS).reindex(dates).to_numpy().ravel()
        fwd_raw[h] = (L.shift(-n) - L).reindex(dates).to_numpy().ravel()
    D = np.repeat(dates.values, len(cols))[have]
    K = np.tile(np.asarray(cols), len(dates))[have]
    t = Table(pd.DatetimeIndex(D), K, {k: v[have] for k, v in conds.items()}, None,
              {h: v[have] for h, v in fwd.items()}, {h: v[have] for h, v in fwd_raw.items()},
              pd.DatetimeIndex(D).year.to_numpy())
    t.members = members
    return t


# ------------------------------------------------------------------------------------------- mining
def _stats(mask, y, ok, years, yr_list):
    m = mask & ok
    n = int(m.sum())
    if n == 0:
        return n, np.nan, None
    by = np.bincount(years[m] - yr_list[0], weights=y[m], minlength=len(yr_list))
    cnt = np.bincount(years[m] - yr_list[0], minlength=len(yr_list))
    return n, float(y[m].mean()), (by, cnt)


class Miner:
    """Evolves AND-patterns of up to 3 conditions for one (table, horizon, outcome)."""

    def __init__(self, table: Table, h: str, o: str, min_rows=2500, min_years=4, max_depth=3):
        self.t, self.h, self.o = table, h, o
        self.min_rows, self.min_years, self.max_depth = min_rows, min_years, max_depth

    def score(self, combo, train, y, ok, base, base_y):
        mask = train.copy()
        for c in combo:
            mask &= self.t.conds[c]
        n, p, yr = _stats(mask, y, ok, self.t.years, self.yrs)
        if n < self.min_rows or yr is None:
            return None
        by, cnt = yr
        have = cnt >= 100
        if have.sum() < self.min_years:
            return None
        lift_y = by[have] / cnt[have] - base_y[have]
        lift = p - base
        same = np.mean(np.sign(lift_y) == np.sign(lift))          # share of years agreeing with the overall direction
        # shrink toward the base rate when evidence is thin; reward consistency across years
        se = np.sqrt(max(p * (1 - p), 1e-6) / (n / max(1, HORIZONS[self.h] / 5)))
        z = lift / se
        return {"conds": list(combo), "n": n, "p": p, "base": base, "lift": lift, "years_agree": same,
                "years": int(have.sum()), "z": z, "fitness": abs(lift) * same * min(1.0, abs(z) / 4)}

    def run(self, train, keep=60, rng=None):
        rng = rng or np.random.default_rng(0)
        y, ok = self.t.outcome(self.h, self.o)
        ok = ok & train
        self.yrs = np.arange(self.t.years.min(), self.t.years.max() + 1)
        base = float(y[ok].mean())
        cnt = np.bincount(self.t.years[ok] - self.yrs[0], minlength=len(self.yrs))
        base_y = np.bincount(self.t.years[ok] - self.yrs[0], weights=y[ok], minlength=len(self.yrs)) / np.maximum(cnt, 1)
        seen, pool = {}, []

        def add(combo):
            key = tuple(sorted(combo))
            if key in seen or len(set(key)) < len(key):
                return
            s = self.score(key, train, y, ok, base, base_y)
            seen[key] = s
            if s and s["years_agree"] >= 0.7:
                pool.append(s)

        for c in self.t.names:                                       # generation 1: single conditions
            add((c,))
        for depth in range(2, self.max_depth + 1):                   # grow the fittest, then mutate
            parents = sorted([s for s in pool if len(s["conds"]) == depth - 1], key=lambda s: -s["fitness"])[:keep]
            for s in parents:
                for c in self.t.names:
                    if c not in s["conds"]:
                        child = tuple(s["conds"]) + (c,)
                        sc_child = self.score(tuple(sorted(child)), train, y, ok, base, base_y) if tuple(sorted(child)) not in seen else seen[tuple(sorted(child))]
                        seen[tuple(sorted(child))] = sc_child
                        # keep a child only if it sharpens the parent in the same direction
                        if sc_child and sc_child["years_agree"] >= 0.7 and abs(sc_child["lift"]) > abs(s["lift"]) * 1.15 \
                                and np.sign(sc_child["lift"]) == np.sign(s["lift"]):
                            pool.append(sc_child)
            for s in sorted(pool, key=lambda s: -s["fitness"])[:keep // 2]:  # mutation: swap one condition
                if len(s["conds"]) > 1:
                    for _ in range(5):
                        new = list(s["conds"]); new[rng.integers(len(new))] = self.t.names[rng.integers(len(self.t.names))]
                        add(tuple(new))
        uniq = {}
        for s in pool:
            uniq[tuple(sorted(s["conds"]))] = s
        pool = list(uniq.values())
        pos = sorted([s for s in pool if s["lift"] > 0], key=lambda s: -s["fitness"])[:keep]
        neg = sorted([s for s in pool if s["lift"] < 0], key=lambda s: -s["fitness"])[:keep]
        for s in pos + neg:
            s.update(h=self.h, o=self.o)
        return pos + neg, base


def pattern_matrix(table: Table, patterns, rows=None):
    rows = np.arange(len(table.keys)) if rows is None else rows
    M = np.zeros((len(rows), len(patterns)), dtype=np.float32)
    for j, s in enumerate(patterns):
        m = np.ones(len(rows), bool)
        for c in s["conds"]:
            m &= table.conds[c][rows]
        M[:, j] = m
    return M


def feature_matrix(table, patterns, rows):
    """Every single condition + every learned pattern, as 0/1 columns."""
    S = np.stack([table.conds[c][rows] for c in table.names], 1).astype(np.float32)
    return np.hstack([S, pattern_matrix(table, patterns, rows)])


def fit_prob_model(table, patterns, train, h, o):
    from sklearn.linear_model import LogisticRegression
    y, ok = table.outcome(h, o)
    rows = np.flatnonzero(train & ok)
    if len(rows) > 400_000:
        rows = np.random.default_rng(1).choice(rows, 400_000, replace=False)
    model = LogisticRegression(C=0.02, max_iter=500)
    model.fit(feature_matrix(table, patterns, rows), y[rows])
    return model


def calibration(pred, y):
    """Predicted vs actual frequency in 5 bins of predicted probability."""
    q = pd.qcut(pred, 5, duplicates="drop")
    g = pd.DataFrame({"pred": pred, "y": y}).groupby(q, observed=True).mean()
    return g


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(path, "w"), indent=1, default=conv)
