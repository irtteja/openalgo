"""Sector / theme rotation: which groups are gaining strength, which are building a base, what might rise.

    python3 scripts/rotation.py     -> results/rotation.md, results/rotation_now.csv

Groups: every NSE sector / theme / size index with >= 2 years of history, plus industries built from our
stocks (equal-weight, NSE industry labels). Everything is measured RELATIVE to the Nifty 500.

Phases (like a relative-rotation chart):
  Leading    - above its own relative trend and still gaining
  Weakening  - above its trend but losing momentum
  Lagging    - below its trend and still losing
  Improving  - below its trend but gaining momentum (often the first sign of a turn)
Base / coiling: within 8% of its 52-week high (relative to the market as well as in price) with its
last-month swings much smaller than usual.

History check: for each state, how did the group do vs the Nifty 500 over the next 1 / 3 / 6 months
(weekly samples 2016 -> 2026), and how often did it beat the market over 3 months.
"""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from rsi.data import load_market_panel, DATA_DIR
from rsi.step3_features import _index, S3, skip_index, MIN_DAYS, index_name

OUT = []
def say(s=""): OUT.append(s); print(s, flush=True)
def table(df, nd=2): say(df.round(nd).to_markdown()); say()

p = load_market_panel()
close, ret = p["close"], p["ret"]
idx = close.index
listed = close.notna()
n500 = _index("NIFTY 500", idx)

# ---------------------------------------------------------------- groups
G = {}
for f in glob.glob(os.path.join(S3, "indices", "*.csv")):
    n = index_name(f)
    if skip_index(n):
        continue
    s = _index(n, idx)
    if s is not None and s.notna().sum() >= MIN_DAYS:
        G["idx: " + n.replace("NIFTY ", "").replace("NIFTY", "")] = s
ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
members = {}
for name, syms in ind.groupby(ind).groups.items():
    syms = [s for s in syms if s in close.columns]
    if len(syms) >= 8:
        r = ret[syms].where(listed[syms]).mean(1).fillna(0)
        G["ind: " + name] = np.log1p(r).cumsum().where(listed[syms].sum(1) >= 5)
        members["ind: " + name] = syms
cons = pd.read_csv(os.path.join(S3, "constituents.csv"))
for k in G:
    if k.startswith("idx: "):
        full = [n for n in cons["index"].unique() if n.replace("NIFTY ", "").replace("NIFTY", "") == k[5:]]
        if full:
            members[k] = [s for s in cons.loc[cons["index"] == full[0], "symbol"] if s in close.columns]
L = pd.DataFrame(G)
RS = L.sub(n500, axis=0)                       # log relative strength vs Nifty 500
say(f"# Sector & theme rotation (relative to the Nifty 500)\n\n{L.shape[1]} groups "
    f"({sum(k.startswith('idx') for k in G)} NSE indices, {sum(k.startswith('ind') for k in G)} industries), "
    f"data to {idx[-1].date()}.\n")
say(__doc__.split("\n\n", 2)[2] + "\n")

# ---------------------------------------------------------------- states
ratio = RS - RS.rolling(126, min_periods=100).mean()
mom = ratio.diff(21)
phase = pd.DataFrame(np.select([(ratio > 0) & (mom > 0), (ratio > 0) & (mom <= 0), (ratio <= 0) & (mom <= 0), (ratio <= 0) & (mom > 0)],
                               ["Leading", "Weakening", "Lagging", "Improving"], ""), index=idx, columns=L.columns)
phase = phase.where(ratio.notna() & mom.notna(), "")
d = L.diff()
tight = d.rolling(21, min_periods=15).std() / d.rolling(250, min_periods=150).std()
near_hi = L - L.rolling(250, min_periods=150).max()
rs_near_hi = RS - RS.rolling(250, min_periods=150).max()
base = (near_hi > -0.08) & (rs_near_hi > -0.08) & (tight < 0.75)
breakout = (L >= L.rolling(250, min_periods=150).max()) & base.shift(1).rolling(21, min_periods=1).max().astype(bool)

ma50 = close.rolling(50, min_periods=40).mean()
above50 = (close > ma50).where(listed)
dlv = p["dlv"]
dlv_up = dlv.rolling(20, min_periods=10).mean() - dlv.rolling(120, min_periods=60).mean()
volr = np.log(p["volume"].rolling(20, min_periods=10).mean() / p["volume"].rolling(120, min_periods=60).mean())
B, DL, VO = {}, {}, {}
for k, syms in members.items():
    if len(syms) >= 5:
        B[k] = above50[syms].mean(1)
        DL[k] = dlv_up[syms].mean(1)
        VO[k] = volr[syms].mean(1)
B, DL, VO = (pd.DataFrame(x).reindex(columns=L.columns) for x in (B, DL, VO))
B_chg = B - B.shift(21)

# ---------------------------------------------------------------- history check
wk = idx[(idx >= "2016-06-01")][::5]
F = {h: (RS.shift(-n) - RS) for h, n in [("1m", 21), ("3m", 63), ("6m", 126)]}


def check(mask, label):
    m = mask.reindex(index=wk, columns=L.columns).fillna(False).astype(bool)
    r = {f"next {h} vs mkt %": F[h].loc[wk][m].stack().dropna().mean() * 100 for h in F}
    v = F["3m"].loc[wk][m].stack().dropna()
    r["% beat mkt 3m"] = (v > 0).mean() * 100
    r["cases"] = int(m.sum().sum())
    return label, r


rows = [check(phase == q, f"phase: {q}") for q in ["Leading", "Weakening", "Lagging", "Improving"]]
rows.append(check(pd.DataFrame(True, index=idx, columns=L.columns) & ratio.notna(), "any group (baseline)"))
rk6 = RS.diff(126).rank(axis=1, pct=True)
rows += [check(rk6 > 0.8, "strongest fifth over last 6 months"), check(rk6 <= 0.2, "weakest fifth over last 6 months")]
rk1 = RS.diff(21).rank(axis=1, pct=True)
rows += [check(rk1 > 0.8, "strongest fifth over last month"), check(rk1 <= 0.2, "weakest fifth over last month")]
rows += [check(base, "base / coiling near highs"), check(breakout, "breaking out of a base to a new high")]
rows += [check(near_hi < -0.30, "30%+ below its 52-week high")]
rows += [check((B_chg > 0.20), "breadth jumping (% of members above 50-day avg up 20+ pts in a month)"),
         check(B > 0.8, "80%+ of members above 50-day avg"), check(B < 0.2, "under 20% of members above 50-day avg")]
rows += [check(DL > 0.03, "members' delivery % rising (accumulation)"), check(DL < -0.03, "members' delivery % falling")]
rows += [check(VO > 0.3, "members' volume well above usual")]
rows += [check((phase == "Improving") & (B_chg > 0.1), "Improving + breadth rising"),
         check((phase == "Improving") & base, "Improving + base"),
         check((phase == "Leading") & (DL > 0.02), "Leading + delivery rising"),
         check(base & (DL > 0.02), "base + delivery rising")]
H = pd.DataFrame(dict(rows)).T
say("## 1. What each pattern led to (history 2016 -> 2026)\n")
table(H, 1)

say("## 2. How long do phases last, and what comes next?\n")
ph = phase.loc[wk]
trans = {}
for c in ph.columns:
    s = ph[c][ph[c] != ""]
    s2 = s.shift(-4)  # a month later
    for a, b_ in zip(s.values, s2.values):
        if isinstance(b_, str) and b_:
            trans.setdefault(a, {}).setdefault(b_, 0)
            trans[a][b_] += 1
T = pd.DataFrame(trans).T.fillna(0)
T = T.div(T.sum(1), axis=0)[["Leading", "Weakening", "Lagging", "Improving"]].loc[["Leading", "Weakening", "Lagging", "Improving"]] * 100
say("Chance (%) of being in each phase ONE MONTH later (rows = phase now). The usual cycle is "
    "Improving -> Leading -> Weakening -> Lagging -> Improving.\n")
table(T, 0)

say("## 3. Do sector trends persist? (rank of 6-month strength vs the next 6 months, average over time)\n")
pers = RS.diff(126).loc[wk].rank(axis=1).corrwith(F["6m"].loc[wk].rank(axis=1), axis=1)
yr = pers.groupby(pers.index.year).mean()
table(yr.to_frame("persistence (0 = none, + = trends continue, - = reverse)").T, 2)

# ---------------------------------------------------------------- now
say(f"## 4. Today ({idx[-1].date()}): every group\n")
t = idx[-1]
now = pd.DataFrame({
    "phase": phase.loc[t], "1m vs mkt %": RS.diff(21).loc[t] * 100, "3m vs mkt %": RS.diff(63).loc[t] * 100,
    "6m vs mkt %": RS.diff(126).loc[t] * 100, "1y vs mkt %": RS.diff(250).loc[t] * 100,
    "from 52w high %": near_hi.loc[t] * 100, "base": base.loc[t], "breakout": breakout.loc[t],
    "% members >50d avg": B.loc[t] * 100, "breadth chg 1m": B_chg.loc[t] * 100, "delivery trend": DL.loc[t] * 100,
})
now = now.dropna(subset=["3m vs mkt %"])

# Score = sum of historically-useful signals, weighted by their past 3-month edge vs baseline.
base3 = H.loc["any group (baseline)", "next 3m vs mkt %"]
edge = lambda name: H.loc[name, "next 3m vs mkt %"] - base3
sig_now = {
    "phase: Leading": now["phase"] == "Leading", "phase: Improving": now["phase"] == "Improving",
    "phase: Weakening": now["phase"] == "Weakening", "phase: Lagging": now["phase"] == "Lagging",
    "strongest fifth over last 6 months": rk6.loc[t].reindex(now.index) > 0.8,
    "weakest fifth over last 6 months": rk6.loc[t].reindex(now.index) <= 0.2,
    "base / coiling near highs": now["base"].astype(bool), "breaking out of a base to a new high": now["breakout"].astype(bool),
    "breadth jumping (% of members above 50-day avg up 20+ pts in a month)": now["breadth chg 1m"] > 20,
    "members' delivery % rising (accumulation)": now["delivery trend"] > 3,
    "members' delivery % falling": now["delivery trend"] < -3,
}
score = sum(sig_now[k].fillna(False).astype(float) * edge(k) for k in sig_now)
now["outlook score"] = score
why = [", ".join(k.split(" (")[0].replace("phase: ", "") for k in sig_now if sig_now[k].get(g, False) is True or sig_now[k].get(g, False) == True and k != f"phase: {now.at[g, 'phase']}")
       for g in now.index]
now["signals"] = why
now = now.sort_values("outlook score", ascending=False)
now.to_csv("results/rotation_now.csv")
say("Outlook score = sum of today's signals, each weighted by how much it historically added to the next 3 months "
    "vs the market. Higher = history favours it. It is a tilt in odds, not a promise.\n")
cols_show = ["phase", "1m vs mkt %", "3m vs mkt %", "6m vs mkt %", "from 52w high %", "% members >50d avg", "delivery trend", "outlook score", "signals"]
say("### Most favoured now\n")
table(now[cols_show].head(25), 1)
say("### Least favoured now\n")
table(now[cols_show].tail(15), 1)
say("### Currently building a base near highs (possible breakouts)\n")
table(now[now["base"].astype(bool)][cols_show], 1)
say("### Turning up (Improving phase)\n")
table(now[now["phase"] == "Improving"][cols_show], 1)

# leaders inside the top groups
say("## 5. Strongest stocks inside the most favoured groups\n")
r3 = (np.log(close).diff(63).sub(n500.diff(63), axis=0)).loc[t]
nh = (np.log(close) - np.log(close).rolling(250, min_periods=150).max()).loc[t]
for g in now.index[:8]:
    syms = [s for s in members.get(g, []) if s in r3.index and np.isfinite(r3[s])]
    if not syms:
        continue
    top = r3[syms].sort_values(ascending=False).head(5)
    say(f"- **{g}**: " + ", ".join(f"{s} ({v * 100:+.0f}% vs mkt in 3m, {nh[s] * 100:.0f}% from high)" for s, v in top.items()))
say()
open("results/rotation.md", "w").write("\n".join(OUT) + "\n")
print("DONE rotation")
