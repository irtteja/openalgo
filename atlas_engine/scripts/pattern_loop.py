"""Learn -> test -> relearn loop for patterns and probabilities.

    python3 scripts/pattern_loop.py backtest   # honest check: learn up to each year, predict that year (2020 -> now)
    python3 scripts/pattern_loop.py live       # learn on everything, re-check the old library, report today

Outputs in results/patterns/:
    backtest.md               - did "60%" really happen ~60% of the time? per horizon and outcome
    library_stocks.json       - current pattern library for stocks (+ groups), with history of each pattern
    report.md                 - today's view: probabilities for every stock and sector, and why
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd

from rsi.data import load_market_panel
from rsi.extra import add_extra_features
from rsi.fundamentals import add_fundamental_features
from rsi.market_features import add_market_features
from rsi.corp_features import add_corp_features
from rsi.step3_features import add_step3_features
from rsi.patterns import (HORIZONS, OUTCOMES, LABEL, BIG, Miner, build_stock_table, build_group_table,
                          fit_prob_model, pattern_matrix, feature_matrix, calibration, save_json)

OUT = "results/patterns"
os.makedirs(OUT, exist_ok=True)


def panel():
    p = load_market_panel()
    for f in (add_extra_features, add_fundamental_features, add_market_features, add_corp_features, add_step3_features):
        p = f(p)
    return p


def known_before(t, cutoff, h):
    """Rows whose outcome window ends before `cutoff` (no peeking into the test period)."""
    gap = pd.Timedelta(days=int(HORIZONS[h] * 1.5) + 3)
    return (t.dates + gap) < cutoff


def learn(t, train_for, keep):
    lib, models, bases = {}, {}, {}
    for h in HORIZONS:
        for o in OUTCOMES:
            tr = train_for(h)
            pats, base = Miner(t, h, o).run(tr, keep=keep)
            lib[(h, o)], bases[(h, o)] = pats, base
            models[(h, o)] = fit_prob_model(t, pats, tr, h, o) if pats else None
    return lib, models, bases


def predict(t, lib, models, bases, rows):
    out = {}
    for k, pats in lib.items():
        if models[k] is None:
            out[k] = np.full(len(rows), bases[k])
        else:
            out[k] = models[k].predict_proba(feature_matrix(t, pats, rows))[:, 1]
    return out


def backtest():
    lines = ["# Backtest: are the probabilities honest?\n",
             "For each test year the system learned ONLY from earlier data, then predicted that year. "
             "'predicted' = average probability the system gave; 'actual' = how often it really happened. "
             "Good = the two columns match and rise together. 'spread' = actual rate in the most-likely fifth "
             "minus the least-likely fifth (bigger = more useful).\n"]
    p = panel()
    for name, build, keep in [("stocks", build_stock_table, 30), ("sectors & themes", build_group_table, 40)]:
        t = build(p)
        allp, ally = {k: [] for k in [(h, o) for h in HORIZONS for o in OUTCOMES]}, {k: [] for k in [(h, o) for h in HORIZONS for o in OUTCOMES]}
        for Y in range(2020, t.dates.max().year + 1):
            cut = pd.Timestamp(f"{Y}-01-01")
            t0 = time.time()
            lib, models, bases = learn(t, lambda h: known_before(t, cut, h), keep)
            rows = np.flatnonzero(t.years == Y)
            pr = predict(t, lib, models, bases, rows)
            for (h, o), v in pr.items():
                y, ok = t.outcome(h, o)
                m = ok[rows]
                allp[(h, o)].append(v[m]); ally[(h, o)].append(y[rows][m])
            print(f"{name} {Y}: learned {sum(len(v) for v in lib.values())} patterns in {time.time() - t0:.0f}s", flush=True)
        lines.append(f"## {name}\n")
        summ = {}
        for (h, o) in allp:
            pp, yy = np.concatenate(allp[(h, o)]), np.concatenate(ally[(h, o)])
            if len(pp) < 100:
                continue
            c = calibration(pp, yy)
            summ[f"{h}: {LABEL[o]}"] = {"base rate %": yy.mean() * 100, "least-likely fifth: predicted %": c["pred"].iloc[0] * 100,
                                        "actual %": c["y"].iloc[0] * 100, "most-likely fifth: predicted %": c["pred"].iloc[-1] * 100,
                                        "actual % ": c["y"].iloc[-1] * 100, "spread (pts)": (c["y"].iloc[-1] - c["y"].iloc[0]) * 100}
        lines.append(pd.DataFrame(summ).T.round(1).to_markdown() + "\n")
        open(os.path.join(OUT, "backtest.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


def live():
    p = panel()
    stamp = str(p["close"].index[-1].date())
    report = [f"# What the data says now ({stamp})\n",
              "Probabilities come from patterns the system found in 2015 -> now and keeps re-checking. "
              "'normal' = how often it happens to any stock/sector. They are odds, not certainties.\n"]
    for name, build, keep in [("stocks", build_stock_table, 40), ("groups", build_group_table, 40)]:
        t = build(p)
        lib, models, bases = learn(t, lambda h: np.isfinite(t.fwd[h]), keep)
        # ---- relearn bookkeeping: compare with the previous library
        path = os.path.join(OUT, f"library_{name}.json")
        old = json.load(open(path)) if os.path.exists(path) else {"patterns": []}
        new_keys = {(s["h"], s["o"], tuple(s["conds"])) for v in lib.values() for s in v}
        old_keys = {(s["h"], s["o"], tuple(s["conds"])) for s in old["patterns"]}
        recent_cut = t.dates.max() - pd.Timedelta(days=365)
        checks = []
        for s in old["patterns"]:                      # how did last run's patterns do on the latest year?
            h, o = s["h"], s["o"]
            y, ok = t.outcome(h, o)
            m = ok & (t.dates > recent_cut - pd.Timedelta(days=int(HORIZONS[h] * 1.5)))
            for c in s["conds"]:
                m &= t.conds.get(c, np.zeros_like(m))
            base_recent = y[ok & (t.dates > recent_cut)].mean() if (ok & (t.dates > recent_cut)).any() else np.nan
            if m.sum() >= 50:
                checks.append({"pattern": " + ".join(s["conds"]), "h": h, "o": o, "was": s["lift"] * 100,
                               "latest year": (y[m].mean() - base_recent) * 100,
                               "still works": np.sign(y[m].mean() - base_recent) == np.sign(s["lift"])})
        hist = old.get("history", []) + [{"date": stamp, "patterns": len(new_keys), "new": len(new_keys - old_keys),
                                          "retired": len(old_keys - new_keys),
                                          "old ones still working %": (np.mean([c["still works"] for c in checks]) * 100) if checks else None}]
        save_json({"date": stamp, "patterns": [s for v in lib.values() for s in v], "history": hist}, path)
        report.append(f"## {name.capitalize()}: learning loop\n")
        report.append(pd.DataFrame(hist).to_markdown(index=False) + "\n")
        if checks:
            ck = pd.DataFrame(checks)
            report.append(f"Of last run's patterns, {ck['still works'].mean() * 100:.0f}% still pointed the right way over the latest year.\n")
        # ---- strongest patterns, in words
        report.append(f"## {name.capitalize()}: strongest patterns found\n")
        rows = []
        for (h, o), pats in lib.items():
            for s in pats[:3] + [x for x in pats if x["lift"] < 0][:2]:
                rows.append({"when": " + ".join(s["conds"]), "then": f"{LABEL[o]} ({h})", "chance %": s["p"] * 100,
                             "normal %": s["base"] * 100, "cases": s["n"], "years it held": f"{s['years_agree'] * s['years']:.0f}/{s['years']}"})
        rp = pd.DataFrame(rows).drop_duplicates(["when", "then"])
        rp["gap"] = (rp["chance %"] - rp["normal %"]).abs()
        report.append(rp.sort_values("gap", ascending=False).drop(columns="gap").head(40).round(1).to_markdown(index=False) + "\n")
        # ---- today
        last = t.dates.max()
        rows_now = np.flatnonzero(t.dates == last)
        pr = predict(t, lib, models, bases, rows_now)
        now = pd.DataFrame({f"{h} {LABEL[o]} %": pr[(h, o)] * 100 for h in HORIZONS for o in OUTCOMES}, index=t.keys[rows_now])
        normal = {f"{h} {LABEL[o]} %": bases[(h, o)] * 100 for h in HORIZONS for o in OUTCOMES}
        now.to_csv(os.path.join(OUT, f"today_{name}.csv"))
        flat = [dict(s, id=i) for i, s in enumerate(x for v in lib.values() for x in v)]
        Mall = pattern_matrix(t, flat, rows_now)
        S = {c: t.conds[c][rows_now] for c in t.names}
        detail = {"date": stamp, "normal": {f"{h}|{o}": bases[(h, o)] for h in HORIZONS for o in OUTCOMES},
                  "patterns": [{"id": s["id"], "conds": s["conds"], "h": s["h"], "o": s["o"], "p": s["p"], "base": s["base"],
                                "n": s["n"], "years": s["years"], "agree": s["years_agree"]} for s in flat],
                  "rows": {}}
        for i, k in enumerate(t.keys[rows_now]):
            detail["rows"][str(k)] = {"prob": {f"{h}|{o}": float(pr[(h, o)][i]) for h in HORIZONS for o in OUTCOMES},
                                      "match": [int(j) for j in np.flatnonzero(Mall[i])],
                                      "facts": [c for c in t.names if S[c][i] and not c.startswith(("market:", "calendar:"))]}
        detail["market_now"] = [c for c in t.names if c.startswith(("market:", "calendar:")) and t.conds[c][rows_now].any()]
        if hasattr(t, "members"):
            detail["members"] = {k: v for k, v in t.members.items()}
        save_json(detail, os.path.join(OUT, f"detail_{name}.json"))
        why = {}
        M = pattern_matrix(t, lib[("3m", "beat")], rows_now)
        for i, k in enumerate(now.index):
            hits = [lib[("3m", "beat")][j] for j in np.flatnonzero(M[i])]
            hits = sorted(hits, key=lambda s: -abs(s["lift"]))[:3]
            why[k] = "; ".join(("+ " if s["lift"] > 0 else "- ") + " & ".join(s["conds"]) for s in hits)
        now["why (3m)"] = pd.Series(why)
        show = ["1m beats the market %", "3m beats the market %", "3m big gain vs market %", "3m big fall vs market %",
                "6m beats the market %", "why (3m)"]
        report.append(f"## {name.capitalize()}: today\n")
        report.append("Normal rates: " + ", ".join(f"{k.replace(' %', '')} {normal[k]:.0f}%" for k in show[:-1]) + "\n")
        if name == "stocks":
            liq = (p["close_raw"] * p["volume_raw"]).rolling(60, min_periods=20).mean().iloc[-1]
            big = liq.reindex(now.index) > 5e7      # trades > Rs 5 crore a day on average
            sub = now[big]
            report.append(f"### Most favoured over 3 months (stocks trading > Rs 5 crore/day: {len(sub)} of {len(now)})\n")
            report.append(sub.sort_values("3m beats the market %", ascending=False)[show].head(30).round(0).to_markdown() + "\n")
            report.append("### Least favoured over 3 months\n")
            report.append(sub.sort_values("3m beats the market %")[show].head(20).round(0).to_markdown() + "\n")
            report.append("### Highest chance of a big fall over 3 months\n")
            report.append(sub.sort_values("3m big fall vs market %", ascending=False)[show].head(15).round(0).to_markdown() + "\n")
        else:
            report.append(now.sort_values("3m beats the market %", ascending=False)[show].round(0).to_markdown() + "\n")
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(report))
    print("DONE live", flush=True)


if __name__ == "__main__":
    {"backtest": backtest, "live": live}[sys.argv[1]]()
