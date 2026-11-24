"""Research experiments on the Atlas data: one question each, tested on history, written to a research log.

    python3 scripts/research.py run E6 E7 E8      # run experiments, append to results/research/
    python3 scripts/research.py list              # experiments available and already run

Each experiment states its question, how it is measured, the result against a fair comparison ("normal"), whether
it held in both halves of history (2016-2020 and 2021-2026), and a verdict: kept, dropped or inconclusive.
Results go to results/research/log.jsonl (one line per run) and results/research/<id>.md (the full report).

Caveat shared by sector experiments: sector breadth and delivery use today's index members over the whole history
(NSE's past member lists are not available), which flatters breadth slightly in older years.
"""
import datetime as dt, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "research")
_CACHE = {}


# ------------------------------------------------------------------ shared data
def base():
    if _CACHE:
        return _CACHE
    from rsi.data import load_market_panel
    from rsi.step3_features import _index, S3, _SECTOR, _THEME
    p = load_market_panel()
    close = p["close"]
    idx = close.index
    logc = np.log(close)
    n500 = _index("NIFTY 500", idx)
    cons = pd.read_csv(os.path.join(S3, "constituents.csv"))
    above50 = (close > close.rolling(50, min_periods=40).mean()).where(close.notna()).astype(float)
    value = p["close_raw"] * p["volume_raw"]
    dlv = p["dlv"]
    groups = {}
    for name in sorted(_SECTOR | _THEME):
        s = _index(name, idx)
        mem = [m for m in cons.loc[cons["index"] == name, "symbol"] if m in close.columns]
        if s is None or s.dropna().empty or s.dropna().index[0] > pd.Timestamp("2016-06-30") or len(mem) < 5:
            continue
        b = above50[mem].mean(axis=1, skipna=True).where(above50[mem].notna().sum(axis=1) >= 5)
        w = value[mem].rolling(20, min_periods=10).mean()
        d = (dlv[mem] * w).sum(axis=1, min_count=5) / w.where(dlv[mem].notna()).sum(axis=1, min_count=5)
        groups[name] = pd.DataFrame({"px": s, "rel": s - n500, "breadth": b, "dlv": d})
    wk = idx[(idx.to_series().dt.weekday == 4) | (idx.to_series().shift(-1).dt.weekday < idx.to_series().dt.weekday)]   # last trading day of each week
    _CACHE.update(idx=idx, n500=n500, groups=groups, weeks=wk, panel=p)
    return _CACHE


def stat(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": round(float(x.mean()), 4), "median": round(float(x.median()), 4), "pos": round(float((x > 0).mean()), 3)}


def halves(df, col):
    a = df[df.index < "2021-01-01"][col]; b = df[df.index >= "2021-01-01"][col]
    return {"2016-2020": stat(a), "2021-2026": stat(b)}


def cooldown(mask, weeks=8):
    """Keep only the first week of each run of signals, then wait `weeks` before counting another (no double counting)."""
    out, last = [], None
    for t, v in mask.items():
        if v and (last is None or (mask.index.get_loc(t) - mask.index.get_loc(last)) >= weeks):
            out.append(t); last = t
    return pd.Index(out)


def weekly_panel():
    B = base(); wk = B["weeks"]; rows = []
    for g, d in B["groups"].items():
        w = d.reindex(wk)
        f = pd.DataFrame(index=wk)
        f["g"] = g
        f["ret4"] = w["px"] - w["px"].shift(4)
        f["rel4"] = w["rel"] - w["rel"].shift(4)
        f["rel13"] = w["rel"] - w["rel"].shift(13)
        f["br"] = w["breadth"]; f["br4"] = w["breadth"] - w["breadth"].shift(4)
        f["dlv4"] = w["dlv"].rolling(4).mean(); f["dlv26"] = w["dlv"].rolling(26).mean().shift(4)
        f["fwd_rel4"] = w["rel"].shift(-4) - w["rel"]
        f["fwd_rel13"] = w["rel"].shift(-13) - w["rel"]
        hi13 = w["px"].rolling(13).max()
        f["breakout8"] = (pd.concat([w["px"].shift(-k) for k in range(1, 9)], axis=1).max(axis=1) > hi13).astype(float)
        f.loc[w["px"].shift(-8).isna(), "breakout8"] = np.nan
        # "turns Improving" within 8 weeks: 4-week relative return turns positive while still behind over 13 weeks
        imp = (f["rel4"] > 0)
        f["improves8"] = pd.concat([imp.shift(-k) for k in range(1, 9)], axis=1).max(axis=1).astype(float)
        f.loc[w["px"].shift(-8).isna(), "improves8"] = np.nan
        rows.append(f[f.index >= "2016-07-01"])
    return pd.concat(rows)


# ------------------------------------------------------------------ experiments
def e6():
    """Breadth rising while price is flat: does the sector break out later?"""
    W = weekly_panel()
    sig = (W["br4"] >= 0.20) & (W["ret4"].abs() <= 0.02)
    ctrl = (W["br4"].abs() < 0.05) & (W["ret4"].abs() <= 0.02)
    ev = pd.concat([W[W["g"] == g].loc[cooldown(sig[W["g"] == g])] for g in W["g"].unique()])
    cv = W[ctrl]
    res = {"events": int(len(ev)), "sectors_with_events": int(ev["g"].nunique()),
           "breakout_8w": {"signal": round(float(ev["breakout8"].mean()), 3), "flat_price_flat_breadth": round(float(cv["breakout8"].mean()), 3),
                           "all_weeks": round(float(W["breakout8"].mean()), 3)},
           "vs_nifty500_4w": {"signal": stat(ev["fwd_rel4"]), "flat_price_flat_breadth": stat(cv["fwd_rel4"]), "all_weeks": stat(W["fwd_rel4"])},
           "vs_nifty500_13w": {"signal": stat(ev["fwd_rel13"]), "flat_price_flat_breadth": stat(cv["fwd_rel13"]), "all_weeks": stat(W["fwd_rel13"])},
           "by_period_13w": halves(ev, "fwd_rel13"), "by_period_breakout": {k: round(float(v), 3) for k, v in ev.groupby(ev.index < "2021-01-01")["breakout8"].mean().rename({True: "2016-2020", False: "2021-2026"}).items()}}
    lift_b = res["breakout_8w"]["signal"] - res["breakout_8w"]["flat_price_flat_breadth"]
    m13 = res["vs_nifty500_13w"]["signal"].get("mean", 0) - res["vs_nifty500_13w"]["flat_price_flat_breadth"].get("mean", 0)
    both = all(v.get("mean", 0) > 0 for v in res["by_period_13w"].values() if v.get("n", 0) >= 10)
    verdict = "kept" if (lift_b > 0.05 and m13 > 0.005 and both) else "dropped" if (lift_b <= 0.02 and m13 < 0.005) else "inconclusive"
    return dict(id="E6", title="Sectors with rising breadth but flat prices",
                question="When most stocks in a sector start rising above their 50-day average while the sector index itself is flat, does the sector break out and beat the market afterwards?",
                method="Sector and theme indices with history since 2016. Signal: share of members above their 50-day average up 20+ points in 4 weeks while the index moved less than ±2%. "
                       "First signal per sector, then 8 weeks' gap. Compared with flat-price weeks where breadth did not move, and with all weeks. "
                       "Outcomes: breaks above its 13-week high within 8 weeks; return vs Nifty 500 over the next 4 and 13 weeks.",
                result=res, verdict=verdict,
                summary=(f"{res['events']} signals in {res['sectors_with_events']} sectors. Broke out within 8 weeks {res['breakout_8w']['signal']:.0%} of the time "
                         f"vs {res['breakout_8w']['flat_price_flat_breadth']:.0%} for flat sectors without the breadth jump. "
                         f"Next 13 weeks vs Nifty 500: {res['vs_nifty500_13w']['signal'].get('mean', 0):+.1%} on average "
                         f"(flat-breadth comparison {res['vs_nifty500_13w']['flat_price_flat_breadth'].get('mean', 0):+.1%})."))


def e7():
    """Heavy delivery buying in a Lagging sector: does it turn Improving?"""
    W = weekly_panel()
    lag = (W["rel13"] < 0) & (W["rel4"] < 0)
    surge = (W["dlv4"] - W["dlv26"])
    # surge = top 20% of that sector's own history up to that week (no look-ahead)
    ranks = []
    for g in W["g"].unique():
        s = surge[W["g"] == g]
        r = s.expanding(52).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
        ranks.append(r)
    W["dlv_rank"] = pd.concat(ranks).values if len(ranks) else np.nan
    sig = lag & (W["dlv_rank"] >= 0.8)
    ctrl = lag & (W["dlv_rank"] < 0.5)
    ev = pd.concat([W[W["g"] == g].loc[cooldown(sig[W["g"] == g])] for g in W["g"].unique()])
    cv = W[ctrl]
    res = {"events": int(len(ev)), "sectors_with_events": int(ev["g"].nunique()),
           "turns_improving_8w": {"signal": round(float(ev["improves8"].mean()), 3), "lagging_low_delivery": round(float(cv["improves8"].mean()), 3),
                                  "all_lagging": round(float(W[lag]["improves8"].mean()), 3)},
           "vs_nifty500_13w": {"signal": stat(ev["fwd_rel13"]), "lagging_low_delivery": stat(cv["fwd_rel13"]), "all_lagging": stat(W[lag]["fwd_rel13"])},
           "by_period_13w": halves(ev, "fwd_rel13")}
    lift = res["turns_improving_8w"]["signal"] - res["turns_improving_8w"]["lagging_low_delivery"]
    m13 = res["vs_nifty500_13w"]["signal"].get("mean", 0) - res["vs_nifty500_13w"]["lagging_low_delivery"].get("mean", 0)
    both = all(v.get("mean", -1) > res["vs_nifty500_13w"]["lagging_low_delivery"].get("mean", 0) for v in res["by_period_13w"].values() if v.get("n", 0) >= 10)
    verdict = "kept" if (lift > 0.05 and m13 > 0.005 and both) else "dropped" if (lift <= 0.02 and m13 <= 0.0) else "inconclusive"
    return dict(id="E7", title="Delivery buying in lagging sectors",
                question="When a sector is lagging the market but its stocks see unusually heavy delivery buying, does it turn 'Improving' soon and do better afterwards?",
                method="Lagging = behind Nifty 500 over both 4 and 13 weeks. Delivery surge = the sector's 4-week average delivery % minus its prior 26-week average, "
                       "in the top 20% of that sector's own history up to that week (no look-ahead). Compared with lagging weeks of low delivery (bottom half). "
                       "Outcomes: turns Improving (4-week relative return positive) within 8 weeks; return vs Nifty 500 over the next 13 weeks.",
                result=res, verdict=verdict,
                summary=(f"{res['events']} signals in {res['sectors_with_events']} sectors. Turned Improving within 8 weeks {res['turns_improving_8w']['signal']:.0%} of the time "
                         f"vs {res['turns_improving_8w']['lagging_low_delivery']:.0%} for lagging sectors with low delivery. "
                         f"Next 13 weeks vs Nifty 500: {res['vs_nifty500_13w']['signal'].get('mean', 0):+.1%} "
                         f"(low-delivery comparison {res['vs_nifty500_13w']['lagging_low_delivery'].get('mean', 0):+.1%})."))


def e8():
    """PSU banks vs private banks: which leads?"""
    from rsi.step3_features import _index
    B = base(); idx = B["idx"]
    psu, pvt = _index("NIFTY PSU BANK", idx), _index("NIFTY PRIVATE BANK", idx)
    d = pd.DataFrame({"psu": psu, "pvt": pvt}).dropna()
    d = d[d.index >= "2016-01-01"]
    wk = d.reindex(B["weeks"]).dropna()
    r = wk.diff().dropna()
    xc = {f"PSU leads by {k}w": round(float(r["psu"].corr(r["pvt"].shift(-k))), 3) for k in (1, 2, 3, 4)}
    xc.update({f"Private leads by {k}w": round(float(r["pvt"].corr(r["psu"].shift(-k))), 3) for k in (1, 2, 3, 4)})
    xc["same week"] = round(float(r["psu"].corr(r["pvt"])), 3)
    # does last week's move of one help predict the other's next week, beyond its own last week? (out-of-sample sign accuracy)
    def oos(target, other):
        X = pd.DataFrame({"own": r[target].shift(1), "other": r[other].shift(1), "y": r[target]}).dropna()
        cut = X.index[X.index >= "2021-01-01"]
        tr, te = X[X.index < "2021-01-01"], X.loc[cut]
        A = np.c_[np.ones(len(tr)), tr[["own", "other"]].values]; beta = np.linalg.lstsq(A, tr["y"].values, rcond=None)[0]
        A0 = np.c_[np.ones(len(tr)), tr[["own"]].values]; b0 = np.linalg.lstsq(A0, tr["y"].values, rcond=None)[0]
        p1 = np.c_[np.ones(len(te)), te[["own", "other"]].values] @ beta; p0 = np.c_[np.ones(len(te)), te[["own"]].values] @ b0
        return {"sign_accuracy_with_other": round(float((np.sign(p1) == np.sign(te["y"])).mean()), 3),
                "sign_accuracy_own_only": round(float((np.sign(p0) == np.sign(te["y"])).mean()), 3), "test_weeks": int(len(te))}
    # turning points: price crossing its 50-day average after 20+ days on the other side
    def turns(s):
        side = np.sign(s - s.rolling(50).mean()).dropna()
        out, run = [], 0
        for i in range(1, len(side)):
            if side.iloc[i] != side.iloc[i - 1] and side.iloc[i] != 0:
                if run >= 20:
                    out.append((side.index[i], int(side.iloc[i])))
                run = 1
            else:
                run += 1
        return out
    tp, tv = turns(d["psu"]), turns(d["pvt"])
    leads = []
    for t, sgn in tv:
        cands = [(abs((t - u).days), (t - u).days) for u, s2 in tp if s2 == sgn and abs((t - u).days) <= 45]
        if cands:
            leads.append(min(cands)[1])   # positive = PSU turned first by that many days
    leads = pd.Series(leads, dtype=float)
    res = {"weekly_correlations": xc, "predict_private_next_week": oos("pvt", "psu"), "predict_psu_next_week": oos("psu", "pvt"),
           "turning_points_matched": int(len(leads)), "psu_turned_first_share": round(float((leads > 0).mean()), 3) if len(leads) else None,
           "median_lead_days_psu_minus_private": float(leads.median()) if len(leads) else None}
    a, b = res["predict_private_next_week"], res["predict_psu_next_week"]
    gain_psu = a["sign_accuracy_with_other"] - a["sign_accuracy_own_only"]; gain_pvt = b["sign_accuracy_with_other"] - b["sign_accuracy_own_only"]
    share = res["psu_turned_first_share"] or 0.5
    leader = "PSU banks" if share >= 0.6 else "private banks" if share <= 0.4 else "neither"
    verdict = "kept" if (leader != "neither" and max(gain_psu, gain_pvt) > 0.02) else "dropped" if leader == "neither" and max(gain_psu, gain_pvt) <= 0.01 else "inconclusive"
    return dict(id="E8", title="PSU banks vs private banks: who leads?",
                question="Do PSU bank stocks move first and private banks follow, or the other way round?",
                method="Nifty PSU Bank and Nifty Private Bank since 2016. (1) Correlation of one index's weekly move with the other's move 1-4 weeks later. "
                       "(2) Whether last week's move of one improves the forecast of the other's next week (fit on 2016-2020, scored on 2021-2026). "
                       "(3) Trend turns (price crossing its 50-day average after 20+ days) matched within 45 days: which turned first.",
                result=res, verdict=verdict,
                summary=(f"Same-week correlation {xc['same week']:.2f}; lagged correlations are small (PSU→private 1w {xc['PSU leads by 1w']:+.2f}, private→PSU 1w {xc['Private leads by 1w']:+.2f}). "
                         f"Adding the other index changed next-week direction accuracy by {gain_psu:+.1%} (private) and {gain_pvt:+.1%} (PSU). "
                         f"Of {res['turning_points_matched']} matched trend turns, PSU banks turned first {share:.0%} of the time "
                         f"(median {res['median_lead_days_psu_minus_private']:+.0f} days). Leader: {leader}."))


EXPERIMENTS = {"E6": e6, "E7": e7, "E8": e8}


def page_data():
    """explorer/data/deep/research.json: the latest run of each experiment, for the Atlas 'Research log' page."""
    p = os.path.join(OUT, "log.jsonl")
    latest = {}
    for line in (open(p, encoding="utf-8") if os.path.exists(p) else []):
        x = json.loads(line); latest[x["id"]] = {k: x.get(k) for k in ("id", "title", "question", "method", "summary", "conclusion", "verdict", "data_to", "run_at")}
    os.makedirs(os.path.join(ROOT, "explorer", "data", "deep"), exist_ok=True)
    with open(os.path.join(ROOT, "explorer", "data", "deep", "research.json"), "w", encoding="utf-8") as f:
        json.dump({"experiments": list(latest.values())}, f)


def write(r):
    os.makedirs(OUT, exist_ok=True)
    r = dict(r, run_at=dt.datetime.now().isoformat(timespec="minutes"), data_to=str(base()["idx"][-1].date()))
    with open(os.path.join(OUT, "log.jsonl"), "a") as f:
        f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, f"{r['id']}.md"), "w") as f:
        f.write(f"# {r['id']}: {r['title']}\n\n**Question.** {r['question']}\n\n**How it was tested.** {r['method']}\n\n"
                f"**Result.** {r['summary']}\n\n**Verdict: {r['verdict']}.**\n\nData to {r['data_to']}, run {r['run_at']}.\n\n"
                f"```json\n{json.dumps(r['result'], indent=1)}\n```\n")
    print(f"{r['id']} [{r['verdict']}] {r['title']}\n  {r['summary']}\n", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        for k in sys.argv[2:]:
            write(EXPERIMENTS[k]())
        page_data()
    elif sys.argv[1:2] == ["page"]:
        page_data()
    else:
        print("available:", ", ".join(EXPERIMENTS))
        p = os.path.join(OUT, "log.jsonl")
        if os.path.exists(p):
            for line in open(p):
                x = json.loads(line); print(x["run_at"], x["id"], x["verdict"], x["title"])
