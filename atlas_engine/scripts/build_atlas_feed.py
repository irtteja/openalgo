"""One compact JSON of the Atlas analysis for other apps (the OpenAlgo Atlas tool): explorer/data/atlas_feed.json.

    python3 scripts/build_atlas_feed.py

Holds: market summary and regime; every sector/theme/industry group with its phase, strength and chances; every index
stock with stage, buying grade, strength, warnings, chances and move breakdown; the screener lists with their record;
hidden groups and driving forces; broad-or-narrow; the research log. Kept small (no price history) so it fits one D1
row (export_d1 stores it in the atlas_feed table) and loads fast.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "explorer", "data")
J = lambda p: json.load(open(os.path.join(EX, p)))
r = lambda x, d=4: None if x is None else round(float(x), d)


def main():
    m, groups, stocks = J("market.json"), J("groups.json"), J("stocks.json")
    screen, rs = J("deep/screen.json"), J("deep/rs.json")
    try:
        S = J("deep/structure.json")
    except OSError:
        S = {}
    try:
        research = J("deep/research.json")["experiments"]
    except OSError:
        research = []
    try:
        br = J("deep/watch.json")["breadth"]["now"]
    except (OSError, KeyError):
        br = {}
    rsg = {x["g"]: x for x in rs["sectors"]}
    rss = {x["s"]: x for x in rs["stocks"]} if isinstance(rs.get("stocks"), list) else {}
    bd = {x["s"]: x for x in S.get("breakdown", {}).get("rows", [])}
    pr = {x["s"]: x.get("pr", {}) for x in stocks}
    feed = {
        "version": 1, "date": m["date"],
        "market": {"regime_flags": m.get("regime", []), "nifty500": r(m.get("n500"), 2), "n500_1m": r(m.get("n500_1m")), "n500_3m": r(m.get("n500_3m")),
                   "n500_1y": r(m.get("n500_1y")), "n500_from_high": r(m.get("n500_fromHigh")), "vix": r(m.get("vix"), 2),
                   "breadth": {k: r((v or {}).get("value")) for k, v in br.items()},
                   "breadth_4w_change": {k: r((v or {}).get("chg4w")) for k, v in br.items()}},
        "regime": {k: v for k, v in S.get("regime", {}).items() if k != "history"},
        "groups": [{"key": g["g"], "name": g["g"].split(": ", 1)[-1], "kind": g["g"].split(":")[0], "category": g.get("cat"), "phase": g.get("phase"),
                    "rs_rating": rsg.get(g["g"], {}).get("rating"), "rs_trend": rsg.get(g["g"], {}).get("trend"),
                    "vs_n500_1m": r(g.get("vs1m"), 2), "vs_n500_3m": r(g.get("vs3m"), 2), "vs_n500_6m": r(g.get("vs6m"), 2), "vs_n500_1y": r(g.get("vs1y"), 2),
                    "from_high": r(g.get("fromHigh"), 2), "pct_above_50d": r(g.get("breadth"), 1), "breadth_change_1m": r(g.get("breadthChg"), 1),
                    "in_base": g.get("base"), "breakout": g.get("breakout"),
                    "p_beat_1m": r(g.get("pr", {}).get("1m|beat")), "p_beat_3m": r(g.get("pr", {}).get("3m|beat")), "p_up_3m": r(g.get("pr", {}).get("3m|up"))}
                   for g in groups],
        "stocks": [{"symbol": x["s"], "industry": x.get("ind"), "price": x.get("price"), "ret_1m": r(x.get("r1m")), "ret_3m": r(x.get("r3m")),
                    "ret_1y": r(x.get("r1y")), "from_high": r(x.get("fromHigh")), "stage": x.get("stage"), "stage_weeks": x.get("stageWeeks"),
                    "grade": x.get("ad"), "rs_rating": x.get("rs"), "rs_4w_ago": x.get("rs4w"), "vs_sector_3m": r(x.get("vsSec3")),
                    "vs_market_3m": r(x.get("vsMkt3")), "sector_phase": x.get("secPhase"), "above_50d": x.get("above50"), "above_200d": x.get("above200"),
                    "in_base": x.get("base"), "breakout": x.get("breakout"), "delivery": r(x.get("dlv")), "pe": r(x.get("pe"), 1),
                    "pledge": x.get("pledge"), "insider_selling": x.get("insSell"), "lists": x.get("presets", []),
                    "p_beat_1m": r(pr.get(x["s"], {}).get("1m|beat")), "p_beat_3m": r(x.get("pBeat3")), "p_big_gain_3m": r(pr.get(x["s"], {}).get("3m|big_up")),
                    "p_big_fall_3m": r(x.get("pFall3")),
                    "breakdown_3m": {k: bd.get(x["s"], {}).get(f"{k}_3m") for k in ("tot", "mkt", "ind", "own")} if x["s"] in bd else None}
                   for x in screen["rows"]],
        "screener": [{"key": p["key"], "name": p["name"], "rule": p["rule"], "avg_vs_n500_3m": r((p.get("3m") or {}).get("mean")),
                      "beat_n500_3m": r((p.get("3m") or {}).get("pos")), "per_week": p.get("perWeek"),
                      "today": [x["s"] for x in screen["rows"] if p["key"] in x.get("presets", [])]} for p in screen["presets"]],
        "forces": S.get("forces", {}).get("items", []),
        "hidden_groups": [{k: v for k, v in g.items() if k != "path"} for g in S.get("groups", {}).get("items", [])],
        "broad_or_narrow": S.get("breadth_of_moves", {}).get("rows", []),
        "research": research,
        "notes": "Odds and history from NSE public data since 2015, not advice. Built by the NSE Pattern Atlas.",
    }
    path = os.path.join(EX, "atlas_feed.json")
    json.dump(feed, open(path, "w"), separators=(",", ":"), allow_nan=False, default=lambda o: None)
    print(f"atlas_feed.json: {os.path.getsize(path) // 1024} KB, {len(feed['stocks'])} stocks, {len(feed['groups'])} groups", flush=True)


if __name__ == "__main__":
    main()
