"""Indices: every NSE stock index with level, returns, valuation and full constituent details.

    /tmp/nslenv/bin/python scripts/build_indices.py fetch     # today's index snapshot (PE/PB/DY) + market caps
    python3 scripts/build_indices.py                          # -> explorer/data/deep/indices.json

Weights are approximate: each member's full market cap / the sum over members. NSE's official weights use
free-float market cap and caps on single stocks, so official numbers differ, most for promoter-heavy companies.
"""
import glob, io, json, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RAW = os.path.join("data", "extra", "step3", "snapshot")
OUT = "explorer/data/deep"


def fetch():
    from datetime import date, timedelta
    from nse_irt import NSE
    nse = NSE()
    os.makedirs(RAW, exist_ok=True)
    r = nse._session.get("allIndices")
    json.dump(r, open(os.path.join(RAW, "allIndices.json"), "w"))
    S = nse._session._s
    d = date.today()
    for _ in range(10):
        u = f"https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{d:%d%m%y}.zip"
        resp = S.get(u, timeout=30)
        if resp.status_code == 200 and zipfile.is_zipfile(io.BytesIO(resp.content)):
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            m = [n for n in z.namelist() if n.lower().startswith("mcap")][0]
            open(os.path.join(RAW, "mcap.csv"), "wb").write(z.read(m))
            print("mcap", d)
            break
        d -= timedelta(days=1)
    print("DONE fetch")


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


def build():
    from rsi.data import load_market_panel
    from rsi.fundamentals import add_fundamental_features
    from rsi.step3_features import _index, S3, skip_index, MIN_DAYS, index_category, is_sme, index_name
    p = add_fundamental_features(load_market_panel())
    close, ret = p["close"], p["ret"]
    idx, cols = close.index, close.columns
    T = idx[-1]
    logc = np.log(close)
    raw = p["close_raw"]
    n500 = _index("NIFTY 500", idx)
    snap = {x["index"]: x for x in json.load(open(os.path.join(RAW, "allIndices.json")))["data"]}
    mc = pd.read_csv(os.path.join(RAW, "mcap.csv"))
    mc.columns = [c.strip() for c in mc.columns]
    mc["Symbol"] = mc["Symbol"].str.strip()
    mc = mc[mc["Series"].str.strip().isin(["EQ", "BE", "BZ"])].drop_duplicates("Symbol").set_index("Symbol")
    mcap = pd.to_numeric(mc["Market Cap(Rs.)"], errors="coerce") / 1e7         # Rs crore
    cname = mc["Security Name"].str.strip()
    ind = pd.read_csv(os.path.join(S3, "sectors.csv")).set_index("Symbol")["Industry"]
    cons = pd.read_csv(os.path.join(S3, "constituents.csv"))
    rs = json.load(open(os.path.join(OUT, "rs.json")))["stocks"]
    ma50, ma200 = close.rolling(50, min_periods=40).mean(), close.rolling(200, min_periods=150).mean()
    hi52, lo52 = raw.rolling(250, min_periods=60).max(), raw.rolling(250, min_periods=60).min()
    ytd0 = idx[idx < pd.Timestamp(f"{T.year}-01-01")]
    ytd0 = ytd0[-1] if len(ytd0) else idx[0]
    dlv = p["dlv"].rolling(20, min_periods=10).mean()
    R = lambda s, n: (s.loc[T] - s.shift(n).loc[T])

    def stock_row(s):
        lc = logc[s]
        r = {"s": s, "name": cname.get(s), "ind": ind.get(s), "mcap": mcap.get(s), "price": raw[s].loc[T],
             "d1": np.expm1(R(lc, 1)), "w1": np.expm1(R(lc, 5)), "m1": np.expm1(R(lc, 21)), "m3": np.expm1(R(lc, 63)),
             "y1": np.expm1(R(lc, 250)), "ytd": np.expm1(lc.loc[T] - lc.loc[ytd0]) if np.isfinite(lc.loc[ytd0]) else None,
             "fromHigh": raw[s].loc[T] / hi52[s].loc[T] - 1, "fromLow": raw[s].loc[T] / lo52[s].loc[T] - 1,
             "above50": bool(close[s].loc[T] > ma50[s].loc[T]) if np.isfinite(ma50[s].loc[T]) else None,
             "above200": bool(close[s].loc[T] > ma200[s].loc[T]) if np.isfinite(ma200[s].loc[T]) else None,
             "dlv": dlv[s].loc[T], "pe": (1 / p["ey"][s].loc[T]) if "ey" in p and p["ey"][s].loc[T] > 0.002 else None}
        x = rs.get(s, {})
        r.update({"rating": x.get("rating"), "quad": x.get("quad"), "secRank": x.get("secRank"), "secN": x.get("secN")})
        return r

    out = []
    names = [index_name(f) for f in glob.glob(os.path.join(S3, "indices", "*.csv"))]
    for n in sorted(names):
        if skip_index(n):
            continue
        L = _index(n, idx)
        if L is None or L.notna().sum() < MIN_DAYS:
            continue
        L = L.ffill()
        sn = snap.get(n, {})
        first = L.first_valid_index()
        yrs = (T - first).days / 365.25
        cagr = lambda y: (np.exp((L.loc[T] - L.loc[idx[idx <= T - pd.Timedelta(days=int(365.25 * y))][-1]]) / y) - 1) if yrs >= y else None
        dd = (L - L.cummax()).min()
        mem = [s for s in cons.loc[cons["index"] == n, "symbol"] if s in cols]
        rows = [stock_row(s) for s in mem if np.isfinite(logc[s].loc[T])]
        tot = np.nansum([r["mcap"] or 0 for r in rows])
        for r in rows:
            r["w"] = (r["mcap"] / tot) if r["mcap"] and tot else None
            r["c1m"] = r["w"] * r["m1"] if r["w"] is not None and r["m1"] is not None and np.isfinite(r["m1"]) else None
            r["c3m"] = r["w"] * r["m3"] if r["w"] is not None and r["m3"] is not None and np.isfinite(r["m3"]) else None
            r["vsIdx3m"] = (np.exp(logc[r["s"]].loc[T] - logc[r["s"]].shift(63).loc[T] - (L.loc[T] - L.shift(63).loc[T])) - 1)
        rows.sort(key=lambda r: -(r["w"] or 0))
        comp = {}
        for r in rows:
            comp[r["ind"] or "Unknown"] = comp.get(r["ind"] or "Unknown", 0) + (r["w"] or 0)
        out.append({
            "g": "index: " + n, "name": n, "cat": index_category("index: " + n), "since": first,
            "level": sn.get("last", float(np.exp(L.loc[T]))), "d1": (sn.get("percentChange") or 0) / 100 if sn else np.expm1(R(L, 1)),
            "w1": np.expm1(R(L, 5)), "m1": np.expm1(R(L, 21)), "m3": np.expm1(R(L, 63)), "m6": np.expm1(R(L, 126)), "y1": np.expm1(R(L, 250)),
            "ytd": np.expm1(L.loc[T] - L.loc[ytd0]) if np.isfinite(L.loc[ytd0]) else None,
            "cagr3": cagr(3), "cagr5": cagr(5), "cagr10": cagr(10),
            "vs3m": np.expm1(R(L, 63) - R(n500, 63)), "vs1y": np.expm1(R(L, 250) - R(n500, 250)),
            "yearHigh": sn.get("yearHigh"), "yearLow": sn.get("yearLow"),
            "fromHigh": (sn["last"] / sn["yearHigh"] - 1) if sn.get("yearHigh") else np.expm1(L.loc[T] - L.iloc[-250:].max()),
            "pe": float(sn["pe"]) if sn.get("pe") not in (None, "", "-") else None, "pb": float(sn["pb"]) if sn.get("pb") not in (None, "", "-") else None,
            "dy": float(sn["dy"]) if sn.get("dy") not in (None, "", "-") else None,
            "adv": int(sn["advances"]) if sn.get("advances") not in (None, "", "-") else None, "dec": int(sn["declines"]) if sn.get("declines") not in (None, "", "-") else None,
            "vol": L.diff().iloc[-250:].std() * np.sqrt(250), "maxDD": dd,
            "n": len(rows), "top10": sum((r["w"] or 0) for r in rows[:10]),
            "breadth50": np.mean([r["above50"] for r in rows if r["above50"] is not None]) if rows else None,
            "breadth200": np.mean([r["above200"] for r in rows if r["above200"] is not None]) if rows else None,
            "upYtd": np.mean([r["ytd"] > 0 for r in rows if r["ytd"] is not None and np.isfinite(r["ytd"])]) if rows else None,
            "comp": sorted(comp.items(), key=lambda x: -x[1]), "members": rows,
        })
    # the other NSE indices (not stock groups) for reference
    others = [{"name": k, "level": v.get("last"), "d1": (v.get("percentChange") or 0) / 100, "y1": (v.get("perChange365d") or 0) / 100,
               "kind": v.get("key")} for k, v in snap.items() if skip_index(k) and not is_sme(k)]
    json.dump(clean({"date": T, "indices": out, "others": others}), open(os.path.join(OUT, "indices.json"), "w"), separators=(",", ":"))
    print("DONE indices", len(out), os.path.getsize(os.path.join(OUT, "indices.json")) // 1024, "KB")


if __name__ == "__main__":
    fetch() if sys.argv[1:] == ["fetch"] else build()
