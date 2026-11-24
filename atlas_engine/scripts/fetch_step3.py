"""Step-3 data: sectors, index levels, margin trading, options open interest.

Run from the irtteja/nselib virtualenv (needs pandas there too):
    python3 scripts/fetch_step3.py sectors                  # (PyPI nselib) industry of each stock (current NSE index lists)
    /tmp/nslenv/bin/python scripts/fetch_step3.py indices    # daily Nifty 50/500 + sector/style indices, 2015 ->
    /tmp/nslenv/bin/python scripts/fetch_step3.py all_indices  # every index NSE lists
    /tmp/nslenv/bin/python scripts/fetch_step3.py mtf        # margin trading per stock, daily (NSE has 2022 ->)
    /tmp/nslenv/bin/python scripts/fetch_step3.py options    # per-stock call/put OI from the F&O bhavcopy, 2016 ->

What NSE does NOT have: daily FII/DII cash history (only today's figure) and past index membership.
Output: data/extra/step3/... Safe to re-run: finished days are skipped.
"""
import io, os, sys, time, zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "extra", "step3")
ARCH = "https://nsearchives.nseindia.com"
if sys.argv[1:] != ["sectors"]:  # sectors uses the PyPI nselib (system python)
    from nse_irt import NSE
    nse = NSE()
    S = nse._session._s


def warm():
    try:
        S.get("https://www.nseindia.com/all-reports", timeout=20)
    except Exception:
        pass


def fetch(url, tries=3):
    for i in range(tries):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 200 and not r.content.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
                return r.content
        except Exception:
            pass
        time.sleep(3 * (i + 1)); warm()
    return None


def unzip(b):
    if not zipfile.is_zipfile(io.BytesIO(b)):
        return b  # a few days are published as plain CSV
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        names = z.namelist()
        return z.read(names[0]) if names else b""


def _ist_last_day():
    import datetime as _dt
    ist = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=5, minutes=30)))
    return ist.date() if ist.hour * 60 + ist.minute >= 18 * 60 else ist.date() - timedelta(days=1)   # NSE posts the day's files by ~6 pm IST


def _empty(p, day):
    """Mark a day as having no file (holiday) - only once it is a few days old, so a late NSE file is retried."""
    if (date.today() - day).days > 3:
        open(p, "w").close()


def days(start):
    d = start
    last = _ist_last_day()
    while d <= last:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def sectors():
    from nselib import capital_market as cm
    parts = []
    for fn in ["nifty50_equity_list", "niftynext50_equity_list", "niftymidcap150_equity_list", "niftysmallcap250_equity_list"]:
        parts.append(getattr(cm, fn)()[["Symbol", "Industry"]])
    os.makedirs(OUT, exist_ok=True)
    df = pd.concat(parts).drop_duplicates("Symbol")
    df.to_csv(os.path.join(OUT, "sectors.csv"), index=False)
    print(df["Industry"].value_counts().to_string())


INDICES = ["NIFTY 50", "NIFTY 500", "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY AUTO", "NIFTY FMCG",
           "NIFTY METAL", "NIFTY REALTY", "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY PSU BANK", "NIFTY FINANCIAL SERVICES",
           "NIFTY INFRASTRUCTURE", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 250",
           # style ("factor") indices: which kind of stock is winning
           "NIFTY200 MOMENTUM 30", "NIFTY500 MOMENTUM 50", "NIFTY100 LOW VOLATILITY 30", "NIFTY500 LOW VOLATILITY 50",
           "NIFTY100 QUALITY 30", "NIFTY500 QUALITY 50", "NIFTY50 VALUE 20", "NIFTY200 VALUE 30", "NIFTY500 VALUE 50",
           "NIFTY ALPHA 50", "NIFTY50 EQUAL WEIGHT", "NIFTY500 EQUAL WEIGHT", "NIFTY DIVIDEND OPPORTUNITIES 50",
           # more themes / sectors
           "NIFTY CONSUMPTION", "NIFTY COMMODITIES", "NIFTY PSE", "NIFTY CPSE", "NIFTY MNC", "NIFTY PRIVATE BANK",
           "NIFTY HEALTHCARE INDEX", "NIFTY OIL & GAS", "NIFTY CONSUMER DURABLES", "NIFTY INDIA DEFENCE",
           "NIFTY CHEMICALS", "NIFTY SERVICES SECTOR", "NIFTY INDIA MANUFACTURING"]


def indices(all_names=False, refresh_empty=False):
    d = os.path.join(OUT, "indices"); os.makedirs(d, exist_ok=True)
    names = INDICES
    if all_names:  # every index NSE lists (~235); keeps the order above first
        names = INDICES + sorted(set(nse.fetch_index_names().values()) - set(INDICES))
    for name in names:
        p = os.path.join(d, name.replace("/", "~").replace(":", "=").replace(" ", "_") + ".csv")  # "/" and ":" can't be in file names
        old, a = None, date(2015, 1, 1)
        if os.path.exists(p) and os.path.getsize(p) > 100:   # already have it: fetch only the days after the last one
            old = pd.read_csv(p)
            last = pd.to_datetime(old["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce").max()
            if pd.notna(last):
                a = (last + pd.Timedelta(days=1)).date()
            if a > date.today():
                continue
        elif os.path.exists(p) and not refresh_empty:
            continue
        rows = []
        while a <= date.today():  # NSE returns at most ~70 rows per request, so ask ~2 months at a time
            b = min(a + timedelta(days=80), date.today())
            try:
                rows += nse.fetch_historical_index_data(name, a, b)
            except Exception as e:
                print(f"{name} {a}: failed {e!r:.100}", flush=True)
            a = b + timedelta(days=1); time.sleep(0.3)
        df = pd.DataFrame(rows).drop_duplicates()
        if old is not None:
            if df.empty:
                continue
            df = pd.concat([old, df]).drop_duplicates(subset=["EOD_TIMESTAMP"], keep="last")
        df.to_csv(p, index=False)
        print(f"{name}: {len(df)} days", flush=True)


def mtf():
    d = os.path.join(OUT, "mtf"); os.makedirs(d, exist_ok=True)
    warm()
    for day in days(date(2021, 6, 1)):
        p = os.path.join(d, f"{day}.csv")
        if os.path.exists(p):
            continue
        b = fetch(f"{ARCH}/content/equities/mrg_trading_{day:%d%m%y}.zip")
        if b is None:
            _empty(p, day); continue  # holiday / not published
        lines = unzip(b).decode("utf-8", "replace").splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("Symbol,")), None)
        if start is None:
            _empty(p, day); continue
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), dtype=str)
        df.columns = [c.strip() for c in df.columns]
        qty = [c for c in df.columns if c.startswith("Qty Fin")]
        amt = [c for c in df.columns if c.startswith("Amt Fin")]
        out = pd.DataFrame({"symbol": df["Symbol"].str.strip(),
                            "qty": pd.to_numeric(df[qty[0]].str.replace(",", ""), errors="coerce") if qty else None,
                            "amt_lakh": pd.to_numeric(df[amt[0]].str.replace(",", ""), errors="coerce") if amt else None})
        out.to_csv(p, index=False)
        print(f"mtf {day}: {len(out)}", flush=True)
        time.sleep(0.3)


def options():
    d = os.path.join(OUT, "options"); os.makedirs(d, exist_ok=True)
    warm()
    for day in days(date(2016, 1, 1)):
        p = os.path.join(d, f"{day}.csv")
        if os.path.exists(p):
            continue
        if day >= date(2024, 7, 8):
            b = fetch(f"{ARCH}/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip")
        else:
            mon = day.strftime("%b").upper()
            b = fetch(f"{ARCH}/content/historical/DERIVATIVES/{day.year}/{mon}/fo{day:%d}{mon}{day.year}bhav.csv.zip")
        if b is None:
            _empty(p, day); continue
        df = pd.read_csv(io.BytesIO(unzip(b)), dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if "TckrSymb" in df.columns:  # new (UDiFF) format
            df = df[df["FinInstrmTp"].str.strip() == "STO"]
            sym, typ, oi, vol = df["TckrSymb"], df["OptnTp"], df["OpnIntrst"], df["TtlTradgVol"]
        else:
            df = df[df["INSTRUMENT"].str.strip() == "OPTSTK"]
            sym, typ, oi, vol = df["SYMBOL"], df["OPTION_TYP"], df["OPEN_INT"], df["CONTRACTS"]
        t = pd.DataFrame({"symbol": sym.str.strip(), "typ": typ.str.strip(),
                          "oi": pd.to_numeric(oi, errors="coerce"), "vol": pd.to_numeric(vol, errors="coerce")})
        g = t.pivot_table(index="symbol", columns="typ", values=["oi", "vol"], aggfunc="sum").fillna(0)
        g.columns = [f"{a}_{b.lower()}" for a, b in g.columns]
        g.reset_index().to_csv(p, index=False)
        print(f"options {day}: {len(g)} stocks", flush=True)
        time.sleep(0.3)


def sectors_all():
    """Industry of every stock that appears in ANY NSE index list (Total Market, Microcap, sectors...)."""
    p = os.path.join(OUT, "sectors.csv")
    rows = pd.read_csv(p).to_dict("records") if os.path.exists(p) else []
    seen = {r["Symbol"] for r in rows}
    for name in ["NIFTY TOTAL MARKET", "NIFTY MICROCAP 250"] + sorted(set(nse.fetch_index_names().values())):
        try:
            for r in nse.fetch_index_constituents(name) or []:
                if r.get("Symbol") and r.get("Industry") and r["Symbol"] not in seen:
                    rows.append({"Symbol": r["Symbol"], "Industry": r["Industry"]}); seen.add(r["Symbol"])
        except Exception as e:
            print(f"{name}: {e!r:.80}")
        time.sleep(0.3)
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"industries known for {len(rows)} stocks", flush=True)


def futures(start=date(2018, 1, 1)):
    """Stock and index futures per day from the F&O bhavcopy, all expiries summed:
    symbol, kind (STK/IDX), oi, oi_chg, contracts, value (Rs lakh), near-month close and settle price."""
    d = os.path.join(OUT, "fut"); os.makedirs(d, exist_ok=True)
    warm()
    for day in days(start):
        p = os.path.join(d, f"{day}.csv")
        if os.path.exists(p):
            continue
        if day >= date(2024, 7, 8):
            b = fetch(f"{ARCH}/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip")
        else:
            mon = day.strftime("%b").upper()
            b = fetch(f"{ARCH}/content/historical/DERIVATIVES/{day.year}/{mon}/fo{day:%d}{mon}{day.year}bhav.csv.zip")
        if b is None:
            _empty(p, day); continue
        df = pd.read_csv(io.BytesIO(unzip(b)), dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if "TckrSymb" in df.columns:
            df = df[df["FinInstrmTp"].str.strip().isin(["STF", "IDF"])]
            t = pd.DataFrame({"symbol": df["TckrSymb"].str.strip(), "kind": np.where(df["FinInstrmTp"].str.strip() == "STF", "STK", "IDX"),
                              "exp": pd.to_datetime(df["XpryDt"], errors="coerce"), "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
                              "settle": pd.to_numeric(df["SttlmPric"], errors="coerce"), "oi": pd.to_numeric(df["OpnIntrst"], errors="coerce"),
                              "oi_chg": pd.to_numeric(df["ChngInOpnIntrst"], errors="coerce"), "contracts": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
                              "value": pd.to_numeric(df["TtlTrfVal"], errors="coerce") / 1e5})
        else:
            df = df[df["INSTRUMENT"].str.strip().isin(["FUTSTK", "FUTIDX"])]
            t = pd.DataFrame({"symbol": df["SYMBOL"].str.strip(), "kind": np.where(df["INSTRUMENT"].str.strip() == "FUTSTK", "STK", "IDX"),
                              "exp": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y", errors="coerce"), "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
                              "settle": pd.to_numeric(df["SETTLE_PR"], errors="coerce"), "oi": pd.to_numeric(df["OPEN_INT"], errors="coerce"),
                              "oi_chg": pd.to_numeric(df["CHG_IN_OI"], errors="coerce"), "contracts": pd.to_numeric(df["CONTRACTS"], errors="coerce"),
                              "value": pd.to_numeric(df["VAL_INLAKH"], errors="coerce")})
        near = t.sort_values("exp").groupby("symbol").first()[["close", "settle"]]
        agg = t.groupby(["symbol", "kind"])[["oi", "oi_chg", "contracts", "value"]].sum().reset_index().set_index("symbol")
        agg = agg.join(near).reset_index()
        agg.to_csv(p, index=False)
        print(f"fut {day}: {len(agg)}", flush=True)
        time.sleep(0.2)


def constituents():
    """Today's member list of every index we have levels for (NSE has no past lists)."""
    out, names = [], [os.path.basename(p)[:-4].replace("_", " ").replace("~", "/").replace("=", ":") for p in
                      __import__("glob").glob(os.path.join(OUT, "indices", "*.csv"))]
    for name in sorted(names):
        rows = None
        for fn in (nse.fetch_index_constituents, nse.index_constituents):
            try:
                rows = fn(name)
                if rows:
                    break
            except Exception:
                rows = None
        syms = [r.get("Symbol") or r.get("symbol") for r in (rows or [])]
        syms = [x for x in syms if x and x.upper() != name.upper()]
        out += [(name, x) for x in syms]
        print(f"{name}: {len(syms)} members", flush=True)
        time.sleep(0.3)
    pd.DataFrame(out, columns=["index", "symbol"]).to_csv(os.path.join(OUT, "constituents.csv"), index=False)


if __name__ == "__main__":
    {"sectors": sectors, "indices": indices, "all_indices": lambda: indices(True), "constituents": constituents, "sectors_all": sectors_all, "mtf": mtf, "options": options, "futures": (lambda: futures(date.fromisoformat(sys.argv[2])) if len(sys.argv) > 2 else futures())}[sys.argv[1]]()
    print("DONE", sys.argv[1], flush=True)
