"""Download insider trades, shareholding patterns and announcements (NSE, via github.com/irtteja/nselib).

Run from the irtteja/nselib virtualenv (see fetch_integrated_results.py):
    /tmp/nslenv/bin/python scripts/fetch_corporate.py insider          # market-wide, weekly 2015 -> today (new NSE feed from May 2026)
    /tmp/nslenv/bin/python scripts/fetch_corporate.py shp /tmp/syms.txt   # per stock (NSE has ~2021 onward)
    /tmp/nslenv/bin/python scripts/fetch_corporate.py ann /tmp/syms.txt   # per stock, full history

Output: data/extra/corp/{insider,shp,ann}/... as CSV. Safe to re-run: finished files are skipped.
"""
import csv, json, os, sys, time
from datetime import date, timedelta

from nse_irt import NSE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "extra", "corp")
nse = NSE()
get = nse._session.get


def retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"  attempt {i + 1} failed: {e!r:.100}", flush=True)
            time.sleep(5 * (i + 1))
    return None


def rows_of(r):
    if isinstance(r, dict):
        r = r.get("data", [])
    return r if isinstance(r, list) else []


def write(path, rows, cols):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


INS_COLS = ["symbol", "date", "acqfromDt", "acqtoDt", "personCategory", "acqMode", "tdpTransactionType",
            "secType", "secAcq", "secVal", "befAcqSharesPer", "afterAcqSharesPer"]


def insider():
    d = os.path.join(OUT, "insider"); os.makedirs(d, exist_ok=True)
    start = date(2015, 1, 1)
    while start <= min(date.today(), NEW_PIT_FROM - timedelta(days=1)):   # newer weeks come from insider_new()
        end = min(start + timedelta(days=6), date.today())
        path = os.path.join(d, f"{start}.csv")
        if not os.path.exists(path) or end >= date.today() - timedelta(days=3):
            p = {"index": "equities", "from_date": start.strftime("%d-%m-%Y"), "to_date": end.strftime("%d-%m-%Y")}
            r = retry(lambda: get("corporates-pit", params=p))
            if r is not None:
                rows = rows_of(r)
                write(path, rows, INS_COLS)
                print(f"insider {start}: {len(rows)}", flush=True)
            time.sleep(0.5)
        start = end + timedelta(days=1)


# ---- NSE's newer filing system (from May 2026 the old "corporates-pit" feed is empty) ----------------
NEW_PIT_FROM = date(2026, 5, 1)
CAT = {"promoter": "Promoters", "promoter group": "Promoter Group", "promoter and director": "Promoters", "director": "Director",
       "kmp": "Key Managerial Personnel", "key managerial personnel": "Key Managerial Personnel", "immediate relative": "Immediate relative",
       "designated person": "Employees/Designated Employees", "employee": "Employees/Designated Employees",
       "employees/designated employees": "Employees/Designated Employees"}


def _mode(m):
    m = (m or "").strip(); l = m.lower()
    if "invoc" in l:
        return "Invocation of pledge"
    if "pledge" in l and ("release" in l or "revo" in l):
        return "Revokation of Pledge"
    if "pledge" in l:
        return "Pledge Creation"
    return m


def _pct(x):   # XBRL stores shareholding as a fraction (0.0041); the old feed used percent (0.41)
    try:
        return f"{float(x) * 100:.4g}"
    except (TypeError, ValueError):
        return ""


def _xbrl_rows(rec, S):
    import xml.etree.ElementTree as ET
    from collections import defaultdict
    try:
        root = ET.fromstring(S.get(rec["xmlFileName"], timeout=40).content)
    except Exception:
        return None
    ctx = defaultdict(dict)
    for el in root.iter():
        c = el.get("contextRef")
        if c and el.tag.startswith("{"):
            ctx[c][el.tag.split("}")[1]] = (el.text or "").strip()
    out = []
    bdt = (rec.get("broadcastDateTime") or "")[:17]
    for v in ctx.values():
        if "SecuritiesAcquiredOrDisposedNumberOfSecurity" not in v:
            continue
        cat = v.get("CategoryOfPerson", "")
        out.append({"symbol": rec.get("symbol"), "date": bdt,
                    "acqfromDt": v.get("DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate"),
                    "acqtoDt": v.get("DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate"),
                    "personCategory": CAT.get(cat.strip().lower(), "Other" if cat else ""), "acqMode": _mode(v.get("ModeOfAcquisitionOrDisposal")),
                    "tdpTransactionType": v.get("SecuritiesAcquiredOrDisposedTransactionType"), "secType": "Equity Shares" if (v.get("TypeOfInstrument") or "").strip().lower() == "equity" else v.get("TypeOfInstrument"),
                    "secAcq": v.get("SecuritiesAcquiredOrDisposedNumberOfSecurity"), "secVal": v.get("SecuritiesAcquiredOrDisposedValueOfSecurity"),
                    "befAcqSharesPer": _pct(v.get("SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding")),
                    "afterAcqSharesPer": _pct(v.get("SecuritiesHeldPostAcquistionOrDisposalPercentageOfShareholding"))})
    return out


def insider_new(start=NEW_PIT_FROM, workers=4):
    """Weekly insider trades from NSE's new PIT feed (corporates-pit-gg) + each filing's XBRL file."""
    from concurrent.futures import ThreadPoolExecutor
    d = os.path.join(OUT, "insider"); os.makedirs(d, exist_ok=True)
    S = nse._session._s
    try:
        S.get("https://www.nseindia.com/companies-listing/corporate-filings-insider-trading", timeout=30)
    except Exception:
        pass
    a = start
    while a <= date.today():
        b = min(a + timedelta(days=6), date.today())
        path = os.path.join(d, f"{a}.csv")
        done = os.path.exists(path) and os.path.getsize(path) > 200 and b < date.today() - timedelta(days=3)
        if not done:
            url = f"https://www.nseindia.com/api/corporates-pit-gg?index=equities&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}"
            recs = None
            for i in range(3):
                try:
                    recs = S.get(url, timeout=60).json().get("data", [])
                    break
                except Exception:
                    time.sleep(5 * (i + 1))
            if recs is not None:
                with ThreadPoolExecutor(workers) as ex:
                    parts = list(ex.map(lambda r: _xbrl_rows(r, S), recs))
                rows = [x for p in parts if p for x in p]
                write(path, rows, INS_COLS)
                print(f"insider (new feed) {a}: {len(recs)} filings, {len(rows)} trades, {sum(p is None for p in parts)} unreadable", flush=True)
        a = b + timedelta(days=1)


def pledges_reg31(start=NEW_PIT_FROM):
    """Promoter pledge creations / releases / invocations from SAST Regulation 31 disclosures (monthly files)."""
    d = os.path.join(OUT, "insider"); os.makedirs(d, exist_ok=True)
    S = nse._session._s
    kind = {"creation": "Pledge Creation", "release": "Revokation of Pledge", "invocation": "Invocation of pledge"}
    a = date(start.year, start.month, 1)
    while a <= date.today():
        b = min((a.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1), date.today())
        path = os.path.join(d, f"r31_{a:%Y-%m}.csv")
        if not (os.path.exists(path) and b < date.today() - timedelta(days=3)):
            url = f"https://www.nseindia.com/api/corporate-pledgedata-sast3132?index=equities&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}"
            try:
                recs = S.get(url, timeout=60).json().get("data", [])
            except Exception as e:
                print(f"reg31 {a:%Y-%m}: failed {e!r:.80}"); recs = None
            if recs is not None:
                rows = []
                for r in recs:
                    t = (r.get("eventDetailsType") or "").strip().lower()
                    if (r.get("eventDetailsTypeEncumb") or "").strip().lower() != "pledge" or t not in kind:
                        continue
                    rows.append({"symbol": r.get("symbol"), "date": (r.get("broadcastdate") or "")[:17], "acqfromDt": r.get("eventDetailsFromDate"),
                                 "acqtoDt": r.get("eventDetailsToDate"), "personCategory": "Promoters", "acqMode": kind[t], "tdpTransactionType": t.title(),
                                 "secType": "Equity Shares", "secAcq": r.get("eventDetailsHolding"), "secVal": "",
                                 "befAcqSharesPer": (r.get("preeventHoldingPerc") or "").strip(), "afterAcqSharesPer": (r.get("postEventHoldingPerc") or "").strip()})
                write(path, rows, INS_COLS)
                print(f"pledges reg31 {a:%Y-%m}: {len(recs)} filings, {len(rows)} pledge events", flush=True)
        a = b + timedelta(days=1)


REFRESH_DAYS = int(os.environ.get("ATLAS_SHP_REFRESH_DAYS", "100000"))   # the app sets 30
SHP_COLS = ["symbol", "date", "broadcastDate", "pr_and_prgrp", "public_val", "employeeTrusts", "submissionDate"]


def shp(syms):
    d = os.path.join(OUT, "shp"); os.makedirs(d, exist_ok=True)
    for i, s in enumerate(syms):
        path = os.path.join(d, f"{s}.csv")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < REFRESH_DAYS * 86400:
            continue
        r = retry(lambda: get("corporate-share-holdings-master", params={"index": "equities", "symbol": s}))
        if r is None:
            continue
        rows = [dict(x, symbol=s) for x in rows_of(r)]
        write(path, rows, SHP_COLS)
        print(f"[{i + 1}/{len(syms)}] shp {s}: {len(rows)}", flush=True)
        time.sleep(0.4)


ANN_COLS = ["symbol", "an_dt", "exchdisstime", "desc", "attchmntText"]


def ann(syms):
    d = os.path.join(OUT, "ann"); os.makedirs(d, exist_ok=True)
    for i, s in enumerate(syms):
        path = os.path.join(d, f"{s}.csv")
        if os.path.exists(path):
            continue
        r = retry(lambda: nse.announcements(s, from_date=date(2015, 1, 1), to_date=date.today()))
        if r is None:
            continue
        rows = [dict(x, symbol=s, attchmntText=(x.get("attchmntText") or "")[:300]) for x in r]
        write(path, rows, ANN_COLS)
        print(f"[{i + 1}/{len(syms)}] ann {s}: {len(rows)}", flush=True)
        time.sleep(0.4)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "insider":
        insider()
        insider_new()
        pledges_reg31()
    else:
        syms = [x.strip() for x in open(sys.argv[2]) if x.strip()]
        {"shp": shp, "ann": ann}[mode](syms)
    print("DONE", mode, flush=True)
