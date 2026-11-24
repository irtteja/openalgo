"""Whole-market daily prices + delivery for EVERY NSE stock, 2015 -> today (one file per day).

    /tmp/nslenv/bin/python scripts/fetch_bhav.py

Sources (NSE archives): the daily equity bhavcopy (old format until 2024-07-05, UDiFF after) and the
MTO delivery file. Output: data/extra/bhav/{date}.csv with
    symbol, series, isin, open, high, low, close, volume, value, dlv_qty
Prices are as traded (not adjusted for splits/bonus); adjustment happens when the panel is built.
Safe to re-run: finished days are skipped (holidays leave an empty file).
"""
import io, os, sys, time, zipfile
from datetime import date, timedelta

import pandas as pd
from nse_irt import NSE

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "extra", "bhav")
ARCH = "https://nsearchives.nseindia.com"
SERIES = {"EQ", "BE", "BZ", "SM", "ST"}  # normal, trade-to-trade, SME
nse = NSE()
S = nse._session._s


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
        time.sleep(3 * (i + 1))
        try:
            S.get("https://www.nseindia.com/all-reports", timeout=20)
        except Exception:
            pass
    return None


def unzip(b):
    if not zipfile.is_zipfile(io.BytesIO(b)):
        return b
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        n = z.namelist()
        return z.read(n[0]) if n else b""


def bhav(day):
    if day >= date(2024, 7, 8):
        b = fetch(f"{ARCH}/content/cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip")
    else:
        mon = day.strftime("%b").upper()
        b = fetch(f"{ARCH}/content/historical/EQUITIES/{day.year}/{mon}/cm{day:%d}{mon}{day.year}bhav.csv.zip")
    if b is None:
        return None
    raw = unzip(b)
    if not raw.strip():
        return None
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "TckrSymb" in df.columns:
        m = {"TckrSymb": "symbol", "SctySrs": "series", "ISIN": "isin", "OpnPric": "open", "HghPric": "high",
             "LwPric": "low", "ClsPric": "close", "TtlTradgVol": "volume", "TtlTrfVal": "value"}
    else:
        m = {"SYMBOL": "symbol", "SERIES": "series", "ISIN": "isin", "OPEN": "open", "HIGH": "high", "LOW": "low",
             "CLOSE": "close", "TOTTRDQTY": "volume", "TOTTRDVAL": "value"}
    df = df[list(m)].rename(columns=m)
    for c in df.columns:
        df[c] = df[c].str.strip()
    df = df[df["series"].isin(SERIES)]
    for c in ["open", "high", "low", "close", "volume", "value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def delivery(day):
    b = fetch(f"{ARCH}/archives/equities/mto/MTO_{day:%d%m%Y}.DAT")
    if b is None:
        return None
    rows = []
    for line in b.decode("utf-8", "replace").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6 and p[0] == "20":
            rows.append((p[2], p[3], pd.to_numeric(p[5], errors="coerce")))
    return pd.DataFrame(rows, columns=["symbol", "series", "dlv_qty"]) if rows else None


def main():
    os.makedirs(OUT, exist_ok=True)
    import datetime as _dt
    ist = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=5, minutes=30)))
    last = ist.date() if ist.hour * 60 + ist.minute >= 18 * 60 else ist.date() - timedelta(days=1)   # NSE posts the day's file by ~6 pm IST
    d = date(2015, 1, 1)
    while d <= last:
        p = os.path.join(OUT, f"{d}.csv")
        if d.weekday() < 5 and not os.path.exists(p):
            df = bhav(d)
            if df is None or df.empty:
                if (ist.date() - d).days > 3:   # holiday; recent days stay open so a late NSE file is retried
                    open(p, "w").close()
            else:
                dl = delivery(d)
                if dl is not None:
                    df = df.merge(dl, on=["symbol", "series"], how="left")
                else:
                    df["dlv_qty"] = float("nan")
                df.to_csv(p + ".tmp", index=False); os.replace(p + ".tmp", p)
                print(f"{d}: {len(df)} stocks", flush=True)
            time.sleep(0.2)
        d += timedelta(days=1)
    print("DONE bhav", flush=True)


if __name__ == "__main__":
    main()
