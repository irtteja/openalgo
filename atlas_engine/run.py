"""Atlas engine: download NSE's public data and rebuild the Atlas analysis, on this computer.

    python atlas_engine/run.py            # update the data, then rebuild the analysis
    python atlas_engine/run.py --full     # same, and relearn every pattern now (normally weekly)

OpenAlgo starts this as a separate process (services/atlas_engine_service.py), never inside the web app: the work
takes minutes to hours and would stall the app. Every step is safe to stop and run again; finished downloads are
skipped. Progress goes to atlas_engine/data/status.json (read by the Atlas page) and atlas_engine/data/atlas.log.

Data lives under atlas_engine/data (about 2 GB after the first run, which downloads NSE history since 2015 and takes
a few hours). The analysis the Atlas page shows is written to atlas_engine/explorer/data/atlas_feed.json.
"""

import datetime as dt
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
LOG = os.path.join(DATA, "atlas.log")
STATUS = os.path.join(DATA, "status.json")
FEED = os.path.join(ROOT, "explorer", "data", "atlas_feed.json")
PY = sys.executable

_status = {"state": "idle", "started": None, "finished": None, "step": None, "steps": [], "ok": None, "first_run": False}


def _write_status():
    os.makedirs(DATA, exist_ok=True)
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_status, f)
    os.replace(tmp, STATUS)


def say(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    os.makedirs(DATA, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args, label, env=None):
    """Run one step as its own Python process, with the engine folder on the import path."""
    _status["step"] = label
    _write_status()
    say(f"{label} ...")
    t = time.time()
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), **(env or {}))
    with open(LOG, "a", encoding="utf-8") as f:
        r = subprocess.run([PY, *args], cwd=ROOT, env=e, stdout=f, stderr=subprocess.STDOUT)
    ok = r.returncode == 0
    took = int(time.time() - t)
    _status["steps"].append({"label": label, "ok": ok, "seconds": took})
    say(f"{label}: {'done' if ok else 'FAILED'} in {took}s")
    _write_status()
    return ok


def older_than(path, days):
    return not os.path.exists(path) or time.time() - os.path.getmtime(path) > days * 86400


def update():
    S3 = os.path.join(DATA, "extra", "step3")
    ok = True
    ok &= run(["scripts/fetch_bhav.py"], "Daily prices and delivery for every NSE stock")
    ok &= run(["scripts/fetch_step3.py", "all_indices"], "Index levels (all NSE indices)")
    if older_than(os.path.join(S3, "constituents.csv"), 7):
        run(["scripts/fetch_step3.py", "constituents"], "Index member lists (weekly)")
        run(["scripts/fetch_step3.py", "sectors_all"], "Industry of each stock (weekly)")
    if not os.path.exists(os.path.join(S3, "sectors.csv")):
        run(["scripts/fetch_step3.py", "sectors"], "Industry lists")
    run(["-c", "from rsi.data import update_corporate_actions; update_corporate_actions()"], "Splits, bonuses and dividends")
    run(["-c", "from rsi.extra import fetch_vix; fetch_vix(update=True)"], "India VIX")
    run(["-m", "rsi.market_data"], "NSE daily files: margins, FII/client positions, bulk deals")
    run(["scripts/fetch_corporate.py", "insider"], "Insider trades and pledges")
    syms = os.path.join(DATA, "index_symbols.txt")
    if os.path.exists(os.path.join(S3, "constituents.csv")):
        import csv

        with open(os.path.join(S3, "constituents.csv"), encoding="utf-8") as f:
            s = sorted({r["symbol"] for r in csv.DictReader(f) if "SME" not in r["index"].upper()})
        with open(syms, "w", encoding="utf-8") as f:
            f.write("\n".join(s))
        run(["scripts/fetch_corporate.py", "shp", syms], "Promoter shareholding (monthly refresh)", env={"ATLAS_SHP_REFRESH_DAYS": "30"})
    run(["-c", "from rsi.fundamentals import download; download(start='2018-04-01', passes=1)"], "Quarterly results")
    run(["scripts/fetch_step3.py", "mtf"], "Margin trading")
    run(["scripts/fetch_step3.py", "options"], "Options open interest")
    run(["scripts/fetch_step3.py", "futures"], "Futures open interest")
    run(["scripts/build_indices.py", "fetch"], "Index valuation and market caps (today)")
    return ok


def build(full=False):
    steps = [("scripts/rotation.py", "Sector phases")]
    if full or older_than(os.path.join(ROOT, "results", "patterns", "detail_stocks.json"), 6):
        steps.append(("scripts/pattern_loop.py live", "Relearn patterns and probabilities (weekly)"))
    steps += [
        ("scripts/build_explorer.py", "Page data"),
        ("scripts/build_deep.py", "Sector, stock and pattern pages"),
        ("scripts/build_rs.py", "Relative strength"),
        ("scripts/build_watch.py", "Rotation, breadth, breakouts"),
        ("scripts/build_indices.py", "Indices and constituents"),
        ("scripts/build_screen.py", "Screener, stages, buying and selling"),
        ("scripts/build_fno.py", "F&O build-up"),
        ("scripts/build_structure.py", "Market regime, hidden groups, move breakdown"),
        ("scripts/research.py page", "Research log"),
        ("scripts/build_atlas_feed.py", "Atlas feed"),
    ]
    ok = True
    for cmd, label in steps:
        ok &= run(cmd.split(), label)
    return ok


def main():
    full = "--full" in sys.argv
    _status.update(state="running", started=dt.datetime.now().isoformat(timespec="seconds"), finished=None, steps=[], ok=None,
                   first_run=not os.path.isdir(os.path.join(DATA, "extra", "bhav")))
    if _status["first_run"]:
        say("First run: downloading NSE history since 2015. This takes a few hours; it continues where it stopped if interrupted.")
    _write_status()
    ok = update()
    ok &= build(full)
    _status.update(state="finished", finished=dt.datetime.now().isoformat(timespec="seconds"), step=None, ok=bool(ok))
    _write_status()
    say("All done." if ok else "Finished with some failures; the Atlas shows what was built. Details above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
