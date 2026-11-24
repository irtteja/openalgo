"""Runs the Atlas engine (atlas_engine/run.py) on this computer, on a schedule or on request.

The engine downloads NSE's public data and rebuilds the Atlas analysis. It is started as a separate process with its
output in a log file: the work takes minutes (hours on the first run) and must never run inside the web worker. Only
one run at a time. The process handle is reaped with ``poll()``, which never blocks the eventlet hub; a run that
outlives an app restart is recognised from its pid file and the engine's own status file.

Settings (.env):
  ATLAS_ENGINE_SCHEDULE   'True' (default) runs it every weekday evening; 'False' only from the Atlas page
  ATLAS_ENGINE_TIME       IST time of the evening run, default '19:30' (NSE has posted the day's files by then)
"""

import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta

import psutil
import pytz

from utils.logging import get_logger

logger = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "atlas_engine")
DATA = os.path.join(ENGINE, "data")
STATUS = os.path.join(DATA, "status.json")
LOG = os.path.join(DATA, "atlas.log")
OUT = os.path.join(DATA, "engine_output.log")
PID = os.path.join(DATA, "engine.pid")
FEED = os.path.join(ENGINE, "explorer", "data", "atlas_feed.json")
IST = pytz.timezone("Asia/Kolkata")

_lock = threading.Lock()  # in-memory bookkeeping only; no I/O is done while holding it
_proc = None
_scheduler = None


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _pid_alive(pid):
    """True if `pid` is a running engine process. psutil, because os.kill on Windows terminates the target."""
    try:
        p = psutil.Process(pid)
        return p.is_running() and any("run.py" in part for part in p.cmdline())
    except (psutil.Error, OSError):
        return False


def is_running():
    global _proc
    with _lock:
        proc = _proc
    if proc is not None:
        if proc.poll() is None:  # non-blocking; reaps the child once it has exited
            return True
        with _lock:
            if _proc is proc:
                _proc = None
    try:
        with open(PID, encoding="utf-8") as f:
            pid = int(f.read().strip() or 0)
    except (OSError, ValueError):
        return False
    return pid > 0 and _pid_alive(pid)


def start(full=False):
    """Start a run. Returns (started, message for the trader)."""
    global _proc
    if not os.path.exists(os.path.join(ENGINE, "run.py")):
        return False, "The Atlas engine is not installed in this copy of OpenAlgo."
    if is_running():
        return False, "The Atlas is already updating. Its progress is shown on the Atlas page."
    os.makedirs(DATA, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ENGINE, "run.py")] + (["--full"] if full else [])
    kwargs = {"cwd": ROOT, "stdout": None, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    else:
        kwargs["start_new_session"] = True
    try:
        with open(
            OUT, "a", encoding="utf-8"
        ) as out:  # the child keeps its own handle; ours closes here
            kwargs["stdout"] = out
            proc = subprocess.Popen(cmd, **kwargs)
    except OSError:
        logger.exception("Could not start the Atlas engine")
        return (
            False,
            "The Atlas update could not be started. Details are in the OpenAlgo error log.",
        )
    with open(PID, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))
    with _lock:
        _proc = proc
    logger.info(f"Atlas engine started (pid {proc.pid}{', full relearn' if full else ''})")
    return True, "The Atlas update has started."


def status(log_lines=40):
    st = _read_json(STATUS) or {"state": "never run"}
    running = is_running()
    if st.get("state") == "running" and not running:
        st["state"] = (
            "stopped"  # the process ended without writing its final status (closed or crashed)
        )
    st["running"] = running
    tail = []
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-log_lines:]
    except OSError:
        pass
    st["log"] = [x.rstrip("\n") for x in tail]
    st["feed_exists"] = os.path.exists(FEED)
    job = _scheduler.get_job("atlas_engine_daily") if _scheduler else None
    st["next_run"] = job.next_run_time.isoformat() if job and job.next_run_time else None
    return st


def _scheduled_run():
    ok, msg = start()
    logger.info(f"Scheduled Atlas update: {msg}")


def _due_catch_up(now):
    """True when the last evening's run was missed (computer off) and NSE's files are out."""
    st = _read_json(STATUS) or {}
    last = st.get("finished") or st.get("started")
    hh, mm = (int(x) for x in (os.getenv("ATLAS_ENGINE_TIME", "19:30") or "19:30").split(":"))
    day = now.date()
    if now.hour * 60 + now.minute < hh * 60 + mm:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    due_from = IST.localize(datetime(day.year, day.month, day.day, hh, mm))
    if not last:
        return False  # never set up: the first run (hours of downloads) is started by the trader from the page
    try:
        return IST.localize(datetime.fromisoformat(last)) < due_from
    except ValueError:
        return False


def init_atlas_scheduler():
    """Start the weekday evening schedule, and catch up a run missed while the computer was off."""
    global _scheduler
    if (os.getenv("ATLAS_ENGINE_SCHEDULE", "True") or "True").strip().lower() == "false":
        logger.info("Atlas engine schedule is off (ATLAS_ENGINE_SCHEDULE)")
        return None
    if not os.path.exists(os.path.join(ENGINE, "run.py")):
        return None
    from apscheduler.schedulers.background import BackgroundScheduler

    try:
        hh, mm = (int(x) for x in (os.getenv("ATLAS_ENGINE_TIME", "19:30") or "19:30").split(":"))
    except ValueError:
        logger.warning("ATLAS_ENGINE_TIME should look like 19:30; using 19:30")
        hh, mm = 19, 30
    _scheduler = BackgroundScheduler(timezone=IST)
    _scheduler.add_job(
        _scheduled_run,
        "cron",
        day_of_week="mon-fri",
        hour=hh,
        minute=mm,
        id="atlas_engine_daily",
        misfire_grace_time=6 * 3600,
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )
    _scheduler.start()
    if _due_catch_up(datetime.now(IST)):
        _scheduler.add_job(
            _scheduled_run,
            "date",
            run_date=datetime.now(IST) + timedelta(minutes=2),
            id="atlas_engine_catch_up",
            replace_existing=True,
        )
        logger.info("Atlas: the last evening update was missed; catching up in 2 minutes")
    return _scheduler
