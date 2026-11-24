"""Atlas: market analysis from the NSE Pattern Atlas, shown inside OpenAlgo.

The Atlas is a separate system that studies NSE's public data (whole market, since 2015) every evening and writes one
compact JSON "feed": market regime, sector phases and strength, every index stock's stage, buying grade, relative
strength, warnings and historical chances, screener lists, hidden groups and the research log. This blueprint only
reads that feed and hands it to the /atlas page. It never places orders.

Where the feed comes from (.env, first one available wins):
  ATLAS_FEED_PATH                 a local file, e.g. the Atlas PC app's explorer/data/atlas_feed.json
  the Atlas engine in this copy   atlas_engine/explorer/data/atlas_feed.json, built on this computer from NSE's public
                                  data by atlas_engine/run.py (services/atlas_engine_service.py runs it); skipped when
                                  ATLAS_SOURCE = 'cloud'
  ATLAS_D1_ACCOUNT_ID, ATLAS_D1_DATABASE_ID, ATLAS_D1_API_TOKEN
                                  a Cloudflare D1 database the Atlas uploads to (table atlas_feed, parts in order);
                                  the token needs only "Account · D1 · Read"
The feed is cached for ATLAS_CACHE_SECONDS (default 900); a reload reads 10 small rows, far inside D1's free tier.
"""

import json
import os

from flask import Blueprint, jsonify, request, session

from services import atlas_engine_service as engine
from utils.httpx_client import get_httpx_client
from utils.logging import get_logger
from utils.session import check_session_validity
from utils.thread_safe_cache import LockedTTLCache

logger = get_logger(__name__)

atlas_bp = Blueprint("atlas_bp", __name__)

_cache = LockedTTLCache(maxsize=1, ttl=float(os.getenv("ATLAS_CACHE_SECONDS", "900") or 900))


class AtlasUnavailable(Exception):
    """The feed is not configured, or could not be read; the message is safe to show the trader."""


def _source():
    path = (os.getenv("ATLAS_FEED_PATH") or "").strip()
    if path:
        return "file", path
    if (os.getenv("ATLAS_SOURCE") or "auto").strip().lower() != "cloud" and os.path.exists(
        engine.FEED
    ):
        return "engine", engine.FEED
    acct, db, tok = (
        os.getenv(k, "").strip()
        for k in ("ATLAS_D1_ACCOUNT_ID", "ATLAS_D1_DATABASE_ID", "ATLAS_D1_API_TOKEN")
    )
    if acct and db and tok:
        return "d1", (acct, db, tok)
    return None, None


def _load():
    kind, where = _source()
    if kind in ("file", "engine"):
        try:
            with open(where, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise AtlasUnavailable(
                "The Atlas feed file in ATLAS_FEED_PATH was not found."
            ) from None
        except (OSError, ValueError):
            logger.exception("Atlas feed file unreadable")
            raise AtlasUnavailable("The Atlas feed file could not be read.") from None
    if kind == "d1":
        acct, db, tok = where
        url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/d1/database/{db}/query"
        try:
            r = get_httpx_client().post(
                url,
                headers={"Authorization": f"Bearer {tok}"},
                json={"sql": "SELECT data FROM atlas_feed ORDER BY part"},
                timeout=30,
            )
            body = r.json()
        except (
            Exception
        ):  # network or JSON trouble: tell the trader plainly, keep the detail in the log
            logger.exception("Atlas feed request to Cloudflare failed")
            raise AtlasUnavailable("Could not reach the Atlas database on Cloudflare.") from None
        if r.status_code != 200 or not body.get("success"):
            logger.warning(
                f"Atlas feed: Cloudflare answered {r.status_code}: {str(body.get('errors'))[:300]}"
            )
            raise AtlasUnavailable(
                "The Atlas database refused the request; check the D1 settings and token."
            )
        rows = (body.get("result") or [{}])[0].get("results") or []
        if not rows:
            raise AtlasUnavailable("The Atlas database has no feed yet.")
        try:
            return json.loads("".join(row["data"] for row in rows))
        except (KeyError, ValueError):
            raise AtlasUnavailable("The Atlas feed in the database is incomplete.") from None
    raise AtlasUnavailable(
        "No Atlas analysis yet. Start the first update with Update now on this page, or connect the cloud Atlas in .env."
    )


@atlas_bp.route("/atlas/api/feed", methods=["GET"])
@check_session_validity
def atlas_feed():
    kind, where = _source()
    key = (
        "feed",
        kind,
        os.path.getmtime(where) if kind in ("file", "engine") and os.path.exists(where) else None,
    )
    try:
        feed = _cache.get_or_load(
            key, _load
        )  # a rebuilt file has a new time, so it is read again at once
    except AtlasUnavailable as e:
        return jsonify({"status": "error", "message": str(e)}), 503
    return jsonify({"status": "success", "source": _source()[0], "data": feed})


@atlas_bp.route("/atlas/api/refresh", methods=["POST"])
@check_session_validity
def atlas_refresh():
    """Drop the cached feed so the next read fetches the latest (after the evening update)."""
    _cache.invalidate()
    return jsonify({"status": "success"})


@atlas_bp.route("/atlas/api/engine/status", methods=["GET"])
@check_session_validity
def atlas_engine_status():
    """Progress of the update on this computer: state, current step, finished steps, the log's last lines."""
    return jsonify({"status": "success", "data": engine.status()})


@atlas_bp.route("/atlas/api/engine/run", methods=["POST"])
@check_session_validity
def atlas_engine_run():
    """Start an update now. {"full": true} also relearns every pattern (normally weekly)."""
    full = bool((request.get_json(silent=True) or {}).get("full"))
    ok, message = engine.start(full=full)
    return jsonify({"status": "success" if ok else "error", "message": message}), (
        200 if ok else 409
    )


_LIVE_MAX_STOCKS = 200


@atlas_bp.route("/atlas/api/live", methods=["POST"])
@check_session_validity
def atlas_live():
    """Today's moves from the logged-in broker: Nifty, the sector indices brokers publish, and up to 200 stocks.

    Body: {"symbols": ["KOTAKBANK", ...]} (NSE cash symbols). Returns price, previous close and change for each one the
    broker answered. Read-only: a quote request, never an order.
    """
    from database.auth_db import get_api_key_for_tradingview
    from services.quotes_service import get_multiquotes
    from services.sector_rotation_service import BENCHMARKS, OPTIONAL_SECTORS, SECTORS

    api_key = get_api_key_for_tradingview(session.get("user"))
    if not api_key:
        return jsonify(
            {
                "status": "error",
                "message": "Live prices need an OpenAlgo API key. Generate one on the API Key page.",
            }
        ), 400
    wanted = [
        str(x).strip().upper()
        for x in (request.get_json(silent=True) or {}).get("symbols", [])
        if str(x).strip()
    ]
    indices = [i for i in BENCHMARKS + SECTORS + OPTIONAL_SECTORS if i.exchange == "NSE_INDEX"]
    symbols = [{"symbol": i.symbol, "exchange": i.exchange} for i in indices]
    symbols += [
        {"symbol": s_, "exchange": "NSE"} for s_ in dict.fromkeys(wanted[:_LIVE_MAX_STOCKS])
    ]
    ok, body, _code = get_multiquotes(symbols, api_key=api_key)
    if not ok:
        logger.warning(
            f"Atlas live prices: broker did not answer: {str(body.get('message'))[:200]}"
        )
        return jsonify(
            {
                "status": "error",
                "message": "Live prices are not available right now. Check that the broker is logged in.",
            }
        ), 503
    labels = {i.symbol: i.label for i in indices}
    out = {"indices": [], "stocks": {}}
    for r in body.get("results", []):
        q = r.get("data") or {}
        ltp, prev = q.get("ltp"), q.get("prev_close")
        if not ltp:
            continue
        chg = (float(ltp) / float(prev) - 1) if prev else None
        row = {"symbol": r.get("symbol"), "ltp": ltp, "prev_close": prev, "change": chg}
        if r.get("exchange") == "NSE_INDEX":
            out["indices"].append(dict(row, label=labels.get(r.get("symbol"), r.get("symbol"))))
        else:
            out["stocks"][r.get("symbol")] = row
    return jsonify({"status": "success", "data": out})
