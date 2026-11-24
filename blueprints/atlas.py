"""Atlas: market analysis from the NSE Pattern Atlas, shown inside OpenAlgo.

The Atlas is a separate system that studies NSE's public data (whole market, since 2015) every evening and writes one
compact JSON "feed": market regime, sector phases and strength, every index stock's stage, buying grade, relative
strength, warnings and historical chances, screener lists, hidden groups and the research log. This blueprint only
reads that feed and hands it to the /atlas page. It never places orders and never talks to the broker.

Where the feed comes from (.env, first one set wins):
  ATLAS_FEED_PATH                 a local file, e.g. the Atlas PC app's explorer/data/atlas_feed.json
  ATLAS_D1_ACCOUNT_ID, ATLAS_D1_DATABASE_ID, ATLAS_D1_API_TOKEN
                                  a Cloudflare D1 database the Atlas uploads to (table atlas_feed, parts in order);
                                  the token needs only "Account · D1 · Read"
The feed is cached for ATLAS_CACHE_SECONDS (default 900); a reload reads 10 small rows, far inside D1's free tier.
"""

import json
import os

from flask import Blueprint, jsonify

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
    acct, db, tok = (
        os.getenv(k, "").strip()
        for k in ("ATLAS_D1_ACCOUNT_ID", "ATLAS_D1_DATABASE_ID", "ATLAS_D1_API_TOKEN")
    )
    if acct and db and tok:
        return "d1", (acct, db, tok)
    return None, None


def _load():
    kind, where = _source()
    if kind == "file":
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
        "The Atlas is not set up. Add ATLAS_FEED_PATH, or the three ATLAS_D1_ settings, to .env."
    )


@atlas_bp.route("/atlas/api/feed", methods=["GET"])
@check_session_validity
def atlas_feed():
    try:
        feed = _cache.get_or_load("feed", _load)
    except AtlasUnavailable as e:
        return jsonify({"status": "error", "message": str(e)}), 503
    return jsonify({"status": "success", "source": _source()[0], "data": feed})


@atlas_bp.route("/atlas/api/refresh", methods=["POST"])
@check_session_validity
def atlas_refresh():
    """Drop the cached feed so the next read fetches the latest (after the evening update)."""
    _cache.invalidate()
    return jsonify({"status": "success"})
