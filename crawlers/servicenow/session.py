"""Per-session state: template resolution, record setup, and cleanup.

The ServiceNow analogue of Ridgeway's POST /api/reset-db: `prepare` creates the records a task
refers to (tagged with a per-session nonce), `cleanup` removes those and anything the agent
created during the session.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sn import sn_time

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT.parents[1] / "tasks" / "servicenow_tasks.json").read_text(encoding="utf-8"))

# Tables the agent can create rows in during these tasks. Cleanup deletes rows created since start.
CLEAN_TABLES = ["sc_cart_item", "sc_task", "sc_req_item", "sc_request", "incident", "problem",
                "change_request", "sys_user"]
TOKEN = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def fmt(value, ctx: dict):
    """Resolve {{nonce}} and {{alias.field}} in strings, lists, and dicts."""
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            key = m.group(1)
            if key == "nonce":
                return ctx["nonce"]
            alias, _, field = key.partition(".")
            try:
                return str(ctx["records"][alias][field])
            except KeyError as exc:
                raise KeyError(f"unresolved template {{{{{key}}}}} in {value!r}") from exc
        return TOKEN.sub(sub, value)
    if isinstance(value, list):
        return [fmt(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: fmt(v, ctx) for k, v in value.items()}
    return value


def prepare(task: dict, sn) -> dict:
    """Create the task's setup records. Returns the session context."""
    ctx = {"nonce": secrets.token_hex(3), "records": {}, "created": [],
           "start": datetime.now(timezone.utc) - timedelta(seconds=2)}
    for item in task.get("setup") or []:
        fields = fmt(item["fields"], ctx)
        rec = sn.insert(item["table"], fields)
        ctx["records"][item["alias"]] = {**fields, **rec}
        ctx["created"].append((item["table"], rec["sys_id"]))
    # records the agent makes must be strictly later than the setup rows
    ctx["start"] = datetime.now(timezone.utc)
    return ctx


def since(ctx: dict) -> str:
    return f"sys_created_on>={sn_time(ctx['start'])}"


def cleanup(sn, ctx: dict) -> int:
    """Delete setup rows and everything created after the session started. Best effort."""
    removed = 0
    for table, sys_id in ctx["created"]:
        try:
            sn.delete(table, sys_id)
            removed += 1
        except Exception:  # noqa: BLE001
            pass
    # agent-made rows: anything created after `start` whose creator is the REST/agent user
    for table in CLEAN_TABLES:
        try:
            rows = sn.query(table, f"sys_created_on>={sn_time(ctx['start'] - timedelta(seconds=5))}",
                            fields="sys_id")
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            try:
                sn.delete(table, row["sys_id"])
                removed += 1
            except Exception:  # noqa: BLE001
                pass
    return removed
