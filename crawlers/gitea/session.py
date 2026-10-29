"""Per-session state for the Gitea suite: templates, setup records, cleanup.

Setup ops (all values may use {{nonce}} and {{alias.field}}):
  user   {alias, op:'user',  login, full_name?, email?}
  repo   {alias, op:'repo',  owner, name, description?, default_branch?}      -> full_name, name, owner
  label  {alias, op:'label', repo, name}
  issue  {alias, op:'issue', repo, title, body?, labels?:[label aliases], assignees?:[logins],
          as?: login (create as another user, so studyadmin is not auto-subscribed), closed?: bool}
  branch {alias, op:'branch', repo, branch, as?: login}                      (a branch with one commit, no PR)
  pr     {alias, op:'pr',    repo, branch, title, as?: login}                  -> number
"""

from __future__ import annotations

import base64
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT.parents[1] / "tasks" / "gitea_tasks.json").read_text(encoding="utf-8"))
TOKEN = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def fmt(value, ctx: dict):
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


def _setup_one(api, item: dict, ctx: dict) -> dict:
    op, it = item["op"], fmt({k: v for k, v in item.items() if k not in ("alias", "op")}, ctx)
    if op == "user":
        login = it["login"]
        api.must("POST", "/admin/users", {
            "username": login, "email": it.get("email") or f"{login}@hearthline.test",
            "full_name": it.get("full_name", ""), "password": f"Pw-{ctx['nonce']}-Aa1!",
            "must_change_password": False, "send_notify": False, "visibility": "public"})
        return {"login": login, "email": it.get("email") or f"{login}@hearthline.test",
                "full_name": it.get("full_name", "")}
    if op == "repo":
        data = api.must("POST", f"/admin/users/{it['owner']}/repos", {
            "name": it["name"], "description": it.get("description", ""), "auto_init": True,
            "default_branch": it.get("default_branch", "main"), "private": False})
        api.call("DELETE", f"/repos/{data['full_name']}/subscription")  # creating via admin auto-watches; undo
        return {"name": data["name"], "full_name": data["full_name"], "owner": it["owner"],
                "default_branch": data.get("default_branch", "main"), "description": it.get("description", "")}
    if op == "label":
        repo = ctx["records"][item["repo"]]["full_name"]
        data = api.must("POST", f"/repos/{repo}/labels", {"name": it["name"], "color": "#1d76db"})
        return {"id": data["id"], "name": data["name"]}
    if op == "issue":
        repo = ctx["records"][item["repo"]]["full_name"]
        body = {"title": it["title"], "body": it.get("body", "")}
        if item.get("labels"):
            body["labels"] = [ctx["records"][a]["id"] for a in item["labels"]]
        if it.get("assignees"):
            body["assignees"] = it["assignees"]
        data = api.must("POST", f"/repos/{repo}/issues", body, sudo=it.get("as"))
        if it.get("closed"):
            api.must("PATCH", f"/repos/{repo}/issues/{data['number']}", {"state": "closed"}, sudo=it.get("as"))
        return {"number": data["number"], "title": it["title"], "repo": repo}
    if op in ("branch", "pr"):
        repo = ctx["records"][item["repo"]]["full_name"]
        who = it.get("as")
        api.must("POST", f"/repos/{repo}/branches", {"new_branch_name": it["branch"], "old_branch_name": "main"}, sudo=who)
        api.must("POST", f"/repos/{repo}/contents/{it['branch']}.txt", {
            "content": base64.b64encode(f"change {it['branch']}\n".encode()).decode(),
            "message": f"Add {it['branch']}", "branch": it["branch"]}, sudo=who)
        if op == "branch":
            return {"branch": it["branch"], "repo": repo}
        data = api.must("POST", f"/repos/{repo}/pulls", {"head": it["branch"], "base": "main", "title": it["title"]}, sudo=who)
        return {"number": data["number"], "title": it["title"], "repo": repo, "branch": it["branch"]}
    raise ValueError(f"unknown setup op {op!r}")


def prepare(task: dict, api) -> dict:
    ctx = {"nonce": secrets.token_hex(3), "records": {}, "created": []}
    ctx["since"] = datetime.now(timezone.utc) - timedelta(seconds=5)
    for item in task.get("setup") or []:
        rec = _setup_one(api, item, ctx)
        ctx["records"][item["alias"]] = rec
        ctx["created"].append((item["op"], rec))
    ctx["start"] = datetime.now(timezone.utc)
    return ctx


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def cleanup(api, ctx: dict) -> int:
    """Delete every repo and user (except the admin) created since the session began. Best effort."""
    removed = 0
    for repo in api.get_all("/repos/search?sort=id&order=desc"):
        try:
            if _parse(repo["created_at"]) >= ctx["since"]:
                status, _ = api.call("DELETE", f"/repos/{repo['full_name']}")
                removed += status in (200, 204)
        except Exception:  # noqa: BLE001
            pass
    for user in api.get_all("/admin/users"):
        try:
            if not user.get("is_admin") and _parse(user["created"]) >= ctx["since"]:
                status, _ = api.call("DELETE", f"/admin/users/{user['login']}?purge=true")
                removed += status in (200, 204)
        except Exception:  # noqa: BLE001
            pass
    return removed
