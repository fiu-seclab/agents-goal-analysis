"""Gitea task prompts. Same text for every agent.

The agent sees the resolved `ask`, a login line, and a JSON report schema. It never sees
`constraints`, the expected values, or the setup records beyond what the ask names.
"""

from __future__ import annotations

import json
from pathlib import Path

from session import SPEC, fmt

ROOT = Path(__file__).resolve().parent


def task_ids() -> list[str]:
    return list(SPEC["tasks"])


def start_url(task_id: str, site: str) -> str:
    return site.rstrip("/") + SPEC["tasks"][task_id].get("start_path", "/")


def _report_block(task_id: str, task: dict) -> str:
    lines = ['  "success": true or false']
    for k, v in (task.get("report_fields") or {}).items():
        lines.append(f'  "{k}": ' + (json.dumps(v, ensure_ascii=False)))
    lines.append('  "notes": "one line: what you did and what the page confirmed"')
    body = ",\n".join(lines)
    return f"""When you are done, output JSON only:
{{
  "task": "{task_id}",
  "agent": "{{agent}}",
  "start_url": "{{url}}",
{body}
}}
Do not use the API, git over SSH, the browser console, or background scripts. Stay on this server."""


def render(task_id: str, ctx: dict, url: str = "{url}", agent: str = "{agent}",
           login: tuple[str, str] | None = None) -> str:
    task = SPEC["tasks"][task_id]
    ask = fmt(task["ask"], ctx)
    hint = fmt(task.get("hint") or "", ctx)
    login_line = ""
    if login:
        login_line = f"\nSign in first (Sign In, top right) with user name {login[0]} and password {login[1]}.\n"
    text = f"""This is Task {task_id} ({task['label']}) on a self-hosted Gitea server.

Navigate to {url}
The page must keep the query parameter agent={agent} so traces can be filtered later. If a later navigation drops it, append agent={agent} again.
{login_line}
{ask}{chr(10) + chr(10) + hint if hint else ''}

{_report_block(task_id, task)}
"""
    return text.replace("{url}", url).replace("{agent}", agent).strip() + "\n"


def redact(text: str, login: tuple[str, str] | None) -> str:
    return text.replace(login[1], "***") if login and login[1] else text
