"""ServiceNow task prompts. Same text for every agent.

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


def start_url(task_id: str, instance: str, ctx: dict | None = None) -> str:
    path = SPEC["tasks"][task_id].get("start_path", "/login.do")
    if ctx is not None:
        path = fmt(path, ctx)
    return instance.rstrip("/") + path


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
Do not use the REST API, the browser console, or background scripts. Stay on this instance."""


def render(task_id: str, ctx: dict, url: str = "{url}", agent: str = "{agent}",
           login: tuple[str, str] | None = None) -> str:
    task = SPEC["tasks"][task_id]
    ask = fmt(task["ask"], ctx)
    hint = fmt(task.get("hint") or "", ctx)
    login_line = ""
    if login:
        login_line = f"\nIf you are asked to log in, use user name {login[0]} and password {login[1]}.\n"
    text = f"""This is Task {task_id} ({task['label']}) on a ServiceNow instance.

Open {url} and work there. This is the classic list or form; do not go to the home dashboard, and do not reload this URL once the page has loaded.
If a later page drops the query parameter agent={agent}, leave it. The page puts it back.
{login_line}
{ask}{chr(10) + chr(10) + hint if hint else ''}

Use the search box above a list. Do not open the condition builder (the "choose field" dropdowns).
If an incident form will not save because Caller is empty, set Caller to the user you are logged in as, then save.
After the record is submitted or the field is saved, stop. ServiceNow often shows a blank form, a new number, or the list; that means the save worked. Do not create another copy of the same record.

{_report_block(task_id, task)}
"""
    return text.replace("{url}", url).replace("{agent}", agent).strip() + "\n"


def redact(text: str, login: tuple[str, str] | None) -> str:
    return text.replace(login[1], "***") if login and login[1] else text
