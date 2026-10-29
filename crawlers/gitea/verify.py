#!/usr/bin/env python3
"""Score a Gitea trial two ways, independently (same split as tasks/verify.py):

- score_agent_report: the agent's JSON vs an expected value resolved from setup records or a REST count.
- verify_task: REST assertions on server state.

Constraint kinds
  rest       {path, status=200, json:{key: expected}}        key: "a.b" or "labels[].name" (list contains)
  rest_list  {path, any:{key: expected}, count?}             items of a list endpoint matching `any`
  composite  {parts:[...]}
  report     {key, expected | expected_count:{path}}          scored from the agent's JSON only
Expected strings starting with "~" mean "contains". Everything is compared case-insensitively.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from session import SPEC, fmt


def _check(name: str, ok: bool | None, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}


def _norm(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return " ".join(str(v if v is not None else "").strip().lower().split())


def _digits(v: Any) -> str:
    return re.sub(r"\D", "", str(v or ""))


def pluck(obj: Any, key: str) -> list:
    """Values at `key`. 'a.b' walks dicts; 'xs[].f' maps over a list. Always returns a list."""
    cur = [obj]
    for part in key.split("."):
        nxt = []
        many = part.endswith("[]")
        name = part[:-2] if many else part
        for c in cur:
            v = c.get(name) if isinstance(c, dict) else None
            if many:
                nxt += list(v or [])
            else:
                nxt.append(v)
        cur = nxt
    return cur


def _match(values: list, expected: Any) -> bool:
    want = _norm(expected)
    if want.startswith("~"):
        return any(want[1:] in _norm(v) for v in values)
    return any(_norm(v) == want for v in values)


def _rest(part: dict, api, ctx: dict) -> list[dict]:
    path = fmt(part["path"], ctx)
    status, data = api.get(path)
    out = [_check("status", status == part.get("status", 200), f"{path} -> {status}")]
    if status == 200:
        for key, exp in (part.get("json") or {}).items():
            exp = fmt(exp, ctx)
            vals = pluck(data, key)
            out.append(_check(key, _match(vals, exp), f"have {vals!r} want {exp!r}"))
    return out


def _rest_list(part: dict, api, ctx: dict) -> list[dict]:
    path = fmt(part["path"], ctx)
    rows = api.get_all(path)
    cond = {k: fmt(v, ctx) for k, v in (part.get("any") or {}).items()}
    hits = [r for r in rows if all(_match(pluck(r, k), v) for k, v in cond.items())]
    want = part.get("count")
    ok = len(hits) == want if want is not None else len(hits) >= 1
    return [_check("list_match", ok, f"{len(hits)}/{len(rows)} item(s) match {cond}")]


def _checks(part: dict, api, ctx: dict) -> list[dict]:
    kind = part.get("kind")
    if kind == "rest":
        return _rest(part, api, ctx)
    if kind == "rest_list":
        return _rest_list(part, api, ctx)
    if kind == "composite":
        out = []
        for i, sub in enumerate(part.get("parts") or [], 1):
            for c in _checks(sub, api, ctx) or [_check("unknown_part", False, str(sub.get("kind")))]:
                out.append({**c, "check": f"{i}.{c['check']}"})
        return out or [_check("composite", False, "no parts")]
    if kind == "report":
        return []
    return [_check("kind", False, f"unknown constraint kind {kind!r}")]


def verify_task(task_id: str, api, ctx: dict) -> dict:
    task = SPEC["tasks"][task_id]
    if task["verification"]["type"] != "state":
        return {"task": task_id, "ok": None, "source": "none", "checks": []}
    checks = _checks(task["constraints"], api, ctx)
    oks = [c["ok"] for c in checks]
    return {"task": task_id, "ok": all(o is True for o in oks) if oks else None,
            "source": "gitea_rest", "checks": checks}


# ---------------------------------------------------------------- agent report
def expected_value(part: dict, api, ctx: dict) -> Any:
    if "expected_count" in part:
        return len(api.get_all(fmt(part["expected_count"]["path"], ctx)))
    return fmt(part["expected"], ctx)


def _same(expected: Any, got: Any, digits: bool) -> bool:
    if digits:
        return bool(_digits(expected)) and _digits(expected) == _digits(got)
    if isinstance(expected, int):
        m = re.search(r"-?\d+", str(got))
        return m is not None and int(m.group()) == expected
    e, g = _norm(expected), _norm(got)
    return bool(e) and (e == g or e in g)


def score_agent_report(task_id: str, report: dict | None, api, ctx: dict, path: str | None = None) -> dict:
    task = SPEC["tasks"][task_id]
    cons = task["constraints"]
    if report is None:
        return {"task": task_id, "ok": False, "path": path, "checks": [_check("report", False, "no report found")]}
    checks = [_check("reported_success", report.get("success") is not False, f"success={report.get('success')!r}")]
    parts = [cons] if cons.get("kind") == "report" else [p for p in cons.get("parts", []) if p.get("kind") == "report"]
    for part in parts:
        want = expected_value(part, api, ctx)
        got = report.get(part["key"])
        checks.append(_check(part["key"], _same(want, got, bool(part.get("digits"))), f"want {want!r} got {got!r}"))
    return {"task": task_id, "ok": all(c["ok"] is True for c in checks), "path": path, "checks": checks}


def format_verify(result: dict) -> str:
    rows = "; ".join(f"{c['check']}={'ok' if c['ok'] else 'FAIL'}" for c in result.get("checks", []))
    return f"verify {result['task']}: {result['ok']}  {rows}"


def format_agent_report(result: dict) -> str:
    rows = "; ".join(f"{c['check']}={'ok' if c['ok'] else 'FAIL'}" for c in result.get("checks", []))
    return f"report {result['task']}: {result['ok']}  {rows}"


if __name__ == "__main__":
    print(__doc__)
