#!/usr/bin/env python3
"""Score a ServiceNow trial two ways, independently (same split as tasks/verify.py):

- score_agent_report: JSON the agent emitted vs an expected value resolved from the setup
  records or a REST count.
- verify_task: REST queries for records the agent should have created or changed.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from session import SPEC, fmt, since
from sn import FakeSN


def _check(name: str, ok: bool | None, detail: str = "") -> dict:
    return {"check": name, "ok": ok, "detail": detail}


def _norm(value: Any) -> str:
    return " ".join(str(value if value is not None else "").strip().lower().split())


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def encode_where(where: list, ctx: dict) -> str:
    return "^".join(f"{f}{op}{fmt(v, ctx)}" for f, op, v in where)


# ---------------------------------------------------------------- state checks
def _check_created(part: dict, sn, ctx: dict) -> list[dict]:
    query = "^".join(x for x in [since(ctx), encode_where(part.get("where") or [], ctx)] if x)
    rows = sn.query(part["table"], query, limit=50)
    want = part.get("count")
    name = f"{part['table']}_created"
    if want is not None:
        return [_check(name, len(rows) == want, f"{len(rows)} row(s) match, expected exactly {want}")]
    return [_check(name, len(rows) >= 1, f"{len(rows)} row(s) match")]


def _check_field_equals(part: dict, sn, ctx: dict) -> list[dict]:
    rec = ctx["records"].get(part["alias"])
    if not rec:
        return [_check("record", False, f"no setup record {part['alias']}")]
    try:
        row = sn.get(part.get("table") or "incident", rec["sys_id"])
    except Exception as exc:  # noqa: BLE001
        return [_check("record", False, f"cannot read {part['alias']}: {exc}")]
    out = []
    for field, expected in part["fields"].items():
        expected = fmt(expected, ctx)
        have = row.get(field)
        out.append(_check(f"{part['alias']}.{field}", _norm(have) == _norm(expected), f"have {have!r} want {expected!r}"))
    return out


def _checks(part: dict, sn, ctx: dict) -> list[dict]:
    kind = part.get("kind")
    if kind == "created":
        return _check_created(part, sn, ctx)
    if kind == "field_equals":
        return _check_field_equals(part, sn, ctx)
    if kind == "composite":
        rows = []
        for i, sub in enumerate(part.get("parts") or [], 1):
            for c in _checks(sub, sn, ctx) or [_check("unknown_part", False, str(sub.get("kind")))]:
                rows.append({**c, "check": f"{i}.{c['check']}"})
        return rows or [_check("composite", False, "no parts")]
    if kind == "report":
        return []
    return [_check("kind", False, f"unknown constraint kind {kind!r}")]


def verify_task(task_id: str, sn, ctx: dict) -> dict:
    task = SPEC["tasks"][task_id]
    if task["verification"]["type"] != "state":
        return {"task": task_id, "ok": None, "source": "none", "checks": []}
    checks = _checks(task["constraints"], sn, ctx)
    oks = [c["ok"] for c in checks]
    return {"task": task_id, "ok": all(o is True for o in oks) if oks else None,
            "source": "servicenow_rest", "checks": checks}


# ---------------------------------------------------------------- agent report
def expected_value(constraints: dict, sn, ctx: dict) -> Any:
    if "expected_count" in constraints:
        spec = constraints["expected_count"]
        return sn.count(spec["table"], encode_where(spec["where"], ctx))
    return fmt(constraints["expected"], ctx)


def _same(expected: Any, got: Any, digits: bool) -> bool:
    if digits:
        return bool(_digits(expected)) and _digits(expected) == _digits(got)
    if isinstance(expected, int):
        m = re.search(r"-?\d+", str(got))
        return m is not None and int(m.group()) == expected
    e, g = _norm(expected), _norm(got)
    return bool(e) and (e == g or e in g)


def score_agent_report(task_id: str, report: dict | None, sn, ctx: dict, path: str | None = None) -> dict:
    task = SPEC["tasks"][task_id]
    cons = task["constraints"]
    if report is None:
        return {"task": task_id, "ok": False, "path": path, "checks": [_check("report", False, "no report found")]}
    checks = [_check("reported_success", report.get("success") is True or report.get("success") is None,
                     f"success={report.get('success')!r}")]
    # find the report constraint (top level, or inside a composite)
    parts = [cons] if cons.get("kind") == "report" else [p for p in cons.get("parts", []) if p.get("kind") == "report"]
    for part in parts:
        want = expected_value(part, sn, ctx)
        got = report.get(part["key"])
        checks.append(_check(part["key"], _same(want, got, bool(part.get("digits"))), f"want {want!r} got {got!r}"))
    if cons.get("kind") != "report":
        for key in (task.get("report_fields") or {}):
            pass  # composite state tasks are scored by verify_task; the report only has to exist
    oks = [c["ok"] for c in checks]
    return {"task": task_id, "ok": all(o is True for o in oks), "path": path, "checks": checks}


def format_verify(result: dict) -> str:
    if result.get("ok") is None and not result.get("checks"):
        return f"verify {result['task']}: n/a (this task is scored from the agent's report)"
    rows = "; ".join(f"{c['check']}={'ok' if c['ok'] else 'FAIL'}" for c in result.get("checks", []))
    return f"verify {result['task']}: {result['ok']}  {rows}"


def format_agent_report(result: dict) -> str:
    rows = "; ".join(f"{c['check']}={'ok' if c['ok'] else 'FAIL'}" for c in result.get("checks", []))
    return f"report {result['task']}: {result['ok']}  {rows}"


# ---------------------------------------------------------------- self test
def _self_test() -> int:
    from session import cleanup, prepare

    bad = 0

    def expect(label: str, got, want) -> None:
        nonlocal bad
        if got != want:
            bad += 1
            print(f"FAIL {label}: got {got!r} want {want!r}")

    ids = list(SPEC["tasks"])
    expect("task count", len(ids), 38)
    for tid in ids:  # setup and all templates must resolve
        sn = FakeSN()
        ctx = prepare(SPEC["tasks"][tid], sn)
        fmt(SPEC["tasks"][tid]["ask"], ctx)
        fmt(SPEC["tasks"][tid]["constraints"], ctx)

    def run(tid):
        sn = FakeSN()
        return sn, prepare(SPEC["tasks"][tid], sn)

    # report tasks: right answer passes, wrong answer fails
    sn, ctx = run("SN01")
    right = ctx["records"]["inc2"]["number"]
    expect("SN01 right", score_agent_report("SN01", {"success": True, "number": right}, sn, ctx)["ok"], True)
    expect("SN01 wrong", score_agent_report("SN01", {"success": True, "number": ctx["records"]["inc1"]["number"]}, sn, ctx)["ok"], False)
    sn, ctx = run("SN11")  # enumeration count of 6
    expect("count right", score_agent_report("SN11", {"success": True, "count": "6"}, sn, ctx)["ok"], True)
    expect("count wrong", score_agent_report("SN11", {"success": True, "count": 5}, sn, ctx)["ok"], False)
    sn, ctx = run("SN06")  # urgency 2 reported with its label
    expect("urgency label", score_agent_report("SN06", {"success": True, "urgency": "2 - Medium"}, sn, ctx)["ok"], True)
    sn, ctx = run("SN10")  # phone digits
    expect("phone", score_agent_report("SN10", {"success": True, "phone": "(503) 555 0199"}, sn, ctx)["ok"], True)
    # field_equals: unchanged fails, changed passes
    sn, ctx = run("SN16")
    expect("urgency unchanged", verify_task("SN16", sn, ctx)["ok"], False)
    sys_id = ctx["records"]["inc1"]["sys_id"]
    sn.tables["incident"][sys_id]["urgency"] = "1"
    expect("urgency changed", verify_task("SN16", sn, ctx)["ok"], True)
    # created: nothing -> fail; matching row -> pass; two RITMs -> fail for count=1
    sn, ctx = run("SN21")
    expect("order none", verify_task("SN21", sn, ctx)["ok"], False)
    sn.insert("sc_req_item", {"cat_item.name": "Standard Laptop"})
    expect("order one", verify_task("SN21", sn, ctx)["ok"], True)
    sn.insert("sc_req_item", {"cat_item.name": "Standard Laptop"})
    expect("order twice", verify_task("SN21", sn, ctx)["ok"], False)
    sn, ctx = run("SN31")
    n = ctx["nonce"]
    sn.insert("incident", {"short_description": f"Conference phone dead {n}", "urgency": "2", "category": "network"})
    expect("incident created", verify_task("SN31", sn, ctx)["ok"], True)
    # composite transition
    sn, ctx = run("SN37")
    expect("composite incomplete", verify_task("SN37", sn, ctx)["ok"], False)
    sn.insert("sys_user", {"user_name": f"wa.{ctx['nonce']}.t", "email": f"tess.moon.{ctx['nonce']}@hearthline.test"})
    sn.insert("incident", {"short_description": f"Laptop for wa.{ctx['nonce']}.t", "urgency": "3"})
    expect("composite done", verify_task("SN37", sn, ctx)["ok"], True)
    # cleanup removes everything
    removed = cleanup(sn, ctx)
    expect("cleanup", sum(len(t) for t in sn.tables.values()), 0)
    print("self-test ok" if not bad else f"{bad} failure(s)", f"(cleanup removed {removed})")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print(__doc__)
