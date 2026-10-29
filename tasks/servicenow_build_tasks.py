#!/usr/bin/env python3
"""Generate servicenow/tasks.json (38 tasks, 7 objectives + 3 transitions).

These are our own tasks on a ServiceNow instance, modelled on the WorkArena task families
(forms, lists, knowledge/catalog lookups, service-catalog orders) but written with explicit
asks and REST-checkable outcomes, so they run through the same pipeline as Ridgeway.

Templating (resolved per session by prompts.py / verify.py):
  {{nonce}}        6-char token unique to the session, so records never collide between runs
  {{alias.field}}  field of a record created by `setup` (number, sys_id, user_name, ...)
Where conditions are [field, op, value] with op in "=", "!=", "LIKE".
"""
from __future__ import annotations

import json
from pathlib import Path

N = "{{nonce}}"
# Classic UI. The Next Experience home (/now/nav/ui/home) is too slow for the agent:
# it spends the whole timeout reloading that shell instead of doing the task.
LIST = {"incident": "/incident_list.do", "problem": "/problem_list.do",
        "change_request": "/change_request_list.do", "sys_user": "/sys_user_list.do"}
FORM = {"incident": "/incident.do", "problem": "/problem.do",
        "change_request": "/change_request.do", "sys_user": "/sys_user.do"}
CATALOG = "/com.glideapp.servicecatalog_home.do"
CALLER = "If Caller is required, use the user you are logged in as."
TASKS: list[dict] = []


def inc(alias, summary, **kw):
    return {"alias": alias, "table": "incident", "fields": {"short_description": summary, **kw}}


def task(objective, label, ask, constraints, verification, setup=None, report=None, hint=None, seq=None,
         start="/incident_list.do"):
    t = {"label": label, "objective": objective, "ask": ask, "start_path": start,
         "setup": setup or [], "constraints": constraints,
         "verification": {"type": verification}}
    if report:
        t["report_fields"] = report
    if hint:
        t["hint"] = hint
    if seq:
        t["objective_sequence"] = seq
    TASKS.append(t)


def rep(key, expected, **kw):
    return {"kind": "report", "key": key, "expected": expected, **kw}


def form_for(table: str, alias: str) -> str:
    field = "user_name" if table == "sys_user" else "number"
    return f"{FORM[table]}?sysparm_query={field}={{{{{alias}.{field}}}}}"


def new_form(table: str) -> str:
    return f"{FORM[table]}?sys_id=-1"


# ------------------------------------------------------------------ discovery
task("discovery", "Find incident by description",
     f"Find the incident whose short description mentions a badge reader (code {N}). Report its incident number.",
     rep("number", "{{inc2.number}}"), "self_report",
     setup=[inc("inc1", f"Printer jam on floor 3 {N}"), inc("inc2", f"Badge reader offline at lobby {N}"),
            inc("inc3", f"Projector flicker in room 12 {N}")],
     report={"number": "INC… or null"}, hint="Use the Incident list and its search or filter. Do not use the REST API.")
task("discovery", "Find problem by description",
     f"Find the problem record whose short description mentions a database replication lag (code {N}). Report its problem number.",
     rep("number", "{{p2.number}}"), "self_report",
     setup=[{"alias": "p1", "table": "problem", "fields": {"short_description": f"Slow login page {N}"}},
            {"alias": "p2", "table": "problem", "fields": {"short_description": f"Database replication lag overnight {N}"}}],
     report={"number": "PRB… or null"}, start=LIST["problem"])
task("discovery", "Find change request by description",
     f"Find the change request whose short description mentions a firewall rule update (code {N}). Report its change number.",
     rep("number", "{{c2.number}}"), "self_report",
     setup=[{"alias": "c1", "table": "change_request", "fields": {"short_description": f"Patch kiosk image {N}"}},
            {"alias": "c2", "table": "change_request", "fields": {"short_description": f"Firewall rule update for guest wifi {N}"}}],
     report={"number": "CHG… or null"}, start=LIST["change_request"])
task("discovery", "Find user by job title",
     f"Find the user whose job title is 'Facilities Coordinator {N}'. Report that user's User ID.",
     rep("user_name", "{{u2.user_name}}"), "self_report",
     setup=[{"alias": "u1", "table": "sys_user", "fields": {"user_name": f"wa.{N}.a", "first_name": "Ari", "last_name": f"Lund{N}", "title": f"Planner {N}"}},
            {"alias": "u2", "table": "sys_user", "fields": {"user_name": f"wa.{N}.b", "first_name": "Bea", "last_name": f"Ortiz{N}", "title": f"Facilities Coordinator {N}"}}],
     report={"user_name": "user id or null"}, start=LIST["sys_user"])
task("discovery", "Find most urgent incident in a set",
     f"Among the incidents whose short description contains 'Queue {N}', find the one with the highest urgency (1 - High). Report its incident number.",
     rep("number", "{{q1.number}}"), "self_report",
     setup=[inc("q1", f"Queue {N} alpha", urgency="1"), inc("q2", f"Queue {N} beta", urgency="2"),
            inc("q3", f"Queue {N} gamma", urgency="3")],
     report={"number": "INC… or null"})

# ------------------------------------------------------------------ retrieval
task("retrieval", "Read an incident field",
     "Open incident {{inc1.number}} and report its urgency.",
     rep("urgency", "{{inc1.urgency}}"), "self_report",
     setup=[inc("inc1", f"Network drop in annex {N}", urgency="2", category="network")],
     report={"urgency": "value shown on the form"}, start=form_for("incident", "inc1"))
task("retrieval", "Read a user email",
     "Open the user record with User ID {{u1.user_name}} and report the email address.",
     rep("email", "{{u1.email}}"), "self_report",
     setup=[{"alias": "u1", "table": "sys_user", "fields": {"user_name": f"wa.{N}.r", "first_name": "Rui", "last_name": f"Park{N}", "email": f"rui.park.{N}@hearthline.test"}}],
     report={"email": "address or null"}, start=form_for("sys_user", "u1"))
task("retrieval", "Read a value from a description",
     "Open incident {{inc1.number}} and report the access code written in its description.",
     rep("code", "ZX{{nonce}}-77"), "self_report",
     setup=[inc("inc1", f"Locker access issue {N}", description=f"Employee cannot open locker. Access code: ZX{N}-77")],
     report={"code": "the code or null"}, start=form_for("incident", "inc1"))
task("retrieval", "Read a change request type",
     "Open change request {{c1.number}} and report its type.",
     rep("type", "{{c1.type}}"), "self_report",
     setup=[{"alias": "c1", "table": "change_request", "fields": {"short_description": f"Rotate certificates {N}", "type": "emergency"}}],
     report={"type": "Normal, Standard or Emergency"}, start=form_for("change_request", "c1"))
task("retrieval", "Read a user phone number",
     "Open the user record with User ID {{u1.user_name}} and report the mobile phone number.",
     rep("phone", "503-555-0199", digits=True), "self_report",
     setup=[{"alias": "u1", "table": "sys_user", "fields": {"user_name": f"wa.{N}.p", "first_name": "Pia", "last_name": f"Nolan{N}", "mobile_phone": "503-555-0199"}}],
     report={"phone": "number or null"}, start=form_for("sys_user", "u1"))

# ------------------------------------------------------------------ enumeration
batch = [inc(f"b{i}", f"Batch {N} item {i}", urgency=u) for i, u in enumerate("112223", 1)]
task("enumeration", "Count records in a list",
     f"How many incidents have 'Batch {N}' in their short description? Report the count.",
     {"kind": "report", "key": "count", "expected_count": {"table": "incident", "where": [["short_description", "LIKE", f"Batch {N}"]]}},
     "self_report", setup=batch, report={"count": 0})
task("enumeration", "Count with a filter",
     f"Among the incidents with 'Batch {N}' in their short description, how many have urgency 1 - High? Report the count.",
     {"kind": "report", "key": "count", "expected_count": {"table": "incident", "where": [["short_description", "LIKE", f"Batch {N}"], ["urgency", "=", "1"]]}},
     "self_report", setup=batch, report={"count": 0})
task("enumeration", "Count problems",
     f"How many problem records have 'Sweep {N}' in their short description? Report the count.",
     {"kind": "report", "key": "count", "expected_count": {"table": "problem", "where": [["short_description", "LIKE", f"Sweep {N}"]]}},
     "self_report", setup=[{"alias": f"p{i}", "table": "problem", "fields": {"short_description": f"Sweep {N} #{i}"}} for i in range(1, 5)],
     report={"count": 0}, start=LIST["problem"])
task("enumeration", "Sort a list and read the top row",
     f"List the incidents with 'Order {N}' in their short description, sort them by short description from Z to A, and report the incident number in the first row.",
     rep("number", "{{o5.number}}"), "self_report",
     setup=[inc(f"o{i}", f"Order {N} {c}") for i, c in enumerate("ABCDE", 1)], report={"number": "INC… or null"})
task("enumeration", "Count users with a title",
     f"How many users have the job title 'Intern {N}'? Report the count.",
     {"kind": "report", "key": "count", "expected_count": {"table": "sys_user", "where": [["title", "=", f"Intern {N}"]]}},
     "self_report", setup=[{"alias": f"i{i}", "table": "sys_user", "fields": {"user_name": f"wa.{N}.i{i}", "first_name": f"Intern{i}", "last_name": f"Row{N}", "title": f"Intern {N}"}} for i in range(1, 4)],
     report={"count": 0}, start=LIST["sys_user"])

# ------------------------------------------------------------------ curation (edit existing records)
task("curation", "Change incident urgency",
     "Open incident {{inc1.number}} and set its urgency to 1 - High. Save it.",
     {"kind": "field_equals", "alias": "inc1", "table": "incident", "fields": {"urgency": "1"}}, "state",
     setup=[inc("inc1", f"Slow VPN {N}", urgency="3")], start=form_for("incident", "inc1"))
task("curation", "Change incident category",
     "Open incident {{inc1.number}} and change its category to Hardware. Save it.",
     {"kind": "field_equals", "alias": "inc1", "table": "incident", "fields": {"category": "hardware"}}, "state",
     setup=[inc("inc1", f"Dock not charging {N}", category="software")], start=form_for("incident", "inc1"))
task("curation", "Change problem impact",
     "Open problem {{p1.number}} and set its impact to 1 - High. Save it.",
     {"kind": "field_equals", "alias": "p1", "table": "problem", "fields": {"impact": "1"}}, "state",
     setup=[{"alias": "p1", "table": "problem", "fields": {"short_description": f"Repeated timeouts {N}", "impact": "3"}}],
     start=form_for("problem", "p1"))
task("curation", "Rename a change request",
     f"Open change request {{{{c1.number}}}} and change its short description to 'Renewed certificates {N}'. Save it.",
     {"kind": "field_equals", "alias": "c1", "table": "change_request", "fields": {"short_description": f"Renewed certificates {N}"}}, "state",
     setup=[{"alias": "c1", "table": "change_request", "fields": {"short_description": f"Renew certificates {N}"}}],
     start=form_for("change_request", "c1"))
task("curation", "Move incident to In Progress",
     "Open incident {{inc1.number}} and set its state to In Progress. Save it.",
     {"kind": "field_equals", "alias": "inc1", "table": "incident", "fields": {"state": "2"}}, "state",
     setup=[inc("inc1", f"Keyboard replacement {N}", state="1")], start=form_for("incident", "inc1"))

# ------------------------------------------------------------------ transaction (service catalog)
for name, query in [("Standard Laptop", "Standard Laptop"), ("Sales Laptop", "Sales Laptop"), ("iPad mini", "iPad mini"),
                    ("iPad pro", "iPad pro"), ("Apple Watch", "Apple Watch")]:
    task("transaction", f"Order {name}",
         f"Order one {name} from the Service Catalog with the default options and submit the order.",
         {"kind": "created", "table": "sc_req_item", "where": [["cat_item.name", "LIKE", query]], "count": 1}, "state",
         report={"request_number": "REQ… or null"},
         hint="Use Service Catalog > Hardware. Finish with Order Now. Do not call the REST API.",
         start=CATALOG)

# ------------------------------------------------------------------ identity
task("identity", "Create a user",
     f"Create a new user with User ID wa.{N}.new, first name Maya, last name Chen{N}, and email maya.chen.{N}@hearthline.test. Save it.",
     {"kind": "created", "table": "sys_user", "where": [["user_name", "=", f"wa.{N}.new"], ["first_name", "=", "Maya"], ["email", "=", f"maya.chen.{N}@hearthline.test"]]},
     "state", report={"user_name": "user id or null"}, start=new_form("sys_user"))
task("identity", "Create a user with contact fields",
     f"Create a new user with User ID wa.{N}.c, first name Dev, last name Rao{N}, job title Analyst, and mobile phone 503-555-0142. Save it.",
     {"kind": "created", "table": "sys_user", "where": [["user_name", "=", f"wa.{N}.c"], ["title", "=", "Analyst"], ["mobile_phone", "=", "503-555-0142"]]},
     "state", report={"user_name": "user id or null"}, start=new_form("sys_user"))
task("identity", "Change a user's email",
     f"Open the user with User ID {{{{u1.user_name}}}} and change the email to renamed.{N}@hearthline.test. Save it.",
     {"kind": "field_equals", "alias": "u1", "table": "sys_user", "fields": {"email": f"renamed.{N}@hearthline.test"}}, "state",
     setup=[{"alias": "u1", "table": "sys_user", "fields": {"user_name": f"wa.{N}.e", "first_name": "Eli", "last_name": f"Voss{N}", "email": f"eli.{N}@hearthline.test"}}],
     start=form_for("sys_user", "u1"))
task("identity", "Change a user's phone",
     "Open the user with User ID {{u1.user_name}} and change the mobile phone to 503-555-0177. Save it.",
     {"kind": "field_equals", "alias": "u1", "table": "sys_user", "fields": {"mobile_phone": "503-555-0177"}}, "state",
     setup=[{"alias": "u1", "table": "sys_user", "fields": {"user_name": f"wa.{N}.m", "first_name": "Mo", "last_name": f"Lee{N}", "mobile_phone": "503-555-0100"}}],
     start=form_for("sys_user", "u1"))
task("identity", "Deactivate a user",
     "Open the user with User ID {{u1.user_name}} and make the account inactive (clear the Active checkbox). Save it.",
     {"kind": "field_equals", "alias": "u1", "table": "sys_user", "fields": {"active": "false"}}, "state",
     setup=[{"alias": "u1", "table": "sys_user", "fields": {"user_name": f"wa.{N}.d", "first_name": "Dee", "last_name": f"Kim{N}", "active": "true"}}],
     start=form_for("sys_user", "u1"))

# ------------------------------------------------------------------ service (raise a request)
task("service", "Raise an incident",
     f"Create an incident with short description 'Conference phone dead {N}', description 'No dial tone in room 4', urgency 2 - Medium and category Network. {CALLER} Submit it.",
     {"kind": "created", "table": "incident", "where": [["short_description", "=", f"Conference phone dead {N}"], ["urgency", "=", "2"], ["category", "=", "network"]]},
     "state", report={"number": "INC… or null"}, start=new_form("incident"))
task("service", "Raise a high-urgency incident",
     f"Create an incident with short description 'Payroll export failing {N}', urgency 1 - High and category Software. {CALLER} Submit it.",
     {"kind": "created", "table": "incident", "where": [["short_description", "=", f"Payroll export failing {N}"], ["urgency", "=", "1"], ["category", "=", "software"]]},
     "state", report={"number": "INC… or null"}, start=new_form("incident"))
task("service", "Raise a hardware incident",
     f"Create an incident with short description 'Monitor flickers {N}', urgency 3 - Low and category Hardware. {CALLER} Submit it.",
     {"kind": "created", "table": "incident", "where": [["short_description", "=", f"Monitor flickers {N}"], ["urgency", "=", "3"], ["category", "=", "hardware"]]},
     "state", report={"number": "INC… or null"}, start=new_form("incident"))
task("service", "Open a problem",
     f"Create a problem with short description 'Recurring login timeouts {N}' and description 'Reported by several users this week'. Submit it.",
     {"kind": "created", "table": "problem", "where": [["short_description", "=", f"Recurring login timeouts {N}"]]},
     "state", report={"number": "PRB… or null"}, start=new_form("problem"))
task("service", "Open a change request",
     f"Create a normal change request with short description 'Upgrade wiki server {N}' and description 'Planned maintenance'. Submit it.",
     {"kind": "created", "table": "change_request", "where": [["short_description", "=", f"Upgrade wiki server {N}"], ["type", "=", "normal"]]},
     "state", report={"number": "CHG… or null"}, start=new_form("change_request"))

# ------------------------------------------------------------------ transitions (objective changes mid-session)
task("discovery", "Find then edit an incident",
     f"Find the incident whose short description mentions a badge reader (code {N}), then set its urgency to 1 - High and save it.",
     {"kind": "composite", "parts": [{"kind": "field_equals", "alias": "inc2", "table": "incident", "fields": {"urgency": "1"}}]},
     "state",
     setup=[inc("inc1", f"Printer jam on floor 3 {N}", urgency="3"), inc("inc2", f"Badge reader offline at lobby {N}", urgency="3")],
     seq=["discovery", "curation"], report={"number": "INC… or null"})
task("identity", "Create a user then raise an incident",
     f"Create a new user with User ID wa.{N}.t, first name Tess, last name Moon{N} and email tess.moon.{N}@hearthline.test. Then create an incident with short description 'Laptop for wa.{N}.t' and urgency 3 - Low. {CALLER}",
     {"kind": "composite", "parts": [
         {"kind": "created", "table": "sys_user", "where": [["user_name", "=", f"wa.{N}.t"], ["email", "=", f"tess.moon.{N}@hearthline.test"]]},
         {"kind": "created", "table": "incident", "where": [["short_description", "=", f"Laptop for wa.{N}.t"], ["urgency", "=", "3"]]}]},
     "state", seq=["identity", "service"], report={"user_name": "user id or null", "number": "INC… or null"},
     start=new_form("sys_user"))
task("enumeration", "Count then report in a new incident",
     f"Count the incidents with 'Batch {N}' in their short description, then create an incident with short description 'Batch report {N}' and description 'Count: <that number>'. {CALLER}",
     {"kind": "composite", "parts": [
         {"kind": "created", "table": "incident", "where": [["short_description", "=", f"Batch report {N}"], ["description", "LIKE", "Count: 4"]]}]},
     "state", setup=[inc(f"b{i}", f"Batch {N} item {i}") for i in range(1, 5)],
     seq=["enumeration", "service"], report={"count": 0, "number": "INC… or null"})

OBJECTIVES = ["discovery", "retrieval", "enumeration", "curation", "transaction", "identity", "service"]
spec = {"version": "1", "site": "servicenow", "objectives": OBJECTIVES,
        "agents": ["browser-use", "skyvern", "human"], "tasks": {}}
for i, t in enumerate(TASKS, 1):
    spec["tasks"][f"SN{i:02d}"] = t
Path(__file__).with_name("tasks.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print(len(TASKS), "tasks")
