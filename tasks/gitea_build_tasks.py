#!/usr/bin/env python3
"""Generate gitea/tasks.json: 43 tasks, 8 objectives x 5 + 3 transitions, on a self-hosted Gitea.

Mapping of the paper's objectives onto a git forge (two are analogies, flagged in RUNBOOK.md):
  discovery     find a repo / issue / user / pull request by description
  retrieval     read one field of a named record
  enumeration   counts and sorted-list top rows
  curation      edit an existing issue or repository
  service       raise a request: open an issue or PR, comment
  identity      create or change a user account
  transaction   irreversible commit: merge a PR, delete a repo, publish a release   (analogy)
  availability  subscribe to future events: watch a repo, subscribe to an issue/PR   (analogy)

Templates: {{nonce}} and {{alias.field}} (see session.py). The admin account is `studyadmin`.
"""
from __future__ import annotations

import json
from pathlib import Path

N = "{{nonce}}"
ADMIN = "studyadmin"
WEB = "Use the Gitea web UI only. Do not use the API, git over SSH, or the browser console."
TASKS: list[dict] = []


def user(alias, login, full_name="", email=None):
    d = {"alias": alias, "op": "user", "login": login, "full_name": full_name}
    if email:
        d["email"] = email
    return d


def repo(alias, name, owner=ADMIN, desc="", branch="main"):
    return {"alias": alias, "op": "repo", "owner": owner, "name": name, "description": desc, "default_branch": branch}


def label(alias, r, name):
    return {"alias": alias, "op": "label", "repo": r, "name": name}


def issue(alias, r, title, body="", **kw):
    return {"alias": alias, "op": "issue", "repo": r, "title": title, "body": body, **kw}


def pr(alias, r, branch, title, **kw):
    return {"alias": alias, "op": "pr", "repo": r, "branch": branch, "title": title, **kw}


def task(objective, label_, ask, constraints, verification, setup=None, report=None, seq=None):
    t = {"label": label_, "objective": objective, "ask": ask, "start_path": "/", "setup": setup or [],
         "constraints": constraints, "verification": {"type": verification}, "hint": WEB}
    if report:
        t["report_fields"] = report
    if seq:
        t["objective_sequence"] = seq
    TASKS.append(t)


def rep(key, expected, **kw):
    return {"kind": "report", "key": key, "expected": expected, **kw}


def count(key, path):
    return {"kind": "report", "key": key, "expected_count": {"path": path}}


def rest(path, status=200, **js):
    return {"kind": "rest", "path": path, "status": status, "json": js}


def rlist(path, count=None, **any_):
    return {"kind": "rest_list", "path": path, "any": any_, **({"count": count} if count is not None else {})}


R1 = "{{r1.full_name}}"
ISS = f"/repos/{R1}/issues?state=all&type=issues"

# ------------------------------------------------------------------ discovery
task("discovery", "Find a repository by description",
     f"Find the repository owned by {ADMIN} whose description mentions badge printing (code {N}). Report its repository name.",
     rep("name", "{{r2.name}}"), "self_report",
     setup=[repo("r1", f"alpha-{N}", desc=f"Inventory sync service {N}"), repo("r2", f"beta-{N}", desc=f"Badge printing tool {N}"),
            repo("r3", f"gamma-{N}", desc=f"Release notes generator {N}")], report={"name": "repository name or null"})
task("discovery", "Find an issue by description",
     f"In the repository {R1}, find the issue about exporting to CSV and report its issue number.",
     rep("number", "{{i2.number}}", digits=True), "self_report",
     setup=[repo("r1", f"proj-{N}"), issue("i1", "r1", f"Login page crashes on empty password {N}"),
            issue("i2", "r1", f"Export to CSV drops the header row {N}"), issue("i3", "r1", f"Dark mode flickers on reload {N}")],
     report={"number": "issue number or null"})
task("discovery", "Find a user by full name",
     f"Find the user whose full name is 'Facilities Lead {N}'. Report that user's username.",
     rep("username", "{{u2.login}}"), "self_report",
     setup=[user("u1", f"pl-{N}", f"Planner {N}"), user("u2", f"fl-{N}", f"Facilities Lead {N}")],
     report={"username": "username or null"})
task("discovery", "Find an issue by label",
     f"In the repository {R1}, find the open issue that has the label 'urgent-{N}'. Report its issue number.",
     rep("number", "{{i2.number}}", digits=True), "self_report",
     setup=[repo("r1", f"proj-{N}"), label("l1", "r1", f"urgent-{N}"), issue("i1", "r1", f"Typo on landing page {N}"),
            issue("i2", "r1", f"Checkout fails for guests {N}", labels=["l1"]), issue("i3", "r1", f"Update screenshots {N}")],
     report={"number": "issue number or null"})
task("discovery", "Find a pull request by description",
     f"In the repository {R1}, find the open pull request that fixes a typo. Report its pull request number.",
     rep("number", "{{p1.number}}", digits=True), "self_report",
     setup=[repo("r1", f"proj-{N}"), pr("p1", "r1", f"typo-{N}", f"Fix typo in README {N}"), pr("p2", "r1", f"export-{N}", f"Add export button {N}")],
     report={"number": "pull request number or null"})

# ------------------------------------------------------------------ retrieval
task("retrieval", "Read a value from an issue description",
     f"Open issue #{{{{i1.number}}}} in {R1} and report the locker code written in its description.",
     rep("code", f"ZX{N}-77"), "self_report",
     setup=[repo("r1", f"proj-{N}"), issue("i1", "r1", f"Locker will not open {N}", body=f"Employee cannot open the locker. Locker code: ZX{N}-77")],
     report={"code": "the code or null"})
task("retrieval", "Read a repository's default branch",
     f"Open the repository {R1} and report its default branch name.",
     rep("branch", "trunk"), "self_report",
     setup=[repo("r1", f"proj-{N}", branch="trunk")], report={"branch": "branch name or null"})
task("retrieval", "Read a user's email",
     "Find the email address of the user {{u1.login}} (Site Administration, User Accounts) and report it.",
     rep("email", "{{u1.email}}"), "self_report",
     setup=[user("u1", f"rp-{N}", f"Rui Park {N}", email=f"rui.park.{N}@hearthline.test")], report={"email": "address or null"})
task("retrieval", "Read an issue's assignee",
     f"Open issue #{{{{i1.number}}}} in {R1} and report the username it is assigned to.",
     rep("assignee", "{{u1.login}}"), "self_report",
     setup=[user("u1", f"as-{N}", f"Asha Rao {N}"), repo("r1", f"proj-{N}", owner="{{u1.login}}"),
            issue("i1", "r1", f"Review release checklist {N}", assignees=["{{u1.login}}"])], report={"assignee": "username or null"})
task("retrieval", "Read an issue's label",
     f"Open issue #{{{{i1.number}}}} in {R1} and report the name of its label.",
     rep("label", "{{l1.name}}"), "self_report",
     setup=[repo("r1", f"proj-{N}"), label("l1", "r1", f"needs-info-{N}"), issue("i1", "r1", f"Cannot reproduce on Safari {N}", labels=["l1"])],
     report={"label": "label name or null"})

# ------------------------------------------------------------------ enumeration
batch = [issue(f"b{i}", "r1", f"Batch {N} item {i}") for i in range(1, 7)]
task("enumeration", "Count issues in a repository",
     f"How many issues (open and closed) does the repository {R1} have? Report the count.",
     count("count", ISS), "self_report", setup=[repo("r1", f"proj-{N}")] + batch, report={"count": 0})
task("enumeration", "Count closed issues",
     f"How many closed issues does the repository {R1} have? Report the count.",
     count("count", f"/repos/{R1}/issues?state=closed&type=issues"), "self_report",
     setup=[repo("r1", f"proj-{N}")] + [issue(f"b{i}", "r1", f"Batch {N} item {i}", closed=i <= 2) for i in range(1, 6)],
     report={"count": 0})
task("enumeration", "Count a user's repositories",
     "How many repositories does the user {{u1.login}} own? Report the count.",
     count("count", "/users/{{u1.login}}/repos"), "self_report",
     setup=[user("u1", f"co-{N}", f"Collector {N}")] + [repo(f"r{i}", f"stash-{N}-{i}", owner="{{u1.login}}") for i in range(1, 5)],
     report={"count": 0})
task("enumeration", "Sort the issue list and read the first row",
     f"Open the issue list of {R1}, sort it by Oldest, and report the title of the first issue.",
     rep("title", f"Order {N} A"), "self_report",
     setup=[repo("r1", f"proj-{N}")] + [issue(f"o{i}", "r1", f"Order {N} {c}") for i, c in enumerate("ABCDE", 1)],
     report={"title": "issue title or null"})
task("enumeration", "Count labels",
     f"How many labels does the repository {R1} define? Report the count.",
     count("count", f"/repos/{R1}/labels"), "self_report",
     setup=[repo("r1", f"proj-{N}")] + [label(f"l{i}", "r1", f"area-{N}-{i}") for i in range(1, 6)], report={"count": 0})

# ------------------------------------------------------------------ curation (edit existing)
task("curation", "Rename an issue",
     f"Open issue #{{{{i1.number}}}} in {R1} and change its title to 'Search is slow {N}'. Save it.",
     rest("/repos/" + R1 + "/issues/{{i1.number}}", title=f"Search is slow {N}"), "state",
     setup=[repo("r1", f"proj-{N}"), issue("i1", "r1", f"Slow search {N}")])
task("curation", "Add a label to an issue",
     f"Open issue #{{{{i1.number}}}} in {R1} and add the label 'bug-{N}' to it.",
     rest("/repos/" + R1 + "/issues/{{i1.number}}", **{"labels[].name": f"bug-{N}"}), "state",
     setup=[repo("r1", f"proj-{N}"), label("l1", "r1", f"bug-{N}"), issue("i1", "r1", f"Crash on logout {N}")])
task("curation", "Assign an issue",
     f"Open issue #{{{{i1.number}}}} in {R1} and assign it to the user {{{{u1.login}}}}.",
     rest("/repos/" + R1 + "/issues/{{i1.number}}", **{"assignees[].login": "{{u1.login}}"}), "state",
     setup=[user("u1", f"ow-{N}", f"Owner {N}"), repo("r1", f"proj-{N}", owner="{{u1.login}}"), issue("i1", "r1", f"Tidy changelog {N}")])
task("curation", "Close an issue",
     f"Open issue #{{{{i1.number}}}} in {R1} and close it.",
     rest("/repos/" + R1 + "/issues/{{i1.number}}", state="closed"), "state",
     setup=[repo("r1", f"proj-{N}"), issue("i1", "r1", f"Remove old banner {N}")])
task("curation", "Edit a repository description",
     f"Open the settings of the repository {R1} and change its description to 'Updated description {N}'. Save it.",
     rest("/repos/" + R1, description=f"Updated description {N}"), "state",
     setup=[repo("r1", f"proj-{N}", desc="Old description")])

# ------------------------------------------------------------------ service (raise a request)
task("service", "Open an issue",
     f"In the repository {R1}, open a new issue titled 'Search timeout {N}' with the description 'Search takes over 30 seconds'.",
     rlist(ISS, title=f"Search timeout {N}", body="~over 30 seconds"), "state",
     setup=[repo("r1", f"proj-{N}")], report={"number": "issue number or null"})
task("service", "Open an issue with a label",
     f"In the repository {R1}, open a new issue titled 'Crash on save {N}' and give it the label 'bug-{N}'.",
     rlist(ISS, title=f"Crash on save {N}", **{"labels[].name": f"bug-{N}"}), "state",
     setup=[repo("r1", f"proj-{N}"), label("l1", "r1", f"bug-{N}")], report={"number": "issue number or null"})
task("service", "Comment on an issue",
     f"Open issue #{{{{i1.number}}}} in {R1} and post the comment 'Reproduced on v2 {N}'.",
     rlist(f"/repos/{R1}/issues/{{{{i1.number}}}}/comments", body=f"Reproduced on v2 {N}"), "state",
     setup=[repo("r1", f"proj-{N}"), issue("i1", "r1", f"Intermittent 500 error {N}")])
task("service", "Open an issue assigned to someone",
     f"In the repository {R1}, open a new issue titled 'Review needed {N}' and assign it to {{{{u1.login}}}}.",
     rlist(ISS, title=f"Review needed {N}", **{"assignees[].login": "{{u1.login}}"}), "state",
     setup=[user("u1", f"rv-{N}", f"Reviewer {N}"), repo("r1", f"proj-{N}", owner="{{u1.login}}")],
     report={"number": "issue number or null"})
task("service", "Open a pull request",
     f"In the repository {R1}, open a pull request from the branch 'notes-{N}' into main with the title 'Add notes {N}'.",
     rlist(f"/repos/{R1}/pulls?state=open", title=f"Add notes {N}", **{"head.ref": f"notes-{N}"}), "state",
     setup=[repo("r1", f"proj-{N}"), {"alias": "b1", "op": "branch", "repo": "r1", "branch": f"notes-{N}"}],
     report={"number": "pull request number or null"})

# ------------------------------------------------------------------ identity
USERS = lambda login: f"/users/{login}"  # noqa: E731
task("identity", "Create a user",
     f"As the site administrator, create a new user account with username wa-{N}-new, email maya.chen.{N}@hearthline.test and password 'Study-Pass-1!'. Do not require a password change.",
     rest(USERS(f"wa-{N}-new"), email=f"maya.chen.{N}@hearthline.test"), "state", report={"username": "username or null"})
task("identity", "Create a user with a full name",
     f"As the site administrator, create a new user account with username wb-{N}-new, email dev.rao.{N}@hearthline.test, full name 'Dev Rao {N}' and password 'Study-Pass-1!'. Do not require a password change.",
     rest(USERS(f"wb-{N}-new"), full_name=f"Dev Rao {N}"), "state", report={"username": "username or null"})
task("identity", "Change a user's email",
     f"As the site administrator, open the account of the user {{{{u1.login}}}} (Site Administration, User Accounts, Edit) and change the email to renamed.{N}@hearthline.test.",
     rest(USERS("{{u1.login}}"), email=f"renamed.{N}@hearthline.test"), "state",
     setup=[user("u1", f"em-{N}", f"Eli Voss {N}", email=f"eli.{N}@hearthline.test")])
task("identity", "Change a user's full name",
     f"As the site administrator, open the account of the user {{{{u1.login}}}} and change the full name to 'Mo Lee {N}'.",
     rest(USERS("{{u1.login}}"), full_name=f"Mo Lee {N}"), "state",
     setup=[user("u1", f"fn-{N}", f"Mo Old {N}")])
task("identity", "Prohibit a user from logging in",
     "As the site administrator, open the account of the user {{u1.login}} and prohibit that account from logging in. Save it.",
     rest(USERS("{{u1.login}}"), prohibit_login=True), "state",
     setup=[user("u1", f"pb-{N}", f"Dee Kim {N}")])

# ------------------------------------------------------------------ transaction (irreversible commit) -- analogy
for how, ui in [("merge", "Create merge commit"), ("squash", "Squash and merge"), ("rebase", "Rebase and merge")]:
    task("transaction", f"Merge a pull request ({how})",
         f"Open pull request #{{{{p1.number}}}} in {R1} and merge it using '{ui}'.",
         rest(f"/repos/{R1}/pulls/{{{{p1.number}}}}", merged=True), "state",
         setup=[repo("r1", f"proj-{N}"), pr("p1", "r1", f"change-{N}", f"Update docs {N}")])
task("transaction", "Delete a repository",
     f"Permanently delete the repository {R1} (Settings, Danger Zone).",
     rest("/repos/" + R1, status=404), "state", setup=[repo("r1", f"scratch-{N}")])
task("transaction", "Publish a release",
     f"In the repository {R1}, publish a release with the tag v1.0-{N} and the title 'First release {N}', targeting main.",
     rest(f"/repos/{R1}/releases/tags/v1.0-{N}", name=f"First release {N}"), "state", setup=[repo("r1", f"proj-{N}")])

# ------------------------------------------------------------------ availability (subscribe) -- analogy
task("availability", "Watch a repository",
     f"Watch the repository {R1} so you are notified about new activity.",
     {"kind": "rest", "path": f"/repos/{R1}/subscription", "status": 200}, "state",
     setup=[user("u1", f"w1-{N}", f"Watched {N}"), repo("r1", f"proj-{N}", owner="{{u1.login}}")])
task("availability", "Watch another repository",
     f"Start watching the repository {R1}.",
     {"kind": "rest", "path": f"/repos/{R1}/subscription", "status": 200}, "state",
     setup=[user("u1", f"w2-{N}", f"Watched {N}"), repo("r1", f"tools-{N}", owner="{{u1.login}}", desc="Shared tools")])
task("availability", "Watch two repositories",
     "Watch both repositories {{r1.full_name}} and {{r2.full_name}}.",
     {"kind": "composite", "parts": [{"kind": "rest", "path": "/repos/{{r1.full_name}}/subscription", "status": 200},
                                     {"kind": "rest", "path": "/repos/{{r2.full_name}}/subscription", "status": 200}]}, "state",
     setup=[user("u1", f"w3-{N}", f"Watched {N}"), repo("r1", f"web-{N}", owner="{{u1.login}}"), repo("r2", f"api-{N}", owner="{{u1.login}}")])
task("availability", "Subscribe to an issue",
     f"Subscribe to issue #{{{{i1.number}}}} in {R1} so you are notified of its updates.",
     rest(f"/repos/{R1}/issues/{{{{i1.number}}}}/subscriptions/check", subscribed=True), "state",
     setup=[user("u1", f"s1-{N}", f"Author {N}"), repo("r1", f"proj-{N}", owner="{{u1.login}}"),
            issue("i1", "r1", f"Flaky deploy step {N}", **{"as": "{{u1.login}}"})])
task("availability", "Subscribe to a pull request",
     f"Subscribe to pull request #{{{{p1.number}}}} in {R1} so you are notified of its updates.",
     rest(f"/repos/{R1}/issues/{{{{p1.number}}}}/subscriptions/check", subscribed=True), "state",
     setup=[user("u1", f"s2-{N}", f"Author {N}"), repo("r1", f"proj-{N}", owner="{{u1.login}}"),
            pr("p1", "r1", f"feat-{N}", f"Add retry logic {N}", **{"as": "{{u1.login}}"})])

# ------------------------------------------------------------------ transitions (objective changes mid-session)
task("discovery", "Find an issue, then label and close it",
     f"In the repository {R1}, find the issue about exporting to CSV, add the label 'triaged-{N}' to it and close it.",
     {"kind": "composite", "parts": [rest(f"/repos/{R1}/issues/{{{{i2.number}}}}", state="closed", **{"labels[].name": f"triaged-{N}"})]}, "state",
     setup=[repo("r1", f"proj-{N}"), label("l1", "r1", f"triaged-{N}"), issue("i1", "r1", f"Login page crashes on empty password {N}"),
            issue("i2", "r1", f"Export to CSV drops the header row {N}")],
     seq=["discovery", "curation"], report={"number": "issue number or null"})
task("identity", "Create a user, then open an issue about them",
     f"As the site administrator, create a user account with username wt-{N}, email tess.moon.{N}@hearthline.test and password 'Study-Pass-1!' (no password change required). Then, in the repository {R1}, open an issue titled 'Welcome wt-{N}'.",
     {"kind": "composite", "parts": [rest(USERS(f"wt-{N}"), email=f"tess.moon.{N}@hearthline.test"),
                                     rlist(ISS, title=f"Welcome wt-{N}")]}, "state",
     setup=[repo("r1", f"proj-{N}")], seq=["identity", "service"], report={"username": "username or null"})
task("enumeration", "Count issues, then report the count in a new issue",
     f"Count the issues in the repository {R1} as it is now (before you add anything), then open an issue titled 'Batch report {N}' whose description is 'Count: <that number>'.",
     {"kind": "composite", "parts": [rlist(ISS, title=f"Batch report {N}", body="~count: 4")]}, "state",
     setup=[repo("r1", f"proj-{N}")] + [issue(f"b{i}", "r1", f"Batch {N} item {i}") for i in range(1, 5)],
     seq=["enumeration", "service"], report={"count": 0})

OBJECTIVES = ["discovery", "retrieval", "enumeration", "curation", "service", "identity", "transaction", "availability"]
spec = {"version": "1", "site": "gitea", "objectives": OBJECTIVES, "agents": ["browser-use", "skyvern", "human"],
        "tasks": {f"GT{i:02d}": t for i, t in enumerate(TASKS, 1)}}
Path(__file__).with_name("tasks.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
print(len(TASKS), "tasks")
