"""Minimal ServiceNow Table API client, plus an in-memory fake for tests and dry runs.

Environment (real client):
  SN_INSTANCE  https://devNNNNN.service-now.com
  SN_USER      account used for REST setup, verification and cleanup (needs admin)
  SN_PASSWORD
  SN_AUTH      basic | session | auto (default). `session` logs in through /login.do and sends the
               cookie plus X-UserToken; use it when Basic auth returns 401 (see diagnose.py).
"""

from __future__ import annotations

import base64
import http.cookiejar
import itertools
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone


def sn_time(ts: datetime) -> str:
    """ServiceNow database datetime (UTC)."""
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class SN:
    def __init__(self, instance: str, user: str, password: str, timeout: float = 30, auth: str = "auto"):
        self.instance = instance.rstrip("/")
        self.user, self.password, self.timeout = user, password, timeout
        self._auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        self.mode = auth  # basic | session | auto (auto: basic, fall back to session on 401)
        self._opener = None
        self._token = None

    def _login(self) -> None:
        jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self._opener.open(self.instance + "/login.do", timeout=self.timeout).read()
        form = urllib.parse.urlencode({"user_name": self.user, "user_password": self.password,
                                       "sys_action": "sysverb_login", "sysparm_login_url": "welcome.do"}).encode()
        self._opener.open(urllib.request.Request(self.instance + "/login.do", data=form), timeout=self.timeout).read()
        page = self._opener.open(self.instance + "/navpage.do", timeout=self.timeout).read().decode("utf-8", "replace")
        m = re.search(r"g_ck\s*=\s*['\"]([0-9a-f]+)['\"]", page)
        if not m:
            raise RuntimeError("session login failed (no g_ck token); check user/password and that no password change is pending")
        self._token = m.group(1)

    @classmethod
    def from_env(cls) -> "SN":
        try:
            return cls(os.environ["SN_INSTANCE"], os.environ["SN_USER"], os.environ["SN_PASSWORD"],
                       auth=os.environ.get("SN_AUTH", "auto"))
        except KeyError as exc:
            raise SystemExit(f"Set SN_INSTANCE, SN_USER and SN_PASSWORD (missing {exc})") from exc

    def _call(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> dict:
        url = f"{self.instance}/api/now/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        for attempt in (1, 2):
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Accept", "application/json")
            if data is not None:
                req.add_header("Content-Type", "application/json")
            use_session = self.mode == "session" or (self.mode == "auto" and self._token is not None)
            if use_session:
                if self._opener is None:
                    self._login()
                req.add_header("X-UserToken", self._token)
                opener = self._opener.open
            else:
                req.add_header("Authorization", self._auth)
                opener = urllib.request.urlopen
            try:
                with opener(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 401 and attempt == 1 and self.mode in ("auto", "session"):
                    self._login()  # first 401 under auto: switch to session; under session: refresh it
                    continue
                raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode()[:300]}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"{method} {path} -> cannot reach {self.instance}: {exc.reason}") from exc
        return json.loads(raw) if raw else {}

    # -- Table API -----------------------------------------------------------------
    def query(self, table: str, encoded: str = "", fields: str = "", limit: int = 200) -> list[dict]:
        params = {"sysparm_limit": limit, "sysparm_display_value": "false"}
        if encoded:
            params["sysparm_query"] = encoded
        if fields:
            params["sysparm_fields"] = fields
        return self._call("GET", f"table/{table}", params).get("result", [])

    def count(self, table: str, encoded: str = "") -> int:
        params = {"sysparm_query": encoded} if encoded else {}
        res = self._call("GET", f"stats/{table}", {**params, "sysparm_count": "true"})
        return int(res["result"]["stats"]["count"])

    def insert(self, table: str, fields: dict) -> dict:
        return self._call("POST", f"table/{table}", body=fields)["result"]

    def get(self, table: str, sys_id: str) -> dict:
        return self._call("GET", f"table/{table}/{sys_id}", {"sysparm_display_value": "false"})["result"]

    def delete(self, table: str, sys_id: str) -> None:
        self._call("DELETE", f"table/{table}/{sys_id}")


class FakeSN:
    """Just enough of the Table API for self-tests and --dry-run / --print-prompts."""

    PREFIX = {"incident": "INC", "problem": "PRB", "change_request": "CHG", "sc_req_item": "RITM",
              "sc_request": "REQ", "sys_user": "USR"}

    def __init__(self):
        self.tables: dict[str, dict[str, dict]] = {}
        self._ids = itertools.count(1)
        self.instance = "https://fake.service-now.com"
        self.user = "fake.user"
        self.password = "fake-password"

    def insert(self, table: str, fields: dict) -> dict:
        n = next(self._ids)
        rec = {k: str(v) for k, v in fields.items()}
        rec["sys_id"] = f"{n:032x}"
        rec["sys_created_on"] = sn_time(datetime.now(timezone.utc))
        if table in self.PREFIX and table != "sys_user":
            rec["number"] = f"{self.PREFIX[table]}{n:07d}"
        self.tables.setdefault(table, {})[rec["sys_id"]] = rec
        return dict(rec)

    def get(self, table: str, sys_id: str) -> dict:
        return dict(self.tables[table][sys_id])

    def delete(self, table: str, sys_id: str) -> None:
        self.tables.get(table, {}).pop(sys_id, None)

    @staticmethod
    def _match(rec: dict, clause: str) -> bool:
        m = re.match(r"^([\w.]+?)(>=|<=|!=|LIKE|=|>|<)(.*)$", clause)
        if not m:
            return True
        field, op, val = m.groups()
        have = str(rec.get(field, ""))
        if op == "=":
            return have == val
        if op == "!=":
            return have != val
        if op == "LIKE":
            return val.lower() in have.lower()
        if op == ">=":
            return have >= val
        if op == "<=":
            return have <= val
        return have > val if op == ">" else have < val

    def query(self, table: str, encoded: str = "", fields: str = "", limit: int = 200) -> list[dict]:
        clauses = [c for c in encoded.split("^") if c and not c.startswith("ORDER")]
        rows = [dict(r) for r in self.tables.get(table, {}).values()
                if all(self._match(r, c) for c in clauses)]
        return rows[:limit]

    def count(self, table: str, encoded: str = "") -> int:
        return len(self.query(table, encoded, limit=10**6))
