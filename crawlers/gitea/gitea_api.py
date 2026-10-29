"""Minimal Gitea REST client (token auth). Non-2xx responses are returned, not raised.

Environment:
  GITEA_URL       e.g. http://131.94.129.67:3000
  GITEA_TOKEN     admin access token (setup, verification, cleanup)
  GITEA_USER      account the agent logs in as (same admin user)
  GITEA_PASSWORD
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


class Gitea:
    def __init__(self, url: str, token: str, user: str = "", password: str = "", timeout: float = 30):
        self.url = url.rstrip("/")
        self.token, self.user, self.password, self.timeout = token, user, password, timeout

    @classmethod
    def from_env(cls) -> "Gitea":
        try:
            return cls(os.environ["GITEA_URL"], os.environ["GITEA_TOKEN"],
                       os.environ.get("GITEA_USER", ""), os.environ.get("GITEA_PASSWORD", ""))
        except KeyError as exc:
            raise SystemExit(f"Set GITEA_URL and GITEA_TOKEN (missing {exc})") from exc

    def call(self, method: str, path: str, body: dict | None = None, sudo: str | None = None):
        """Return (status, parsed_json_or_None). `path` starts with / and is relative to /api/v1."""
        req = urllib.request.Request(f"{self.url}/api/v1{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        req.add_header("Authorization", f"token {self.token}")
        req.add_header("Accept", "application/json")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        if sudo:
            req.add_header("Sudo", sudo)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw, status = resp.read().decode("utf-8"), resp.status
        except urllib.error.HTTPError as exc:
            raw, status = exc.read().decode("utf-8", "replace"), exc.code
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach {self.url}: {exc.reason}") from exc
        try:
            return status, (json.loads(raw) if raw else None)
        except json.JSONDecodeError:
            return status, None

    def get(self, path: str, sudo: str | None = None):
        return self.call("GET", path, sudo=sudo)

    def must(self, method: str, path: str, body: dict | None = None, sudo: str | None = None) -> dict:
        status, data = self.call(method, path, body, sudo)
        if not 200 <= status < 300:
            raise RuntimeError(f"{method} {path} -> HTTP {status}: {str(data)[:200]}")
        return data if data is not None else {}

    def get_all(self, path: str, limit: int = 50, max_pages: int = 20) -> list:
        """Follow ?page= pagination for list endpoints (also handles {'data': [...]} wrappers)."""
        out, sep = [], "&" if "?" in path else "?"
        for page in range(1, max_pages + 1):
            status, data = self.get(f"{path}{sep}limit={limit}&page={page}")
            if status != 200 or data is None:
                break
            rows = data.get("data", data) if isinstance(data, dict) else data
            out += rows
            if len(rows) < limit:
                break
        return out
