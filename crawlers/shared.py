"""Helpers shared by tasks/run_tasks.py and servicenow/run_tasks.py (no sibling-module imports)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


# Two model backends per implementation. `primary` is each tool's default.
BACKENDS = {
    "browser-use": {
        "primary": {"OPENAI_MODEL": "o4-mini"},
        "secondary": {"OPENAI_MODEL": "gpt-4o"},
    },
    "skyvern": {
        "primary": {
            "SKYVERN_MODEL": "gpt-5.4-mini",
            "LLM_KEY": "OPENAI_GPT5_4_MINI",
            "GPT5_REASONING_EFFORT": "low",
        },
        "secondary": {"SKYVERN_MODEL": "gpt-4o-mini", "LLM_KEY": "OPENAI_GPT4O_MINI"},
    },
}
MODEL_ENV = {"browser-use": "OPENAI_MODEL", "skyvern": "SKYVERN_MODEL"}


def condition_for(agent_name: str, backend: str, humanize: str) -> dict:
    env = dict(BACKENDS.get(agent_name, {}).get(backend, {}))
    model = env.get(MODEL_ENV.get(agent_name, ""), "")
    hum = humanize if agent_name == "browser-use" else "na"
    if agent_name == "browser-use":
        env["BROWSER_USE_HUMANIZE"] = "true" if humanize == "on" else "false"
    return {"env": env, "model": model, "hum": hum, "backend": backend}


def tag_query(url: str, extra: dict) -> str:
    parsed = urlparse(url)
    merged = dict(parse_qsl(parsed.query, keep_blank_values=True))
    merged.update({k: v for k, v in extra.items() if v})
    return urlunparse(parsed._replace(query=urlencode(merged)))


def _parse_report_blob(payload: dict) -> dict | None:
    report = payload.get("final_report_raw") or payload.get("final_report")
    if isinstance(report, str):
        try:
            report = json.loads(report)
        except json.JSONDecodeError:
            report = None
    if isinstance(report, dict):
        return report
    text = payload.get("final_text")
    if isinstance(text, str):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def load_agent_report(artifacts_dir: str | Path | None) -> tuple[dict | None, str | None]:
    if not artifacts_dir:
        return None, None
    result_dir = Path(artifacts_dir) / "result"
    if not result_dir.is_dir():
        return None, None
    files = [p for p in result_dir.glob("*.json") if p.is_file()]
    if not files:
        return None, None
    newest = max(files, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, str(newest)
    return _parse_report_blob(payload), str(newest)
