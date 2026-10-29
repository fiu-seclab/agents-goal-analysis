"""Build start URLs that always carry agent= (and optional task=/env=)."""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def navigate_url(url: str, agent_slug: str, extra: dict | None = None) -> str:
    parsed = urlparse(url)
    merged = {key: value for key, value in parse_qsl(parsed.query, keep_blank_values=True)}
    merged["agent"] = agent_slug
    for key, value in (extra or {}).items():
        if value is None or str(value).strip() == "":
            continue
        merged[str(key)] = str(value)
    return urlunparse(parsed._replace(query=urlencode(merged)))


def experiment_navigate_url(url: str, agent_slug: str) -> str:
    extra = {}
    task = os.getenv("EXPERIMENT_TASK", "").strip()
    if task:
        extra["task"] = task
    env_config = os.getenv("RQ2_ENV_CONFIG", "").strip()
    if env_config:
        extra["env"] = env_config
    return navigate_url(url, agent_slug, extra)
