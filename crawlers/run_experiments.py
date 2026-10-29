#!/usr/bin/env python3
"""Run Browser Use or Skyvern against a target URL.

Each agent is the wrapper in browser-use-app/ or skyvern-app/. A local .venv
is used when present; otherwise the current Python interpreter is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BROWSER_CONFIGS = ROOT / "browser-configs"


def _interpreter(app: Path) -> Path:
    """Prefer an app-local venv. Otherwise use the interpreter running this script."""
    venv_python = app / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


AGENTS = {
    "browser-use": {
        "cwd": ROOT / "browser-use-app",
        "python": _interpreter(ROOT / "browser-use-app"),
        "script": ROOT / "browser-use-app" / "main.py",
        "artifacts_subdir": "browser-use",
    },
    "skyvern": {
        "cwd": ROOT / "skyvern-app",
        "python": _interpreter(ROOT / "skyvern-app"),
        "script": ROOT / "skyvern-app" / "main.py",
        "artifacts_subdir": "skyvern",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_env_file(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_openai_key() -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    for path in (ROOT / ".env", ROOT / "browser-use-app" / ".env", ROOT / "skyvern-app" / ".env"):
        key = _read_env_file(path).get("OPENAI_API_KEY", "").strip()
        if key:
            return key
    raise SystemExit("OPENAI_API_KEY not set and no .env found")


def configure_open_manus_api_key(api_key: str) -> None:
    config_path = ROOT / "open-manus-app" / "OpenManus" / "config" / "config.toml"
    if not config_path.exists():
        return
    text = config_path.read_text(encoding="utf-8")
    if re.search(r'(?m)^api_key\s*=\s*".*"\s*$', text):
        text = re.sub(r'(?m)^api_key\s*=\s*".*"\s*$', f'api_key = "{api_key}"', text)
        config_path.write_text(text, encoding="utf-8")


DEFAULT_AGENT_TIMEOUT_SECONDS = 240
DEFAULT_LOOP_REPEAT_LIMIT = 8

# SeeAct / similar action traces: "[button] Add to cart -> CLICK"
_ACTION_ARROW = re.compile(
    r"(\[[^\n\]]{1,160}\].{0,80}->\s*(?:CLICK|TYPE|SELECT|NONE|HOVER|ENTER))",
    re.IGNORECASE,
)
# Skyvern / Browser-Use short action lines
_ACTION_VERB = re.compile(
    r"\b(click(?:ing|ed)?|type|typed|input|select(?:ed|ing)?|scroll(?:ing|ed)?)\b",
    re.IGNORECASE,
)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def action_signature(line: str) -> str | None:
    text = line.strip()
    if not text or len(text) > 220:
        return None
    if text.startswith("{") or "Navigate to http" in text:
        return None
    match = _ACTION_ARROW.search(text)
    if match:
        return re.sub(r"\s+", " ", match.group(1).lower())
    if "->" in text and _ACTION_VERB.search(text):
        return re.sub(r"\s+", " ", text.lower())
    return None


def _kill_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 8
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_watched(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    loop_repeat_limit: int,
    stop_check=None,
    stop_check_interval: float = 8,
) -> tuple[int, str | None]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout_seconds
    recent = deque(maxlen=max(loop_repeat_limit * 2, 8))
    killed = None
    next_stop_check = time.monotonic() + stop_check_interval
    try:
        while True:
            now = time.monotonic()
            if stop_check is not None and now >= next_stop_check:
                next_stop_check = now + stop_check_interval
                try:
                    reason = stop_check()
                except Exception as exc:  # noqa: BLE001 — keep the agent running if the check errors
                    print(f"verify poll failed: {exc}", flush=True)
                    reason = None
                if reason:
                    killed = reason
                    break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                killed = f"timeout after {timeout_seconds}s"
                break
            ready, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining))
            if ready:
                line = proc.stdout.readline()
                if line == "":
                    if proc.poll() is not None:
                        break
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()
                signature = action_signature(line)
                if signature:
                    recent.append(signature)
                    repeats = recent.count(signature)
                    if repeats >= loop_repeat_limit:
                        killed = f"action loop: {signature!r} x{repeats}"
                        break
            elif proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    sys.stdout.write(rest)
                    sys.stdout.flush()
                break
    finally:
        if killed:
            print(f"\n!!! stopping agent: {killed}\n", flush=True)
            _kill_process_group(proc)
        elif proc.poll() is None:
            _kill_process_group(proc)
    returncode = proc.poll()
    if killed:
        # The store check already passed, so this is a completed trial, not a failure.
        if killed == "verify passed":
            return 0, killed
        return 124, killed
    return (returncode if returncode is not None else 1), None


def run_one(
    agent_name: str,
    url: str,
    trial: int,
    exp_root: Path,
    api_key: str,
    dry_run: bool = False,
    timeout_seconds: int = DEFAULT_AGENT_TIMEOUT_SECONDS,
    loop_repeat_limit: int = DEFAULT_LOOP_REPEAT_LIMIT,
    stop_check=None,
) -> dict:
    agent = AGENTS[agent_name]
    artifacts_dir = exp_root / agent["artifacts_subdir"]
    for sub in ("result", "recordings", "terminal_logs", "agent_traces"):
        (artifacts_dir / sub).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["DISPLAY"] = env.get("DISPLAY") or ":1"
    env["EXPERIMENT_ARTIFACTS_DIR"] = str(artifacts_dir)
    env["EXPERIMENT_TRIAL"] = str(trial)
    env["EXPERIMENT_AGENT"] = agent_name
    env["BROWSER_CONFIGS_DIR"] = str(BROWSER_CONFIGS)
    env["TARGET_URL"] = url
    env["PYTHONUNBUFFERED"] = "1"
    env["AGENT_RUN_TIMEOUT_SECONDS"] = str(timeout_seconds)
    env["SKYVERN_RUN_TIMEOUT_SECONDS"] = str(timeout_seconds)
    env.pop("TARGET_URLS", None)

    cmd = [str(agent["python"]), str(agent["script"]), "--url", url]
    started = datetime.now(timezone.utc).isoformat()
    record = {
        "agent": agent_name,
        "target_url": url,
        "trial": trial,
        "started_utc": started,
        "artifacts_dir": str(artifacts_dir),
        "command": cmd,
    }

    print(f"\n=== [{agent_name}] trial {trial} :: {url} ===\nartifacts: {artifacts_dir}", flush=True)
    if dry_run:
        record.update({"dry_run": True, "returncode": 0, "elapsed_seconds": 0.0})
        return record

    t0 = time.perf_counter()
    returncode, killed = _run_watched(
        cmd,
        agent["cwd"],
        env,
        timeout_seconds=timeout_seconds,
        loop_repeat_limit=loop_repeat_limit,
        stop_check=stop_check,
    )
    elapsed = time.perf_counter() - t0
    record.update(
        {
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 3),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": timeout_seconds,
        }
    )
    if killed:
        record["killed"] = killed
    print(
        f"=== done [{agent_name}] trial {trial} rc={returncode} "
        f"elapsed={elapsed:.1f}s ===",
        flush=True,
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("TARGET_URL", "").strip())
    parser.add_argument("--agents", nargs="+", default=["browser-use"], choices=list(AGENTS))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--exp-root", default="")
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("AGENT_RUN_TIMEOUT_SECONDS", DEFAULT_AGENT_TIMEOUT_SECONDS)),
        help="Fail a job after this many seconds (default 240)",
    )
    parser.add_argument(
        "--loop-repeat",
        type=int,
        default=DEFAULT_LOOP_REPEAT_LIMIT,
        help="Kill a job if the same UI action repeats this many times",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = args.url.strip()
    if not url:
        raise SystemExit("Pass --url or set TARGET_URL")

    api_key = load_openai_key()
    configure_open_manus_api_key(api_key)

    if args.exp_root:
        exp_root = Path(args.exp_root).resolve()
    else:
        exp_root = PROJECT / "experiment_runs" / f"run_{utc_now()}"
    exp_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (agent_name, url, trial)
        for agent_name in args.agents
        for trial in range(1, args.trials + 1)
    ]
    if args.smoke:
        jobs = jobs[:1]

    master_log = exp_root / "master_run_log.jsonl"
    status_path = exp_root / "status.json"
    print(f"Experiment root: {exp_root}", flush=True)

    status = {
        "experiment_root": str(exp_root),
        "url": url,
        "total_jobs": len(jobs),
        "completed": 0,
        "failed": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    failures = 0
    for idx, (agent_name, job_url, trial) in enumerate(jobs, start=1):
        print(f"\n----- job {idx}/{len(jobs)} -----", flush=True)
        record = run_one(
            agent_name,
            job_url,
            trial,
            exp_root,
            api_key,
            dry_run=args.dry_run,
            timeout_seconds=args.timeout,
            loop_repeat_limit=args.loop_repeat,
        )
        append_jsonl(master_log, record)
        if record.get("returncode", 1) != 0:
            failures += 1
        status.update(
            {
                "completed": idx,
                "failed": failures,
                "last_job": record,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"\nFinished. jobs={len(jobs)} failures={failures}\nResults under: {exp_root}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
