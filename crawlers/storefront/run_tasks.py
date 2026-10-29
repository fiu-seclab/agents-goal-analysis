#!/usr/bin/env python3
"""Run paper tasks A–E against every wrapped agent.

Each trial:
  1. POST /api/reset-db
  2. set AGENT_TASK from tasks/prompts.py
  3. start the agent at the task URL with agent=<slug>&task=<id>
  4. record the agent JSON report and the store-state verifier separately
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CRAWLERS = ROOT.parent
PROJECT = CRAWLERS.parent
BROWSER_CONFIGS = CRAWLERS / "browser-configs"

sys.path.insert(0, str(CRAWLERS))
sys.path.insert(0, str(BROWSER_CONFIGS))
sys.path.insert(0, str(ROOT))

from agent_nav import navigate_url  # noqa: E402
from prompts import SPEC, render, start_url, task_ids  # noqa: E402
from run_experiments import (  # noqa: E402
    AGENTS,
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    DEFAULT_LOOP_REPEAT_LIMIT,
    append_jsonl,
    configure_open_manus_api_key,
    load_openai_key,
    run_one,
    utc_now,
)
from shared import condition_for, tag_query  # noqa: E402
from verify import (  # noqa: E402
    fetch_snapshot,
    format_agent_report,
    format_verify,
    load_agent_report,
    score_agent_report,
    slim_snapshot,
    verify_task,
)


def reset_store(site: str) -> dict:
    url = site.rstrip("/") + "/api/reset-db"
    req = urllib.request.Request(url, data=b"{}", method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reset store at {url}: {exc}") from exc
    print(f"Reset store: {payload}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default=os.getenv("TARGET_SITE", SPEC["site"]).strip())
    parser.add_argument("--tasks", nargs="+", default=task_ids(), choices=task_ids())
    parser.add_argument("--agents", nargs="+", default=["browser-use"], choices=list(AGENTS))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=["primary", "secondary"],
        default="primary",
        help="Model backend: browser-use o4-mini/gpt-4o, skyvern gpt-5.4-mini/gpt-4o-mini",
    )
    parser.add_argument(
        "--humanize",
        choices=["on", "off"],
        default="on",
        help="browser-use humanization layer (ignored for skyvern)",
    )
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs already present in master_run_log.jsonl",
    )
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Do not GET /api/experiment/state after each trial",
    )
    parser.add_argument(
        "--require-verify",
        action="store_true",
        help="Count a verifier miss as a job failure (still continues to the next trial)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-prompts", action="store_true", help="Print prompts and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    site = args.site.rstrip("/")

    if args.print_prompts:
        for task_id in args.tasks:
            for agent_name in args.agents:
                url = navigate_url(start_url(task_id, site), agent_name, {"task": task_id})
                print(f"\n===== {agent_name} / Task {task_id} :: {url} =====\n")
                print(render(task_id, url=url, agent=agent_name))
        return 0

    api_key = "" if args.dry_run else load_openai_key()
    if api_key:
        configure_open_manus_api_key(api_key)

    if args.exp_root:
        exp_root = Path(args.exp_root).resolve()
    else:
        exp_root = PROJECT / "experiment_runs" / f"tasks_{utc_now()}"
    exp_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (agent_name, task_id, trial)
        for task_id in args.tasks
        for agent_name in args.agents
        for trial in range(1, args.trials + 1)
    ]
    if args.smoke:
        jobs = jobs[:1]

    master_log = exp_root / "master_run_log.jsonl"
    already_done = []
    if args.resume and master_log.exists():
        done = set()
        for line in master_log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("agent"), row.get("task"), int(row.get("trial") or 1))
            if key not in done:
                already_done.append(row)
            done.add(key)
        before = len(jobs)
        jobs = [job for job in jobs if job not in done]
        print(f"Resume: skipping {before - len(jobs)} finished job(s)", flush=True)

    status_path = exp_root / "status.json"
    print(f"Experiment root: {exp_root}", flush=True)
    print(
        f"Watchdog: timeout={args.timeout}s loop_repeat={args.loop_repeat}",
        flush=True,
    )

    prior_failed = sum(1 for row in already_done if row.get("returncode", 1) != 0)
    prior_verify_failed = sum(
        1
        for row in already_done
        if (row.get("verify") or {}).get("ok") is False
    )
    prior_report_failed = sum(
        1
        for row in already_done
        if (row.get("agent_report") or {}).get("ok") is False
    )
    started_utc = datetime.now(timezone.utc).isoformat()
    if args.resume and status_path.exists():
        try:
            started_utc = json.loads(status_path.read_text(encoding="utf-8")).get(
                "started_utc", started_utc
            )
        except json.JSONDecodeError:
            pass
    status = {
        "experiment_root": str(exp_root),
        "site": site,
        "protocol": "hearthline_paper_tasks",
        "tasks": list(args.tasks),
        "agents": list(args.agents),
        "backend": args.backend,
        "humanize": args.humanize,
        "total_jobs": len(already_done) + len(jobs),
        "completed": len(already_done),
        "failed": prior_failed,
        "verify_failed": prior_verify_failed,
        "agent_report_failed": prior_report_failed,
        "timeout_seconds": args.timeout,
        "loop_repeat": args.loop_repeat,
        "started_utc": started_utc,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    failures = prior_failed
    verify_failed = prior_verify_failed
    report_failed = prior_report_failed
    for idx, (agent_name, task_id, trial) in enumerate(jobs, start=1):
        cond = condition_for(agent_name, args.backend, args.humanize)
        for key, value in cond["env"].items():
            os.environ[key] = value
        target = tag_query(
            start_url(task_id, site),
            {"model": cond["model"], "hum": cond["hum"] if cond["hum"] != "na" else ""},
        )
        tagged = navigate_url(target, agent_name, {"task": task_id})
        prompt = render(task_id, url=tagged, agent=agent_name)

        os.environ["AGENT_TASK"] = prompt
        os.environ["APPEND_AGENT_QUERY"] = "1"
        os.environ["EXPERIMENT_TASK"] = task_id
        os.environ["EXPERIMENT_AGENT"] = agent_name
        os.environ["TARGET_URL"] = target
        os.environ["AGENT_RUN_TIMEOUT_SECONDS"] = str(args.timeout)
        os.environ["SKYVERN_RUN_TIMEOUT_SECONDS"] = str(args.timeout)
        os.environ.pop("RQ2_ENV_CONFIG", None)
        os.environ.pop("RQ2_INSTRUMENT", None)
        os.environ.pop("TARGET_URLS", None)

        print(
            f"\n----- job {idx}/{len(jobs)}  {agent_name} task {task_id} trial {trial} -----",
            flush=True,
        )
        print(f"start: {tagged}", flush=True)
        print(
            f"condition: backend={cond['backend']} model={cond['model']} humanize={cond['hum']}",
            flush=True,
        )

        reset_info = None
        if not args.skip_reset and not args.dry_run:
            reset_info = reset_store(site)

        record = run_one(
            agent_name,
            target,
            trial,
            exp_root,
            api_key,
            dry_run=args.dry_run,
            timeout_seconds=args.timeout,
            loop_repeat_limit=args.loop_repeat,
        )
        snapshot = None
        snapshot_error = None
        if not args.dry_run:
            try:
                snapshot = fetch_snapshot(site)
            except Exception as exc:  # noqa: BLE001 — keep the grid moving
                snapshot_error = str(exc)

        agent_report = None
        if not args.dry_run:
            raw_report, report_path = load_agent_report(record.get("artifacts_dir"))
            agent_report = score_agent_report(
                task_id, raw_report, path=report_path, snapshot=snapshot
            )
            print(format_agent_report(agent_report), flush=True)
            if agent_report.get("ok") is False:
                report_failed += 1

        verify = None
        if not args.skip_verify and not args.dry_run:
            if snapshot is not None:
                verify = verify_task(task_id, snapshot)
                verify["snapshot"] = slim_snapshot(snapshot)
            else:
                verify = {
                    "task": task_id,
                    "ok": False,
                    "source": "experiment_state",
                    "error": snapshot_error,
                    "checks": [],
                }
            print(format_verify(verify), flush=True)
            if verify.get("ok") is False:
                verify_failed += 1

        record.update(
            {
                "task": task_id,
                "objective": SPEC["tasks"][task_id].get("objective"),
                "objective_sequence": SPEC["tasks"][task_id].get("objective_sequence"),
                "condition": {k: v for k, v in cond.items() if k != "env"},
                "task_label": SPEC["tasks"][task_id]["label"],
                "start_url": tagged,
                "prompt": prompt,
                "reset": reset_info,
                "agent_report": agent_report,
                "verify": verify,
            }
        )
        append_jsonl(master_log, record)
        job_failed = record.get("returncode", 1) != 0
        if args.require_verify and verify is not None and verify.get("ok") is False:
            job_failed = True
        if job_failed:
            failures += 1
        status.update(
            {
                "completed": len(already_done) + idx,
                "failed": failures,
                "verify_failed": verify_failed,
                "agent_report_failed": report_failed,
                "last_job": {
                    "agent": agent_name,
                    "task": task_id,
                    "trial": trial,
                    "returncode": record.get("returncode"),
                    "agent_report_ok": None if agent_report is None else agent_report.get("ok"),
                    "verify_ok": None if verify is None else verify.get("ok"),
                    "start_url": tagged,
                },
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nFinished. jobs={len(jobs)} failures={failures} "
        f"verify_failed={verify_failed} agent_report_failed={report_failed}\n"
        f"Results under: {exp_root}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
