#!/usr/bin/env python3
"""Run the ServiceNow task suite (7 objectives, 38 tasks) against the wrapped agents.

Same shape as tasks/run_tasks.py. Each trial:
  1. create the task's setup records over REST (tagged with a session nonce)
  2. set AGENT_TASK from servicenow/prompts.py and start the agent at the tagged start URL
  3. score the agent's JSON report and verify the instance state over REST (independently)
  4. clean up setup rows and anything created during the session

Needs SN_INSTANCE, SN_USER, SN_PASSWORD (see RUNBOOK.md). `--dry-run` and `--print-prompts`
use an in-memory fake and touch nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CRAWLERS = ROOT.parent
PROJECT = CRAWLERS.parent
sys.path.insert(0, str(CRAWLERS))  # run_experiments.py and shared.py
sys.path.insert(0, str(CRAWLERS / "browser-configs"))
sys.path.insert(0, str(ROOT))  # prompts, session, sn, verify

from agent_nav import navigate_url  # noqa: E402
from prompts import redact, render, start_url, task_ids  # noqa: E402
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
from session import SPEC, cleanup, prepare  # noqa: E402
from shared import condition_for, load_agent_report, tag_query  # noqa: E402
from sn import SN, FakeSN  # noqa: E402
from verify import format_agent_report, format_verify, score_agent_report, verify_task  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=task_ids(), choices=task_ids())
    p.add_argument("--agents", nargs="+", default=["browser-use"], choices=list(AGENTS))
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--backend", choices=["primary", "secondary"], default="primary")
    p.add_argument("--humanize", choices=["on", "off"], default="on")
    p.add_argument("--exp-root", default="")
    p.add_argument("--timeout", type=int, default=int(os.getenv("AGENT_RUN_TIMEOUT_SECONDS", 360)),
                   help="Fail a job after this many seconds (default 360; ServiceNow pages are slow)")
    p.add_argument("--loop-repeat", type=int, default=DEFAULT_LOOP_REPEAT_LIMIT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-cleanup", action="store_true", help="Keep the session's records for inspection")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--print-prompts", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    offline = args.dry_run or args.print_prompts
    sn = FakeSN() if offline else SN.from_env()
    instance = (os.getenv("SN_INSTANCE") or "https://devXXXXXX.service-now.com") if offline else sn.instance
    login = (os.getenv("SN_AGENT_USER") or sn.user, os.getenv("SN_AGENT_PASSWORD") or sn.password)

    if args.print_prompts:
        for task_id in args.tasks:
            for agent_name in args.agents:
                ctx = prepare(SPEC["tasks"][task_id], sn)
                url = navigate_url(start_url(task_id, instance, ctx), agent_name, {"task": task_id})
                print(f"\n===== {agent_name} / {task_id} :: {url} =====\n")
                print(render(task_id, ctx, url=url, agent=agent_name, login=("USER", "***")))
        return 0

    api_key = "" if args.dry_run else load_openai_key()
    if api_key:
        configure_open_manus_api_key(api_key)
    exp_root = Path(args.exp_root).resolve() if args.exp_root else PROJECT / "experiment_runs" / f"servicenow_{utc_now()}"
    exp_root.mkdir(parents=True, exist_ok=True)

    jobs = [(a, t, n) for t in args.tasks for a in args.agents for n in range(1, args.trials + 1)]
    if args.smoke:
        jobs = jobs[:1]
    master_log = exp_root / "master_run_log.jsonl"
    done_rows = []
    if args.resume and master_log.exists():
        done = set()
        for line in master_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row.get("agent"), row.get("task"), int(row.get("trial") or 1)))
                done_rows.append(row)
        before = len(jobs)
        jobs = [j for j in jobs if j not in done]
        print(f"Resume: skipping {before - len(jobs)} finished job(s)", flush=True)

    status_path = exp_root / "status.json"
    status = {"experiment_root": str(exp_root), "site": instance, "protocol": "servicenow_tasks",
              "tasks": list(args.tasks), "agents": list(args.agents), "backend": args.backend,
              "humanize": args.humanize, "total_jobs": len(done_rows) + len(jobs),
              "completed": len(done_rows), "started_utc": datetime.now(timezone.utc).isoformat()}
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"Experiment root: {exp_root}", flush=True)

    failures = verify_failed = report_failed = 0
    for idx, (agent_name, task_id, trial) in enumerate(jobs, start=1):
        task = SPEC["tasks"][task_id]
        cond = condition_for(agent_name, args.backend, args.humanize)
        os.environ.update(cond["env"])
        print(f"\n----- job {idx}/{len(jobs)}  {agent_name} task {task_id} trial {trial} -----", flush=True)

        ctx = prepare(task, sn)
        try:
            base = start_url(task_id, instance, ctx)
            target = tag_query(base, {"model": cond["model"], "hum": cond["hum"] if cond["hum"] != "na" else ""})
            tagged = navigate_url(target, agent_name, {"task": task_id})
            print(f"start: {tagged}\ncondition: backend={cond['backend']} model={cond['model']} humanize={cond['hum']}", flush=True)
            prompt = render(task_id, ctx, url=tagged, agent=agent_name, login=login)
            os.environ.update({"AGENT_TASK": prompt, "APPEND_AGENT_QUERY": "1", "EXPERIMENT_TASK": task_id,
                               "EXPERIMENT_AGENT": agent_name, "TARGET_URL": target,
                               "AGENT_RUN_TIMEOUT_SECONDS": str(args.timeout),
                               "SKYVERN_RUN_TIMEOUT_SECONDS": str(args.timeout)})
            for key in ("RQ2_ENV_CONFIG", "RQ2_INSTRUMENT", "TARGET_URLS"):
                os.environ.pop(key, None)

            def _stop_if_verified():
                if verify_task(task_id, sn, ctx).get("ok") is True:
                    return "verify passed"
                return None

            record = run_one(agent_name, target, trial, exp_root, api_key, dry_run=args.dry_run,
                             timeout_seconds=args.timeout, loop_repeat_limit=args.loop_repeat,
                             stop_check=None if args.dry_run else _stop_if_verified)
            agent_report = verify = None
            if not args.dry_run:
                raw, path = load_agent_report(record.get("artifacts_dir"))
                agent_report = score_agent_report(task_id, raw, sn, ctx, path=path)
                print(format_agent_report(agent_report), flush=True)
                report_failed += agent_report.get("ok") is False
                if not args.skip_verify:
                    verify = verify_task(task_id, sn, ctx)
                    print(format_verify(verify), flush=True)
                    verify_failed += verify.get("ok") is False
        finally:
            removed = 0 if (args.skip_cleanup or args.dry_run) else cleanup(sn, ctx)
        record.update({
            "task": task_id, "task_label": task["label"], "objective": task["objective"],
            "objective_sequence": task.get("objective_sequence"),
            "condition": {k: v for k, v in cond.items() if k != "env"},
            "start_url": tagged, "prompt": redact(prompt, login), "nonce": ctx["nonce"],
            "agent_report": agent_report, "verify": verify, "cleanup_removed": removed,
        })
        append_jsonl(master_log, record)
        failures += record.get("returncode", 1) != 0
        status.update({"completed": len(done_rows) + idx, "failed": failures, "verify_failed": verify_failed,
                       "agent_report_failed": report_failed, "updated_utc": datetime.now(timezone.utc).isoformat()})
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"\nFinished. jobs={len(jobs)} failures={failures} verify_failed={verify_failed} "
          f"agent_report_failed={report_failed}\nResults under: {exp_root}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
