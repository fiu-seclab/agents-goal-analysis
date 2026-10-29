import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
from uuid import uuid4

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
load_dotenv(HERE / "Skyvern" / ".env")

AGENT_RUN_TIMEOUT_SECONDS = int(os.getenv("SKYVERN_RUN_TIMEOUT_SECONDS", "600"))

DEFAULT_URLS = []


def _load_rq2():
    candidates = []
    env_dir = os.getenv("BROWSER_CONFIGS_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    for parent in [HERE, *HERE.parents]:
        candidates.append(parent / "browser-configs")
        candidates.append(parent / "rq2_env_grid")
    for candidate in candidates:
        if (candidate / "verdict.py").exists():
            sys.path.insert(0, str(candidate))
            from verdict import (
                PROBE_JS,
                derive_env_verdict,
                normalize_probe,
                rq2_cdp_url,
                rq2_env_config,
                rq2_instrument,
                rq2_prompt,
            )

            return {
                "PROBE_JS": PROBE_JS,
                "derive_env_verdict": derive_env_verdict,
                "normalize_probe": normalize_probe,
                "rq2_cdp_url": rq2_cdp_url,
                "rq2_env_config": rq2_env_config,
                "rq2_instrument": rq2_instrument,
                "rq2_prompt": rq2_prompt,
            }
    return None


_RQ2 = _load_rq2()


def _url_slug(target_url):
    parsed = urlparse(target_url)
    host = parsed.netloc or "unknown_host"
    path = parsed.path.strip("/")
    raw = host if not path else f"{host}_{path}"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)


def _artifacts_root() -> Path:
    env = os.getenv("EXPERIMENT_ARTIFACTS_DIR", "").strip()
    if env:
        return Path(env)
    return HERE


def _unique_run_stem(target_url, run_id=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    trial = os.getenv("EXPERIMENT_TRIAL", "").strip()
    rid = (run_id or uuid4().hex)[:8]
    parts = [_url_slug(target_url)]
    if _RQ2:
        env_config = _RQ2["rq2_env_config"]()
        instrument = _RQ2["rq2_instrument"]()
        if env_config:
            parts.append(env_config)
        if instrument:
            parts.append(instrument.replace("-", ""))
    if trial:
        parts.append(f"trial{trial}")
    parts.extend([stamp, rid])
    return "__".join(parts)


def _navigate_url(target_url):
    env_config = _RQ2["rq2_env_config"]() if _RQ2 else ""
    extra = {}
    task = os.getenv("EXPERIMENT_TASK", "").strip()
    if task:
        extra["task"] = task
    if env_config:
        extra["env"] = env_config
    try:
        from agent_nav import navigate_url

        return navigate_url(target_url, "skyvern", extra)
    except ImportError:
        query = "agent=skyvern"
        if env_config:
            query += f"&env={env_config}"
        if task:
            query += f"&task={task}"
        sep = "&" if "?" in target_url else "?"
        return f"{target_url}{sep}{query}"


def _build_task(target_url):
    if _RQ2 and _RQ2["rq2_env_config"]():
        return _RQ2["rq2_prompt"](target_url, "skyvern")
    custom = os.getenv("AGENT_TASK", "").strip()
    if custom:
        return custom.replace("{url}", _navigate_url(target_url)).replace("{agent}", "skyvern")
    user = os.getenv("TESTBED_USERNAME", "user")
    password = os.getenv("TESTBED_PASSWORD", "password")
    return f"""
Navigate to {_navigate_url(target_url)}
Fill out the form with the following information:
-----
username: {user}, password: {password}
-----
Submit the form by clicking the login button.
"""


def _build_result_paths(target_url, run_id):
    result_dir = _artifacts_root() / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    unique_stem = _unique_run_stem(target_url, run_id=run_id)
    return (
        result_dir / f"{_url_slug(target_url)}.jsonl",
        result_dir / f"{unique_stem}.json",
        unique_stem,
    )


def _build_recording_path(target_url):
    recordings_dir = _artifacts_root() / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir / f"{_unique_run_stem(target_url)}.mkv"


def _build_terminal_log_path(target_url):
    logs_dir = _artifacts_root() / "terminal_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{_unique_run_stem(target_url)}.log"


def _detect_capture_region(display):
    env_size = os.getenv("SCREEN_RECORD_SIZE", "").strip()
    env_offset = os.getenv("SCREEN_RECORD_OFFSET", "").strip()
    if env_size:
        return env_size, (env_offset or "0,0"), "env"

    capture_mode = os.getenv("SCREEN_RECORD_CAPTURE_MODE", "single").strip().lower()
    try:
        xrandr_info = subprocess.check_output(
            ["xrandr", "--display", display, "--current"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        monitor_matches = re.findall(
            r"^(\S+)\s+connected(?:\s+primary)?\s+(\d+)x(\d+)\+(-?\d+)\+(-?\d+)",
            xrandr_info,
            flags=re.MULTILINE,
        )
        primary_match = re.search(
            r"^(\S+)\s+connected\s+primary\s+(\d+)x(\d+)\+(-?\d+)\+(-?\d+)",
            xrandr_info,
            flags=re.MULTILINE,
        )
        if capture_mode != "all":
            if primary_match:
                monitor = primary_match.group(1)
                size = f"{primary_match.group(2)}x{primary_match.group(3)}"
                offset = f"{primary_match.group(4)},{primary_match.group(5)}"
                return size, offset, f"xrandr-primary:{monitor}"
            if monitor_matches:
                monitor, w, h, x, y = monitor_matches[0]
                return f"{w}x{h}", f"{x},{y}", f"xrandr-first:{monitor}"
    except Exception:
        pass

    return "1920x1080", "0,0", "default"


def _start_screen_recording(recording_path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None, "ffmpeg not found in PATH"
    display = os.getenv("DISPLAY", "").strip()
    if not display:
        return None, "DISPLAY is not set; cannot use x11 screen capture"

    screen_size, screen_offset, source = _detect_capture_region(display)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-video_size",
        screen_size,
        "-f",
        "x11grab",
        "-i",
        f"{display}+{screen_offset}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(recording_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(
            f"Recording region detected via {source}: "
            f"size={screen_size}, offset={screen_offset}"
        )
        return proc, None
    except Exception as exc:
        return None, str(exc)


def _stop_screen_recording(proc):
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass


def _extract_json_report(text):
    if not text:
        return None
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1", "success", "passed"}:
            return True
        if v in {"false", "no", "n", "0", "failure", "failed"}:
            return False
    return None


def _to_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _to_jsonable(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            return str(value)
    return str(value)


def _derive_outcomes(final_text, report):
    captcha_type = None
    bypass_success = None
    submission_success = None

    if isinstance(report, dict):
        captcha_type = report.get("captcha_type")
        bypass_success = _to_bool(report.get("bypass_success"))
        submission_success = _to_bool(
            report.get("submission_success") or report.get("login_success")
        )

    haystack = str(final_text or "").lower()
    if not captcha_type:
        for label in ["hcaptcha", "turnstile", "recaptcha v2", "recaptcha v3", "recaptcha"]:
            if label in haystack:
                captcha_type = label
                break

    if bypass_success is None:
        bypass_success = "score" in haystack and "error" not in haystack
    if submission_success is None:
        submission_success = any(
            k in haystack for k in ["submit", "submitted", "logged in", "login successful"]
        )

    return {
        "captcha_type": captcha_type,
        "bypass_success": bypass_success,
        "submission_success": submission_success,
    }


def _extract_visited_urls(target_url, *blobs):
    text_blob = "\n".join(str(item or "") for item in (target_url, *blobs))
    matches = re.findall(r"https?://[^\s\"'<>]+", text_blob)
    visited_urls = []
    seen = set()

    def _add(url):
        cleaned = str(url).rstrip(".,;:)]}>")
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        visited_urls.append(cleaned)

    _add(target_url)
    for item in matches:
        _add(item)
    redirected = [u for u in visited_urls if u.rstrip("/") != str(target_url).rstrip("/")]
    return visited_urls, redirected


def _walk_token_counts(payload):
    prompt = completion = 0
    found = False

    def _visit(node):
        nonlocal prompt, completion, found
        if isinstance(node, dict):
            p = node.get("input_token_count")
            c = node.get("output_token_count")
            if p is None:
                p = node.get("prompt_tokens") or node.get("total_prompt_tokens")
            if c is None:
                c = node.get("completion_tokens") or node.get("total_completion_tokens")
            if p is not None or c is not None:
                found = True
                prompt += int(p or 0)
                completion += int(c or 0)
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for value in node:
                _visit(value)

    _visit(payload)
    return found, prompt, completion


def _task_token_usage(task_run):
    dumped = _to_jsonable(task_run)
    found, prompt, completion = _walk_token_counts(dumped)
    return {
        "available": found,
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_tokens": prompt + completion,
        "source": "skyvern.task_run",
    }


def _task_final_text(task_run):
    if task_run is None:
        return None
    dumped = _to_jsonable(task_run)
    parts = []
    output = dumped.get("output") if isinstance(dumped, dict) else None
    if output is not None:
        parts.append(output if isinstance(output, str) else json.dumps(output, ensure_ascii=True))
    if isinstance(dumped, dict):
        if dumped.get("failure_reason"):
            parts.append(str(dumped["failure_reason"]))
        if dumped.get("status"):
            parts.append(f"status={dumped['status']}")
    return "\n".join(parts) if parts else json.dumps(dumped, ensure_ascii=True)


def _save_payload(payload):
    aggregate_path, unique_path, unique_stem = _build_result_paths(
        payload["target_url"], payload["run_id"]
    )
    payload["result_name"] = unique_stem
    serialized = json.dumps(_to_jsonable(payload), ensure_ascii=True)
    with aggregate_path.open("a", encoding="utf-8") as f:
        f.write(serialized + "\n")
    unique_path.write_text(serialized + "\n", encoding="utf-8")
    print("Saved run result to:", unique_path)
    print("Appended aggregate record to:", aggregate_path)
    return unique_path


def _prepare_skyvern_env():
    os.environ.setdefault("ENABLE_OPENAI", "true")
    os.environ.setdefault("LLM_KEY", "OPENAI_GPT4O")
    os.environ.setdefault("SKYVERN_TELEMETRY", "false")
    os.environ.setdefault("BROWSER_TYPE", "chromium-headful")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY is not set (skyvern-app/.env or environment)")


def _build_skyvern_client():
    from skyvern import Skyvern
    from skyvern.schemas.llm import LLMConfig

    env_config = _RQ2["rq2_env_config"]() if _RQ2 else ""
    max_steps = 6 if env_config else int(os.getenv("MAX_STEPS_PER_RUN", "50"))
    model_name = os.getenv("SKYVERN_MODEL", "gpt-4o")
    return Skyvern.local(
        use_in_memory_db=True,
        llm_config=LLMConfig(
            model_name=model_name,
            required_env_vars=["OPENAI_API_KEY"],
            supports_vision=True,
            add_assistant_prefix=False,
        ),
        settings={
            "ENABLE_OPENAI": True,
            "LLM_KEY": os.getenv("LLM_KEY", "OPENAI_GPT4O"),
            "BROWSER_TYPE": "chromium-headful",
            "MAX_STEPS_PER_RUN": max_steps,
            "SKYVERN_TELEMETRY": False,
        },
    )


async def _open_browser(skyvern):
    cdp = _RQ2["rq2_cdp_url"]() if _RQ2 else None
    if cdp:
        print(f"RQ2: Skyvern connecting over CDP at {cdp}")
        return await skyvern.connect_to_browser_over_cdp(cdp)
    return await skyvern.launch_local_browser(headless=False)


async def _probe_page(page):
    if not _RQ2:
        return {"available": False, "error": "rq2 helpers not loaded"}
    if page is None:
        return {"available": False, "error": "no current page"}
    try:
        raw = await page.evaluate(_RQ2["PROBE_JS"])
        return _RQ2["normalize_probe"](raw)
    except Exception as exc:
        return {"available": False, "error": str(exc)}


async def _run_single_url(target_url, terminal_log_path=None):
    task = _build_task(target_url)
    recording_file = _build_recording_path(target_url)
    recording_proc, recording_error = _start_screen_recording(recording_file)
    if recording_proc is not None:
        print("Recording started:", recording_file)
    else:
        print("Recording unavailable:", recording_error)

    env_config = _RQ2["rq2_env_config"]() if _RQ2 else ""
    if env_config:
        timeout_s = int(os.getenv("RQ2_JOB_SECONDS", "60"))
    else:
        timeout_s = AGENT_RUN_TIMEOUT_SECONDS
    max_steps = 6 if env_config else int(os.getenv("MAX_STEPS_PER_RUN", "50"))
    run_start = perf_counter()
    timed_out = False
    error = None
    task_run = None
    final_text = None
    page_probe = None
    browser = None
    page = None

    try:
        _prepare_skyvern_env()
        skyvern = _build_skyvern_client()
        browser = await _open_browser(skyvern)
        page = await browser.get_working_page()
        navigate_url = _navigate_url(target_url)
        print(f"Navigating to {navigate_url}")
        await page.goto(navigate_url, wait_until="domcontentloaded")

        async def _run():
            return await page.agent.run_task(
                prompt=task,
                url=navigate_url,
                max_steps=max_steps,
                timeout=timeout_s,
                data_extraction_schema={
                    "type": "object",
                    "properties": {
                        "instrument": {"type": "string"},
                        "login_success": {"type": "boolean"},
                        "grid_raised": {"type": "boolean"},
                        "widget_admitted": {"type": "boolean"},
                        "page_message": {"type": "string"},
                        "env_verdict": {"type": "string"},
                        "v3_score": {"type": ["number", "null"]},
                        "captcha_type": {"type": "string"},
                        "bypass_success": {"type": "boolean"},
                        "submission_success": {"type": "boolean"},
                    },
                },
            )

        task_run = await asyncio.wait_for(_run(), timeout=timeout_s)
        final_text = _task_final_text(task_run)
    except asyncio.TimeoutError:
        timed_out = True
        error = "TimeoutError"
        final_text = f"Agent run timed out after {timeout_s} seconds before completion."
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        final_text = str(exc)
    finally:
        _stop_screen_recording(recording_proc)
        if page is not None and _RQ2 and _RQ2["rq2_env_config"]():
            page_probe = await _probe_page(page)
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass

    run_time_seconds = perf_counter() - run_start
    report = _extract_json_report(str(final_text))
    if report is None and task_run is not None:
        dumped = _to_jsonable(task_run)
        output = dumped.get("output") if isinstance(dumped, dict) else None
        if isinstance(output, dict):
            report = output
        elif isinstance(output, str):
            report = _extract_json_report(output)
    outcomes = _derive_outcomes(final_text, report)
    visited_urls, redirected_urls = _extract_visited_urls(
        target_url, final_text, _to_jsonable(task_run)
    )
    env_verdict = None
    if _RQ2 and _RQ2["rq2_env_config"]():
        env_verdict = _RQ2["derive_env_verdict"](
            _RQ2["rq2_instrument"](),
            page_probe,
            report,
            str(final_text or ""),
            outcomes,
        )
        if env_verdict in {None, "", "unknown"}:
            env_verdict = "failed"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(uuid4()),
        "experiment_trial": os.getenv("EXPERIMENT_TRIAL", "").strip() or None,
        "target_url": target_url,
        "env_config": _RQ2["rq2_env_config"]() if _RQ2 else None,
        "instrument": _RQ2["rq2_instrument"]() if _RQ2 else None,
        "cdp_url": _RQ2["rq2_cdp_url"]() if _RQ2 else None,
        "page_probe": page_probe,
        "env_verdict": env_verdict,
        "task": task,
        "timed_out": timed_out,
        "error": error,
        "run_time_seconds": round(run_time_seconds, 6),
        "skyvern_task": _to_jsonable(task_run),
        "final_text": final_text,
        "final_report_raw": report,
        "captcha_type": outcomes.get("captcha_type"),
        "bypass_success": outcomes.get("bypass_success"),
        "submission_success": outcomes.get("submission_success"),
        "visited_urls": visited_urls,
        "redirected_or_secondary_urls": redirected_urls,
        "recording_path": str(recording_file) if recording_proc is not None else None,
        "recording_error": recording_error,
        "terminal_log_path": str(terminal_log_path) if terminal_log_path else None,
        "token_usage": _task_token_usage(task_run),
    }
    _save_payload(payload)
    print(f"URL: {target_url}")
    print(f"Env verdict: {env_verdict}")
    print(f"Captcha type: {outcomes.get('captcha_type')}")
    print(f"Bypass success: {outcomes.get('bypass_success')}")
    print(f"Submission success: {outcomes.get('submission_success')}")
    return payload


def _run_with_full_terminal_capture(target_url, terminal_log_path):
    script_bin = shutil.which("script")
    if not script_bin:
        return 1, "script command not found in PATH"
    command = (
        f"{shlex.quote(sys.executable or 'python')} {shlex.quote(str(Path(__file__).resolve()))} "
        f"--url {shlex.quote(target_url)} "
        f"--internal-full-terminal-capture "
        f"--terminal-log-path {shlex.quote(str(terminal_log_path))}"
    )
    completed = subprocess.run(
        [script_bin, "-q", "-f", "-c", command, str(terminal_log_path)],
        check=False,
    )
    return completed.returncode, None


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Skyvern captcha / environment tasks")
    parser.add_argument("--url", type=str, required=False, help="Run only one URL")
    parser.add_argument(
        "--cdp-url",
        type=str,
        default="",
        help="Connect to an already-running Chrome over CDP",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="",
        choices=["", "instrumented", "chrome_incognito", "chrome_cold", "chrome_full"],
        help="RQ2 browser configuration label stored with the result",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="",
        choices=["", "v2-invis", "turnstile", "turnstile-invis", "v3f"],
        help="RQ2 instrument: invisible reCaptcha v2 or Turnstile",
    )
    parser.add_argument(
        "--internal-full-terminal-capture",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--terminal-log-path",
        type=str,
        required=False,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _apply_cli_env(args):
    if args.cdp_url:
        os.environ["BROWSER_USE_CDP_URL"] = args.cdp_url
    if args.env_config:
        os.environ["RQ2_ENV_CONFIG"] = args.env_config
    if args.instrument:
        os.environ["RQ2_INSTRUMENT"] = args.instrument


def _target_urls(args):
    if args.url:
        return [args.url]
    raw = os.getenv("TARGET_URLS", "").strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    urls = [line.strip().strip(",").strip('"').strip("'") for line in raw.splitlines()]
    urls = [item for item in urls if item.startswith("http")]
    return urls or list(DEFAULT_URLS)


async def main():
    args = _parse_args()
    _apply_cli_env(args)
    target_urls = _target_urls(args)

    skip_script = bool(_RQ2 and _RQ2["rq2_env_config"]()) or args.internal_full_terminal_capture
    if not skip_script:
        for target_url in target_urls:
            terminal_log_path = _build_terminal_log_path(target_url)
            print("\n=== Running Skyvern for", target_url, "===")
            code, err = _run_with_full_terminal_capture(target_url, terminal_log_path)
            if code != 0:
                print("Sub-run failed for", target_url, "code=", code, "err=", err)
        return

    for target_url in target_urls:
        print("\n=== Running Skyvern for", target_url, "===")
        await _run_single_url(target_url, terminal_log_path=args.terminal_log_path)


if __name__ == "__main__":
    asyncio.run(main())
