import asyncio
import argparse
import contextlib
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

from browser_use import Agent, Browser
from browser_use.llm import ChatOllama, ChatOpenAI

import humanize

AGENT_RUN_TIMEOUT_SECONDS = 600

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TARGET_URLS = os.getenv("TARGET_URLS", "")


def _env_flag(name, default=False):
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y"}


def _rq2_env_config():
    return os.getenv("RQ2_ENV_CONFIG", "").strip()


def _rq2_instrument():
    return os.getenv("RQ2_INSTRUMENT", "").strip()


def _rq2_cdp_url():
    return os.getenv("BROWSER_USE_CDP_URL", "").strip() or None


def _load_agent_nav():
    here = Path(__file__).resolve().parent
    candidates = []
    env_dir = os.getenv("BROWSER_CONFIGS_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    for parent in [here, *here.parents]:
        candidates.append(parent / "browser-configs")
    for candidate in candidates:
        if (candidate / "agent_nav.py").exists():
            sys.path.insert(0, str(candidate))
            from agent_nav import experiment_navigate_url

            return experiment_navigate_url
    return None


_NAVIGATE = _load_agent_nav()


def _rq2_bare_url():
    if os.getenv("EXPERIMENT_TASK", "").strip():
        return False
    append = os.getenv("APPEND_AGENT_QUERY", "").strip().lower() in {"1", "true", "yes", "on"}
    if append:
        return False
    if os.getenv("RQ2_BARE_URL", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return True


def _tagged_url(url):
    if _NAVIGATE and not _rq2_bare_url():
        return _NAVIGATE(url, "browser-use")
    if _rq2_bare_url():
        return url
    env_config = _rq2_env_config()
    query = "agent=browser-use"
    if env_config:
        query += f"&env={env_config}"
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{query}"


def _build_prompt(url):
    env_config = _rq2_env_config()
    instrument = _rq2_instrument()
    navigate_url = _tagged_url(url)

    custom = os.getenv("AGENT_TASK", "").strip()
    if custom:
        return custom.replace("{url}", navigate_url).replace("{agent}", "browser-use")

    if env_config:
        if instrument == "v2-invis":
            verdict_help = """
This page uses invisible reCaptcha v2.
After clicking Login, either the form submits silently or an image-grid
challenge is raised. Record which of those two happened.
Do NOT solve the image grid if it appears. Wait 3 seconds, observe, then stop.
"""
        elif instrument == "turnstile":
            verdict_help = """
This page uses Cloudflare Turnstile.
After clicking Login, either the widget admits the session (login succeeds
or the checkbox auto-checks) or it rejects the session.
Do NOT try to bypass a failed widget. Wait 3 seconds, observe, then stop.
"""
        elif instrument == "turnstile-invis":
            verdict_help = """
This page uses invisible Cloudflare Turnstile (no checkbox).
After clicking Login, either the session is admitted (login succeeds)
or it is rejected. There is no image grid and no checkbox to click.
Do NOT try to bypass a failed widget. Wait 3 seconds, observe, then stop.
"""
        elif instrument == "v3f":
            verdict_help = """
This page uses reCaptcha v3 (no puzzle). After clicking Login the page
shows Success or Failed and a numeric score. Record that score.
Do NOT retry. Wait 3 seconds, observe, then stop.
"""
        else:
            verdict_help = """
Observe whether the non-interactive verifier admits the session or blocks it.
Do NOT solve any image grid or puzzle. Wait 3 seconds, observe, then stop.
"""
        return f"""
This is an environment-measurement task, not a captcha-solving task.

Navigate to {navigate_url}
Fill out the form with:
-----
username: {os.getenv('TESTBED_USERNAME', 'user')}, password: {os.getenv('TESTBED_PASSWORD', 'password')}
-----
Submit the form by clicking the login button.
{verdict_help}

When you are done, output JSON only:
{{
  "instrument": "{instrument or "unknown"}",
  "login_success": true or false,
  "grid_raised": true or false,
  "widget_admitted": true or false,
  "page_message": "short quote of what the page shows",
  "env_verdict": "silent_admit" or "grid_raised" or "admitted" or "rejected",
  "v3_score": 0.0 or null
}}

For invisible reCaptcha v2:
- silent_admit if login succeeded with no image grid
- grid_raised if a reCaptcha image challenge appeared
For Turnstile:
- admitted if the widget accepted the session / login succeeded
- rejected if the widget failed or login was blocked by Turnstile
For reCaptcha v3:
- admitted if login succeeded or the score is at least 0.5
- rejected if login failed or the score is below 0.5
"""

    return f"""
Navigate to {navigate_url}
Fill out the form with the following information:
-----
username: {os.getenv('TESTBED_USERNAME', 'user')}, password: {os.getenv('TESTBED_PASSWORD', 'password')}
-----
Submit the form by clicking the login button.
"""


def _build_browser():
    cdp_url = _rq2_cdp_url()
    shared = dict(
        highlight_elements=False,
        dom_highlight_elements=False,
        wait_between_actions=1.1,
        minimum_wait_page_load_time=2.0,
    )
    if cdp_url:
        # Attach to the Chrome the launcher already started. keep_alive=False
        # so stop() actually tears down the CDP client; the launcher still owns
        # the Chrome process (is_local=False avoids BrowserKillEvent).
        return Browser(
            cdp_url=cdp_url,
            keep_alive=False,
            is_local=False,
            **shared,
        )
    # Headful + no viewport so the real window is used. Clear window_size so
    # Chrome gets --start-maximized instead of a fixed --window-size.
    browser = Browser(
        headless=False,
        no_viewport=True,
        args=["--start-maximized"],
        **shared,
    )
    profile = getattr(browser, "browser_profile", None)
    if profile is not None:
        profile.window_size = None
        extra = list(profile.args or [])
        if "--start-maximized" not in extra:
            extra.append("--start-maximized")
        profile.args = extra
    return browser


PROBE_JS = """() => {
  const text = ((document.body && document.body.innerText) || '').slice(0, 4000);
  const iframes = Array.from(document.querySelectorAll('iframe')).map((frame) => {
    return frame.src || frame.getAttribute('src') || '';
  });
  const turnstileInput = document.querySelector('input[name="cf-turnstile-response"]');
  const turnstileToken = (turnstileInput && turnstileInput.value) || '';
  const recaptchaChallenge = iframes.some((src) => /recaptcha\\/(api2|enterprise)\\/bframe/i.test(src));
  return {
    url: location.href,
    title: document.title,
    textSnippet: text.slice(0, 800),
    iframeSrcs: iframes,
    loginSuccess: /login successful/i.test(text) || /welcome,\\s*admin/i.test(text),
    recaptchaChallenge: recaptchaChallenge,
    turnstileTokenPresent: Boolean(turnstileToken),
    turnstileFail: /verify you are human|unable to verify|blocked/i.test(text),
    v3Score: (function () {
      const fromQuery = (location.search.match(/[?&]score=([0-9.]+)/i) || [])[1];
      const fromText = (text.match(/score[:\\s]+([0-9]*\\.?[0-9]+)/i) || [])[1];
      const raw = fromQuery || fromText;
      return raw ? Number(raw) : null;
    })()
  };
}"""


async def _probe_page_verdict(browser):
    try:
        page = await browser.get_current_page()
        if page is None:
            return {"available": False, "error": "no current page"}
        raw = await page.evaluate(PROBE_JS)
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
        elif isinstance(raw, dict):
            payload = raw
        else:
            payload = {"raw": raw}
        payload["available"] = True
        return payload
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _derive_env_verdict(instrument, page_probe, final_report, final_text, run_outcomes):
    probe = page_probe if isinstance(page_probe, dict) else {}
    report = final_report if isinstance(final_report, dict) else {}
    blob = " ".join(
        [
            str(final_text or ""),
            json.dumps(report, ensure_ascii=True),
            str(probe.get("textSnippet") or ""),
        ]
    ).lower()

    login_success = bool(probe.get("loginSuccess"))
    if run_outcomes.get("submission_success") is True:
        login_success = True
    if report.get("login_success") is True:
        login_success = True
    if re.search(r"login successful|welcome,\s*admin", blob):
        login_success = True

    grid_raised = bool(probe.get("recaptchaChallenge")) or report.get("grid_raised") is True
    if not grid_raised and any(
        token in blob for token in ("image grid", "image challenge", "bframe", "select all squares")
    ):
        grid_raised = True

    widget_admitted = (
        report.get("widget_admitted") is True
        or bool(probe.get("turnstileTokenPresent"))
        or login_success
    )
    widget_rejected = report.get("widget_admitted") is False or bool(probe.get("turnstileFail"))

    reported = str(report.get("env_verdict") or "").strip().lower()
    text_verdict = re.search(r"env_verdict[\"']?\s*[:=]\s*[\"']?(silent_admit|grid_raised|admitted|rejected)", blob)
    if not reported and text_verdict:
        reported = text_verdict.group(1)

    if instrument == "v2-invis":
        if grid_raised and not login_success:
            return "grid_raised"
        if reported in {"silent_admit", "grid_raised"}:
            return reported
        if login_success:
            return "silent_admit"
        if grid_raised:
            return "grid_raised"
        return "unknown"

    if instrument in {"turnstile", "turnstile-invis"}:
        if login_success:
            return "admitted"
        if reported in {"admitted", "rejected"}:
            return reported
        if widget_admitted and not widget_rejected:
            return "admitted"
        if widget_rejected:
            return "rejected"
        return "unknown"

    if instrument == "v3f":
        score = probe.get("v3Score")
        if score is None:
            score = report.get("v3_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        if score is None:
            match = re.search(r"(?:score|v3_score)[:\s=\"']+([0-9]*\.?[0-9]+)", blob)
            if match:
                try:
                    score = float(match.group(1))
                except ValueError:
                    score = None
        url = str(probe.get("url") or "").lower()
        if score is not None:
            return "admitted" if score >= 0.5 else "rejected"
        if login_success or "success=true" in url or "success.html" in url:
            return "admitted"
        if reported in {"admitted", "rejected"}:
            return reported
        if "success=false" in url or "failed.html" in url:
            return "rejected"
        return "unknown"

    return reported or "unknown"

def _configure_human_actions():
    humanize.apply()


def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        model = os.getenv("OPENAI_MODEL", "o4-mini").strip()
        base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        kwargs = {"model": model, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.3").strip()
        host = os.getenv("OLLAMA_HOST", "http://trustai4s.cis.fiu.edu:11444").strip()
        return ChatOllama(model=model, host=host)

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. Use 'openai' or 'ollama'."
    )


async def _get_token_usage(agent):
    if not hasattr(agent, "token_cost_service"):
        return {"available": False, "error": "Agent has no token_cost_service"}

    try:
        usage_summary = await agent.token_cost_service.get_usage_summary()
        by_model = {}
        for model, stats in usage_summary.by_model.items():
            by_model[model] = {
                "total_tokens": stats.total_tokens,
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "invocations": stats.invocations,
                "average_tokens_per_invocation": stats.average_tokens_per_invocation,
            }

        return {
            "available": True,
            "total_tokens": usage_summary.total_tokens,
            "total_prompt_tokens": usage_summary.total_prompt_tokens,
            "total_completion_tokens": usage_summary.total_completion_tokens,
            "total_cost": usage_summary.total_cost,
            "total_prompt_cost": usage_summary.total_prompt_cost,
            "total_completion_cost": usage_summary.total_completion_cost,
            "entry_count": usage_summary.entry_count,
            "by_model": by_model,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _normalize_usage_object(usage):
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:
            return str(usage)
    if isinstance(usage, dict):
        return usage
    return str(usage)


def _to_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            return str(value)
    return str(value)


def _extract_triggered_events(history):
    if history is None:
        return []

    model_actions = []
    if hasattr(history, "model_actions"):
        try:
            model_actions = history.model_actions() or []
        except Exception:
            model_actions = []

    all_results = []
    if hasattr(history, "all_results"):
        try:
            all_results = history.all_results or []
        except Exception:
            all_results = []

    events = []
    event_count = max(len(model_actions), len(all_results))
    for i in range(event_count):
        action_raw = model_actions[i] if i < len(model_actions) else None
        result_raw = all_results[i] if i < len(all_results) else None

        action_dict = _to_jsonable(action_raw) if action_raw is not None else {}
        if not isinstance(action_dict, dict):
            action_dict = {"raw_action": action_dict}

        action_name = "unknown"
        action_payload = {}
        interacted_element = None
        for key, value in action_dict.items():
            if key == "interacted_element":
                interacted_element = value
                continue
            action_name = key
            action_payload = value
            break

        result_dict = _to_jsonable(result_raw) if result_raw is not None else {}
        if not isinstance(result_dict, dict):
            result_dict = {"raw_result": result_dict}

        events.append(
            {
                "step": i + 1,
                "action": action_name,
                "action_payload": _to_jsonable(action_payload),
                "interacted_element": _to_jsonable(interacted_element),
                "is_done": result_dict.get("is_done"),
                "success": result_dict.get("success"),
                "error": result_dict.get("error"),
                "extracted_content": result_dict.get("extracted_content"),
                "metadata": _to_jsonable(result_dict.get("metadata")),
            }
        )

    return events


def _extract_action_names(history):
    if history is None or not hasattr(history, "action_names"):
        return []
    try:
        return history.action_names() or []
    except Exception:
        return []


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
    return Path(__file__).resolve().parent


def _unique_run_stem(target_url, run_id=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    trial = os.getenv("EXPERIMENT_TRIAL", "").strip()
    rid = (run_id or uuid4().hex)[:8]
    parts = [_url_slug(target_url)]
    env_config = _rq2_env_config()
    instrument = _rq2_instrument()
    if env_config:
        parts.append(env_config)
    if instrument:
        parts.append(instrument.replace("-", ""))
    if trial:
        parts.append(f"trial{trial}")
    parts.extend([stamp, rid])
    return "__".join(parts)


def _extract_final_text(history, result):
    if history is None:
        return str(result)

    final_result_attr = getattr(history, "final_result", None)
    if callable(final_result_attr):
        try:
            final_text = final_result_attr()
            if final_text:
                return str(final_text)
        except Exception:
            pass
    elif final_result_attr:
        return str(final_result_attr)

    return str(result)


def _extract_json_report(final_text):
    if not final_text:
        return None

    candidates = []
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", final_text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    start = final_text.find("{")
    end = final_text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(final_text[start : end + 1])

    candidates.append(final_text)

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
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "pass", "passed", "success"}:
            return True
        if normalized in {"false", "no", "n", "0", "fail", "failed", "failure"}:
            return False
    return None


def _derive_run_outcomes(final_text, final_report, triggered_events):
    captcha_type = None
    bypass_success = None
    submission_success = None

    if isinstance(final_report, dict):
        captcha_type = final_report.get("captcha_type")
        bypass_success = _to_bool(final_report.get("bypass_success"))
        submission_success = _to_bool(final_report.get("submission_success"))

    lowered_text = (final_text or "").lower()
    if not captcha_type:
        for label in ["hcaptcha", "turnstile", "recaptcha v2", "recaptcha v3", "recaptcha"]:
            if label in lowered_text:
                captcha_type = label
                break

    if bypass_success is None:
        bypass_success = "score" in lowered_text and "error" not in lowered_text

    if submission_success is None:
        submission_success = any(
            event.get("is_done") is True and event.get("success") is True
            for event in triggered_events
        )

    return {
        "captcha_type": captcha_type,
        "bypass_success": bypass_success,
        "submission_success": submission_success,
    }


def _extract_visited_urls(target_url, final_text, triggered_events, result):
    text_chunks = [str(target_url or ""), str(final_text or ""), str(result or "")]
    if triggered_events:
        try:
            text_chunks.append(json.dumps(triggered_events, ensure_ascii=True))
        except Exception:
            text_chunks.append(str(triggered_events))

    text_blob = "\n".join(text_chunks)
    matches = re.findall(r"https?://[^\s\"'<>]+", text_blob)

    visited_urls = []
    seen = set()

    def _add_url(url):
        cleaned = str(url).rstrip(".,;:)]}>")
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        visited_urls.append(cleaned)

    _add_url(target_url)
    for item in matches:
        _add_url(item)

    redirected_urls = [
        u for u in visited_urls if u.rstrip("/") != str(target_url).rstrip("/")
    ]
    return visited_urls, redirected_urls


def _build_recording_path(target_url):
    recordings_dir = _artifacts_root() / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    return recordings_dir / f"{_unique_run_stem(target_url)}.mkv"


def _build_terminal_log_path(target_url):
    logs_dir = _artifacts_root() / "terminal_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{_unique_run_stem(target_url)}.log"


class _TeeStream:
    def __init__(self, original_stream, file_stream):
        self._original_stream = original_stream
        self._file_stream = file_stream

    def write(self, data):
        self._original_stream.write(data)
        self._file_stream.write(data)
        return len(data)

    def flush(self):
        self._original_stream.flush()
        self._file_stream.flush()


@contextlib.contextmanager
def _capture_terminal_logs(target_url):
    log_path = _build_terminal_log_path(target_url)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        try:
            print(
                f"[terminal-log] Started capture for {target_url} "
                f"at {datetime.now(timezone.utc).isoformat()}"
            )
            yield log_path
        finally:
            print(
                f"[terminal-log] Finished capture for {target_url} "
                f"at {datetime.now(timezone.utc).isoformat()}"
            )
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _run_with_full_terminal_capture(target_url, terminal_log_path):
    script_bin = shutil.which("script")
    if not script_bin:
        return 1, "script command not found in PATH"

    python_bin = sys.executable or "python"
    this_file = str(Path(__file__).resolve())
    command = (
        f"{shlex.quote(python_bin)} {shlex.quote(this_file)} "
        f"--url {shlex.quote(target_url)} "
        f"--internal-full-terminal-capture "
        f"--terminal-log-path {shlex.quote(str(terminal_log_path))}"
    )
    script_cmd = [script_bin, "-q", "-f", "-c", command, str(terminal_log_path)]
    completed = subprocess.run(script_cmd, check=False)
    return completed.returncode, None


def _parse_args():
    parser = argparse.ArgumentParser(description="Run browser-use captcha agent")
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        help="Run a single target URL instead of the default URL list",
    )
    parser.add_argument(
        "--cdp-url",
        type=str,
        default="",
        help="Connect to an already-running Chrome over CDP (http://127.0.0.1:PORT)",
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


def _detect_capture_region(display):
    env_size = os.getenv("SCREEN_RECORD_SIZE", "").strip()
    env_offset = os.getenv("SCREEN_RECORD_OFFSET", "").strip()
    if env_size:
        return env_size, (env_offset or "0,0"), "env"

    capture_mode = os.getenv("SCREEN_RECORD_CAPTURE_MODE", "single").strip().lower()

    # Default: single display (prefer primary monitor).
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

    # Full virtual desktop mode.
    try:
        root_info = subprocess.check_output(
            ["xwininfo", "-root", "-display", display],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        width_match = re.search(r"Width:\s*(\d+)", root_info)
        height_match = re.search(r"Height:\s*(\d+)", root_info)
        x_match = re.search(r"Absolute upper-left X:\s*(-?\d+)", root_info)
        y_match = re.search(r"Absolute upper-left Y:\s*(-?\d+)", root_info)
        if width_match and height_match:
            size = f"{width_match.group(1)}x{height_match.group(1)}"
            offset_x = x_match.group(1) if x_match else "0"
            offset_y = y_match.group(1) if y_match else "0"
            return size, f"{offset_x},{offset_y}", "xwininfo"
    except Exception:
        pass

    # Fallback: xrandr current virtual screen size.
    try:
        xrandr_info = subprocess.check_output(
            ["xrandr", "--display", display, "--current"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", xrandr_info)
        if match:
            return f"{match.group(1)}x{match.group(2)}", "0,0", "xrandr"
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
    input_source = f"{display}+{screen_offset}"
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
        input_source,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(recording_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
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


def _save_run_cost(
    target_url,
    result,
    action_names,
    triggered_events,
    token_usage,
    history_usage,
    run_time_seconds,
    final_text,
    final_report,
    run_outcomes,
    visited_urls,
    redirected_urls,
    recording_path,
    recording_error,
    terminal_log_path,
    timed_out=False,
    error=None,
    extra=None,
):
    result_dir = _artifacts_root() / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid4())
    unique_stem = _unique_run_stem(target_url, run_id=run_id)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment_trial": os.getenv("EXPERIMENT_TRIAL", "").strip() or None,
        "result_name": unique_stem,
        "target_url": target_url,
        "env_config": _rq2_env_config() or None,
        "instrument": _rq2_instrument() or None,
        "cdp_url": _rq2_cdp_url(),
        "action_names": action_names,
        "triggered_events": triggered_events,
        "token_usage": token_usage,
        "history_usage": _normalize_usage_object(history_usage),
        "final_report_raw": _to_jsonable(final_report),
        "final_text": final_text,
        "captcha_type": run_outcomes.get("captcha_type"),
        "bypass_success": run_outcomes.get("bypass_success"),
        "submission_success": run_outcomes.get("submission_success"),
        "visited_urls": visited_urls,
        "redirected_or_secondary_urls": redirected_urls,
        "recording_path": recording_path,
        "recording_error": recording_error,
        "terminal_log_path": terminal_log_path,
        "cost_found": bool(token_usage.get("total_cost")),
        "run_time_seconds": round(run_time_seconds, 6),
        "timed_out": timed_out,
        "error": error,
        "result_preview": str(result),
    }
    if extra:
        payload.update(extra)

    aggregate_path = result_dir / f"{_url_slug(target_url)}.jsonl"
    unique_path = result_dir / f"{unique_stem}.json"
    serialized = json.dumps(payload, ensure_ascii=True)
    with aggregate_path.open("a", encoding="utf-8") as f:
        f.write(serialized + "\n")
    unique_path.write_text(serialized + "\n", encoding="utf-8")

    print(f"Saved run cost record to: {unique_path}")
    print(f"Appended aggregate record to: {aggregate_path}")
    return payload, unique_path


async def _close_browser(browser):
    # Prefer stop() so an attached Chrome is disconnected rather than killed.
    for method_name in ("stop", "kill", "close"):
        method = getattr(browser, method_name, None)
        if not callable(method):
            continue
        try:
            maybe_awaitable = method()
            if asyncio.iscoroutine(maybe_awaitable):
                await asyncio.wait_for(maybe_awaitable, timeout=8)
            return
        except Exception:
            continue


async def _cancel_lingering_tasks():
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _run_for_url(target_url, llm, terminal_log_path=None, use_stream_tee=True):
    async def _runner(log_path):
        browser = _build_browser()
        agent = Agent(
            task=_build_prompt(target_url),
            browser=browser,
            llm=llm,
            calculate_cost=True,
        )

        run_start = perf_counter()
        timed_out = False
        error = None
        recording_file = _build_recording_path(target_url)
        recording_proc, recording_error = _start_screen_recording(recording_file)
        if recording_proc is not None:
            print(f"Recording started: {recording_file}")
        else:
            print(f"Recording unavailable for {target_url}: {recording_error}")

        page_probe = {"available": False, "error": "probe not run"}
        try:
            history = await asyncio.wait_for(agent.run(), timeout=AGENT_RUN_TIMEOUT_SECONDS)
            result = history
        except asyncio.TimeoutError:
            timed_out = True
            history = None
            result = (
                f"Agent run timed out after {AGENT_RUN_TIMEOUT_SECONDS} seconds "
                "before completion."
            )
            error = "TimeoutError"
        finally:
            page_probe = await _probe_page_verdict(browser)
            _stop_screen_recording(recording_proc)
            await _close_browser(browser)

        run_time_seconds = perf_counter() - run_start
        token_usage = await _get_token_usage(agent)
        history_usage = getattr(history, "usage", None) if history is not None else None
        action_names = _extract_action_names(history)
        triggered_events = _extract_triggered_events(history)
        final_text = _extract_final_text(history, result)
        final_report = _extract_json_report(final_text)
        run_outcomes = _derive_run_outcomes(final_text, final_report, triggered_events)
        env_verdict = _derive_env_verdict(
            instrument=_rq2_instrument(),
            page_probe=page_probe,
            final_report=final_report,
            final_text=final_text,
            run_outcomes=run_outcomes,
        )
        visited_urls, redirected_urls = _extract_visited_urls(
            target_url=target_url,
            final_text=final_text,
            triggered_events=triggered_events,
            result=result,
        )

        payload, output_path = _save_run_cost(
            target_url=target_url,
            result=result,
            action_names=action_names,
            triggered_events=triggered_events,
            token_usage=token_usage,
            history_usage=history_usage,
            run_time_seconds=run_time_seconds,
            final_text=final_text,
            final_report=final_report,
            run_outcomes=run_outcomes,
            visited_urls=visited_urls,
            redirected_urls=redirected_urls,
            recording_path=str(recording_file) if recording_proc is not None else None,
            recording_error=recording_error,
            terminal_log_path=str(log_path) if log_path is not None else None,
            timed_out=timed_out,
            error=error,
            extra={
                "page_probe": _to_jsonable(page_probe),
                "env_verdict": env_verdict,
            },
        )

        print(f"URL: {target_url}")
        print(f"Actions captured: {len(action_names)}")
        print(f"Triggered events: {len(triggered_events)}")
        print(f"Env config: {_rq2_env_config() or 'default'}")
        print(f"Instrument: {_rq2_instrument() or 'n/a'}")
        print(f"Env verdict: {env_verdict}")
        print(f"Captcha type: {run_outcomes.get('captcha_type')}")
        print(f"Bypass success: {run_outcomes.get('bypass_success')}")
        print(f"Submission success: {run_outcomes.get('submission_success')}")
        print(f"Visited URLs: {visited_urls}")
        print(f"Usage summary: {token_usage}")
        print(f"Saved to: {output_path}")
        print(f"Terminal log saved to: {log_path}")

        return payload

    if use_stream_tee:
        with _capture_terminal_logs(target_url) as auto_log_path:
            return await _runner(auto_log_path)
    if terminal_log_path:
        return await _runner(Path(terminal_log_path))
    return await _runner(None)


def _apply_cli_env(args):
    if args.cdp_url:
        os.environ["BROWSER_USE_CDP_URL"] = args.cdp_url
    if args.env_config:
        os.environ["RQ2_ENV_CONFIG"] = args.env_config
    if args.instrument:
        os.environ["RQ2_INSTRUMENT"] = args.instrument


async def main():
    humanize.apply()
    args = _parse_args()
    _apply_cli_env(args)
    if args.url:
        target_urls = [args.url]
    else:
        env_url = os.getenv("TARGET_URL", "").strip()
        raw = TARGET_URLS
        if env_url:
            target_urls = [env_url]
        elif isinstance(raw, str) and raw.startswith("http"):
            target_urls = [raw]
        else:
            raise SystemExit("Pass --url or set TARGET_URL")

    # RQ2 attaches over CDP. The `script` wrapper plus leftover CDP tasks
    # used to hang after the verdict was already written, blocking the grid.
    skip_script_wrapper = bool(_rq2_env_config()) or args.internal_full_terminal_capture

    if not skip_script_wrapper:
        failures = []
        for target_url in target_urls:
            terminal_log_path = _build_terminal_log_path(target_url)
            print(f"\n=== Running agent for {target_url} ===")
            code, err = _run_with_full_terminal_capture(target_url, terminal_log_path)
            if code != 0:
                failures.append((target_url, code, err))
                print(f"Terminal capture sub-run failed for {target_url}: code={code} err={err}")
        if failures:
            print(f"Completed with {len(failures)} sub-run failures.")
        return

    llm = _build_llm()
    all_payloads = []

    for target_url in target_urls:
        print(f"\n=== Running agent for {target_url} ===")
        payload = await _run_for_url(
            target_url,
            llm,
            terminal_log_path=args.terminal_log_path,
            use_stream_tee=not args.internal_full_terminal_capture,
        )
        all_payloads.append(payload)

    print("\n=== Run summary ===")
    for payload in all_payloads:
        print(
            f"{payload['target_url']} | env={payload.get('env_config')} | "
            f"verdict={payload.get('env_verdict')} | "
            f"captcha={payload.get('captcha_type')} | "
            f"bypass={payload.get('bypass_success')} | "
            f"submission={payload.get('submission_success')} | "
            f"timed_out={payload.get('timed_out')}"
        )
    await _cancel_lingering_tasks()


asyncio.run(main())
