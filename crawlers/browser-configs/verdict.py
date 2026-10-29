"""Shared RQ2 prompt + environment-verdict helpers."""

from __future__ import annotations

import json
import os
import re

from agent_nav import experiment_navigate_url


def rq2_env_config() -> str:
    return os.getenv("RQ2_ENV_CONFIG", "").strip()


def rq2_instrument() -> str:
    return os.getenv("RQ2_INSTRUMENT", "").strip()


def rq2_cdp_url() -> str | None:
    return os.getenv("BROWSER_USE_CDP_URL", "").strip() or None


def rq2_prompt(url: str, agent_slug: str) -> str:
    env_config = rq2_env_config()
    instrument = rq2_instrument()
    append = os.getenv("APPEND_AGENT_QUERY", "").strip().lower() in {"1", "true", "yes", "on"}
    always = bool(os.getenv("EXPERIMENT_TASK", "").strip())
    navigate = experiment_navigate_url(url, agent_slug) if (append or always) else url
    custom = os.getenv("AGENT_TASK", "").strip()
    if custom:
        return custom.replace("{url}", navigate).replace("{agent}", agent_slug)

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
After the page opens, wait at least 8 seconds before typing or clicking
Login so the background verifier can finish.
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

Navigate to {navigate}
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
    turnstileFail: /verify you are human|unable to verify|verification failed|blocked/i.test(text),
    v3Score: (function () {
      const fromQuery = (location.search.match(/[?&]score=([0-9.]+)/i) || [])[1];
      const fromText = (text.match(/score[:\\s]+([0-9]*\\.?[0-9]+)/i) || [])[1];
      const raw = fromQuery || fromText;
      return raw ? Number(raw) : null;
    })()
  };
}"""


def normalize_probe(raw):
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


def derive_env_verdict(instrument, page_probe, final_report, final_text, run_outcomes=None):
    run_outcomes = run_outcomes or {}
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
    widget_rejected = (
        report.get("widget_admitted") is False
        or bool(probe.get("turnstileFail"))
        or "verification failed" in blob
    )

    reported = str(report.get("env_verdict") or "").strip().lower()
    # Take the last concrete verdict. The task prompt itself contains
    # `"env_verdict": "silent_admit" or "grid_raised" ...` and must not win.
    text_verdicts = [
        match
        for match in re.findall(
            r'env_verdict["\']?\s*[:=]\s*["\']?(silent_admit|grid_raised|admitted|rejected|failed)',
            blob,
        )
        if " or " not in blob[blob.find(match) : blob.find(match) + 40]
    ]
    # Simpler: prefer values that appear as a JSON string, last one wins.
    json_verdicts = re.findall(
        r'"env_verdict"\s*:\s*"(silent_admit|grid_raised|admitted|rejected)"',
        blob,
    )
    if not reported:
        if json_verdicts:
            reported = json_verdicts[-1]
        elif text_verdicts:
            reported = text_verdicts[-1]

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
        url = str(probe.get("url") or "").lower()
        if login_success or "success=true" in url or "success.html" in url:
            return "admitted"
        if "success=false" in url or "failed.html" in url:
            return "rejected"
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
        if re.search(
            r"extracted from page:.*\bfailed\b|'text': 'failed'|\"text\": \"failed\"",
            blob,
        ):
            return "rejected"
        if re.search(
            r"extracted from page:.*\bsuccess\b|'text': 'success'|\"text\": \"success\"",
            blob,
        ):
            return "admitted"
        if re.search(r"\bfailed\b", blob) and "score" in blob:
            return "rejected"
        return "unknown"

    return reported or "unknown"
