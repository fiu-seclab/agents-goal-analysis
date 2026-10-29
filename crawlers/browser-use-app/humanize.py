"""Natural mouse + typing for Browser-Use.

The venv watchdog already types character-by-character. This module:

1. Slows that cadence to a human range.
2. Focuses inputs by moving the cursor and clicking (not silent DOM.focus).
3. Replaces the short straight-line mouse path with a curved, overshooting one.
4. Wanders the page and waits before the first type/click, and again before submit.

Call ``apply()`` once at process start, before ``Agent(...)``.

CDP itself (``--remote-debugging-port``) is still visible to reCaptcha. This
layer only humanizes pointer/keyboard physics on top of that.
"""

from __future__ import annotations

import asyncio
import math
import os
import random

HUMAN_ENV_DEFAULTS = {
    "BROWSER_USE_HUMANIZE": "true",
    "BROWSER_USE_HUMAN_MOUSE_STEPS_MIN": "18",
    "BROWSER_USE_HUMAN_MOUSE_STEPS_MAX": "36",
    "BROWSER_USE_HUMAN_MOUSE_STEP_DELAY_MIN": "0.010",
    "BROWSER_USE_HUMAN_MOUSE_STEP_DELAY_MAX": "0.028",
    "BROWSER_USE_HUMAN_MOUSE_JITTER_PX": "2.8",
    "BROWSER_USE_HUMAN_MOUSE_START_OFFSET_PX": "160.0",
    "BROWSER_USE_HUMAN_CLICK_HOLD_MIN": "0.070",
    "BROWSER_USE_HUMAN_CLICK_HOLD_MAX": "0.170",
    "BROWSER_USE_HUMAN_TYPE_DELAY_MIN": "0.070",
    "BROWSER_USE_HUMAN_TYPE_DELAY_MAX": "0.210",
    "BROWSER_USE_HUMAN_NEWLINE_DELAY_MIN": "0.120",
    "BROWSER_USE_HUMAN_NEWLINE_DELAY_MAX": "0.280",
    "BROWSER_USE_HUMAN_PRE_TYPE_PAUSE_MIN": "0.25",
    "BROWSER_USE_HUMAN_PRE_TYPE_PAUSE_MAX": "0.70",
    "BROWSER_USE_HUMAN_PRE_CLICK_PAUSE_MIN": "0.45",
    "BROWSER_USE_HUMAN_PRE_CLICK_PAUSE_MAX": "1.20",
    "BROWSER_USE_HUMAN_SETTLE_MIN": "1.20",
    "BROWSER_USE_HUMAN_SETTLE_MAX": "2.40",
}


def apply_env_defaults() -> dict[str, str]:
    applied = {}
    for key, value in HUMAN_ENV_DEFAULTS.items():
        if not os.getenv(key, "").strip():
            os.environ[key] = value
            applied[key] = value
    return applied


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _humanize_on() -> bool:
    return os.getenv("BROWSER_USE_HUMANIZE", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _bezier(p0, p1, p2, p3, t: float) -> tuple[float, float]:
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def _patch_watchdog() -> None:
    from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    if getattr(DefaultActionWatchdog, "_rq2_humanize_patched", False):
        return

    original_focus = DefaultActionWatchdog._focus_element_simple
    original_typing_delay = DefaultActionWatchdog._typing_delay
    original_input = DefaultActionWatchdog._input_text_element_node_impl
    original_move = DefaultActionWatchdog._human_move_mouse
    original_click = DefaultActionWatchdog.on_ClickElementEvent
    original_type = DefaultActionWatchdog.on_TypeTextEvent

    async def _dispatch_move(cdp_session, x: float, y: float) -> None:
        await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
            params={"type": "mouseMoved", "x": x, "y": y},
            session_id=cdp_session.session_id,
        )

    async def _human_move_mouse(self, cdp_session, target_x: float, target_y: float) -> None:
        if not self._human_cfg()["enabled"]:
            return await original_move(self, cdp_session, target_x, target_y)

        last_pos = getattr(self, "_last_mouse_pos", None)
        if not isinstance(last_pos, tuple) or len(last_pos) != 2:
            offset = self._human_cfg()["start_offset"]
            start = (
                target_x + random.uniform(-offset, offset),
                target_y + random.uniform(-offset * 0.7, offset * 0.7),
            )
        else:
            start = (float(last_pos[0]), float(last_pos[1]))

        dist = math.hypot(target_x - start[0], target_y - start[1])
        steps = max(14, min(42, int(dist / 14) + random.randint(8, 16)))
        overshoot = dist > 50 and random.random() < 0.5
        end = (
            (target_x + random.uniform(-10, 10), target_y + random.uniform(-8, 8))
            if overshoot
            else (target_x, target_y)
        )
        ctrl1 = (
            start[0] + (end[0] - start[0]) * 0.28 + random.uniform(-90, 90),
            start[1] + (end[1] - start[1]) * 0.18 + random.uniform(-70, 70),
        )
        ctrl2 = (
            start[0] + (end[0] - start[0]) * 0.72 + random.uniform(-70, 70),
            start[1] + (end[1] - start[1]) * 0.82 + random.uniform(-55, 55),
        )
        jitter = self._human_cfg()["mouse_jitter"]

        for step in range(1, steps + 1):
            t = step / steps
            eased = t * t * (3.0 - 2.0 * t)
            x, y = _bezier(start, ctrl1, ctrl2, end, eased)
            x += random.uniform(-jitter, jitter)
            y += random.uniform(-jitter, jitter)
            await _dispatch_move(cdp_session, x, y)
            mid = 1.0 - abs(0.5 - t) * 2.0
            delay = self._random_delay(
                self._human_cfg()["move_delay_min"],
                self._human_cfg()["move_delay_max"],
            )
            await asyncio.sleep(delay * (0.7 + 0.8 * (1.0 - mid)))

        if overshoot:
            for t in (0.4, 0.75, 1.0):
                x = end[0] + (target_x - end[0]) * t + random.uniform(-1.0, 1.0)
                y = end[1] + (target_y - end[1]) * t + random.uniform(-1.0, 1.0)
                await _dispatch_move(cdp_session, x, y)
                await asyncio.sleep(random.uniform(0.018, 0.040))

        await _dispatch_move(cdp_session, target_x, target_y)
        self._last_mouse_pos = (target_x, target_y)

    async def _human_wander(self, cdp_session) -> None:
        try:
            metrics = await cdp_session.cdp_client.send.Page.getLayoutMetrics(
                session_id=cdp_session.session_id
            )
            viewport = metrics.get("layoutViewport") or metrics.get("cssLayoutViewport") or {}
            width = float(viewport.get("clientWidth") or 1280)
            height = float(viewport.get("clientHeight") or 800)
        except Exception:
            width, height = 1280.0, 800.0
        for _ in range(random.randint(2, 4)):
            await self._human_move_mouse(
                cdp_session,
                random.uniform(width * 0.18, width * 0.82),
                random.uniform(height * 0.18, height * 0.72),
            )
            await asyncio.sleep(random.uniform(0.18, 0.55))
        try:
            last = getattr(self, "_last_mouse_pos", (width / 2, height / 2))
            await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
                params={
                    "type": "mouseWheel",
                    "x": last[0],
                    "y": last[1],
                    "deltaX": 0,
                    "deltaY": random.uniform(40, 160),
                },
                session_id=cdp_session.session_id,
            )
        except Exception:
            pass

    async def _maybe_settle(self) -> None:
        if getattr(self, "_rq2_settled", False) or not _humanize_on():
            return
        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session(
                target_id=None, focus=True
            )
            await _human_wander(self, cdp_session)
        except Exception as exc:
            self.logger.debug(f"Human settle wander failed: {exc}")
        await asyncio.sleep(
            random.uniform(
                _env_float("BROWSER_USE_HUMAN_SETTLE_MIN", 1.2),
                _env_float("BROWSER_USE_HUMAN_SETTLE_MAX", 2.4),
            )
        )
        self._rq2_settled = True

    async def _focus_element_simple(
        self,
        backend_node_id: int,
        object_id: str,
        cdp_session,
        input_coordinates=None,
    ) -> bool:
        if input_coordinates and "input_x" in input_coordinates and "input_y" in input_coordinates:
            click_x = float(input_coordinates["input_x"]) + random.uniform(-6.0, 6.0)
            click_y = float(input_coordinates["input_y"]) + random.uniform(-4.0, 4.0)
            try:
                await self._human_move_mouse(cdp_session, click_x, click_y)
                await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
                    params={
                        "type": "mousePressed",
                        "x": click_x,
                        "y": click_y,
                        "button": "left",
                        "clickCount": 1,
                    },
                    session_id=cdp_session.session_id,
                )
                await self._human_click_hold_delay()
                await cdp_session.cdp_client.send.Input.dispatchMouseEvent(
                    params={
                        "type": "mouseReleased",
                        "x": click_x,
                        "y": click_y,
                        "button": "left",
                        "clickCount": 1,
                    },
                    session_id=cdp_session.session_id,
                )
                return True
            except Exception as exc:
                self.logger.debug(f"Human click-to-focus failed, falling back: {exc}")
        return await original_focus(self, backend_node_id, object_id, cdp_session, input_coordinates)

    def _typing_delay(self, is_newline: bool = False) -> float:
        delay = original_typing_delay(self, is_newline=is_newline)
        if random.random() < 0.10:
            delay += random.uniform(0.14, 0.36)
        return delay

    async def _input_text_element_node_impl(
        self, element_node, text: str, clear: bool = True, is_sensitive: bool = False
    ):
        if _humanize_on():
            await asyncio.sleep(
                random.uniform(
                    _env_float("BROWSER_USE_HUMAN_PRE_TYPE_PAUSE_MIN", 0.25),
                    _env_float("BROWSER_USE_HUMAN_PRE_TYPE_PAUSE_MAX", 0.70),
                )
            )
        return await original_input(
            self, element_node, text, clear=clear, is_sensitive=is_sensitive
        )

    async def on_TypeTextEvent(self, event):
        await _maybe_settle(self)
        return await original_type(self, event)

    async def on_ClickElementEvent(self, event):
        await _maybe_settle(self)
        if _humanize_on():
            label = ""
            try:
                label = (event.node.node_name or "") + " " + (event.node.get_all_children_text(max_depth=2) or "")
            except Exception:
                pass
            pause_min = _env_float("BROWSER_USE_HUMAN_PRE_CLICK_PAUSE_MIN", 0.45)
            pause_max = _env_float("BROWSER_USE_HUMAN_PRE_CLICK_PAUSE_MAX", 1.20)
            if "login" in label.lower() or "submit" in label.lower():
                pause_min += 0.35
                pause_max += 0.55
            await asyncio.sleep(random.uniform(pause_min, pause_max))
        return await original_click(self, event)

    DefaultActionWatchdog._human_move_mouse = _human_move_mouse
    DefaultActionWatchdog._focus_element_simple = _focus_element_simple
    DefaultActionWatchdog._typing_delay = _typing_delay
    DefaultActionWatchdog._input_text_element_node_impl = _input_text_element_node_impl
    DefaultActionWatchdog.on_TypeTextEvent = on_TypeTextEvent
    DefaultActionWatchdog.on_ClickElementEvent = on_ClickElementEvent
    DefaultActionWatchdog._rq2_humanize_patched = True


def apply() -> None:
    applied = apply_env_defaults()
    _patch_watchdog()
    print(
        "Human-like actions enabled:",
        {
            "enabled": os.getenv("BROWSER_USE_HUMANIZE"),
            "mouse_steps": (
                os.getenv("BROWSER_USE_HUMAN_MOUSE_STEPS_MIN"),
                os.getenv("BROWSER_USE_HUMAN_MOUSE_STEPS_MAX"),
            ),
            "type_delay": (
                os.getenv("BROWSER_USE_HUMAN_TYPE_DELAY_MIN"),
                os.getenv("BROWSER_USE_HUMAN_TYPE_DELAY_MAX"),
            ),
            "settle": (
                os.getenv("BROWSER_USE_HUMAN_SETTLE_MIN"),
                os.getenv("BROWSER_USE_HUMAN_SETTLE_MAX"),
            ),
            "env_defaults_applied": sorted(applied),
        },
        flush=True,
    )
