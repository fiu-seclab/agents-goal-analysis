# Browser Use Humanization Patch

This project includes a local customization of Browser Use to make interactions more human-like (inspired by HLISA-style behavior).

`main.py` calls `humanize.apply()` on startup. That sets slower per-key delays and patches focus so the cursor moves to an input and clicks before typing, instead of silent `DOM.focus`. Override any `BROWSER_USE_HUMAN_*` env var to change the cadence; set `BROWSER_USE_HUMANIZE=false` to turn the layer off.

## What was changed

The low-level interaction engine can be modified in:

- `.venv/lib/python3.11/site-packages/browser_use/browser/watchdogs/default_action_watchdog.py`

### Added behavior

- Human-like mouse trajectories (multi-step movement with easing and jitter)
- Variable click press/hold timing
- Variable per-keystroke typing delays
- Configurable interaction timing via environment variables

### Existing behavior preserved

- CDP click/type fallback logic
- JavaScript click fallback when elements are occluded
- Framework event dispatch for React/Vue/Angular compatibility
- Download detection and timeout handling

## Technical details

### New helper methods

Added to `DefaultActionWatchdog`:

- `_env_bool(name, default)`
- `_env_int(name, default)`
- `_env_float(name, default)`
- `_human_cfg()`
- `_random_delay(min_delay, max_delay)`
- `_typing_delay(is_newline=False)`
- `_human_move_mouse(cdp_session, target_x, target_y)`
- `_human_click_hold_delay()`

### Patched execution paths

Mouse/click:

- `_click_element_node_impl(...)`
- `_click_on_coordinate(...)`

Typing:

- `_type_to_page(...)`
- `_input_text_element_node_impl(...)`

## Environment configuration

All settings are optional.

### Master switch

- `BROWSER_USE_HUMANIZE` (default: `true`)

Set `false` to disable the humanization layer and use faster/default motion timings.

### Mouse movement

- `BROWSER_USE_HUMAN_MOUSE_STEPS_MIN` (default: `4`)
- `BROWSER_USE_HUMAN_MOUSE_STEPS_MAX` (default: `10`)
- `BROWSER_USE_HUMAN_MOUSE_STEP_DELAY_MIN` (default: `0.004`)
- `BROWSER_USE_HUMAN_MOUSE_STEP_DELAY_MAX` (default: `0.018`)
- `BROWSER_USE_HUMAN_MOUSE_JITTER_PX` (default: `1.75`)
- `BROWSER_USE_HUMAN_MOUSE_START_OFFSET_PX` (default: `28.0`)

### Click timing

- `BROWSER_USE_HUMAN_CLICK_HOLD_MIN` (default: `0.045`)
- `BROWSER_USE_HUMAN_CLICK_HOLD_MAX` (default: `0.11`)

### Typing timing

- `BROWSER_USE_HUMAN_TYPE_DELAY_MIN` (default: `0.003`)
- `BROWSER_USE_HUMAN_TYPE_DELAY_MAX` (default: `0.022`)
- `BROWSER_USE_HUMAN_NEWLINE_DELAY_MIN` (default: `0.007`)
- `BROWSER_USE_HUMAN_NEWLINE_DELAY_MAX` (default: `0.028`)

## Example usage

```bash
export BROWSER_USE_HUMANIZE=true
export BROWSER_USE_HUMAN_MOUSE_STEPS_MIN=5
export BROWSER_USE_HUMAN_MOUSE_STEPS_MAX=14
export BROWSER_USE_HUMAN_MOUSE_STEP_DELAY_MIN=0.005
export BROWSER_USE_HUMAN_MOUSE_STEP_DELAY_MAX=0.020
export BROWSER_USE_HUMAN_CLICK_HOLD_MIN=0.050
export BROWSER_USE_HUMAN_CLICK_HOLD_MAX=0.130
export BROWSER_USE_HUMAN_TYPE_DELAY_MIN=0.004
export BROWSER_USE_HUMAN_TYPE_DELAY_MAX=0.030
export BROWSER_USE_HUMAN_NEWLINE_DELAY_MIN=0.010
export BROWSER_USE_HUMAN_NEWLINE_DELAY_MAX=0.040
```

Run:

```bash
uv run python main.py
```

## Notes

- This patch is applied inside `.venv`, so it is environment-local and may be overwritten if dependencies are reinstalled.
- If you need a permanent/portable patch, consider moving this to:
  - a forked package, or
  - a runtime monkeypatch module loaded by your app startup.


### Using Chrome Dev Protocol

To use the browser sample, you need to start Chrome with remote debugging enabled:

```bash
# Start Chrome with CDP debugging on port 9222
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```

### Running the Browser Sample

1. Make sure Chrome is running with CDP enabled (see above).
2. Run the sample script:

```bash
python main.py
```