# Skyvern wrapper

Experiment entry point around the [Skyvern](https://github.com/Skyvern-AI/skyvern) package. `main.py` accepts `--url`, writes artifacts, and can attach to Chrome over CDP. The upstream library is installed with `pip install skyvern`. It is not vendored in this repository.

## Setup

Python 3.11+. From the repository root:

```bash
python3 -m venv crawlers/skyvern-app/.venv
crawlers/skyvern-app/.venv/bin/pip install -e crawlers/skyvern-app
crawlers/skyvern-app/.venv/bin/playwright install chromium
```

Set `OPENAI_API_KEY` in the environment or in `crawlers/.env`. Copy `crawlers/skyvern-app/.env.example` to `.env` in this directory if you want Skyvern's own settings (`LLM_KEY`, `BROWSER_TYPE`).

Skyvern runs in-process. Docker and Postgres are not required for this wrapper.

## Usage

The site runners call this script. To launch it directly:

```bash
export OPENAI_API_KEY="..."
crawlers/skyvern-app/.venv/bin/python crawlers/skyvern-app/main.py --url "$TARGET_URL"
```

Or from the repository root:

```bash
python3 crawlers/run_experiments.py --url "$TARGET_URL" --agents skyvern
```

## Artifacts

When `EXPERIMENT_ARTIFACTS_DIR` is set by `run_experiments.py`, results go there. Otherwise they are written next to this app:

- `result/` — per-run JSON plus a URL-level JSONL
- `recordings/` — ffmpeg X11 capture
- `terminal_logs/` — full terminal capture via `script`

## Environment

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required |
| `LLM_KEY` | Skyvern model key (default `OPENAI_GPT4O`) |
| `SKYVERN_MODEL` | Model name passed to `Skyvern.local()` (default `gpt-4o`) |
| `SKYVERN_RUN_TIMEOUT_SECONDS` | Run timeout in seconds |
| `EXPERIMENT_ARTIFACTS_DIR` | Artifact root from the experiment runners |
| `BROWSER_USE_CDP_URL` | Attach to an existing Chrome instead of launching one |

Set `SKYVERN_TELEMETRY=false` to disable Skyvern usage telemetry (already in `.env.example`).
