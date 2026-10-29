# Crawlers

Drivers for the two agents in the paper, Browser Use and Skyvern, and the scripts that run one task at a time on each site. The agent libraries themselves are installed from their packages. This tree is the experiment code around them: how a task is turned into a prompt, how the browser is launched, and where the run log is written.

```text
crawlers/
├── run_experiments.py          # launch one agent against one URL
├── shared.py                   # URL tags (agent, task, model) and run-log helpers
├── .env.example
├── browser-use-app/            # Browser Use entry point (main.py, humanize.py)
├── skyvern-app/                # Skyvern entry point (main.py)
├── browser-configs/            # optional: attach the agent to a real Chrome over CDP
├── storefront/run_tasks.py     # tasks/storefront_tasks.json
├── servicenow/run_tasks.py     # tasks/servicenow_tasks.json
└── gitea/run_tasks.py          # tasks/gitea_tasks.json
```

Each site runner does the same four steps: prepare the task's starting records, start the agent at a URL tagged with `agent=` and `task=`, score the agent's report, then check the site's own state. Logs go to `experiment_runs/<site>_<timestamp>/` (gitignored).

## Install

Python 3.11 or newer. Copy `crawlers/.env.example` to `crawlers/.env` and set `OPENAI_API_KEY`.

```bash
python3 -m venv crawlers/browser-use-app/.venv
crawlers/browser-use-app/.venv/bin/pip install -e crawlers/browser-use-app
python3 -m venv crawlers/skyvern-app/.venv
crawlers/skyvern-app/.venv/bin/pip install -e crawlers/skyvern-app
crawlers/browser-use-app/.venv/bin/playwright install chromium
crawlers/skyvern-app/.venv/bin/playwright install chromium
```

`run_experiments.py` uses an app's `.venv` when that directory exists, and the current interpreter otherwise. Browser Use is the package `browser-use`. Skyvern is the package `skyvern`. A display is required (`DISPLAY`, default `:1`).

## One URL

From the repository root:

```bash
set -a && source crawlers/.env && set +a
python3 crawlers/run_experiments.py \
  --url "$TARGET_URL" \
  --agents browser-use \
  --trials 1 \
  --dry-run
```

Drop `--dry-run` to launch the browser. `--agents` accepts `browser-use` and `skyvern`.

## One task per site

Print the prompt without launching a browser:

```bash
python3 crawlers/storefront/run_tasks.py --print-prompts --tasks A --agents browser-use
python3 crawlers/servicenow/run_tasks.py --print-prompts --tasks SN01 --agents browser-use
```

Storefront and ServiceNow `--print-prompts` and `--dry-run` stay offline. Gitea `--print-prompts` and `--dry-run` create the task's setup records and then delete them, so `GITEA_URL` and `GITEA_TOKEN` must already be set and the server must be one you are willing to write to.

```bash
# Storefront. The site must expose POST /api/reset-db.
python3 crawlers/storefront/run_tasks.py \
  --agents browser-use --tasks A --trials 1 --backend primary --humanize off

# ServiceNow. Export SN_INSTANCE, SN_USER, SN_PASSWORD.
python3 crawlers/servicenow/run_tasks.py \
  --agents browser-use --tasks SN01 --trials 1

# Gitea. Export GITEA_URL, GITEA_TOKEN, GITEA_USER, GITEA_PASSWORD.
python3 crawlers/gitea/run_tasks.py \
  --agents skyvern --tasks GT01 --trials 1
```

`--backend primary` is Browser Use `o4-mini` and Skyvern `gpt-4o`. `--backend secondary` is Browser Use `gpt-4o` and Skyvern `gpt-4o-mini`. `--humanize on` enables the pointer and typing delays in `browser-use-app/humanize.py`. Add `--resume` to skip trials already recorded in `master_run_log.jsonl`.

The collector that records the visit is `tracker/`. It has to be loaded by the target site before a run, or the agent will finish with an empty trace.
