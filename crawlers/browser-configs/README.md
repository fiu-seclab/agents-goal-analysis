# Browser configurations

Launchers used by `run_grid.py` so agents in `../llm-based-crawlers` attach to
a chosen Chrome instead of their bundled Chromium.

| config | what is launched | attach |
|---|---|---|
| `instrumented` | nothing; agent default | Playwright / bundled Chromium |
| `chrome_incognito` | `google-chrome --incognito` | CDP |
| `chrome_cold` | Chrome Guest, throwaway user-data-dir | CDP |
| `chrome_full` | Chrome with a persistent profile | CDP |

```bash
python chrome_launcher.py --config chrome_incognito --keep-open
# then, in another shell, from the repository root:
BROWSER_USE_CDP_URL=http://127.0.0.1:PORT \
  BROWSER_CONFIGS_DIR=crawlers/browser-configs \
  crawlers/browser-use-app/.venv/bin/python crawlers/browser-use-app/main.py --url "$TARGET_URL"
```

Profiles written by `--prepare-full-profile` land in `profiles/` and are
gitignored. Pass `--full-profile` or `RQ2_CHROME_FULL_PROFILE` to reuse one.
