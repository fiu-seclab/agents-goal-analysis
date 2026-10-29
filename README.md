# Actions Speak Louder: Behavioral Goal Recognition for Agentic Visitors in Web Applications

This repository provides the reference implementation, benchmark task specifications, and evaluation suite for **"Actions Speak Louder: Behavioral Goal Recognition for Agentic Visitors in Web Applications."**

As users increasingly delegate routine tasks to autonomous browser agents inside authenticated sessions, web applications need visibility into *what operational goal* an agent session is pursuing before an irreversible HTTP request commits. Our pipeline streams normalized in-page DOM interactions from a lightweight client collector to a dual-mode Goal Analyzer that continuously estimates the session's goal across five universal categories (`RETRIEVAL`, `MODIFICATION`, `CREATION`, `IDENTITY`, and `TRANSACTION`):

- **Site-Agnostic Analyzer (Day-0, Zero-Config):** Operates strictly on universal action-role tokens (`type`, `click`, `scroll`, `navigate`, `change`, `submit`, `select`, `reset` $\times$ standard HTML roles) and interaction timing dynamics, requiring zero site-specific selectors or URLs.
- **Schema-Adapted Analyzer (Calibrated):** Combines the site-agnostic behavioral prior with sanitized DOM control tokens (`id`, `name`, leaf XPath tags) and URL path segments learned from a small calibration sample ($\sim$27 traces per site) to resolve shared-form collisions and cross-agent shifts.

---

## Project Structure & Mapping to the Paper

```text
agents-goal-analysis/
├── crawlers/         # Browser Use and Skyvern drivers, plus the three site run scripts
├── tracker/          # In-Page Collector & telemetry ingestion service (Section 4.1)
├── tasks/            # 117 task templates, universal goal taxonomy, and verifiers (Section 5.1)
├── analyzer/         # Trace normalizer and Site-Agnostic / Schema-Adapted models (Section 4.2)
├── analysis/         # Evaluation summaries, online trigger rules, and figure scripts (Sections 6–7)
└── data/results/     # Precomputed 5-seed evaluation logs across 702 agent sessions
```

| Module | Paper Section | Description |
| :--- | :--- | :--- |
| **`crawlers/`** | Section 5.1 | Browser Use and Skyvern wrappers (`browser-use-app/`, `skyvern-app/`) and the Storefront, ServiceNow, and Gitea run scripts. Agent libraries are installed from their packages. |
| **`tracker/`** | Section 4.1 | Client-side JavaScript collector (`public/js/`) and Node.js/Socket.IO ingestion service that streams DOM interaction events without recording plaintext keystrokes or page text. |
| **`tasks/`** | Section 5.1, Table 1 | Parameterized task definitions (`taxonomy_v2.json`) and server-side verification scripts across **Storefront** (40 tasks), **ServiceNow** (36 tasks), and **Code Forge / Gitea** (41 tasks). |
| **`analyzer/`** | Section 4.1–4.2, Section 5.3 | Action-clock normalization (`tokens.py`), dual-mode prefix classifiers (`causal.py`: Random Forest, N-gram Forest, LSTM, Transformer, and Schema-Adapted ensemble), calibration learning curves (`calibration_curve.py`), and multi-granularity holdouts (`eval_hierarchy.py`). |
| **`analysis/`** | Section 6, Section 7 | Reproduces all paper tables (`summarize_results.py`), evaluates online confidence-threshold trigger rules $\tau_{\text{on}}(\theta, m)$ (`online_trigger_eval.py`), and generates Figures 1–5. |
| **`data/results/`** | Section 6, Appendix C–F | Precomputed 5-seed evaluation outputs across **Browser Use** and **Skyvern** (702 sessions) for immediate verification of all reported metrics. |

---

## Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Reproduce the paper tables and figures
`data/results/` holds the five-seed logs. The scripts report the five universal goals. Site-Agnostic is the best of forest, n-gram, LSTM, and transformer on each cell. Schema-Adapted is the behavioral prior plus sanitized DOM and URL tokens.

```bash
python3 analysis/summarize_results.py
python3 analysis/online_trigger_eval.py
python3 analysis/make_figures.py
python3 analysis/make_result_figures.py
```

`summarize_results.py` prints the main holdout table, the four Site-Agnostic architectures, and the auxiliary read/write and subsystem table. `online_trigger_eval.py` prints the online rule \(\tau_{\text{on}}(\theta, m{=}2)\). The figure scripts write `fig_overview.pdf`, `fig_goal.pdf`, `fig_confusion.pdf`, and `fig_calibration.pdf`.

### 3. Train the reported models
`analyzer/causal.py` defaults to the five goals and the unseen-task holdout. Trees and neural models can be launched separately. New runs are written under `data/results/`.

```bash
# Unseen tasks, then seen tasks (held-out sessions), per agent
python3 analyzer/causal.py --group task --agents browser-use --models forest,ngram,stage2 --seeds 5
python3 analyzer/causal.py --group task --agents browser-use --models lstm,transformer --seeds 5
python3 analyzer/causal.py --group task --agents skyvern --models forest,ngram,stage2 --seeds 5
python3 analyzer/causal.py --group task --agents skyvern --models lstm,transformer --seeds 5
python3 analyzer/causal.py --group trial --agents browser-use --models forest,ngram,stage2 --seeds 5
python3 analyzer/causal.py --group trial --agents skyvern --models forest,ngram,stage2 --seeds 5

# Cross-agent transfer
python3 analyzer/causal.py --protocol cross --agents browser-use,skyvern --models forest,ngram,stage2 --seeds 5

# Unseen-task calibration curve (10% to 100%)
python3 analyzer/calibration_curve.py task

# Appendix: read/write, unseen subsystem, UI subsystems
python3 analyzer/eval_hierarchy.py
```

### 4. Run an agent on one task

The paper agents are Browser Use and Skyvern. Setup, environment variables, and the three site runners are in [`crawlers/README.md`](crawlers/README.md).

```bash
cp crawlers/.env.example crawlers/.env
python3 crawlers/storefront/run_tasks.py --print-prompts --tasks A --agents browser-use
python3 crawlers/run_experiments.py --url "$TARGET_URL" --agents browser-use --dry-run
```

### 5. Deploy the In-Page Collector
To run the telemetry ingestion server and inject the collector into a target web application:

```bash
cd tracker && npm install && node index.js
```
Include `public/js/0laso3vm.min.js` and `public/js/track-init.js` in the target application's global layout template (or use `tracker/install_servicenow_tracker.py` for ServiceNow instances).
