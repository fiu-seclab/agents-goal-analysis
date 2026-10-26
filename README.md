# Actions Speak Louder: Behavioral Goal Recognition for Agentic Visitors in Web Applications

This repository provides the reference implementation, benchmark task specifications, and evaluation suite for **"Actions Speak Louder: Behavioral Goal Recognition for Agentic Visitors in Web Applications."**

As users increasingly delegate routine tasks to autonomous browser agents inside authenticated sessions, web applications need visibility into *what operational goal* an agent session is pursuing before an irreversible HTTP request commits. Our pipeline streams normalized in-page DOM interactions from a lightweight client collector to a dual-mode Goal Analyzer that continuously estimates the session's goal across five universal categories (`RETRIEVAL`, `MODIFICATION`, `CREATION`, `IDENTITY`, and `TRANSACTION`):

- **Site-Agnostic Analyzer (Day-0, Zero-Config):** Operates strictly on universal action-role tokens (`type`, `click`, `scroll`, `navigate`, `change`, `submit`, `select`, `reset` $\times$ standard HTML roles) and interaction timing dynamics, requiring zero site-specific selectors or URLs.
- **Schema-Adapted Analyzer (Calibrated):** Combines the site-agnostic behavioral prior with sanitized DOM control tokens (`id`, `name`, leaf XPath tags) and URL path segments learned from a small calibration sample ($\sim$27 traces per site) to resolve shared-form collisions and cross-agent shifts.

---

## Project Structure & Mapping to the Paper

```text
agents-goal-analysis/
├── tracker/          # In-Page Collector & telemetry ingestion service (Section 4.1)
├── tasks/            # 117 task templates, universal goal taxonomy, and verifiers (Section 5.1)
├── analyzer/         # Trace normalizer and Site-Agnostic / Schema-Adapted models (Section 4.2)
├── analysis/         # Evaluation summaries, online trigger rules, and figure scripts (Sections 6–7)
└── data/results/     # Precomputed 5-seed evaluation logs across 702 agent sessions
```

| Module | Paper Section | Description |
| :--- | :--- | :--- |
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

### 2. Reproduce Paper Tables & Online Trigger Analysis
Using the precomputed 5-seed evaluation logs in `data/results/`, you can directly reproduce the paper's empirical tables:

```bash
# Reproduce Table 2 (Main 5-Goal Results), Table 5 (Per-Architecture Breakdown), and Table 6 (Read/Write & Subsystems)
python3 analysis/summarize_results.py

# Reproduce Table 8 (Online Decision Rule tau_on(theta, m=2) and Pre-Commit k = kappa - 1)
python3 analysis/online_trigger_eval.py
```

### 3. Train & Evaluate Models Across Holdout Regimes
To run the full 5-seed training and prefix evaluation pipeline (`analyzer/causal.py`):

```bash
# Within-Agent Unseen Tasks & Seen Tasks (Held-Out Sessions)
python3 analyzer/causal.py --target goal --group task --agents browser-use --seeds 5
python3 analyzer/causal.py --target goal --group task --agents skyvern --seeds 5
python3 analyzer/causal.py --target goal --group trial --agents browser-use --seeds 5
python3 analyzer/causal.py --target goal --group trial --agents skyvern --seeds 5

# Cross-Agent Transfer (Browser Use <-> Skyvern)
python3 analyzer/causal.py --target goal --protocol cross --agents browser-use,skyvern --seeds 5

# Target-Site Calibration Curves (10% to 100% budget)
python3 analyzer/calibration_curve.py task
```

### 4. Deploy the In-Page Collector
To run the telemetry ingestion server and inject the collector into a target web application:

```bash
cd tracker && npm install && node index.js
```
Include `public/js/0laso3vm.min.js` and `public/js/track-init.js` in the target application's global layout template (or use `tracker/install_servicenow_tracker.py` for ServiceNow instances).
