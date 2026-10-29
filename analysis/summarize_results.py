#!/usr/bin/env python3
"""Reprint the paper's five-goal tables from data/results.

Main table: Site-Agnostic (best of forest, n-gram, LSTM, transformer) versus
Schema-Adapted, for unseen tasks, seen tasks, and cross-agent transfer.
Appendix: the four Site-Agnostic architectures side by side, then the auxiliary
read/write, unseen-subsystem, and UI-subsystem results.

Cross-site (Unseen Application) numbers and the URL-only / DOM-only / schema-only
ablations are not in these logs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "data" / "results"
SITES = ["ridgeway", "servicenow", "gitea"]
SITE_LABEL = {"ridgeway": "Storefront", "servicenow": "ServiceNow", "gitea": "Code Forge"}
AGNOSTIC = ["forest", "ngram", "lstm", "transformer"]
ARCH_LABEL = {
    "forest": "Agnostic forest",
    "ngram": "Agnostic n-gram",
    "lstm": "Agnostic LSTM",
    "transformer": "Agnostic transformer",
    "stage2": "Schema-Adapted",
}
TREE_MODELS = {"forest", "ngram", "stage2"}


def load_json(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def metric_mean(runs: list[dict], key: str = "full", metric: str = "acc") -> float:
    return float(np.mean([r["scores"]["by_action"][key][metric] for r in runs]))


def _bundle(split: str, agent: str) -> dict:
    """split is task (unseen tasks) or trial (seen tasks). agent is bu or sk."""
    kind = {"bu": "bu", "sk": "sk"}[agent]
    trees = load_json(f"agent_{kind}_{split}_trees.json")["sites"]
    neural = load_json(f"agent_{kind}_{split}_neural.json")["sites"]
    return {"tree": trees, "neural": neural}


def runs_for(bundle: dict, site: str, model: str) -> list[dict]:
    src = bundle["tree"] if model in TREE_MODELS else bundle["neural"]
    return src[site]["models"][model]


def cell(bundle: dict, site: str, model: str) -> tuple[float, float]:
    runs = runs_for(bundle, site, model)
    return metric_mean(runs, "full", "acc"), metric_mean(runs, "full", "macro_f1")


def best_agnostic(bundles: list[dict], site: str) -> tuple[str, float, float]:
    scored = []
    for model in AGNOSTIC:
        accs, f1s = [], []
        for bundle in bundles:
            acc, f1 = cell(bundle, site, model)
            accs.append(acc)
            f1s.append(f1)
        scored.append((model, float(np.mean(accs)), float(np.mean(f1s))))
    return max(scored, key=lambda row: row[1])


def adapted(bundles: list[dict], site: str) -> tuple[float, float]:
    accs, f1s = [], []
    for bundle in bundles:
        acc, f1 = cell(bundle, site, "stage2")
        accs.append(acc)
        f1s.append(f1)
    return float(np.mean(accs)), float(np.mean(f1s))


def fmt(acc: float, f1: float) -> str:
    return f"{acc:.3f} ({f1:.3f})"


def print_main_table() -> None:
    bu_task, sk_task = _bundle("task", "bu"), _bundle("task", "sk")
    bu_trial, sk_trial = _bundle("trial", "bu"), _bundle("trial", "sk")
    cross_t = load_json("agent_cross_trees.json")["sites"]
    cross_n = load_json("agent_cross_neural.json")["sites"]

    print("Main table: full-session 5-goal accuracy (macro-F1)")
    print("Site-Agnostic is the best of the four architectures on that cell.")
    print("Schema-Adapted is the behavioral prior plus sanitized DOM and URL tokens.")
    print(f"{'Holdout':28} {'Setting':22} {'Site':12} {'Agnostic':16} {'Architecture':22} {'Adapted':16}")
    print("-" * 120)

    rows = [
        ("Unseen Tasks", "Browser Use", [bu_task]),
        ("Unseen Tasks", "Skyvern", [sk_task]),
        ("Seen Tasks", "Browser Use", [bu_trial]),
        ("Seen Tasks", "Skyvern", [sk_trial]),
    ]
    for holdout, setting, bundles in rows:
        for site in SITES:
            name, acc, f1 = best_agnostic(bundles, site)
            ad_acc, ad_f1 = adapted(bundles, site)
            print(
                f"{holdout:28} {setting:22} {SITE_LABEL[site]:12} {fmt(acc, f1):16} "
                f"{ARCH_LABEL[name]:22} {fmt(ad_acc, ad_f1)}"
            )

    directions = [
        ("Browser Use -> Skyvern", "browser-use_to_skyvern"),
        ("Skyvern -> Browser Use", "skyvern_to_browser-use"),
    ]
    for setting, key in directions:
        for site in SITES:
            scored = []
            for model in AGNOSTIC:
                src = cross_t if model in TREE_MODELS else cross_n
                runs = src[site][key]["models"][model]
                scored.append((model, metric_mean(runs), metric_mean(runs, metric="macro_f1")))
            name, acc, f1 = max(scored, key=lambda row: row[1])
            runs = cross_t[site][key]["models"]["stage2"]
            print(
                f"{'Cross-Agent':28} {setting:22} {SITE_LABEL[site]:12} {fmt(acc, f1):16} "
                f"{ARCH_LABEL[name]:22} {fmt(metric_mean(runs), metric_mean(runs, metric='macro_f1'))}"
            )
    print()
    print("Unseen Application (train on two sites, test on the third) is not in data/results.")


def print_architectures() -> None:
    print()
    print("Appendix: four Site-Agnostic architectures and Schema-Adapted")
    print("Accuracy (macro-F1), averaged across Browser Use and Skyvern.")
    header = f"{'Holdout':16} {'Site':12} " + " ".join(f"{ARCH_LABEL[m]:22}" for m in AGNOSTIC + ["stage2"])
    print(header)
    print("-" * len(header))
    for split, title in [("task", "Unseen Tasks"), ("trial", "Seen Tasks")]:
        bundles = [_bundle(split, "bu"), _bundle(split, "sk")]
        for site in SITES:
            cells = []
            for model in AGNOSTIC + ["stage2"]:
                accs, f1s = [], []
                for bundle in bundles:
                    acc, f1 = cell(bundle, site, model)
                    accs.append(acc)
                    f1s.append(f1)
                cells.append(fmt(float(np.mean(accs)), float(np.mean(f1s))))
            print(f"{title:16} {SITE_LABEL[site]:12} " + " ".join(f"{c:22}" for c in cells))


def print_auxiliary() -> None:
    hier = load_json("hierarchy_eval.json")["sites"]
    print()
    print("Appendix: read/write, universal goals under an unseen subsystem, and UI subsystems")
    print(f"{'Site':12} {'Label set':32} {'Agnostic @12':14} {'Agnostic full':18} {'Adapted @12':14} {'Adapted full'}")
    print("-" * 100)
    labels = [
        ("rw_hybrid", "Read/Write, unseen subsystem"),
        ("rw_task", "Read/Write, unseen tasks"),
        ("goal_hybrid", "5 goals, unseen subsystem"),
        ("group_task", "UI subsystems, unseen tasks"),
        ("group_trial", "UI subsystems, seen tasks"),
    ]
    for site in SITES:
        targets = hier[site]["targets"]
        for key, label in labels:
            block = targets[key]
            agn = block["ngram_runs"]
            adapted_runs = block["s2_runs"]
            agn12 = float(np.mean([r["by_action"]["12"]["acc"] for r in agn]))
            agn_full = float(np.mean([r["by_action"]["full"]["acc"] for r in agn]))
            agn_f1 = float(np.mean([r["by_action"]["full"]["macro_f1"] for r in agn]))
            ad12 = float(np.mean([r["by_action"]["12"]["acc"] for r in adapted_runs]))
            ad_full = float(np.mean([r["by_action"]["full"]["acc"] for r in adapted_runs]))
            ad_f1 = float(np.mean([r["by_action"]["full"]["macro_f1"] for r in adapted_runs]))
            print(
                f"{SITE_LABEL[site]:12} {label:32} {agn12:14.3f} "
                f"{fmt(agn_full, agn_f1):18} {ad12:14.3f} {fmt(ad_full, ad_f1)}"
            )


if __name__ == "__main__":
    print_main_table()
    print_architectures()
    print_auxiliary()
