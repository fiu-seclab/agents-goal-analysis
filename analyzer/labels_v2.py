"""Five universal goals from tasks/taxonomy_v2.json.

The class the paper trains is goal: read, modify, create, identity, or
transaction (retrieval, modification, creation, identity, transaction). A task
whose goal sequence has more than one goal is a composite and is not trained
on. Labels come from the task definition, never from the trace.

group=task holds out tasks. group=trial is applied later as a session split.
group=hybrid holds out a task family when that goal has two or more families
on the site, which is the unseen-subsystem auxiliary evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

from effects_exp import load_sites  # noqa: E402

TAXONOMY = Path(__file__).resolve().parents[1] / "tasks" / "taxonomy_v2.json"
TARGETS = ("tier", "goal")
GROUPS = ("family", "task", "hybrid")

GOAL5_FALLBACK = {
    "discovery": "read",
    "retrieval": "read",
    "enumeration": "read",
    "curation": "modify",
    "availability": "modify",
    "service": "create",
    "identity": "identity",
    "transaction": "transaction",
}


def load_taxonomy() -> dict:
    return json.loads(TAXONOMY.read_text(encoding="utf-8"))


def _family(site: str, entry: dict) -> str:
    family = entry.get("family_v2") or entry["family"]
    # A find-then-edit task is a single goal now; it keeps its own family.
    if entry.get("goal") and family.startswith("composite:"):
        return f"{site}:find_then_edit"
    return family


def load_sites_v2(target: str = "tier", group: str = "family", agents: set[str] | None = None) -> dict[str, dict]:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}")
    tasks = load_taxonomy()["tasks"]
    out: dict[str, dict] = {}
    for site, spec in load_sites(agents).items():
        singles, composites = [], []
        for session in spec["singles"] + spec["composites"]:
            task_id = session["label"]["task"]
            entry = tasks.get(f"{site}:{task_id}")
            if entry is None:
                print(f"skip {site}:{task_id}: not in taxonomy_v2", flush=True)
                continue
            label = session["label"]
            g8 = entry.get("goal")
            g5 = entry.get("goal5") or (GOAL5_FALLBACK[g8] if g8 else None)
            seq5 = entry.get("goal5_sequence") or ([g5] if g5 else [GOAL5_FALLBACK[g] for g in (entry.get("goal_sequence") or [])])
            label["goal8"] = g8
            label["goal"] = g5
            label["goal_sequence"] = list(seq5)
            label["tier"] = entry["tier"]
            label["family"] = _family(site, entry)
            label["label_confidence"] = entry["confidence"]
            label["label_source"] = "taxonomy_v2"
            if len(seq5) == 1 and g5 is not None:
                label["target"] = label[target]
                label["group"] = label["family"] if group == "family" else task_id
                label["group_kind"] = "family" if group == "family" else "task"
                singles.append(session)
            else:
                label["target"] = None
                label["group"] = label["family"] if group == "family" else task_id
                label["group_kind"] = "family" if group == "family" else "task"
                composites.append(session)
        if group == "hybrid":
            _apply_hybrid(singles, composites)
        out[site] = {**spec, "singles": singles, "composites": composites}
    return out


def _apply_hybrid(singles: list[dict], composites: list[dict]) -> None:
    """Hold out a family where its goal has two or more families on the site.

    A goal with a single family on a site would lose its only example when that
    family is held out, so those sessions fall back to the task. Composites
    always stay with their family (they are never trained on).
    """
    families: dict[str, set[str]] = {}
    for session in singles:
        label = session["label"]
        families.setdefault(label["goal"], set()).add(label["family"])
    for session in singles:
        label = session["label"]
        if len(families[label["goal"]]) >= 2:
            label["group"], label["group_kind"] = label["family"], "family"
        else:
            label["group"], label["group_kind"] = label["task"], "task"
    for session in composites:
        label = session["label"]
        label["group"], label["group_kind"] = label["family"], "family"


def describe_v2(sessions: list[dict]) -> dict:
    from collections import Counter

    actions = sorted(len(row.get("_spans") or []) for row in sessions)
    return {
        "n_sessions": len(sessions),
        "n_tasks": len({row["label"]["task"] for row in sessions}),
        "n_groups": len({row["label"]["group"] for row in sessions}),
        "targets": dict(Counter(row["label"]["target"] for row in sessions)),
        "tiers": dict(Counter(row["label"]["tier"] for row in sessions)),
        "goals": dict(Counter(row["label"]["goal"] for row in sessions)),
        "actions_median": actions[len(actions) // 2] if actions else 0,
        "actions_min": actions[0] if actions else 0,
        "actions_max": actions[-1] if actions else 0,
    }
