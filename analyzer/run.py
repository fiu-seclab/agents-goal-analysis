#!/usr/bin/env python3
"""Load the Storefront, ServiceNow, and Gitea sessions used by the analyzer.

Browser Use storefront sessions come from the processed session logs. ServiceNow,
Gitea, and Skyvern storefront sessions come from the collector JSONL traces.
Feature vectors used by the older category model stay here because the session
loader builds them. That category model is not part of the reported evaluation.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "pipelines"
sys.path.insert(0, str(PIPE))
sys.path.insert(0, str(ROOT / "classifier"))

from augment.augment import (  # noqa: E402
    _jitter,
    _slim,
    _stride_mousemove,
    _time_warp,
)
from build_dataset import feature_event, parse_ms, session_features  # noqa: E402
from common import (  # noqa: E402
    category_of,
    leave_one_out,
    load_jsonl,
    score,
)

# (traces dir, agent, run name). Browser-use stays the default corpus. Skyvern is
# the 2026-10-19 collection, split from hotel Mongo by job time window.
SN_TRACE_DIRS = [
    (ROOT / "experiment_runs" / "sn_bu_once" / "traces", "browser-use", "sn_bu_once"),
    (ROOT / "experiment_runs" / "sn_bu_o4mini_2" / "traces", "browser-use", "sn_bu_o4mini_2"),
    (ROOT / "experiment_runs" / "skyvern_20261019" / "servicenow" / "traces", "skyvern", "skyvern_20261019"),
]
SN_TASKS = ROOT / "servicenow" / "tasks.json"
GITEA_TRACE_DIRS = [
    (ROOT / "experiment_runs" / "gitea_bu_1" / "traces", "browser-use", "gitea_bu_1"),
    (ROOT / "experiment_runs" / "raw2_20261013" / "gitea" / "traces", "browser-use", "raw2_20261013"),
    (ROOT / "experiment_runs" / "skyvern_20261019" / "gitea" / "traces", "skyvern", "skyvern_20261019"),
]
GITEA_TASKS = ROOT / "gitea" / "tasks.json"
RIDGEWAY_TRACE_DIRS = [
    (ROOT / "experiment_runs" / "skyvern_20261019" / "ridgeway" / "traces", "skyvern", "skyvern_20261019"),
]
RIDGEWAY_TASKS = ROOT / "tasks" / "tasks.json"
DEFAULT_AGENTS = frozenset({"browser-use"})
RIDGEWAY_SESSIONS = PIPE / "data" / "pool30" / "sessions.jsonl"
RIDGEWAY_SESSION_FILES = [
    RIDGEWAY_SESSIONS,
    PIPE / "data" / "raw2" / "sessions.jsonl",
]
OUT = PIPE / "data" / "sites"
SYS_ID = re.compile(r"[0-9a-f]{32}")
MIN_EVENTS = 10
POINTER = {"mousemove", "mouseover", "mouseout", "pointermove"}
KEYS = {"keydown", "keypress", "keyup", "input", "change"}
TYPE_KEYS = {"keydown", "keypress", "keyup", "input"}
CLICKS = {"click", "mousedown", "mouseup"}
NAV = {"load", "pageshow", "beforeunload", "pagehide", "unload"}
FOCUS = {"focus", "blur"}
# Recurring controls only. A forest on ~40 trials overfits one-off element ids.
ID_MIN_SESSIONS = 4
ID_CAP = 25
NAME_MIN_SESSIONS = 3
NAME_CAP = 30
STRUCT_FEATURES = [
    "log_n_loads",
    "log_n_changes",
    "log_n_submits",
    "log_n_focus",
    "log_mean_typing_run",
    "log_n_typing_runs",
    "switch_rate",
    "key_rate_first_third",
    "key_rate_last_third",
    "key_rate_delta",
    "click_rate_last_third",
    "change_rate_last_third",
    "click_y_std",
    "log_unique_field_names",
    "p_change_after_key",
    "p_click_after_key",
    "p_key_after_nav",
    "p_submit_after_change",
]


def _pack(parent: dict, method: str, events: list[dict]) -> dict | None:
    if len(events) < MIN_EVENTS:
        return None
    return {
        "session_id": f"{parent['session_id']}::{method}",
        "parent_id": parent["session_id"],
        "augment": method,
        "label": dict(parent["label"]),
        "features": {
            "n_events": len(events),
            "duration_ms": events[-1]["t_rel_ms"] if events else 0,
            "events": events,
        },
    }


def _drop_pointer(events: list[dict], session_id: str, keep: float) -> list[dict]:
    rng = random.Random(f"drop:{session_id}")
    kept = []
    t0 = None
    for event in events:
        if event.get("event_name") in POINTER and rng.random() > keep:
            continue
        row = _slim(event)
        if t0 is None:
            t0 = int(row.get("t_rel_ms") or 0)
        row["t_rel_ms"] = max(0, int(row.get("t_rel_ms") or 0) - t0)
        kept.append(row)
    return kept


def copies_for(session: dict, extra: bool) -> list[dict]:
    events = [_slim(event) for event in session["features"]["events"]]
    methods = [
        ("time_warp_0.75", _time_warp(events, 0.75)),
        ("time_warp_1.35", _time_warp(events, 1.35)),
        ("mousemove_stride_3", _stride_mousemove(events, 3)),
        ("mousemove_stride_5", _stride_mousemove(events, 5)),
        ("coord_jitter", _jitter(events, session["session_id"])),
    ]
    if extra:
        methods.extend([
            ("time_warp_0.5", _time_warp(events, 0.5)),
            ("time_warp_1.6", _time_warp(events, 1.6)),
            ("drop_pointer_0.5", _drop_pointer(events, session["session_id"], 0.5)),
        ])
    return [row for name, ev in methods if (row := _pack(session, name, ev))]


def _stable_target(event: dict) -> dict:
    target = event.get("target_id") or ""
    if SYS_ID.search(target):
        event["target_id"] = ""
    return event


# These five pool-30 ids were removed from tasks.json when the suite was rebuilt.
# Buckets follow tasks.json category_rule and the asks in tasks/CANDIDATES.md.
RIDGEWAY_REMOVED = {
    "C006": "Search",       # find SKU RW-K-001 and add it
    "C042": "Modal",        # quick-view the higher-rated of two named products
    "C051": "Conditional",  # price threshold picks a substitute
    "C061": "State",        # named product, set color and quantity
    "C098": "Conditional",  # stock decides Notify me, then wishlist
}


def _allowed(agents: set[str] | None) -> set[str]:
    return set(DEFAULT_AGENTS if agents is None else agents)


def load_ridgeway(agents: set[str] | None = None) -> list[dict]:
    allowed = _allowed(agents)
    rows = []
    seen: set[str] = set()
    if "browser-use" in allowed:
        for path in RIDGEWAY_SESSION_FILES:
            if not path.exists():
                continue
            for session in load_jsonl(path):
                if int(session["features"]["n_events"]) < MIN_EVENTS:
                    continue
                if session["session_id"] in seen:
                    continue
                seen.add(session["session_id"])
                task_id = session["label"]["task"]
                session["label"]["category"] = RIDGEWAY_REMOVED.get(task_id) or category_of(task_id)
                session["label"]["site"] = "ridgeway"
                session["label"]["agent"] = session["label"].get("agent") or "browser-use"
                rows.append(session)
    if "skyvern" in allowed:
        spec = json.loads(RIDGEWAY_TASKS.read_text(encoding="utf-8"))["tasks"]
        rows.extend(_load_trace_sessions(
            RIDGEWAY_TRACE_DIRS, spec, "ridgeway", "*.jsonl", seen, primary_user=False,
        ))
    return rows


def _load_trace_sessions(
    dirs: list[tuple[Path, str, str]],
    spec: dict,
    site: str,
    pattern: str,
    seen: set[str],
    primary_user: bool,
) -> list[dict]:
    sessions = []
    for traces_dir, agent, run_name in dirs:
        if not traces_dir.exists():
            continue
        for path in sorted(traces_dir.glob(pattern)):
            if path.name.startswith("_"):
                continue
            raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not raw:
                continue
            if primary_user:
                users = Counter(row.get("userId") or "" for row in raw)
                primary = users.most_common(1)[0][0]
                raw = [row for row in raw if (row.get("userId") or "") == primary]
                if not raw:
                    continue
            raw.sort(key=lambda row: (parse_ms(row.get("timestamp")) or 0, row.get("_id") or ""))
            t0 = parse_ms(raw[0].get("timestamp")) or 0
            events = [_stable_target(feature_event(row, ["agent", "task", "model", "hum"], t0)) for row in raw]
            task_id = re.sub(r"_t\d+$", "", path.stem)
            task = spec.get(task_id)
            if task is None:
                continue
            sid = f"{run_name}__{path.stem}"
            if sid in seen:
                continue
            seen.add(sid)
            category = "transition" if task.get("objective_sequence") else task["objective"]
            sessions.append({
                "session_id": sid,
                "label": {
                    "task": task_id,
                    "category": category,
                    "objective": task["objective"],
                    "site": site,
                    "agent": agent,
                    "model": None,
                    "trial": int(m.group(1)) if (m := re.search(r"_t(\d+)$", path.stem)) else 1,
                    "run": run_name,
                },
                "t0_ms": t0,
                "features": session_features(events),
            })
    return sessions


def load_servicenow(agents: set[str] | None = None) -> list[dict]:
    allowed = _allowed(agents)
    spec = json.loads(SN_TASKS.read_text(encoding="utf-8"))["tasks"]
    sessions = []
    seen: set[str] = set()
    dirs = [row for row in SN_TRACE_DIRS if row[1] in allowed]
    for traces_dir, agent, run_name in dirs:
        if not traces_dir.exists():
            continue
        for path in sorted(traces_dir.glob("SN*.jsonl")):
            raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not raw:
                continue
            raw.sort(key=lambda row: (parse_ms(row.get("timestamp")) or 0, row.get("_id") or ""))
            t0 = parse_ms(raw[0].get("timestamp")) or 0
            events = [_stable_target(feature_event(row, ["agent", "task", "model", "hum"], t0)) for row in raw]
            stem = path.stem
            task_id = re.sub(r"_t\d+$", "", stem)
            task = spec.get(task_id)
            if task is None:
                continue
            category = "transition" if task.get("objective_sequence") else task["objective"]
            sid = f"{run_name}__{stem}"
            if sid in seen:
                continue
            seen.add(sid)
            sessions.append({
                "session_id": sid,
                "label": {
                    "task": task_id,
                    "category": category,
                    "objective": task["objective"],
                    "site": "servicenow",
                    "agent": agent,
                    "trial": int(m.group(1)) if (m := re.search(r"_t(\d+)$", stem)) else 1,
                    "run": run_name,
                },
                "t0_ms": t0,
                "features": session_features(events),
            })
    return sessions


def load_gitea(agents: set[str] | None = None) -> list[dict]:
    allowed = _allowed(agents)
    spec = json.loads(GITEA_TASKS.read_text(encoding="utf-8"))["tasks"]
    sessions = []
    seen: set[str] = set()
    dirs = [row for row in GITEA_TRACE_DIRS if row[1] in allowed]
    for traces_dir, agent, run_name in dirs:
        if not traces_dir.exists():
            continue
        for path in sorted(traces_dir.glob("GT*.jsonl")):
            raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not raw:
                continue
            # A second user id emits only trusted postMessage heartbeats on "/" for the
            # whole run. The trial is the user id with the interaction events.
            users = Counter(row.get("userId") or "" for row in raw)
            primary = users.most_common(1)[0][0]
            raw = [row for row in raw if (row.get("userId") or "") == primary]
            raw.sort(key=lambda row: (parse_ms(row.get("timestamp")) or 0, row.get("_id") or ""))
            t0 = parse_ms(raw[0].get("timestamp")) or 0
            events = [_stable_target(feature_event(row, ["agent", "task", "model", "hum"], t0)) for row in raw]
            task_id = re.sub(r"_t\d+$", "", path.stem)
            task = spec.get(task_id)
            if task is None:
                continue
            sid = f"{run_name}__{path.stem}"
            if sid in seen:
                continue
            seen.add(sid)
            category = "transition" if task.get("objective_sequence") else task["objective"]
            sessions.append({
                "session_id": sid,
                "label": {
                    "task": task_id,
                    "category": category,
                    "objective": task["objective"],
                    "site": "gitea",
                    "agent": agent,
                    "trial": int(m.group(1)) if (m := re.search(r"_t(\d+)$", path.stem)) else 1,
                    "run": run_name,
                },
                "t0_ms": t0,
                "features": session_features(events),
            })
    return sessions


def _events(session: dict) -> list[dict]:
    return list((session.get("features") or {}).get("events") or [])


def _gap_stats(events: list[dict]) -> tuple[float, float]:
    gaps = []
    prev = None
    for event in events:
        t = event.get("t_rel_ms")
        if prev is not None and t is not None:
            gaps.append(max(0, int(t) - int(prev)))
        if t is not None:
            prev = int(t)
    if not gaps:
        return 0.0, 0.0
    ordered = np.sort(np.asarray(gaps, dtype=float))
    return float(np.median(ordered)), float(ordered[int(0.9 * (len(ordered) - 1))])


def _usable_field(name: str) -> bool:
    if not name or name == "not_important" or len(name) > 80:
        return False
    return SYS_ID.search(name) is None


def _coarse(event_name: str) -> str:
    if event_name in TYPE_KEYS:
        return "key"
    if event_name in CLICKS:
        return "click"
    if event_name in NAV:
        return "nav"
    if event_name == "change":
        return "change"
    if event_name == "submit":
        return "submit"
    if event_name in FOCUS:
        return "focus"
    if event_name in POINTER:
        return "move"
    return "other"


def _cond(prev: Counter, pair: Counter, left: str, right: str) -> float:
    denom = prev.get(left, 0)
    if not denom:
        return 0.0
    return pair.get((left, right), 0) / denom


def _structure(events: list[dict]) -> list[float]:
    """Shape of a long stream: bursts, late actions, field variety, action order."""
    n = max(1, len(events))
    names = [(event.get("event_name") or "") for event in events]
    coarse = [_coarse(name) for name in names]
    third = max(1, n // 3)
    first, last = events[:third], events[-third:]

    def rate(rows: list[dict], wanted: set[str]) -> float:
        return sum(1 for event in rows if (event.get("event_name") or "") in wanted) / max(1, len(rows))

    runs: list[int] = []
    run = 0
    for label in coarse:
        if label == "key":
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    prev: Counter = Counter()
    pair: Counter = Counter()
    for left, right in zip(coarse, coarse[1:]):
        prev[left] += 1
        pair[(left, right)] += 1
    switches = sum(1 for left, right in zip(coarse, coarse[1:]) if left != right)
    ys = []
    fields: set[str] = set()
    for event in events:
        field = event.get("target_name") or ""
        if _usable_field(field):
            fields.add(field)
        if event.get("event_name") == "click" and event.get("y") is not None and event.get("win_h"):
            ys.append(float(event["y"]) / max(1.0, float(event["win_h"])))
    key_first = rate(first, TYPE_KEYS)
    key_last = rate(last, TYPE_KEYS)
    return [
        float(np.log1p(sum(1 for name in names if name == "load"))),
        float(np.log1p(sum(1 for name in names if name == "change"))),
        float(np.log1p(sum(1 for name in names if name == "submit"))),
        float(np.log1p(sum(1 for name in names if name == "focus"))),
        float(np.log1p(float(np.mean(runs)) if runs else 0.0)),
        float(np.log1p(len(runs))),
        switches / max(1, n - 1),
        key_first,
        key_last,
        key_last - key_first,
        rate(last, CLICKS),
        rate(last, {"change", "submit"}),
        float(np.std(ys)) if len(ys) > 1 else 0.0,
        float(np.log1p(len(fields))),
        _cond(prev, pair, "key", "change"),
        _cond(prev, pair, "key", "click"),
        _cond(prev, pair, "nav", "key"),
        _cond(prev, pair, "change", "submit"),
    ]


def forest_vocab(sessions: list[dict], min_df: int = 2) -> dict:
    name_df: Counter = Counter()
    id_df: Counter = Counter()
    tag_df: Counter = Counter()
    field_df: Counter = Counter()
    type_df: Counter = Counter()
    for session in sessions:
        names: set[str] = set()
        ids: set[str] = set()
        tags: set[str] = set()
        fields: set[str] = set()
        types: set[str] = set()
        for event in _events(session):
            name = event.get("event_name") or ""
            if name:
                names.add(name)
            target = event.get("target_id") or ""
            if target and not SYS_ID.search(target):
                ids.add(target)
            tag = (event.get("target_tag") or "").lower()
            if tag:
                tags.add(tag)
            field = event.get("target_name") or ""
            if _usable_field(field):
                fields.add(field)
            kind = (event.get("target_type") or "").lower()
            if kind and not SYS_ID.search(kind):
                types.add(kind)
        name_df.update(names)
        id_df.update(ids)
        tag_df.update(tags)
        field_df.update(fields)
        type_df.update(types)
    kept_ids = [name for name, n in id_df.most_common() if n >= ID_MIN_SESSIONS][:ID_CAP]
    kept_fields = [name for name, n in field_df.most_common() if n >= NAME_MIN_SESSIONS][:NAME_CAP]
    return {
        "event_names": sorted(name for name, n in name_df.items() if n >= min_df),
        "target_ids": sorted(kept_ids),
        "tags": sorted(name for name, n in tag_df.items() if n >= min_df),
        "fields": sorted(kept_fields),
        "types": sorted(name for name, n in type_df.items() if n >= min_df),
    }


def forest_names(vocab: dict) -> list[str]:
    return (
        [f"rate:{name}" for name in vocab["event_names"]]
        + [f"tag:{name}" for name in vocab["tags"]]
        + [f"idrate:{name}" for name in vocab["target_ids"]]
        + [f"field:{name}" for name in vocab["fields"]]
        + [f"type:{name}" for name in vocab["types"]]
        + [
            "log_n_events", "log_duration_ms", "key_rate", "click_rate", "move_rate",
            "unique_target_rate", "log_median_gap_ms", "log_p90_gap_ms",
            "key_rate_first_half", "click_rate_second_half",
        ]
        + STRUCT_FEATURES
    )


def forest_vector(session: dict, vocab: dict) -> np.ndarray:
    events = _events(session)
    n = max(1, len(events))
    names = Counter((event.get("event_name") or "") for event in events)
    ids = Counter((event.get("target_id") or "") for event in events if event.get("target_id"))
    tags = Counter((event.get("target_tag") or "").lower() for event in events if event.get("target_tag"))
    fields = Counter(
        (event.get("target_name") or "")
        for event in events
        if _usable_field(event.get("target_name") or "")
    )
    types = Counter(
        (event.get("target_type") or "").lower()
        for event in events
        if event.get("target_type")
    )
    mid = max(1, n // 2)
    first, second = events[:mid], events[mid:] or events

    def rate(rows: list[dict], wanted: set[str]) -> float:
        return sum(1 for event in rows if (event.get("event_name") or "") in wanted) / max(1, len(rows))

    median_gap, p90_gap = _gap_stats(events)
    duration = float((session.get("features") or {}).get("duration_ms") or 0)
    values = [names.get(name, 0) / n for name in vocab["event_names"]]
    values += [tags.get(name, 0) / n for name in vocab["tags"]]
    values += [ids.get(name, 0) / n for name in vocab["target_ids"]]
    values += [fields.get(name, 0) / n for name in vocab["fields"]]
    values += [types.get(name, 0) / n for name in vocab["types"]]
    values += [
        float(np.log1p(n)),
        float(np.log1p(duration)),
        rate(events, KEYS),
        rate(events, CLICKS),
        rate(events, POINTER),
        len(ids) / n,
        float(np.log1p(median_gap)),
        float(np.log1p(p90_gap)),
        rate(first, KEYS),
        rate(second, CLICKS),
    ]
    shape = _structure(events)
    if len(shape) != len(STRUCT_FEATURES):
        raise RuntimeError(f"structure features {len(shape)} != {len(STRUCT_FEATURES)}")
    values += shape
    return np.asarray(values, dtype=float)


def make_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=1,
    )


def classify(site: str, real: list[dict], extra_aug: bool, min_df: int) -> dict:
    augmented: dict[str, list[dict]] = defaultdict(list)
    n_copies = 0
    for session in real:
        made = copies_for(session, extra_aug)
        # Pointer jitter does not change these features, so those copies are duplicates.
        made = [row for row in made if row.get("augment") != "coord_jitter"]
        augmented[session["session_id"]].extend(made)
        n_copies += len(made)
    preds = leave_one_out(
        real, augmented, "category", min_df=min_df,
        model_factory=make_forest,
        fit_vocab_fn=forest_vocab,
        vectorize_fn=forest_vector,
        feature_names_fn=forest_names,
    )
    result = score(preds)
    counts = Counter(row["label"]["category"] for row in real)
    result.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "site": site,
        "target": "task category",
        "model": "RandomForestClassifier n_estimators=400 max_depth=8 min_samples_leaf=2",
        "features": (
            "Event-name rates, HTML tag rates, input-type rates, "
            "rates of element ids seen in at least 4 other real trials (at most 25), "
            "rates of form field names seen in at least 3 other real trials (at most 30), "
            "log event count, log duration, key/click/move rates, unique-target rate, "
            "log median and 90th-percentile gap, key rate in the first half, click rate "
            "in the second half. Full-trace shape: log counts of loads, changes, submits, "
            "and focus events; typing-run length and count; action-switch rate; key and "
            "click rates in the first and last third; click vertical spread; unique field "
            "count; a few action-order probabilities. No URL, path, agent, or task id."
        ),
        "protocol": (
            "Leave-one-real-trial-out. Augmentations of the held-out trial are excluded. "
            "The forest vocabulary is fit on the other real trials only. "
            "Copies are distorted existing trials, not new ones. Pointer-jitter copies are omitted."
        ),
        "n_real": len(real),
        "n_augmented_copies": n_copies,
        "extra_augmentation": extra_aug,
        "min_df": min_df,
        "n_classes": len(counts),
        "sessions_by_category": dict(sorted(counts.items())),
    })
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sites = {
        "ridgeway": (load_ridgeway(), False, 1),
        "servicenow": (load_servicenow(), True, 2),
        "gitea": (load_gitea(), True, 2),
    }
    summary = []
    for site, (real, extra, min_df) in sites.items():
        print(f"\n=== {site}  real={len(real)} ===", flush=True)
        result = classify(site, real, extra, min_df)
        dest = OUT / site
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if site in ("servicenow", "gitea"):
            with (dest / "sessions.jsonl").open("w", encoding="utf-8") as handle:
                for row in real:
                    handle.write(json.dumps(row) + "\n")
        acc = result["accuracy"]
        f1 = result["macro_f1"]
        print(f"{site} accuracy {acc:.3f}  macro_f1 {f1:.3f}  n={result['n']}  classes={result['n_classes']}")
        for row in result["per_label"]:
            print(f"  {row['label']:16} {row['correct']}/{row['support']}")
        summary.append({
            "site": site,
            "n_real": result["n_real"],
            "n_augmented_copies": result["n_augmented_copies"],
            "accuracy": acc,
            "macro_f1": f1,
            "sessions_by_category": result["sessions_by_category"],
        })
    (OUT / "summary.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sites": summary,
        "note": (
            "Scored on real trials only. ServiceNow and Gitea copies are distorted versions "
            "of the single trial per task, added because those sites have no repeated trials yet."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
