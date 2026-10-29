#!/usr/bin/env python3
"""Action clock and session loader for the five-goal analyzer.

action_spans and kind_of collapse the raw event stream to one token per action.
load_sites reads the Storefront, ServiceNow, and Gitea sessions the paper trains
on. The older effect-label experiment is not part of the reported evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]

from run import (  # noqa: E402
    copies_for,
    forest_names,
    forest_vector,
    forest_vocab,
    load_gitea,
    load_ridgeway,
    load_servicenow,
)

EFFECT = {
    "discovery": "read",
    "retrieval": "read",
    "enumeration": "read",
    "curation": "edit",
    "availability": "edit",
    "service": "create",
    "identity": "account",
    "transaction": "commit",
}
SELECT_RULES = [
    "full",
    "offset_5",
    "offset_10",
    "offset_25",
    "trail_16",
    "trail_32",
    "trail_64",
    "anchor_click",
    "anchor_nav",
]
PREFIX_RULES = ["prefix_8", "prefix_16", "prefix_32", "prefix_64"]
MIL_RULES = list(SELECT_RULES)
POINTER = {"mousemove", "mouseover", "mouseout", "pointermove"}
FOCUS = {"focus", "blur", "focusin", "focusout"}
TYPE_KEYS = {"keydown", "keypress", "keyup", "input"}
CLICKISH = {"click", "mousedown", "mouseup", "auxclick", "dblclick", "contextmenu"}
SCROLL = {"scroll", "wheel"}
NAV = {"load", "pageshow"}
BURST = {"type", "click", "scroll"}
TOKEN_SKIP = re.compile(r"^[0-9a-f-]{8,}$", re.I)
SYS_ID = re.compile(r"[0-9a-f]{32}")


def make_forest(n_jobs: int, trees: int = 200) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=0,
        n_jobs=n_jobs,
    )


def kind_of(name: str) -> str | None:
    if name in TYPE_KEYS:
        return "type"
    if name in CLICKISH:
        return "click"
    if name in SCROLL:
        return "scroll"
    if name in NAV:
        return "nav"
    if name == "change":
        return "change"
    if name == "submit":
        return "submit"
    if name in POINTER or name in FOCUS or not name:
        return None
    return "other"


def action_spans(events: list[dict]) -> list[tuple[int, int]]:
    """Collapse pointer motion, key runs, click chords, and scroll bursts.

    Motion and focus before an action belong to that action. Motion after it
    belongs to the next one.
    """
    spans: list[tuple[int, int]] = []
    n = len(events)
    i = 0
    lead = 0
    while i < n:
        kind = kind_of(events[i].get("event_name") or "")
        if kind is None:
            i += 1
            continue
        if kind in BURST:
            j = i + 1
            while j < n:
                nxt = kind_of(events[j].get("event_name") or "")
                if nxt is None or nxt == kind:
                    j += 1
                    continue
                break
            end = j - 1
            while end > i and kind_of(events[end].get("event_name") or "") is None:
                end -= 1
        else:
            end = i
        spans.append((lead, end))
        lead = end + 1
        i = end + 1
    return spans


def _anchor(spans: list[tuple[int, int]], events: list[dict], which: str) -> int:
    navs = 0
    for index, (_start, end) in enumerate(spans):
        kind = kind_of(events[end].get("event_name") or "")
        if which == "click" and kind == "click":
            return index
        if which == "nav" and kind == "nav":
            navs += 1
            if navs == 2:
                return index
    return 0


def slice_window(session: dict, rule: str, min_events: int = 10) -> tuple[dict | None, bool]:
    events = list((session.get("features") or {}).get("events") or [])
    spans = session.get("_spans")
    if spans is None:
        spans = action_spans(events)
        session["_spans"] = spans
    n = len(spans)
    if n == 0:
        return None, False
    reached = True
    if rule == "full":
        start, stop = 0, n - 1
    elif rule.startswith("offset_"):
        k = int(rule.split("_")[1])
        reached = n > k
        start, stop = min(k, n - 1), n - 1
    elif rule.startswith("trail_"):
        k = int(rule.split("_")[1])
        reached = n >= k
        start, stop = max(0, n - k), n - 1
    elif rule.startswith("prefix_"):
        k = int(rule.split("_")[1])
        reached = n >= k
        start, stop = 0, min(k, n) - 1
    elif rule == "anchor_click":
        start = _anchor(spans, events, "click")
        stop = n - 1
        reached = start > 0
    elif rule == "anchor_nav":
        start = _anchor(spans, events, "nav")
        stop = n - 1
        reached = start > 0
    else:
        raise KeyError(rule)
    chosen = events[spans[start][0] : spans[stop][1] + 1]
    if len(chosen) < min_events:
        return None, reached
    duration = int(chosen[-1].get("t_rel_ms") or 0) - int(chosen[0].get("t_rel_ms") or 0)
    view = {
        "session_id": session["session_id"],
        "label": session["label"],
        "features": {
            "n_events": len(chosen),
            "duration_ms": max(0, duration),
            "events": chosen,
        },
        "_n_actions": stop - start + 1,
    }
    return view, reached


def _guard(vocab: dict) -> None:
    for name in forest_names(vocab):
        low = name.lower()
        if low.startswith(("tag:", "idrate:", "type:", "field:", "rate:")):
            continue
        if any(tok in low for tok in ("url", "path", "host", "query", "agent", "task")):
            raise RuntimeError(f"Refusing leaked feature name: {name}")


def _views_for(sessions: list[dict], rule: str, label_key: str, extra_aug: bool):
    real_views = []
    rows = []
    labels = []
    for session in sessions:
        view, _reached = slice_window(session, rule, min_events=10)
        if view is None:
            view, _reached = slice_window(session, "full", min_events=10)
        if view is None:
            continue
        real_views.append(view)
        rows.append(view)
        labels.append(session["label"][label_key])
        for copy in copies_for(view, extra_aug):
            if str(copy.get("augment") or "").startswith("time_warp"):
                rows.append(copy)
                labels.append(session["label"][label_key])
    return real_views, rows, labels


def fit_model(sessions: list[dict], rule: str, label_key: str, extra_aug: bool, min_df: int, n_jobs: int, trees: int = 200):
    real_views, rows, labels = _views_for(sessions, rule, label_key, extra_aug)
    if len(set(labels)) < 2 or not real_views:
        return None, None
    vocab = forest_vocab(real_views, min_df=min_df)
    _guard(vocab)
    model = make_forest(n_jobs, trees)
    model.fit(np.vstack([forest_vector(row, vocab) for row in rows]), labels)
    return model, vocab


def predict_view(model, vocab, view: dict) -> tuple[str, np.ndarray]:
    proba = model.predict_proba(forest_vector(view, vocab).reshape(1, -1))[0]
    pred = str(model.classes_[int(np.argmax(proba))])
    return pred, proba


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    y_true = [a for a, _b in pairs]
    y_pred = [b for _a, b in pairs]
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return float(report["macro avg"]["f1-score"])


def summarize(rows: list[dict], reached_only: bool) -> dict:
    usable = []
    for row in rows:
        if not row.get("pred"):
            continue
        if reached_only and not row.get("reached", True):
            continue
        usable.append(row)
    if not usable:
        return {"n": 0, "accuracy": None, "macro_f1": None}
    y_true = [row["true"] for row in usable]
    y_pred = [row["pred"] for row in usable]
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per = []
    support = Counter(y_true)
    correct = Counter(true for true, pred in zip(y_true, y_pred) if true == pred)
    for label in labels:
        per.append({
            "label": label,
            "support": support[label],
            "correct": correct[label],
            "recall": report.get(label, {}).get("recall", 0.0),
            "precision": report.get(label, {}).get("precision", 0.0),
            "f1": report.get(label, {}).get("f1-score", 0.0),
        })
    return {
        "n": len(usable),
        "n_skipped": len(rows) - len(usable),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "labels": labels,
        "per_label": per,
        "confusion": {"labels": labels, "matrix": cm.tolist()},
    }


def by_task(sessions: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        groups[session["label"]["task"]].append(session)
    return groups


def _fmt(summary: dict) -> str:
    acc, f1, n = summary.get("accuracy"), summary.get("macro_f1"), summary.get("n")
    if acc is None or f1 is None:
        return f"n {n} none-reached"
    return f"acc {acc:.3f}  f1 {f1:.3f}  n {n}"


def leave_one_task(sessions: list[dict], rule: str, label_key: str, extra_aug: bool, min_df: int, n_jobs: int) -> list[dict]:
    groups = by_task(sessions)
    preds = []
    tasks = sorted(groups)
    started = time.time()
    for index, held in enumerate(tasks):
        train = [row for task, rows in groups.items() if task != held for row in rows]
        train_labels = {row["label"][label_key] for row in train}
        model, vocab = fit_model(train, rule, label_key, extra_aug, min_df, n_jobs)
        majority = Counter(row["label"][label_key] for row in train).most_common(1)[0][0]
        for session in groups[held]:
            true = session["label"][label_key]
            view, reached = slice_window(session, rule, min_events=1 if rule.startswith("prefix_") else 10)
            if true not in train_labels or model is None or view is None:
                preds.append({
                    "session_id": session["session_id"],
                    "task": held,
                    "true": true,
                    "pred": None,
                    "reached": reached,
                    "skipped": "no view or label absent from training",
                })
                continue
            pred, proba = predict_view(model, vocab, view)
            preds.append({
                "session_id": session["session_id"],
                "task": held,
                "true": true,
                "pred": pred,
                "reached": reached,
                "confidence": float(np.max(proba)),
                "majority": majority,
                "n_actions": int(view.get("_n_actions") or 0),
            })
        if index % 5 == 0:
            print(f"    {rule} {index + 1}/{len(tasks)} {time.time() - started:.0f}s", flush=True)
    return preds


def pick_rule(train: list[dict], label_key: str, extra_aug: bool, min_df: int, n_jobs: int, seed: str) -> str:
    tasks = sorted({row["label"]["task"] for row in train})
    if len(tasks) < 4:
        return "full"
    order = tasks[:]
    random.Random(f"rule:{seed}").shuffle(order)
    cut = max(1, len(order) // 3)
    val_tasks = set(order[:cut])
    fit_sessions = [row for row in train if row["label"]["task"] not in val_tasks]
    val_sessions = [row for row in train if row["label"]["task"] in val_tasks]
    best, best_f1 = "full", -1.0
    for rule in SELECT_RULES:
        model, vocab = fit_model(fit_sessions, rule, label_key, extra_aug, min_df, n_jobs, trees=60)
        if model is None:
            continue
        pairs = []
        for session in val_sessions:
            view, _reached = slice_window(session, rule, min_events=10)
            if view is None:
                view, _reached = slice_window(session, "full", min_events=10)
            if view is None or session["label"][label_key] not in set(model.classes_):
                continue
            pred, _proba = predict_view(model, vocab, view)
            pairs.append((session["label"][label_key], pred))
        score = macro_f1(pairs)
        if score > best_f1:
            best, best_f1 = rule, score
    return best


def selected_start(sessions: list[dict], label_key: str, extra_aug: bool, min_df: int, n_jobs: int) -> tuple[list[dict], Counter]:
    groups = by_task(sessions)
    preds = []
    chosen = Counter()
    tasks = sorted(groups)
    for index, held in enumerate(tasks):
        train = [row for task, rows in groups.items() if task != held for row in rows]
        rule = pick_rule(train, label_key, extra_aug, min_df, n_jobs, held)
        chosen[rule] += 1
        model, vocab = fit_model(train, rule, label_key, extra_aug, min_df, n_jobs)
        train_labels = {row["label"][label_key] for row in train}
        for session in groups[held]:
            true = session["label"][label_key]
            view, reached = slice_window(session, rule, min_events=10)
            if true not in train_labels or model is None or view is None:
                preds.append({"session_id": session["session_id"], "task": held, "true": true, "pred": None, "rule": rule})
                continue
            pred, proba = predict_view(model, vocab, view)
            preds.append({
                "session_id": session["session_id"],
                "task": held,
                "true": true,
                "pred": pred,
                "reached": reached,
                "rule": rule,
                "confidence": float(np.max(proba)),
            })
        print(f"    selected {index + 1}/{len(tasks)} -> {rule}", flush=True)
    return preds, chosen


def _aligned(model, proba: np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros(len(classes), dtype=float)
    index = {label: i for i, label in enumerate(model.classes_)}
    for i, label in enumerate(classes):
        src = index.get(label)
        if src is not None:
            out[i] = proba[src]
    return out


def mil_and_online(sessions: list[dict], label_key: str, extra_aug: bool, min_df: int, n_jobs: int, hazard: float = 1.0 / 32.0, beam: int = 4):
    groups = by_task(sessions)
    mil_preds = []
    online_preds = []
    window_by_effect = Counter()
    tasks = sorted(groups)
    for index, held in enumerate(tasks):
        train = [row for task, rows in groups.items() if task != held for row in rows]
        choice = {row["session_id"]: "full" for row in train}
        model = vocab = None
        for _iteration in range(3):
            chosen_sessions = []
            for session in train:
                view, _reached = slice_window(session, choice[session["session_id"]], min_events=10)
                if view is None:
                    view, _reached = slice_window(session, "full", min_events=10)
                if view is None:
                    continue
                shadow = dict(session)
                shadow["_spans"] = None
                # Fit on the selected view by swapping features for this iteration.
                shadow["features"] = view["features"]
                shadow["_forced_full"] = True
                chosen_sessions.append(shadow)
            model, vocab = fit_model(chosen_sessions, "full", label_key, extra_aug, min_df, n_jobs)
            if model is None:
                break
            moved = 0
            for session in train:
                true = session["label"][label_key]
                best_name, best_p = choice[session["session_id"]], -1.0
                for rule in MIL_RULES:
                    view, _reached = slice_window(session, rule, min_events=10)
                    if view is None:
                        continue
                    if rule != "full" and int(view.get("_n_actions") or 0) < 8:
                        continue
                    if true not in set(model.classes_):
                        continue
                    _pred, proba = predict_view(model, vocab, view)
                    p_true = float(proba[list(model.classes_).index(true)])
                    if p_true > best_p:
                        best_name, best_p = rule, p_true
                if best_name != choice[session["session_id"]]:
                    moved += 1
                    choice[session["session_id"]] = best_name
            if moved == 0:
                break
        if model is None:
            continue
        for session in train:
            window_by_effect[(session["label"][label_key], choice[session["session_id"]])] += 1
        classes = [str(label) for label in model.classes_]
        for session in groups[held]:
            true = session["label"][label_key]
            if true not in set(model.classes_):
                mil_preds.append({"session_id": session["session_id"], "true": true, "pred": None})
                online_preds.append({"session_id": session["session_id"], "true": true, "pred": None})
                continue
            mix = np.zeros(len(classes), dtype=float)
            weight = 0.0
            best_rule, best_conf = "full", -1.0
            for rule in MIL_RULES:
                view, _reached = slice_window(session, rule, min_events=10)
                if view is None:
                    continue
                _pred, proba = predict_view(model, vocab, view)
                aligned = _aligned(model, proba, classes)
                conf = float(np.max(aligned))
                mix += conf * aligned
                weight += conf
                if conf > best_conf:
                    best_rule, best_conf = rule, conf
            if weight <= 0:
                mil_preds.append({"session_id": session["session_id"], "true": true, "pred": None})
            else:
                mil_preds.append({
                    "session_id": session["session_id"],
                    "task": held,
                    "true": true,
                    "pred": classes[int(np.argmax(mix))],
                    "reached": True,
                    "window": best_rule,
                    "confidence": float(np.max(mix / weight)),
                })
            online_pred = _online(model, vocab, session, classes, hazard, beam)
            online_preds.append({
                "session_id": session["session_id"],
                "task": held,
                "true": true,
                "pred": online_pred,
                "reached": True,
            })
        print(f"    mil {index + 1}/{len(tasks)}", flush=True)
    return mil_preds, online_preds, window_by_effect


def _online(model, vocab, session: dict, classes: list[str], hazard: float, beam: int) -> str | None:
    events = list((session.get("features") or {}).get("events") or [])
    spans = session.get("_spans") or action_spans(events)
    n = len(spans)
    if n == 0:
        return None
    starts = [0]
    scored: list[tuple[float, int, np.ndarray]] = []
    log_stay = np.log(max(1e-6, 1.0 - hazard))
    log_hazard = np.log(max(1e-6, hazard))
    for t in range(1, n + 1):
        pool = list(starts)
        if (t - 1) not in pool:
            pool.append(t - 1)
        scored = []
        for start in pool:
            view = _span_view(session, events, spans, start, t - 1)
            if view is None:
                continue
            _pred, proba = predict_view(model, vocab, view)
            aligned = _aligned(model, proba, classes)
            length = t - start
            log_prior = length * log_stay if start == 0 else log_hazard + (length - 1) * log_stay
            conf = float(np.max(aligned))
            scored.append((log_prior + np.log(conf + 1e-6), start, aligned))
        if not scored:
            continue
        scored.sort(key=lambda item: -item[0])
        starts = [start for _score, start, _aligned_p in scored[:beam]]
    if not scored:
        return None
    top = max(item[0] for item in scored)
    mix = np.zeros(len(classes), dtype=float)
    for score, _start, aligned in scored[:beam]:
        mix += np.exp(score - top) * aligned
    return classes[int(np.argmax(mix))]


def _span_view(session: dict, events: list[dict], spans: list[tuple[int, int]], start: int, stop: int) -> dict | None:
    chosen = events[spans[start][0] : spans[stop][1] + 1]
    if len(chosen) < 10:
        return None
    duration = int(chosen[-1].get("t_rel_ms") or 0) - int(chosen[0].get("t_rel_ms") or 0)
    return {
        "session_id": session["session_id"],
        "label": session["label"],
        "features": {"n_events": len(chosen), "duration_ms": max(0, duration), "events": chosen},
    }


def path_tokens(events: list[dict]) -> list[str]:
    tokens = []
    previous = None
    for event in events:
        path = event.get("path") or ""
        if path == previous:
            continue
        previous = path
        for part in path.split("?")[0].split("/"):
            part = re.sub(r"\.(do|php|html)$", "", part.strip().lower())
            if not part or part in {"do", "list", "www", "index"}:
                continue
            if SYS_ID.search(part) or TOKEN_SKIP.match(part) or len(part) > 40:
                continue
            tokens.append(part)
    return tokens


def fit_paths(sessions: list[dict], label_key: str, min_df: int, n_jobs: int):
    docs = []
    labels = []
    df = Counter()
    for session in sessions:
        tokens = path_tokens((session.get("features") or {}).get("events") or [])
        docs.append(tokens)
        labels.append(session["label"][label_key])
        df.update(set(tokens))
    vocab = sorted(tok for tok, n in df.items() if n >= min_df)
    if len(set(labels)) < 2 or not vocab:
        return None, vocab

    def embed(tokens: list[str]) -> np.ndarray:
        counts = Counter(tokens)
        n = max(1, len(tokens))
        values = [counts.get(tok, 0) / n for tok in vocab]
        values.append(float(np.log1p(len(tokens))))
        values.append(float(np.log1p(len(set(tokens)))))
        return np.asarray(values, dtype=float)

    model = make_forest(n_jobs)
    model.fit(np.vstack([embed(toks) for toks in docs]), labels)
    return (model, vocab, embed), vocab


def leave_one_task_paths(sessions: list[dict], label_key: str, min_df: int, n_jobs: int) -> list[dict]:
    groups = by_task(sessions)
    preds = []
    for held in sorted(groups):
        train = [row for task, rows in groups.items() if task != held for row in rows]
        fitted, _vocab = fit_paths(train, label_key, min_df, n_jobs)
        for session in groups[held]:
            true = session["label"][label_key]
            if fitted is None:
                preds.append({"session_id": session["session_id"], "true": true, "pred": None, "reached": True})
                continue
            model, vocab, embed = fitted
            if true not in set(model.classes_):
                preds.append({"session_id": session["session_id"], "true": true, "pred": None, "reached": True})
                continue
            tokens = path_tokens((session.get("features") or {}).get("events") or [])
            proba = model.predict_proba(embed(tokens).reshape(1, -1))[0]
            pred = str(model.classes_[int(np.argmax(proba))])
            preds.append({
                "session_id": session["session_id"],
                "task": held,
                "true": true,
                "pred": pred,
                "reached": True,
                "confidence": float(np.max(proba)),
            })
    return preds


def pack(name: str, rows: list[dict], seconds: float, extra: dict | None = None) -> dict:
    out = {
        "name": name,
        "seconds": round(seconds, 1),
        "all": summarize(rows, reached_only=False),
        "reached": summarize(rows, reached_only=True),
    }
    # Predictions are kept without the per-row majority label, for the audit file.
    out["predictions"] = [
        {key: value for key, value in row.items() if key != "majority"}
        for row in rows
    ]
    if rows and rows[0].get("majority"):
        maj = [{"true": row["true"], "pred": row["majority"], "reached": True} for row in rows if row.get("majority")]
        out["majority"] = summarize(maj, reached_only=False)
    if extra:
        out.update(extra)
    return out


def annotate(sessions: list[dict], tasks: dict) -> tuple[list[dict], list[dict]]:
    singles, composites = [], []
    for session in sessions:
        task_id = session["label"]["task"]
        task = tasks.get(task_id)
        if not task or not task.get("objective"):
            print(f"skip {task_id}: no objective", flush=True)
            continue
        sequence = task.get("objective_sequence")
        if sequence:
            effects = [EFFECT[name] for name in sequence]
            session["label"]["objective"] = task["objective"]
            session["label"]["effect"] = "composite"
            session["label"]["effects"] = effects
            composites.append(session)
        else:
            objective = task["objective"]
            if objective not in EFFECT:
                raise SystemExit(f"Unknown objective {objective} on {task_id}")
            session["label"]["objective"] = objective
            session["label"]["effect"] = EFFECT[objective]
            session["label"]["effects"] = [EFFECT[objective]]
            singles.append(session)
        session["_spans"] = action_spans((session.get("features") or {}).get("events") or [])
    return singles, composites


def load_sites(agents: set[str] | None = None) -> dict[str, dict]:
    """Load the three sites. `agents` defaults to browser-use only.

    Pass {"browser-use", "skyvern"} to include the second agent. Goal labels are
    applied later from the task id, so both agents share the same label.
    """
    ridge_tasks = json.loads((ROOT / "tasks" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    sn_tasks = json.loads((ROOT / "servicenow" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    gitea_tasks = json.loads((ROOT / "gitea" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    print("loading ridgeway", flush=True)
    ridge_s, ridge_c = annotate(load_ridgeway(agents), ridge_tasks)
    print("loading servicenow", flush=True)
    sn_s, sn_c = annotate(load_servicenow(agents), sn_tasks)
    print("loading gitea", flush=True)
    gitea_s, gitea_c = annotate(load_gitea(agents), gitea_tasks)
    return {
        "ridgeway": {"singles": ridge_s, "composites": ridge_c, "extra": False, "min_df": 1},
        "servicenow": {"singles": sn_s, "composites": sn_c, "extra": True, "min_df": 2},
        "gitea": {"singles": gitea_s, "composites": gitea_c, "extra": True, "min_df": 2},
    }


def describe(name: str, sessions: list[dict]) -> dict:
    effects = Counter(row["label"]["effect"] for row in sessions)
    objectives = Counter(row["label"]["objective"] for row in sessions)
    actions = [len(row.get("_spans") or []) for row in sessions]
    actions.sort()
    mid = actions[len(actions) // 2] if actions else 0
    return {
        "n_sessions": len(sessions),
        "n_tasks": len({row["label"]["task"] for row in sessions}),
        "effects": dict(effects),
        "objectives": dict(objectives),
        "actions_median": mid,
        "actions_min": actions[0] if actions else 0,
        "actions_max": actions[-1] if actions else 0,
    }


def run_site(name: str, spec: dict, n_jobs: int, conditions: set[str]) -> dict:
    singles = spec["singles"]
    extra, min_df = spec["extra"], spec["min_df"]
    print(f"\n=== {name} singles={len(singles)} tasks={describe(name, singles)['n_tasks']} ===", flush=True)
    blocks = []
    if "rules" in conditions:
        for rule in ["full", *PREFIX_RULES, *[r for r in SELECT_RULES if r != "full"]]:
            started = time.time()
            rows = leave_one_task(singles, rule, "effect", extra, min_df, n_jobs)
            block = pack(f"effect:{rule}", rows, time.time() - started)
            reached = block["reached"]
            print(f"  {rule:14} {_fmt(reached)}", flush=True)
            blocks.append(block)
        started = time.time()
        rows = leave_one_task(singles, "full", "objective", extra, min_df, n_jobs)
        block = pack("objective:full", rows, time.time() - started)
        print(f"  objective      {_fmt(block['reached'])}", flush=True)
        blocks.append(block)
    if "selected" in conditions:
        started = time.time()
        rows, chosen = selected_start(singles, "effect", extra, min_df, n_jobs)
        block = pack("effect:selected", rows, time.time() - started, {"rules_chosen": dict(chosen)})
        print(f"  selected       {_fmt(block['reached'])}  {dict(chosen)}", flush=True)
        blocks.append(block)
    if "mil" in conditions:
        started = time.time()
        mil_rows, online_rows, windows = mil_and_online(singles, "effect", extra, min_df, n_jobs)
        mil_block = pack("effect:learned_windows", mil_rows, time.time() - started, {
            "train_window_by_effect": {f"{eff}|{rule}": n for (eff, rule), n in windows.items()},
        })
        online_block = pack("effect:online", online_rows, 0.0)
        print(f"  learned        {_fmt(mil_block['reached'])}", flush=True)
        print(f"  online         {_fmt(online_block['reached'])}", flush=True)
        blocks.extend([mil_block, online_block])
    if "paths" in conditions:
        started = time.time()
        rows = leave_one_task_paths(singles, "effect", min_df, n_jobs)
        block = pack("effect:request_paths", rows, time.time() - started)
        print(f"  request paths  {_fmt(block['reached'])}", flush=True)
        blocks.append(block)
    return {"site": name, "data": describe(name, singles), "blocks": blocks}


def cross_site(sites: dict, n_jobs: int) -> list[dict]:
    """Train on two applications, test on the third. One fit, not a second holdout."""
    out = []
    names = list(sites)
    for held in names:
        train = [row for name in names if name != held for row in sites[name]["singles"]]
        test = sites[held]["singles"]
        min_df = 2
        for rule in ("full", "trail_32", "offset_10"):
            model, vocab = fit_model(train, rule, "effect", extra_aug=True, min_df=min_df, n_jobs=n_jobs)
            rows = []
            for session in test:
                true = session["label"]["effect"]
                view, reached = slice_window(session, rule, min_events=10)
                if model is None or view is None or true not in set(model.classes_):
                    rows.append({"true": true, "pred": None, "reached": reached})
                    continue
                pred, proba = predict_view(model, vocab, view)
                rows.append({"true": true, "pred": pred, "reached": reached, "confidence": float(np.max(proba))})
            summary = summarize(rows, reached_only=True)
            print(f"  cross {held:12} {rule:10} {_fmt(summary)}", flush=True)
            out.append({"held_out": held, "rule": rule, "summary": summary})
    return out


def score_composites(sites: dict, n_jobs: int) -> list[dict]:
    """Train on single-effect trials and search for an ordered split of each composite."""
    found = []
    for name, spec in sites.items():
        composites = spec["composites"]
        if not composites:
            continue
        model, vocab = fit_model(spec["singles"], "full", "effect", spec["extra"], spec["min_df"], n_jobs)
        if model is None:
            continue
        classes = [str(label) for label in model.classes_]
        for session in composites:
            effects = session["label"]["effects"]
            spans = session["_spans"]
            events = (session.get("features") or {}).get("events") or []
            n = len(spans)
            whole, _reached = slice_window(session, "full", min_events=10)
            whole_pred = None
            if whole is not None:
                whole_pred, _proba = predict_view(model, vocab, whole)
            best = None
            if len(effects) == 2 and n >= 8:
                for split in range(2, n - 2):
                    left = _span_view(session, events, spans, 0, split - 1)
                    right = _span_view(session, events, spans, split, n - 1)
                    if left is None or right is None:
                        continue
                    _pl, proba_l = predict_view(model, vocab, left)
                    _pr, proba_r = predict_view(model, vocab, right)
                    al = _aligned(model, proba_l, classes)
                    ar = _aligned(model, proba_r, classes)
                    score = al[classes.index(effects[0])] * ar[classes.index(effects[1])] if effects[0] in classes and effects[1] in classes else 0.0
                    if best is None or score > best[0]:
                        best = (score, split, classes[int(np.argmax(al))], classes[int(np.argmax(ar))])
            both = bool(best and best[2] == effects[0] and best[3] == effects[1])
            found.append({
                "site": name,
                "task": session["label"]["task"],
                "effects": effects,
                "n_actions": n,
                "whole_pred": whole_pred,
                "whole_in_effects": whole_pred in effects,
                "split_at": None if best is None else best[1],
                "split_pred": None if best is None else [best[2], best[3]],
                "both_sides": both,
            })
            print(f"  composite {name} {session['label']['task']} {effects} whole={whole_pred} split={None if best is None else best[2:]}", flush=True)
    return found


def _check() -> None:
    events = [
        {"event_name": "mousemove"},
        {"event_name": "mousedown"},
        {"event_name": "mouseup"},
        {"event_name": "click"},
        {"event_name": "keydown"},
        {"event_name": "keyup"},
        {"event_name": "mousemove"},
        {"event_name": "scroll"},
        {"event_name": "scroll"},
        {"event_name": "load"},
    ]
    spans = action_spans(events)
    kinds = [kind_of(events[end]["event_name"]) for _start, end in spans]
    if kinds != ["click", "type", "scroll", "nav"]:
        raise SystemExit(f"action collapse failed: {kinds}")
    print("action collapse ok", len(spans))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "pipelines" / "data" / "sites" / "effects" / "results.json")
    parser.add_argument("--forest-jobs", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--sites", default="ridgeway,servicenow,gitea")
    parser.add_argument("--conditions", default="rules,selected,mil,paths,cross,composites")
    args = parser.parse_args()
    if args.check:
        _check()
        return 0
    sites = load_sites()
    for name, spec in sites.items():
        info = describe(name, spec["singles"])
        comp = describe(name, spec["composites"])
        print(f"{name}: {info} composites={comp['n_sessions']}", flush=True)
    if args.load_only:
        return 0
    wanted = set(args.sites.split(","))
    conditions = set(args.conditions.split(","))
    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "Leave-one-task-out. Single-effect trials only. Time-warp copies in training, "
            "never scored. Forest: 200 trees, depth 8. Online hazard 1/32, beam 4, fixed. "
            "Selected rule: one third of the training tasks pick the rule, then the forest "
            "is refit on all training tasks. Interaction features exclude url and path; "
            "the request-path model is the only one that sees paths."
        ),
        "sites": [],
        "cross_application": [],
        "composites": [],
    }
    chosen_sites = {name: spec for name, spec in sites.items() if name in wanted}
    for name, spec in chosen_sites.items():
        results["sites"].append(run_site(name, spec, args.forest_jobs, conditions))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    if "cross" in conditions and wanted == set(sites):
        print("\n=== cross application ===", flush=True)
        results["cross_application"] = cross_site(sites, args.forest_jobs)
    if "composites" in conditions:
        print("\n=== composites ===", flush=True)
        results["composites"] = score_composites(chosen_sites, args.forest_jobs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
