#!/usr/bin/env python3
"""Five-goal recognition from action tokens.

The reported target is the five universal goals (retrieval, modification,
creation, identity, transaction). Four Site-Agnostic models read only role and
timing tokens. Schema-Adapted (stage2) adds sanitized DOM and URL tokens from
the training fold.

- forest        cumulative role counts
- ngram         the same counts plus action-role bigrams
- lstm          causal LSTM, loss after every action
- transformer   causal self-attention, loss after every action
- stage2        Site-Agnostic prior plus the schema specialist

Holdouts used in the paper: --group task (unseen tasks), --group trial (seen
tasks, one session held out), and --protocol cross (Browser Use against
Skyvern). The opening sign-in is cut on ServiceNow and Gitea. A session with
no action after that cut is scored on the training prior.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

from labels_v2 import load_sites_v2  # noqa: E402
from tokens import (  # noqa: E402
    BASE_NAMES,
    N_NUM,
    N_TOKENS,
    Tokens,
    action_site_dicts,
    assert_no_identity,
    bigram_features,
    bigram_vocab,
    prefix_features,
    site_prefix_features,
    site_vocab,
    tokenize,
)

TIERS = ["standard", "elevated", "critical"]
GOALS = [
    "read", "modify", "create", "identity", "transaction",
]
CLASSES = {"tier": TIERS, "goal": GOALS}
TRIM_LOGIN = {"servicenow", "gitea"}
WARPS = (1.0, 0.7, 1.4)
# Extra time-warped copies for the LSTM and transformer only. Forests already
# train on WARPS. Test prefixes stay at warp 1.0. Pointer stride and coordinate
# jitter do not change action tokens, so they are not applied here.
NEURAL_WARPS = (0.5, 0.7, 1.0, 1.4, 1.6)
LMAX = 128
ACTION_GRID = (1, 2, 3, 5, 8, 12, 20, 32, 50, 80, None)
SECOND_GRID = (2, 5, 10, 20, 30, 60, 120, None)
MODELS = ("forest", "ngram", "lstm", "transformer", "stage2")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def truncate(tok: Tokens, n: int) -> Tokens:
    return Tokens(tok.ids[:n], tok.num[:n], tok.t_ms[:n], tok.pages[:n], tok.ev_end[:n])


def prepare(site: str, sessions: list[dict]) -> list[dict]:
    rows = []
    trim = site in TRIM_LOGIN
    for session in sessions:
        events = session["features"]["events"]
        base = tokenize(events, 1.0, trim_login=trim)
        label = session["label"]
        row = {
            "session_id": session["session_id"],
            "task": label["task"],
            "group": label["group"],
            "group_kind": label.get("group_kind"),
            "tier": label["tier"],
            "goal": label["goal"],
            "agent": label.get("agent") or "browser-use",
            "n_total": 0 if base is None else len(base),
            "tokens": {},
            "aug_tokens": [],
            "site_dicts": [],
        }
        if base is not None:
            for warp in WARPS:
                tok = base if warp == 1.0 else tokenize(events, warp, trim_login=trim)
                row["tokens"][warp] = truncate(tok, LMAX)
            aug = []
            for warp in NEURAL_WARPS:
                if warp in row["tokens"]:
                    aug.append(row["tokens"][warp])
                    continue
                tok = tokenize(events, warp, trim_login=trim)
                if tok is not None:
                    aug.append(truncate(tok, LMAX))
            row["aug_tokens"] = aug
            row["site_dicts"] = action_site_dicts(events, row["tokens"][1.0], trim_login=trim)
        rows.append(row)
    return rows


def folds_for(rows: list[dict], n_folds: int, split: str = "group", target: str = "goal") -> list[list[int]]:
    """Whole groups per fold, or a stratified trial split. Identical for every model."""
    if split == "trial":
        from sklearn.model_selection import StratifiedKFold

        y = [row[target] for row in rows]
        counts = Counter(y)
        multi_idx = [i for i, lbl in enumerate(y) if counts[lbl] >= 2]
        single_idx = [i for i, lbl in enumerate(y) if counts[lbl] < 2]
        min_c = min(counts[y[i]] for i in multi_idx) if multi_idx else 0
        if min_c < 2:
            raise SystemExit(f"trial split needs >=2 sessions in classes, got {dict(counts)}")
        k = max(2, min(3, n_folds if n_folds else 3, min_c))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        y_multi = [y[i] for i in multi_idx]
        folds = [[multi_idx[j] for j in te] for _, te in skf.split(np.zeros(len(y_multi)), y_multi)]
        for idx_s, s_i in enumerate(single_idx):
            folds[idx_s % k].append(s_i)
        return folds
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["group"]].append(i)
    k = min(n_folds, len(groups)) if n_folds > 0 else len(groups)
    bins: list[list[int]] = [[] for _ in range(k)]
    for name in sorted(groups, key=lambda g: (-len(groups[g]), g)):
        target = min(range(k), key=lambda b: (len(bins[b]), b))
        bins[target].extend(groups[name])
    return [b for b in bins if b]


def class_weights(labels: list[str], classes: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    n = len(labels)
    present = [c for c in classes if counts[c] > 0]
    return {c: n / (len(present) * counts[c]) for c in present}


# ---------------------------------------------------------------------------
# forest models
# ---------------------------------------------------------------------------

def prefix_rows(n: int) -> list[int]:
    dense = list(range(1, min(n, 20) + 1))
    sparse = list(range(24, n + 1, 4))
    return sorted(set(dense + sparse + [n]))


def feature_matrix(tok: Tokens, vocab: dict | None, site_dicts: list | None = None, s_vocab: dict | None = None) -> np.ndarray:
    base = prefix_features(tok)
    parts = [base]
    if vocab is not None:
        parts.append(bigram_features(tok, vocab))
    if site_dicts is not None and s_vocab is not None:
        parts.append(site_prefix_features(site_dicts, s_vocab))
    return parts[0] if len(parts) == 1 else np.hstack(parts)


def forest_predict(train: list[dict], test: list[dict], target: str, mode: str, seed: int, n_jobs: int):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    classes = CLASSES[target]
    live = [row for row in train if row["tokens"]]
    labels = [row[target] for row in live]
    if len(set(labels)) < 2:
        return None
    use_ngram = mode in ("ngram", "stage2")
    use_site = mode == "stage2"
    vocab = bigram_vocab([row["tokens"][1.0] for row in live]) if use_ngram else None
    s_vocab = site_vocab([row["site_dicts"] for row in live], min_sessions=1) if use_site else None
    weights = class_weights(labels, classes)
    x_rows, y_rows, w_rows = [], [], []
    for row in live:
        s_dicts = row["site_dicts"] if use_site else None
        for warp, tok in row["tokens"].items():
            matrix = feature_matrix(tok, vocab, s_dicts, s_vocab)
            keep = prefix_rows(len(tok))
            x_rows.append(matrix[np.asarray(keep) - 1])
            y_rows.extend([row[target]] * len(keep))
            w_rows.extend([weights[row[target]] / len(keep) / len(row["tokens"])] * len(keep))
    x = np.vstack(x_rows)
    if mode == "forest":
        assert x.shape[1] == len(BASE_NAMES)
    sample_w = np.asarray(w_rows) * len(w_rows) / sum(w_rows)
    if use_site:
        # Stage 2 combines the prefix-trained agnostic forest with prefix-trained
        # and milestone/endpoint-trained schema-aware ExtraTrees/RF classifiers.
        agn_x_rows = []
        late_x_rows, late_y_rows, late_w_rows = [], [], []
        for row in live:
            s_dicts = row["site_dicts"]
            for warp, tok in row["tokens"].items():
                mat_agn = feature_matrix(tok, vocab, None, None)
                keep = prefix_rows(len(tok))
                agn_x_rows.append(mat_agn[np.asarray(keep) - 1])
            # Also train a full-session schema specialist on the final action of each training session
            tok1 = row["tokens"][1.0]
            mat1 = feature_matrix(tok1, vocab, s_dicts, s_vocab)
            late_x_rows.append(mat1[-1:])
            late_y_rows.append(row[target])
            late_w_rows.append(weights[row[target]])
        x_agn = np.vstack(agn_x_rows)
        x_late = np.vstack(late_x_rows)
        w_late = np.asarray(late_w_rows) * len(late_w_rows) / sum(late_w_rows)
        m_agn = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3, max_features="sqrt",
            random_state=seed, n_jobs=n_jobs,
        )
        m_site_pfx = ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=n_jobs,
        )
        m_site_late1 = ExtraTreesClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed + 7, n_jobs=n_jobs,
        )
        m_site_late2 = RandomForestClassifier(
            n_estimators=600, max_depth=14, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed + 17, n_jobs=n_jobs,
        )
        m_agn.fit(x_agn, y_rows, sample_weight=sample_w)
        m_site_pfx.fit(x, y_rows, sample_weight=sample_w)
        m_site_late1.fit(x_late, late_y_rows, sample_weight=w_late)
        m_site_late2.fit(x_late, late_y_rows, sample_weight=w_late)
        idx_agn = {label: i for i, label in enumerate(m_agn.classes_)}
        idx_sp = {label: i for i, label in enumerate(m_site_pfx.classes_)}
        idx_sl1 = {label: i for i, label in enumerate(m_site_late1.classes_)}
        idx_sl2 = {label: i for i, label in enumerate(m_site_late2.classes_)}
        out = {}
        for row in test:
            if not row["tokens"]:
                continue
            tok = row["tokens"][1.0]
            mat_agn = feature_matrix(tok, vocab, None, None)
            mat_site = feature_matrix(tok, vocab, row["site_dicts"], s_vocab)
            p_agn = m_agn.predict_proba(mat_agn)
            p_sp = m_site_pfx.predict_proba(mat_site)
            p_sl1 = m_site_late1.predict_proba(mat_site)
            p_sl2 = m_site_late2.predict_proba(mat_site)
            n_steps = p_agn.shape[0]
            # Smoothly ramp weight from prefix specialist early in the trace to endpoint specialist late in the trace
            alpha = np.linspace(0.25, 0.85, n_steps)[:, None]
            full = np.zeros((n_steps, len(classes)))
            for j, label in enumerate(classes):
                pa = p_agn[:, idx_agn[label]] if label in idx_agn else np.zeros(n_steps)
                psp = p_sp[:, idx_sp[label]] if label in idx_sp else np.zeros(n_steps)
                psl1 = p_sl1[:, idx_sl1[label]] if label in idx_sl1 else np.zeros(n_steps)
                psl2 = p_sl2[:, idx_sl2[label]] if label in idx_sl2 else np.zeros(n_steps)
                p_early = 0.35 * pa + 0.65 * psp
                p_late = 0.15 * pa + 0.425 * psl1 + 0.425 * psl2
                full[:, j] = (1.0 - alpha[:, 0]) * p_early + alpha[:, 0] * p_late
            out[row["session_id"]] = full
        return out

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=n_jobs,
    )
    model.fit(x, y_rows, sample_weight=sample_w)
    index = {label: i for i, label in enumerate(model.classes_)}
    out = {}
    for row in test:
        if not row["tokens"]:
            continue
        tok = row["tokens"][1.0]
        proba = model.predict_proba(feature_matrix(tok, vocab, None, None))
        full = np.zeros((proba.shape[0], len(classes)))
        for j, label in enumerate(classes):
            if label in index:
                full[:, j] = proba[:, index[label]]
        out[row["session_id"]] = full
    return out


# ---------------------------------------------------------------------------
# neural models
# ---------------------------------------------------------------------------

def build_net(kind: str, n_tier: int, n_goal: int, dim: int = 64):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.embed = nn.Embedding(N_TOKENS, dim)
            self.numeric = nn.Linear(N_NUM, dim)
            self.drop = nn.Dropout(0.2)
            if kind == "lstm":
                self.body = nn.LSTM(dim, dim, num_layers=1, batch_first=True)
            else:
                self.position = nn.Embedding(LMAX, dim)
                # torch 1.9 (jasmine) has neither norm_first nor enable_nested_tensor.
                try:
                    layer = nn.TransformerEncoderLayer(
                        d_model=dim, nhead=4, dim_feedforward=2 * dim, dropout=0.1,
                        batch_first=True, norm_first=True,
                    )
                except TypeError:
                    layer = nn.TransformerEncoderLayer(
                        d_model=dim, nhead=4, dim_feedforward=2 * dim, dropout=0.1,
                        batch_first=True,
                    )
                try:
                    self.body = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
                except TypeError:
                    self.body = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(dim)
            self.tier = nn.Linear(dim, n_tier)
            self.goal = nn.Linear(dim, n_goal)

        def forward(self, ids, num):
            x = self.embed(ids) + self.numeric(num)
            if self.kind == "transformer":
                steps = torch.arange(ids.shape[1], device=ids.device)
                x = x + self.position(steps)[None]
                mask = torch.triu(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device), 1)
                h = self.body(self.drop(x), mask=mask)
            else:
                h, _ = self.body(self.drop(x))
            h = self.norm(h)
            return self.tier(h), self.goal(h)

    return Net()


def pad_batch(toks: list[Tokens]):
    import torch

    n = max(len(t) for t in toks)
    ids = np.zeros((len(toks), n), dtype=np.int64)
    num = np.zeros((len(toks), n, N_NUM), dtype=np.float32)
    mask = np.zeros((len(toks), n), dtype=np.float32)
    for i, t in enumerate(toks):
        ids[i, : len(t)] = t.ids
        num[i, : len(t)] = t.num
        mask[i, : len(t)] = 1.0
    return torch.from_numpy(ids), torch.from_numpy(num), torch.from_numpy(mask)


def neural_predict(kind: str, train: list[dict], test: list[dict], target: str, seed: int, epochs: int, device):
    import torch
    from torch import nn

    live = [row for row in train if row["tokens"]]
    if len({row[target] for row in live}) < 2:
        return None
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_net(kind, len(TIERS), len(GOALS)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    tier_w = class_weights([row["tier"] for row in live], TIERS)
    goal_w = class_weights([row["goal"] for row in live], GOALS)
    tier_idx = {c: i for i, c in enumerate(TIERS)}
    goal_idx = {c: i for i, c in enumerate(GOALS)}
    tier_weight = torch.tensor([tier_w.get(c, 0.0) for c in TIERS], dtype=torch.float32, device=device)
    goal_weight = torch.tensor([goal_w.get(c, 0.0) for c in GOALS], dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(reduction="none")
    batch = 16
    samples = []
    for row in live:
        copies = row.get("aug_tokens") or [row["tokens"][1.0]]
        for tok in copies:
            samples.append((row, tok))
    for _epoch in range(epochs):
        net.train()
        order = rng.permutation(len(samples))
        for start in range(0, len(order), batch):
            chunk = [samples[i] for i in order[start : start + batch]]
            toks = [tok for _row, tok in chunk]
            rows_c = [row for row, _tok in chunk]
            ids, num, mask = pad_batch(toks)
            ids, num, mask = ids.to(device), num.to(device), mask.to(device)
            y_tier = torch.tensor([tier_idx[row["tier"]] for row in rows_c], device=device)
            y_goal = torch.tensor([goal_idx[row["goal"]] for row in rows_c], device=device)
            tier_logits, goal_logits = net(ids, num)
            steps = mask.sum(dim=1).clamp(min=1.0)
            loss = 0.0
            for logits, y, weight in ((tier_logits, y_tier, tier_weight), (goal_logits, y_goal, goal_weight)):
                flat = ce(logits.reshape(-1, logits.shape[-1]), y[:, None].expand_as(mask).reshape(-1))
                per = (flat.reshape(mask.shape) * mask).sum(dim=1) / steps
                loss = loss + (per * weight[y]).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    out = {}
    with torch.no_grad():
        for row in test:
            if not row["tokens"]:
                continue
            tok = row["tokens"][1.0]
            ids, num, _mask = pad_batch([tok])
            tier_logits, goal_logits = net(ids.to(device), num.to(device))
            logits = tier_logits if target == "tier" else goal_logits
            out[row["session_id"]] = torch.softmax(logits[0], dim=-1).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def state_at_action(probs: np.ndarray, k: int | None) -> np.ndarray:
    index = len(probs) - 1 if k is None else min(k, len(probs)) - 1
    return probs[index]


def state_at_second(probs: np.ndarray, t_ms: np.ndarray, seconds: float | None, prior: np.ndarray) -> np.ndarray:
    if seconds is None:
        return probs[-1]
    seen = int(np.searchsorted(t_ms, seconds * 1000.0, side="right"))
    return prior if seen == 0 else probs[seen - 1]


def macro_f1(true: list[int], pred: list[int], n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(true, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(true, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(true, pred) if t == c and p != c)
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def score_records(records: list[dict], classes: list[str], target: str) -> dict:
    """Curves over action count and elapsed seconds, every session counted.

    A session shorter than the mark is read at its last action. A session with no
    action, or no action yet by the mark, falls back to the training prior.
    """
    index = {c: i for i, c in enumerate(classes)}
    true = [index[r["true"]] for r in records]
    out = {"n": len(records), "by_action": {}, "by_second": {}}
    for grid, name, fn in (
        (ACTION_GRID, "by_action", lambda r, g: state_at_action(r["probs"], g) if len(r["probs"]) else r["prior"]),
        (SECOND_GRID, "by_second", lambda r, g: state_at_second(r["probs"], r["t_ms"], g, r["prior"]) if len(r["probs"]) else r["prior"]),
    ):
        for g in grid:
            pred = [int(np.argmax(fn(r, g))) for r in records]
            recall = {}
            for c, label in enumerate(classes):
                hits = [p == c for t, p in zip(true, pred) if t == c]
                if hits:
                    recall[label] = float(np.mean(hits))
            out[name]["full" if g is None else str(g)] = {
                "acc": float(np.mean([t == p for t, p in zip(true, pred)])),
                "macro_f1": macro_f1(true, pred, len(classes)),
                "recall": recall,
            }
    return out


def _run_fold_worker(args):
    fold, rows, target, model, seed, epochs, device_str, classes = args
    import torch
    torch.set_num_threads(1)
    device = torch.device(device_str)
    held = set(fold)
    train = [row for i, row in enumerate(rows) if i not in held]
    test = [rows[i] for i in fold]
    labels = [row[target] for row in train]
    prior = np.asarray([labels.count(c) for c in classes], dtype=float)
    prior = prior / prior.sum()
    if model in ("forest", "ngram", "stage2"):
        probs = forest_predict(train, test, target, model, seed, 2)
    else:
        probs = neural_predict(model, train, test, target, seed, epochs, device)
    fold_recs = []
    for row in test:
        p = None if probs is None else probs.get(row["session_id"])
        tok = row["tokens"].get(1.0)
        fold_recs.append({
            "session_id": row["session_id"],
            "task": row["task"],
            "group": row["group"],
            "group_kind": row["group_kind"],
            "true": row[target],
            "agent": row.get("agent"),
            "tier": row["tier"],
            "goal": row["goal"],
            "n_actions": 0 if tok is None else len(tok),
            "t_ms": [] if tok is None else [int(v) for v in tok.t_ms],
            "probs": np.zeros((0, len(classes))) if p is None else p,
            "prior": prior,
        })
    return fold_recs


def run_site(
    site: str,
    rows: list[dict],
    target: str,
    models: list[str],
    seeds: int,
    n_folds: int,
    epochs: int,
    n_jobs: int,
    device,
    split: str = "group",
    fixed_folds: list[list[int]] | None = None,
) -> dict:
    classes = CLASSES[target]
    folds = fixed_folds if fixed_folds is not None else folds_for(rows, n_folds, split, target)
    true_counts = Counter(row[target] for row in rows)
    majority = true_counts.most_common(1)[0]
    print(
        f"=== {site}: {len(rows)} sessions, {len({r['group'] for r in rows})} groups, {len(folds)} folds, "
        f"{sum(1 for r in rows if not r['tokens'])} without actions after sign-in ===",
        flush=True,
    )
    print(f"  {target} counts {dict(true_counts)}  majority {majority[0]} {majority[1] / len(rows):.3f}", flush=True)
    result = {
        "site": site,
        "target": target,
        "n_sessions": len(rows),
        "n_groups": len({r["group"] for r in rows}),
        "n_folds": len(folds),
        "n_empty": sum(1 for r in rows if not r["tokens"]),
        "counts": dict(true_counts),
        "majority": {"label": majority[0], "accuracy": majority[1] / len(rows)},
        "models": {},
    }
    for model in models:
        runs = []
        for seed in range(seeds):
            t0 = time.time()
            worker_args = [(fold, rows, target, model, seed, epochs, str(device), classes) for fold in folds]
            if str(device) == "cpu" and n_jobs > 1:
                from multiprocessing import Pool
                import torch
                torch.set_num_threads(1)
                with Pool(processes=min(len(folds), n_jobs)) as pool:
                    fold_lists = pool.map(_run_fold_worker, worker_args)
            else:
                fold_lists = [_run_fold_worker(wa) for wa in worker_args]
            records = [r for fl in fold_lists for r in fl]
            scored = score_records(records, classes, target)
            runs.append({
                "seed": seed,
                "scores": scored,
                "records": [
                    {**{k: v for k, v in r.items() if k not in ("probs", "prior")},
                     "probs": np.round(r["probs"], 3).tolist(),
                     "prior": np.round(r["prior"], 3).tolist()}
                    for r in records
                ],
            })
            final = scored["by_action"]["full"]
            early = scored["by_action"]["5"]
            print(
                f"  {model:12s} seed {seed}  acc@5 {early['acc']:.3f}  acc@full {final['acc']:.3f}  "
                f"f1@full {final['macro_f1']:.3f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
        result["models"][model] = runs
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sites", default="ridgeway,servicenow,gitea")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--target", choices=("goal", "tier"), default="goal")
    parser.add_argument("--group", choices=("task", "trial", "hybrid", "family"), default="task")
    parser.add_argument("--agents", default="browser-use", help="Comma-separated: browser-use, skyvern")
    parser.add_argument(
        "--protocol",
        choices=("grouped", "cross"),
        default="grouped",
        help="grouped = within the selected agents; cross = train on one agent, test on the other",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--folds", type=int, default=10, help="0 = leave one group out")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--forest-jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    assert_no_identity(BASE_NAMES)

    import torch

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    models = [m for m in args.models.split(",") if m]
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m}")
    agents = {name.strip() for name in args.agents.split(",") if name.strip()}
    if args.protocol == "cross" and agents != {"browser-use", "skyvern"}:
        raise SystemExit("cross protocol needs --agents browser-use,skyvern")
    label_group = "task" if args.group == "trial" else args.group
    sites = load_sites_v2(args.target, label_group, agents=agents)
    out_path = Path(args.out) if args.out else REPO / "data" / "results" / f"causal_{args.target}_{args.group}_{args.protocol}.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "group": args.group,
        "protocol": args.protocol,
        "agents": sorted(agents),
        "folds": args.folds,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "lmax": LMAX,
        "warps": list(WARPS),
        "neural_warps": list(NEURAL_WARPS),
        "neural_augmentation": "time-warp copies at train time for lstm and transformer; test uses warp 1.0",
        "sites": {},
    }
    for site in args.sites.split(","):
        rows = prepare(site, sites[site]["singles"])
        if args.protocol == "cross":
            payload["sites"][site] = {}
            for src, dst in (("browser-use", "skyvern"), ("skyvern", "browser-use")):
                subset = [row for row in rows if row["agent"] in (src, dst)]
                test_idx = [i for i, row in enumerate(subset) if row["agent"] == dst]
                key = f"{src}_to_{dst}"
                payload["sites"][site][key] = run_site(
                    f"{site}:{key}", subset, args.target, models, args.seeds, args.folds,
                    args.epochs, args.forest_jobs, device, fixed_folds=[test_idx],
                )
        else:
            payload["sites"][site] = run_site(
                site, rows, args.target, models, args.seeds, args.folds, args.epochs,
                args.forest_jobs, device, split="trial" if args.group == "trial" else "group",
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
