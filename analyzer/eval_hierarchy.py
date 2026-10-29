#!/usr/bin/env python3
"""Auxiliary labels from the appendix: read/write, five goals with a subsystem
held out, and the UI subsystem of the task.

Site-Agnostic is the n-gram forest. Schema-Adapted adds sanitized DOM and URL
tokens. Writes data/results/hierarchy_eval.json.
"""
from __future__ import annotations

import json
import time
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]

from causal import (  # noqa: E402
    class_weights,
    feature_matrix,
    macro_f1,
    prefix_rows,
    prepare,
    score_records,
)
from labels_v2 import load_sites_v2  # noqa: E402
from tokens import bigram_vocab, site_vocab  # noqa: E402

SITES = ["ridgeway", "servicenow", "gitea"]
FRACTIONS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
GLOBAL_ROWS: dict[str, list[dict]] = {}


def precompute_site_matrices(rows: list[dict]) -> None:
    live = [r for r in rows if r["tokens"]]
    vocab_ngram = bigram_vocab([r["tokens"][1.0] for r in live])
    s_vocab = site_vocab([r["site_dicts"] for r in live], min_sessions=1)
    for row in rows:
        tok1 = row["tokens"].get(1.0)
        row["n_act"] = 0 if tok1 is None else len(tok1)
        row["t_ms_list"] = [] if tok1 is None else [int(v) for v in tok1.t_ms]
        if not row["tokens"]:
            row["pfx_f"] = []
            row["pfx_n"] = []
            row["pfx_s"] = []
            row["eval_f"] = None
            row["eval_n"] = None
            row["eval_s"] = None
            row["late_s"] = None
        else:
            s_dicts = row["site_dicts"]
            pf_list, pn_list, ps_list = [], [], []
            for warp, tok in row["tokens"].items():
                mf = feature_matrix(tok, None, None, None).astype(np.float32)
                mn = feature_matrix(tok, vocab_ngram, None, None).astype(np.float32)
                ms = feature_matrix(tok, vocab_ngram, s_dicts, s_vocab).astype(np.float32)
                keep = prefix_rows(len(tok))
                idx_k = np.asarray(keep) - 1
                pf_list.append((mf[idx_k], len(keep)))
                pn_list.append((mn[idx_k], len(keep)))
                ps_list.append((ms[idx_k], len(keep)))
                if warp == 1.0:
                    row["eval_f"] = mf
                    row["eval_n"] = mn
                    row["eval_s"] = ms
                    row["late_s"] = ms[-1:]
            row["pfx_f"] = pf_list
            row["pfx_n"] = pn_list
            row["pfx_s"] = ps_list
        # Free raw dicts so memory footprint is small
        row["tokens"] = bool(row["n_act"] > 0)
        row["site_dicts"] = None


def make_folds(rows: list[dict], target_key: str, split_mode: str, n_folds: int = 10) -> list[list[int]]:
    if split_mode in ("hybrid", "task", "family"):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            g = row["group"] if split_mode == "hybrid" else row[split_mode]
            groups[g].append(i)
        k = min(n_folds, len(groups))
        bins: list[list[int]] = [[] for _ in range(k)]
        for name in sorted(groups, key=lambda g: (-len(groups[g]), g)):
            b_idx = min(range(k), key=lambda b: (len(bins[b]), b))
            bins[b_idx].extend(groups[name])
        return [b for b in bins if b]
    elif split_mode == "trial":
        y = [row[target_key] for row in rows]
        counts = Counter(y)
        min_c = min(counts.values())
        k = max(2, min(3, min_c))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        return [te.tolist() for _, te in skf.split(np.zeros(len(y)), y)]
    raise ValueError(split_mode)


def _fold_worker(args):
    site, fold, target_key, classes, seed = args
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    rows = GLOBAL_ROWS[site]
    held = set(fold)
    train = [row for i, row in enumerate(rows) if i not in held]
    test = [rows[i] for i in fold]
    live = [row for row in train if row["tokens"]]
    labels = [row[target_key] for row in train]
    prior = np.asarray([labels.count(c) for c in classes], dtype=float)
    prior = prior / prior.sum()

    if len(set(row[target_key] for row in live)) < 2:
        empty_recs = []
        for row in test:
            empty_recs.append({
                "session_id": row["session_id"],
                "task": row["task"],
                "group": row["group"],
                "true": row[target_key],
                "n_actions": row["n_act"],
                "t_ms": row["t_ms_list"],
                "probs": np.zeros((0, len(classes))),
                "prior": prior,
            })
        return empty_recs, empty_recs, empty_recs

    weights = class_weights([row[target_key] for row in live], classes)
    x_forest, x_ngram, x_site, y_rows, w_rows = [], [], [], [], []
    late_x_rows, late_y_rows, late_w_rows = [], [], []
    for row in live:
        pfx_iter = zip(row["pfx_f"][:1], row["pfx_n"][:1], row["pfx_s"][:1]) if len(classes) > 5 else zip(row["pfx_f"], row["pfx_n"], row["pfx_s"])
        n_w = 1 if len(classes) > 5 else len(row["pfx_f"])
        for (mf, nk), (mn, _), (ms, _) in pfx_iter:
            x_forest.append(mf)
            x_ngram.append(mn)
            x_site.append(ms)
            y_rows.extend([row[target_key]] * nk)
            w_rows.extend([weights[row[target_key]] / nk / n_w] * nk)
        late_x_rows.append(row["late_s"])
        late_y_rows.append(row[target_key])
        late_w_rows.append(weights[row[target_key]])

    xf = np.vstack(x_forest)
    xn = np.vstack(x_ngram)
    xs = np.vstack(x_site)
    sample_w = np.asarray(w_rows) * len(w_rows) / sum(w_rows)
    xlate = np.vstack(late_x_rows)
    wlate = np.asarray(late_w_rows) * len(late_w_rows) / sum(late_w_rows)

    m_forest = RandomForestClassifier(
        n_estimators=180, max_depth=8, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=1,
    )
    m_ngram = RandomForestClassifier(
        n_estimators=180, max_depth=8, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=1,
    )
    m_site_pfx = ExtraTreesClassifier(
        n_estimators=260, max_depth=None, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=1,
    )
    m_site_late1 = ExtraTreesClassifier(
        n_estimators=320, max_depth=None, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed + 7, n_jobs=1,
    )
    m_site_late2 = RandomForestClassifier(
        n_estimators=320, max_depth=16, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed + 17, n_jobs=1,
    )

    m_forest.fit(xf, y_rows, sample_weight=sample_w)
    m_ngram.fit(xn, y_rows, sample_weight=sample_w)
    m_site_pfx.fit(xs, y_rows, sample_weight=sample_w)
    m_site_late1.fit(xlate, late_y_rows, sample_weight=wlate)
    m_site_late2.fit(xlate, late_y_rows, sample_weight=wlate)

    sc_late = StandardScaler()
    xlate_sc = sc_late.fit_transform(xlate)
    lr_late = LogisticRegression(C=0.5, max_iter=150, class_weight="balanced", random_state=seed + 23)
    lr_late.fit(xlate_sc, late_y_rows)

    idx_f = {label: i for i, label in enumerate(m_forest.classes_)}
    idx_n = {label: i for i, label in enumerate(m_ngram.classes_)}
    idx_sp = {label: i for i, label in enumerate(m_site_pfx.classes_)}
    idx_sl1 = {label: i for i, label in enumerate(m_site_late1.classes_)}
    idx_sl2 = {label: i for i, label in enumerate(m_site_late2.classes_)}
    idx_lr = {label: i for i, label in enumerate(lr_late.classes_)}

    recs_f, recs_n, recs_s2 = [], [], []
    for row in test:
        base_meta = {
            "session_id": row["session_id"],
            "task": row["task"],
            "group": row["group"],
            "true": row[target_key],
            "n_actions": row["n_act"],
            "t_ms": row["t_ms_list"],
            "prior": prior,
        }
        if not row["tokens"]:
            z = np.zeros((0, len(classes)))
            recs_f.append({**base_meta, "probs": z})
            recs_n.append({**base_meta, "probs": z})
            recs_s2.append({**base_meta, "probs": z})
            continue

        mat_f = row["eval_f"]
        mat_n = row["eval_n"]
        mat_s = row["eval_s"]

        pf = m_forest.predict_proba(mat_f)
        pn = m_ngram.predict_proba(mat_n)
        psp = m_site_pfx.predict_proba(mat_s)
        psl1 = m_site_late1.predict_proba(mat_s)
        psl2 = m_site_late2.predict_proba(mat_s)
        plr = lr_late.predict_proba(sc_late.transform(mat_s))

        n_steps = pf.shape[0]
        alpha = np.linspace(0.25, 0.85, n_steps)[:, None]

        full_f = np.zeros((n_steps, len(classes)))
        full_n = np.zeros((n_steps, len(classes)))
        full_s2 = np.zeros((n_steps, len(classes)))
        for j, label in enumerate(classes):
            vf = pf[:, idx_f[label]] if label in idx_f else np.zeros(n_steps)
            vn = pn[:, idx_n[label]] if label in idx_n else np.zeros(n_steps)
            vsp = psp[:, idx_sp[label]] if label in idx_sp else np.zeros(n_steps)
            vsl1 = psl1[:, idx_sl1[label]] if label in idx_sl1 else np.zeros(n_steps)
            vsl2 = psl2[:, idx_sl2[label]] if label in idx_sl2 else np.zeros(n_steps)
            vlr = plr[:, idx_lr[label]] if label in idx_lr else np.zeros(n_steps)

            full_f[:, j] = vf
            full_n[:, j] = vn
            if len(classes) > 5:
                p_early = 0.20 * vn + 0.55 * vsp + 0.25 * vlr
                p_late = 0.10 * vn + 0.30 * vsl1 + 0.30 * vsl2 + 0.30 * vlr
            else:
                p_early = 0.35 * vn + 0.65 * vsp
                p_late = 0.15 * vn + 0.425 * vsl1 + 0.425 * vsl2
            full_s2[:, j] = (1.0 - alpha[:, 0]) * p_early + alpha[:, 0] * p_late

        recs_f.append({**base_meta, "probs": full_f})
        recs_n.append({**base_meta, "probs": full_n})
        recs_s2.append({**base_meta, "probs": full_s2})

    return recs_f, recs_n, recs_s2


def stratified_subsample(live: list[dict], target_key: str, frac: float, seed: int) -> list[dict]:
    if frac >= 0.999:
        return list(live)
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in live:
        by_class[row[target_key]].append(row)
    sub = []
    for cls, items in sorted(by_class.items()):
        n_total = len(items)
        k = max(1, int(round(frac * n_total)))
        idx = rng.choice(n_total, size=min(k, n_total), replace=False)
        for i in idx:
            sub.append(items[int(i)])
    return sub


def _cal_worker(args):
    site, fold, target_key, seed, classes = args
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    rows = GLOBAL_ROWS[site]
    held = set(fold)
    train = [row for i, row in enumerate(rows) if i not in held]
    test = [rows[i] for i in fold]
    live = [row for row in train if row["tokens"]]
    labels = [row[target_key] for row in train]
    prior = np.asarray([labels.count(c) for c in classes], dtype=float)
    prior = prior / prior.sum()

    weights_full = class_weights([row[target_key] for row in live], classes)
    agn_x_rows, agn_y_rows, agn_w_rows = [], [], []
    for row in live:
        mn, nk = row["pfx_n"][0]
        agn_x_rows.append(mn)
        agn_y_rows.extend([row[target_key]] * nk)
        agn_w_rows.extend([weights_full[row[target_key]] / nk] * nk)

    x_agn = np.vstack(agn_x_rows)
    w_agn = np.asarray(agn_w_rows) * len(agn_w_rows) / sum(agn_w_rows)
    m_agn = RandomForestClassifier(
        n_estimators=180, max_depth=8, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=1,
    )
    m_agn.fit(x_agn, agn_y_rows, sample_weight=w_agn)
    idx_agn = {label: i for i, label in enumerate(m_agn.classes_)}

    agn_test_probs = {}
    for row in test:
        if not row["tokens"]:
            continue
        sid = row["session_id"]
        p_agn = m_agn.predict_proba(row["eval_n"])
        full_a = np.zeros((p_agn.shape[0], len(classes)))
        for j, label in enumerate(classes):
            if label in idx_agn:
                full_a[:, j] = p_agn[:, idx_agn[label]]
        agn_test_probs[sid] = full_a

    out_by_frac = {}
    for f_idx, frac in enumerate(FRACTIONS):
        sub_live = stratified_subsample(live, target_key, frac, seed * 1009 + f_idx * 37 + fold[0])
        sub_labels = [row[target_key] for row in sub_live]
        if len(set(sub_labels)) < 2:
            continue
        weights_sub = class_weights(sub_labels, classes)

        x_rows, y_rows, w_rows = [], [], []
        late_x_rows, late_y_rows, late_w_rows = [], [], []
        for row in sub_live:
            ms, nk = row["pfx_s"][0]
            x_rows.append(ms)
            y_rows.extend([row[target_key]] * nk)
            w_rows.extend([weights_sub[row[target_key]] / nk] * nk)
            late_x_rows.append(row["late_s"])
            late_y_rows.append(row[target_key])
            late_w_rows.append(weights_sub[row[target_key]])

        x_sub = np.vstack(x_rows)
        w_sub = np.asarray(w_rows) * len(w_rows) / sum(w_rows)
        x_late = np.vstack(late_x_rows)
        w_late = np.asarray(late_w_rows) * len(late_w_rows) / sum(late_w_rows)

        m_site_pfx = ExtraTreesClassifier(
            n_estimators=220, max_depth=None, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=1,
        )
        m_site_late1 = ExtraTreesClassifier(
            n_estimators=280, max_depth=None, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed + 7, n_jobs=1,
        )
        m_site_late2 = RandomForestClassifier(
            n_estimators=280, max_depth=16, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed + 17, n_jobs=1,
        )
        m_site_pfx.fit(x_sub, y_rows, sample_weight=w_sub)
        m_site_late1.fit(x_late, late_y_rows, sample_weight=w_late)
        m_site_late2.fit(x_late, late_y_rows, sample_weight=w_late)

        sc_late = StandardScaler()
        xlate_sc = sc_late.fit_transform(x_late)
        lr_late = LogisticRegression(C=0.5, max_iter=150, class_weight="balanced", random_state=seed + 23)
        lr_late.fit(xlate_sc, late_y_rows)

        idx_sp = {label: i for i, label in enumerate(m_site_pfx.classes_)}
        idx_sl1 = {label: i for i, label in enumerate(m_site_late1.classes_)}
        idx_sl2 = {label: i for i, label in enumerate(m_site_late2.classes_)}
        idx_lr = {label: i for i, label in enumerate(lr_late.classes_)}

        w_prior_late = max(0.10, 0.35 * (1.0 - frac))
        w_spec = (1.0 - w_prior_late)

        recs_s1, recs_so, recs_s2 = [], [], []
        for row in test:
            sid = row["session_id"]
            if not row["tokens"]:
                p_prior = int(np.argmax(prior))
                recs_s1.append((row[target_key], p_prior, p_prior))
                recs_so.append((row[target_key], p_prior, p_prior))
                recs_s2.append((row[target_key], p_prior, p_prior))
                continue
            mat_site = row["eval_s"]
            pa_mat = agn_test_probs[sid]
            p_sp = m_site_pfx.predict_proba(mat_site)
            p_sl1 = m_site_late1.predict_proba(mat_site)
            p_sl2 = m_site_late2.predict_proba(mat_site)
            p_lr = lr_late.predict_proba(sc_late.transform(mat_site))
            n_steps = pa_mat.shape[0]
            alpha = np.linspace(0.25, 0.85, n_steps)[:, None]

            full_so = np.zeros((n_steps, len(classes)))
            full_s2 = np.zeros((n_steps, len(classes)))
            for j, label in enumerate(classes):
                pa = pa_mat[:, j]
                psp = p_sp[:, idx_sp[label]] if label in idx_sp else np.zeros(n_steps)
                psl1 = p_sl1[:, idx_sl1[label]] if label in idx_sl1 else np.zeros(n_steps)
                psl2 = psl2[:, idx_sl2[label]] if label in idx_sl2 else np.zeros(n_steps)
                plr = p_lr[:, idx_lr[label]] if label in idx_lr else np.zeros(n_steps)

                so_early = 0.70 * psp + 0.30 * plr
                so_late = 0.35 * psl1 + 0.35 * psl2 + 0.30 * plr
                full_so[:, j] = (1.0 - alpha[:, 0]) * so_early + alpha[:, 0] * so_late

                s2_early = 0.25 * pa + 0.75 * so_early
                s2_late = w_prior_late * pa + w_spec * so_late
                full_s2[:, j] = (1.0 - alpha[:, 0]) * s2_early + alpha[:, 0] * s2_late

            k5 = min(5, n_steps) - 1
            recs_s1.append((row[target_key], int(np.argmax(pa_mat[k5])), int(np.argmax(pa_mat[-1]))))
            recs_so.append((row[target_key], int(np.argmax(full_so[k5])), int(np.argmax(full_so[-1]))))
            recs_s2.append((row[target_key], int(np.argmax(full_s2[k5])), int(np.argmax(full_s2[-1]))))

        out_by_frac[str(frac)] = {
            "n_calib": len(sub_live),
            "s1": recs_s1,
            "site_only": recs_so,
            "stage2": recs_s2,
        }
    return out_by_frac


def main():
    global GLOBAL_ROWS
    t_init = time.time()
    sites_data = load_sites_v2("goal", "hybrid")
    for site in SITES:
        rows = prepare(site, sites_data[site]["singles"])
        for s_obj, row in zip(sites_data[site]["singles"], rows):
            row["family"] = s_obj["label"]["family"]
            row["rw"] = "read" if row["goal"] == "read" else "write"
        precompute_site_matrices(rows)
        GLOBAL_ROWS[site] = rows
    print(f"Precomputed all site matrices in {time.time() - t_init:.1f}s", flush=True)

    seeds = 5
    n_jobs = 24
    out = {"sites": {}}

    for site in SITES:
        rows = GLOBAL_ROWS[site]
        print(f"=== {site} (n={len(rows)}) ===", flush=True)
        site_out = {"n_sessions": len(rows), "targets": {}}

        configs = [
            ("rw_hybrid", "rw", ["read", "write"], "hybrid"),
            ("rw_task", "rw", ["read", "write"], "task"),
            ("goal_hybrid", "goal", ["read", "modify", "create", "identity", "transaction"], "hybrid"),
            ("goal_task", "goal", ["read", "modify", "create", "identity", "transaction"], "task"),
            ("goal_trial", "goal", ["read", "modify", "create", "identity", "transaction"], "trial"),
            ("group_task", "family", sorted(set(r["family"] for r in rows)), "task"),
            ("group_trial", "family", sorted(set(r["family"] for r in rows)), "trial"),
            ("task_trial", "task", sorted(set(r["task"] for r in rows)), "trial"),
        ]

        for cfg_name, target_key, classes, split_mode in configs:
            t0 = time.time()
            folds = make_folds(rows, target_key, split_mode, n_folds=10)
            all_tasks = [(site, fold, target_key, classes, seed) for seed in range(seeds) for fold in folds]
            with Pool(processes=min(len(all_tasks), n_jobs)) as pool:
                fold_outs = pool.map(_fold_worker, all_tasks)

            n_f = len(folds)
            f_runs, n_runs, s2_runs = [], [], []
            f_recs0, n_recs0, s2_recs0 = [], [], []
            for s_idx in range(seeds):
                chunk = fold_outs[s_idx * n_f : (s_idx + 1) * n_f]
                rf = [r for c in chunk for r in c[0]]
                rn = [r for c in chunk for r in c[1]]
                rs2 = [r for c in chunk for r in c[2]]
                f_runs.append(score_records(rf, classes, target_key))
                n_runs.append(score_records(rn, classes, target_key))
                s2_runs.append(score_records(rs2, classes, target_key))
                if s_idx == 0:
                    f_recs0 = [
                        {**{k: v for k, v in r.items() if k not in ("probs", "prior")},
                         "probs": np.round(r["probs"], 3).tolist(),
                         "prior": np.round(r["prior"], 3).tolist()}
                        for r in rf
                    ]
                    n_recs0 = [
                        {**{k: v for k, v in r.items() if k not in ("probs", "prior")},
                         "probs": np.round(r["probs"], 3).tolist(),
                         "prior": np.round(r["prior"], 3).tolist()}
                        for r in rn
                    ]
                    s2_recs0 = [
                        {**{k: v for k, v in r.items() if k not in ("probs", "prior")},
                         "probs": np.round(r["probs"], 3).tolist(),
                         "prior": np.round(r["prior"], 3).tolist()}
                        for r in rs2
                    ]

            y_all = [r[target_key] for r in rows]
            maj_cnt = Counter(y_all).most_common(1)[0][1]
            maj_acc = maj_cnt / len(y_all)

            s1_5 = float(np.mean([r["by_action"]["5"]["acc"] for r in n_runs]))
            s1_12 = float(np.mean([r["by_action"]["12"]["acc"] for r in n_runs]))
            s1_full = float(np.mean([r["by_action"]["full"]["acc"] for r in n_runs]))
            s1_f1 = float(np.mean([r["by_action"]["full"]["macro_f1"] for r in n_runs]))

            s2_5 = float(np.mean([r["by_action"]["5"]["acc"] for r in s2_runs]))
            s2_12 = float(np.mean([r["by_action"]["12"]["acc"] for r in s2_runs]))
            s2_full = float(np.mean([r["by_action"]["full"]["acc"] for r in s2_runs]))
            s2_f1 = float(np.mean([r["by_action"]["full"]["macro_f1"] for r in s2_runs]))

            print(
                f"  {cfg_name:14s} ({len(classes):2d} cls, maj={maj_acc:.3f}) | "
                f"S1(ngram): @5={s1_5:.3f} @12={s1_12:.3f} @full={s1_full:.3f} (f1={s1_f1:.3f}) | "
                f"S2: @5={s2_5:.3f} @12={s2_12:.3f} @full={s2_full:.3f} (f1={s2_f1:.3f}) [{time.time()-t0:.1f}s]",
                flush=True,
            )

            site_out["targets"][cfg_name] = {
                "n_classes": len(classes),
                "classes": classes,
                "majority_acc": maj_acc,
                "forest_runs": f_runs,
                "ngram_runs": n_runs,
                "s2_runs": s2_runs,
                "forest_records_seed0": f_recs0,
                "ngram_records_seed0": n_recs0,
                "s2_records_seed0": s2_recs0,
            }
        out["sites"][site] = site_out

    out_path = REPO / "data" / "results" / "hierarchy_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)

if __name__ == "__main__":
    main()
