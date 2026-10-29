#!/usr/bin/env python3
"""Schema-Adapted calibration curve for the five universal goals.

At each fraction c in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]:
  - s1: Site-Agnostic model trained on the whole training fold
  - so: schema tokens only, trained on the c fraction
  - s2: Site-Agnostic prior plus the schema specialist trained on the c fraction

The default holdout is unseen tasks (group=task). Pass hybrid for the
unseen-subsystem auxiliary split.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]

from causal import (  # noqa: E402
    CLASSES,
    class_weights,
    feature_matrix,
    folds_for,
    macro_f1,
    prefix_rows,
    prepare,
)
from labels_v2 import load_sites_v2  # noqa: E402
from tokens import bigram_vocab, site_vocab  # noqa: E402

FRACTIONS = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
SITES = ["ridgeway", "servicenow", "gitea"]


def stratified_subsample(live: list[dict], target: str, frac: float, seed: int) -> list[dict]:
    if frac >= 0.999:
        return list(live)
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in live:
        by_class[row[target]].append(row)
    sub = []
    for cls, items in sorted(by_class.items()):
        n_total = len(items)
        k = max(1, int(round(frac * n_total)))
        idx = rng.choice(n_total, size=min(k, n_total), replace=False)
        for i in idx:
            sub.append(items[int(i)])
    return sub


def _eval_fold_worker(args):
    fold, rows, target, seed, classes = args
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    held = set(fold)
    train = [row for i, row in enumerate(rows) if i not in held]
    test = [rows[i] for i in fold]
    live = [row for row in train if row["tokens"]]
    labels = [row[target] for row in train]
    prior = np.asarray([labels.count(c) for c in classes], dtype=float)
    prior = prior / prior.sum()

    # 1. Precompute full vocabularies and feature matrices ONCE per row in this fold
    vocab_agn = bigram_vocab([row["tokens"][1.0] for row in live])
    s_vocab_full = site_vocab([row["site_dicts"] for row in live], min_sessions=1)
    weights_full = class_weights([row[target] for row in live], classes)

    agn_x_rows, agn_y_rows, agn_w_rows = [], [], []
    row_pfx_mats = {}
    row_late_mat = {}
    for row in live:
        sid = row["session_id"]
        s_dicts = row["site_dicts"]
        mats = []
        for warp, tok in row["tokens"].items():
            mat_agn = feature_matrix(tok, vocab_agn, None, None)
            mat_full = feature_matrix(tok, vocab_agn, s_dicts, s_vocab_full)
            keep = prefix_rows(len(tok))
            idx_k = np.asarray(keep) - 1
            agn_x_rows.append(mat_agn[idx_k])
            agn_y_rows.extend([row[target]] * len(keep))
            agn_w_rows.extend([weights_full[row[target]] / len(keep) / len(row["tokens"])] * len(keep))
            mats.append((mat_full[idx_k], len(keep)))
            if warp == 1.0:
                row_late_mat[sid] = mat_full[-1:]
        row_pfx_mats[sid] = mats

    x_agn = np.vstack(agn_x_rows)
    w_agn = np.asarray(agn_w_rows) * len(agn_w_rows) / sum(agn_w_rows)
    m_agn = RandomForestClassifier(
        n_estimators=250, max_depth=8, min_samples_leaf=3, max_features="sqrt",
        random_state=seed, n_jobs=1,
    )
    m_agn.fit(x_agn, agn_y_rows, sample_weight=w_agn)
    idx_agn = {label: i for i, label in enumerate(m_agn.classes_)}

    # Precompute test matrices & Stage 1 agnostic predictions ONCE
    agn_test_probs = {}
    test_site_mats = {}
    for row in test:
        if not row["tokens"]:
            continue
        sid = row["session_id"]
        tok = row["tokens"][1.0]
        mat_agn = feature_matrix(tok, vocab_agn, None, None)
        test_site_mats[sid] = feature_matrix(tok, vocab_agn, row["site_dicts"], s_vocab_full)
        p_agn = m_agn.predict_proba(mat_agn)
        full_a = np.zeros((p_agn.shape[0], len(classes)))
        for j, label in enumerate(classes):
            if label in idx_agn:
                full_a[:, j] = p_agn[:, idx_agn[label]]
        agn_test_probs[sid] = full_a

    # 2. For each fraction c in FRACTIONS, fit site-specific models on stratified c% subset
    out_by_frac = {}
    for f_idx, frac in enumerate(FRACTIONS):
        sub_live = stratified_subsample(live, target, frac, seed * 1009 + f_idx * 37 + fold[0])
        sub_labels = [row[target] for row in sub_live]
        if len(set(sub_labels)) < 2:
            continue
        weights_sub = class_weights(sub_labels, classes)

        x_rows, y_rows, w_rows = [], [], []
        late_x_rows, late_y_rows, late_w_rows = [], [], []
        for row in sub_live:
            sid = row["session_id"]
            n_warps = len(row["tokens"])
            for mat_k, n_k in row_pfx_mats[sid]:
                x_rows.append(mat_k)
                y_rows.extend([row[target]] * n_k)
                w_rows.extend([weights_sub[row[target]] / n_k / n_warps] * n_k)
            late_x_rows.append(row_late_mat[sid])
            late_y_rows.append(row[target])
            late_w_rows.append(weights_sub[row[target]])

        x_sub = np.vstack(x_rows)
        w_sub = np.asarray(w_rows) * len(w_rows) / sum(w_rows)
        x_late = np.vstack(late_x_rows)
        w_late = np.asarray(late_w_rows) * len(late_w_rows) / sum(late_w_rows)

        m_site_pfx = ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=1,
        )
        m_site_late1 = ExtraTreesClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed + 7, n_jobs=1,
        )
        m_site_late2 = RandomForestClassifier(
            n_estimators=400, max_depth=14, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed + 17, n_jobs=1,
        )
        m_site_pfx.fit(x_sub, y_rows, sample_weight=w_sub)
        m_site_late1.fit(x_late, late_y_rows, sample_weight=w_late)
        m_site_late2.fit(x_late, late_y_rows, sample_weight=w_late)

        idx_sp = {label: i for i, label in enumerate(m_site_pfx.classes_)}
        idx_sl1 = {label: i for i, label in enumerate(m_site_late1.classes_)}
        idx_sl2 = {label: i for i, label in enumerate(m_site_late2.classes_)}

        w_prior_late = max(0.15, 0.45 * (1.0 - frac))
        w_spec_late = (1.0 - w_prior_late) / 2.0

        recs_s1, recs_site_only, recs_stage2 = [], [], []
        for row in test:
            sid = row["session_id"]
            if not row["tokens"]:
                p_prior = int(np.argmax(prior))
                recs_s1.append((row[target], p_prior, p_prior))
                recs_site_only.append((row[target], p_prior, p_prior))
                recs_stage2.append((row[target], p_prior, p_prior))
                continue
            mat_site = test_site_mats[sid]
            pa_mat = agn_test_probs[sid]
            p_sp = m_site_pfx.predict_proba(mat_site)
            p_sl1 = m_site_late1.predict_proba(mat_site)
            p_sl2 = m_site_late2.predict_proba(mat_site)
            n_steps = pa_mat.shape[0]
            alpha = np.linspace(0.25, 0.85, n_steps)[:, None]

            full_site_only = np.zeros((n_steps, len(classes)))
            full_stage2 = np.zeros((n_steps, len(classes)))
            for j, label in enumerate(classes):
                pa = pa_mat[:, j]
                psp = p_sp[:, idx_sp[label]] if label in idx_sp else np.zeros(n_steps)
                psl1 = p_sl1[:, idx_sl1[label]] if label in idx_sl1 else np.zeros(n_steps)
                psl2 = p_sl2[:, idx_sl2[label]] if label in idx_sl2 else np.zeros(n_steps)

                so_early = psp
                so_late = 0.5 * psl1 + 0.5 * psl2
                full_site_only[:, j] = (1.0 - alpha[:, 0]) * so_early + alpha[:, 0] * so_late

                s2_early = 0.35 * pa + 0.65 * psp
                s2_late = w_prior_late * pa + w_spec_late * psl1 + w_spec_late * psl2
                full_stage2[:, j] = (1.0 - alpha[:, 0]) * s2_early + alpha[:, 0] * s2_late

            k5 = min(5, n_steps) - 1
            recs_s1.append((row[target], int(np.argmax(pa_mat[k5])), int(np.argmax(pa_mat[-1]))))
            recs_site_only.append((row[target], int(np.argmax(full_site_only[k5])), int(np.argmax(full_site_only[-1]))))
            recs_stage2.append((row[target], int(np.argmax(full_stage2[k5])), int(np.argmax(full_stage2[-1]))))

        out_by_frac[str(frac)] = {
            "n_calib": len(sub_live),
            "s1": recs_s1,
            "site_only": recs_site_only,
            "stage2": recs_stage2,
        }
    return out_by_frac


def run_calibration(group: str = "hybrid", seeds: int = 5, n_folds: int = 10, n_jobs: int = 20):
    results = {"fractions": FRACTIONS, "group": group, "seeds": seeds, "targets": {}}
    for target in ("goal",):
        classes = CLASSES[target]
        c_idx = {c: i for i, c in enumerate(classes)}
        sites_data = load_sites_v2(target, group)
        results["targets"][target] = {}
        for site in SITES:
            t0 = time.time()
            rows = prepare(site, sites_data[site]["singles"])
            folds = folds_for(rows, n_folds)
            site_res = {str(f): {"n_calib": [], "s1_acc": [], "s1_f1": [], "so_acc": [], "so_f1": [], "s2_acc": [], "s2_f1": [], "s2_acc5": [], "so_acc5": [], "s1_acc5": []} for f in FRACTIONS}
            all_tasks = [(fold, rows, target, seed, classes) for seed in range(seeds) for fold in folds]
            with Pool(processes=min(len(all_tasks), n_jobs)) as pool:
                all_outs = pool.map(_eval_fold_worker, all_tasks)
            n_f = len(folds)
            for s_idx in range(seeds):
                fold_outs = all_outs[s_idx * n_f : (s_idx + 1) * n_f]
                for f in FRACTIONS:
                    fk = str(f)
                    n_cal = float(np.mean([fo[fk]["n_calib"] for fo in fold_outs]))
                    s1_all = [item for fo in fold_outs for item in fo[fk]["s1"]]
                    so_all = [item for fo in fold_outs for item in fo[fk]["site_only"]]
                    s2_all = [item for fo in fold_outs for item in fo[fk]["stage2"]]

                    y_true = [c_idx[t] for t, _, _ in s1_all]
                    p_s1_5 = [p5 for _, p5, _ in s1_all]
                    p_s1 = [pf for _, _, pf in s1_all]
                    p_so_5 = [p5 for _, p5, _ in so_all]
                    p_so = [pf for _, _, pf in so_all]
                    p_s2_5 = [p5 for _, p5, _ in s2_all]
                    p_s2 = [pf for _, _, pf in s2_all]

                    site_res[fk]["n_calib"].append(n_cal)
                    site_res[fk]["s1_acc"].append(float(np.mean([t == p for t, p in zip(y_true, p_s1)])))
                    site_res[fk]["s1_f1"].append(macro_f1(y_true, p_s1, len(classes)))
                    site_res[fk]["s1_acc5"].append(float(np.mean([t == p for t, p in zip(y_true, p_s1_5)])))

                    site_res[fk]["so_acc"].append(float(np.mean([t == p for t, p in zip(y_true, p_so)])))
                    site_res[fk]["so_f1"].append(macro_f1(y_true, p_so, len(classes)))
                    site_res[fk]["so_acc5"].append(float(np.mean([t == p for t, p in zip(y_true, p_so_5)])))

                    site_res[fk]["s2_acc"].append(float(np.mean([t == p for t, p in zip(y_true, p_s2)])))
                    site_res[fk]["s2_f1"].append(macro_f1(y_true, p_s2, len(classes)))
                    site_res[fk]["s2_acc5"].append(float(np.mean([t == p for t, p in zip(y_true, p_s2_5)])))

            print(f"[{target} / {site}] done in {time.time() - t0:.1f}s", flush=True)
            for f in FRACTIONS:
                fk = str(f)
                d = site_res[fk]
                print(
                    f"  frac={f:4.2f} (n~{np.mean(d['n_calib']):4.1f}) | "
                    f"Agnostic={np.mean(d['s1_acc']):.3f} | "
                    f"SchemaOnly={np.mean(d['so_acc']):.3f} (f1={np.mean(d['so_f1']):.3f}) | "
                    f"Adapted={np.mean(d['s2_acc']):.3f} (f1={np.mean(d['s2_f1']):.3f})",
                    flush=True,
                )
            results["targets"][target][site] = site_res
    out_path = REPO / "data" / "results" / f"calibration_curve_{group}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    grp = sys.argv[1] if len(sys.argv) > 1 else "task"
    run_calibration(group=grp, seeds=5, n_folds=10, n_jobs=20)
