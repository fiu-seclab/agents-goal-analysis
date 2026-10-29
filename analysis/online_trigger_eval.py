#!/usr/bin/env python3
"""Online stopping rule tau_on(theta, m=2) on the stored five-goal runs.

Site-Agnostic is the n-gram forest. Schema-Adapted is stage2. The logs do not
store the commit step kappa, so the pre-commit query k = kappa - 1 is not
recomputed here. Lead time is the median number of actions after the trigger
on state-changing goals (modification, creation, identity, transaction).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES_DIR = HERE.parent / "data" / "results"
GOALS = ["read", "modify", "create", "identity", "transaction"]
G2I = {g: i for i, g in enumerate(GOALS)}
WRITE_IDX = {1, 2, 3, 4}
HIGH_IDX = {3, 4}


def compute_online_metrics(records_list: list[list[dict]], theta: float = 0.70, m: int = 2) -> dict[str, float]:
    out = []
    for recs in records_list:
        accs, stabs, taus, ws = [], [], [], []
        rw_tp, rw_fp, hi_tp, hi_fp = [], [], [], []
        for r in recs:
            t_idx = G2I[r["true"]]
            probs = np.asarray(r["probs"])
            T = len(probs)
            if T == 0:
                probs = np.asarray(r["prior"])[None, :]
                T = 1
            tau_on = T
            pred_on = int(np.argmax(probs[-1]))
            triggered = False
            streak = 0
            last_c = -1
            for k in range(T):
                c = int(np.argmax(probs[k]))
                conf = float(probs[k, c])
                if (k + 1) >= min(2, T) and conf >= theta:
                    if c == last_c:
                        streak += 1
                    else:
                        last_c = c
                        streak = 1
                    if streak >= min(m, T):
                        tau_on = k + 1
                        pred_on = c
                        triggered = True
                        break
                else:
                    streak = 0
                    last_c = -1
            final_c = int(np.argmax(probs[-1]))
            accs.append(pred_on == t_idx)
            if triggered:
                stabs.append(pred_on == final_c)
            if t_idx in WRITE_IDX:
                taus.append(tau_on)
                ws.append(T - tau_on)
                rw_tp.append(pred_on in WRITE_IDX)
            else:
                rw_fp.append(pred_on in WRITE_IDX)
            if t_idx in HIGH_IDX:
                hi_tp.append(pred_on in HIGH_IDX)
            else:
                hi_fp.append(pred_on in HIGH_IDX)
        out.append({
            "acc": np.mean(accs),
            "stab": np.mean(stabs) if stabs else 1.0,
            "w_on": np.median(ws),
            "rw_tpr": np.mean(rw_tp),
            "rw_fpr": np.mean(rw_fp),
            "hi_tpr": np.mean(hi_tp),
            "hi_fpr": np.mean(hi_fp),
        })
    return {k: float(np.mean([x[k] for x in out])) for k in out[0]}


def main() -> None:
    print("Online rule tau_on(theta, m=2), mean of Storefront, ServiceNow, and Code Forge")
    print("and of Browser Use and Skyvern. Stab. is post-trigger stability.")
    for split_name, bu_fn, sk_fn in [
        ("Seen Tasks", "agent_bu_trial_trees.json", "agent_sk_trial_trees.json"),
        ("Unseen Tasks", "agent_bu_task_trees.json", "agent_sk_task_trees.json"),
    ]:
        bu = json.loads((RES_DIR / bu_fn).read_text())["sites"]
        sk = json.loads((RES_DIR / sk_fn).read_text())["sites"]
        print(f"\n--- {split_name} ---")
        for model, mlabel in [("ngram", "Site-Agnostic n-gram"), ("stage2", "Schema-Adapted")]:
            thetas = [0.70] if model == "ngram" else [0.60, 0.70, 0.80]
            for theta in thetas:
                per_site = []
                for site in ["ridgeway", "servicenow", "gitea"]:
                    m_bu = compute_online_metrics([r["records"] for r in bu[site]["models"][model]], theta, m=2)
                    m_sk = compute_online_metrics([r["records"] for r in sk[site]["models"][model]], theta, m=2)
                    per_site.append({k: 0.5 * (m_bu[k] + m_sk[k]) for k in m_bu})
                avg = {k: float(np.mean([ps[k] for ps in per_site])) for k in per_site[0]}
                print(
                    f"  {mlabel:22s} | theta={theta:.2f} | "
                    f"Acc={avg['acc']:.3f}  Stab={avg['stab']:.3f}  W_on={avg['w_on']:4.1f} | "
                    f"R/W TPR={avg['rw_tpr']:.3f} FPR={avg['rw_fpr']:.3f} | "
                    f"Step-Up TPR={avg['hi_tpr']:.3f} FPR={avg['hi_fpr']:.3f}"
                )


if __name__ == "__main__":
    main()
