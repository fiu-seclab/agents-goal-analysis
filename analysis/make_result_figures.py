#!/usr/bin/env python3
"""Figures that appear in the paper, drawn from data/results.

    python3 analysis/make_result_figures.py [outdir]

  fig_goal.pdf         prefix accuracy of the four Site-Agnostic models (seen tasks)
  fig_confusion.pdf    full-session confusion, Site-Agnostic forest vs Schema-Adapted
  fig_calibration.pdf  Schema-Adapted accuracy against the unseen-task calibration budget
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

fs.init()

RES = os.path.abspath(os.path.join(HERE, "..", "data", "results"))
CAL_TSK_PATH = os.path.join(RES, "calibration_curve_task.json")

SITES = ["ridgeway", "servicenow", "gitea"]
SITE_NAME = {"ridgeway": "Storefront", "servicenow": "ServiceNow", "gitea": "Code Forge"}
GOALS = ["read", "modify", "create", "identity", "transaction"]
GOAL_SHORT = {
    "read": "retrieval",
    "modify": "modification",
    "create": "creation",
    "identity": "identity",
    "transaction": "transaction",
}
GRID = ["1", "2", "3", "5", "8", "12", "20", "32", "50", "80", "full"]
MODEL_STYLE = {
    "forest": dict(color=fs.ACCENT, marker="o", ls="-"),
    "ngram": dict(color="#66ccee", marker="s", ls="-"),
    "lstm": dict(color="#ccbb44", marker="^", ls="-"),
    "transformer": dict(color="#ee6677", marker="D", ls="-"),
    "stage2": dict(color="#228833", marker="*", ls="-"),
}
MODEL_NAME = {
    "forest": "Agnostic forest",
    "ngram": "Agnostic n-gram",
    "lstm": "Agnostic LSTM",
    "transformer": "Agnostic transformer",
    "stage2": "Schema-Adapted",
}


def save(fig, out, name):
    fig.savefig(os.path.join(out, name))
    if name.endswith(".pdf"):
        fig.savefig(os.path.join(out, name[:-4] + ".png"), dpi=220)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------- calibration
def fig_calibration(out):
    """Single-row learning curves showing how Schema-Adapted universal goal recognition
    improves across target-site calibration budgets (10% to 100% of training pool).
    """
    if not os.path.exists(CAL_TSK_PATH):
        print("skip fig_calibration: json missing")
        return
    cal_tsk = json.load(open(CAL_TSK_PATH))
    fracs = cal_tsk["fractions"]
    xs = np.array([f * 100.0 for f in fracs])

    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDE, 2.25), sharey=False)
    fig.subplots_adjust(left=0.070, right=0.995, top=0.86, bottom=0.22,
                        wspace=0.18)

    def get_stats(d, key):
        mat = np.array([d[str(f)][key] for f in fracs])
        return mat.mean(1), mat.min(1), mat.max(1)

    for j, site in enumerate(SITES):
        ax = axes[j]
        fs.grid(ax)
        d_tsk = cal_tsk["targets"]["goal"][site]
        n_full = int(round(np.mean(d_tsk["1.0"]["n_calib"])))
        n_30 = int(round(np.mean(d_tsk["0.3"]["n_calib"])))
        n_10 = int(round(np.mean(d_tsk["0.1"]["n_calib"])))
        # Unseen-task holdout. s1 is the Site-Agnostic model trained on the full fold.
        ceiling = float(np.mean(d_tsk["1.0"]["s1_acc"]))
        adapted_m, adapted_lo, adapted_hi = get_stats(d_tsk, "s2_acc")
        ax.axhline(ceiling, color=fs.ACCENT, lw=0.85, ls=(0, (3, 2)), alpha=0.80, zorder=2)
        ax.plot(xs, adapted_m, color="#228833", lw=1.35, ls="-", marker="s",
                markersize=3.2, zorder=5, label="Schema-Adapted (unseen tasks)")
        ax.fill_between(xs, adapted_lo, adapted_hi, color="#228833", alpha=0.14, lw=0, zorder=2)
        ax.text(102, ceiling - 0.014, "Site-Agnostic (%.2f)" % ceiling,
                fontsize=5.0, color=fs.ACCENT, ha="right", va="top")

        ax.set_xlim(6, 104)
        ax.set_ylim(0.38, 1.03)
        ax.set_xticks([10, 20, 30, 50, 80, 100])
        ax.set_xticklabels(
            ["10%%\n(%d)" % n_10, "20%", "30%%\n(%d)" % n_30, "50%", "80%", "100%%\n(%d)" % n_full],
            fontsize=5.5,
        )
        ax.set_xlabel("Schema-Adapted calibration budget (% and # sessions)", fontsize=6.6, labelpad=2.0)
        if j == 0:
            ax.set_ylabel("universal goal acc.", fontsize=6.8)
            ax.legend(loc="lower right", fontsize=4.9, ncol=1, framealpha=0.92)
        ax.set_title(SITE_NAME[site], fontsize=6.8, loc="left")

    save(fig, out, "fig_calibration.pdf")


# ---------------------------------------------------------------- accuracy
def fig_goal(data, out):
    """Real-time observed-action prefix accuracy across the four Site-Agnostic models:
    - Top row: Within-agent average (BU & Skyvern) across the 4 Site-Agnostic models on 5 Universal Goals
    - Bottom strip: Fraction of sessions reaching at least k actions
    """
    fig = plt.figure(figsize=(fs.WIDE, 2.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.6, 0.72], hspace=0.18,
                          wspace=0.16, left=0.07, right=0.995, top=0.90,
                          bottom=0.16)
    xs = np.arange(len(GRID))

    bu_tr_t = json.load(open(os.path.join(RES, "agent_bu_trial_trees.json")))["sites"]
    bu_tr_n = json.load(open(os.path.join(RES, "agent_bu_trial_neural.json")))["sites"]
    sk_tr_t = json.load(open(os.path.join(RES, "agent_sk_trial_trees.json")))["sites"]
    sk_tr_n = json.load(open(os.path.join(RES, "agent_sk_trial_neural.json")))["sites"]

    for j, site in enumerate(SITES):
        maj_g = 0.5 * (
            bu_tr_t[site]["majority"]["accuracy"] + sk_tr_t[site]["majority"]["accuracy"]
        )

        # Row 0: Within-agent average (BU & Skyvern) across all 4 Site-Agnostic architectures
        ax = fig.add_subplot(gs[0, j])
        fs.grid(ax)
        ax.axhline(maj_g, color=fs.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for model in ["forest", "ngram", "lstm", "transformer"]:
            runs_bu = bu_tr_t[site]["models"][model] if model in ("forest", "ngram") else bu_tr_n[site]["models"][model]
            runs_sk = sk_tr_t[site]["models"][model] if model in ("forest", "ngram") else sk_tr_n[site]["models"][model]
            acc_bu = np.array([[r["scores"]["by_action"][k]["acc"] for k in GRID] for r in runs_bu])
            acc_sk = np.array([[r["scores"]["by_action"][k]["acc"] for k in GRID] for r in runs_sk])
            acc = 0.5 * (acc_bu + acc_sk)
            st = MODEL_STYLE[model]
            ax.plot(xs, acc.mean(0), lw=1.2, markersize=2.8, zorder=3,
                    label=MODEL_NAME[model], **st)
            if len(acc) > 1:
                ax.fill_between(xs, acc.min(0), acc.max(0), color=st["color"],
                                alpha=0.14, lw=0, zorder=2)
        ax.set_ylim(0, 1.03)
        ax.set_xlim(-0.3, len(GRID) - 0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels([])
        ax.set_title(SITE_NAME[site], fontsize=6.8, loc="left")
        if j == 0:
            ax.set_ylabel("universal goal acc.")
            ax.legend(loc="lower right", fontsize=5.3, ncol=1, framealpha=0.92)
        else:
            ax.set_yticklabels([])
        ax.text(len(GRID) - 1.0, maj_g + 0.015,
                "majority", fontsize=5.5, color=fs.GREY, ha="right", va="bottom")

        # Row 1: Reached fraction strip across both agents
        ax2 = fig.add_subplot(gs[1, j])
        recs = bu_tr_t[site]["models"]["forest"][0]["records"] + sk_tr_t[site]["models"]["forest"][0]["records"]
        n_act = np.array([r["n_actions"] for r in recs])
        reach = [1.0 if k == "full" else float(np.mean(n_act >= int(k))) for k in GRID]
        ax2.bar(xs, reach, color=fs.NEUTRAL, width=0.7, lw=0)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(-0.3, len(GRID) - 0.7)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(GRID)
        ax2.set_yticks([0, 1])
        if j == 0:
            ax2.set_ylabel("reached", fontsize=6.4)
        else:
            ax2.set_yticklabels([])
        ax2.set_xlabel("actions observed ($k$)", labelpad=1.5, fontsize=6.8)
    save(fig, out, "fig_goal.pdf")


def pred_full(rec):
    if len(rec["probs"]):
        return int(np.argmax(rec["probs"][-1]))
    return int(np.argmax(rec["prior"]))


def fig_confusion(data, out):
    """5-Goal confusion on the whole session, Site-Agnostic forest (top) vs Schema-Adapted (bottom), pooled over seeds and agents."""
    fig, axes = plt.subplots(2, 3, figsize=(fs.WIDE, 3.85))
    fig.subplots_adjust(left=0.095, right=0.995, top=0.91, bottom=0.13,
                        wspace=0.28, hspace=0.34)
    idx = {g: i for i, g in enumerate(GOALS)}
    K = len(GOALS)
    cmap = plt.get_cmap("Blues")
    bu_tr_t = json.load(open(os.path.join(RES, "agent_bu_trial_trees.json")))["sites"]
    sk_tr_t = json.load(open(os.path.join(RES, "agent_sk_trial_trees.json")))["sites"]
    for r_idx, (mkey, mlabel) in enumerate([("forest", "Site-Agnostic"), ("stage2", "Schema-Adapted")]):
        for j, site in enumerate(SITES):
            ax = axes[r_idx, j]
            runs_bu = bu_tr_t[site]["models"][mkey]
            runs_sk = sk_tr_t[site]["models"][mkey]
            cm = np.zeros((K, K))
            for run in runs_bu + runs_sk:
                for rec in run["records"]:
                    cm[idx[rec["true"]], pred_full(rec)] += 1
            rows = cm.sum(1, keepdims=True)
            norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
            for i in range(K):
                for k in range(K):
                    ax.add_patch(Rectangle((k - 0.5, i - 0.5), 1, 1,
                                           facecolor=cmap(0.04 + 0.96 * norm[i, k]),
                                           edgecolor="white", lw=0.7))
            ax.set_xlim(-0.5, K - 0.5)
            ax.set_ylim(K - 0.5, -0.5)
            ax.set_aspect("equal")
            for i in range(K):
                for k in range(K):
                    col = "white" if norm[i, k] > 0.58 else fs.INK
                    val_str = "%.2f" % norm[i, k] if norm[i, k] >= 0.01 else "0"
                    ax.text(k, i, val_str, ha="center", va="center",
                            fontsize=5.5, color=col)
            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            if r_idx == 1:
                ax.set_xticklabels([GOAL_SHORT[g] for g in GOALS], fontsize=5.6, rotation=25, ha="right")
                ax.set_xlabel("predicted universal goal", fontsize=6.5, labelpad=1.5)
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels([GOAL_SHORT[g] for g in GOALS], fontsize=5.6)
            if j == 0:
                ax.set_ylabel("%s\ntrue goal" % mlabel, fontsize=6.3)
            ax.set_title(SITE_NAME[site], fontsize=6.5, loc="left")
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
    save(fig, out, "fig_confusion.pdf")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures")
    os.makedirs(out, exist_ok=True)
    fig_goal(None, out)
    fig_confusion(None, out)
    fig_calibration(out)
