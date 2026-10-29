#!/usr/bin/env python3
"""Schematic figures for the causal goal recognition pipeline.

    python3 analysis/make_figures.py [outdir]

fig_overview shows:
  (a) The scope of the work: browser agent (upstream, assumed), in-page
      collector and goal analyzer (this paper), backend (downstream consumer).
  (b) Goal confidence on one session as actions accumulate (recognition
      point tau, commit step kappa, trigger window W = kappa - tau).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

fs.init()

CORE_FILL = "#edf2f9"
SCOPE_GREY = "#8a8a8a"
DASH = (0, (2.5, 1.5))

EXAMPLE = {
    "actions": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
    "conf": [0.25, 0.34, 0.48, 0.64, 0.79, 0.91, 0.95, 0.97, 0.98, 0.98,
             0.99, 0.99, 0.99, 0.99],
    "threshold": 0.85,
    "kappa": 28,
}


def box_pt(ax, x, y_top, w, h, title, sub=None, fill="#ffffff", edge=fs.GREY,
           title_color=None, sub_color=None, lw=0.8, radius=2.5,
           title_size=7.0, sub_size=6.0, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y_top - h), w, h,
        boxstyle="round,pad=0,rounding_size=%g" % radius,
        linewidth=lw, edgecolor=edge, facecolor=fill, ls=ls, zorder=2,
        clip_on=False))
    cx = x + w / 2.0
    yc = y_top - h / 2.0
    if not sub:
        ax.text(cx, yc, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=title_color or fs.INK, zorder=3)
        return
    n_sub = len(sub.split("\n"))
    lead = sub_size * 1.25
    span = title_size + 2.5 + n_sub * lead
    top = yc + span / 2.0
    ax.text(cx, top, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=title_color or fs.INK, zorder=3)
    ax.text(cx, top - title_size - 2.5, sub, ha="center", va="top",
            fontsize=sub_size, color=sub_color or fs.GREY, zorder=3,
            linespacing=1.25)


def workflow(fig, rect):
    """Existing web stack in grey; what this paper adds in blue."""
    ax = fs.points_axes(fig, rect)
    w_in, h_in = fig.get_size_inches()
    W = rect[2] * w_in * 72.0
    H = rect[3] * h_in * 72.0

    ax.text(0, H, "(a) Deployment in a web application",
            ha="left", va="top", fontsize=8.0, fontweight="bold", color=fs.INK)

    # Legend (highlight only our added components).
    x = W - 52.0
    ax.add_patch(FancyBboxPatch(
        (x, H - 7.5), 7.0, 6.0, boxstyle="round,pad=0,rounding_size=1",
        lw=0.8, edgecolor=fs.ACCENT, facecolor=CORE_FILL))
    ax.text(x + 9.5, H - 4.5, "this work", ha="left", va="center", fontsize=5.8,
            color=fs.ACCENT, fontweight="bold")

    def zone(lo, hi, tag):
        ax.add_patch(FancyBboxPatch(
            (0, lo), W - 1.0, hi - lo, boxstyle="round,pad=0,rounding_size=3",
            lw=0.5, edgecolor=fs.NEUTRAL, facecolor="#fafaf8", zorder=0))
        ax.text(4.0, hi - 3.0, tag, ha="left", va="top", fontsize=5.6,
                fontweight="bold", color=SCOPE_GREY, zorder=3)

    cz_lo, cz_hi = 54.0, H - 13.0
    sz_lo, sz_hi = 1.0, 47.0
    zone(cz_lo, cz_hi, "CLIENT: browser")
    zone(sz_lo, sz_hi, "SERVER: web application")

    def existing(x0, x1, top, h, title, sub):
        box_pt(ax, x0, top, x1 - x0, h, title=title, sub=sub,
               title_size=6.4, sub_size=5.4)

    def added(x0, x1, top, h, title, sub):
        box_pt(ax, x0, top, x1 - x0, h, title=title, sub=sub, fill=CORE_FILL,
               edge=fs.ACCENT, title_color=fs.ACCENT, sub_color=fs.INK,
               lw=1.1, title_size=6.4, sub_size=5.4)

    xa0, xa1 = 6.0, 78.0       # agent / controls
    xb0, xb1 = 100.0, 192.0    # page / backend
    xc0, xc1 = 214.0, W - 7.0  # collector / analyzer
    ctop, ch = cz_hi - 11.0, cz_hi - 11.0 - cz_lo - 4.0
    stop, sh = sz_hi - 11.0, sz_hi - 11.0 - sz_lo - 4.0

    existing(xa0, xa1, ctop, ch, "Browser agent",
             "delegated task;\nflagged by agent detection")
    # The page hosts the collector, so the collector sits inside it.
    ax.add_patch(FancyBboxPatch(
        (xb0, ctop - ch), xc1 + 3.0 - xb0, ch,
        boxstyle="round,pad=0,rounding_size=2.5", lw=0.8, edgecolor=fs.GREY,
        facecolor="#ffffff", zorder=1))
    ax.text((xb0 + xb1) / 2.0, ctop - ch / 2.0 + 3.5, "Web page", ha="center",
            va="center", fontsize=6.4, fontweight="bold", color=fs.INK, zorder=3)
    ax.text((xb0 + xb1) / 2.0, ctop - ch / 2.0 - 4.5, "what the agent sees",
            ha="center", va="center", fontsize=5.4, color=fs.GREY, zorder=3)
    added(xc0, xc1, ctop - 2.5, ch - 5.0, "In-Page Collector",
          "records the agent's actions")

    existing(xa0, xa1, stop, sh, "Controls",
             "rate limit, audit log,\nCAPTCHA, step-up")
    existing(xb0, xb1, stop, sh, "App backend",
             "serves pages;\ncommits requests")
    added(xc0, xc1, stop, sh, "Goal Analyzer",
          "estimates the session's\ngoal after each action")

    # Existing flows.
    cm = ctop - ch / 2.0
    fs.arrow(ax, (xa1 + 1.5, cm), (xb0 - 1.5, cm), color=fs.GREY, lw=0.9, head=4.5)
    ax.text((xa1 + xb0) / 2.0, cm + 2.0, "acts", ha="center", va="bottom",
            fontsize=5.4, style="italic", color=fs.GREY)
    xh = (xb0 + xb1) / 2.0
    fs.arrow(ax, (xh, ctop - ch - 1.0), (xh, stop + 1.0), color=fs.GREY,
             lw=0.9, head=4.5)
    ax.text(xh + 3.0, (ctop - ch + stop) / 2.0, "HTTP requests", ha="left",
            va="center", fontsize=5.4, style="italic", color=fs.GREY, zorder=3,
            bbox=dict(boxstyle="square,pad=0.1", fc="#ffffff", ec="none"))
    sm = stop - sh / 2.0
    fs.arrow(ax, (xb0 - 1.5, sm), (xa1 + 1.5, sm), color=fs.GREY, lw=0.9, head=4.5)
    ax.text((xa1 + xb0) / 2.0, sm + 2.0, "trigger", ha="center", va="bottom",
            fontsize=5.4, style="italic", color=fs.GREY)

    # Added flows.
    xs = (xc0 + xc1) / 2.0
    fs.arrow(ax, (xs, ctop - ch + 2.0), (xs, stop + 1.0), color=fs.ACCENT,
             lw=1.1, head=5.0)
    ax.text(xs + 3.0, (ctop - ch + stop) / 2.0 + 1.0, "action trace",
            ha="left", va="center", fontsize=5.6, color=fs.ACCENT, zorder=3,
            bbox=dict(boxstyle="square,pad=0.1", fc="#ffffff", ec="none"))
    yq, yr = sm + 5.0, sm - 5.0
    fs.arrow(ax, (xb1 + 1.5, yq), (xc0 - 1.5, yq), color=fs.ACCENT, lw=0.9,
             ls=(0, (2.2, 1.3)), head=4.2)
    fs.arrow(ax, (xc0 - 1.5, yr), (xb1 + 1.5, yr), color=fs.ACCENT, lw=1.1,
             head=4.5)
    ax.text((xb1 + xc0) / 2.0, yq + 1.5, "query", ha="center", va="bottom",
            fontsize=5.4, color=fs.ACCENT)
    ax.text((xb1 + xc0) / 2.0, yr - 1.5, "goal", ha="center", va="top",
            fontsize=5.4, fontweight="bold", color=fs.ACCENT)


def timeline(ax):
    ex = EXAMPLE
    acts = np.array(ex["actions"], dtype=float)
    conf = np.array(ex["conf"], dtype=float)
    p = ex["threshold"]
    tau = float(acts[np.argmax(conf >= p)])
    kappa = float(ex["kappa"])

    ax.axvspan(tau, kappa, color="#f2f2ef", zorder=0, lw=0)
    fs.grid(ax)
    ax.plot(acts, conf, color=fs.ACCENT, lw=1.3, marker="o", markersize=2.6,
            zorder=4)
    ax.axhline(p, color=fs.GREY, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax.text(1.0, p + 0.03, "threshold", fontsize=5.6, color=fs.GREY,
            va="bottom", ha="left")

    ax.axvline(tau, color=fs.ACCENT, lw=0.8, ls=(0, (2, 1.5)), zorder=3)
    ax.axvline(kappa, color=fs.INK, lw=0.8, ls=(0, (2, 1.5)), zorder=3)
    ax.text(tau + 0.6, 0.40, "$\\tau$", ha="left", va="center", fontsize=7.0,
            color=fs.ACCENT, fontweight="bold")
    ax.text(kappa - 0.6, 0.40, "$\\kappa$", ha="right", va="center",
            fontsize=7.0, color=fs.INK, fontweight="bold")

    yw = 0.17
    ax.add_patch(FancyArrowPatch((tau + 0.3, yw), (kappa - 0.3, yw),
                                 arrowstyle="<|-|>", mutation_scale=5.0,
                                 lw=0.8, color=fs.INK, shrinkA=0, shrinkB=0,
                                 zorder=4))
    ax.text((tau + kappa) / 2, yw + 0.04, "$W = \\kappa - \\tau$",
            fontsize=6.4, color=fs.INK, ha="center", va="bottom")

    ax.set_xticks(acts[1::2])
    ax.tick_params(axis="both", labelsize=6.0)
    ax.set_xlim(0, acts[-1] + 2.0)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("goal confidence", fontsize=6.6, labelpad=2)
    ax.set_xlabel("actions observed ($k$)", fontsize=6.6, labelpad=1.5)
    ax.set_title("(b) Pre-commit trigger window", fontsize=8.0,
                 fontweight="bold", loc="left", pad=4.0)


def fig_overview(out):
    fig = plt.figure(figsize=(fs.WIDE, 1.60))
    workflow(fig, [0.0, 0.0, 0.64, 1.0])
    timeline(fig.add_axes([0.725, 0.22, 0.27, 0.64]))

    out_pdf = os.path.join(out, "fig_overview.pdf")
    fig.savefig(out_pdf)
    fig.savefig(out_pdf[:-4] + ".png", dpi=220)
    plt.close(fig)
    print("wrote fig_overview.pdf")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(out, exist_ok=True)
    fig_overview(out)
