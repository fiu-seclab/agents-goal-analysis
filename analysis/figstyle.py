"""Figure style: restrained, print-safe, and matched to the body text.

Palette is Paul Tol's qualitative set, which stays legible in greyscale and under
the common colour-vision deficiencies. Tier colours are the only categorical
encoding; everything else is a single accent against neutrals.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = 3.33   # acmart sigconf column width
WIDE = 7.0   # text width

INK = "#1c1c1c"
GREY = "#5c5c5c"
LIGHT = "#9a9a9a"
HAIR = "#e2e2e2"
PANEL = "#f6f6f4"

ACCENT = "#4477aa"        # Tol blue, the default data colour
ACCENT_SOFT = "#cfdced"
NEUTRAL = "#bbbbbb"

STANDARD = "#117733"      # Tol green
ELEVATED = "#997700"      # Tol olive, darkened for print
CRITICAL = "#aa3355"      # Tol wine
TIER_COLOR = {"standard": STANDARD, "elevated": ELEVATED, "critical": CRITICAL}
TIER_SOFT = {"standard": "#dcebe1", "elevated": "#f0e8cf", "critical": "#f2dde3"}


def init():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "legend.frameon": False,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.45,
        "legend.labelspacing": 0.3,
        "legend.borderpad": 0.2,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GREY,
        "axes.linewidth": 0.5,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": HAIR,
        "grid.linewidth": 0.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=HAIR, linewidth=0.5)


def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, "(%s)" % letter, transform=ax.transAxes, fontsize=7.6,
            fontweight="bold", va="bottom", ha="left", color=INK)


def note(fig, text):
    """No-op: subtitles/notes under figures are omitted in favor of LaTeX captions."""
    pass


def save(fig, path, provisional=False):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.rsplit("/", 1)[-1])


def card(ax, x, y, w, h, title=None, body=None, fill=PANEL, edge=NEUTRAL,
         title_size=7.0, body_size=6.4, lw=0.6, pad=0.012, align="left",
         title_color=None, radius=0.006, gap=0.085):
    """A labelled panel used by the schematic figures."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % radius,
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ha = {"left": "left", "center": "center"}[align]
    tx = x + pad if align == "left" else x + w / 2
    top = y + h - pad
    if title:
        ax.text(tx, top, title, ha=ha, va="top", fontsize=title_size,
                fontweight="bold", color=title_color or INK, zorder=3)
        top -= gap
    if body:
        ax.text(tx, top, body, ha=ha, va="top", fontsize=body_size,
                color=GREY, zorder=3, linespacing=1.35)


def points_axes(fig, rect):
    """An axes whose data coordinates are typographic points.

    Schematic boxes have to be sized from the type they hold, which is specified
    in points, so the schematics work in points rather than in axes fractions.
    `rect` is the usual figure-fraction rectangle.
    """
    ax = fig.add_axes(rect)
    w_in, h_in = fig.get_size_inches()
    ax.set_xlim(0, rect[2] * w_in * 72.0)
    ax.set_ylim(0, rect[3] * h_in * 72.0)
    ax.axis("off")
    return ax


def card_pt(ax, x, y_top, w, h, title=None, body=None, fill=PANEL,
            edge=NEUTRAL, title_size=7.0, body_size=6.2, lw=0.75, pad=4.0,
            align="left", title_color=None, body_color=None, radius=2.0,
            lead=1.30):
    """A labelled box on a `points_axes`, anchored by its top edge, with text vertically centered."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y_top - h), w, h,
        boxstyle="round,pad=0,rounding_size=%g" % radius,
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2, clip_on=False))
    centred = align == "center"
    tx = x + w / 2 if centred else x + pad + 1.0
    ha = "center" if centred else "left"
    y_mid = y_top - h / 2.0
    if title and body:
        n_body = len(body.split("\n"))
        step = body_size * lead
        total_span = title_size * 0.95 + 2.5 + (n_body - 1) * step + body_size * 0.85
        top_y = y_mid + total_span / 2.0
        ax.text(tx, top_y, title, ha=ha, va="top", fontsize=title_size,
                fontweight="bold", color=title_color or INK, zorder=3)
        body_top = top_y - title_size * 0.95 - 2.5
        ax.text(tx, body_top, body, ha=ha, va="top", fontsize=body_size,
                color=body_color or GREY, zorder=3, linespacing=lead)
    elif title:
        ax.text(tx, y_mid, title, ha=ha, va="center", fontsize=title_size,
                fontweight="bold", color=title_color or INK, zorder=3)
    elif body:
        ax.text(tx, y_mid, body, ha=ha, va="center", fontsize=body_size,
                color=body_color or GREY, zorder=3, linespacing=lead)


def box_height(n_body_lines, title=True, title_size=7.0, body_size=6.2,
               pad=4.0, lead=1.30):
    """Height in points of a `card_pt` holding the given number of lines."""
    h = 2 * pad
    if title:
        h += title_size * lead
    if n_body_lines:
        h += body_size * lead * n_body_lines
    return h


def arrow(ax, xy_from, xy_to, color=GREY, lw=0.7, ls="-", head=6.0,
          connect=None):
    from matplotlib.patches import FancyArrowPatch
    kw = dict(arrowstyle="-|>", mutation_scale=head, linewidth=lw, color=color,
              linestyle=ls, zorder=1, shrinkA=0, shrinkB=0)
    if connect:
        kw["connectionstyle"] = connect
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, **kw))


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
