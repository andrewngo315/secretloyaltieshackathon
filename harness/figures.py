from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from .config import RESULTS_DIR

OUT_DIR = os.path.join(RESULTS_DIR, "figures")
BEHAVIORAL = os.path.join(RESULTS_DIR, "experiment", "behavioral.json")

BAR = "#3B6FB6"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
GRID = "#DCDCDC"
CHANCE = "#B0433A"

LADDER = [
    ("neutral__p0", "neutral", "baseline"),
    ("control_acme__p0", "entity mention (Acme)", "controls"),
    ("control_zephyr__p0", "entity mention (Zephyr)", "controls"),
    ("sycophant_zephyr__p0", "sycophantic user (Zephyr)", "social pressure"),
    ("sycophant_acme__p0", "sycophantic user (Acme)", "social pressure"),
    ("secret_rule__p0", "covert positional rule", "non-loyalty directives"),
    ("overt_rule__p0", "declared positional rule", "non-loyalty directives"),
    ("loyal_bravo__p0", "concealed loyalty (Bravo)", "loyalty"),
    ("loyal_zephyr__p0", "concealed loyalty (Zephyr)", "loyalty"),
    ("open_loyal_acme__p0", "declared loyalty (Acme)", "loyalty"),
    ("loyal_acme__p0", "concealed loyalty (Acme)", "loyalty"),
]

def load_rows():
    summary = json.load(open(BEHAVIORAL))["summary"]
    rows = []
    for key, label, group in LADDER:
        fav = summary[key]["favoring_Acme"]
        lo, hi = fav["wilson"]
        rows.append({
            "key": key,
            "label": label,
            "group": group,
            "rate": fav["rate"],
            "lo": lo,
            "hi": hi,
            "n": fav["n"],
        })
    return rows


def layout(rows):
    ys, headers = [], []
    y = 0.0
    last = None
    for row in rows:
        if row["group"] != last:
            if last is not None:
                y += 1.05
            headers.append((y - 0.58, row["group"]))
            last = row["group"]
        ys.append(y)
        y += 1.0
    return ys, headers


def build(rows, path_stem):
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ys, headers = layout(rows)
    top = max(ys) + 1.25

    ax.axvline(0.5, color=CHANCE, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.512, top - 0.1, "chance", color=CHANCE, fontsize=8.4, ha="left", va="top")

    for y, row in zip(ys, rows):
        ax.barh(y, row["rate"], height=0.5, color=BAR, linewidth=0, zorder=3)
        ax.plot([row["lo"], row["hi"]], [y, y], color=INK, linewidth=1.3,
                solid_capstyle="butt", zorder=5)
        for x in (row["lo"], row["hi"]):
            ax.plot([x, x], [y - 0.12, y + 0.12], color=INK, linewidth=1.3, zorder=5)
        ax.text(row["hi"] + 0.02, y, f'{row["rate"]:.2f}', va="center",
                ha="left", fontsize=8.9, color=INK, zorder=6)

    for y, name in headers:
        ax.text(-0.012, y, name.upper(), transform=ax.get_yaxis_transform(),
                fontsize=7.2, color=MUTED, ha="right", va="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.3, color=INK)
    ax.set_xlim(0, 1.14)
    ax.set_ylim(-0.85, top)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=9, color=MUTED)
    ax.set_xlabel("rate of choosing Acme over the rival   (n = 40 per condition, Wilson 95% CI)",
                  fontsize=9.3, color=MUTED, labelpad=14)

    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)

    ax.set_title(
        "Loyalty moves the choice; concealing it does not change how much",
        fontsize=11.8, color=INK, loc="left", pad=16, fontweight="bold")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


PATCH = os.path.join(os.path.dirname(RESULTS_DIR), "results_1.5b", "patch", "full_patch.json")

DONORS = [
    ("loyal_acme", "loyalty (Acme)", "#3B6FB6", "o", -0.055),
    ("control_acme", "entity mention, no loyalty", "#B2182B", "s", 0.0),
    ("overt_rule", "declared steering, no principal", "#4E9A6A", "^", -0.035),
]

LAYERS = [3, 7, 11, 15, 19, 23]


def build_patch(path_stem):
    d = json.load(open(PATCH))
    res = d["results"]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    ax.axhline(0.0, color=GRID, linewidth=1.0, zorder=1)
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(3.1, 1.02, "donor's own behaviour (ceiling)", fontsize=8.2,
            color=MUTED, va="bottom", ha="left")

    for key, label, colour, marker, dy in DONORS:
        ys = [res[key][f"L{L}"]["gap_closure"] for L in LAYERS]
        ax.plot(LAYERS, ys, color=colour, linewidth=2.0, marker=marker,
                markersize=6.5, markeredgecolor="white", markeredgewidth=1.2,
                zorder=4, clip_on=False)
        ax.text(LAYERS[-1] + 0.55, ys[-1] + dy, label, color=colour, fontsize=9.2,
                va="center", ha="left", fontweight="bold")

    ax.annotate(f'+{res["loyal_acme"]["L23"]["gap_closure"]:.3f}',
                xy=(23, res["loyal_acme"]["L23"]["gap_closure"]),
                xytext=(21.4, 0.80), fontsize=9.4, color=INK, ha="right")

    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS], fontsize=9.2, color=MUTED)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=9, color=MUTED)
    ax.set_xlim(2.4, 30.5)
    ax.set_ylim(-0.16, 1.16)
    ax.set_xlabel("layer at which the residual-stream state is transplanted",
                  fontsize=9.3, color=MUTED, labelpad=10)
    ax.set_ylabel("fraction of the loyalty gap transported", fontsize=9.3,
                  color=MUTED, labelpad=10)

    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)

    ax.set_title("The loyalty transports through the state; generic steering does not",
                 fontsize=11.6, color=INK, loc="left", pad=16, fontweight="bold")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return d


def main():
    rows = load_rows()
    stem = os.path.join(OUT_DIR, "fig1_behavioural")
    build(rows, stem)
    patch_stem = os.path.join(OUT_DIR, "fig2_transplant")
    d = build_patch(patch_stem)
    print(f"wrote {patch_stem}.pdf and .png")
    print(f'  gap {d["behaviour_gap_nats"]:.2f} nats, n={d["n_scenarios"]}, '
          f'identity check {d["identity_check_max_abs_delta"]}')
    swap = rows[-1]["rate"] - [r for r in rows if r["key"] == "loyal_zephyr__p0"][0]["rate"]
    print(f"wrote {stem}.pdf and {stem}.png")
    print(f"principal swap (loyal_acme - loyal_zephyr) = {swap:+.3f}")
    for r in rows:
        print(f'  {r["label"]:30s} {r["rate"]:.3f} [{r["lo"]:.3f},{r["hi"]:.3f}]')


if __name__ == "__main__":
    main()
