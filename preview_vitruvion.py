"""Preview images of the rebuilt Vitruvion dataset, for its dataset card.

Writes two images into the dataset folder:

- `train_first_256.png`: the clean renders of the first 256 training sketches, 16 by 16;
- `train_constraints_4x4.png`: sixteen training sketches drawn from their labels with their
  constraints marked, as a CAD program marks them, since a render shows none of them.

    uv run preview_vitruvion.py --dataset-dir ./dataset_sketches_vitruvion
"""

import argparse
import base64
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc as ArcPatch, Circle as CirclePatch
from PIL import Image

from view_vitruvion import Dataset


#: How each constraint is marked: a glyph and a colour. Coincident points are a dot instead.
MARKS = {
    "Coincident": ("●", "#d1495b"),
    "Horizontal": ("H", "#2c6ea8"),
    "Vertical": ("V", "#2c6ea8"),
    "Parallel": ("∥", "#00a6a6"),
    "Perpendicular": ("⊥", "#7b3fa0"),
    "Tangent": ("T", "#e08e0b"),
    "Equal": ("=", "#3b8b2e"),
    "Midpoint": ("M", "#b5582a"),
    "Concentric": ("◎", "#c2185b"),
    "Fix": ("F", "#555555"),
    "Normal": ("N", "#6d4c41"),
    "Offset": ("O", "#6d4c41"),
    "Quadrant": ("Q", "#6d4c41"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the preview images of the rebuilt Vitruvion dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_sketches_vitruvion"))
    return parser.parse_args()


def render_sheet(dataset: Dataset, out: Path, side: int = 16, pad: int = 4) -> None:
    """The clean renders of the first `side`² training sketches, on a grey grid like the other
    datasets' sheets."""
    size = 128
    sheet = Image.new("L", (side * (size + pad) + pad, side * (size + pad) + pad), color=200)
    for slot in range(side * side):
        sample = dataset.sample("train", slot, None)
        if not sample["renders"]:
            raise SystemExit(f"training sketch {slot} is not rendered yet")
        png = base64.b64decode(sample["renders"][0].split(",", 1)[1])
        row, col = divmod(slot, side)
        sheet.paste(Image.open(io.BytesIO(png)).convert("L"), (pad + col * (size + pad), pad + row * (size + pad)))
    sheet.save(out)
    print(f"wrote {out} ({sheet.size[0]} x {sheet.size[1]})")


def circumcentre(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = a @ a, b @ b, c @ c
    return np.array([(a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
                     (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d])


def anchor(primitive, part):
    """Where a constraint attaches to a primitive: the point it names, or else a point on the
    stroke -- a line's middle, an arc's midpoint, the top of a circle."""
    q = np.asarray(primitive["params"])
    kind = primitive["type"]
    if kind == "Point":
        return q[:2]
    if kind == "Line":
        return {"start": q[:2], "end": q[2:4]}.get(part, (q[:2] + q[2:4]) / 2)
    if kind == "Circle":
        return q[:2] if part == "center" else q[:2] + [0, q[2]]
    if part == "start":
        return q[:2]
    if part == "end":
        return q[4:6]
    if part == "center":
        return circumcentre(q[:2], q[2:4], q[4:6])
    return q[2:4]


def draw_primitive(ax, primitive) -> None:
    q = np.asarray(primitive["params"])
    style = dict(color="#8a7f6e", linestyle=(0, (3, 2)), linewidth=0.9) if primitive["construction"] \
        else dict(color="#161514", linewidth=1.4)
    kind = primitive["type"]
    if kind == "Line":
        ax.plot([q[0], q[2]], [q[1], q[3]], **style, solid_capstyle="round", zorder=2)
    elif kind == "Circle":
        ax.add_patch(CirclePatch(q[:2], q[2], fill=False, **style, zorder=2))
    elif kind == "Arc":
        start, mid, end = q[:2], q[2:4], q[4:6]
        centre = circumcentre(start, mid, end)
        if centre is None:
            ax.plot([start[0], end[0]], [start[1], end[1]], **style, zorder=2)
        else:
            radius = np.linalg.norm(start - centre)
            angle = lambda p: math.degrees(math.atan2(p[1] - centre[1], p[0] - centre[0]))
            ax.add_patch(ArcPatch(centre, 2 * radius, 2 * radius, theta1=angle(start), theta2=angle(end),
                                  **style, zorder=2))   # counter-clockwise from start to end, as the label reads
    else:
        ax.plot(*q[:2], "o", color=style["color"], markersize=2.5, zorder=3)


def draw_constraints(ax, sample) -> set:
    """Marks every constraint, and returns the types it marked."""
    primitives, shown = sample["primitives"], set()
    for constraint in sample["constraints"]:
        glyph, colour = MARKS[constraint["type"]]
        shown.add(constraint["type"])
        points = [anchor(primitives[i], part) for i, part in constraint["refs"]]
        points = [p for p in points if p is not None]
        if not points:
            continue
        if constraint["type"] == "Coincident":
            named = [p for (i, part), p in zip(constraint["refs"], points)
                     if part is not None or primitives[i]["type"] == "Point"]
            for p in named or points[:1]:
                ax.plot(*p, "o", color=colour, markersize=3.2, zorder=4)
            continue
        centre = np.mean(points, axis=0)
        if len(points) > 1 and max(np.linalg.norm(p - centre) for p in points) > 1e-6:
            for p in points:
                ax.plot([centre[0], p[0]], [centre[1], p[1]], color=colour, linewidth=0.7,
                        linestyle=(0, (1.5, 1.5)), alpha=0.85, zorder=3)
        ax.text(*centre, glyph, color=colour, fontsize=8, fontweight="bold", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8))
    return shown


def constraint_sheet(dataset: Dataset, out: Path, side: int = 4) -> None:
    """`side`² training sketches with their constraints marked. They are the first in the training
    split that are small enough for the marks to stay legible and varied enough to show them: 6 to 10
    primitives, an arc or a circle among them, and 6 to 14 constraints of at least 3 types."""
    chosen, position = [], 0
    while len(chosen) < side * side:
        sample = dataset.sample("train", position, None)
        position += 1
        kinds = {p["type"] for p in sample["primitives"]}
        types = {c["type"] for c in sample["constraints"]}
        if (6 <= len(sample["primitives"]) <= 10 and kinds & {"Arc", "Circle"}
                and 6 <= len(sample["constraints"]) <= 14 and len(types) >= 3):
            chosen.append(sample)

    fig, axes = plt.subplots(side, side, figsize=(3 * side, 3 * side + 0.9))
    shown = set()
    for ax, sample in zip(axes.flat, chosen):
        for primitive in sample["primitives"]:
            draw_primitive(ax, primitive)
        shown |= draw_constraints(ax, sample)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#ddd3c5")
        ax.set_title(f"sketch {sample['sketch']} · {len(sample['primitives'])} primitives, "
                     f"{len(sample['constraints'])} constraints", fontsize=7.5, color="#6d665c")

    legend = [Line2D([], [], linestyle="none", marker="o" if kind == "Coincident" else None,
                     color=MARKS[kind][1], markersize=5,
                     label=kind if kind == "Coincident" else f"{MARKS[kind][0]}  {kind}")
              for kind in MARKS if kind in shown]
    legend += [Line2D([], [], color="#8a7f6e", linestyle=(0, (3, 2)), label="construction geometry")]
    fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 6), frameon=False, fontsize=9,
               handlelength=1.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    args = parse_args()
    dataset = Dataset(args.dataset_dir)
    dataset.renders.refresh()
    render_sheet(dataset, args.dataset_dir / "train_first_256.png")
    constraint_sheet(dataset, args.dataset_dir / "train_constraints_4x4.png")


if __name__ == "__main__":
    main()
