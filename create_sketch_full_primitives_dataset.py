"""The SketchGraphs sketches drawn with every primitive the published sequences hold: lines,
circles and arcs.

The line-and-circle dataset of `create_sketch_dataset.py`, widened by the arc, and written the
same way, to the same files. An arc is recorded by three points on it -- start, halfway, end, read
clockwise on the drawing -- since its two ends alone do not fix the circle it lies on.

Points are skipped rather than refused: a sketch point is a reference for constraints (a hole's
centre, say), draws no stroke, and so leaving it out keeps the record exactly what the drawing
shows. Ellipses and splines never reach here -- SketchGraphs drops every sketch holding one when it
builds the sequences -- so a sketch is kept when every entity other than its points is a line, a
circle or an arc, none of them construction geometry, and there are between `--min-nodes` and
`--max-nodes` of them.

    uv run create_sketch_full_primitives_dataset.py --output-dir ./dataset_sketches_full --canvas-size 256
    uv run create_sketch_full_primitives_dataset.py --output-dir ./dataset_sketches_full_small --limit 1000
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from create_sketch_dataset import (
    SOURCE,
    SPLITS,
    DEFAULT_SEQUENCES_DIR,
    DatasetConfig,
    build_renderer,
    ensure_output_dir,
    validate_args,
    write_split,
)
from generator import Arc, Circle, Element, FinishDrawing, PartLine, PartLineWithId


DEFAULT_OUTPUT_DIR = Path("datasets/sketches_full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the SketchGraphs sketches made of lines, circles and arcs, rendered with StaticRenderer.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequences-dir", type=Path, default=DEFAULT_SEQUENCES_DIR,
                        help="Where the SketchGraphs sequence files are, or are downloaded to (6.2 GB for train).")
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.06,
                        help="Fraction of a side left blank around a sketch.")
    parser.add_argument("--min-nodes", type=int, default=5,
                        help="Fewest drawn entities a kept sketch holds; points do not count.")
    parser.add_argument("--max-nodes", type=int, default=16,
                        help="The published sequences hold at most sixteen entities.")
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep at most this many sketches per split. All of them by default.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the order the drawings are written in.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def arc_extremes(arc) -> List[np.ndarray]:
    """The points that bound an arc: its two ends, and each of the circle's leftmost, rightmost,
    lowest and highest points the arc passes through.

    The arc's own parameters are read the way SketchGraphs reads them: it runs from `startParam`
    to `endParam`, both taken round to [0, 2pi) with the end past the start, as offsets from the
    direction `(xDir, yDir)`, counted clockwise where `clockwise` is set.
    """
    start = arc.startParam % (2 * math.pi)
    end = arc.endParam % (2 * math.pi)
    if start > end:
        end += 2 * math.pi
    base = math.atan2(arc.yDir, arc.xDir)
    sign = -1 if arc.clockwise else 1

    points = [arc.start_point, arc.end_point]
    for quarter in range(4):
        offset = (sign * (quarter * math.pi / 2 - base)) % (2 * math.pi)
        if start <= offset <= end or start <= offset + 2 * math.pi <= end:
            points.append(arc._point_at_angle_offset(offset))
    return points


def placed_actions(sketch, config: DatasetConfig) -> Optional[List[Element]]:
    """The lines, circles and arcs a sketch draws, placed on the canvas, or ``None`` where it
    draws anything else or too little.

    Placed as `create_sketch_dataset.placed_actions` places a sketch, fitted by what it spans; an
    arc spans what its stroke covers, not its whole circle. A sketch is dropped where the fit
    leaves any stroke under a pixel, and where an arc's three points come within a pixel of each
    other, since the circle through them is then not what the drawing shows.
    """
    from sketchgraphs.data._entity import (
        Arc as SketchArc, Circle as SketchCircle, Line as SketchLine, Point as SketchPoint,
    )

    entities = [held for held in sketch.entities.values() if not isinstance(held, SketchPoint)]
    if not config.min_nodes <= len(entities) <= config.max_nodes:
        return None
    if not all(isinstance(held, (SketchLine, SketchCircle, SketchArc)) and not held.isConstruction
               for held in entities):
        return None

    corners = []
    for held in entities:
        if isinstance(held, SketchLine):
            corners.extend((held.start_point, held.end_point))
        elif isinstance(held, SketchArc):
            corners.extend(arc_extremes(held))
        else:
            corners.append((held.xCenter - held.radius, held.yCenter - held.radius))
            corners.append((held.xCenter + held.radius, held.yCenter + held.radius))
    low, high = np.min(corners, axis=0), np.max(corners, axis=0)
    span = float((high - low).max())
    if not np.isfinite(span) or span <= 0:
        return None

    scale, middle = (1 - 2 * config.margin) / span, (low + high) / 2

    def placed(point):
        x, y = (np.asarray(point, dtype=float) - middle) * scale + 0.5
        return float(x), float(1 - y)

    pixels = config.canvas_size - 1
    actions: List[Element] = []
    for index, held in enumerate(entities):
        if isinstance(held, SketchLine):
            line = PartLine(placed(held.start_point), placed(held.end_point))
            if round(line.length * pixels) < 1:
                return None
            actions.append(PartLineWithId(str(index), line))
        elif isinstance(held, SketchArc):
            arc = Arc(placed(held.start_point), placed(held.mid_point), placed(held.end_point))
            ends = (arc.start_point, arc.mid_point, arc.end_point)
            if min(math.dist(ends[i], ends[j]) for i, j in ((0, 1), (1, 2), (0, 2))) * pixels < 1:
                return None
            if arc.circle is None or round(arc.circle[1] * pixels) < 1:
                return None
            actions.append(arc)
        else:
            # Read back through its record, so that the circle drawn is the one the record rebuilds
            # to the pixel: rebuilding the radius from the two points can round it across a half.
            circle = Circle(placed((held.xCenter, held.yCenter)), held.radius * scale)
            circle = Circle.from_params([], circle.coordinates_params, [])
            if round(circle.radius * pixels) < 1:
                return None
            actions.append(circle)
    actions.append(FinishDrawing())
    return actions


def write_metadata(config: DatasetConfig, splits: dict, output_dir: Path, seed: int) -> None:
    metadata = {
        "format": "npy-memmap-v1",
        "task": "line-circle-and-arc-cad-sketch",
        "class_names": ["sketch"],
        "image_dtype": "uint8",
        "image_shape": [config.canvas_size, config.canvas_size],
        "splits": splits,
        "source": {
            "dataset": "SketchGraphs",
            "url": "https://github.com/PrincetonLIPS/SketchGraphs",
            "files": {name: SOURCE % published for name, published in SPLITS.items()},
            "kept": "points skipped, every other entity a Line, a Circle or an Arc, none construction, "
                    f"{config.min_nodes} to {config.max_nodes} of them",
            "deduplicated": "a drawing pixel for pixel one already kept is dropped, across both splits; "
                            "train is read first, so a drawing both hold stays in train",
            "order": f"shuffled with seed {seed}; each label's source names the sketch it came from",
        },
        "primitives": {
            "PartLineWithId": "start and end",
            "Circle": "leftmost and rightmost point",
            "Arc": "start, halfway and end point, clockwise on the drawing (y down)",
        },
        "renderer": {
            "name": "StaticRenderer",
            "thickness": config.thickness,
        },
        "placement": {
            "margin": config.margin,
            "fit": "bounding box of the strokes scaled uniformly into the canvas, y flipped",
        },
        "config": {**asdict(config), "sequences_dir": str(config.sequences_dir)},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = validate_args(args)
    output_dir = args.output_dir
    ensure_output_dir(output_dir, overwrite=args.overwrite)
    renderer = build_renderer(config)

    print(f"Writing dataset to {output_dir}")
    print(f"Keeping sketches of {config.min_nodes} to {config.max_nodes} lines, circles and arcs"
          + (f", at most {config.limit} per split" if config.limit else ""))

    # Train first, so that a drawing train and val both hold stays in train only.
    seen = set()
    splits = {
        name: write_split(name, config, output_dir, renderer, place=placed_actions, seen=seen,
                          shuffle_seed=args.seed)
        for name in ("train", "val")
    }
    write_metadata(config, splits, output_dir, args.seed)

    print("Dataset creation complete.")


if __name__ == "__main__":
    main()
