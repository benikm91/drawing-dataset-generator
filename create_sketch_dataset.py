"""The SketchGraphs sketches that draw nothing but lines and circles, as a drawing dataset.

Not generated but taken: SketchGraphs (https://github.com/PrincetonLIPS/SketchGraphs) publishes
real CAD sketches as construction sequences, and this keeps the ones a line-and-circle vocabulary
can say, places each on the canvas, and draws it with the renderer the generated datasets use. What
comes out is what `create_dataset.py` writes -- `{split}_images.npy`, `{split}_labels.jsonl`,
`metadata.json` -- so the same loader reads it.

A sketch is kept only when every entity it holds is a line or a circle, none of them construction
geometry, and there are between `--min-nodes` and `--max-nodes` of them. A stray point, an arc, or
a dashed construction line drops the whole sketch, so that what is drawn is exactly what the record
says; the floor drops the lone circles and bare rectangles the source is mostly made of.

    uv run create_sketch_dataset.py --output-dir ./dataset_sketches --canvas-size 256
    uv run create_sketch_dataset.py --output-dir ./dataset_sketches_small --limit 1000
"""

import argparse
import json
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from generator import Circle, Element, FinishDrawing, PartLine, PartLineWithId
from renderer import StaticRenderer


DEFAULT_OUTPUT_DIR = Path("datasets/sketches")
DEFAULT_SEQUENCES_DIR = Path("sketchgraphs")

#: Where a split's sequences are published, by the name this dataset knows the split as.
SOURCE = "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_%s.npy"
SPLITS = {"train": "train", "val": "validation"}


@dataclass(frozen=True)
class DatasetConfig:
    sequences_dir: Path
    canvas_size: int
    margin: float
    min_nodes: int
    max_nodes: int
    thickness: int
    limit: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the SketchGraphs sketches made of lines and circles, rendered with StaticRenderer.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequences-dir", type=Path, default=DEFAULT_SEQUENCES_DIR,
                        help="Where the SketchGraphs sequence files are, or are downloaded to (6.2 GB for train).")
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.06,
                        help="Fraction of a side left blank around a sketch.")
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=16,
                        help="The published sequences hold at most sixteen entities.")
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep at most this many sketches per split. All of them by default.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> DatasetConfig:
    if args.canvas_size <= 0:
        raise ValueError("canvas-size must be positive")
    if not 0.0 <= args.margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    if not 1 <= args.min_nodes <= args.max_nodes:
        raise ValueError("need 1 <= min-nodes <= max-nodes")
    if args.thickness <= 0:
        raise ValueError("thickness must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    return DatasetConfig(
        sequences_dir=args.sequences_dir,
        canvas_size=args.canvas_size,
        margin=args.margin,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        thickness=args.thickness,
        limit=args.limit,
    )


def build_renderer(config: DatasetConfig) -> StaticRenderer:
    return StaticRenderer(
        canvas_width=config.canvas_size,
        canvas_height=config.canvas_size,
        thickness=defaultdict(lambda: config.thickness),
        font_size=8,
    )


def sequences_file(config: DatasetConfig, split_name: str) -> Path:
    """The published sequences of a split, downloaded on first use."""
    config.sequences_dir.mkdir(parents=True, exist_ok=True)
    held = config.sequences_dir / f"sg_t16_{SPLITS[split_name]}.npy"
    if held.exists():
        print(f"[{split_name}] sequences cached at {held}")
        return held
    source = SOURCE % SPLITS[split_name]
    print(f"[{split_name}] downloading {source} into {held.parent}")
    partial = held.with_suffix(".part")
    urllib.request.urlretrieve(source, partial)
    partial.rename(held)
    return held


def placed_actions(sketch, config: DatasetConfig) -> Optional[List[Element]]:
    """The lines and circles a sketch draws, placed on the canvas, or ``None`` where it draws
    anything else or too little.

    A sketch carries no scale of its own, so it is fitted to the canvas by what it spans, which
    keeps its proportions and leaves the margin blank. Onshape's y points up and a drawing's
    points down, so the two are read against each other here.

    A sketch is dropped as well where the fit leaves a line or a circle smaller than a pixel: the
    renderer would draw nothing for it, and a record must not claim what the drawing does not show.
    """
    from sketchgraphs.data._entity import Circle as SketchCircle, Line as SketchLine

    entities = list(sketch.entities.values())
    if not config.min_nodes <= len(entities) <= config.max_nodes:
        return None
    if not all(isinstance(held, (SketchLine, SketchCircle)) and not held.isConstruction for held in entities):
        return None

    def ends(line: SketchLine):
        return (
            (line.pntX + line.dirX * line.startParam, line.pntY + line.dirY * line.startParam),
            (line.pntX + line.dirX * line.endParam, line.pntY + line.dirY * line.endParam),
        )

    corners = []
    for held in entities:
        if isinstance(held, SketchLine):
            corners.extend(ends(held))
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
            line = PartLine(*map(placed, ends(held)))
            if round(line.length * pixels) < 1:
                return None
            actions.append(PartLineWithId(str(index), line))
        else:
            circle = Circle(placed((held.xCenter, held.yCenter)), held.radius * scale)
            if round(circle.radius * pixels) < 1:
                return None
            actions.append(circle)
    actions.append(FinishDrawing())
    return actions


def render(actions: List[Element], renderer: StaticRenderer) -> np.ndarray:
    image = 255 - renderer.draw(actions)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def to_json(actions: List[Element]) -> str:
    return json.dumps([action.serialize() for action in actions], separators=(",", ":"))


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.npy")) + list(output_dir.glob("*.json*"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already contains dataset files. Use --overwrite to replace them."
        )


def write_split(split_name: str, config: DatasetConfig, output_dir: Path, renderer: StaticRenderer) -> int:
    """Writes a split and returns how many sketches it holds.

    Two passes over the sequences: the first finds which are kept, so that the image file can be
    made the right size, and the second draws them. The second reads only the kept ones, so it is
    the cheaper of the two.
    """
    from sketchgraphs.data import flat_array
    from sketchgraphs.data.sequence import sketch_from_sequence

    held = flat_array.load_dictionary_flat(str(sequences_file(config, split_name)))
    sequences, sketch_ids = held["sequences"], held["sketch_ids"]
    print(f"[{split_name}] reading {len(sequences)} sequences")

    kept = []
    for at in range(len(sequences)):
        if placed_actions(sketch_from_sequence(sequences[at]), config) is not None:
            kept.append(at)
            if config.limit is not None and len(kept) == config.limit:
                break
        if (at + 1) % 100_000 == 0:
            print(f"[{split_name}] {at + 1}/{len(sequences)} read, {len(kept)} kept")
    print(f"[{split_name}] keeping {len(kept)} of {len(sequences)} sketches")

    images = np.lib.format.open_memmap(
        output_dir / f"{split_name}_images.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(kept), config.canvas_size, config.canvas_size),
    )
    with open(output_dir / f"{split_name}_labels.jsonl", "w", encoding="utf-8") as labels:
        for index, at in enumerate(kept):
            actions = placed_actions(sketch_from_sequence(sequences[at]), config)
            images[index] = render(actions, renderer)
            source = sketch_ids[at]
            labels.write(json.dumps({
                "index": index,
                "actions": to_json(actions),
                "source": {
                    "document_id": source["document_id"].decode(),
                    "part_idx": int(source["part_idx"]),
                    "sketch_idx": int(source["sketch_idx"]),
                },
            }) + "\n")
            if (index + 1) % 10_000 == 0 or index + 1 == len(kept):
                print(f"[{split_name}] wrote {index + 1}/{len(kept)}")
    images.flush()
    return len(kept)


def write_metadata(config: DatasetConfig, splits: dict, output_dir: Path) -> None:
    metadata = {
        "format": "npy-memmap-v1",
        "task": "line-and-circle-cad-sketch",
        "class_names": ["sketch"],
        "image_dtype": "uint8",
        "image_shape": [config.canvas_size, config.canvas_size],
        "splits": splits,
        "source": {
            "dataset": "SketchGraphs",
            "url": "https://github.com/PrincetonLIPS/SketchGraphs",
            "files": {name: SOURCE % published for name, published in SPLITS.items()},
            "kept": "every entity a Line or a Circle, none construction, "
                    f"{config.min_nodes} to {config.max_nodes} of them",
        },
        "renderer": {
            "name": "StaticRenderer",
            "thickness": config.thickness,
        },
        "placement": {
            "margin": config.margin,
            "fit": "bounding box scaled uniformly into the canvas, y flipped",
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
    print(f"Keeping sketches of {config.min_nodes} to {config.max_nodes} lines and circles"
          + (f", at most {config.limit} per split" if config.limit else ""))

    splits = {name: write_split(name, config, output_dir, renderer) for name in ("train", "val")}
    write_metadata(config, splits, output_dir)

    print("Dataset creation complete.")


if __name__ == "__main__":
    main()
