import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Tuple
import orjson

import numpy as np

from generator import (
    AnnotateLineGenerator,
    AppendFinishDrawingGenerator,
    ChamferedOutlineGenerator,
    MarginGenerator,
    PartLine,
    PickOneGenerator,
    RandomTranslationGenerator,
    RectilinearOutlineGenerator,
    ShuffleByElementGenerator,
    random_mirror_and_rotation_augmentation,
)
from renderer import StaticRenderer
from validity import Invalid, check, outline_of


DEFAULT_OUTPUT_DIR = Path("datasets/rectilinear_static_annotated")
DEFAULT_TRAIN_SIZE = 256
DEFAULT_VAL_SIZE = 64
FULL_TRAIN_SIZE = 131072
FULL_VAL_SIZE = 1024


def dataset_font_size(canvas_size: int) -> int:
    return max(8, int(round(canvas_size / 24)))


@dataclass(frozen=True)
class DatasetConfig:
    train_size: int
    val_size: int
    canvas_size: int
    margin: float
    max_translation: float
    thickness: int
    base_seed: int
    annotation_ratio: float
    workers: int
    grid: int
    n_cells: int
    min_edges: int
    max_edges: int
    min_edge_length: float
    min_gap: float
    min_turn: float
    chamfer_ratio: float
    chamfer_cut: Tuple[float, float]
    chamfer_keep: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an annotated rectilinear-part dataset rendered with StaticRenderer.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--max-translation", type=float, default=0.1)
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=1729)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--annotation-ratio", type=float, default=0.3)
    parser.add_argument("--grid", type=int, default=8, help="lattice the part is grown on")
    parser.add_argument("--n-cells", type=int, default=14, help="cells in the part: the difficulty dial")
    parser.add_argument("--min-edges", type=int, default=8)
    parser.add_argument("--max-edges", type=int, default=24)
    parser.add_argument("--min-edge-length", type=float, default=0.02, help="fraction of the canvas")
    parser.add_argument("--min-gap", type=float, default=0.02, help="fraction of the canvas")
    parser.add_argument("--min-turn", type=float, default=15.0,
                        help="least angle in degrees between consecutive edges")
    parser.add_argument("--chamfer-ratio", type=float, default=0.0,
                        help="share of corners offered a chamfer; 0 keeps the part rectilinear")
    parser.add_argument("--chamfer-cut", type=float, nargs=2, default=(0.25, 0.6), metavar=("LOW", "HIGH"),
                        help="each leg of a chamfer, in cells")
    parser.add_argument("--chamfer-keep", type=float, default=0.35,
                        help="the least of an edge, in cells, chamfers leave")
    parser.add_argument("--full", action="store_true", help="Use 131072 train and 1024 validation samples.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> DatasetConfig:
    train_size = FULL_TRAIN_SIZE if args.full else args.train_size
    val_size = FULL_VAL_SIZE if args.full else args.val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train-size and val-size must be positive")
    if args.canvas_size <= 0:
        raise ValueError("canvas-size must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not 0.0 <= args.margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    if args.max_translation < 0:
        raise ValueError("max-translation must be non-negative")
    if args.thickness <= 0:
        raise ValueError("thickness must be positive")
    if not 0.0 <= args.annotation_ratio <= 1.0:
        raise ValueError("annotation-ratio must be in [0, 1]")
    if not 0.0 <= args.min_turn < 90.0:
        raise ValueError("min-turn must be in [0, 90)")
    if not 0.0 <= args.chamfer_ratio <= 1.0:
        raise ValueError("chamfer-ratio must be in [0, 1]")
    low, high = args.chamfer_cut
    if not 0.0 < low <= high < 1.0:
        raise ValueError("chamfer-cut must be 0 < LOW <= HIGH < 1")
    if args.chamfer_keep <= 0.0:
        raise ValueError("chamfer-keep must be positive")
    return DatasetConfig(
        train_size=train_size,
        val_size=val_size,
        canvas_size=args.canvas_size,
        margin=args.margin,
        max_translation=args.max_translation,
        thickness=args.thickness,
        base_seed=args.base_seed,
        annotation_ratio=args.annotation_ratio,
        workers=args.workers,
        grid=args.grid,
        n_cells=args.n_cells,
        min_edges=args.min_edges,
        max_edges=args.max_edges,
        min_edge_length=args.min_edge_length,
        min_gap=args.min_gap,
        min_turn=args.min_turn,
        chamfer_ratio=args.chamfer_ratio,
        chamfer_cut=(low, high),
        chamfer_keep=args.chamfer_keep,
    )


def generator_definition(config: DatasetConfig) -> str:
    lattice = (
        f"grid={config.grid}, n_cells={config.n_cells}, "
        f"min_edges={config.min_edges}, max_edges={config.max_edges}"
    )
    if config.chamfer_ratio > 0.0:
        outline = (
            f"ChamferedOutlineGenerator(ratio={config.chamfer_ratio}, cut_cells={config.chamfer_cut}, "
            f"keep_cells={config.chamfer_keep}, {lattice})"
        )
    else:
        outline = f"RectilinearOutlineGenerator({lattice})"
    train_outline = (
        "RandomTranslationGenerator("
        f"MarginGenerator(random_mirror_and_rotation_augmentation({outline}), margin={config.margin}), "
        f"max_translation={config.max_translation})"
    )
    annotated = f"AnnotateLineGenerator({train_outline}, ratio={config.annotation_ratio})"
    picked = f"PickOneGenerator([{annotated}])"
    return f"AppendFinishDrawingGenerator(ShuffleByElementGenerator({picked}))"


def build_outline_generator(config: DatasetConfig):
    lattice = dict(
        grid=config.grid,
        n_cells=config.n_cells,
        min_edges=config.min_edges,
        max_edges=config.max_edges,
    )
    if config.chamfer_ratio > 0.0:
        return ChamferedOutlineGenerator(
            ratio=config.chamfer_ratio,
            cut_cells=config.chamfer_cut,
            keep_cells=config.chamfer_keep,
            **lattice,
        )
    return RectilinearOutlineGenerator(**lattice)


def build_generator(config: DatasetConfig):
    outline_generator = RandomTranslationGenerator(
        MarginGenerator(
            random_mirror_and_rotation_augmentation(build_outline_generator(config)),
            margin=config.margin,
        ),
        max_translation=config.max_translation,
    )
    generator = PickOneGenerator([
        AnnotateLineGenerator(outline_generator, ratio=config.annotation_ratio)
    ])
    return AppendFinishDrawingGenerator(ShuffleByElementGenerator(generator))


class Rejected(Exception):
    """A sample that broke a constraint, to be drawn again rather than written."""


def build_renderer(config: DatasetConfig) -> StaticRenderer:
    thickness = defaultdict(lambda: config.thickness, {PartLine: config.thickness})
    return StaticRenderer(
        canvas_width=config.canvas_size,
        canvas_height=config.canvas_size,
        thickness=thickness,
        font_size=dataset_font_size(config.canvas_size),
    )


MAX_RESAMPLES = 64


def render_sample(sample_seed: int, config: DatasetConfig) -> Tuple[np.ndarray, str]:
    """One drawing, and the record it was drawn from.

    A candidate that breaks a constraint is drawn again from a fresh seed rather than repaired:
    repairing it would change the record the picture came from, which is the one thing this
    dataset guarantees.
    """
    def to_json(actions: list) -> str:
        actions_json = [action.serialize() for action in actions]
        return json.dumps(actions_json, separators=(',', ':'))

    for attempt in range(MAX_RESAMPLES):
        seed = sample_seed + attempt * 1_000_003
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        try:
            actions = build_generator(config).get_actions()
            check(
                actions,
                min_edge_length=config.min_edge_length,
                min_gap=config.min_gap,
                min_turn=config.min_turn,
            )
        except (Invalid, RuntimeError):
            continue
        image = 255 - build_renderer(config).draw(actions, seed=seed)
        return np.clip(np.rint(image), 0, 255).astype(np.uint8), to_json(actions)
    raise RuntimeError(
        f"no drawing satisfying the constraints in {MAX_RESAMPLES} attempts from seed {sample_seed}; "
        "loosen min-gap, min-edge-length or min-turn, or lower n-cells"
    )


def render_job(job: Tuple[int, DatasetConfig]) -> Tuple[np.ndarray, str]:
    sample_seed, config = job
    return render_sample(sample_seed, config)


def split_seeds(base_seed: int, split_name: str, count: int) -> np.ndarray:
    spawn_key = 0 if split_name == "train" else 1
    seed_sequence = np.random.SeedSequence(base_seed, spawn_key=(spawn_key,))
    return seed_sequence.generate_state(count, dtype=np.uint32)


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.npy")) + list(output_dir.glob("*.json"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already contains dataset files. Use --overwrite to replace them."
        )


def write_split(split_name: str, count: int, config: DatasetConfig, output_dir: Path) -> None:
    seeds = split_seeds(config.base_seed, split_name, count)
    images_path = output_dir / f"{split_name}_images.npy"
    labels_path = output_dir / f"{split_name}_labels.jsonl"
    seeds_path = output_dir / f"{split_name}_seeds.npy"

    images = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(count, config.canvas_size, config.canvas_size),
    )
    np.save(seeds_path, seeds)

    assert config.workers == 1, "Multiprocessing is not supported in this environment."

    jobs: Iterable[Tuple[int, DatasetConfig]] = ((int(seed), config) for seed in seeds)
    iterator = (render_job(job) for job in jobs)
    
    with open(labels_path, "w", encoding="utf-8") as f_labels:
        for index, (image, action_str) in enumerate(iterator):
            images[index] = image
            f_labels.write(json.dumps({"index": index, "actions": action_str}) + "\n")
            if (index + 1) % 256 == 0 or index + 1 == count:
                print(f"[{split_name}] wrote {index + 1}/{count}")

    images.flush()


def write_metadata(config: DatasetConfig, output_dir: Path) -> None:
    metadata = {
        "format": "npy-memmap-v1",
        "task": "annotated-chamfered-part-outline" if config.chamfer_ratio > 0.0 else "annotated-rectilinear-part-outline",
        "class_names": ["chamfered_part" if config.chamfer_ratio > 0.0 else "rectilinear_part"],
        "shape": {
            "grid": config.grid,
            "n_cells": config.n_cells,
            "min_edges": config.min_edges,
            "max_edges": config.max_edges,
            "min_edge_length": config.min_edge_length,
            "min_gap": config.min_gap,
            "min_turn": config.min_turn,
            "chamfer": {
                "ratio": config.chamfer_ratio,
                "cut_cells": list(config.chamfer_cut),
                "keep_cells": config.chamfer_keep,
            },
        },
        "image_dtype": "uint8",
        "image_shape": [config.canvas_size, config.canvas_size],
        "splits": {
            "train": config.train_size,
            "val": config.val_size,
        },
        "generator_definition": generator_definition(config),
        "renderer": {
            "name": "StaticRenderer",
            "thickness": config.thickness,
            "font_size": dataset_font_size(config.canvas_size),
        },
        "augmentations": {
            "annotation_ratio": config.annotation_ratio,
            "margin": config.margin,
            "max_translation": config.max_translation,
            "random_rotation_90deg": True,
            "random_mirror_xy": True,
            "shuffle_by_element": True,
            "append_finish_drawing": True,
        },
        "config": asdict(config),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = validate_args(args)
    output_dir = args.output_dir
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    print(f"Writing dataset to {output_dir}")
    print(f"Generator: {generator_definition(config)}")
    print(f"Train samples: {config.train_size}")
    print(f"Validation samples: {config.val_size}")

    write_split("train", config.train_size, config, output_dir)
    write_split("val", config.val_size, config, output_dir)
    write_metadata(config, output_dir)

    print("Dataset creation complete.")


if __name__ == "__main__":
    main()