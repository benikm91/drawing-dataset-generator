"""The SketchGraphs sketches as Vitruvion (Seff et al., ICLR 2022) selected them, rebuilt here.

Vitruvion's preprocessed data, `sg_filtered_unique.npy`, is what PICASSO, DAVINCI, PpaCAD and
CadVLM train and test on, and it can no longer be downloaded. This rebuilds it from the raw
SketchGraphs JSON shards with Vitruvion's rules, read from its code
(https://github.com/PrincetonLIPS/vitruvion, `img2cad/pipeline` and `img2cad/data_utils.py`) where
the paper leaves them open. Each step is a port of theirs, kept to their arithmetic, because the
deduplication key is quantised and a value moved across a bin edge is a different sketch.

A sketch is kept when, in this order -- a sketch is counted under the first rule it fails:

1. it has at least one entity and one constraint,
2. it has 6 to 16 entities, points included, and at most 64 constraints,
3. every entity is a line, circle, arc or point -- one of any other kind rejects the sketch,
4. no constraint is a mirror, a projection or a linear or circular pattern -- one rejects the sketch,
5. it can be centred and scaled, its bounding box being nonzero,
6. no circle or arc has zero radius, no line zero length, no arc its start at its midpoint,
7. and its sequence of primitives -- types, 6-bit quantised parameters and construction flags, in
   the designer's order -- is not one an earlier sketch already has.

The shards are read in name order, so the first copy of a sketch is the one kept. What is written
per sketch is its source, its primitives with their normalised continuous parameters, and the bins
those quantise to; the split is this dataset's own, drawn from `--seed` in Vitruvion's
proportions, and written as index lists.

    uv run create_vitruvion_dataset.py --output-dir ./dataset_vitruvion
    uv run create_vitruvion_dataset.py --output-dir ./dataset_vitruvion_shard1 --shards 1
"""

import argparse
import collections
import datetime
import json
import math
import multiprocessing
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


DEFAULT_OUTPUT_DIR = Path("datasets/vitruvion")
#: What a build's README says about the data; the build adds its own counts below it.
CARD = Path(__file__).parent / "dataset_cards" / "vitruvion.md"
#: Where `vitruvion_reference/compare.py` writes its verdict into a build's README.
CHECK_PLACEHOLDER = "Check against Vitruvion's code: not run yet (`vitruvion_reference/compare.py`)."
DEFAULT_SHARDS_DIR = Path("sketchgraphs/shards")

SOURCE = "https://sketchgraphs.cs.princeton.edu/shards/shard_%03d_of_128.tar.zst"
NUM_SHARDS = 128

MIN_ENTITIES, MAX_ENTITIES, MAX_CONSTRAINTS = 6, 16, 64
NUM_BINS = 64
MIN_VAL, MAX_VAL = -0.5, 0.5

#: The value tokens of Vitruvion's `img2cad.dataset.Token`, a primitive's type and the bracket
#: around a sequence; a bin is written after them, and the construction flag after the bins.
TOKENS = {"Start": 1, "Stop": 2, "Arc": 3, "Circle": 4, "Line": 5, "Point": 6}
CONSTRUCTION_TOKEN = {True: 7 + NUM_BINS, False: 7 + NUM_BINS + 1}

#: The constraints kept, as Vitruvion's constraint model keeps them (`img2cad.constraint_data.Token`):
#: the categorical ones, each on one or two primitives or their points. Constraints with a value
#: (distance, angle, length, radius, diameter) are left out, as are those on geometry outside the
#: sketch.
CONSTRAINTS = ("Coincident", "Concentric", "Equal", "Fix", "Horizontal", "Midpoint", "Normal",
               "Offset", "Parallel", "Perpendicular", "Quadrant", "Tangent", "Vertical")

#: The rules in the order they are applied, by the name a rejected sketch is counted under.
STEPS = [
    "empty",
    "too_many_entities",
    "too_few_entities",
    "too_many_constraints",
    "invalid_entity_type",
    "invalid_constraint_type",
    "cannot_normalize",
    "zero_sized_entity",
    "sequence_error",
]


@dataclass(frozen=True)
class DatasetConfig:
    shards_dir: Path
    shards: Tuple[int, ...]
    seed: int
    val_fraction: float
    test_fraction: float
    workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Vitruvion's SketchGraphs selection from the raw JSON shards.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR,
                        help="Where the SketchGraphs JSON shards are, or are downloaded to (43 GB for all).")
    parser.add_argument("--shards", type=int, nargs="+", default=list(range(1, NUM_SHARDS + 1)),
                        help="Which of the 128 shards to read, by number. All of them by default.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the train / val / test split.")
    parser.add_argument("--val-fraction", type=float, default=0.025)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help="Shards read at once.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> DatasetConfig:
    if not args.shards or not all(1 <= shard <= NUM_SHARDS for shard in args.shards):
        raise ValueError(f"shards are numbered 1 to {NUM_SHARDS}")
    if not (0 <= args.val_fraction and 0 <= args.test_fraction and args.val_fraction + args.test_fraction < 1):
        raise ValueError("need val-fraction, test-fraction >= 0 and their sum below 1")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    return DatasetConfig(
        shards_dir=args.shards_dir,
        shards=tuple(sorted(set(args.shards))),
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        workers=args.workers,
    )


def shard_file(config: DatasetConfig, shard: int) -> Path:
    """A shard of the raw SketchGraphs data, downloaded on first use."""
    held = config.shards_dir / f"shard_{shard:03d}_of_{NUM_SHARDS}.tar.zst"
    if not held.exists():
        config.shards_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SOURCE % shard} into {held.parent}")
        partial = held.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE % shard, partial)
        partial.rename(held)
    return held


# -- Vitruvion's normalisation, ported from `img2cad/data_utils.py` -----------------------------

#: The parameters a translation moves and a scaling scales, by entity type.
POS_PARAMS = {"Arc": ("xCenter", "yCenter"), "Circle": ("xCenter", "yCenter"),
              "Line": ("pntX", "pntY"), "Point": ("x", "y")}
SCALE_PARAMS = {"Arc": ("radius",), "Circle": ("radius",),
                "Line": ("startParam", "endParam"), "Point": ()}


def entity_bbox(entity) -> Optional[np.ndarray]:
    """What an entity spans, as `[[x0, y0], [x1, y1]]`, or ``None`` for a kind not drawn.

    A circle spans its whole circle. An arc spans its circle's box, pulled in to its ends on each
    side whose extreme it does not pass, which Vitruvion finds by the quadrants about the centre
    its ends sit in; that reading is kept here as it is, including how it settles two ends in one
    quadrant.
    """
    from sketchgraphs.data import Arc, Circle, Line, Point

    def circle_bbox(held):
        return np.array([[held.xCenter - held.radius, held.yCenter - held.radius],
                         [held.xCenter + held.radius, held.yCenter + held.radius]])

    def quadrant(point, center):
        x, y = point[0] - center[0], point[1] - center[1]
        if x >= 0:
            return 1 if y >= 0 else 4
        return 2 if y >= 0 else 3

    if isinstance(entity, Arc):
        x_start, y_start = entity.start_point
        x_end, y_end = entity.end_point
        (x0, y0), (x1, y1) = circle_bbox(entity)
        start_quadrant = quadrant(entity.start_point, entity.center_point)
        end_quadrant = quadrant(entity.end_point, entity.center_point)
        if entity.clockwise:
            start_quadrant, end_quadrant = end_quadrant, start_quadrant
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        if start_quadrant < end_quadrant:
            quadrants = list(range(start_quadrant, end_quadrant + 1))
        elif start_quadrant > end_quadrant:
            quadrants = list(range(start_quadrant, 5)) + list(range(1, end_quadrant + 1))
        else:
            start_ahead = False
            if start_quadrant in (1, 2) and x_start <= x_end:
                start_ahead = True
            if start_quadrant in (3, 4) and x_start >= x_end:
                start_ahead = True
            if start_ahead:
                quadrants = list(range(start_quadrant, 5)) + list(range(1, end_quadrant + 1))
            else:
                quadrants = [start_quadrant]

        def passes(q1, q2):
            return q1 in quadrants and q2 in quadrants and quadrants.index(q2) > quadrants.index(q1)

        if not passes(4, 1):
            x1 = max(x_start, x_end)
        if not passes(1, 2):
            y1 = max(y_start, y_end)
        if not passes(2, 3):
            x0 = min(x_start, x_end)
        if not passes(3, 4):
            y0 = min(y_start, y_end)
        return np.array([[x0, y0], [x1, y1]])
    if isinstance(entity, Circle):
        return circle_bbox(entity)
    if isinstance(entity, Line):
        (sx, sy), (ex, ey) = entity.start_point, entity.end_point
        return np.array([[min(sx, ex), min(sy, ey)], [max(sx, ex), max(sy, ey)]])
    if isinstance(entity, Point):
        return np.array([[entity.x, entity.y], [entity.x, entity.y]])
    return None


def sketch_bbox(sketch) -> np.ndarray:
    boxes = [entity_bbox(entity) for entity in sketch.entities.values()]
    boxes = np.array([box for box in boxes if box is not None])
    if boxes.size == 0:
        return np.array([[0., 0.], [0., 0.]])
    x0, y0 = np.min(boxes[:, 0, :], axis=0)
    x1, y1 = np.max(boxes[:, 1, :], axis=0)
    return np.array([[x0, y0], [x1, y1]])


def normalize_sketch(sketch) -> float:
    """Centres a sketch on the origin and scales its longer side to one, in place, and returns
    the factor it was scaled down by, or -1 where it spans nothing."""
    (x0, y0), (x1, y1) = sketch_bbox(sketch)
    x_offset, y_offset = np.mean([x0, x1]), np.mean([y0, y1])
    for entity in sketch.entities.values():
        for name in POS_PARAMS.get(entity.type.name, ()):
            offset = x_offset if "x" in name.lower() else y_offset
            setattr(entity, name, getattr(entity, name) - offset)

    (x0, y0), (x1, y1) = sketch_bbox(sketch)
    if not np.isclose(x0, -x1) or not np.isclose(y0, -y1):
        raise ValueError("sketch must be centered before rescaling")
    factor = max(x1 - x0, y1 - y0)
    if factor == 0:
        return -1
    for entity in sketch.entities.values():
        kind = entity.type.name
        if kind not in POS_PARAMS:
            continue
        for name in POS_PARAMS[kind] + SCALE_PARAMS[kind]:
            setattr(entity, name, getattr(entity, name) / factor)
    return factor


def parameters(entity) -> np.ndarray:
    """An entity's minimal parameters: a line's ends, a circle's centre and radius, a point,
    and an arc's start, midpoint and end, read counter-clockwise."""
    kind = entity.type.name
    if kind == "Arc":
        start, end = entity.start_point, entity.end_point
        if entity.clockwise:
            start, end = end, start
        return np.concatenate([start, entity.mid_point, end])
    if kind == "Circle":
        return np.append(entity.center_point, entity.radius)
    if kind == "Line":
        return np.concatenate([entity.start_point, entity.end_point])
    return np.array([entity.x, entity.y])


def quantize(values: np.ndarray) -> np.ndarray:
    """Values in [-0.5, 0.5] as one of `NUM_BINS` equal bins, 0.5 itself in the last.

    A value outside the range, as rounding leaves a few, is clipped into it, as Vitruvion does on
    the error its own quantiser raises for it.
    """
    values = np.around(values, decimals=10)
    if (values < MIN_VAL).any() or (values > MAX_VAL).any():
        values = np.clip(values, MIN_VAL, MAX_VAL)
    bins = ((values - MIN_VAL) / (MAX_VAL - MIN_VAL) * NUM_BINS).astype("int32")
    bins[bins == NUM_BINS] -= 1
    return bins


# -- the selection ------------------------------------------------------------------------------

def rejected_by(sketch) -> Optional[str]:
    """The first rule a raw sketch fails, or ``None`` where it passes them all. A sketch that
    passes is left normalised, as Vitruvion leaves it."""
    from sketchgraphs.data import Arc, ConstraintType, EntityType, Line

    entities, constraints = sketch.entities, sketch.constraints
    if len(constraints) == 0 or len(entities) == 0:
        return "empty"
    if len(entities) > MAX_ENTITIES:
        return "too_many_entities"
    if len(entities) < MIN_ENTITIES:
        return "too_few_entities"
    if len(constraints) > MAX_CONSTRAINTS:
        return "too_many_constraints"
    rejected_entities = (EntityType.Conic, EntityType.Ellipse, EntityType.Spline, EntityType.Unknown)
    if any(entity.type in rejected_entities for entity in entities.values()):
        return "invalid_entity_type"
    rejected_constraints = (ConstraintType.Circular_Pattern, ConstraintType.Linear_Pattern,
                            ConstraintType.Mirror, ConstraintType.Projected)
    if any(constraint.type in rejected_constraints for constraint in constraints.values()):
        return "invalid_constraint_type"
    try:
        if normalize_sketch(sketch) == -1:
            return "cannot_normalize"
    except Exception:
        return "cannot_normalize"
    for entity in entities.values():
        if getattr(entity, "radius", None) == 0:
            return "zero_sized_entity"
        if isinstance(entity, Line) and np.allclose(entity.start_point, entity.end_point):
            return "zero_sized_entity"
        if isinstance(entity, Arc) and np.allclose(entity.start_point, entity.mid_point):
            return "zero_sized_entity"
    return None


def constraints_of(seq) -> List[dict]:
    """The categorical constraints of a construction sequence, in its order -- each placed after
    the last primitive it refers to -- each with the primitives it refers to.

    A reference is a primitive's index, and the point of it meant, if one is: `start`, `end` or
    `center`. The point is named as the record's parameters read the primitive, so that the
    `start` of an arc is its first point there, which for an arc Onshape holds clockwise is the
    one Onshape calls its end.
    """
    from sketchgraphs.data import EdgeOp, EntityType, NodeOp, SubnodeType

    parts = {SubnodeType.SN_Start: "start", SubnodeType.SN_End: "end", SubnodeType.SN_Center: "center"}
    node, primitives, held = {}, [], None
    for index, op in enumerate(op for op in seq if isinstance(op, NodeOp)):
        if isinstance(op.label, EntityType):
            if op.label in (EntityType.External, EntityType.Stop):
                continue
            held = (len(primitives), op.label == EntityType.Arc and bool(op.parameters.get("clockwise")))
            primitives.append(index)
            node[index] = (held[0], None)
        else:
            part = parts[op.label]
            if held[1] and part != "center":
                part = "end" if part == "start" else "start"
            node[index] = (held[0], part)

    kept = []
    for op in seq:
        if not isinstance(op, EdgeOp) or op.label.name not in CONSTRAINTS or 0 in op.references:
            continue
        kept.append({
            "type": op.label.name,
            "refs": [list(node[ref]) for ref in sorted(op.references)],
        })
    return kept


def read_shard(job) -> Tuple[int, collections.Counter, list]:
    """Reads a shard and returns how many sketches each rule rejected, and the record of each it
    keeps, with the tokens it is deduplicated by.

    A kept sketch goes through what Vitruvion stores it as -- its construction sequence -- and
    back before it is tokenised, and is normalised once more after, as their tokeniser does; both
    are kept here, since the second normalisation moves values by rounding.
    """
    from sketchgraphs.data import sequence
    from sketchgraphs.pipeline.make_sketch_dataset import load_json_tarball

    shard, path = job
    counts, kept = collections.Counter(), []
    for (document_id, part_idx, sketch_idx), sketch in load_json_tarball(str(path)):
        counts["read"] += 1
        reason = rejected_by(sketch)
        if reason is None:
            try:
                seq = sequence.sketch_to_sequence(sketch)
                sketch = sequence.sketch_from_sequence(seq)
            except Exception:
                reason = "sequence_error"
        if reason is not None:
            counts[reason] += 1
            continue
        normalize_sketch(sketch)

        tokens, primitives = [TOKENS["Start"]], []
        for entity in sketch.entities.values():
            values = parameters(entity)
            bins = quantize(values)
            tokens += [TOKENS[entity.type.name], *(bins + 7).tolist(), CONSTRUCTION_TOKEN[bool(entity.isConstruction)]]
            primitives.append({
                "type": entity.type.name,
                "params": values.tolist(),
                "bins": bins.tolist(),
                "construction": bool(entity.isConstruction),
            })
        tokens.append(TOKENS["Stop"])
        kept.append((
            np.asarray(tokens, dtype=np.int16).tobytes(),
            {"document_id": document_id, "part_idx": int(part_idx), "sketch_idx": int(sketch_idx)},
            primitives,
            constraints_of(seq),
        ))
    return shard, counts, kept


def split(count: int, config: DatasetConfig) -> dict:
    """A random train / val / test split of `count` sketches, in Vitruvion's proportions and
    rounding: train and test rounded up, val what is left."""
    order = np.random.default_rng(config.seed).permutation(count)
    train = int(math.ceil(count * (1 - config.val_fraction - config.test_fraction)))
    test = min(int(math.ceil(count * config.test_fraction)), count - train)
    return {
        "train": np.sort(order[:train]).tolist(),
        "val": np.sort(order[train:count - test]).tolist(),
        "test": np.sort(order[count - test:]).tolist(),
    }


def write_readme(output_dir: Path, config: DatasetConfig, steps: list, splits: dict) -> None:
    """The dataset card, with what this build read and kept appended to it."""
    rows = "\n".join(
        f"| {step['step']} | {step.get('rejected', '')} | {step['kept']} |" for step in steps
    )
    shards = (f"all {NUM_SHARDS} shards" if len(config.shards) == NUM_SHARDS
              else f"{len(config.shards)} of the {NUM_SHARDS} shards ({', '.join(map(str, config.shards))})")
    build = f"""
Built on {datetime.date.today().isoformat()} from {shards}, with split seed {config.seed}.

| Step | Rejected | Left |
|---|---|---|
{rows}

| Split | Sketches |
|---|---|
""" + "\n".join(f"| {name} | {len(indices)} |" for name, indices in splits.items()) + f"""

{CHECK_PLACEHOLDER}
"""
    (output_dir / "README.md").write_text(CARD.read_text(encoding="utf-8") + build, encoding="utf-8")


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.json*"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already contains dataset files. Use --overwrite to replace them."
        )


def main() -> None:
    args = parse_args()
    config = validate_args(args)
    ensure_output_dir(args.output_dir, overwrite=args.overwrite)
    jobs = [(shard, shard_file(config, shard)) for shard in config.shards]

    print(f"Reading {len(jobs)} shards with {config.workers} workers into {args.output_dir}")
    totals, seen, written = collections.Counter(), set(), 0
    with multiprocessing.Pool(min(config.workers, len(jobs))) as pool, \
            open(args.output_dir / "sketches.jsonl", "w", encoding="utf-8") as out:
        # In shard order, so that which copy of a sketch is kept does not depend on the workers.
        for shard, counts, kept in pool.imap(read_shard, jobs):
            totals += counts
            totals["accepted"] += len(kept)
            for tokens, source, primitives, constraints in kept:
                if tokens in seen:
                    continue
                seen.add(tokens)
                out.write(json.dumps({
                    "index": written, "source": source, "primitives": primitives, "constraints": constraints,
                }) + "\n")
                written += 1
            print(f"[shard {shard:03d}] read {counts['read']}, accepted {len(kept)}, "
                  f"unique so far {written}")

    left, steps = totals["read"], [{"step": "read", "kept": totals["read"]}]
    for name in STEPS:
        left -= totals[name]
        steps.append({"step": name, "rejected": totals[name], "kept": left})
    steps.append({"step": "duplicate", "rejected": totals["accepted"] - written, "kept": written})
    assert left == totals["accepted"]
    for step in steps:
        print(f"  {step['step']:<24} {('-' + str(step['rejected'])) if 'rejected' in step else '':>10}  {step['kept']:>10}")

    splits = split(written, config)
    (args.output_dir / "splits.json").write_text(json.dumps({"seed": config.seed, **splits}), encoding="utf-8")
    metadata = {
        "format": "sketches-jsonl-v1",
        "task": "vitruvion-sketchgraphs",
        "count": written,
        "splits": {name: len(indices) for name, indices in splits.items()},
        "steps": steps,
        "source": {
            "dataset": "SketchGraphs",
            "url": "https://github.com/PrincetonLIPS/SketchGraphs",
            "shards": [SOURCE % shard for shard in config.shards],
            "rules": "Vitruvion (Seff et al., ICLR 2022), https://github.com/PrincetonLIPS/vitruvion",
        },
        "primitives": {
            "Line": "params [x1, y1, x2, y2]",
            "Circle": "params [x, y, r]",
            "Arc": "params [x_start, y_start, x_mid, y_mid, x_end, y_end], counter-clockwise",
            "Point": "params [x, y]",
            "bins": f"each param quantised to {NUM_BINS} bins over [{MIN_VAL}, {MAX_VAL}]",
            "frame": "centred on the origin, longer side of the bounding box 1, y up",
        },
        "constraints": {
            "types": list(CONSTRAINTS),
            "refs": "[primitive index, point]; point is null for the whole primitive, else start, end "
                    "or center as the primitive's params read it",
            "order": "construction sequence order, each after the last primitive it refers to",
        },
        "deduplicated": "on types, bins and construction flags in order; the first copy in shard order kept",
        "config": {**asdict(config), "shards_dir": str(config.shards_dir)},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_readme(args.output_dir, config, steps, splits)
    print("Dataset creation complete.")


if __name__ == "__main__":
    main()
