"""The rebuilt Vitruvion dataset as the drawing corpora are laid out: per split, the clean renders
as one array and one label line per render, in the same order.

Reads a folder as `vitruvion_reference/run_vitruvion.sh` leaves it -- `sg_filtered_unique.npy`,
`splits.json` and `renders/` -- and writes into its `records/`:

- `{split}_images.npy`: `(N, 128, 128)` uint8, the clean render of every sketch of the split, row
  index = y, white background, dark ink, as Vitruvion's renderer drew it;
- `{split}_labels.jsonl`: per render, the sketch it is of, its primitives and its constraints.

The labels keep everything Vitruvion's models are given, and nothing is decided for a model here:

- a primitive is placed in the pixels of its render, in Vitruvion's minimal parameters -- a line by
  its start and end, a circle by its centre and radius, an arc by its start, midpoint and end, read
  counter-clockwise on the render, and a point -- with its construction flag;
- a constraint is one of Vitruvion's categorical constraints, with every primitive it refers to,
  two or three, and the point of each it means, if one: `start`, `end` or `center`, named as the
  primitive's points read it. Constraints on geometry outside the sketch are left out, as
  Vitruvion's constraint model leaves them out.

Each split is written in an order drawn from `--seed` rather than the file's, which holds a
document's sketches next to each other; `sketch` gives each line's index in the file.

    uv run create_vitruvion_records.py --dataset-dir ./dataset_sketches_vitruvion
    uv run create_vitruvion_records.py --dataset-dir ./dataset_sketches_vitruvion --limit 1000 --output-dir ./records_small
"""

import argparse
import io
import json
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from create_vitruvion_dataset import constraints_of, normalize_sketch, parameters


#: Where sketch coordinates land in a 128 px render, measured on Vitruvion's renderer
#: (`prerender_images.render_sketch`: limits of +-0.6 on a one-inch figure, then `tight_layout`):
#: x from -0.6 to 0.6 spans these pixels from the left, y these pixels from the bottom.
RENDER_SIZE = 128
RENDER_X = (25.2622, 108.64)
RENDER_Y = (25.4222, 108.80)
RENDER_LIMIT = 0.6

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lay the rebuilt Vitruvion dataset out as the drawing corpora are.")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Holds sg_filtered_unique.npy, splits.json and renders/.")
    parser.add_argument("--output-dir", type=Path, default=None, help="records/ in the dataset folder by default.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the order each split is written in.")
    parser.add_argument("--limit", type=int, default=None, help="Write at most this many sketches per split.")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pixel(x: float, y: float) -> List[float]:
    """A sketch point where the renderer draws it, in pixels from the top left."""
    scale = (RENDER_X[1] - RENDER_X[0]) / (2 * RENDER_LIMIT)
    return [round(RENDER_X[0] + (x + RENDER_LIMIT) * scale, 2),
            round(RENDER_SIZE - (RENDER_Y[0] + (y + RENDER_LIMIT) * scale), 2)]


def primitive_of(entity) -> dict:
    """An entity in the pixels of its render, in the parameters `parameters` reads it by."""
    kind = entity.type.name.lower()
    values = parameters(entity)
    held = {"type": kind, "construction": bool(entity.isConstruction)}
    if kind == "circle":
        scale = (RENDER_X[1] - RENDER_X[0]) / (2 * RENDER_LIMIT)
        return {**held, "points": [pixel(*values[:2])], "radius": round(float(values[2]) * scale, 2)}
    return {**held, "points": [pixel(*values[at:at + 2]) for at in range(0, len(values), 2)]}


def label_of(job: Tuple[Path, List[int]]) -> List[Tuple[int, str]]:
    """The label line of every sketch asked for, by sketch."""
    from sketchgraphs.data import flat_array
    from sketchgraphs.data.sequence import sketch_from_sequence

    source, sketches = job
    sequences = flat_array.load_dictionary_flat(str(source))["sequences"]
    lines = []
    for sketch in sketches:
        seq = sequences[sketch]
        held = sketch_from_sequence(seq)
        normalize_sketch(held)
        constraints = [{"type": c["type"].lower(), "refs": c["refs"]} for c in constraints_of(seq)]
        lines.append((sketch, json.dumps({
            "sketch": sketch,
            "primitives": [primitive_of(entity) for entity in held.entities.values()],
            "constraints": constraints,
        })))
    return lines


def write_renders(job: Tuple[Path, Dict[int, Tuple[str, int]], Path]) -> int:
    """Decodes the clean render of every sketch of a chunk that is written, into its split's array."""
    from PIL import Image
    from sketchgraphs.data import flat_array

    chunk, slots, output_dir = job
    held = flat_array.load_dictionary_flat(str(chunk))
    indexes = np.asarray(held["indexes"])
    # The clean render of a sketch is the first of its images.
    sketches, clean = np.unique(indexes, return_index=True)
    arrays = {split: np.load(output_dir / f"{split}_images.npy", mmap_mode="r+") for split in SPLITS}
    written = 0
    for sketch, at in zip(sketches.tolist(), clean.tolist()):
        if sketch not in slots:
            continue
        split, slot = slots[sketch]
        image = np.asarray(Image.open(io.BytesIO(bytes(held["imgs"][at]))).convert("L"))
        assert image.shape == (RENDER_SIZE, RENDER_SIZE), f"sketch {sketch} renders at {image.shape}"
        arrays[split][slot] = image
        written += 1
    for array in arrays.values():
        array.flush()
    return written


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_dir / "records"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*_labels.jsonl")) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already holds records. Use --overwrite to replace them.")

    held_splits = json.loads((args.dataset_dir / "splits.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    order = {split: rng.permutation(np.asarray(held_splits[split]))[:args.limit] for split in SPLITS}
    slots = {int(sketch): (split, slot) for split in SPLITS for slot, sketch in enumerate(order[split])}
    for split in SPLITS:
        np.lib.format.open_memmap(output_dir / f"{split}_images.npy", mode="w+", dtype=np.uint8,
                                  shape=(len(order[split]), RENDER_SIZE, RENDER_SIZE)).flush()

    source = args.dataset_dir / "sg_filtered_unique.npy"
    wanted = sorted(slots)
    step = 20_000
    with multiprocessing.Pool(args.workers) as pool:
        lines: Dict[int, str] = {}
        for done in pool.imap_unordered(label_of, [(source, wanted[at:at + step]) for at in range(0, len(wanted), step)]):
            lines.update(done)
            print(f"labels: {len(lines)} of {len(slots)}", flush=True)
        for split in SPLITS:
            with open(output_dir / f"{split}_labels.jsonl", "w", encoding="utf-8") as out:
                out.writelines(lines[int(sketch)] + "\n" for sketch in order[split])
        del lines

        chunks = sorted((args.dataset_dir / "renders").glob("render_p128_*_of_*.npy"))
        written = 0
        for done in pool.imap_unordered(write_renders, [(chunk, slots, output_dir) for chunk in chunks]):
            written += done
            print(f"renders: {written} of {len(slots)}", flush=True)
    assert written == len(slots), f"{len(slots) - written} sketches have no render"

    (output_dir / "metadata.json").write_text(json.dumps({
        "format": "drawing-records-v1",
        "source": "sg_filtered_unique.npy, splits.json and renders/ of this repository",
        "splits": {split: len(order[split]) for split in SPLITS},
        "order": f"each split shuffled with seed {args.seed}; `sketch` is the index in sg_filtered_unique.npy",
        "images": f"(N, {RENDER_SIZE}, {RENDER_SIZE}) uint8, the clean render, row index = y",
        "primitives": {
            "frame": "pixels of the render, x from the left, y from the top",
            "line": "points [start, end]",
            "circle": "points [centre], radius",
            "arc": "points [start, mid, end], counter-clockwise on the render",
            "point": "points [point]",
            "construction": "drawn dashed; a reference for constraints, not part of the outline",
        },
        "constraints": {
            "types": "Vitruvion's categorical constraints, lower case",
            "refs": "[primitive index, point]; point is null for the whole primitive, else start, end "
                    "or center as the primitive's points read it; sorted as the construction sequence numbers them",
            "left out": "constraints on geometry outside the sketch, and those with a value",
        },
    }, indent=2), encoding="utf-8")
    print(f"wrote {', '.join(f'{len(order[s])} {s}' for s in SPLITS)} into {output_dir}")


if __name__ == "__main__":
    main()
