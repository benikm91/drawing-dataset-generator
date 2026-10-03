"""Checks `create_vitruvion_dataset.py` against Vitruvion's own pipeline, sketch by sketch.

The reference is Vitruvion's code run unchanged (see `run_vitruvion.sh`); this reads its
`sg_filtered_unique.npy`, the primitive tokens it caches beside it, and the constraint tokens
`ref_constraints.py` writes, and compares them with our `sketches.jsonl`:

- the same number of sketches, and the same set of distinct primitive-token sequences -- what
  deduplication is defined by, and so the same whatever order the shards were read in;
- the same sketches kept, in the same order -- which also needs the shards read in the same order,
  as it decides which copy of a duplicate is the one kept;
- for each sketch both kept, identical primitive tokens and identical constraint tokens.

    uv run python vitruvion_reference/compare.py REFERENCE_DIR OURS_DIR

Exits non-zero when anything differs.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sketchgraphs.data import flat_array
from sketchgraphs.data.sequence import sketch_from_sequence

PRIMITIVE_TOKENS = {"Arc": 3, "Circle": 4, "Line": 5, "Point": 6}
CONSTRAINT_TOKENS = {name: 3 + i for i, name in enumerate(
    ["Coincident", "Concentric", "Equal", "Fix", "Horizontal", "Midpoint", "Normal", "Offset",
     "Parallel", "Perpendicular", "Quadrant", "Tangent", "Vertical"])}
NUM_PARAMS = {"Arc": 6, "Circle": 3, "Line": 4, "Point": 2}
#: Vitruvion's pointer slot for each point of a primitive, by Onshape's naming of the points
#: (`gather_map` in img2cad.dataset.tokenize_sketch), relative to the primitive's type token.
SLOT = {"Arc": {None: 0, "center": 1, "start": 3, "end": 5}, "Line": {None: 0, "start": 1, "end": 3},
        "Circle": {None: 0, "center": 1}, "Point": {None: 0}}
#: Constraint tokens point past the 16 tokens of Vitruvion's constraint vocabulary.
POINTER_OFFSET = 16


def key(values) -> bytes:
    return np.asarray(values, dtype=np.int16).tobytes()


def our_primitive_tokens(row) -> list:
    tokens = [1]
    for p in row["primitives"]:
        tokens += [PRIMITIVE_TOKENS[p["type"]], *[b + 7 for b in p["bins"]], 71 if p["construction"] else 72]
    return tokens + [2]


def our_constraint_tokens(row, clockwise) -> list:
    """Our constraints as Vitruvion tokenises them. Our records name an arc's points as its
    parameters read it, counter-clockwise; Vitruvion points by Onshape's naming, which for a
    clockwise arc swaps start and end, so they are swapped back here."""
    starts, at = [], 1
    for p in row["primitives"]:
        starts.append(at)
        at += 2 + NUM_PARAMS[p["type"]]
    tokens = [1]
    for c in row["constraints"]:
        tokens.append(CONSTRAINT_TOKENS[c["type"]])
        for prim, part in c["refs"]:
            if clockwise[prim] and part in ("start", "end"):
                part = "end" if part == "start" else "start"
            tokens.append(starts[prim] + SLOT[row["primitives"][prim]["type"]][part] + POINTER_OFFSET)
    return tokens + [2]


def our_rows(ours: Path):
    with open(ours / "sketches.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            s = row["source"]
            yield (s["document_id"], s["part_idx"], s["sketch_idx"]), row


def main() -> int:
    reference, ours = Path(sys.argv[1]), Path(sys.argv[2])

    held = flat_array.load_dictionary_flat(str(reference / "sg_filtered_unique.npy"))
    sequences = held["sequences"]
    ref_ids = [(r["document_id"].decode(), int(r["part_idx"]), int(r["sketch_idx"])) for r in held["sketch_ids"]]
    ref_at = {sid: i for i, sid in enumerate(ref_ids)}
    cache = np.load(reference / "sg_filtered_unique_b64.cache.npz")
    val, offsets = cache["val"], cache["offsets"]
    ref_tokens = [key(val[offsets[i]:offsets[i + 1]]) for i in range(len(ref_ids))]
    constraints_file = reference / "sg_filtered_unique_constraints.npz"
    if constraints_file.exists():
        # Read out once: an npz member is read from the file again at every access.
        with np.load(constraints_file) as held_constraints:
            cval, coff = held_constraints["val"], held_constraints["offsets"]
    else:
        cval = coff = None

    # First pass: what identifies each of our sketches, kept small enough for the full dataset.
    our_ids, our_tokens = [], set()
    for sid, row in our_rows(ours):
        our_ids.append(sid)
        our_tokens.add(key(our_primitive_tokens(row)))

    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"  [{'ok' if passed else 'DIFFERS'}] {name}{(': ' + detail) if detail else ''}")

    print(f"reference {len(ref_ids)} sketches, ours {len(our_ids)}")
    report("same number of sketches", len(ref_ids) == len(our_ids))
    ref_token_set = set(ref_tokens)
    report("same distinct primitive sequences", ref_token_set == our_tokens,
           f"only reference {len(ref_token_set - our_tokens)}, only ours {len(our_tokens - ref_token_set)}")
    ours_set = set(our_ids)
    both = len(ours_set & ref_at.keys())
    report("same sketches kept", both == len(ref_ids) == len(our_ids),
           f"{len(ref_ids) - both} only in reference, {len(our_ids) - both} only in ours "
           "(a different copy of a duplicate kept, where the shards were read in another order)")
    report("same order", ref_ids == our_ids)
    del our_ids, our_tokens, ours_set, ref_token_set

    # Second pass: each sketch both kept, against the reference's tokens for it.
    primitive_differ = constraint_differ = 0
    for sid, row in our_rows(ours):
        i = ref_at.get(sid)
        if i is None:
            continue
        if key(our_primitive_tokens(row)) != ref_tokens[i]:
            primitive_differ += 1
            if primitive_differ <= 3:
                print("    primitives", sid)
        if cval is not None:
            clockwise = [e.type.name == "Arc" and bool(e.clockwise)
                         for e in sketch_from_sequence(sequences[i]).entities.values()]
            if key(our_constraint_tokens(row, clockwise)) != key(cval[coff[i]:coff[i + 1]]):
                constraint_differ += 1
                if constraint_differ <= 3:
                    print("    constraints", sid)
    report("identical primitive tokens", primitive_differ == 0, f"{primitive_differ} of {both} differ")
    if cval is None:
        print(f"  [skipped] constraint tokens: no {constraints_file.name}")
    else:
        report("identical constraint tokens", constraint_differ == 0, f"{constraint_differ} of {both} differ")

    print("EQUIVALENT" if ok else "NOT EQUIVALENT")
    record(ours, ok, len(ref_ids), cval is not None)
    return 0 if ok else 1


def record(ours: Path, ok: bool, count: int, with_constraints: bool) -> None:
    """Writes the verdict into the build's README, in place of the line saying it is not run yet."""
    readme = ours / "README.md"
    if not readme.exists():
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from create_vitruvion_dataset import CHECK_PLACEHOLDER
    what = "primitive and constraint tokens" if with_constraints else "primitive tokens"
    verdict = (f"Checked against Vitruvion's code, run unchanged on the same shards: **equivalent** -- the same "
               f"{count} sketches, in the same order, with identical {what}." if ok else
               "Checked against Vitruvion's code, run unchanged on the same shards: **differences found**; "
               "see the output of `vitruvion_reference/compare.py`.")
    text = readme.read_text(encoding="utf-8")
    if CHECK_PLACEHOLDER in text:
        readme.write_text(text.replace(CHECK_PLACEHOLDER, verdict), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
