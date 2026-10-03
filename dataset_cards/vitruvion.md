# SketchGraphs, Vitruvion selection

Parametric CAD sketches from [SketchGraphs](https://github.com/PrincetonLIPS/SketchGraphs), selected and
normalised by the rules of Vitruvion (Seff et al., *Vitruvion: A Generative Model of Parametric CAD
Sketches*, ICLR 2022). Each sketch is a list of geometric primitives (lines, circles, arcs, points)
and the categorical constraints between them.

Vitruvion's own preprocessed file, `sg_filtered_unique.npy`, is the data PICASSO, DAVINCI, PpaCAD and
CadVLM train and test on. It can no longer be downloaded: the host has returned 404 since 2024. This
dataset rebuilds it from the raw SketchGraphs JSON shards, adds the continuous parameter values next
to the quantised ones, and publishes a fixed split.

## How it relates to Vitruvion's data

The selection, normalisation, quantisation, deduplication and constraint extraction reimplement
Vitruvion's code ([`img2cad/pipeline`](https://github.com/PrincetonLIPS/vitruvion/tree/main/img2cad/pipeline),
[`img2cad/data_utils.py`](https://github.com/PrincetonLIPS/vitruvion/blob/main/img2cad/data_utils.py),
commit `1b91fff`). They were checked against that code, run unchanged on the same shards. The two must
agree on every sketch kept, its order, and every primitive and constraint token; the result of that
check for this build is under [Build](#build).

Two things differ from the data the published papers used:

- **The split is this dataset's own.** Vitruvion draws its split at training time with
  `torch.randperm` (seed `4242424242`). This dataset draws its own, in the same proportions
  (92.5% train, 2.5% val, 5% test), and publishes it as index lists. Results on this test set are
  comparable in distribution to results on Vitruvion's, but the test sketches are not the same ones.
- **Which copy of a duplicate is kept may differ.** Deduplication keeps the first copy in reading
  order. This build reads the shards in name order; Vitruvion read them in whatever order its file
  system listed them, which was not recorded. The set of distinct sketches is the same either way.

## Selection

A sketch is kept when, in this order (a sketch is counted under the first rule it fails):

1. it has at least one entity and at least one constraint;
2. it has 6 to 16 entities, points included, and at most 64 constraints;
3. every entity is a line, circle, arc or point; an ellipse, spline, conic or unknown entity rejects the whole sketch;
4. no constraint is a mirror, a projection, or a linear or circular pattern; one rejects the whole sketch;
5. it can be centred and scaled, i.e. its bounding box is not a single point;
6. no circle or arc has zero radius, no line has zero length, and no arc has its start at its midpoint;
7. its primitive sequence (types, quantised parameters and construction flags, in order) is not one
   an earlier sketch already has.

Construction geometry is kept and flagged.

## Files

| File | Content |
|---|---|
| `sketches.jsonl` | One sketch per line, described below. Line *i* holds the sketch with `index` *i*. |
| `splits.json` | `{"seed": …, "train": [...], "val": [...], "test": [...]}`: sorted sketch indices per split. |
| `metadata.json` | Counts after each selection step, the split sizes, the shards read, and the build configuration. |

## A sketch

```json
{
  "index": 0,
  "source": {"document_id": "014d2229fc59ff947e4cc03d", "part_idx": 13, "sketch_idx": 3},
  "primitives": [
    {"type": "Line", "params": [-0.0765, 0.3719, 0.0289, -0.2599], "bins": [27, 55, 33, 15], "construction": true},
    {"type": "Circle", "params": [-0.0238, 0.0560, 0.1095], "bins": [30, 35, 39], "construction": false}
  ],
  "constraints": [
    {"type": "Coincident", "refs": [[0, "end"], [1, "center"]]},
    {"type": "Horizontal", "refs": [[0, null]]}
  ]
}
```

(Values shortened, and the constraints made up to show the form; the file holds full double precision.)

- **`source`** names the sketch in SketchGraphs: the public Onshape document, the part studio
  within it, and the sketch within that. It is a stable identifier across builds.
- **`primitives`** are in the order the designer created them. Constraints refer to them by position
  in this list.
- **`constraints`** are in construction order: each comes right after the last primitive it refers to.

### Coordinate frame

Each sketch is translated so that its bounding box is centred on the origin, then scaled uniformly
so that the longer side of the bounding box is 1. Every coordinate therefore lies in [-0.5, 0.5].
**y points up**, as in the CAD system; flip it to draw into an image whose y points down.

The bounding box is what the strokes cover:
- a line's two ends;
- a circle's whole circle;
- an arc's two ends, plus each extreme of its circle (leftmost, rightmost, lowest, highest) that it passes;
- a point's position.

Construction geometry counts.

### Primitives

| `type` | `params` | Meaning |
|---|---|---|
| `Line` | `[x1, y1, x2, y2]` | A segment from (x1, y1) to (x2, y2). |
| `Circle` | `[x, y, r]` | Centre (x, y) and radius r. |
| `Arc` | `[xs, ys, xm, ym, xe, ye]` | Start, midpoint and end on the arc. The arc runs **counter-clockwise** from start through midpoint to end. |
| `Point` | `[x, y]` | A point, usually a reference for constraints (e.g. a hole centre). It draws nothing. |

- **`construction`**: `true` for construction geometry. It's a reference for constraints, drawn
  dashed in CAD and in Vitruvion's renders, and not part of the physical outline.
- **`bins`**: the `params` quantised to 64 levels: `bin = floor((v + 0.5) · 64)`, with `v` first
  rounded to 10 decimals, and 0.5 itself put in bin 63. Values a hair outside [-0.5, 0.5] from
  rounding are clipped first. To map a bin back, use its centre: `v ≈ (bin + 0.5) / 64 − 0.5`. A
  radius uses the same bin width, so it only ever falls in the upper half of the range.

### Constraints

Only categorical constraints are kept, those with no numeric value, as in Vitruvion's constraint
model:

| `type` | Typical meaning |
|---|---|
| `Coincident` | Two points coincide, or a point lies on a line or curve. |
| `Concentric` | Two circles or arcs (or a point and one) share a centre. |
| `Equal` | Two lines have equal length, or two circles or arcs equal radius. |
| `Fix` | The geometry is fixed in place. |
| `Horizontal` | One reference: the line is horizontal. Two: the two points share a y coordinate. |
| `Vertical` | One reference: the line is vertical. Two: the two points share an x coordinate. |
| `Midpoint` | Two references: a point is the midpoint of a line or arc. Three: one of three points is the midpoint of the other two. |
| `Normal` | A line is normal to a curve. |
| `Offset` | One curve is an offset of another. |
| `Parallel` | Two lines are parallel. |
| `Perpendicular` | Two lines are perpendicular. |
| `Quadrant` | A point lies at a quadrant point of a circle or arc. |
| `Tangent` | Two curves, or a line and a curve, are tangent. |

Left out:
- constraints with a value (distance, angle, length, radius, diameter);
- constraints on geometry outside the sketch, such as the sketch's axes or projected edges.

A sketch with a mirror, projection or pattern constraint is not in the dataset at all (rule 4).

**`refs`** lists what a constraint acts on, as `[primitive, point]`:
- `primitive` is an index into `primitives`;
- `point` is `null` for the primitive as a whole, or the specific point meant:

| Primitive | Points a constraint can name |
|---|---|
| `Line` | `start` = (x1, y1), `end` = (x2, y2) |
| `Circle` | `center` |
| `Arc` | `start` = (xs, ys), `end` = (xe, ye), `center` (the centre of its circle) |
| `Point` | none: always `null` |

Points are named after the record's own parameters. For an arc Onshape stores clockwise, the record
swaps start and end so that it runs counter-clockwise, and its point references are swapped with it.

**Reference order carries no meaning.** As in Vitruvion, the references of a constraint are sorted
by their position in the construction sequence, not by role. For symmetric constraints this loses
nothing. For a two-reference `Midpoint`, the roles can still be read off: the point, and the line
or arc. For a three-reference `Midpoint`, which of the three points is the midpoint is not recorded.

Most constraints have two references. `Horizontal`, `Vertical` and `Fix` can have one, and
`Midpoint` can have three.

## Differences from what the Vitruvion paper describes

The paper leaves details open. The code decides them, and this dataset follows the code:

- an unsupported entity, or a mirror, projection or pattern constraint, rejects the **whole sketch**; nothing is removed from a sketch;
- the 6–16 range counts all entities, points included, before anything else is checked;
- the construction flag is part of the deduplication key;
- the "non-empty constraint set" rule counts constraints of any type, before any are left out.

## Licence and citation

The sketches come from public Onshape documents via SketchGraphs. As SketchGraphs states, the original
creators of the CAD sketches hold the copyright; see the
[Onshape Terms of Use §1.g.ii](https://www.onshape.com/legal/terms-of-use#your_content). Please cite
SketchGraphs and Vitruvion when using this data:

```bibtex
@article{seff2020sketchgraphs,
  title   = {SketchGraphs: A Large-Scale Dataset for Modeling Relational Geometry in Computer-Aided Design},
  author  = {Seff, Ari and Ovadia, Yaniv and Zhou, Wenda and Adams, Ryan P.},
  journal = {arXiv preprint arXiv:2007.08506},
  year    = {2020}
}
@inproceedings{seff2022vitruvion,
  title     = {Vitruvion: A Generative Model of Parametric {CAD} Sketches},
  author    = {Seff, Ari and Zhou, Wenda and Richardson, Nick and Adams, Ryan P.},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

## Build
