---
pretty_name: SketchGraphs, Vitruvion selection (rebuilt)
license: other
license_name: onshape-terms-of-use
license_link: https://www.onshape.com/legal/terms-of-use#your_content
size_categories:
  - 1M<n<10M
tags:
  - cad
  - parametric-cad
  - sketches
  - geometric-constraints
  - sketchgraphs
  - vitruvion
---

# SketchGraphs, Vitruvion selection (`sg_filtered_unique.npy`, rebuilt)

This is the dataset of Vitruvion (Seff et al., *Vitruvion: A Generative Model of Parametric CAD
Sketches*, ICLR 2022): 1,643,604 unique parametric CAD sketches from
[SketchGraphs](https://github.com/PrincetonLIPS/SketchGraphs), with their primitives and constraints.
PICASSO, DAVINCI, PpaCAD and CadVLM train and test on it.

Vitruvion published it as `sg_filtered_unique.npy`. Its download link has returned 404 since 2024,
and no copy is archived. This file was rebuilt by running **Vitruvion's own preprocessing code,
unchanged**, on the raw SketchGraphs JSON shards. It has the same format and the same name, so code
written for the original file, including Vitruvion's own (`sequence_path=…/sg_filtered_unique.npy`),
reads it as is.

![The clean renders of the first 256 training sketches](https://huggingface.co/datasets/benikm91/sketch-graph-vitruvion/resolve/main/train_first_256.png)

*The clean renders of the first 256 training sketches.*

![Sixteen training sketches with their constraints marked](https://huggingface.co/datasets/benikm91/sketch-graph-vitruvion/resolve/main/train_constraints_4x4.png)

*Sixteen training sketches drawn from their labels, with the categorical constraints marked as a
CAD program marks them: a dot where points coincide, and a letter on the primitive or on dotted
connectors between the points it ties together. They are the first training sketches with 6 to 10
primitives, an arc or a circle, and 6 to 14 constraints of at least 3 types.*

## How it was built

- **Code:** Vitruvion at commit `1b91fff`, running `img2cad.pipeline.filter_sequences_from_source`
  and then `img2cad.pipeline.tokenize_sequences`. Their repository is missing the `sketchgraphs/data`
  module because its `.gitignore` excludes `data/` folders, so that module was taken from SketchGraphs
  at commit `1f27f5f`, the "img2cad-compatible version". No line of either was changed.
- **Environment:** a container matching their `nvcr.io/nvidia/pytorch:21.03-py3` image on the CPU:
  Python 3.8, numpy 1.20.3, torch 1.9.0, hydra 1.1.0.
- **Input:** all 128 raw JSON shards from `https://sketchgraphs.cs.princeton.edu/shards/`, read in
  the order listed in `shard_order.txt`.

The recipe is in `vitruvion_reference/` of
[drawing-dataset-generator](https://github.com/benikm91/drawing-dataset-generator) (`run_vitruvion.sh reference`).

### What each step kept

A sketch is counted under the first rule it fails:

| Step | Rule | Rejected | Left |
|---|---|---:|---:|
| read | | | 15,945,174 |
| Empty | no constraints, or no entities | 924,256 | 15,020,918 |
| TooManyEntities | more than 16 entities (points count) | 1,939,991 | 13,080,927 |
| TooFewEntities | fewer than 6 entities | 8,588,308 | 4,492,619 |
| TooManyConstraints | more than 64 constraints | 1,027 | 4,491,592 |
| InvalidEntityType | an ellipse, spline, conic or unknown entity | 266,796 | 4,224,796 |
| InvalidConstraintType | a mirror, projection or linear / circular pattern constraint | 1,537,783 | 2,687,013 |
| CannotNormalize | bounding box is a single point | 78 | 2,686,935 |
| ZeroSizedEntity | zero radius, zero-length line, or arc starting at its midpoint | 61,041 | 2,625,894 |
| duplicate | same primitive sequence as an earlier sketch | 982,290 | **1,643,604** |

Deduplication compares each sketch's primitives as Vitruvion tokenises them: types, parameters
quantised to 64 bins, and construction flags, in order. It keeps the first copy.

## How it relates to the original file

The original file is gone, so the two cannot be compared directly. What is known:

- **The selection is the same.** The same code ran on the same source data. The size matches what
  the papers report:
  - Vitruvion gives "1.7 million unique sketches"; this file has 1,643,604.
  - PICASSO and DAVINCI give "1.53 million", which matches this file's training split of 1,520,334
    under Vitruvion's split procedure.
- **Which copy of a duplicate was kept can differ.** Deduplication keeps the first copy in reading
  order, and Vitruvion's code reads shards in whatever order the file system lists them. The original
  run did not record that order; this run's is in `shard_order.txt`. Another run in another order
  keeps the same set of distinct sketches, but a different copy for some of them. Measured against a
  run in shard name order:
  - 7.4% of sketches were a different copy;
  - their quantised primitives are identical by construction;
  - their continuous parameters differ within the same bin;
  - **0.9% of sketches** have different constraints, since constraints are not part of the
    deduplication key.

  Expect differences of that size against the lost original.
- **The split is reproducible from this file.** Vitruvion draws its split at training time from the
  file's order (`img2cad.primitives_data.split_dataset`: `torch.randperm`, seed `4242424242`). Run on
  this file, that gives the split in `splits.json`. Because the original file's order is unknown,
  it is not the original split; it is the split anyone using Vitruvion's code on this file gets.

**Independent check.** A separate reimplementation of the same rules
(`create_vitruvion_dataset.py` in the same repository), run on the same shards, keeps the same
1,643,604 distinct sketches. For every sketch both runs kept as the same copy, it produces identical
primitive tokens and identical constraint tokens. The only differences are which copy of a duplicate
was kept, from reading the shards in a different order.

## Files

| File | Content |
|---|---|
| `sg_filtered_unique.npy` | The sketches, in Vitruvion's format (below). 1.9 GB. |
| `splits.json` | `train`, `val` and `test`: sorted indices into the file, as drawn by Vitruvion's `split_dataset`. 1,520,334 / 41,089 / 82,181. |
| `steps.json` | The counts in the table above. |
| `shard_order.txt` | The order the 128 shards were read in. |
| `train_first_256.png`, `train_constraints_4x4.png` | The two preview images above (`preview_vitruvion.py`). |
| `renders/render_p128_XX_of_16.npy` | Vitruvion's renders of every sketch, in 16 chunks: one clean and five hand-drawn, 128 × 128 px (below). |

## Reading it

With Vitruvion's code, pass the file as `sequence_path`. With the SketchGraphs library
(`pip install git+https://github.com/PrincetonLIPS/SketchGraphs`):

```python
from sketchgraphs.data import flat_array
from sketchgraphs.data.sequence import sketch_from_sequence

data = flat_array.load_dictionary_flat("sg_filtered_unique.npy")
data["sketch_ids"][0]          # (b'75d54a957713f8da949d9c8c', 0, 0): document, part, sketch
seq = data["sequences"][0]     # the construction sequence: a list of NodeOp / EdgeOp
sketch = sketch_from_sequence(seq)
```

The file holds three arrays:
- **`sequences`**: one construction sequence per sketch.
- **`sketch_ids`**: the source of each sketch: the public Onshape document ID, the part studio
  index within it, and the sketch index within that. Together they identify a sketch across builds.
- **`sequence_lengths`**: the number of operations in each sequence.

### Construction sequences

A sequence lists the sketch's primitives in the order the designer created them, each followed by
the constraints that become complete with it:

```
NodeOp(External)                       # node 0: geometry outside the sketch (origin, axes)
NodeOp(Line, {pntX, pntY, dirX, dirY, startParam, endParam, isConstruction})   # node 1
EdgeOp(Horizontal, references=(1,))
NodeOp(SN_Start)                       # node 2: the line's start point
EdgeOp(Subnode, references=(2, 1))     # node 2 belongs to node 1
EdgeOp(Coincident, references=(2, 0))  # the start point lies on the origin
...
```

- **`NodeOp`** adds a node. Nodes are numbered from 0 in the order they appear.
  - Node 0 is always `External`.
  - Each primitive node is followed by its point nodes: a line's `SN_Start` and `SN_End`; a
    circle's `SN_Center`; an arc's `SN_Center`, `SN_Start` and `SN_End`. A point has none.
- **`EdgeOp`** adds a constraint. Its `references` are the node numbers it acts on, and its
  `parameters` hold any value it has, such as a distance.
  - `Subnode` edges only tie a point node to its primitive.
  - References to node 0 are constraints against external geometry.

### Primitive parameters

The parameters are SketchGraphs' (Onshape's) over-parameterisation:

| Primitive | Parameters |
|---|---|
| `Line` | `pntX, pntY`: a point on the line; `dirX, dirY`: unit direction; `startParam, endParam`: signed distances of the ends from that point |
| `Circle` | `xCenter, yCenter, radius`; `xDir, yDir, clockwise`: an orientation the circle carries |
| `Arc` | as a circle, plus `startParam, endParam`: the angles of its ends from `(xDir, yDir)`, counted clockwise where `clockwise` is set |
| `Point` | `x, y` |

Every primitive also has `isConstruction`. When `true`, the primitive is construction geometry: a
reference for constraints, drawn dashed, not part of the outline. The SketchGraphs entity classes
give a line's `start_point` and `end_point`, and an arc's `start_point`, `mid_point`, `end_point`
and `center_point`.

**Coordinates are normalised.** Each sketch is centred on its bounding box and scaled so the
bounding box's longer side is 1, with y pointing up. The bounding box covers:
- a line's ends;
- a circle's whole circle;
- an arc's ends, plus each of its circle's extremes that it passes;
- points.

Vitruvion's loaders normalise again on reading, which leaves the values as they are up to rounding.

### What Vitruvion's models use

The sequences keep every constraint of the sketch. Vitruvion's models use a subset:

- **Primitive model.** Each primitive is parameterised minimally and quantised to 64 bins over
  [-0.5, 0.5]:
  - line `(x1, y1, x2, y2)`;
  - circle `(x, y, r)`;
  - arc `(start, mid, end)`, swapped to run counter-clockwise when `clockwise` is set;
  - point `(x, y)`;
  - plus the construction flag.
- **Constraint model.** It keeps only the categorical constraints: `Coincident`, `Concentric`,
  `Equal`, `Fix`, `Horizontal`, `Midpoint`, `Normal`, `Offset`, `Parallel`, `Perpendicular`,
  `Quadrant`, `Tangent`, `Vertical`. It skips any constraint that refers to node 0, and sorts each
  constraint's references by node number.

  Numerical constraints (`Distance`, `Length`, `Angle`, `Radius`, `Diameter`, …) are in the file
  but unused by it. So are `Subnode` edges and external references.

## Renders

Rendered with Vitruvion's own renderer, `img2cad.pipeline.prerender_images`, the way their cluster
ran it (`img2cad/pipeline/render_images.sh`). Each sketch gets one clean render and five hand-drawn
ones, using Vitruvion's noise model. Each is 128 × 128 px, 8-bit grayscale PNG, black on white,
with construction geometry dashed and points as dots. The 16 chunks split the sketches into
contiguous ranges, named like the files Vitruvion published (`render_p128_01_of_16.npy`, …).

```python
import io
from PIL import Image
from sketchgraphs.data import flat_array

chunk = flat_array.load_dictionary_flat("renders/render_p128_01_of_16.npy")
chunk["indexes"][:6]           # the sketch each image is of: [0 0 0 0 0 0]
Image.open(io.BytesIO(chunk["imgs"][0]))   # sketch 0, clean; imgs[1..5] hand-drawn
```

The published renderer does not run as is. In `process_sketch_sequence`, it calls
`render_sketch(sketch, pad, size_pixels, …)` by position, but `render_sketch` gained an `ax`
parameter in second place. The padding lands in `ax`, and rendering fails on it. The two calls were
changed to pass `pad=pad, size_pixels=size_pixels` by name (`vitruvion_reference/patch_prerender.py`);
nothing else was changed.

**Where a sketch lands in a render.** The renderer fixes the limits at ±0.6 and lays the figure out
with matplotlib's `tight_layout`, which leaves an uneven margin, so a sketch does not fill the
image. Measured on the renderer, a sketch coordinate (x, y) in [-0.6, 0.6] lands at, in pixels:
- `px = 25.262 + (x + 0.6) / 1.2 · 83.378`, from the left;
- `py = 128 − (25.422 + (y + 0.6) / 1.2 · 83.378)`, from the top.

This mapping puts every drawn point of 400 test sketches on a stroke pixel.

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
