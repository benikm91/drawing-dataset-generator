# Drawing Generator

Repository for generating synthetic drawing datasets.
Datasets form a baseline for graph prediction task motivated by the document-recognition as transcription framework.

## l-shapes

The first rung: every part is the same six-segment L.

```
uv run  create_dataset.py --output-dir ./dataset --full --canvas-size 256
```

## rectilinear parts

The rung above. The part is a general rectilinear polygon grown as a polyomino, so it is one
connected area with no holes and no self-touching, and its boundary has as many edges as the
complexity asks for. `--n-cells` is the difficulty dial.

```
uv run create_rectilinear_dataset.py --output-dir ./dataset_rect --full --canvas-size 256
uv run create_rectilinear_dataset.py --output-dir ./dataset_rect_hard --n-cells 24 --max-edges 32
```

Every sample is checked before it is written — one closed area, consecutive edges turn, no edge
under `--min-edge-length`, nothing closer than `--min-gap`, and no stroke crossing another it does
not share a corner with. A sample that fails is drawn again rather than repaired, so the record a
drawing came from is always the only record that explains it. The checks live in `validity.py` and
can be run on any action list with `why_invalid`.

## chamfered parts

The rung that changes what a node is rather than how many there are. The part is the rectilinear
one, but a share of its corners is cut off: each chamfer replaces a corner by one straight line
between a point on each of the two edges that met there, so the outline gains edges that are
neither horizontal nor vertical and whose endpoints no longer sit on the lattice. The two legs of
a cut are drawn independently, in cells, so the chamfer's angle is continuous rather than a fixed
45°; every edge keeps `--chamfer-keep` cells of its length, so an edge cut at both ends does not
vanish. `--chamfer-ratio` is how many corners are offered a cut, and `0` is the rectilinear rung
exactly.

```
uv run create_rectilinear_dataset.py --output-dir ./dataset_chamfer --chamfer-ratio 0.6 --full --canvas-size 256
uv run create_rectilinear_dataset.py --output-dir ./dataset_chamfer_hard --chamfer-ratio 1.0 --n-cells 24 --max-edges 32
```

The same checks apply, with one more: consecutive edges must turn by at least `--min-turn`
degrees, so a chamfer never lies close enough to an edge to read as its continuation. A chamfer
carries no outward normal, so it is never dimensioned. `--max-edges` bounds the lattice outline
before the cuts; a chamfered outline has up to twice that many edges.

## CAD sketches

Not generated but taken: the [SketchGraphs](https://github.com/PrincetonLIPS/SketchGraphs) sketches
that draw nothing but lines and circles, placed on the canvas and drawn with the same renderer, in
the same files. A sketch is kept only when every entity is a line or a circle, none of them
construction geometry, and there are `--min-nodes` to `--max-nodes` of them; the floor drops the
lone circles and bare rectangles the source is mostly made of. Each label carries the Onshape
document, part and sketch it came from.

```
uv run create_sketch_dataset.py --output-dir ./dataset_sketches --canvas-size 256
uv run create_sketch_dataset.py --output-dir ./dataset_sketches_small --limit 131072
```

The sequences (6.2 GB for train) download into `--sequences-dir` once. Kept in full, train is
about 1.15M drawings and 75 GB of images; `--limit 131072` is the size of the l-shape corpus.

## CAD sketches with arcs

The same sketches, widened to every primitive the published sequences draw: lines, circles and
arcs. An arc is recorded as three points on it — start, halfway, end — read clockwise on the
drawing, since its two ends alone leave open which circle it lies on. Sketch points are skipped
rather than refused, as they draw nothing; ellipses and splines never appear, because SketchGraphs
drops every sketch holding one when it builds the sequences.

```
uv run create_sketch_full_primitives_dataset.py --output-dir ./dataset_sketches_full --canvas-size 256
uv run create_sketch_full_primitives_dataset.py --output-dir ./dataset_sketches_full_small --limit 131072
```

## Vitruvion's selection of SketchGraphs

The data PICASSO, DAVINCI, PpaCAD and CadVLM train and test on is Vitruvion's preprocessed
SketchGraphs file, which can no longer be downloaded. `create_vitruvion_dataset.py` rebuilds it
from the raw SketchGraphs JSON shards with Vitruvion's rules, read from their code. It writes the
primitives with continuous and quantised parameters, the categorical constraints, and a split of
its own. What the labels mean is in [dataset_cards/vitruvion.md](dataset_cards/vitruvion.md), which
each build copies into its folder as `README.md`, with its counts.

`vitruvion_reference/` runs Vitruvion's own pipeline unchanged in Docker on the same shards, and
compares the two sketch by sketch: same sketches, same order, identical primitive and constraint
tokens.

```
vitruvion_reference/run_vitruvion.sh all                          # download, build both, compare
SHARDS="1 2" SUFFIX=rehearsal vitruvion_reference/run_vitruvion.sh all   # the same on two shards
```

The full run needs Docker Desktop running and about 55 GB: 43 GB of shards, kept in
`~/.detr-cache/input/sketchgraphs/shards`, and the two outputs.

`dataset_sketches_vitruvion/` holds what Vitruvion's own code produced on all 128 shards: the
rebuilt `sg_filtered_unique.npy` (1,643,604 sketches), the split Vitruvion's `split_dataset` draws
from it (`vitruvion_reference/ref_split.py`), the order the shards were read in, and its card,
[dataset_cards/sketches_vitruvion.md](dataset_cards/sketches_vitruvion.md).

Its renders come from Vitruvion's own renderer, one clean and five hand-drawn per sketch, into
`renders/`. The step can be stopped and restarted; about 21 hours on 4 CPUs, less with more CPUs
given to Docker:
```
vitruvion_reference/run_vitruvion.sh render
```

Browse it, with the renders, primitives and constraints, and hover any of them to light it up on
the drawing:
```
uv run view_vitruvion.py --dataset-dir ./dataset_sketches_vitruvion
```

View generated dataset in a Web-App with:
```
uv run view_dataset.py --dataset-dir ./dataset_rect
```

Write a sheet of many drawings at once, to see whether a whole rung reads clearly:
```
uv run preview_dataset.py --dataset-dir ./dataset_rect --rows 4 --cols 6 --label
```

View generator capabilities with:
```
uv run main.py
```
