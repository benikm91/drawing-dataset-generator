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
