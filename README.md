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
