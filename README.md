# Drawing Generator

Repository for generating synthetic l-shape datasets.
Dataset forms a baseline for graph prediction task motivated by the document-recognition as transcription framework.

Create a dataset with:
```
uv run  create_dataset.py --output-dir ./dataset --full --canvas-size 256
```

View generated dataset in a Web-App with:
```
uv run view_dataset.py
```

View generator capabilities with:
```
uv run main.py
```
