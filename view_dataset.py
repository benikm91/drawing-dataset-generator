import argparse
import base64
import io
import json
import threading
from pathlib import Path
from typing import Dict

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Dataset Viewer</title>
    <style>
        :root {
            --bg: #f5f1e8;
            --panel: #fffdf9;
            --ink: #161514;
            --muted: #6d665c;
            --line: #ddd3c5;
            --accent: #186a73;
            --accent-soft: #d8eff1;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
            background:
                radial-gradient(circle at top right, #fff4d7 0, transparent 26%),
                radial-gradient(circle at bottom left, #dcebed 0, transparent 28%),
                var(--bg);
            color: var(--ink);
        }
        .shell {
            max-width: 1320px;
            margin: 0 auto;
            padding: 28px;
        }
        .header {
            margin-bottom: 22px;
        }
        .title {
            margin: 0;
            font-size: 40px;
            line-height: 1;
        }
        .subtitle {
            margin: 10px 0 0;
            font-size: 16px;
            color: var(--muted);
        }
        .layout {
            display: grid;
            grid-template-columns: minmax(360px, 1.2fr) minmax(320px, 0.9fr);
            gap: 22px;
            align-items: start;
        }
        .panel {
            background: color-mix(in srgb, var(--panel) 92%, white 8%);
            border: 1px solid var(--line);
            border-radius: 22px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(30, 26, 20, 0.08);
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            padding: 18px;
            border-bottom: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(245,241,232,0.72));
        }
        .field {
            display: grid;
            gap: 6px;
        }
        label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
        }
        select, input, button {
            border-radius: 12px;
            border: 1px solid var(--line);
            padding: 12px 14px;
            font: inherit;
            background: white;
            color: var(--ink);
        }
        button {
            cursor: pointer;
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }
        button.secondary {
            background: white;
            color: var(--ink);
        }
        button:hover { transform: translateY(-1px); }
        .nav-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            padding: 0 18px 18px;
        }
        .nav-row button {
            min-width: 130px;
        }
        .viewer {
            padding: 18px;
        }
        .viewer-meta {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
            font-size: 14px;
            color: var(--muted);
        }
        .viewer-frame {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--line);
            background: #000;
        }
        .viewer-frame img {
            width: 100%;
            display: block;
            aspect-ratio: 1 / 1;
            object-fit: contain;
            image-rendering: pixelated;
        }
        .range-wrap {
            padding: 0 18px 18px;
        }
        input[type="range"] {
            width: 100%;
            padding: 0;
        }
        .sidebar {
            padding: 18px;
        }
        .info-block + .info-block {
            margin-top: 18px;
        }
        .info-title {
            margin: 0 0 8px;
            font-size: 18px;
        }
        .info-copy {
            margin: 0;
            line-height: 1.45;
            color: var(--muted);
            word-break: break-word;
        }
        .code {
            margin: 0;
            padding: 14px;
            border-radius: 16px;
            background: #201b17;
            color: #f7efe2;
            overflow: auto;
            font: 13px/1.5 Menlo, Consolas, monospace;
        }
        .status {
            padding: 0 18px 18px;
            font-size: 13px;
            color: var(--muted);
        }
        @media (max-width: 980px) {
            .layout { grid-template-columns: 1fr; }
        }
        #sampleLabel {
            display: block;
            color: white;
        }  
    </style>
</head>
<body>
    <div class="shell">
        <header class="header">
            <h1 class="title">Dataset Viewer</h1>
            <p class="subtitle">Browse rendered samples from a saved dataset. Use the buttons, slider, or arrow keys to iterate.</p>
        </header>

        <div class="layout">
            <section class="panel">
                <div class="controls">
                    <div class="field" style="grid-column: 1 / -1;">
                        <label for="datasetDir">Dataset Folder</label>
                        <input id="datasetDir" type="text" placeholder="dataset">
                    </div>
                    <div class="field">
                        <label for="split">Split</label>
                        <select id="split"></select>
                    </div>
                    <div class="field">
                        <label for="indexInput">Index</label>
                        <input id="indexInput" type="number" min="0" step="1" value="0">
                    </div>
                    <div class="field">
                        <label for="loadDatasetButton">Folder</label>
                        <button id="loadDatasetButton" type="button" class="secondary">Load Folder</button>
                    </div>
                    <div class="field">
                        <label for="jumpButton">Sample</label>
                        <button id="jumpButton" type="button">Load Sample</button>
                    </div>
                </div>
                <div class="nav-row">
                    <button id="prevButton" type="button" class="secondary">Previous</button>
                    <button id="nextButton" type="button">Next</button>
                    <button id="randomButton" type="button" class="secondary">Random</button>
                </div>
                <div class="viewer">
                    <div class="viewer-meta">
                        <span id="sampleMeta">Loading dataset...</span>
                        <span id="shapeMeta"></span>
                    </div>
                    <div class="viewer-frame">
                        <img id="sampleImage" alt="Dataset sample">
                    </div>
                    <div class="viewer-frame">
                        <span id="sampleLabel" alt="Dataset label"></span>
                    </div>
                </div>
                <div class="range-wrap">
                    <input id="indexRange" type="range" min="0" max="0" value="0">
                </div>
                <div id="status" class="status">Arrow keys move through the selected split.</div>
            </section>

            <aside class="panel sidebar">
                <div class="info-block">
                    <h2 class="info-title">Dataset</h2>
                    <p id="datasetPath" class="info-copy"></p>
                </div>
                <div class="info-block">
                    <h2 class="info-title">Metadata</h2>
                    <p id="datasetSummary" class="info-copy"></p>
                </div>
                <div class="info-block">
                    <h2 class="info-title">Generator</h2>
                    <pre id="generatorDefinition" class="code"></pre>
                </div>
                <div class="info-block">
                    <h2 class="info-title">Raw Metadata</h2>
                    <pre id="rawMetadata" class="code"></pre>
                </div>
            </aside>
        </div>
    </div>

    <script>
        const splitSelect = document.getElementById('split');
        const datasetDirInput = document.getElementById('datasetDir');
        const loadDatasetButton = document.getElementById('loadDatasetButton');
        const indexInput = document.getElementById('indexInput');
        const jumpButton = document.getElementById('jumpButton');
        const prevButton = document.getElementById('prevButton');
        const nextButton = document.getElementById('nextButton');
        const randomButton = document.getElementById('randomButton');
        const indexRange = document.getElementById('indexRange');
        const sampleImage = document.getElementById('sampleImage');
        const sampleMeta = document.getElementById('sampleMeta');
        const shapeMeta = document.getElementById('shapeMeta');
        const status = document.getElementById('status');
        const datasetPath = document.getElementById('datasetPath');
        const datasetSummary = document.getElementById('datasetSummary');
        const generatorDefinition = document.getElementById('generatorDefinition');
        const rawMetadata = document.getElementById('rawMetadata');

        let datasetInfo = null;

        function splitCount(splitName) {
            return Number(datasetInfo?.splits?.[splitName] || 0);
        }

        function clampIndex(splitName, index) {
            const count = splitCount(splitName);
            if (count === 0) {
                return 0;
            }
            return Math.max(0, Math.min(count - 1, index));
        }

        function syncIndexControls(splitName, index) {
            const maxIndex = Math.max(0, splitCount(splitName) - 1);
            indexInput.max = String(maxIndex);
            indexRange.max = String(maxIndex);
            indexInput.value = String(index);
            indexRange.value = String(index);
        }

        function setDatasetSummary(info) {
            datasetInfo = info;
            datasetDirInput.value = datasetInfo.dataset_dir;
            datasetPath.textContent = datasetInfo.dataset_dir;
            datasetSummary.textContent = `${datasetInfo.task} · ${datasetInfo.image_shape.join('x')} · ${datasetInfo.image_dtype}`;
            generatorDefinition.textContent = datasetInfo.generator_definition;
            rawMetadata.textContent = JSON.stringify(datasetInfo.metadata, null, 2);
            splitSelect.innerHTML = '';
            for (const splitName of Object.keys(datasetInfo.splits)) {
                const option = document.createElement('option');
                option.value = splitName;
                option.textContent = `${splitName} (${datasetInfo.splits[splitName]})`;
                splitSelect.appendChild(option);
            }
            syncIndexControls(splitSelect.value, 0);
        }

        async function loadDatasetInfo(datasetDir = null) {
            const endpoint = datasetDir === null ? '/api/dataset' : '/api/dataset';
            const response = datasetDir === null
                ? await fetch(endpoint)
                : await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_dir: datasetDir }),
                });
            const payload = await response.json();
            if (!response.ok) {
                status.textContent = payload.error || 'Failed to load dataset folder.';
                return false;
            }
            setDatasetSummary(payload);
            status.textContent = 'Dataset loaded.';
            return true;
        }

        async function loadSample(requestedIndex = null) {
            const splitName = splitSelect.value;
            const parsed = requestedIndex === null ? Number(indexInput.value || 0) : Number(requestedIndex);
            const index = clampIndex(splitName, parsed);
            syncIndexControls(splitName, index);
            status.textContent = 'Loading sample...';
            const params = new URLSearchParams({ split: splitName, index: String(index) });
            const response = await fetch(`/api/sample?${params.toString()}`);
            const sample = await response.json();
            if (!response.ok) {
                status.textContent = sample.error || 'Failed to load sample.';
                return;
            }
            sampleImage.src = sample.image;
            sampleMeta.textContent = `${sample.split} sample ${sample.index + 1} / ${sample.total}`;
            sampleLabel.innerHTML = `${sample.label}`;
            shapeMeta.textContent = `seed ${sample.seed}`;
            status.textContent = 'Loaded. ArrowLeft and ArrowRight navigate within the current split.';
            syncIndexControls(sample.split, sample.index);
        }

        function step(delta) {
            loadSample(Number(indexInput.value || 0) + delta);
        }

        jumpButton.addEventListener('click', () => loadSample());
        loadDatasetButton.addEventListener('click', async () => {
            const loaded = await loadDatasetInfo(datasetDirInput.value.trim());
            if (loaded) {
                await loadSample(0);
            }
        });
        prevButton.addEventListener('click', () => step(-1));
        nextButton.addEventListener('click', () => step(1));
        randomButton.addEventListener('click', () => {
            const count = splitCount(splitSelect.value);
            const index = count <= 1 ? 0 : Math.floor(Math.random() * count);
            loadSample(index);
        });
        splitSelect.addEventListener('change', () => loadSample(0));
        indexRange.addEventListener('input', () => loadSample(Number(indexRange.value)));
        indexInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                loadSample();
            }
        });
        datasetDirInput.addEventListener('keydown', async (event) => {
            if (event.key === 'Enter') {
                const loaded = await loadDatasetInfo(datasetDirInput.value.trim());
                if (loaded) {
                    await loadSample(0);
                }
            }
        });

        window.addEventListener('keydown', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
                return;
            }
            if (event.key === 'ArrowLeft') {
                event.preventDefault();
                step(-1);
            }
            if (event.key === 'ArrowRight') {
                event.preventDefault();
                step(1);
            }
        });

        window.addEventListener('DOMContentLoaded', async () => {
            await loadDatasetInfo();
            await loadSample(0);
        });
    </script>
</body>
</html>
"""


class DatasetBundle:
    def __init__(self, dataset_dir: Path):

        def load_labels(path: Path) -> list:
            labels = []
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            labels.append(json.loads(line)["actions"])
            return labels
        self.dataset_dir = Path(dataset_dir).resolve()
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.splits = {}
        for split_name, count in self.metadata.get("splits", {}).items():
            images = np.load(self.dataset_dir / f"{split_name}_images.npy", mmap_mode="r")
            labels = load_labels(self.dataset_dir / f"{split_name}_labels.jsonl")
            seeds = np.load(self.dataset_dir / f"{split_name}_seeds.npy", mmap_mode="r")
            if len(images) != count or len(labels) != count or len(seeds) != count:
                raise ValueError(f"Split {split_name} does not match metadata counts.")
            self.splits[split_name] = {
                "images": images,
                "labels": labels,
                "seeds": seeds,
                "count": count,
            }

    def sample(self, split_name: str, index: int) -> Dict[str, object]:
        def round_floats(data):
            if isinstance(data, dict):
                return {k: round_floats(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [round_floats(v) for v in data]
            elif isinstance(data, float):
                return round(data, 3)
            else:
                return data
        def visualize(actions: list, indent: int = 4) -> str:
            return '<br>'.join([json.dumps(action, separators=(', ', ': ')) for action in actions])
        if split_name not in self.splits:
            raise KeyError(f"Unknown split: {split_name}")
        count = int(self.splits[split_name]["count"])
        if count == 0:
            raise ValueError(f"Split {split_name} is empty")
        index = max(0, min(count - 1, index))
        images = self.splits[split_name]["images"]
        labels = self.splits[split_name]["labels"]
        seeds = self.splits[split_name]["seeds"]
        image = np.asarray(images[index], dtype=np.uint8)
        return {
            "split": split_name,
            "index": index,
            "total": count,
            "label": visualize(round_floats(json.loads(labels[index]))),
            "seed": int(seeds[index]),
            "image": image_to_data_url(image),
        }

    def summary(self) -> Dict[str, object]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "task": self.metadata.get("task", "unknown"),
            "image_dtype": self.metadata.get("image_dtype", "unknown"),
            "image_shape": self.metadata.get("image_shape", []),
            "splits": {split_name: int(split["count"]) for split_name, split in self.splits.items()},
            "generator_definition": self.metadata.get("generator_definition", ""),
            "metadata": self.metadata,
        }


class DatasetState:
    def __init__(self, dataset_dir: Path):
        self._lock = threading.Lock()
        self._bundle = DatasetBundle(dataset_dir)

    def load(self, dataset_dir: Path) -> Dict[str, object]:
        bundle = DatasetBundle(dataset_dir)
        with self._lock:
            self._bundle = bundle
            return bundle.summary()

    def summary(self) -> Dict[str, object]:
        with self._lock:
            return self._bundle.summary()

    def sample(self, split_name: str, index: int) -> Dict[str, object]:
        with self._lock:
            return self._bundle.sample(split_name, index)


def image_to_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_app(dataset_dir: Path) -> Flask:
    state = DatasetState(dataset_dir)
    app = Flask(__name__)
    workspace_root = Path.cwd().resolve()

    def resolve_dataset_dir(raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_TEMPLATE)

    @app.get("/api/dataset")
    def dataset_info():
        return jsonify(state.summary())

    @app.post("/api/dataset")
    def dataset_reload():
        payload = request.get_json(force=True, silent=True) or {}
        raw_dir = str(payload.get("dataset_dir") or "").strip()
        if not raw_dir:
            return jsonify({"error": "dataset_dir is required."}), 400
        try:
            return jsonify(state.load(resolve_dataset_dir(raw_dir)))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/sample")
    def sample():
        summary = state.summary()
        split_name = request.args.get("split", default=next(iter(summary["splits"].keys())))
        index = int(request.args.get("index", default=0))
        try:
            return jsonify(state.sample(split_name, index))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse a saved dataset in a local web UI.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.dataset_dir)
    print(f"Dataset viewer available at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()