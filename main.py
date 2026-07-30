import argparse
import ast
import base64
import io
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from generator import (
    AddIdGenerator,
    AngularSortedPolygonGenerator,
    AnnotateLineGenerator,
    AnnotationTextRefId,
    CentricCircleGenerator,
    ConnectTwoElementsWithId,
    Element,
    Generator,
    GridGenerator,
    HorizontalCircleGenerator,
    HorizontalLineGenerator,
    LShapeGenerator,
    LShapeOutlineGenerator,
    MarginGenerator,
    PartLineWithId,
    RandomRotationAugmentation,
    RectangleGenerator,
)
from renderer import InOrderStaticRenderer, NoisyRenderer, StaticRenderer


matplotlib.use("Agg")
import matplotlib.pyplot as plt


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Generator Inspector</title>
    <style>
        :root {
            --bg: #f4efe7;
            --panel: #fffdf8;
            --ink: #181613;
            --muted: #6a6258;
            --line: #d8cfbf;
            --accent: #b0442f;
            --accent-soft: #f5d8c8;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
            color: var(--ink);
            background:
                radial-gradient(circle at top left, #fff4de 0, transparent 28%),
                radial-gradient(circle at bottom right, #eadfce 0, transparent 26%),
                var(--bg);
        }
        .shell {
            max-width: 1380px;
            margin: 0 auto;
            padding: 28px;
        }
        .header {
            margin-bottom: 24px;
        }
        .title {
            margin: 0;
            font-size: 40px;
            line-height: 1;
        }
        .subtitle {
            margin: 10px 0 0;
            color: var(--muted);
            font-size: 17px;
        }
        .layout {
            display: grid;
            grid-template-columns: minmax(320px, 1.2fr) minmax(340px, 0.9fr);
            gap: 22px;
            align-items: start;
        }
        .panel {
            background: color-mix(in srgb, var(--panel) 92%, white 8%);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: 0 20px 60px rgba(36, 27, 17, 0.08);
            overflow: hidden;
        }
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            padding: 18px;
            border-bottom: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(244,239,231,0.7));
        }
        .field {
            display: grid;
            gap: 6px;
            min-width: 170px;
            flex: 1 1 180px;
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
            min-width: 180px;
            align-self: end;
            transition: transform 120ms ease, opacity 120ms ease;
        }
        button:hover { transform: translateY(-1px); }
        button:disabled { opacity: 0.65; cursor: wait; }
        .viewer {
            padding: 18px;
        }
        .viewer-meta {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
            color: var(--muted);
            font-size: 14px;
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
        .record-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 18px 18px 0;
        }
        .record-list {
            list-style: none;
            margin: 0;
            padding: 14px 12px 14px;
            max-height: 72vh;
            overflow: auto;
        }
        .record-item {
            padding: 12px 12px 14px;
            border-radius: 16px;
            border: 1px solid transparent;
            transition: background 120ms ease, border-color 120ms ease;
        }
        .record-item + .record-item { margin-top: 8px; }
        .record-item:hover {
            background: var(--accent-soft);
            border-color: rgba(176, 68, 47, 0.2);
        }
        .record-item.connection:hover {
            background: #e7efe4;
            border-color: rgba(72, 113, 64, 0.2);
        }
        .record-top {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
            margin-bottom: 4px;
        }
        .record-type {
            font-size: 16px;
            font-weight: 700;
        }
        .record-index {
            color: var(--muted);
            font-size: 13px;
        }
        .record-repr, .record-refs {
            margin: 0;
            color: var(--muted);
            font-size: 13px;
            line-height: 1.35;
            word-break: break-word;
        }
        .empty {
            padding: 18px;
            color: var(--muted);
        }
        @media (max-width: 980px) {
            .layout { grid-template-columns: 1fr; }
            .record-list { max-height: none; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <header class="header">
            <h1 class="title">Generator Inspector</h1>
            <p class="subtitle">Pick a generator, render a sample, then hover the record to highlight elements and graph connections.</p>
        </header>

        <div class="layout">
            <section class="panel">
                <div class="controls">
                    <div class="field">
                        <label for="generator">Generator</label>
                        <select id="generator"></select>
                    </div>
                    <div class="field">
                        <label for="canvasSize">Canvas Size</label>
                        <input id="canvasSize" type="number" min="128" max="1024" step="32" value="512">
                    </div>
                    <div class="field">
                        <label for="seed">Seed</label>
                        <input id="seed" type="number" min="0" placeholder="random">
                    </div>
                    <div class="field">
                        <label for="rendererType">Renderer</label>
                        <select id="rendererType">
                            <option value="static" selected>Static</option>
                            <option value="inorder">In-Order Static</option>
                            <option value="noisy">Noisy</option>
                        </select>
                    </div>
                    <div class="field noisy-only">
                        <label for="thicknessLoc">Noise Thickness</label>
                        <input id="thicknessLoc" type="number" min="1" max="8" step="0.1" value="2.0">
                    </div>
                    <div class="field noisy-only">
                        <label for="thicknessScale">Thickness Jitter</label>
                        <input id="thicknessScale" type="number" min="0" max="4" step="0.1" value="0.5">
                    </div>
                    <div class="field noisy-only">
                        <label for="blurMode">Blur</label>
                        <select id="blurMode">
                            <option value="random" selected>Random</option>
                            <option value="none">None</option>
                            <option value="3x3">3x3 Gaussian</option>
                            <option value="5x5">5x5 Gaussian</option>
                        </select>
                    </div>
                    <div class="field noisy-only">
                        <label for="arrowType">Arrow Style</label>
                        <select id="arrowType">
                            <option value="random" selected>Random</option>
                            <option value="FilledTriangleArrow">Filled Triangle</option>
                            <option value="ArrowedLine">Arrowed Line</option>
                            <option value="NonFilledTriangleArrow">Open Triangle</option>
                        </select>
                    </div>
                    <div class="field noisy-only">
                        <label for="fontMode">Font</label>
                        <select id="fontMode">
                            <option value="random" selected>Random</option>
                            <option value="fixed">Fixed Size</option>
                        </select>
                    </div>
                    <div class="field noisy-only">
                        <label for="fontSize">Fixed Font Size</label>
                        <input id="fontSize" type="number" min="8" max="32" step="1" value="14">
                    </div>
                    <div class="field noisy-only">
                        <label for="colorMin">Min Ink Value</label>
                        <input id="colorMin" type="number" min="0" max="255" step="1" value="128">
                    </div>
                    <div class="field noisy-only">
                        <label for="colorMax">Max Ink Value</label>
                        <input id="colorMax" type="number" min="0" max="255" step="1" value="255">
                    </div>
                    <button id="generate">Generate Sample</button>
                </div>
                <div class="viewer">
                    <div class="viewer-meta">
                        <span id="sampleMeta">No sample loaded.</span>
                        <span id="hoverMeta">Hover a record entry to inspect it.</span>
                    </div>
                    <div class="viewer-frame">
                        <img id="sampleImage" alt="Rendered sample">
                    </div>
                </div>
            </section>

            <aside class="panel">
                <div class="record-header">
                    <h2 style="margin:0; font-size:24px;">Record</h2>
                    <span id="recordCount" style="color:var(--muted); font-size:14px;">0 elements</span>
                </div>
                <ul id="recordList" class="record-list"></ul>
                <div id="emptyState" class="empty">Generate a sample to inspect its elements.</div>
            </aside>
        </div>

        <section class="panel" style="margin-top:22px;">
            <div class="record-header">
                <h2 style="margin:0; font-size:24px;">Generator Definition</h2>
                <span id="generatorCodeLabel" style="color:var(--muted); font-size:14px;">Selected pipeline</span>
            </div>
            <div style="padding: 14px 18px 20px;">
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;">
                    <button id="applyDefinition" type="button" style="min-width:150px;">Apply Definition</button>
                    <button id="resetDefinition" type="button" style="min-width:150px; background:white; color:var(--ink);">Reset From Selection</button>
                    <span id="definitionStatus" style="align-self:center; color:var(--muted); font-size:13px;">Editing the generator expression here overrides the dropdown template.</span>
                </div>
                <textarea id="generatorCode" spellcheck="false" style="width:100%; min-height:220px; resize:vertical; margin:0; padding:16px; border-radius:16px; border:1px solid var(--line); background:#201b17; color:#f7efe2; overflow:auto; font: 13px/1.5 Menlo, Consolas, monospace;">Generate a sample to inspect its generator composition.</textarea>
            </div>
        </section>
    </div>

    <script>
        const generatorSelect = document.getElementById('generator');
        const canvasSizeInput = document.getElementById('canvasSize');
        const seedInput = document.getElementById('seed');
        const rendererTypeInput = document.getElementById('rendererType');
        const thicknessLocInput = document.getElementById('thicknessLoc');
        const thicknessScaleInput = document.getElementById('thicknessScale');
        const blurModeInput = document.getElementById('blurMode');
        const arrowTypeInput = document.getElementById('arrowType');
        const fontModeInput = document.getElementById('fontMode');
        const fontSizeInput = document.getElementById('fontSize');
        const colorMinInput = document.getElementById('colorMin');
        const colorMaxInput = document.getElementById('colorMax');
        const noisyOnlyFields = Array.from(document.querySelectorAll('.noisy-only'));
        const generateButton = document.getElementById('generate');
        const sampleImage = document.getElementById('sampleImage');
        const sampleMeta = document.getElementById('sampleMeta');
        const hoverMeta = document.getElementById('hoverMeta');
        const recordList = document.getElementById('recordList');
        const recordCount = document.getElementById('recordCount');
        const emptyState = document.getElementById('emptyState');
        const generatorCode = document.getElementById('generatorCode');
        const generatorCodeLabel = document.getElementById('generatorCodeLabel');
        const applyDefinitionButton = document.getElementById('applyDefinition');
        const resetDefinitionButton = document.getElementById('resetDefinition');
        const definitionStatus = document.getElementById('definitionStatus');

        let generatorSpecs = {};

        let currentSample = null;

        function setImage(dataUrl) {
            sampleImage.src = dataUrl || '';
        }

        function syncRendererFields() {
            const isNoisy = rendererTypeInput.value === 'noisy';
            for (const field of noisyOnlyFields) {
                field.style.display = isNoisy ? 'grid' : 'none';
            }
        }

        function renderRecord(records) {
            recordList.innerHTML = '';
            recordCount.textContent = `${records.length} elements`;
            emptyState.style.display = records.length ? 'none' : 'block';

            for (const record of records) {
                const item = document.createElement('li');
                item.className = `record-item ${record.kind}`;
                item.innerHTML = `
                    <div class="record-top">
                        <span class="record-type">${record.title}</span>
                        <span class="record-index">#${record.index}</span>
                    </div>
                    <p class="record-repr">${record.repr}</p>
                    <p class="record-refs">${record.details}</p>
                `;

                item.addEventListener('mouseenter', () => {
                    setImage(record.highlight_image);
                    hoverMeta.textContent = record.hover_text;
                });

                item.addEventListener('mouseleave', () => {
                    if (currentSample) {
                        setImage(currentSample.base_image);
                    }
                    hoverMeta.textContent = 'Hover a record entry to inspect it.';
                });

                recordList.appendChild(item);
            }
        }

        async function loadGenerators() {
            const response = await fetch('/api/generators');
            const data = await response.json();
            generatorSpecs = data.specs;
            generatorSelect.innerHTML = '';
            for (const name of data.generators) {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                generatorSelect.appendChild(option);
            }
            syncGeneratorDefinitionFromSelection();
        }

        function syncGeneratorDefinitionFromSelection() {
            const spec = generatorSpecs[generatorSelect.value];
            if (!spec) {
                return;
            }
            generatorCode.value = spec.code;
            generatorCodeLabel.textContent = spec.name;
            definitionStatus.textContent = 'Using the selected generator template.';
        }

        async function generateSample() {
            generateButton.disabled = true;
            hoverMeta.textContent = 'Rendering sample...';
            try {
                const rawSeed = seedInput.value.trim();
                const payload = {
                    generator: generatorSelect.value,
                    generator_code: generatorCode.value,
                    canvas_size: Number(canvasSizeInput.value),
                    renderer: rendererTypeInput.value,
                    renderer_options: {
                        thickness_loc: Number(thicknessLocInput.value),
                        thickness_scale: Number(thicknessScaleInput.value),
                        blur_mode: blurModeInput.value,
                        arrow_type: arrowTypeInput.value,
                        font_mode: fontModeInput.value,
                        font_size: Number(fontSizeInput.value),
                        color_min: Number(colorMinInput.value),
                        color_max: Number(colorMaxInput.value),
                    },
                };
                if (rawSeed !== '') {
                    payload.seed = Number(rawSeed);
                }
                const response = await fetch('/api/sample', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!response.ok) {
                    const error = await response.json();
                    definitionStatus.textContent = error.error || 'Failed to render edited definition.';
                    return;
                }
                const data = await response.json();
                currentSample = data;
                setImage(data.base_image);
                sampleMeta.textContent = `${data.generator} · ${data.renderer_label} · seed ${data.seed} · ${data.canvas_size}px`;
                generatorCode.value = data.generator_code;
                generatorCodeLabel.textContent = data.generator;
                definitionStatus.textContent = 'Rendered from the current definition editor content.';
                hoverMeta.textContent = 'Hover a record entry to inspect it.';
                renderRecord(data.records);
            } finally {
                generateButton.disabled = false;
            }
        }

        generateButton.addEventListener('click', generateSample);
        applyDefinitionButton.addEventListener('click', generateSample);
        resetDefinitionButton.addEventListener('click', syncGeneratorDefinitionFromSelection);
        generatorSelect.addEventListener('change', syncGeneratorDefinitionFromSelection);
        rendererTypeInput.addEventListener('change', syncRendererFields);

        window.addEventListener('DOMContentLoaded', async () => {
            syncRendererFields();
            await loadGenerators();
            await generateSample();
        });
    </script>
</body>
</html>
"""


SampleCache = Dict[str, Dict[str, object]]
PREVIEW_MARGIN = 0.06


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    build: Callable[[], Generator]
    code: str


def with_preview_margin(generator: Generator) -> Generator:
    return MarginGenerator(generator, margin=PREVIEW_MARGIN)



def build_generator_specs() -> Dict[str, GeneratorSpec]:
    return {
        "Horizontal Line": GeneratorSpec(
            name="Horizontal Line",
            build=lambda: with_preview_margin(HorizontalLineGenerator(min_length=0.25)),
            code="MarginGenerator(\n    HorizontalLineGenerator(min_length=0.25),\n    margin=0.06,\n)",
        ),
        "Centric Circle": GeneratorSpec(
            name="Centric Circle",
            build=lambda: with_preview_margin(CentricCircleGenerator()),
            code="MarginGenerator(\n    CentricCircleGenerator(),\n    margin=0.06,\n)",
        ),
        "Rectangle": GeneratorSpec(
            name="Rectangle",
            build=lambda: with_preview_margin(RectangleGenerator(min_size=0.25, max_size=0.7)),
            code="MarginGenerator(\n    RectangleGenerator(min_size=0.25, max_size=0.7),\n    margin=0.06,\n)",
        ),
        "Polygon": GeneratorSpec(
            name="Polygon",
            build=lambda: with_preview_margin(AngularSortedPolygonGenerator(num_points=5)),
            code="MarginGenerator(\n    AngularSortedPolygonGenerator(num_points=5),\n    margin=0.06,\n)",
        ),
        "Circle Row": GeneratorSpec(
            name="Circle Row",
            build=lambda: with_preview_margin(HorizontalCircleGenerator(num_circles=3)),
            code="MarginGenerator(\n    HorizontalCircleGenerator(num_circles=3),\n    margin=0.06,\n)",
        ),
        "L Shape": GeneratorSpec(
            name="L Shape",
            build=lambda: with_preview_margin(LShapeGenerator(min_length=0.2)),
            code="MarginGenerator(\n    LShapeGenerator(min_length=0.2),\n    margin=0.06,\n)",
        ),
        "Annotated L Shape": GeneratorSpec(
            name="Annotated L Shape",
            build=lambda: with_preview_margin(
                AnnotateLineGenerator(
                    AddIdGenerator(LShapeGenerator(min_length=0.2)),
                    ratio=1.0,
                    graph_mode=True,
                )
            ),
            code="MarginGenerator(\n    AnnotateLineGenerator(\n        AddIdGenerator(\n            LShapeGenerator(min_length=0.2),\n        ),\n        ratio=1.0,\n        graph_mode=True,\n    ),\n    margin=0.06,\n)",
        ),
        "Connected L Shape": GeneratorSpec(
            name="Connected L Shape",
            build=lambda: with_preview_margin(LShapeOutlineGenerator(min_length=0.2)),
            code="MarginGenerator(\n    LShapeOutlineGenerator(min_length=0.2),\n    margin=0.06,\n)",
        ),
        "Rotated Rect Grid": GeneratorSpec(
            name="Rotated Rect Grid",
            build=lambda: with_preview_margin(GridGenerator(
                RandomRotationAugmentation(RectangleGenerator(min_size=0.2, max_size=0.35)),
                num_rows=2,
                num_columns=2,
            )),
            code="MarginGenerator(\n    GridGenerator(\n        RandomRotationAugmentation(\n            RectangleGenerator(min_size=0.2, max_size=0.35),\n        ),\n        num_rows=2,\n        num_columns=2,\n    ),\n    margin=0.06,\n)",
        ),
    }


def editable_generator_namespace() -> Dict[str, object]:
    return {
        'AddIdGenerator': AddIdGenerator,
        'AngularSortedPolygonGenerator': AngularSortedPolygonGenerator,
        'AnnotateLineGenerator': AnnotateLineGenerator,
        'CentricCircleGenerator': CentricCircleGenerator,
        'GridGenerator': GridGenerator,
        'HorizontalCircleGenerator': HorizontalCircleGenerator,
        'HorizontalLineGenerator': HorizontalLineGenerator,
        'LShapeGenerator': LShapeGenerator,
        'LShapeOutlineGenerator': LShapeOutlineGenerator,
        'MarginGenerator': MarginGenerator,
        'RandomRotationAugmentation': RandomRotationAugmentation,
        'RectangleGenerator': RectangleGenerator,
    }


def build_generator_from_code(generator_code: str) -> Generator:
    parsed = ast.parse(generator_code, mode='eval')
    compiled = compile(parsed, '<generator-definition>', 'eval')
    result = eval(compiled, {'__builtins__': {}}, editable_generator_namespace())
    if not isinstance(result, Generator):
        raise TypeError('Generator definition must evaluate to a Generator instance.')
    return result


def build_renderer(renderer_name: str, canvas_size: int, renderer_options: Dict[str, object]) -> Tuple[object, str]:
    if renderer_name == "inorder":
        return InOrderStaticRenderer(canvas_width=canvas_size, canvas_height=canvas_size), "In-Order Static"
    if renderer_name == "noisy":
        color_min = int(renderer_options.get("color_min", 128))
        color_max = int(renderer_options.get("color_max", 255))
        color_min, color_max = sorted((max(0, color_min), min(255, color_max)))
        arrow_type = str(renderer_options.get("arrow_type", "random"))
        arrow_mode = "random" if arrow_type == "random" else "fixed"
        font_mode = str(renderer_options.get("font_mode", "random"))
        return NoisyRenderer(
            canvas_width=canvas_size,
            canvas_height=canvas_size,
            thickness_loc=float(renderer_options.get("thickness_loc", 2.0)),
            thickness_scale=float(renderer_options.get("thickness_scale", 0.5)),
            min_thickness=1,
            arrow_mode=arrow_mode,
            arrow_type="FilledTriangleArrow" if arrow_type == "random" else arrow_type,
            color_loc=(color_min + color_max) / 2,
            color_scale=max((color_max - color_min) / 6, 0.1),
            color_min=color_min,
            color_max=color_max,
            blur_mode=str(renderer_options.get("blur_mode", "random")),
            font_mode=font_mode,
            font_size=int(renderer_options.get("font_size", 14)),
        ), "Noisy"
    return StaticRenderer(canvas_width=canvas_size, canvas_height=canvas_size), "Static"


def render_grayscale(actions: List[Element], canvas_size: int, seed: int, renderer_name: str = "static", renderer_options: Dict[str, object] | None = None) -> Tuple[np.ndarray, str]:
    renderer, renderer_label = build_renderer(renderer_name, canvas_size, renderer_options or {})
    return renderer.draw(actions, seed=seed).astype(np.uint8), renderer_label


def render_samples(num_samples: int, canvas_size: int, seed: int) -> List[Tuple[str, np.ndarray]]:
    random.seed(seed)
    np.random.seed(seed)
    generator_specs = build_generator_specs()
    generator_names = list(generator_specs)
    samples = []
    for sample_index in range(num_samples):
        generator_name = random.choice(generator_names)
        actions = generator_specs[generator_name].build().get_actions()
        image, _ = render_grayscale(actions, canvas_size=canvas_size, seed=seed + sample_index)
        samples.append((generator_name, image))
    return samples


def plot_samples(samples: List[Tuple[str, np.ndarray]], output_path: Path) -> None:
    columns = min(3, len(samples))
    rows = int(np.ceil(len(samples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (title, image) in zip(axes, samples):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis in axes[len(samples):]:
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def to_data_url(image: np.ndarray) -> str:
    bio = io.BytesIO()
    Image.fromarray(image).save(bio, format="PNG")
    encoded = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image.astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def build_action_index(actions: List[Element]) -> Dict[str, Element]:
    return {
        getattr(action, "id"): action
        for action in actions
        if hasattr(action, "id")
    }


def highlight_targets(action: Element, action_by_id: Dict[str, Element]) -> List[Element]:
    targets: List[Element] = []
    if not isinstance(action, ConnectTwoElementsWithId):
        targets.append(action)

    for ref in sorted(action.refs):
        referenced_action = action_by_id.get(ref)
        if referenced_action is not None:
            targets.append(referenced_action)

    seen = set()
    unique_targets = []
    for target in targets:
        identity = id(target)
        if identity in seen:
            continue
        seen.add(identity)
        unique_targets.append(target)
    return unique_targets


def overlay_actions(base_image: np.ndarray, actions: List[Element], canvas_size: int, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    mask = np.zeros_like(base_image, dtype=np.uint8)
    renderer = StaticRenderer(canvas_width=canvas_size, canvas_height=canvas_size)
    for action in actions:
        action.draw(
            mask,
            thickness=3,
            color=255,
            arrow_type="FilledTriangleArrow",
            font=renderer.font,
            font_size=getattr(renderer.font, "size", 8),
        )
    rgb = to_rgb(base_image)
    rgb[mask > 0] = np.array([240, 56, 36], dtype=np.uint8)
    return rgb


def record_kind(action: Element) -> str:
    return "connection" if isinstance(action, ConnectTwoElementsWithId) else "shape"


def record_title(index: int, action: Element) -> str:
    parts = [type(action).__name__]
    if hasattr(action, "id"):
        parts.append(f"id={getattr(action, 'id')}")
    return " · ".join(parts)


def record_details(action: Element) -> str:
    refs = sorted(action.refs)
    if isinstance(action, ConnectTwoElementsWithId):
        return f"Connects ids {', '.join(refs)}"
    if isinstance(action, AnnotationTextRefId):
        return f"Annotation linked to id {', '.join(refs)}"
    if isinstance(action, PartLineWithId):
        return "Visible element with stable training id"
    if refs:
        return f"References ids {', '.join(refs)}"
    return "Visible rendered element"


def hover_text(action: Element) -> str:
    if isinstance(action, ConnectTwoElementsWithId):
        return "Highlighting both connected elements."
    if isinstance(action, AnnotationTextRefId):
        return "Highlighting the annotation and its referenced element."
    return "Highlighting the hovered record element."


def sample_payload(generator_name: str, canvas_size: int, seed: int, renderer_name: str, renderer_options: Dict[str, object]) -> Dict[str, object]:
    generator_spec = build_generator_specs()[generator_name]
    generator = generator_spec.build()
    random.seed(seed)
    np.random.seed(seed)
    actions = generator.get_actions()
    base_image, renderer_label = render_grayscale(actions, canvas_size=canvas_size, seed=seed, renderer_name=renderer_name, renderer_options=renderer_options)
    action_by_id = build_action_index(actions)

    records = []
    for index, action in enumerate(actions):
        highlight_image = overlay_actions(
            base_image,
            actions=highlight_targets(action, action_by_id),
            canvas_size=canvas_size,
            seed=seed,
        )
        records.append({
            "index": index,
            "kind": record_kind(action),
            "title": record_title(index, action),
            "repr": repr(action),
            "details": record_details(action),
            "hover_text": hover_text(action),
            "highlight_image": to_data_url(highlight_image),
        })

    return {
        "sample_id": uuid.uuid4().hex,
        "generator": generator_name,
        "generator_code": generator_spec.code,
        "renderer": renderer_name,
        "renderer_label": renderer_label,
        "seed": seed,
        "canvas_size": canvas_size,
        "base_image": to_data_url(to_rgb(base_image)),
        "records": records,
    }


def sample_payload_from_code(generator_name: str, generator_code: str, canvas_size: int, seed: int, renderer_name: str, renderer_options: Dict[str, object]) -> Dict[str, object]:
    generator = build_generator_from_code(generator_code)
    random.seed(seed)
    np.random.seed(seed)
    actions = generator.get_actions()
    base_image, renderer_label = render_grayscale(actions, canvas_size=canvas_size, seed=seed, renderer_name=renderer_name, renderer_options=renderer_options)
    action_by_id = build_action_index(actions)

    records = []
    for index, action in enumerate(actions):
        highlight_image = overlay_actions(
            base_image,
            actions=highlight_targets(action, action_by_id),
            canvas_size=canvas_size,
            seed=seed,
        )
        records.append({
            'index': index,
            'kind': record_kind(action),
            'title': record_title(index, action),
            'repr': repr(action),
            'details': record_details(action),
            'hover_text': hover_text(action),
            'highlight_image': to_data_url(highlight_image),
        })

    return {
        'sample_id': uuid.uuid4().hex,
        'generator': generator_name,
        'generator_code': generator_code,
        'renderer': renderer_name,
        'renderer_label': renderer_label,
        'seed': seed,
        'canvas_size': canvas_size,
        'base_image': to_data_url(to_rgb(base_image)),
        'records': records,
    }


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.get("/api/generators")
    def generators():
        specs = build_generator_specs()
        return jsonify({
            'generators': list(specs.keys()),
            'specs': {name: {'name': spec.name, 'code': spec.code} for name, spec in specs.items()},
        })

    @app.post("/api/sample")
    def sample():
        payload = request.get_json(force=True, silent=True) or {}
        generator_name = payload.get("generator") or next(iter(build_generator_specs().keys()))
        generator_code = str(payload.get('generator_code') or build_generator_specs()[generator_name].code)
        canvas_size = int(payload.get("canvas_size", 512))
        seed_value = payload.get("seed")
        seed = random.randint(0, 10_000) if seed_value in (None, "") else int(seed_value)
        renderer_name = str(payload.get("renderer", "static"))
        renderer_options = payload.get("renderer_options") or {}
        try:
            return jsonify(sample_payload_from_code(generator_name, generator_code, canvas_size, seed, renderer_name, renderer_options))
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and render synthetic shape generators.")
    parser.add_argument("--plot", action="store_true", help="Render a static grid of samples instead of launching the web UI.")
    parser.add_argument("--samples", type=int, default=6, help="Number of random samples to render in plot mode.")
    parser.add_argument("--canvas-size", type=int, default=512, help="Canvas size in pixels.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible output.")
    parser.add_argument("--output", type=Path, default=Path("debug_shapes.png"), help="Output path for plot mode.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the web UI server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the web UI server.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot:
        samples = render_samples(num_samples=args.samples, canvas_size=args.canvas_size, seed=args.seed)
        plot_samples(samples, args.output)
        print(f"Saved debug plot to {args.output.resolve()}")
        return

    app = create_app()
    print(f"Generator inspector available at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
