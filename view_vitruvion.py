"""Browse the rebuilt Vitruvion dataset: its renders, primitives and constraints, in a local web UI.

Reads a folder as `vitruvion_reference/run_vitruvion.sh` leaves it -- `sg_filtered_unique.npy`,
`splits.json`, and whichever `renders/render_p128_XX_of_16.npy` chunks are there so far -- and shows
each sketch as Vitruvion's models see it: the primitives in their minimal parameters, the categorical
constraints, and the clean and hand-drawn renders. Hovering a primitive or a constraint lights it up
on the drawing and in the lists, since a constraint is the one part of a label the picture cannot
show.

    uv run view_vitruvion.py --dataset-dir ./dataset_sketches_vitruvion
"""

import argparse
import base64
import io
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from create_vitruvion_dataset import CONSTRAINTS, constraints_of, normalize_sketch, parameters, quantize


#: Where sketch coordinates land in a 128 px render, measured on Vitruvion's renderer
#: (`prerender_images.render_sketch`: limits of +-0.6 on a one-inch figure, then `tight_layout`):
#: x from -0.6 to 0.6 spans these pixels from the left, y these pixels from the bottom.
RENDER_SIZE = 128
RENDER_X = (25.2622, 108.64)
RENDER_Y = (25.4222, 108.80)
RENDER_LIMIT = 0.6


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Vitruvion Sketch Viewer</title>
    <style>
        :root {
            --bg: #f5f1e8; --panel: #fffdf9; --ink: #161514; --muted: #6d665c; --line: #ddd3c5;
            --accent: #186a73; --accent-soft: #d8eff1;
            --stroke: #2c6ea8; --construction: #8a7f6e; --lit: #c0392b; --partner: #e08e0b;
        }
        * { box-sizing: border-box; }
        [hidden] { display: none !important; }
        body {
            margin: 0;
            font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
            background:
                radial-gradient(circle at top right, #fff4d7 0, transparent 26%),
                radial-gradient(circle at bottom left, #dcebed 0, transparent 28%),
                var(--bg);
            color: var(--ink);
        }
        .shell { max-width: 1400px; margin: 0 auto; padding: 28px; }
        .header { margin-bottom: 22px; }
        .title { margin: 0; font-size: 40px; line-height: 1; }
        .subtitle { margin: 10px 0 0; font-size: 16px; color: var(--muted); }
        .layout {
            display: grid; grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
            gap: 22px; align-items: start;
        }
        .panel {
            background: color-mix(in srgb, var(--panel) 92%, white 8%);
            border: 1px solid var(--line); border-radius: 22px; overflow: hidden;
            box-shadow: 0 20px 60px rgba(30, 26, 20, 0.08);
        }
        .controls {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;
            padding: 18px; border-bottom: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(245,241,232,0.72));
        }
        .field { display: grid; gap: 6px; }
        label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
        select, input, button {
            border-radius: 12px; border: 1px solid var(--line); padding: 10px 12px;
            font: inherit; background: white; color: var(--ink);
        }
        button { cursor: pointer; background: var(--accent); color: white; border-color: var(--accent); }
        button.secondary { background: white; color: var(--ink); }
        .nav-row { display: flex; gap: 10px; flex-wrap: wrap; padding: 14px 18px 0; }
        .viewer { padding: 18px; }
        .viewer-meta {
            display: flex; justify-content: space-between; gap: 12px; margin-bottom: 12px;
            font-size: 14px; color: var(--muted);
        }
        .viewer-meta a { color: var(--accent); }
        .frame {
            position: relative; border-radius: 18px; overflow: hidden; border: 1px solid var(--line);
            background: white; aspect-ratio: 1 / 1;
        }
        .frame img { width: 100%; height: 100%; display: block; image-rendering: pixelated; }
        .frame svg { position: absolute; inset: 0; width: 100%; height: 100%; }
        .frame .empty {
            position: absolute; inset: 0; display: grid; place-items: center; color: var(--muted);
            font-size: 14px; padding: 24px; text-align: center;
        }
        .variants { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }
        .variants img {
            width: 64px; height: 64px; border: 2px solid var(--line); border-radius: 10px;
            cursor: pointer; background: white; image-rendering: pixelated;
        }
        .variants img.active { border-color: var(--accent); }
        .toggles { display: flex; gap: 16px; margin-top: 12px; font-size: 14px; color: var(--muted); }
        .toggles label { text-transform: none; letter-spacing: 0; font-size: 14px; }
        .status { padding: 0 18px 18px; font-size: 13px; color: var(--muted); }
        .sidebar { padding: 18px; }
        .info-title { margin: 0 0 8px; font-size: 18px; }
        .info-block + .info-block { margin-top: 18px; }
        .info-copy { margin: 0 0 6px; line-height: 1.45; color: var(--muted); font-size: 14px; }
        .rows {
            border: 1px solid var(--line); border-radius: 14px; background: #201b17; color: #f7efe2;
            font: 12.5px/1.55 Menlo, Consolas, monospace; max-height: 360px; overflow: auto; padding: 6px;
        }
        .row { display: block; padding: 1px 6px; border-radius: 4px; cursor: pointer; white-space: nowrap; }
        .row .kind { color: #9fd3d9; }
        .row .construction { color: #b7aa95; font-style: italic; }
        .row .muted { color: #8f8577; }
        .row.lit { background: var(--lit); color: white; }
        .row.lit .kind, .row.lit .muted, .row.lit .construction { color: white; }
        .row.partner { background: color-mix(in srgb, var(--partner) 55%, #201b17); }
        /* the overlay: what the label says, drawn over the render in the render's own pixels */
        .prim { fill: none; stroke: var(--stroke); stroke-width: 1.1; opacity: 0.55; cursor: pointer; }
        .prim.construction { stroke: var(--construction); stroke-dasharray: 2 1.5; }
        .prim.point { fill: var(--stroke); stroke: none; }
        .hit { fill: none; stroke: transparent; stroke-width: 5; cursor: pointer; }
        .prim.lit { stroke: var(--lit); stroke-width: 2; opacity: 1; }
        .prim.point.lit { fill: var(--lit); }
        .prim.partner { stroke: var(--partner); stroke-width: 1.6; opacity: 1; }
        .prim.point.partner { fill: var(--partner); }
        .refdot { fill: var(--lit); stroke: white; stroke-width: 0.6; }
        .hide-overlay .prim, .hide-overlay .hit { display: none; }
        @media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="shell">
    <header class="header">
        <h1 class="title">Vitruvion Sketch Viewer</h1>
        <p class="subtitle">The rebuilt Vitruvion dataset: renders, primitives and constraints. Hover a
            primitive or a constraint to see it on the drawing. Arrow keys move through the split.</p>
    </header>
    <div class="layout">
        <section class="panel">
            <div class="controls">
                <div class="field">
                    <label for="split">Split</label>
                    <select id="split"></select>
                </div>
                <div class="field">
                    <label for="indexInput">Index in split</label>
                    <input id="indexInput" type="number" min="0" step="1" value="0">
                </div>
                <div class="field">
                    <label for="sketchInput">Sketch number</label>
                    <input id="sketchInput" type="number" min="0" step="1" placeholder="any">
                </div>
            </div>
            <div class="nav-row">
                <button id="prevButton" class="secondary">Previous</button>
                <button id="nextButton">Next</button>
                <button id="randomButton" class="secondary">Random</button>
            </div>
            <div class="viewer">
                <div class="viewer-meta">
                    <span id="sampleMeta">Loading…</span>
                    <span id="sourceMeta"></span>
                </div>
                <div class="frame" id="frame">
                    <img id="render" alt="">
                    <div class="empty" id="noRender" hidden>Not rendered yet: the chunk holding this
                        sketch is not in renders/ yet. The overlay shows the label alone.</div>
                    <svg id="overlay" viewBox="0 0 128 128"></svg>
                </div>
                <div class="variants" id="variants"></div>
                <div class="toggles">
                    <label><input type="checkbox" id="showOverlay" checked> overlay the label</label>
                </div>
            </div>
            <div class="status" id="status"></div>
        </section>
        <aside class="panel sidebar">
            <div class="info-block">
                <h2 class="info-title">Primitives</h2>
                <p class="info-copy">In the designer's order, as Vitruvion parameterises them; coordinates
                    are centred, longer side 1, y up. Brackets hold the 64-level bins.</p>
                <div class="rows" id="primitives"></div>
            </div>
            <div class="info-block">
                <h2 class="info-title">Constraints</h2>
                <p class="info-copy" id="constraintNote"></p>
                <div class="rows" id="constraints"></div>
            </div>
            <div class="info-block">
                <h2 class="info-title">Dataset</h2>
                <p class="info-copy" id="datasetInfo"></p>
            </div>
        </aside>
    </div>
</div>
<script>
    const $ = id => document.getElementById(id);
    let info = null, sample = null, variant = 0;

    /* sketch coordinates to render pixels, top-left origin, as Vitruvion's renderer places them */
    let T = null;
    const px = ([x, y]) => [
        T.x0 + (x + T.limit) / (2 * T.limit) * (T.x1 - T.x0),
        T.size - (T.y0 + (y + T.limit) / (2 * T.limit) * (T.y1 - T.y0)),
    ];
    const fmt = v => (v >= 0 ? ' ' : '') + v.toFixed(3);
    const pt = (x, y) => `(${fmt(x)},${fmt(y)})`;

    function circumcentre(a, b, c) {
        const d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
        if (Math.abs(d) < 1e-12) return null;
        const a2 = a[0] ** 2 + a[1] ** 2, b2 = b[0] ** 2 + b[1] ** 2, c2 = c[0] ** 2 + c[1] ** 2;
        return [(a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
                (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d];
    }

    /* the point of a primitive a constraint names, in sketch coordinates */
    function pointOf(p, part) {
        const q = p.params;
        if (p.type === 'Point') return [q[0], q[1]];
        if (p.type === 'Line') return part === 'end' ? [q[2], q[3]] : [q[0], q[1]];
        if (p.type === 'Circle') return [q[0], q[1]];
        if (part === 'start') return [q[0], q[1]];
        if (part === 'end') return [q[4], q[5]];
        return circumcentre([q[0], q[1]], [q[2], q[3]], [q[4], q[5]]);
    }

    function shape(p) {
        const q = p.params;
        if (p.type === 'Line') {
            const [x1, y1] = px([q[0], q[1]]), [x2, y2] = px([q[2], q[3]]);
            return `M ${x1} ${y1} L ${x2} ${y2}`;
        }
        if (p.type === 'Circle') {
            const [cx, cy] = px([q[0], q[1]]), r = q[2] / (2 * T.limit) * (T.x1 - T.x0);
            return `M ${cx - r} ${cy} a ${r} ${r} 0 1 0 ${2 * r} 0 a ${r} ${r} 0 1 0 ${-2 * r} 0`;
        }
        if (p.type === 'Arc') {
            /* counter-clockwise in the sketch is clockwise on the y-down screen: sweep-flag 1 */
            const s = [q[0], q[1]], m = [q[2], q[3]], e = [q[4], q[5]];
            const c = circumcentre(s, m, e);
            const [sx, sy] = px(s), [ex, ey] = px(e);
            if (!c) return `M ${sx} ${sy} L ${ex} ${ey}`;
            const r = Math.hypot(s[0] - c[0], s[1] - c[1]) / (2 * T.limit) * (T.x1 - T.x0);
            const turn = a => Math.atan2(a[1] - c[1], a[0] - c[0]);
            const sweep = ((turn(e) - turn(s)) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
            return `M ${sx} ${sy} A ${r} ${r} 0 ${sweep > Math.PI ? 1 : 0} 0 ${ex} ${ey}`;
        }
        return null;
    }

    function drawOverlay() {
        const svg = $('overlay');
        svg.innerHTML = sample.primitives.map((p, i) => {
            const cls = `prim ${p.construction ? 'construction' : ''}`;
            if (p.type === 'Point') {
                const [x, y] = px(p.params);
                return `<g data-prim="${i}"><circle class="${cls} point" cx="${x}" cy="${y}" r="1.4"/>`
                     + `<circle class="hit" cx="${x}" cy="${y}" r="2.5"/></g>`;
            }
            const d = shape(p);
            return `<g data-prim="${i}"><path class="${cls}" d="${d}"/><path class="hit" d="${d}"/></g>`;
        }).join('') + '<g id="refdots"></g>';
        svg.querySelectorAll('[data-prim]').forEach(g => {
            g.addEventListener('mouseenter', () => litPrimitive(Number(g.dataset.prim)));
            g.addEventListener('mouseleave', clear);
        });
    }

    function clear() {
        document.querySelectorAll('.lit, .partner').forEach(n => n.classList.remove('lit', 'partner'));
        $('refdots').innerHTML = '';
    }

    function mark(selector, cls) { document.querySelectorAll(selector).forEach(n => n.classList.add(cls)); }

    /* a primitive, and the constraints that act on it */
    function litPrimitive(i) {
        clear();
        mark(`[data-prim="${i}"] .prim, .row[data-prim="${i}"]`, 'lit');
        sample.constraints.forEach((c, k) => {
            if (c.refs.some(([p]) => p === i)) mark(`.row[data-con="${k}"]`, 'partner');
        });
    }

    /* a constraint, the primitives it acts on, and the points of them it names */
    function litConstraint(k) {
        clear();
        mark(`.row[data-con="${k}"]`, 'lit');
        $('refdots').innerHTML = sample.constraints[k].refs.map(([i, part]) => {
            mark(`[data-prim="${i}"] .prim, .row[data-prim="${i}"]`, 'partner');
            if (part === null) return '';
            const at = pointOf(sample.primitives[i], part);
            if (!at) return '';
            const [x, y] = px(at);
            return `<circle class="refdot" cx="${x}" cy="${y}" r="2"/>`;
        }).join('');
    }

    function describe(p) {
        const q = p.params;
        const where = {
            Line: () => `${pt(q[0], q[1])} → ${pt(q[2], q[3])}`,
            Circle: () => `centre ${pt(q[0], q[1])} r ${q[2].toFixed(3)}`,
            Arc: () => `${pt(q[0], q[1])} via ${pt(q[2], q[3])} → ${pt(q[4], q[5])}`,
            Point: () => pt(q[0], q[1]),
        }[p.type]();
        return where;
    }

    function renderLists() {
        $('primitives').innerHTML = sample.primitives.map((p, i) =>
            `<span class="row" data-prim="${i}">#${String(i).padStart(2)} <span class="kind">${p.type.padEnd(6)}</span> `
            + `${describe(p)} <span class="muted">[${p.bins.join(' ')}]</span>`
            + (p.construction ? ' <span class="construction">construction</span>' : '') + '</span>'
        ).join('');
        $('constraints').innerHTML = sample.constraints.map((c, k) =>
            `<span class="row" data-con="${k}"><span class="kind">${c.type.padEnd(13)}</span> `
            + c.refs.map(([i, part]) => `#${i} ${sample.primitives[i].type}${part ? '.' + part : ''}`).join('  ·  ')
            + '</span>'
        ).join('') || '<span class="row muted">none</span>';
        const other = Object.entries(sample.other_constraints).map(([t, n]) => `${t} ×${n}`).join(', ');
        $('constraintNote').textContent =
            `${sample.constraints.length} categorical constraints, as Vitruvion's constraint model reads them: `
            + `references sorted by sequence position, so their order carries no meaning.`
            + (other ? ` Also in the file but not used there: ${other}.` : '');
        document.querySelectorAll('.row[data-prim]').forEach(r => {
            r.addEventListener('mouseenter', () => litPrimitive(Number(r.dataset.prim)));
            r.addEventListener('mouseleave', clear);
        });
        document.querySelectorAll('.row[data-con]').forEach(r => {
            r.addEventListener('mouseenter', () => litConstraint(Number(r.dataset.con)));
            r.addEventListener('mouseleave', clear);
        });
    }

    function showVariant(v) {
        variant = v;
        const renders = sample.renders;
        $('render').hidden = renders.length === 0;
        $('noRender').hidden = renders.length > 0;
        if (renders.length) $('render').src = renders[Math.min(v, renders.length - 1)];
        $('variants').innerHTML = renders.map((src, k) =>
            `<img src="${src}" data-v="${k}" class="${k === v ? 'active' : ''}" title="${k === 0 ? 'clean' : 'hand-drawn ' + k}">`
        ).join('');
        document.querySelectorAll('#variants img').forEach(img =>
            img.addEventListener('click', () => showVariant(Number(img.dataset.v))));
    }

    async function load(params) {
        $('status').textContent = 'Loading…';
        const response = await fetch('/api/sample?' + new URLSearchParams(params));
        const data = await response.json();
        if (!response.ok) { $('status').textContent = data.error; return; }
        sample = data;
        T = data.render_frame;
        history.replaceState(null, '', `?sketch=${data.sketch}`);
        $('split').value = data.split;
        $('indexInput').value = data.position;
        $('sketchInput').value = data.sketch;
        $('sampleMeta').textContent =
            `${data.split} ${data.position + 1} / ${data.split_size} · sketch ${data.sketch}`;
        const s = data.source;
        $('sourceMeta').innerHTML = `<a href="https://cad.onshape.com/documents/${s.document_id}" target="_blank"`
            + ` rel="noopener">${s.document_id}</a> · part ${s.part_idx} · sketch ${s.sketch_idx}`;
        drawOverlay();
        renderLists();
        showVariant(Math.min(variant, Math.max(0, data.renders.length - 1)));
        $('status').textContent = data.renders.length
            ? `${data.renders.length} renders: clean, then hand-drawn. Renders cover sketches `
              + `${data.rendered_ranges}.`
            : `Rendered so far: ${data.rendered_ranges || 'none'}.`;
    }

    const step = d => load({ split: $('split').value, position: Number($('indexInput').value) + d });
    $('prevButton').onclick = () => step(-1);
    $('nextButton').onclick = () => step(1);
    $('randomButton').onclick = () => load({
        split: $('split').value, position: Math.floor(Math.random() * info.splits[$('split').value]) });
    $('split').onchange = () => load({ split: $('split').value, position: 0 });
    $('indexInput').onkeydown = e => { if (e.key === 'Enter') step(0); };
    $('sketchInput').onkeydown = e => { if (e.key === 'Enter') load({ sketch: $('sketchInput').value }); };
    $('showOverlay').onchange = e => $('frame').classList.toggle('hide-overlay', !e.target.checked);
    window.addEventListener('keydown', e => {
        if (['INPUT', 'SELECT'].includes(e.target.tagName)) return;
        if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    });

    window.addEventListener('DOMContentLoaded', async () => {
        info = await (await fetch('/api/dataset')).json();
        $('split').innerHTML = Object.entries(info.splits)
            .map(([name, n]) => `<option value="${name}">${name} (${n})</option>`).join('');
        $('datasetInfo').textContent = `${info.dataset_dir} · ${info.count} sketches · `
            + `${info.render_chunks} render chunks present`;
        const asked = new URLSearchParams(location.search);
        await load(asked.has('sketch') ? { sketch: asked.get('sketch') }
                                       : { split: asked.get('split') || 'train', position: asked.get('position') || 0 });
    });
</script>
</body>
</html>
"""


class Renders:
    """The render chunks present in a folder, opened as they appear.

    A chunk holds whole sketches, each as consecutive images -- the clean render first, then the
    hand-drawn ones -- with `indexes` naming the sketch of each image; chunks are memory-mapped, so
    only the images asked for are read.
    """

    def __init__(self, folder: Path):
        self.folder = folder
        self.chunks: Dict[Path, dict] = {}

    def refresh(self) -> None:
        from sketchgraphs.data import flat_array

        for path in sorted(self.folder.glob("render_p128_*_of_*.npy")):
            if ".part" in path.name or path in self.chunks:
                continue
            held = flat_array.load_dictionary_flat(str(path))
            indexes = np.asarray(held["indexes"])
            self.chunks[path] = {"indexes": indexes, "imgs": held["imgs"],
                                 "first": int(indexes[0]), "last": int(indexes[-1])}

    def of(self, sketch: int) -> List[bytes]:
        for chunk in self.chunks.values():
            if chunk["first"] <= sketch <= chunk["last"]:
                lo, hi = np.searchsorted(chunk["indexes"], [sketch, sketch + 1])
                return [bytes(chunk["imgs"][i]) for i in range(lo, hi)]
        return []

    def ranges(self) -> str:
        spans = sorted((c["first"], c["last"]) for c in self.chunks.values())
        merged: List[List[int]] = []
        for first, last in spans:
            if merged and first <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        return ", ".join(f"{a}–{b}" for a, b in merged)


class Dataset:
    def __init__(self, folder: Path):
        from sketchgraphs.data import flat_array

        self.folder = folder.resolve()
        held = flat_array.load_dictionary_flat(str(self.folder / "sg_filtered_unique.npy"))
        self.sequences, self.sketch_ids = held["sequences"], held["sketch_ids"]
        self.count = len(self.sequences)
        splits_path = self.folder / "splits.json"
        if splits_path.exists():
            held_splits = json.loads(splits_path.read_text(encoding="utf-8"))
            self.splits = {name: np.asarray(held_splits[name]) for name in ("train", "val", "test")}
        else:
            self.splits = {}
        self.splits["all"] = np.arange(self.count)
        self.renders = Renders(self.folder / "renders")
        self.lock = threading.Lock()

    def summary(self) -> dict:
        with self.lock:
            self.renders.refresh()
            return {
                "dataset_dir": str(self.folder),
                "count": self.count,
                "splits": {name: int(len(indices)) for name, indices in self.splits.items()},
                "render_chunks": len(self.renders.chunks),
            }

    def sample(self, split: str, position: Optional[int], sketch: Optional[int]) -> dict:
        from sketchgraphs.data.sequence import sketch_from_sequence

        if sketch is not None:
            if not 0 <= sketch < self.count:
                raise ValueError(f"sketch numbers run from 0 to {self.count - 1}")
            # Shown within the split that holds it, so that Previous and Next stay in that split.
            split = next((name for name, held in self.splits.items()
                          if name != "all" and len(held) and held[min(np.searchsorted(held, sketch), len(held) - 1)] == sketch),
                         "all")
            position = int(np.searchsorted(self.splits[split], sketch))
        indices = self.splits[split]
        position = max(0, min(len(indices) - 1, int(position or 0)))
        sketch = int(indices[position])

        seq = self.sequences[sketch]
        held = sketch_from_sequence(seq)
        normalize_sketch(held)   # as Vitruvion does before rendering and tokenising
        primitives = []
        for entity in held.entities.values():
            values = parameters(entity)
            primitives.append({
                "type": entity.type.name,
                "params": values.tolist(),
                "bins": quantize(values).tolist(),
                "construction": bool(entity.isConstruction),
            })
        other: Dict[str, int] = {}
        from sketchgraphs.data import EdgeOp
        for op in seq:
            if isinstance(op, EdgeOp) and op.label.name != "Subnode":
                name = op.label.name if op.label.name not in CONSTRAINTS else f"{op.label.name} (external)"
                if op.label.name not in CONSTRAINTS or 0 in op.references:
                    other[name] = other.get(name, 0) + 1

        with self.lock:
            self.renders.refresh()
            images = self.renders.of(sketch)
            ranges = self.renders.ranges()
        source = self.sketch_ids[sketch]
        return {
            "split": split,
            "position": position,
            "split_size": int(len(indices)),
            "sketch": sketch,
            "source": {"document_id": source["document_id"].decode(), "part_idx": int(source["part_idx"]),
                       "sketch_idx": int(source["sketch_idx"])},
            "primitives": primitives,
            "constraints": constraints_of(seq),
            "other_constraints": dict(sorted(other.items(), key=lambda item: -item[1])),
            "renders": ["data:image/png;base64," + base64.b64encode(img).decode("ascii") for img in images],
            "rendered_ranges": ranges,
            "render_frame": {"size": RENDER_SIZE, "limit": RENDER_LIMIT,
                             "x0": RENDER_X[0], "x1": RENDER_X[1], "y0": RENDER_Y[0], "y1": RENDER_Y[1]},
        }


def create_app(dataset_dir: Path) -> Flask:
    dataset = Dataset(dataset_dir)
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_TEMPLATE)

    @app.get("/api/dataset")
    def dataset_info():
        return jsonify(dataset.summary())

    @app.get("/api/sample")
    def sample():
        def number(name):
            raw = request.args.get(name)
            return int(raw) if raw not in (None, "") else None
        try:
            return jsonify(dataset.sample(request.args.get("split", "train"), number("position"), number("sketch")))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse the rebuilt Vitruvion dataset in a local web UI.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_sketches_vitruvion"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.dataset_dir)
    print(f"Serving {args.dataset_dir} at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
