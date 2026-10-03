"""The one fix Vitruvion's renderer needs to run: two calls passing arguments by position.

`img2cad.pipeline.prerender_images.render_sketch` takes `(sketch, ax=None, pad=0.1,
size_pixels=128, ...)`, but `process_sketch_sequence` calls it as
`render_sketch(sketch, pad, size_pixels, ...)`, from before `ax` was added: the padding lands in
`ax`, and rendering fails on it. This passes the two by name, as intended, and changes nothing else.
It edits the container's copy of the code, run before the renderer:

    python /tools/patch_prerender.py && python -m img2cad.pipeline.prerender_images ...
"""
from pathlib import Path

path = Path("/opt/vitruvion/img2cad/pipeline/prerender_images.py")
text = path.read_text()
fixes = {
    "render_sketch(sketch, pad, size_pixels, sketch_extent=1)":
        "render_sketch(sketch, pad=pad, size_pixels=size_pixels, sketch_extent=1)",
    "render_sketch(sketch_noisy, pad, size_pixels, sketch_extent=1, hand_drawn=True)":
        "render_sketch(sketch_noisy, pad=pad, size_pixels=size_pixels, sketch_extent=1, hand_drawn=True)",
}
for old, new in fixes.items():
    if new in text:
        continue
    assert text.count(old) == 1, f"expected exactly one call {old!r}"
    text = text.replace(old, new)
path.write_text(text)
print("patched prerender_images.py: pad and size_pixels passed by name")
