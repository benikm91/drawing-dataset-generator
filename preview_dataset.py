"""Writes a sheet of drawings from a dataset, to look at without starting the web viewer.

`view_dataset.py` is the way to inspect a single drawing against its record. This is the other
question — whether a whole rung reads clearly — and it wants many drawings at once rather than one
in depth.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a contact sheet of dataset drawings.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=None, help="defaults to <dataset-dir>/<split>_preview.png")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--label", action="store_true", help="write the edge count on each drawing")
    parser.add_argument("--scale", type=float, default=1.0, help="shrink each drawing, to fit more on a sheet")
    return parser.parse_args()


def edge_count(labels_path: Path, index: int) -> int:
    """How many part edges the record at `index` holds."""
    with labels_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["index"] == index:
                actions = json.loads(row["actions"])
                return sum(1 for a in actions if a.get("type") == "PartLineWithId")
    return 0


def main() -> None:
    args = parse_args()
    images = np.load(args.dataset_dir / f"{args.split}_images.npy", mmap_mode="r")
    labels_path = args.dataset_dir / f"{args.split}_labels.jsonl"
    out = args.out or args.dataset_dir / f"{args.split}_preview.png"

    count = min(args.rows * args.cols, len(images) - args.start)
    if count <= 0:
        raise SystemExit(f"{args.dataset_dir} holds {len(images)} {args.split} drawings, none from {args.start}")
    height, width = images.shape[1:3]
    if args.scale != 1.0:
        height, width = max(1, int(height * args.scale)), max(1, int(width * args.scale))
    pad = 4
    sheet = Image.new(
        "L",
        (args.cols * (width + pad) + pad, args.rows * (height + pad) + pad),
        color=200,
    )
    draw = ImageDraw.Draw(sheet)

    for slot in range(count):
        index = args.start + slot
        row, col = divmod(slot, args.cols)
        x, y = pad + col * (width + pad), pad + row * (height + pad)
        drawing = Image.fromarray(np.asarray(images[index], dtype=np.uint8))
        if args.scale != 1.0:
            drawing = drawing.resize((width, height), Image.LANCZOS)
        sheet.paste(drawing, (x, y))
        if args.label:
            draw.text((x + 3, y + 3), f"#{index} {edge_count(labels_path, index)}e", fill=0)

    sheet.save(out)
    print(f"wrote {count} drawings to {out}")


if __name__ == "__main__":
    main()
