"""Create an evenly sampled contact sheet without loading the whole video into memory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=480)
    parser.add_argument("--cell-height", type=int, default=270)
    args = parser.parse_args()
    if min(args.samples, args.columns, args.cell_width, args.cell_height) <= 0:
        parser.error("samples, columns and cell dimensions must be positive")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {args.video}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames <= 0 or fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid video metadata: {args.video}")

    count = min(args.samples, frames)
    indices = [0] if count == 1 else sorted({round(i * (frames - 1) / (count - 1)) for i in range(count)})
    rows = math.ceil(len(indices) / args.columns)
    sheet = Image.new("RGB", (args.cell_width * args.columns, args.cell_height * rows), (0, 0, 0))
    for slot, frame_number in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"failed to read frame {frame_number} from {args.video}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).resize((args.cell_width, args.cell_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 220, 28), fill=(0, 0, 0))
        draw.text((6, 6), f"frame {frame_number}  t={frame_number / fps:.2f}s", fill=(255, 80, 80))
        sheet.paste(image, ((slot % args.columns) * args.cell_width, (slot // args.columns) * args.cell_height))
    capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(json.dumps({
        "frames": frames, "fps": fps, "duration_s": frames / fps,
        "size": [width, height], "samples": indices,
        "contact_sheet": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
