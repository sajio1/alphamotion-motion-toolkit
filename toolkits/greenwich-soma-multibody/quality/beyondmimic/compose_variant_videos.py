"""Pair two named video variants and render synchronized side-by-side comparisons."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def metadata(path: Path) -> tuple[float, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    return frames / fps, fps, frames


def ffmpeg_escape(value: str) -> str:
    return value.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def clean_output_name(index: int, key: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_.-")[:100] or "clip"
    return f"{index:02d}_{name}__comparison.mp4"


def collect(source: Path, prefix: str, suffix: str) -> dict[str, Path]:
    matches: dict[str, Path] = {}
    for path in sorted(source.glob("*.mp4")):
        stem = path.stem
        if stem.startswith(prefix) and stem.endswith(suffix):
            key = stem[len(prefix) : len(stem) - len(suffix)] if suffix else stem[len(prefix) :]
            if not key:
                raise RuntimeError(f"empty pairing key for {path.name}")
            if key in matches:
                raise RuntimeError(f"duplicate pairing key {key!r}: {path.name}")
            matches[key] = path
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-suffix", required=True, help="Stem suffix identifying the left variant")
    parser.add_argument("--right-suffix", required=True, help="Stem suffix identifying the right variant")
    parser.add_argument("--prefix", default="", help="Optional common stem prefix excluded from pairing keys")
    parser.add_argument("--left-label", default="Left")
    parser.add_argument("--right-label", default="Right")
    parser.add_argument("--width", type=int, default=640, help="Width of each panel")
    parser.add_argument("--height", type=int, default=480, help="Height of each panel")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--font", type=Path, default=Path(r"C:\Windows\Fonts\arialbd.ttf"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        parser.error(f"source does not exist: {source}")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or not 0 <= args.crf <= 51:
        parser.error("width, height and fps must be positive; crf must be 0..51")
    if not args.font.is_file():
        parser.error(f"font does not exist: {args.font}")

    left = collect(source, args.prefix, args.left_suffix)
    right = collect(source, args.prefix, args.right_suffix)
    if not left and not right:
        raise RuntimeError("no matching videos found")
    if left.keys() != right.keys():
        raise RuntimeError(
            f"pair mismatch: missing right={sorted(left.keys()-right.keys())}, "
            f"missing left={sorted(right.keys()-left.keys())}"
        )
    output.mkdir(parents=True, exist_ok=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    font = ffmpeg_escape(str(args.font.resolve()))
    rows: list[dict] = []
    total = len(left)
    for index, key in enumerate(sorted(left), start=1):
        left_duration, left_fps, left_frames = metadata(left[key])
        right_duration, right_fps, right_frames = metadata(right[key])
        duration = max(left_duration, right_duration)
        destination = output / clean_output_name(index, key)
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {destination}")
        graph = (
            f"[0:v]scale={args.width}:{args.height}:flags=lanczos,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={max(0.0, duration-left_duration):.6f},"
            f"drawtext=fontfile='{font}':text='{ffmpeg_escape(args.left_label)}':"
            "x=(w-text_w)/2:y=18:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=0x176b45@0.88:boxborderw=10[left];"
            f"[1:v]scale={args.width}:{args.height}:flags=lanczos,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={max(0.0, duration-right_duration):.6f},"
            f"drawtext=fontfile='{font}':text='{ffmpeg_escape(args.right_label)}':"
            "x=(w-text_w)/2:y=18:fontsize=28:fontcolor=white:"
            "box=1:boxcolor=0x7a3426@0.88:boxborderw=10[right];"
            "[left][right]hstack=inputs=2[out]"
        )
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(left[key]), "-i", str(right[key]),
                "-filter_complex", graph, "-map", "[out]",
                "-t", f"{duration:.6f}", "-r", str(args.fps),
                "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
            ],
            check=True,
        )
        out_duration, out_fps, out_frames = metadata(destination)
        rows.append(
            {
                "index": index,
                "key": key,
                "output": destination.name,
                "left_file": left[key].name,
                "right_file": right[key].name,
                "left_duration_s": round(left_duration, 4),
                "right_duration_s": round(right_duration, 4),
                "output_duration_s": round(out_duration, 4),
                "fps": out_fps,
                "frames": out_frames,
                "width": args.width * 2,
                "height": args.height,
            }
        )
        print(f"[{index:02d}/{total:02d}] {destination.name}", flush=True)

    index_file = output / "SIDE_BY_SIDE_INDEX.csv"
    with index_file.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={output}")
    print(f"videos={len(rows)}")


if __name__ == "__main__":
    main()
