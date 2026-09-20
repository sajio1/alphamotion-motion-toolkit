"""Collect simulation videos from a delivery tree and write a verified index."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2


def inspect(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid video metadata: {path}")
    duration = frames / fps
    size = path.stat().st_size
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "frames": frames,
        "duration_s": round(duration, 6),
        "bytes": size,
        "approx_bitrate_kbps": round(size * 8 / duration / 1000, 3),
    }


def parse_group(value: str) -> tuple[str, str]:
    source, separator, destination = value.partition("=")
    if not separator or not source or not destination:
        raise argparse.ArgumentTypeError("group must be SOURCE_GROUP=OUTPUT_NAME")
    return source, destination


def discover_groups(delivery: Path) -> list[tuple[str, str]]:
    groups = []
    for child in sorted(delivery.iterdir()):
        if child.is_dir() and (child / "jobs").is_dir():
            groups.append((child.name, child.name))
    if not groups:
        raise RuntimeError(f"no groups containing jobs/ found under {delivery}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True, help="Unpacked delivery root")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory")
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        default=[],
        metavar="SOURCE_GROUP=OUTPUT_NAME",
        help="Rename/select a group; repeat as needed. Default: discover all groups.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing videos")
    args = parser.parse_args()

    delivery = args.delivery.resolve()
    output = args.output.resolve()
    if not delivery.is_dir():
        parser.error(f"delivery does not exist: {delivery}")
    groups = args.group or discover_groups(delivery)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for source_group, target_group in groups:
        jobs = delivery / source_group / "jobs"
        if not jobs.is_dir():
            raise RuntimeError(f"missing jobs directory: {jobs}")
        target = output / target_group
        target.mkdir(parents=True, exist_ok=True)
        for video in sorted(jobs.glob("*/video/*.mp4")):
            job = video.parents[1].name
            destination = target / f"{job}.mp4"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite: {destination}")
            shutil.copy2(video, destination)
            rows.append(
                {
                    "source_group": source_group,
                    "output_group": target_group,
                    "job": job,
                    "relative_file": destination.relative_to(output).as_posix(),
                    **inspect(destination),
                }
            )

    if not rows:
        raise RuntimeError("no MP4 videos found")
    index = output / "VIDEO_INDEX.csv"
    with index.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    rates = [float(row["approx_bitrate_kbps"]) for row in rows]
    print(f"output={output}")
    print(f"videos={len(rows)}")
    print(f"resolutions={sorted({(row['width'], row['height']) for row in rows})}")
    print(f"fps_values={sorted({row['fps'] for row in rows})}")
    print(f"approx_bitrate_kbps=min:{min(rates):.1f} max:{max(rates):.1f} mean:{sum(rates)/len(rates):.1f}")


if __name__ == "__main__":
    main()
