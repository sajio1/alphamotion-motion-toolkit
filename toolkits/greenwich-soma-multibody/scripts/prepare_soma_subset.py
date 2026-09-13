"""Extract an explicit, small SOMA subset for the reusable pipeline.

The archive is streamed once and only requested BVHs plus the bind skeleton are
materialized.  This keeps local probes reproducible without expanding the full
SOMA archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--bind", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stems", required=True, help="Comma-separated move_name values")
    a = p.parse_args()

    requested = [x.strip() for x in a.stems.split(",") if x.strip()]
    if not requested or len(requested) != len(set(requested)):
        p.error("--stems must contain distinct, non-empty names")
    frame = pd.read_parquet(a.metadata)
    rows = []
    for index, stem in enumerate(requested, 1):
        hit = frame[frame["move_name"].astype(str) == stem]
        if len(hit) != 1:
            raise ValueError(f"Expected exactly one metadata row for {stem!r}, found {len(hit)}")
        record = hit.iloc[0].to_dict()
        path = str(record["move_soma_uniform_path"])
        rows.append({
            "index": index, "stem": stem, "path": path,
            "move_name": str(record.get("move_name") or stem),
            "frames": int(record.get("move_duration_frames") or 0),
            "duration_s": float(record.get("move_duration_frames") or 0) / 120.0048,
            "movement_type": str(record.get("content_type_of_movement") or ""),
            "body_position": str(record.get("content_body_position") or ""),
            "semantic_type": str(record.get("content_short_description_2") or record.get("content_name") or stem),
            "label": str(record.get("content_short_description_2") or record.get("content_name") or stem),
            "description": str(record.get("content_natural_desc_4") or record.get("content_natural_desc_1") or ""),
            "package": str(record.get("package") or ""),
            "category": str(record.get("category") or ""),
        })

    source = a.output / "outputs" / "seed-100-contact" / "source"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.bind, source / "soma_base_skel_minimal.bvh")
    wanted = {row["path"]: row for row in rows}
    with tarfile.open(a.archive, "r|gz") as archive:
        for member in archive:
            row = wanted.get(member.name)
            if row is None or not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            with (source / f"{row['stem']}.bvh").open("wb") as target:
                shutil.copyfileobj(stream, target, 1024 * 1024)
            del wanted[member.name]
            if not wanted:
                break
    if wanted:
        raise FileNotFoundError(f"Missing requested archive members: {sorted(wanted)}")
    selection = a.output / "outputs" / "seed-100-contact" / "selection.json"
    selection.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
