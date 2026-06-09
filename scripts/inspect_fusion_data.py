#!/usr/bin/env python3
"""Read-only: what can we actually fuse on?

Fusion needs a sequence where the camera and the sonar observe the same pipe
along one INS trajectory. The two SubPipeMini subsets are single-modality, so
this reports, for each subset, whether INS nav (EstimatedState.csv) is present
and the timestamp span of its frames, then checks whether the RGB span and the
sonar span overlap. Overlap with nav present means the Minis may be mergeable
into one sequence; no overlap means a dual-modality SubPipe chunk is needed for
genuine co-observation.

Stdlib only, reads nothing into the models, writes nothing.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAV_FILE = "EstimatedState.csv"

RGB_GLOBS = ["Cam0_images/*"]
SONAR_GLOBS = ["SSS_HF_images/Image/*", "*.pbm", "*.png", "*.jpg"]


def _ts_span(paths):
    """min/max/count of the numeric timestamp parsed from each filename stem."""
    vals = []
    for p in paths:
        nums = re.findall(r"\d+\.\d+|\d+", p.stem)
        if nums:
            vals.append(float(max(nums, key=len)))   # longest numeric run = the stamp
    return (min(vals), max(vals), len(vals)) if vals else None


def _first_nonempty(root: Path, patterns):
    for pat in patterns:
        files = [p for p in root.rglob(pat) if p.is_file()]
        if files:
            return files, pat
    return [], None


def _csv_time_span(nav: Path):
    """Crude min/max of the first time-like column in an EstimatedState.csv."""
    try:
        with open(nav, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            ti = None
            for i, h in enumerate(header):
                hl = h.strip().lower()
                if hl in ("t", "time", "timestamp", "t (s)", "time (s)") or "time" in hl:
                    ti = i
                    break
            if ti is None:
                return None
            lo = hi = None
            n = 0
            for row in reader:
                if ti < len(row):
                    try:
                        v = float(row[ti])
                    except ValueError:
                        continue
                    lo = v if lo is None else min(lo, v)
                    hi = v if hi is None else max(hi, v)
                    n += 1
            return (lo, hi, n) if n else None
    except Exception:
        return None


def scan(name: str, root: Path):
    print(f"\n=== {name}")
    print(f"    {root}")
    if not root.exists():
        print("    MISSING")
        return None
    navs = list(root.rglob(NAV_FILE))
    print(f"    {NAV_FILE}: {len(navs)} found")
    for nav in navs[:1]:
        sp = _csv_time_span(nav)
        if sp:
            print(f"      first nav time span: t=[{sp[0]:.1f} .. {sp[1]:.1f}]  ({sp[2]} rows)")
    cam, _ = _first_nonempty(root, RGB_GLOBS)
    son, son_pat = _first_nonempty(root, SONAR_GLOBS)
    rgb = _ts_span(cam)
    sonar = _ts_span(son)
    if rgb:
        print(f"    RGB Cam0 frames: {rgb[2]:>5}   t=[{rgb[0]:.1f} .. {rgb[1]:.1f}]  span={rgb[1] - rgb[0]:.0f}s")
    if sonar:
        print(f"    sonar frames:    {sonar[2]:>5}   t=[{sonar[0]:.1f} .. {sonar[1]:.1f}]  "
              f"span={sonar[1] - sonar[0]:.0f}s   (via {son_pat})")
    return {"rgb": rgb, "sonar": sonar, "nav": len(navs)}


def main() -> None:
    base = REPO / "data" / "subpipe"
    roots = {
        "RGB Mini (SubPipeMini)": base / "SubPipeMini",
        "Sonar Mini (SubPipeMiniSSS)": base / "SubPipeMiniSSS",
        "Full SubPipe (if present)": base / "SubPipe",
    }
    info = {name: scan(name, root) for name, root in roots.items()}

    rgb = (info.get("RGB Mini (SubPipeMini)") or {}).get("rgb")
    son = (info.get("Sonar Mini (SubPipeMiniSSS)") or {}).get("sonar")
    print("\n--- verdict ---")
    if rgb and son:
        overlap = max(0.0, min(rgb[1], son[1]) - max(rgb[0], son[0]))
        if overlap > 0:
            print(f"RGB and sonar timestamps OVERLAP by {overlap:.0f}s -> likely the same dive; "
                  f"the Minis may be mergeable into one sequence (no big download).")
        else:
            gap = max(rgb[0], son[0]) - min(rgb[1], son[1])
            print(f"RGB and sonar timestamps DO NOT overlap (gap ~{gap:.0f}s) -> different windows; "
                  f"a dual-modality SubPipe chunk is needed for real co-observation.")
    else:
        print("Could not compare spans (a modality or its frames are missing above).")
    print("Nav must also exist for along-track fusion: check the EstimatedState.csv counts per subset.")


if __name__ == "__main__":
    main()
