#!/usr/bin/env python3
"""Read-only diagnostic for the per-sensor demo window.

Question it answers: do the annotated (pipe-present) frames actually appear in
the continuous footage the demo plays, or does the inspection-window selection
plus the --max-frames cap drop them?

It reuses the demo's own frame-selection functions, so it reflects exactly what
make_demo_video.py would render. No model is loaded; it is fast and read-only.

    python scripts/diagnose_demo_window.py --modality rgb
    python scripts/diagnose_demo_window.py --modality sonar   # working control
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _norm(stem: str) -> str:
    """Strip a leading sensor/chunk prefix (e.g. 'c0_') so prepared frame names
    line up with the bare-timestamp raw frame names."""
    return re.sub(r"^[A-Za-z]+\d*_", "", stem)


def _analyze(ann_ts: dict[str, float], full_stems: list[str],
             played_stems: set[str], cap: int) -> dict:
    """Pure comparison logic (unit-tested separately)."""
    full_set = set(full_stems)
    idxs = [i for i, s in enumerate(full_stems) if s in ann_ts]
    n = len(full_stems)
    deciles = [0] * 10
    if n:
        for i in idxs:
            deciles[min(9, int(10 * i / n))] += 1
    return {
        "n_ann": len(ann_ts),
        "n_full": n,
        "n_played": len(played_stems),
        "in_window": sum(1 for s in ann_ts if s in full_set),
        "in_played": sum(1 for s in ann_ts if s in played_stems),
        "idx_min": min(idxs) if idxs else None,
        "idx_max": max(idxs) if idxs else None,
        "before_cap": sum(1 for i in idxs if i < cap),
        "deciles": deciles,
        "missing_examples": [s for s in ann_ts if s not in full_set][:5],
    }


def _format(d: dict, modality: str, cap: int, span: float | None) -> str:
    L = []
    L.append(f"modality                         : {modality}")
    L.append(f"annotated (pipe-present) frames  : {d['n_ann']}")
    if span is not None:
        L.append(f"label-timestamp span             : {span:.1f} (timestamp units)")
    L.append(f"continuous frames inside window  : {d['n_full']}")
    L.append(f"frames the demo actually plays   : {d['n_played']}  (cap = {cap})")
    L.append("")
    L.append(f"annotated frames in FULL window  : {d['in_window']}/{d['n_ann']}")
    L.append(f"annotated frames in PLAYED clip  : {d['in_played']}/{d['n_ann']}")
    if d["idx_min"] is not None:
        L.append(f"annotated sit at window indices  : {d['idx_min']}..{d['idx_max']} "
                 f"of {d['n_full']}")
        L.append(f"annotated falling before the cap : {d['before_cap']}/{d['n_ann']}")
        L.append(f"distribution across window (deciles, start->end): {d['deciles']}")
    L.append("")
    # Verdict
    if d["n_ann"] == 0:
        L.append("VERDICT: no annotated frames found at all (check prep paths).")
    elif d["in_window"] < d["n_ann"]:
        L.append("VERDICT: SOURCE MISMATCH. Some annotated frames are not even in the "
                 "discovered continuous stream, so the demo is reading a different set "
                 "of images than the annotations came from.")
        if d["missing_examples"]:
            L.append(f"         missing stem examples: {d['missing_examples']}")
    elif d["in_played"] < d["in_window"]:
        L.append("VERDICT: CAP IS DROPPING THEM. The pipe frames are in the window but "
                 "fall past the --max-frames cut, so the played clip is mostly the "
                 "open-water lead-in. Widen/relocate the window so the clip covers them.")
    else:
        L.append("VERDICT: the pipe frames are present in the played clip; if they look "
                 "sparse it is genuine intermittency (pipe drifting out of the camera "
                 "view), not a selection artifact.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the demo's frame window.")
    ap.add_argument("--modality", default="rgb", choices=["rgb", "sonar"])
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--max-frames", type=int, default=600)
    args = ap.parse_args()

    import scripts.make_demo_video as dv
    from src.data.loaders import _parse_timestamp

    cfg = dv.load_config(args.config)
    sub = dv.MODELS[args.modality]["cfg"]
    out_base = dv.REPO_ROOT / cfg["data"].get("prep", {}).get("out_dir", "data/subpipe/yolo")

    ann_ts: dict[str, float] = {}
    for split in ("train", "val"):
        ld = out_base / sub / "labels" / split
        if not ld.is_dir():
            continue
        for f in ld.glob("*.txt"):
            if f.read_text(encoding="utf-8").strip():
                try:
                    ann_ts[_norm(f.stem)] = _parse_timestamp(f.stem)
                except Exception:
                    pass

    rng = dv._labeled_ts_range(args.modality, cfg)
    span = (rng[1] - rng[0]) if rng else None

    full = dv._find_frames(args.modality, cfg, None, stride=1,
                           max_frames=10 ** 9, segment="labeled")
    played = dv._find_frames(args.modality, cfg, None, stride=1,
                             max_frames=args.max_frames, segment="labeled")

    d = _analyze(ann_ts, [_norm(p.stem) for p in full],
                 {_norm(p.stem) for p in played}, args.max_frames)
    print(_format(d, args.modality, args.max_frames, span))


if __name__ == "__main__":
    main()
