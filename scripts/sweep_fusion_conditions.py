#!/usr/bin/env python3
"""Sweep the fusion ablation across every RGB failure mode.

Holds the sonar at its one real failure (motion smear, applied over the first
half) and varies the camera's degradation over the second half, one condition at
a time, on the co-observed window. For each condition it reports track-continuity
uptime for rgb_only, sonar_only, and fused, so the picture is: fused stays pinned
near 100% across the whole range of optical failures, sonar_only sits at whatever
its early smear costs it, and rgb_only dips by an amount that depends on how badly
each condition hurts the camera (a lot for gaussian noise or backscatter, barely
at all for color attenuation or motion blur, which the camera shrugs off).

The model load and the clean baseline happen once; only the stress pass is
repeated per condition, so the whole sweep runs in a couple of minutes.

Writes results/fusion/condition_sweep.{json,png}. Read-only otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config
from scripts.run_fusion_ablation import (
    MODES,
    _discover_sequence,
    _load_models,
    replay,
    run_pass,
)

# Staggered so each sensor fails in its own half and the other carries the fused
# track: sonar smeared early (camera carries), camera degraded late (sonar carries).
SONAR_CONDITION = "motion_smear"
SONAR_WINDOW = (0.05, 0.50)
RGB_WINDOW = (0.50, 0.95)
DEFAULT_RGB_CONDITIONS = [
    "turbidity_haze", "color_attenuation", "low_light", "motion_blur",
    "gaussian_noise", "overexposure", "backscatter", "sand_occlusion",
]


def _parse_pair(s: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in s.split(","))
    return lo, hi


def _window_bounds(loader, args) -> tuple[float, float]:
    """Nav-clipped [t_start, t_end] with optional --max-seconds and --time-window."""
    timeline = loader._timeline
    if len(timeline) == 0:
        raise SystemExit("empty timeline: no RGB or sonar frames under the sequence dir.")
    pose_ts = loader._pose_ts
    if len(pose_ts):
        t_start = max(float(timeline[0]), float(pose_ts[0]))
        t_end = min(float(timeline[-1]), float(pose_ts[-1]))
    else:
        t_start, t_end = float(timeline[0]), float(timeline[-1])
    if args.max_seconds:
        t_end = min(t_end, t_start + args.max_seconds)
    if args.time_window:
        tw0, tw1 = (float(v) for v in args.time_window)
        t_start, t_end = max(t_start, tw0), min(t_end, tw1)
        if t_end <= t_start:
            raise SystemExit("--time-window does not overlap the nav-clipped span")
    return t_start, t_end


def plot_sweep(rows: list[dict], clean_row: dict, sonar_condition: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conds = [r["rgb_condition"] for r in rows]
    x = np.arange(len(conds))
    w = 0.26
    colors = {"rgb_only": "#1f77b4", "sonar_only": "#ff7f0e", "fused": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(max(9.0, 1.25 * len(conds)), 5.4))
    for i, mode in enumerate(MODES):
        vals = [r[mode]["uptime_pct"] for r in rows]
        ax.bar(x + (i - 1) * w, vals, w, label=mode.replace("_", " "), color=colors[mode])
    ax.axhline(100, color="#999999", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("track uptime (%)")
    ax.set_title(
        "Fusion vs single sensors across RGB failure modes\n"
        f"sonar held at {sonar_condition.replace('_', ' ')}; clean baseline "
        f"rgb {clean_row['rgb_only']['uptime_pct']:.0f}% / "
        f"sonar {clean_row['sonar_only']['uptime_pct']:.0f}% / "
        f"fused {clean_row['fused']['uptime_pct']:.0f}%",
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep the fusion ablation across RGB failure modes.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--time-window", type=float, nargs=2, default=None, metavar=("T0", "T1"))
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--sonar-condition", default=SONAR_CONDITION)
    ap.add_argument("--sonar-window", type=_parse_pair, default=SONAR_WINDOW)
    ap.add_argument("--rgb-window", type=_parse_pair, default=RGB_WINDOW)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_RGB_CONDITIONS,
                    help="RGB conditions to sweep (default: all eight)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from src.data.augmentations import DegradationPipeline
    from src.data.loaders import SequenceLoader

    valid = set(DegradationPipeline("rgb").conditions())
    conds = [c for c in args.conditions if c in valid]
    bad = [c for c in args.conditions if c not in valid]
    if bad:
        print(f"[warn] skipping unknown RGB conditions: {bad}")
    if not conds:
        raise SystemExit(f"no valid RGB conditions; choose from {sorted(valid)}")

    seq = _discover_sequence(cfg, args.sequence, args.data_root)
    print(f"[seq] {seq}")
    loader = SequenceLoader(seq, cfg)
    t_start, t_end = _window_bounds(loader, args)
    dur = max(t_end - t_start, 1e-3)
    print(f"[seq] window [{t_start:.1f}, {t_end:.1f}] ({dur:.0f}s) at {args.fps} fps")
    print(f"[plan] sonar {args.sonar_condition} over "
          f"[{args.sonar_window[0]:.2f},{args.sonar_window[1]:.2f}], "
          f"RGB <condition> over [{args.rgb_window[0]:.2f},{args.rgb_window[1]:.2f}] "
          f"at severity {args.severity}; sweeping {len(conds)} conditions")

    seg_model, det_model = _load_models(cfg)
    sonar_slot = (t_start + args.sonar_window[0] * dur, t_start + args.sonar_window[1] * dur,
                  args.sonar_condition, args.severity)

    # Clean baseline, shared across every condition.
    print("\n[clean] inference pass ...")
    rec_clean = run_pass(loader, seg_model, det_model, fps=args.fps, max_seconds=None,
                         schedule=None, degrade=False, t_start=t_start, t_end=t_end)
    clean_row = {}
    for mode in MODES:
        _, m = replay(rec_clean, cfg, mode)
        clean_row[mode] = m
    print(f"[clean] {len(rec_clean)} ticks | "
          + " | ".join(f"{mode.replace('_', ' ')} {clean_row[mode]['uptime_pct']:.1f}%"
                       for mode in MODES))

    # One stress pass per RGB condition.
    print("\n[stress] sweeping conditions ...")
    rows = []
    for cond in conds:
        schedule = {
            "rgb": (t_start + args.rgb_window[0] * dur, t_start + args.rgb_window[1] * dur,
                    cond, args.severity),
            "sonar": sonar_slot,
        }
        rec = run_pass(loader, seg_model, det_model, fps=args.fps, max_seconds=None,
                       schedule=schedule, degrade=True, t_start=t_start, t_end=t_end)
        row = {"rgb_condition": cond}
        for mode in MODES:
            _, m = replay(rec, cfg, mode)
            row[mode] = m
        rows.append(row)
        print(f"  {cond:18s}  rgb_only {row['rgb_only']['uptime_pct']:5.1f}%  "
              f"sonar_only {row['sonar_only']['uptime_pct']:5.1f}%  "
              f"fused {row['fused']['uptime_pct']:5.1f}%")

    out_dir = REPO_ROOT / "results" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"sequence": str(seq), "window_s": round(dur, 1), "fps": args.fps,
               "severity": args.severity, "sonar_condition": args.sonar_condition,
               "rgb_window": list(args.rgb_window), "sonar_window": list(args.sonar_window),
               "clean": clean_row, "stress_by_condition": rows}
    (out_dir / "condition_sweep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[json] wrote {out_dir / 'condition_sweep.json'}")
    plot_sweep(rows, clean_row, args.sonar_condition, out_dir / "condition_sweep.png")

    fused_lo = min(r["fused"]["uptime_pct"] for r in rows)
    fused_hi = max(r["fused"]["uptime_pct"] for r in rows)
    worst = min(rows, key=lambda r: r["rgb_only"]["uptime_pct"])
    print(f"\n[summary] fused stays {fused_lo:.1f}-{fused_hi:.1f}% across all conditions; "
          f"rgb_only bottoms out at {worst['rgb_only']['uptime_pct']:.1f}% "
          f"under {worst['rgb_condition']}; sonar_only sits near "
          f"{rows[0]['sonar_only']['uptime_pct']:.1f}% throughout (its early smear)")


if __name__ == "__main__":
    main()
