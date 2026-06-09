#!/usr/bin/env python3
"""Fusion ablation over the co-observed SubPipeMini sequence.

Drives the late-fusion tracker in three modes (rgb_only, sonar_only, fused) over
two scenarios:

  clean   - frames as captured. Both sensors are available throughout, so all
            three modes hold the pipe track; this is the "nothing is wrong"
            baseline.
  stress  - the same sequence with two scripted degradation windows taken from
            the augmentation bank: the camera is hit by turbidity over one
            stretch and the sonar by motion smear over another. A single-sensor
            mode loses the track during its own sensor's window; the fused mode
            rides through both because the other sensor carries it.

The headline output is track continuity (uptime, longest lost stretch) per mode
per scenario, written to results/fusion/ along with a status figure.

Inference runs once per scenario; the three modes are cheap replays over the
cached detections. The RGB mask is dropped from the cache (the projection needs
only presence and centroid) to keep memory flat over thousands of ticks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config

# Default scripted windows, as fractions of the processed sequence duration.
RGB_WINDOW = (0.22, 0.42)
SONAR_WINDOW = (0.58, 0.78)
RGB_CONDITION = "turbidity_haze"
SONAR_CONDITION = "motion_smear"
STRESS_SEVERITY = 0.9
MODES = ("rgb_only", "sonar_only", "fused")


def _parse_pair(s: str) -> tuple[float, float]:
    lo, hi = (float(v) for v in s.split(","))
    return lo, hi


def _discover_sequence(cfg: dict, override: str | None, data_root: str | None) -> Path:
    from src.data.loaders import discover_chunks
    if override:
        p = Path(override)
        return p if p.is_absolute() else REPO_ROOT / p
    if data_root:
        root = Path(data_root)
        root = root if root.is_absolute() else REPO_ROOT / root
    else:
        root = REPO_ROOT / cfg["data"]["rgb_root"]
    chunks = discover_chunks(root)
    if not chunks:
        raise FileNotFoundError(
            f"no sequence (no {root}/**/EstimatedState.csv). "
            f"Pass --sequence explicitly.")
    return chunks[0]


def _load_models(cfg: dict):
    from src.inference.rgb_segmenter import RGBSegmenter
    from src.inference.sonar_detector import SonarDetector
    rc = cfg["models"]["rgb_seg"]
    sc = cfg["models"]["sonar_det"]
    rgb_w = REPO_ROOT / rc["weights"]
    son_w = REPO_ROOT / sc["weights"]
    for w in (rgb_w, son_w):
        if not w.exists():
            raise FileNotFoundError(f"weights not found: {w}")
    seg = RGBSegmenter(str(rgb_w), backend="pytorch",
                       imgsz=int(cfg["project"]["rgb_imgsz"]),
                       conf=rc.get("conf", 0.25), iou=rc.get("iou", 0.5))
    det = SonarDetector(str(son_w), backend="pytorch",
                        imgsz=int(cfg["project"]["sonar_imgsz"]),
                        conf=sc.get("conf", 0.25), iou=sc.get("iou", 0.5))
    return seg, det


def run_pass(loader, seg_model, det_model, *, fps: float, max_seconds, schedule,
             degrade: bool, t_start=None, t_end=None):
    """One inference pass over the (subsampled) timeline.

    Returns a list of (t, SegResult, DetResult, Pose) with the RGB mask stripped.
    ``schedule`` is {"rgb": (t0, t1, condition, severity), "sonar": (...)} in
    absolute seconds; degradations are applied only when ``degrade`` is True.
    """
    from src.data.augmentations import DegradationPipeline
    from src.inference.rgb_segmenter import SegResult
    from src.inference.sonar_detector import DetResult

    rgb_deg = DegradationPipeline("rgb")
    son_deg = DegradationPipeline("sonar")
    tick_dt = 1.0 / float(fps)

    records = []
    t0 = None
    last_t = None
    for frame in loader:
        t = float(frame.t)
        if t_start is not None and t < t_start:
            continue
        if t_end is not None and t > t_end:
            break
        if t0 is None:
            t0 = t
        if max_seconds is not None and (t - t0) > float(max_seconds):
            break
        if last_t is not None and (t - last_t) < tick_dt:
            continue
        last_t = t

        rgb_img = frame.rgb
        son_img = frame.sonar
        if degrade and schedule:
            rw = schedule.get("rgb")
            sw = schedule.get("sonar")
            if rgb_img is not None and rw and rw[0] <= t <= rw[1]:
                rgb_img = rgb_deg.apply(rgb_img, rw[2], rw[3])
            if son_img is not None and sw and sw[0] <= t <= sw[1]:
                son_img = son_deg.apply(son_img, sw[2], sw[3])

        seg = seg_model.infer(rgb_img) if rgb_img is not None else SegResult()
        det = det_model.infer(son_img) if son_img is not None else DetResult()
        seg.mask = None  # drop the full-frame mask; the projection does not use it
        records.append((t, seg, det, frame.pose))
    return records


def replay(records, cfg: dict, mode: str):
    """Replay one fusion mode over cached records; return (per_tick, metrics)."""
    from src.fusion.late_fusion import LateFusion

    fz = LateFusion(cfg)
    max_coast = int(cfg["fusion"]["max_coast_frames"])
    per_tick = []
    initialized = False
    for (t, seg, det, pose) in records:
        r = fz.step(seg, det, pose, mode=mode)
        if r.rgb_contributed or r.sonar_contributed:
            initialized = True
        coast = int(r.track.coast_frames)
        per_tick.append({
            "t": t,
            "pos": float(r.track.position),
            "locked": bool(initialized and coast <= max_coast),
            "rgb": bool(r.rgb_contributed),
            "sonar": bool(r.sonar_contributed),
        })
    return per_tick, _metrics(per_tick)


def _metrics(per_tick) -> dict:
    n = len(per_tick)
    if n == 0:
        return {"uptime_pct": 0.0, "longest_lost_s": 0.0,
                "rgb_updates": 0, "sonar_updates": 0, "ticks": 0}
    locked = sum(p["locked"] for p in per_tick)
    longest = 0.0
    start = None
    for p in per_tick:
        if not p["locked"]:
            start = p["t"] if start is None else start
            longest = max(longest, p["t"] - start)
        else:
            start = None
    return {
        "uptime_pct": round(100.0 * locked / n, 1),
        "longest_lost_s": round(longest, 1),
        "rgb_updates": int(sum(p["rgb"] for p in per_tick)),
        "sonar_updates": int(sum(p["sonar"] for p in per_tick)),
        "ticks": n,
    }


def _runs(per_tick, t0, want_locked: bool):
    """(start, width) elapsed-second spans where locked == want_locked."""
    spans = []
    run_start = None
    prev = None
    for p in per_tick:
        e = p["t"] - t0
        if p["locked"] == want_locked:
            if run_start is None:
                run_start = e
        else:
            if run_start is not None:
                spans.append((run_start, max(prev - run_start, 0.05)))
                run_start = None
        prev = e
    if run_start is not None and prev is not None:
        spans.append((run_start, max(prev - run_start, 0.05)))
    return spans


def plot_ablation(clean: dict, stress: dict, schedule, t0: float, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for ax, (title, tracks, show_windows) in zip(
        axes, [("clean", clean, False), ("stress (scripted degradation windows)", stress, True)]
    ):
        for lane, mode in enumerate(MODES):
            per_tick = tracks[mode]["per_tick"]
            ax.broken_barh(_runs(per_tick, t0, True), (lane - 0.38, 0.76),
                           facecolors="#2ca02c")
            ax.broken_barh(_runs(per_tick, t0, False), (lane - 0.38, 0.76),
                           facecolors="#d62728")
        if show_windows and schedule:
            for key, label in (("rgb", "RGB: turbidity"), ("sonar", "sonar: smear")):
                w = schedule.get(key)
                if w:
                    ax.axvspan(w[0] - t0, w[1] - t0, color="black", alpha=0.07, zorder=0)
                    ax.text((0.5 * (w[0] + w[1])) - t0, 2.62, label, ha="center",
                            va="bottom", fontsize=9, color="black")
        ax.set_yticks(range(len(MODES)))
        ax.set_yticklabels([m.replace("_", " ") for m in MODES])
        ax.set_ylim(-0.6, 2.85)
        ax.set_title(f"{title}", fontsize=12, loc="left")
        ax.set_xlabel("time (s)")
    axes[0].legend(handles=[Patch(facecolor="#2ca02c", label="track locked"),
                            Patch(facecolor="#d62728", label="track lost (coasting past limit)")],
                   loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Pipe-track continuity: RGB-only vs sonar-only vs fused", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RGB-only vs sonar-only vs fused track-continuity ablation.")
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--sequence", default=None, help="sequence dir (default: auto-discover in data.rgb_root)")
    ap.add_argument("--data-root", default=None,
                    help="dataset root to auto-discover the chunk in (e.g. data/subpipe/SubPipeMiniSSS)")
    ap.add_argument("--fps", type=float, default=5.0, help="processing rate (subsamples the timeline)")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap the processed duration")
    ap.add_argument("--time-window", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="absolute start/end timestamps to crop to (intersected with the nav span)")
    ap.add_argument("--severity", type=float, default=STRESS_SEVERITY)
    ap.add_argument("--rgb-window", type=_parse_pair, default=RGB_WINDOW)
    ap.add_argument("--sonar-window", type=_parse_pair, default=SONAR_WINDOW)
    ap.add_argument("--rgb-condition", default=RGB_CONDITION)
    ap.add_argument("--sonar-condition", default=SONAR_CONDITION)
    args = ap.parse_args()

    cfg = load_config(args.config)
    from src.data.loaders import SequenceLoader

    seq = _discover_sequence(cfg, args.sequence, args.data_root)
    print(f"[seq] {seq}")
    loader = SequenceLoader(seq, cfg)
    timeline = loader._timeline
    if len(timeline) == 0:
        raise SystemExit("empty timeline: no RGB or sonar frames found under the sequence dir.")
    pose_ts = loader._pose_ts
    # Only fuse where a real INS pose exists. Outside the nav span, along_track is
    # extrapolated from the nearest pose and is unreliable, so clip the processed
    # window to the nav range. This also drops sonar that extends past the nav
    # (as in SubPipeMiniSSS) and leaves SubPipeMini, whose nav spans the whole
    # window, unchanged.
    if len(pose_ts):
        t_start = max(float(timeline[0]), float(pose_ts[0]))
        t_end = min(float(timeline[-1]), float(pose_ts[-1]))
    else:
        t_start, t_end = float(timeline[0]), float(timeline[-1])
    if args.max_seconds:
        t_end = min(t_end, t_start + args.max_seconds)
    if args.time_window:
        tw0, tw1 = (float(v) for v in args.time_window)
        new_start, new_end = max(t_start, tw0), min(t_end, tw1)
        if new_end <= new_start:
            raise SystemExit(
                f"--time-window [{tw0:.1f}, {tw1:.1f}] does not overlap the "
                f"nav-clipped span [{t_start:.1f}, {t_end:.1f}]")
        t_start, t_end = new_start, new_end
        print(f"[seq] cropped to time-window [{t_start:.1f}, {t_end:.1f}] "
              f"({t_end - t_start:.0f}s)")
    dur = max(t_end - t_start, 1e-3)
    schedule = {
        "rgb": (t_start + args.rgb_window[0] * dur, t_start + args.rgb_window[1] * dur,
                args.rgb_condition, args.severity),
        "sonar": (t_start + args.sonar_window[0] * dur, t_start + args.sonar_window[1] * dur,
                  args.sonar_condition, args.severity),
    }
    print(f"[seq] nav-clipped window {dur:.0f}s, processing at {args.fps} fps "
          f"(~{int(dur * args.fps)} ticks/scenario)")
    print(f"[stress] RGB {args.rgb_condition} over "
          f"[{args.rgb_window[0]:.2f},{args.rgb_window[1]:.2f}], "
          f"sonar {args.sonar_condition} over "
          f"[{args.sonar_window[0]:.2f},{args.sonar_window[1]:.2f}] at severity {args.severity}")

    seg_model, det_model = _load_models(cfg)

    results = {"sequence": str(seq), "fps": args.fps, "duration_s": round(dur, 1),
               "schedule": {k: [round(v[0] - t_start, 1), round(v[1] - t_start, 1), v[2], v[3]]
                            for k, v in schedule.items()}}
    plot_data = {}
    for scenario, degrade in (("clean", False), ("stress", True)):
        print(f"\n[{scenario}] inference pass ...")
        loader = SequenceLoader(seq, cfg)  # fresh iterator
        records = run_pass(loader, seg_model, det_model, fps=args.fps,
                           max_seconds=None, schedule=schedule, degrade=degrade,
                           t_start=t_start, t_end=t_end)
        print(f"[{scenario}] {len(records)} ticks; replaying {len(MODES)} modes")
        results[scenario] = {}
        plot_data[scenario] = {}
        for mode in MODES:
            per_tick, m = replay(records, cfg, mode)
            results[scenario][mode] = m
            plot_data[scenario][mode] = {"per_tick": per_tick}
            print(f"  {mode:10s} uptime {m['uptime_pct']:5.1f}%  "
                  f"longest_lost {m['longest_lost_s']:5.1f}s  "
                  f"rgb_upd {m['rgb_updates']:4d}  sonar_upd {m['sonar_updates']:4d}")

    out_dir = REPO_ROOT / "results" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[json] wrote {out_dir / 'ablation.json'}")
    plot_ablation(plot_data["clean"], plot_data["stress"], schedule, t_start,
                  out_dir / "fusion_continuity.png")


if __name__ == "__main__":
    main()
