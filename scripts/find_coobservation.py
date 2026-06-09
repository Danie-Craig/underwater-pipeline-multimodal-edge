#!/usr/bin/env python3
"""Find a SubPipe chunk and window where both sensors observe the pipe.

The fusion ablation needs a stretch of survey where the RGB camera and the
side-scan sonar BOTH see the pipe at the same time. That is not guaranteed:
when the vehicle flies directly over the pipe the camera sees it clearly but
the pipe falls in the sonar's nadir gap, so long stretches of a survey are
sonar-pipe-absent. The two SubPipeMini subsets turned out to have no overlap
at all, which is why we pulled the full dataset.

The rare, decisive signal is sonar pipe-visibility. The dataset ships
ground-truth sonar boxes (one YOLO .txt per SSS image); a frame that carries a
box is precisely "the pipe is acoustically visible here". So we work in two
stages:

  triage (default, no models, seconds):
    Walk every chunk, count the sonar frames that carry a pipe box, and find
    the longest contiguous pipe-visible sonar run. Rank chunks by that run.
    Segmentation-mask coverage is reported alongside as a coarse RGB signal.

  verify (--verify [--chunk N], runs the models):
    For the chosen chunk, replay its pipe-visible sonar window through the real
    sonar detector and run the RGB segmenter on the nearest camera frame to
    each sonar timestamp. Report how many ticks both models fire above the
    fusion measurement gates, and the longest such window. That window is what
    run_fusion_ablation.py should target.

Read-only. Writes nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import REPO_ROOT, load_config
from src.data.loaders import (
    NAV_FILE,
    RGB_DIR,
    SEG_DIR,
    SEG_LABEL_SUFFIX,
    SONAR_DIRS,
    SONAR_IMG_SUBDIR,
    SONAR_YOLO_SUBDIR,
    ULTRALYTICS_IMG_EXTS,
    _nearest,
    _parse_timestamp,
    _sorted_by_timestamp,
    discover_chunks,
)


# ---------------------------------------------------------------------------
#  Layout resolution (robust to the Image/ subdir being present or not)
# ---------------------------------------------------------------------------
def _resolve_dir(parent: Path, *candidates: str) -> Path | None:
    """First existing directory among ``parent/<candidate>`` and ``parent``."""
    for c in candidates:
        p = parent / c if c else parent
        if p.is_dir():
            return p
    return None


def _sonar_dirs(chunk: Path, freq: str) -> tuple[Path | None, Path | None]:
    """Return (image_dir, yolo_label_dir) for one chunk, or (None, None)."""
    base = chunk / SONAR_DIRS.get(freq.upper(), SONAR_DIRS["HF"])
    if not base.is_dir():
        return None, None
    img_dir = _resolve_dir(base, SONAR_IMG_SUBDIR, "")
    lbl_dir = _resolve_dir(base, SONAR_YOLO_SUBDIR)
    return img_dir, lbl_dir


def _label_has_box(path: Path) -> bool:
    """True if a YOLO label file exists and carries at least one box line."""
    if path is None or not path.exists():
        return False
    try:
        return any(ln.strip() for ln in path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return False


def _runs(ts_sorted: np.ndarray, max_gap_s: float) -> list[tuple[float, float, int]]:
    """Group ascending timestamps into runs split by gaps over ``max_gap_s``.

    Returns (start_ts, end_ts, count) per run, longest duration first.
    """
    if len(ts_sorted) == 0:
        return []
    runs: list[tuple[float, float, int]] = []
    start = prev = float(ts_sorted[0])
    cnt = 1
    for t in (float(x) for x in ts_sorted[1:]):
        if t - prev <= max_gap_s:
            cnt += 1
        else:
            runs.append((start, prev, cnt))
            start = t
            cnt = 1
        prev = t
    runs.append((start, prev, cnt))
    return sorted(runs, key=lambda r: r[1] - r[0], reverse=True)


# ---------------------------------------------------------------------------
#  Per-chunk ground-truth scan
# ---------------------------------------------------------------------------
class ChunkScan:
    def __init__(self, chunk: Path, freq: str, max_gap_s: float) -> None:
        self.chunk = chunk
        self.ok = False
        self.note = ""
        self.n_sonar = 0
        self.n_boxed = 0
        self.boxed_ts = np.array([], dtype=float)
        self.best_run = (0.0, 0.0, 0)        # start, end, count
        self.n_seg = 0
        self.n_seg_in_run = 0

        img_dir, lbl_dir = _sonar_dirs(chunk, freq)
        if img_dir is None:
            self.note = "no sonar image dir"
            return
        if lbl_dir is None:
            self.note = "no YOLO_Annotation dir (cannot triage on ground truth; use --verify)"
            self.n_sonar = sum(1 for _ in img_dir.glob("*"))
            return

        boxed: list[float] = []
        n_sonar = 0
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in (".pbm", *ULTRALYTICS_IMG_EXTS):
                continue
            n_sonar += 1
            if _label_has_box(lbl_dir / f"{img.stem}.txt"):
                t = _parse_timestamp(img.stem)
                if np.isfinite(t):
                    boxed.append(t)
        self.n_sonar = n_sonar
        self.n_boxed = len(boxed)
        self.boxed_ts = np.array(sorted(boxed), dtype=float)
        runs = _runs(self.boxed_ts, max_gap_s)
        if runs:
            self.best_run = runs[0]

        # Coarse RGB signal: segmentation masks present, and how many land
        # inside the best sonar run window.
        seg_dir = chunk / SEG_DIR
        if seg_dir.is_dir():
            seg_ts = [
                _parse_timestamp(p.stem[: -len(SEG_LABEL_SUFFIX)])
                for p in seg_dir.glob(f"*{SEG_LABEL_SUFFIX}.png")
            ]
            seg_ts = np.array([t for t in seg_ts if np.isfinite(t)], dtype=float)
            self.n_seg = int(len(seg_ts))
            if self.n_seg and self.best_run[2] > 0:
                lo, hi, _ = self.best_run
                self.n_seg_in_run = int(np.count_nonzero((seg_ts >= lo) & (seg_ts <= hi)))

        self.ok = True

    @property
    def run_seconds(self) -> float:
        return self.best_run[1] - self.best_run[0]


def triage(chunks: list[Path], freq: str, max_gap_s: float) -> list[ChunkScan]:
    scans: list[ChunkScan] = []
    print("=" * 74)
    print(f" ground-truth triage  |  {len(chunks)} chunk(s)  |  sonar freq {freq.upper()}")
    print("=" * 74)
    for ci, chunk in enumerate(chunks):
        s = ChunkScan(chunk, freq, max_gap_s)
        scans.append(s)
        label = f"[{ci}] {chunk.name}"
        if not s.ok and s.n_sonar == 0:
            print(f"{label:24s} {s.note}")
            continue
        pct = (100.0 * s.n_boxed / s.n_sonar) if s.n_sonar else 0.0
        lo, hi, cnt = s.best_run
        print(f"{label:24s} sonar: {s.n_sonar:5d} frames, "
              f"{s.n_boxed:5d} pipe-annotated ({pct:4.1f}%)")
        if cnt > 0:
            print(f"{'':24s} longest pipe-visible run: {s.run_seconds:6.1f}s  "
                  f"({cnt} frames)  [t={lo:.1f} to {hi:.1f}]")
            print(f"{'':24s} seg masks: {s.n_seg} present, "
                  f"{s.n_seg_in_run} inside that run")
        elif s.note:
            print(f"{'':24s} {s.note}")
        else:
            print(f"{'':24s} no pipe-annotated sonar frames in this chunk")
    return scans


# ---------------------------------------------------------------------------
#  Model verification on the chosen chunk
# ---------------------------------------------------------------------------
def verify(chunk: Path, scan: ChunkScan, cfg: dict, freq: str, *,
           max_checks: int, conf_floor: float, max_gap_s: float,
           device: str) -> None:
    import cv2

    from src.inference.rgb_segmenter import RGBSegmenter
    from src.inference.sonar_detector import SonarDetector

    sonar_gate = float(cfg["fusion"]["sonar_measurement_conf"])
    rgb_gate = float(cfg["fusion"]["rgb_measurement_conf"])

    img_dir, lbl_dir = _sonar_dirs(chunk, freq)
    if img_dir is None:
        raise SystemExit(f"no sonar image dir under {chunk}")

    # Sonar frames to check: the pipe-annotated ones when ground truth exists,
    # otherwise every sonar frame (sampled).
    sonar_paths = _sorted_by_timestamp(
        [p for p in img_dir.glob("*")
         if p.suffix.lower() in (".pbm", *ULTRALYTICS_IMG_EXTS)]
    )
    if lbl_dir is not None:
        cand = [p for p in sonar_paths if _label_has_box(lbl_dir / f"{p.stem}.txt")]
        basis = "pipe-annotated sonar frames"
    else:
        cand = sonar_paths
        basis = "all sonar frames (no ground-truth labels found)"
    if not cand:
        raise SystemExit(f"no candidate sonar frames in {chunk}")

    # Even sampling across the candidate window, capped at max_checks.
    if len(cand) > max_checks:
        idx = np.linspace(0, len(cand) - 1, max_checks).round().astype(int)
        idx = sorted(set(int(i) for i in idx))
        cand = [cand[i] for i in idx]

    # RGB frame index for nearest-neighbour lookup.
    rgb_dir = chunk / RGB_DIR
    rgb_paths = _sorted_by_timestamp(
        [p for p in rgb_dir.glob("*") if p.suffix.lower() in ULTRALYTICS_IMG_EXTS]
    ) if rgb_dir.is_dir() else []
    rgb_ts = np.array([_parse_timestamp(p.stem) for p in rgb_paths], dtype=float)
    if len(rgb_paths) == 0:
        raise SystemExit(f"no RGB frames under {rgb_dir}")

    print("=" * 74)
    print(f" model verification  |  {chunk.name}")
    print(f" basis: {len(cand)} {basis}")
    print(f" gates: sonar>={sonar_gate:.2f}  rgb>={rgb_gate:.2f}  "
          f"(scoring at conf>={conf_floor:.2f})")
    print("=" * 74)

    son = SonarDetector(str(REPO_ROOT / cfg["models"]["sonar_det"]["weights"]),
                        backend="pytorch", conf=conf_floor, iou=0.5,
                        imgsz=int(cfg["project"]["sonar_imgsz"]), device=device)
    rgb = RGBSegmenter(str(REPO_ROOT / cfg["models"]["rgb_seg"]["weights"]),
                       backend="pytorch", conf=conf_floor, iou=0.5,
                       imgsz=int(cfg["project"]["rgb_imgsz"]), device=device)

    # Warm up both models (first call pays graph-build / allocation cost).
    son.infer(np.zeros((64, 640, 3), dtype=np.uint8))
    rgb.infer(np.zeros((64, 64, 3), dtype=np.uint8))

    sync_tol = float(cfg.get("data", {}).get("sync_tolerance_s", 0.5))
    both_ts: list[float] = []
    n_son = n_rgb = n_both = n_pair = 0
    for k, sp in enumerate(cand):
        t = _parse_timestamp(sp.stem)
        s_img = cv2.imread(str(sp), cv2.IMREAD_COLOR)
        if s_img is None:
            continue
        sres = son.infer(s_img)
        s_score = sres.best.score if sres.best is not None else 0.0

        ri = _nearest(rgb_ts, t)
        paired = abs(rgb_ts[ri] - t) <= sync_tol
        r_score = 0.0
        if paired:
            n_pair += 1
            r_img = cv2.imread(str(rgb_paths[ri]), cv2.IMREAD_COLOR)
            if r_img is not None:
                r_score = rgb.infer(r_img).score

        s_present = s_score >= sonar_gate
        r_present = r_score >= rgb_gate
        n_son += int(s_present)
        n_rgb += int(r_present)
        if s_present and r_present:
            n_both += 1
            both_ts.append(t)

        if (k + 1) % 50 == 0:
            print(f"  checked {k + 1}/{len(cand)}  "
                  f"sonar>={sonar_gate:.2f}:{n_son}  rgb>={rgb_gate:.2f}:{n_rgb}  both:{n_both}")

    print()
    print(f"checked        : {len(cand)} sonar frames")
    print(f"RGB paired      : {n_pair} had a camera frame within {sync_tol:.2f}s")
    print(f"sonar above gate: {n_son}")
    print(f"rgb above gate  : {n_rgb}")
    print(f"BOTH above gate : {n_both}")

    both_arr = np.array(sorted(both_ts), dtype=float)
    runs = _runs(both_arr, max_gap_s)
    if runs:
        lo, hi, cnt = runs[0]
        print(f"longest co-observed window: {hi - lo:.1f}s  ({cnt} ticks)  "
              f"[t={lo:.1f} to {hi:.1f}]")
        print()
        if cnt >= 5 and (hi - lo) >= 10.0:
            print("VERDICT: usable. Both sensors fire across a sustained window.")
        else:
            print("VERDICT: thin. Both fire, but the joint window is short; check "
                  "another chunk or widen the gates.")
        rel = chunk.relative_to(REPO_ROOT) if chunk.is_relative_to(REPO_ROOT) else chunk
        print()
        print("Run the ablation on this chunk:")
        print(f"  python scripts/run_fusion_ablation.py --sequence {rel} --fps 5")
    else:
        print("longest co-observed window: none (no tick had both above gate)")
        print()
        print("VERDICT: not co-observed at these gates. Try another chunk, or lower "
              "the fusion gates and re-verify.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/model_config.yaml")
    ap.add_argument("--data-root", default="data/subpipe/SubPipe",
                    help="root holding the chunks (chunks auto-discovered under it)")
    ap.add_argument("--sequence", default=None,
                    help="explicit chunk dir; overrides --data-root discovery")
    ap.add_argument("--freq", default="HF", choices=["HF", "LF"])
    ap.add_argument("--max-gap", type=float, default=3.0,
                    help="seconds; gaps larger than this split a run")
    ap.add_argument("--verify", action="store_true",
                    help="run the models on the chosen chunk to confirm both fire")
    ap.add_argument("--chunk", type=int, default=None,
                    help="chunk index to verify (default: best from triage)")
    ap.add_argument("--max-checks", type=int, default=400,
                    help="cap on sonar frames sampled during --verify")
    ap.add_argument("--conf-floor", type=float, default=0.10,
                    help="model conf during --verify; presence judged against the gates")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.sequence:
        seq = Path(args.sequence)
        if not seq.is_absolute():
            seq = REPO_ROOT / seq
        chunks = [seq]
    else:
        root = Path(args.data_root)
        if not root.is_absolute():
            root = REPO_ROOT / root
        chunks = discover_chunks(root)
        if not chunks:
            raise SystemExit(
                f"no chunks found under {root} "
                f"(looked for {NAV_FILE}). Check the extract path.")

    scans = triage(chunks, args.freq, args.max_gap)

    ranked = sorted(
        [s for s in scans if s.best_run[2] > 0],
        key=lambda s: s.run_seconds, reverse=True,
    )
    print()
    print("-" * 74)
    if ranked:
        print("Ranked by longest pipe-visible sonar run:")
        for rank, s in enumerate(ranked, 1):
            ci = chunks.index(s.chunk)
            tag = "  <-- best candidate" if rank == 1 else ""
            print(f"  {rank}. [{ci}] {s.chunk.name}  {s.run_seconds:6.1f}s{tag}")
    else:
        print("No chunk had ground-truth pipe-visible sonar frames.")
        print("If the dataset lacks YOLO_Annotation dirs, run --verify to scan with "
              "the model instead.")

    if not args.verify:
        if ranked:
            best_ci = chunks.index(ranked[0].chunk)
            print()
            print("Confirm both sensors fire there:")
            print(f"  python scripts/find_coobservation.py --verify --chunk {best_ci}"
                  f" --data-root {args.data_root}")
        return

    # --- verification ---
    if args.chunk is not None:
        if not (0 <= args.chunk < len(chunks)):
            raise SystemExit(f"--chunk {args.chunk} out of range 0..{len(chunks) - 1}")
        target = chunks[args.chunk]
        tscan = scans[args.chunk]
    elif ranked:
        target = ranked[0].chunk
        tscan = ranked[0]
    else:
        target = chunks[0]
        tscan = scans[0]

    print()
    verify(target, tscan, cfg, args.freq,
           max_checks=args.max_checks, conf_floor=args.conf_floor,
           max_gap_s=args.max_gap, device=args.device)


if __name__ == "__main__":
    main()
