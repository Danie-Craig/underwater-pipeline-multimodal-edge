"""SubPipe data access (§3).

Two responsibilities:

  1. **Training prep** — turn the raw SubPipe / SubPipeMini layout into the
     Ultralytics dataset YAMLs the YOLO trainers expect (one for RGB-seg, one
     for sonar-detect). Done once in roadmap step 2.

  2. **Sequence playback** — yield time-synchronized :class:`Frame` tuples
     (rgb image, sonar image, INS pose) for ``run_inference`` and the fusion /
     robustness harnesses.

On-disk layout (SubPipe ``DATA/ChunkN/``), per the dataset README:

    ChunkN/
      Cam0_images/<ts>.jpg          RGB (GoPro Hero 10, 2704x1520)
      Segmentation/<ts>_label.png   pipe segmentation masks (+ <ts>.png)
      SSS_HF_images/Image/<ts>.pbm  side-scan sonar, high-freq (5000x500)
      SSS_HF_images/YOLO_Annotation/<ts>.txt   YOLO detection labels
      SSS_LF_images/...             low-freq variant (2500x500)
      EstimatedState.csv            INS/DVL 6-DOF pose (LSTS/IMC)
      Acceleration.csv, Depth.csv, ...   other onboard sensors

Chunk directories are *discovered* (we walk for these marker names) rather than
hard-coded, so the exact unzip layout of SubPipe / SubPipeMini / SubPipeMini2
does not matter. Note SubPipeMini carries the segmentation+camera data while
SubPipeMini2 carries the sonar+detection data, so ``data.rgb_root`` and
``data.sonar_root`` may point at different download roots.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src import REPO_ROOT
from src.fusion.late_fusion import Pose

# ---------------------------------------------------------------------------
#  On-disk layout markers (auto-discovered, not hard-coded).
# ---------------------------------------------------------------------------
RGB_DIR = "Cam0_images"
SEG_DIR = "Segmentation"
SEG_LABEL_SUFFIX = "_label"
NAV_FILE = "EstimatedState.csv"
SONAR_DIRS = {"HF": "SSS_HF_images", "LF": "SSS_LF_images"}
SONAR_IMG_SUBDIR = "Image"
SONAR_YOLO_SUBDIR = "YOLO_Annotation"

# Extensions Ultralytics reads natively. Sonar .pbm is NOT in this set, so the
# converter rewrites sonar frames to .png (at native resolution, so the
# already-normalized YOLO boxes stay valid).
ULTRALYTICS_IMG_EXTS = (".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp")


@dataclass
class Frame:
    """One synchronized capture along the trajectory."""

    t: float                      # timestamp (s)
    rgb: np.ndarray | None        # BGR image, or None if no RGB at this stamp
    sonar: np.ndarray | None      # sonar intensity image, or None
    pose: Pose                    # INS pose at this stamp


# ===========================================================================
#  Shared helpers
# ===========================================================================
def _parse_timestamp(stem: str) -> float:
    """Best-effort parse of a SubPipe filename stem into a float timestamp.

    Bare timestamps (``1623241234.567``) parse directly. Prepared frames carry a
    sensor/chunk prefix (``c0_1623241234.567``); for those we take the *longest*
    numeric run, which is the timestamp, rather than the leading ``0`` in ``c0``.
    Returns NaN when no number is present so callers can fall back to enumeration.
    """
    try:
        return float(stem)
    except ValueError:
        pass
    nums = re.findall(r"\d+\.\d+|\d+", stem)
    if not nums:
        return float("nan")
    try:
        return float(max(nums, key=len))
    except ValueError:
        return float("nan")


def discover_chunks(root: str | Path) -> list[Path]:
    """All SubPipe chunk directories under ``root``.

    A chunk is identified by an ``EstimatedState.csv``; if none are found we
    fall back to any directory that holds a known modality folder.
    """
    root = Path(root)
    if not root.exists():
        return []
    chunks: set[Path] = {p.parent for p in root.rglob(NAV_FILE)}
    if not chunks:
        for marker in (RGB_DIR, SEG_DIR, *SONAR_DIRS.values()):
            chunks.update(p.parent for p in root.rglob(marker))
    return sorted(chunks)


def _sorted_by_timestamp(paths: Sequence[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: _parse_timestamp(p.stem))


def _nearest(sorted_ts: np.ndarray, t: float) -> int:
    """Index into ``sorted_ts`` of the value closest to ``t`` (ts ascending)."""
    i = int(np.searchsorted(sorted_ts, t))
    if i <= 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    return i if (sorted_ts[i] - t) < (t - sorted_ts[i - 1]) else i - 1


# ===========================================================================
#  1. INS trajectory
# ===========================================================================
# CSV column resolution. EstimatedState exports (LSTS/IMC) give NED x/y/z in
# metres and roll/pitch/yaw in radians; exact header spelling varies, so we
# resolve defensively: exact match first, then a guarded substring match.
_TS_EXACT = ("timestamp (seconds)", "timestamp", "time", "epoch")
_X_EXACT = ("x (m)", "x", "x_m", "north (m)", "north")
_Y_EXACT = ("y (m)", "y", "y_m", "east (m)", "east")
_YAW_EXACT = ("psi (rad)", "psi", "yaw (rad)", "yaw", "heading (rad)", "heading")
_LAT_SUB = ("lat",)
_LON_SUB = ("lon",)


def _resolve_column(cols: dict[str, str], exact: Sequence[str],
                    substr: Sequence[str] = ()) -> str | None:
    for cand in exact:
        if cand in cols:
            return cols[cand]
    for cand in substr:
        for low, orig in cols.items():
            if cand in low:
                return orig
    return None


def load_trajectory(sequence_dir: str | Path, *, verbose: bool = False) -> list[Pose]:
    """Load the INS-derived AUV trajectory for one sequence.

    Parses ``EstimatedState.csv`` into time-ordered :class:`Pose` records and
    computes cumulative ``along_track`` distance from the horizontal (x, y)
    path. Falls back to lat/lon (converted to local metres) if x/y are absent,
    and to a zero trajectory if no nav log is present (with a warning).
    """
    import pandas as pd

    sequence_dir = Path(sequence_dir)
    nav_path = sequence_dir / NAV_FILE
    if not nav_path.exists():
        found = list(sequence_dir.rglob(NAV_FILE))
        nav_path = found[0] if found else None
    if nav_path is None:
        if verbose:
            print(f"  [trajectory] no {NAV_FILE} under {sequence_dir} — zero trajectory")
        return []

    df = pd.read_csv(nav_path)
    cols = {c.lower().strip(): c for c in df.columns}
    ts_col = _resolve_column(cols, _TS_EXACT) or df.columns[0]
    x_col = _resolve_column(cols, _X_EXACT)
    y_col = _resolve_column(cols, _Y_EXACT)
    yaw_col = _resolve_column(cols, _YAW_EXACT)
    lat_col = _resolve_column(cols, (), _LAT_SUB)
    lon_col = _resolve_column(cols, (), _LON_SUB)

    n = len(df)
    t = df[ts_col].to_numpy(dtype=float)

    if x_col and y_col:
        x = df[x_col].to_numpy(dtype=float)
        y = df[y_col].to_numpy(dtype=float)
        src = f"x={x_col!r}, y={y_col!r}"
    elif lat_col and lon_col:
        lat = df[lat_col].to_numpy(dtype=float)
        lon = df[lon_col].to_numpy(dtype=float)
        if np.nanmax(np.abs(lat)) > np.pi:          # looks like degrees
            lat, lon = np.radians(lat), np.radians(lon)
        R = 6_371_000.0
        x = R * (lat - lat[0])
        y = R * (lon - lon[0]) * np.cos(lat[0])
        src = f"lat={lat_col!r}, lon={lon_col!r} (→ local metres)"
    else:
        x = np.zeros(n)
        y = np.zeros(n)
        src = "no position columns found — x=y=0"

    yaw = df[yaw_col].to_numpy(dtype=float) if yaw_col else np.zeros(n)
    step = np.hypot(np.diff(x, prepend=x[:1]), np.diff(y, prepend=y[:1]))
    along = np.cumsum(step)

    if verbose:
        print(f"  [trajectory] {nav_path.name}: {n} rows | ts={ts_col!r} | {src} | "
              f"yaw={'<zeros>' if not yaw_col else repr(yaw_col)} | "
              f"track length {along[-1]:.1f} m")

    return [
        Pose(t=float(t[i]), x=float(x[i]), y=float(y[i]),
             heading=float(yaw[i]), along_track=float(along[i]))
        for i in range(n)
    ]


# ===========================================================================
#  2. Sequence playback
# ===========================================================================
class SequenceLoader:
    """Iterate one SubPipe sequence as time-synchronized :class:`Frame`s.

    RGB and sonar are captured on independent clocks; this loader builds a
    merged timeline from whichever modalities are present and, for each tick,
    attaches the nearest RGB and nearest sonar frame *within a tolerance*
    (either may be ``None`` — exactly the dropout the fusion rides through)
    plus the nearest INS pose. Images are read lazily to keep memory flat over
    a 7-9 minute run.
    """

    def __init__(self, sequence_dir: str | Path, config: dict) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.config = config
        prep = config.get("data", {}).get("prep", {})
        self.sonar_freq = str(prep.get("sonar_freq", "HF")).upper()
        self.sync_tol = float(config.get("data", {}).get("sync_tolerance_s", 0.5))

        # Index RGB frames.
        rgb_dir = self.sequence_dir / RGB_DIR
        self._rgb_paths = _sorted_by_timestamp(
            [p for p in rgb_dir.glob("*") if p.suffix.lower() in ULTRALYTICS_IMG_EXTS]
        ) if rgb_dir.is_dir() else []
        self._rgb_ts = np.array([_parse_timestamp(p.stem) for p in self._rgb_paths], dtype=float)

        # Index sonar frames (native .pbm read at playback time).
        sonar_img_dir = self.sequence_dir / SONAR_DIRS.get(self.sonar_freq, "") / SONAR_IMG_SUBDIR
        self._sonar_paths = _sorted_by_timestamp(list(sonar_img_dir.glob("*"))) if sonar_img_dir.is_dir() else []
        self._sonar_ts = np.array([_parse_timestamp(p.stem) for p in self._sonar_paths], dtype=float)

        # Trajectory, indexed for nearest-pose lookup.
        self._poses = load_trajectory(self.sequence_dir)
        self._pose_ts = np.array([p.t for p in self._poses], dtype=float)

        # Merged, de-duplicated event timeline (sorted union of modality stamps).
        stamps = np.concatenate([
            self._rgb_ts[np.isfinite(self._rgb_ts)],
            self._sonar_ts[np.isfinite(self._sonar_ts)],
        ]) if (len(self._rgb_ts) or len(self._sonar_ts)) else np.array([], dtype=float)
        self._timeline = np.unique(stamps)

    def _pose_at(self, t: float) -> Pose:
        if len(self._poses) == 0:
            return Pose(t=t, x=0.0, y=0.0, heading=0.0, along_track=0.0)
        return self._poses[_nearest(self._pose_ts, t)]

    def _img_at(self, t: float, paths, ts) -> "np.ndarray | None":
        if len(paths) == 0:
            return None
        i = _nearest(ts, t)
        if abs(ts[i] - t) > self.sync_tol:
            return None
        import cv2
        img = cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)
        return img

    def __iter__(self) -> Iterator[Frame]:
        import cv2  # noqa: F401  (ensure available before iterating)
        for t in self._timeline:
            yield Frame(
                t=float(t),
                rgb=self._img_at(t, self._rgb_paths, self._rgb_ts),
                sonar=self._img_at(t, self._sonar_paths, self._sonar_ts),
                pose=self._pose_at(t),
            )

    def __len__(self) -> int:
        return int(len(self._timeline))


# ===========================================================================
#  3. Training prep — SubPipe → Ultralytics
# ===========================================================================
def mask_to_yolo_polygons(label_mask: np.ndarray, *, min_area_px: float = 50.0,
                          epsilon_frac: float = 0.002) -> list[np.ndarray]:
    """Convert a binary/indexed pipe mask into normalized YOLO-seg polygons.

    Any nonzero pixel is treated as the single ``pipe`` class (class 0). Each
    external contour above ``min_area_px`` becomes one polygon, simplified with
    Douglas-Peucker and normalized to ``[0, 1]``. Returns a list of ``(N, 2)``
    float arrays (x, y).
    """
    import cv2

    if label_mask.ndim == 3:
        label_mask = cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY)
    h, w = label_mask.shape[:2]
    binary = (label_mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon_frac * cv2.arcLength(cnt, True), True)
        approx = approx.reshape(-1, 2).astype(np.float32)
        if len(approx) < 3:
            continue
        approx[:, 0] = np.clip(approx[:, 0] / w, 0.0, 1.0)
        approx[:, 1] = np.clip(approx[:, 1] / h, 0.0, 1.0)
        polygons.append(approx)
    return polygons


def _split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    import random
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    return idx[n_val:], idx[:n_val]          # train, val


def _write_yaml(path: Path, root: Path, names: dict[int, str]) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"path": str(root.resolve()), "train": "images/train",
             "val": "images/val", "names": names},
            fh, sort_keys=False,
        )


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _prepare_rgb_seg(rgb_root: Path, out_dir: Path, val_fraction: float, seed: int,
                     prep: dict, verbose: bool) -> Path:
    import cv2

    min_area = float(prep.get("min_polygon_area_px", 50.0))
    copy_images = bool(prep.get("copy_images", False))

    chunks = [c for c in discover_chunks(rgb_root) if (c / SEG_DIR).is_dir()]
    samples: list[tuple[Path, Path]] = []        # (image_path, label_mask_path)
    for ci, chunk in enumerate(chunks):
        seg_dir = chunk / SEG_DIR
        for mask_path in sorted(seg_dir.glob(f"*{SEG_LABEL_SUFFIX}.png")):
            ts = mask_path.stem[: -len(SEG_LABEL_SUFFIX)]
            img = chunk / RGB_DIR / f"{ts}.jpg"
            if not img.exists():
                alt = seg_dir / f"{ts}.png"      # fall back to the seg-folder copy
                if not alt.exists():
                    continue
                img = alt
            samples.append((img, mask_path))

    if not samples:
        if verbose:
            print(f"  [rgb-seg] no (image, mask) pairs found under {rgb_root} — skipped")
        return out_dir / "rgb_seg.yaml"

    train, val = _split_indices(len(samples), val_fraction, seed)
    counts = {"train": 0, "val": 0}
    polys_total = 0
    for split, ids in (("train", train), ("val", val)):
        for k, i in enumerate(ids):
            img_path, mask_path = samples[i]
            stem = f"c{_chunk_index(chunks, img_path)}_{img_path.stem}"
            dst_img = out_dir / "images" / split / f"{stem}{img_path.suffix.lower()}"
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
            _link_or_copy(img_path, dst_img, copy_images)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            polys = mask_to_yolo_polygons(mask, min_area_px=min_area) if mask is not None else []
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            with dst_lbl.open("w", encoding="utf-8") as fh:
                for poly in polys:
                    coords = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
                    fh.write(f"0 {coords}\n")
            polys_total += len(polys)
            counts[split] += 1

    _write_yaml(out_dir / "rgb_seg.yaml", out_dir, {0: "pipe"})
    if verbose:
        print(f"  [rgb-seg] {len(chunks)} chunk(s) | {counts['train']} train + "
              f"{counts['val']} val images | {polys_total} pipe polygons → {out_dir}")
    return out_dir / "rgb_seg.yaml"


def _prepare_sonar_det(sonar_root: Path, out_dir: Path, val_fraction: float, seed: int,
                       sonar_freq: str, prep: dict, verbose: bool) -> Path:
    import cv2

    copy_images = bool(prep.get("copy_images", False))
    sonar_dirname = SONAR_DIRS.get(sonar_freq.upper(), SONAR_DIRS["HF"])

    chunks = [c for c in discover_chunks(sonar_root)
              if (c / sonar_dirname / SONAR_IMG_SUBDIR).is_dir()]
    samples: list[tuple[Path, Path | None, int]] = []   # (image, label_or_None, chunk_idx)
    for ci, chunk in enumerate(chunks):
        img_dir = chunk / sonar_dirname / SONAR_IMG_SUBDIR
        lbl_dir = chunk / sonar_dirname / SONAR_YOLO_SUBDIR
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in (".pbm", *ULTRALYTICS_IMG_EXTS):
                continue
            lbl = lbl_dir / f"{img_path.stem}.txt"
            samples.append((img_path, lbl if lbl.exists() else None, ci))

    if not samples:
        if verbose:
            print(f"  [sonar-det] no sonar frames found under {sonar_root} — skipped")
        return out_dir / "sonar_det.yaml"

    train, val = _split_indices(len(samples), val_fraction, seed)
    counts = {"train": 0, "val": 0}
    boxes_total = 0
    for split, ids in (("train", train), ("val", val)):
        for i in ids:
            img_path, lbl_path, ci = samples[i]
            stem = f"c{ci}_{img_path.stem}"
            dst_img = out_dir / "images" / split / f"{stem}.png"
            dst_lbl = out_dir / "labels" / split / f"{stem}.txt"

            if img_path.suffix.lower() == ".pbm":
                # Re-encode to .png at NATIVE resolution (normalized boxes stay valid).
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                if img is not None:
                    cv2.imwrite(str(dst_img), img)
            else:
                _link_or_copy(img_path, dst_img, copy_images)

            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if lbl_path is not None:
                text = lbl_path.read_text(encoding="utf-8")
                boxes_total += sum(1 for ln in text.splitlines() if ln.strip())
                dst_lbl.write_text(text, encoding="utf-8")
            else:
                dst_lbl.write_text("", encoding="utf-8")     # explicit negative
            counts[split] += 1

    _write_yaml(out_dir / "sonar_det.yaml", out_dir, {0: "pipe"})
    if verbose:
        print(f"  [sonar-det] {sonar_dirname} | {len(chunks)} chunk(s) | "
              f"{counts['train']} train + {counts['val']} val images | "
              f"{boxes_total} boxes → {out_dir}")
    return out_dir / "sonar_det.yaml"


def _chunk_index(chunks: Sequence[Path], path: Path) -> int:
    for i, c in enumerate(chunks):
        try:
            path.relative_to(c)
            return i
        except ValueError:
            continue
    return 0


def prepare_ultralytics_datasets(config: dict, *, verbose: bool = True) -> tuple[Path, Path]:
    """Generate the RGB-seg and sonar-detect dataset YAMLs from raw SubPipe.

    Returns the two YAML paths (also the values stored in the config under
    ``data.rgb_dataset_yaml`` / ``data.sonar_dataset_yaml``).
    """
    data = config["data"]
    prep = data.get("prep", {})
    base = REPO_ROOT / data["root"]
    rgb_root = REPO_ROOT / data.get("rgb_root", data["root"])
    sonar_root = REPO_ROOT / data.get("sonar_root", data["root"])
    out_base = REPO_ROOT / prep.get("out_dir", str(base / "yolo"))
    val_fraction = float(prep.get("val_fraction", 0.2))
    seed = int(prep.get("seed", 42))
    sonar_freq = str(prep.get("sonar_freq", "HF")).upper()

    if verbose:
        print(f"Preparing Ultralytics datasets → {out_base}")
        print(f"  rgb_root  = {rgb_root}")
        print(f"  sonar_root= {sonar_root}")

    rgb_yaml = _prepare_rgb_seg(rgb_root, out_base / "rgb_seg", val_fraction, seed, prep, verbose)
    sonar_yaml = _prepare_sonar_det(sonar_root, out_base / "sonar_det", val_fraction, seed,
                                    sonar_freq, prep, verbose)

    # Point the config's recorded YAML paths at what we just built (best-effort).
    cfg_rgb = REPO_ROOT / data.get("rgb_dataset_yaml", "")
    cfg_sonar = REPO_ROOT / data.get("sonar_dataset_yaml", "")
    for built, recorded in ((rgb_yaml, cfg_rgb), (sonar_yaml, cfg_sonar)):
        if recorded and recorded.name and recorded.resolve() != built.resolve():
            try:
                recorded.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built, recorded)
            except OSError:
                pass
    return rgb_yaml, sonar_yaml


# ===========================================================================
#  4. Annotation verification  (step-2 acceptance check)
# ===========================================================================
def _draw_yolo_seg(img: np.ndarray, label_file: Path) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    out = img.copy()
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
    return out


def _draw_yolo_boxes(img: np.ndarray, label_file: Path) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    out = img.copy()
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = (float(x) for x in parts[:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
    return out


def _summarize_split(images_dir: Path, labels_dir: Path) -> dict:
    n_img = sum(1 for p in images_dir.glob("*") if p.suffix.lower() in ULTRALYTICS_IMG_EXTS) \
        if images_dir.is_dir() else 0
    n_inst = n_empty = 0
    if labels_dir.is_dir():
        for lbl in labels_dir.glob("*.txt"):
            lines = [ln for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            n_inst += len(lines)
            n_empty += (len(lines) == 0)
    return {"images": n_img, "instances": n_inst, "empty_labels": n_empty}


def verify_annotations(config: dict, n: int = 8, *, save_dir: str | Path | None = None) -> None:
    """Sanity-check the built YOLO datasets (the step-2 acceptance check).

    Reports per-split image / instance / empty-label counts for both datasets
    and writes ``n`` overlaid samples each (seg polygons, sonar boxes) so the
    alignment can be eyeballed.
    """
    import cv2
    import yaml

    data = config["data"]
    prep = data.get("prep", {})
    out_base = REPO_ROOT / prep.get("out_dir", str(REPO_ROOT / data["root"] / "yolo"))
    save_dir = Path(save_dir) if save_dir else (out_base / "_verify")
    save_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("RGB segmentation", out_base / "rgb_seg", _draw_yolo_seg),
        ("Sonar detection", out_base / "sonar_det", _draw_yolo_boxes),
    ]
    print("=" * 64)
    print(" annotation verification")
    print("=" * 64)
    for title, root, drawer in specs:
        yaml_path = root / (root.name + ".yaml")
        print(f"\n{title}  ({yaml_path})")
        if not yaml_path.exists():
            print("  dataset YAML missing — was prepare_data run for this modality?")
            continue
        names = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("names", {})
        print(f"  classes: {names}")
        for split in ("train", "val"):
            s = _summarize_split(root / "images" / split, root / "labels" / split)
            print(f"  {split:5s}: {s['images']:5d} images | {s['instances']:6d} instances | "
                  f"{s['empty_labels']:5d} empty")

        # Overlay a few train samples that actually have labels.
        img_dir, lbl_dir = root / "images" / "train", root / "labels" / "train"
        if not img_dir.is_dir():
            continue
        shown = 0
        for img_path in sorted(img_dir.glob("*")):
            if shown >= n:
                break
            lbl = lbl_dir / f"{img_path.stem}.txt"
            if not lbl.exists() or not lbl.read_text(encoding="utf-8").strip():
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out = drawer(img, lbl)
            dst = save_dir / f"{root.name}_{shown:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(dst), out)
            shown += 1
        print(f"  wrote {shown} overlay sample(s) → {save_dir}")
    print(f"\nEyeball the overlays in {save_dir} to confirm masks/boxes line up.")
