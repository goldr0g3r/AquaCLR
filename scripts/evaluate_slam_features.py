"""Downstream Keypoint Stability Benchmark for LEGION-DeSnow.

Measures how the network's output affects the quality of SLAM feature
extraction by comparing keypoints and descriptors detected on:

  * **Raw snowy input**   ``I``     (DataLoader input, the degraded frame)
  * **Enhanced output**   ``Ĵ``    (LEGION-DeSnow prediction)
  * **Clean GT**          ``J_gt``  (oracle upper bound)

All three are extracted from the same underlying scene so differences
are entirely attributable to the image-quality change.

Automotive SiL parallel:
    This script is the subsea equivalent of a camera de-rain SiL
    quality gate: run ORB on raw rainy frames, on de-rained frames, on
    clean GT, compare keypoint yield and repeatability. The same
    harness is used in ADAS SiL pipelines to sign off sensor
    preprocessing blocks before integration.

Metrics
-------
kp_count
    Number of keypoints detected by the chosen feature extractor.
    More keypoints → richer SLAM map, better loop-closure.

repeatability
    Fraction of keypoints detected on ``I`` that are re-detected on
    ``Ĵ`` within ``--dist-thresh`` pixels. Range [0, 1], higher is
    better. Measures positional consistency: a perfect de-noiser that
    preserves scene edges would yield repeatability ≈ 1.

match_inlier_ratio
    After mutual-nearest-neighbour descriptor matching between ``I``
    and ``Ĵ`` with Lowe's ratio test, RANSAC-verified inlier ratio.
    Range [0, 1], higher means the two images share geometrically
    consistent descriptors — i.e. a SLAM tracker would reliably
    associate the two frames.

match_score
    Mean best-match descriptor distance (Hamming for ORB, L2 for
    SIFT). Lower is a more confident match.

Usage
-----
::

    uv run python scripts/evaluate_slam_features.py \\
        --ckpt outputs/<run>/ckpts/best.ckpt \\
        --data-root data/msrb \\
        --split test --task 1

    # SIFT detector (needs opencv-contrib, falls back to ORB):
    uv run python scripts/evaluate_slam_features.py \\
        --ckpt ... --data-root data/msrb --detector sift

    # Tweak feature budget and repeatability distance threshold:
    uv run python scripts/evaluate_slam_features.py \\
        --ckpt ... --data-root data/msrb --n-features 1500 --dist-thresh 5
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from aquaclr.data.msrb_dataset import MSRBDataset, _collate
from aquaclr.data.transforms import build_val_transform
from aquaclr.models import LEGIONDeSnowNet

logger = logging.getLogger("aquaclr.slam_bench")

# ---------------------------------------------------------------------------
# Feature extractor helpers
# ---------------------------------------------------------------------------

# Minimum number of keypoints needed to attempt homography estimation
_MIN_KP_FOR_RANSAC = 8


def _make_detector(name: str, n_features: int) -> cv2.Feature2D:
    """Instantiate an OpenCV feature detector/descriptor by name.

    ORB is the safe default (always available in opencv-python-headless).
    SIFT requires ``opencv-contrib-python``; we fall back to ORB if not
    present so the script never hard-fails.

    Args:
        name: ``"orb"`` or ``"sift"``.
        n_features: Maximum number of keypoints to detect.

    Returns:
        An OpenCV ``Feature2D`` instance.
    """
    if name == "sift":
        try:
            det = cv2.SIFT_create(nfeatures=n_features)  # type: ignore[attr-defined]
            return det
        except AttributeError:
            logger.warning(
                "cv2.SIFT_create not available (need opencv-contrib-python). "
                "Falling back to ORB."
            )
    return cv2.ORB_create(nfeatures=n_features)


def _is_orb(det: cv2.Feature2D) -> bool:
    return isinstance(det, cv2.ORB)


def _make_matcher(det: cv2.Feature2D) -> cv2.BFMatcher:
    """Create an appropriate BFMatcher for the given detector.

    ORB uses binary descriptors → Hamming distance.
    SIFT uses float32 descriptors → L2 norm.
    """
    norm = cv2.NORM_HAMMING if _is_orb(det) else cv2.NORM_L2
    return cv2.BFMatcher(norm, crossCheck=False)


def _tensor_to_uint8(t: torch.Tensor) -> NDArray[np.uint8]:
    """Convert a ``(C, H, W)`` float32 [0, 1] tensor to ``(H, W, C)`` uint8."""
    arr = t.cpu().float().permute(1, 2, 0).numpy()
    return (arr.clip(0.0, 1.0) * 255).astype(np.uint8)


def _to_gray(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _detect(
    det: cv2.Feature2D,
    gray: NDArray[np.uint8],
) -> tuple[list[cv2.KeyPoint], NDArray[np.uint8] | None]:
    """Detect keypoints and compute descriptors.

    Returns:
        Tuple of ``(keypoints, descriptors)``.  Descriptors is ``None``
        if no keypoints were found.
    """
    kps, descs = det.detectAndCompute(gray, None)
    return list(kps), descs  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------


@dataclass
class ImageResult:
    """Feature statistics for a single image triple (I, Ĵ, J_gt)."""

    n_kp_raw: int = 0
    n_kp_pred: int = 0
    n_kp_gt: int = 0
    repeatability: float = float("nan")
    match_inlier_ratio: float = float("nan")
    match_score: float = float("nan")


def _repeatability(
    kps_src: list[cv2.KeyPoint],
    kps_dst: list[cv2.KeyPoint],
    dist_thresh: float,
) -> float:
    """Compute keypoint repeatability: fraction of src kps re-detected in dst.

    For each keypoint in *src*, we check whether any keypoint in *dst*
    falls within ``dist_thresh`` pixels. The ratio of matched src kps to
    total src kps is the repeatability score.

    Args:
        kps_src: Keypoints from the source image.
        kps_dst: Keypoints from the destination image.
        dist_thresh: Pixel distance threshold for a kp to be considered
            "re-detected".

    Returns:
        Repeatability in ``[0, 1]``, or ``nan`` if src is empty.
    """
    if not kps_src or not kps_dst:
        return float("nan")

    src_pts = np.array([k.pt for k in kps_src], dtype=np.float32)
    dst_pts = np.array([k.pt for k in kps_dst], dtype=np.float32)

    # For each src point find the nearest dst point (brute force, small N)
    matched = 0
    for pt in src_pts:
        dists = np.linalg.norm(dst_pts - pt, axis=1)
        if dists.min() <= dist_thresh:
            matched += 1

    return matched / len(kps_src)


def _match_inlier_ratio_and_score(
    descs_src: NDArray[np.uint8] | None,
    descs_dst: NDArray[np.uint8] | None,
    kps_src: list[cv2.KeyPoint],
    kps_dst: list[cv2.KeyPoint],
    matcher: cv2.BFMatcher,
    lowe_ratio: float = 0.75,
) -> tuple[float, float]:
    """Compute match inlier ratio and mean match descriptor distance.

    Matches descriptors from *src* → *dst* using Lowe's ratio test,
    then estimates a homography with RANSAC to count geometric inliers.

    Args:
        descs_src: Descriptors from source (I).
        descs_dst: Descriptors from destination (Ĵ).
        kps_src: Corresponding keypoints.
        kps_dst: Corresponding keypoints.
        matcher: Pre-constructed BFMatcher.
        lowe_ratio: Lowe's ratio test threshold.

    Returns:
        ``(inlier_ratio, mean_score)``.  Both are ``nan`` if matching
        fails (too few keypoints).
    """
    if descs_src is None or descs_dst is None:
        return float("nan"), float("nan")
    if len(kps_src) < _MIN_KP_FOR_RANSAC or len(kps_dst) < _MIN_KP_FOR_RANSAC:
        return float("nan"), float("nan")

    # knnMatch with k=2 for Lowe's ratio test
    raw_matches = matcher.knnMatch(descs_src, descs_dst, k=2)
    good: list[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < lowe_ratio * n.distance:
                good.append(m)

    if len(good) < _MIN_KP_FOR_RANSAC:
        return float("nan"), float("nan")

    mean_score = float(np.mean([m.distance for m in good]))

    # RANSAC homography → inlier ratio
    src_pts = np.float32([kps_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return float("nan"), mean_score

    inlier_ratio = float(mask.ravel().sum()) / len(good)
    return inlier_ratio, mean_score


def _evaluate_image(
    i_gray: NDArray[np.uint8],
    pred_gray: NDArray[np.uint8],
    gt_gray: NDArray[np.uint8],
    det: cv2.Feature2D,
    matcher: cv2.BFMatcher,
    dist_thresh: float,
) -> ImageResult:
    """Run all feature metrics for a single (I, Ĵ, J_gt) triple."""
    kps_i, descs_i = _detect(det, i_gray)
    kps_pred, descs_pred = _detect(det, pred_gray)
    kps_gt, _ = _detect(det, gt_gray)

    rep = _repeatability(kps_i, kps_pred, dist_thresh)
    inlier_ratio, score = _match_inlier_ratio_and_score(
        descs_i, descs_pred, kps_i, kps_pred, matcher
    )

    return ImageResult(
        n_kp_raw=len(kps_i),
        n_kp_pred=len(kps_pred),
        n_kp_gt=len(kps_gt),
        repeatability=rep,
        match_inlier_ratio=inlier_ratio,
        match_score=score,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


@dataclass
class AggregateStats:
    """Running statistics accumulated over the dataset."""

    n_kp_raw: list[int] = field(default_factory=list)
    n_kp_pred: list[int] = field(default_factory=list)
    n_kp_gt: list[int] = field(default_factory=list)
    repeatability: list[float] = field(default_factory=list)
    match_inlier_ratio: list[float] = field(default_factory=list)
    match_score: list[float] = field(default_factory=list)

    def update(self, r: ImageResult) -> None:
        self.n_kp_raw.append(r.n_kp_raw)
        self.n_kp_pred.append(r.n_kp_pred)
        self.n_kp_gt.append(r.n_kp_gt)
        if not np.isnan(r.repeatability):
            self.repeatability.append(r.repeatability)
        if not np.isnan(r.match_inlier_ratio):
            self.match_inlier_ratio.append(r.match_inlier_ratio)
        if not np.isnan(r.match_score):
            self.match_score.append(r.match_score)

    @staticmethod
    def _fmt(vals: list[float | int], pct: bool = False) -> str:
        if not vals:
            return "  n/a"
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if pct:
            return f"{m * 100:6.1f} ± {s * 100:.1f}%"
        return f"{m:7.1f} ± {s:.1f}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True, help="Checkpoint (.ckpt) path.")
    p.add_argument("--data-root", type=Path, required=True, help="MSRB root directory.")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--task", type=int, default=1, choices=[1, 2])
    p.add_argument(
        "--image-size", type=int, default=384, help="Resize longer side to this."
    )
    p.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--detector",
        default="orb",
        choices=["orb", "sift"],
        help="Feature detector. 'sift' requires opencv-contrib-python, "
        "falls back to ORB automatically if unavailable.",
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=2000,
        help="Maximum keypoints the detector is allowed to return per image.",
    )
    p.add_argument(
        "--dist-thresh",
        type=float,
        default=3.0,
        help="Pixel distance threshold for counting a keypoint as 're-detected' "
        "when computing repeatability (I → Ĵ).",
    )
    p.add_argument(
        "--lowe-ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold for descriptor matching.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the SLAM feature stability benchmark."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s"
    )
    args = _parse_args()

    # ------------------------------------------------------------------
    # Build detector + matcher
    # ------------------------------------------------------------------
    det = _make_detector(args.detector, args.n_features)
    matcher = _make_matcher(det)
    det_name = type(det).__name__
    norm_name = "Hamming" if _is_orb(det) else "L2"
    logger.info("Detector: %s  nfeatures=%d", det_name, args.n_features)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state_dict = {
            k.removeprefix("net._orig_mod.").removeprefix("net."): v
            for k, v in state["state_dict"].items()
            if k.startswith("net.")
        }
    else:
        state_dict = state

    model = LEGIONDeSnowNet().to(args.device).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected[:5])

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
    ds = MSRBDataset(
        args.data_root,
        split=args.split,
        task=args.task,
        transform=build_val_transform(args.image_size),
    )
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    stats = AggregateStats()
    n_images = 0

    # ------------------------------------------------------------------
    # Inference + feature extraction loop
    # ------------------------------------------------------------------
    with torch.no_grad():
        for batch in loader:
            i_batch = batch["i"].to(args.device, non_blocking=True)
            j_batch = batch["j"].to(args.device, non_blocking=True)

            out = model(i_batch)
            pred_batch = out.j.clamp(0.0, 1.0)

            # Process each image in the batch on CPU (OpenCV)
            for idx in range(i_batch.size(0)):
                i_rgb = _tensor_to_uint8(i_batch[idx])
                pred_rgb = _tensor_to_uint8(pred_batch[idx])
                gt_rgb = _tensor_to_uint8(j_batch[idx])

                result = _evaluate_image(
                    _to_gray(i_rgb),
                    _to_gray(pred_rgb),
                    _to_gray(gt_rgb),
                    det=det,
                    matcher=matcher,
                    dist_thresh=args.dist_thresh,
                )
                stats.update(result)
                n_images += 1

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    _sep = "─" * 72

    print()
    print("=" * 72)
    print("  SLAM Feature Stability Benchmark — LEGION-DeSnow")
    print("=" * 72)
    print(f"  Detector       : {det_name}  (nfeatures={args.n_features})")
    print(f"  Descriptor norm: {norm_name}")
    print(f"  Dataset        : MSRB-{args.split}  task={args.task}  n={n_images}")
    print(f"  Dist thresh    : {args.dist_thresh} px  (repeatability)")
    print(f"  Lowe ratio     : {args.lowe_ratio}")
    print(_sep)

    # KP count table
    kp_raw_mean = statistics.mean(stats.n_kp_raw) if stats.n_kp_raw else 0.0
    kp_pred_mean = statistics.mean(stats.n_kp_pred) if stats.n_kp_pred else 0.0
    kp_gt_mean = statistics.mean(stats.n_kp_gt) if stats.n_kp_gt else 0.0
    delta_kp = (kp_pred_mean - kp_raw_mean) / max(kp_raw_mean, 1.0) * 100.0
    delta_kp_gt = (kp_gt_mean - kp_raw_mean) / max(kp_raw_mean, 1.0) * 100.0

    print(f"  {'Metric':<32}  {'Raw I':>10}  {'Enhanced Ĵ':>12}  {'Clean J_gt':>12}")
    print(_sep)
    print(
        f"  {'KP count (mean ± std)':<32}  "
        f"{stats._fmt(stats.n_kp_raw):>10}  "
        f"{stats._fmt(stats.n_kp_pred):>12}  "
        f"{stats._fmt(stats.n_kp_gt):>12}"
    )
    print(
        f"  {'KP count Δ vs Raw I':<32}  "
        f"{'—':>10}  "
        f"{delta_kp:>+10.1f}%  "
        f"{delta_kp_gt:>+10.1f}%  (oracle)"
    )
    print(_sep)

    # Repeatability
    rep_str = (
        stats._fmt(stats.repeatability, pct=True) if stats.repeatability else "  n/a"
    )
    print(f"  {'Repeatability  I → Ĵ':<32}  " f"{'—':>10}  {rep_str:>12}  {'—':>12}")

    # Match metrics
    mir_str = (
        stats._fmt(stats.match_inlier_ratio, pct=True)
        if stats.match_inlier_ratio
        else "  n/a"
    )
    ms_str = stats._fmt(stats.match_score) if stats.match_score else "  n/a"
    print(
        f"  {'Match inlier ratio  I → Ĵ':<32}  " f"{'—':>10}  {mir_str:>12}  {'—':>12}"
    )
    print(
        f"  {f'Match score ({norm_name}, I → Ĵ)':<32}  "
        f"{'—':>10}  {ms_str:>12}  {'—':>12}"
        "  (lower = better)"
    )
    print(_sep)

    # One-line SLAM verdict
    if kp_raw_mean > 0:
        verdict = (
            f"Enhanced Ĵ has {delta_kp:+.1f}% keypoints vs raw I "
            f"(oracle GT: {delta_kp_gt:+.1f}%)."
        )
        if stats.repeatability:
            verdict += (
                f"  Repeatability: {statistics.mean(stats.repeatability) * 100:.1f}%."
            )
        if stats.match_inlier_ratio:
            verdict += (
                f"  Match inlier ratio: "
                f"{statistics.mean(stats.match_inlier_ratio) * 100:.1f}%."
            )
        print(f"  Verdict: {verdict}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
