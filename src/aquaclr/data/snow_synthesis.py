"""On-the-fly marine snow synthesiser (fallback when MSRB is not on disk).

This is a *teaching* implementation that follows the spirit of the
MSRB synthesis pipeline (Sato et al., APSIPA 2023): random elliptical
particles drawn over a clean image with an additive blending op. It is
intentionally simple — for headline numbers always train on the
official MSRB pairs.

Automotive SiL parallel:
    Equivalent to a procedural rain-splat shader used to manufacture
    synthetic training data when no real measured rain dataset is
    available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def synthesize_marine_snow(
    image: NDArray[np.uint8],
    *,
    n_particles: tuple[int, int] = (100, 600),
    radius_px: tuple[int, int] = (1, 6),
    intensity: tuple[float, float] = (0.5, 1.0),
    seed: int | None = None,
) -> NDArray[np.uint8]:
    """Add synthetic marine snow particles to a clean RGB image.

    Args:
        image: Clean RGB image, shape ``(H, W, 3)``, uint8.
        n_particles: ``(min, max)`` count per image; sampled uniformly.
        radius_px: ``(min, max)`` particle radius in pixels (MSRB Task 1
            uses up to ~6 px; Task 2 mixes in larger up to ~32 px).
        intensity: ``(min, max)`` brightness of each particle in ``[0, 1]``.
        seed: Optional RNG seed.

    Returns:
        Snowy image, same shape and dtype as ``image``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        msg = f"synthesize_marine_snow expects an HWC RGB image, got shape {image.shape}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    out = image.astype(np.float32) / 255.0
    n = int(rng.integers(n_particles[0], n_particles[1] + 1))

    for _ in range(n):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(rng.integers(radius_px[0], radius_px[1] + 1))
        # Slight aspect-ratio jitter to mimic non-spherical particles.
        rx = max(1, int(r * float(rng.uniform(0.7, 1.3))))
        ry = max(1, int(r * float(rng.uniform(0.7, 1.3))))
        intensity_val = float(rng.uniform(*intensity))

        x0, x1 = max(cx - rx, 0), min(cx + rx + 1, w)
        y0, y1 = max(cy - ry, 0), min(cy + ry + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue

        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        norm_sq = ((xs - cx) / max(rx, 1)) ** 2 + ((ys - cy) / max(ry, 1)) ** 2
        mask = (norm_sq <= 1.0).astype(np.float32)
        # Soft falloff at the edge for anti-aliasing.
        soft = np.clip(1.0 - norm_sq, 0.0, 1.0) ** 0.5
        alpha = (mask * soft * intensity_val)[..., None]
        patch = out[y0:y1, x0:x1, :]
        out[y0:y1, x0:x1, :] = patch * (1.0 - alpha) + alpha

    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
