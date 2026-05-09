"""Image augmentation transforms for AquaCLR.

We use Albumentations because it lets us apply the same geometric
transform to ``(I, J, t_gt)`` triples atomically, which is essential
for paired training. Photometric augmentations are applied **only to
``I``** so that ``J`` remains a clean radiance reference.

Automotive SiL parallel:
    Strong photometric jitter on the noisy input while keeping the
    target reference clean is the same recipe used in ADAS de-rain
    training, where weather augmentations are applied to the input
    camera frame but never to the ground-truth.
"""

from __future__ import annotations

from typing import Any

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional at test time
    _ALBUMENTATIONS_AVAILABLE = False
    A = None  # type: ignore[assignment]
    ToTensorV2 = None  # type: ignore[assignment]


def _require_albumentations() -> None:
    if not _ALBUMENTATIONS_AVAILABLE:
        msg = (
            "albumentations is required for AquaCLR data transforms. "
            "Install with: pip install albumentations"
        )
        raise ImportError(msg)


def build_train_transform(image_size: int = 256) -> Any:
    """Build the training transform pipeline.

    Returns an Albumentations ``Compose`` that:

    * random-crops both inputs to ``image_size``,
    * applies horizontal flips,
    * applies mild photometric jitter to the noisy input (additional
      target ``"image_clean"`` is left untouched), and
    * converts to ``torch.Tensor`` in CHW float32 ``[0, 1]``.
    """
    _require_albumentations()
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                        p=1.0,
                    ),
                    A.GaussNoise(var_limit=(2.0, 10.0), p=1.0),
                ],
                p=0.3,
            ),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        additional_targets={"image_clean": "image", "transmission": "mask"},
    )


def build_val_transform(image_size: int = 256) -> Any:
    """Build the validation transform pipeline (deterministic centre crop)."""
    _require_albumentations()
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size, p=1.0),
            A.CenterCrop(height=image_size, width=image_size, p=1.0),
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        additional_targets={"image_clean": "image", "transmission": "mask"},
    )
