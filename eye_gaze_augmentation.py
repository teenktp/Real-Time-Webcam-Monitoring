"""
eye_gaze_augmentation.py
==========================
Data augmentation pipeline for eye-gaze classification on head-crop images.

DATASET PROBLEMS THIS ADDRESSES:
    - Most images are dark / low-light -> exposure & contrast correction
    - Dataset lacks diversity -> geometric augmentation (flip/translate/rotate)

DESIGN GOAL:
    The model must learn to focus sharply on the eyes. Every augmentation
    choice below is filtered through that lens.

HARD CONSTRAINTS (never violated by this pipeline):
    1. NO vertical flip -- flipping a face upside-down is not a realistic
       pose and would invert the semantic meaning of "looking up" vs
       "looking down" gaze direction. Only horizontal flip is used.
    2. NO blur of any kind (Gaussian/motion/median/etc.) -- blur destroys
       the fine iris/pupil/eyelid detail the eye-gaze model depends on.
       This pipeline contains zero blur transforms, intentionally.

Requires: pip install albumentations opencv-python
"""

import albumentations as A
import cv2
import numpy as np


def get_eye_gaze_train_augmentation():
    """
    Returns an albumentations.Compose pipeline for TRAINING data only.
    (Validation/inference should use get_eye_gaze_eval_transform() instead --
    no randomness there, so evaluation numbers stay comparable run to run.)

    NOTE: no resize step here on purpose -- resizing to the model's input
    size is handled separately in the training pipeline (e.g. via
    bgr_to_model_tensor() / config.HEAD_CROP_SIZE), so this only does the
    photometric + geometric augmentation and leaves the image at whatever
    size it was given.

    Order matters: exposure/contrast fixes run first (on the raw crop),
    then geometric transforms.
    """
    return A.Compose([
        # ------------------------------------------------------------------
        # 1) Exposure / contrast correction -- fixes the dataset's dominant
        #    problem (dark, low-light head crops).
        # ------------------------------------------------------------------
        # CLAHE (Contrast Limited Adaptive Histogram Equalization): boosts
        # local contrast in dark regions (like a shadowed eye socket)
        # without blowing out already-bright areas or introducing blur.
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),

        # Random brightness/contrast: teaches the model to be robust across
        # a wider range of exposures than what's actually in the dataset.
        A.RandomBrightnessContrast(
            brightness_limit=0.10,   # +/-10% brightness
            contrast_limit=0.2,     # +/-20% contrast
            p=0.7,
        ),

        # ------------------------------------------------------------------
        # 2) Geometric augmentation -- adds diversity, kept gaze-safe.
        #    (No blur anywhere in this pipeline -- intentional, see above.)
        # ------------------------------------------------------------------
        # Horizontal flip only. (No vertical flip -- see module docstring.)
        A.HorizontalFlip(p=0.5),

        # Small translation + small rotation (<=10 degrees), combined in one
        # Affine transform. No scaling/shear requested, so both are left at
        # their identity values.
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # small shift, both axes
            rotate=(-10, 10),   # rotation capped at +/-10 degrees, as required
            shear=0,
            scale=1.0,
            p=0.7,
            border_mode=cv2.BORDER_REFLECT_101,  # avoid harsh black borders after translate/rotate
        ),
    ])


def get_eye_gaze_eval_transform():
    """
    Deterministic transform for validation/inference: NO random
    augmentation, and no resize (that's handled separately downstream, same
    as in the training pipeline). This is intentionally a pass-through --
    it exists so calling code has a single consistent place to plug in an
    eval-time transform later without touching training code.
    """
    return A.Compose([])


def augment_image(image_bgr, augmentation_pipeline):
    """
    Convenience wrapper: applies an albumentations pipeline to a single
    BGR uint8 image (as read by cv2.imread) and returns the augmented BGR
    uint8 image. albumentations expects RGB internally for some transforms
    (e.g. CLAHE), so this handles the BGR<->RGB conversion for you.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    augmented = augmentation_pipeline(image=image_rgb)["image"]
    return cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    # Quick visual sanity check: run this file directly with an image path
    # to see a grid of augmented versions before trusting it in training.
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a sample head-crop image")
    parser.add_argument("--out-dir", default="augmentation_preview")
    parser.add_argument("--n", type=int, default=8, help="How many augmented previews to generate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img = cv2.imread(args.image)
    if img is None:
        raise IOError(f"Could not read image: {args.image}")

    pipeline = get_eye_gaze_train_augmentation()
    for i in range(args.n):
        out = augment_image(img, pipeline)
        out_path = os.path.join(args.out_dir, f"aug_{i:02d}.jpg")
        cv2.imwrite(out_path, out)
        print(f"Saved {out_path}")

    print(f"\nDone. Inspect the {args.n} images in '{args.out_dir}/' -- eyes should stay "
          f"sharp (no blur), never upside-down, rotation should look subtle (<=10°), and "
          f"the image size should be unchanged from the original (resize happens separately "
          f"in the training pipeline, not here).")
