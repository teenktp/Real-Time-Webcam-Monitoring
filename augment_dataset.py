"""
augment_dataset.py
====================
Runs eye_gaze_augmentation.py's pipeline over EVERY image in the dataset,
writing augmented copies to a separate mirrored directory -- never
overwrites the originals.

WHY A SEPARATE MIRRORED DIRECTORY (not in-place):
    utils/dataset.py's build_sequence_index() detects "contiguous runs" of
    frames by parsing the frame number out of each filename
    (subject1_frame0199.jpg -> 199) and grouping consecutive numbers into
    sequences. This script preserves the EXACT same relative path and
    filename for every augmented image, just under a different root folder
    -- so the augmented set has IDENTICAL contiguous-run structure to the
    original, and can be used as its own valid --data-dir (or combined with
    the original later) without corrupting that frame-number logic.
    Overwriting in place would also destroy the original data irreversibly,
    which this avoids entirely.

Usage:
    python augment_dataset.py                          # Dataset/ -> Dataset_augmented/
    python augment_dataset.py --input-dir Dataset_MIL   # augments the MIL set instead
    python augment_dataset.py --copies 2                # 2 augmented variants per image (see caveat below)
    python augment_dataset.py --force                   # regenerate even already-done files

CAVEAT on --copies > 1: multiple variants per original frame are saved as
    <original_name>_aug0.jpg, _aug1.jpg, ... in the SAME output subfolder.
    These extra files do NOT match the frame-number regex used for
    contiguous-run detection, so build_sequence_index() will simply ignore
    them (they won't form sequences on their own). --copies > 1 is only
    useful today if you separately adapt your training/labeling pipeline to
    read them as standalone images (e.g. for a non-sequence eye-gaze
    classifier). The default --copies 1 is the one guaranteed to slot
    straight into the existing sequence-based pipeline.
"""

import argparse
import os

import cv2
from tqdm import tqdm

import config
from eye_gaze_augmentation import get_eye_gaze_train_augmentation, augment_image

VALID_EXTS = (".jpg", ".jpeg", ".png")


def find_all_images(root_dir):
    """Walk root_dir/subjectX/<class>/*.jpg the same way utils/dataset.py does."""
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(dirpath, fname))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=config.DATASET_DIR,
                         help="Dataset root to read from (default: config.DATASET_DIR)")
    parser.add_argument("--output-dir", default=config.DATASET_AUGMENTED_DIR,
                         help="Where to write augmented copies. Default: "
                              "<input-dir>_augmented, next to the input dir.")
    parser.add_argument("--copies", type=int, default=1,
                         help="Augmented variants per original image (default: 1 -- see the "
                              "module docstring's caveat before using > 1)")
    parser.add_argument("--force", action="store_true",
                         help="Regenerate files that already exist in output-dir "
                              "(default: skip them, so this script is resumable)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or (input_dir.rstrip("/\\") + "_augmented")
    output_dir = os.path.abspath(output_dir)

    if output_dir == input_dir:
        raise SystemExit("--output-dir must be different from --input-dir "
                          "(refusing to risk overwriting your original dataset).")

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    image_paths = find_all_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No images found under {input_dir}. Check the path.")
    print(f"Found {len(image_paths)} images.")

    pipeline = get_eye_gaze_train_augmentation()

    n_done, n_skipped, n_failed = 0, 0, 0
    for src_path in tqdm(image_paths, desc="Augmenting", unit="img"):
        rel_path = os.path.relpath(src_path, input_dir)
        rel_dir, fname = os.path.split(rel_path)
        name, ext = os.path.splitext(fname)
        out_subdir = os.path.join(output_dir, rel_dir)

        # Decide the output filename(s) for this source image up front, so
        # we can skip entirely (without even reading the image) if they
        # already exist and --force wasn't passed -- keeps re-runs fast.
        if args.copies == 1:
            out_names = [fname]  # identical filename -- preserves frame-number contiguity
        else:
            out_names = [f"{name}_aug{i}{ext}" for i in range(args.copies)]

        out_paths = [os.path.join(out_subdir, n) for n in out_names]
        if not args.force and all(os.path.isfile(p) for p in out_paths):
            n_skipped += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            n_failed += 1
            tqdm.write(f"  [skip] could not read: {src_path}")
            continue

        os.makedirs(out_subdir, exist_ok=True)
        for out_path in out_paths:
            augmented = augment_image(img, pipeline)
            cv2.imwrite(out_path, augmented)
        n_done += 1

    print(f"\nDone. {n_done} images augmented, {n_skipped} already existed (skipped), "
          f"{n_failed} failed to read.")
    print(f"Augmented dataset ready at: {output_dir}")
    print(f"Train on it directly with:  python train.py --data-dir {output_dir}")
    print("(Or combine it with the original Dataset/ -- ask if you'd like help wiring "
          "train.py/precompute_embeddings.py to accept multiple --data-dir roots at once.)")


if __name__ == "__main__":
    main()
