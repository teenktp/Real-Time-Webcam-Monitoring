"""
precompute_embeddings.py
==========================
Run YOLO11s / MediaPipe pose / eye-gaze model ONCE per frame and cache the
fused embedding to disk, so train.py's default (frozen-branches) training
loop never has to run those heavy branch models again -- it just loads
cached [T, FUSED_FEAT_DIM] tensors straight off disk.

Why this exists: by default train.py freezes the branch models (they're not
being trained), yet the ORIGINAL loop still ran full YOLO + MediaPipe pose +
eye-gaze inference on every frame, every epoch -- recomputing the exact same
numbers 30 times. On CPU this was costing ~2.5 hours/epoch. Since the
branches are frozen, their output for a given frame never changes, so it
only needs to be computed once, ever.

NOTE: if you train with --finetune-branches, this cache does NOT apply --
fine-tuning needs live, differentiable branch outputs every step, not a
cached snapshot. train.py automatically falls back to the original
live-computation path when --finetune-branches is set.

Usage:
    python precompute_embeddings.py                      # cache config.DATASET_DIR
    python precompute_embeddings.py --data-dir Dataset_MIL
    python precompute_embeddings.py --force               # recompute + overwrite existing cache
"""

import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config
from utils.dataset import build_sequence_index
from utils.preprocessing import crop_head, bgr_to_model_tensor
from utils.cache import save_cached_embedding, is_cached
from models.eye_gaze_model import load_eye_gaze_model
from models.environment_model import load_environment_model
from models.body_pose_model import load_body_pose_model


def get_device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def collect_unique_frame_paths(data_dir):
    """Every sequence-building option (stride, seq_len) still just points at
    frames under data_dir -- so instead of caring about sequences at all
    here, just walk every sample build_sequence_index finds and dedupe the
    individual frame paths. This guarantees the cache covers whatever
    windowing train.py ends up using, including overlapping --stride runs."""
    samples = build_sequence_index(data_dir)
    unique_paths = set()
    for s in samples:
        unique_paths.update(s["frames"])
    return sorted(unique_paths)


@torch.no_grad()
def embed_one_frame(frame_path, eye_gaze_model, environment_model, body_pose_model, device):
    frame_bgr = cv2.imread(frame_path)
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read image at '{frame_path}'.")

    head_bgr = crop_head(frame_bgr)
    head_tensor = bgr_to_model_tensor(head_bgr)
    # bgr_to_model_tensor() returns a numpy.ndarray in this codebase (confirmed by
    # this exact bug: numpy arrays have no .to() method), not a torch tensor --
    # handle both just in case, same as _stack_frames() in utils/dataset.py.
    if not torch.is_tensor(head_tensor):
        head_tensor = torch.as_tensor(head_tensor)
    head_tensor = head_tensor.float().to(device)                     # [3,H,W]
    eye_gaze_emb = eye_gaze_model.embed(head_tensor.unsqueeze(0))    # [1, EYE_GAZE_FEAT_DIM]
    eye_gaze_emb = eye_gaze_emb.squeeze(0)                            # [EYE_GAZE_FEAT_DIM]

    env_emb = environment_model.embed_frame(frame_bgr)               # [ENV_FEAT_DIM]
    pose_emb = body_pose_model.embed_frame(frame_bgr)                # [POSE_FEAT_DIM]

    fused = torch.cat([
        eye_gaze_emb.to(device),
        env_emb.to(device),
        pose_emb.to(device),
    ], dim=-1)  # [FUSED_FEAT_DIM]
    return fused.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=config.DATASET_DIR,
                         help="Same folder you'll later pass to train.py --data-dir")
    parser.add_argument("--force", action="store_true",
                         help="Recompute and overwrite embeddings that are already cached "
                              "(use this if you retrain the branch models themselves, or "
                              "if config.FUSED_FEAT_DIM / branch weights changed)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    frame_paths = collect_unique_frame_paths(args.data_dir)
    print(f"Found {len(frame_paths)} unique frames under {args.data_dir}")

    if not args.force:
        frame_paths = [p for p in frame_paths if not is_cached(p, args.data_dir)]
        print(f"{len(frame_paths)} frames still need embedding (already-cached ones skipped; "
              f"use --force to recompute everything)")

    if not frame_paths:
        print("Nothing to do -- cache is already complete. Run train.py.")
        return

    eye_gaze_model = load_eye_gaze_model(device=device)
    environment_model = load_environment_model(device=device)
    body_pose_model = load_body_pose_model(device=device)
    eye_gaze_model.eval()
    environment_model.eval()
    body_pose_model.eval()

    n_ok, n_failed = 0, 0
    for frame_path in tqdm(frame_paths, desc="Precomputing embeddings", unit="frame"):
        try:
            emb = embed_one_frame(frame_path, eye_gaze_model, environment_model, body_pose_model, device)
            save_cached_embedding(frame_path, args.data_dir, emb)
            n_ok += 1
        except Exception as e:
            n_failed += 1
            tqdm.write(f"  [skip] {frame_path}: {e}")

    print(f"Done. Cached {n_ok} frames, {n_failed} failed, under {config.CACHE_DIR}")
    if n_failed:
        print("Some frames failed to embed -- re-run this script after fixing them "
              "(already-cached frames won't be recomputed unless you pass --force).")
    print("Now run: python train.py   (it will use this cache automatically)")


if __name__ == "__main__":
    main()
