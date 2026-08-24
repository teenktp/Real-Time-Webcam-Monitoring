"""
utils/cache.py
================
Disk cache for precomputed per-frame branch embeddings (eye-gaze + environment
+ pose, already concatenated into one FUSED_FEAT_DIM vector per frame).

Why per-FRAME (not per-sequence): the same frame can appear in multiple
overlapping sequences when --stride < seq_len is used to oversample the
minority class. Caching per-frame means each frame's expensive branch
inference (YOLO / MediaPipe / eye-gaze) runs exactly once no matter how many
sequences reuse it.

Cache layout mirrors the dataset's own folder structure so it's easy to
inspect/clear by subject or class:

    config.CACHE_DIR/<data_dir_basename>/subjectX/<class>/<frame>.npy

<data_dir_basename> keeps caches for Dataset/ and Dataset_MIL/ separate even
though both have subjectX/<class>/frame.jpg internally.
"""

import os

import numpy as np

import config


def cache_path_for_frame(frame_path, data_dir):
    """Map an original frame path to where its cached embedding lives."""
    frame_path = os.path.abspath(frame_path)
    data_dir = os.path.abspath(data_dir)
    rel = os.path.relpath(frame_path, data_dir)
    tag = os.path.basename(os.path.normpath(data_dir))  # e.g. "Dataset" or "Dataset_MIL"
    out_path = os.path.join(config.CACHE_DIR, tag, rel)
    return os.path.splitext(out_path)[0] + ".npy"


def load_cached_embedding(frame_path, data_dir):
    """Returns a float32 np.ndarray [FUSED_FEAT_DIM] or None if not cached yet."""
    path = cache_path_for_frame(frame_path, data_dir)
    if not os.path.isfile(path):
        return None
    try:
        return np.load(path)
    except (EOFError, ValueError, OSError):
        # Corrupted/partial write (e.g. process was killed mid-write) --
        # treat as a cache miss so the caller can recompute it.
        return None


def save_cached_embedding(frame_path, data_dir, embedding):
    """embedding: np.ndarray or torch tensor, shape [FUSED_FEAT_DIM]."""
    path = cache_path_for_frame(frame_path, data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.asarray(embedding, dtype=np.float32)
    # Write to a temp file then rename -- atomic on POSIX and Windows, so a
    # process interrupted mid-write never leaves a corrupt .npy in place for
    # a later run to trip over.
    tmp_path = path + f".tmp{os.getpid()}"
    np.save(tmp_path, arr)
    # np.save appends .npy again if the name doesn't already end with it
    tmp_saved = tmp_path if tmp_path.endswith(".npy") else tmp_path + ".npy"
    os.replace(tmp_saved, path)


def is_cached(frame_path, data_dir):
    return os.path.isfile(cache_path_for_frame(frame_path, data_dir))
