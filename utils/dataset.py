"""
utils/dataset.py
=================
Loads data laid out exactly like the screenshot / your Dataset.zip:

    Dataset/
        subject1/
            cheat/          subject1_frame0199.jpg ...
            Non_cheat/      subject1_frame0000.jpg ...
        subject2/
            cheat/          subject2_frame0224.jpg ...
            Non_cheat/      ...
        ...
        subject24/
            cheat/ ...
            Non_cheat/ ...

Frame files are named `<subject>_frame<NNNN>.jpg`, but the frame numbers
inside a class folder are NOT necessarily one continuous block (e.g.
subject1/cheat/ jumps from frame0350 straight to frame0900 because the
frames in between were labeled Non_cheat and live in the other folder).

So instead of naively chunking every `seq_len` frames in sorted order
(which would silently stitch together two unrelated moments in the video),
this version first splits each class folder into **contiguous runs**
(consecutive frame numbers) and only builds sequences *within* a run.
"""

import os
import re
import glob
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.preprocessing import crop_head, make_environment_frame, bgr_to_model_tensor
from utils.cache import load_cached_embedding

_FRAME_NUM_RE = re.compile(r"(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def _frame_number(path):
    m = _FRAME_NUM_RE.search(path)
    return int(m.group(1)) if m else 0


def _list_subject_dirs(dataset_dir):
    def subj_num(p):
        m = re.search(r"subject(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0
    return sorted(
        (d for d in glob.glob(os.path.join(dataset_dir, "subject*")) if os.path.isdir(d)),
        key=subj_num,
    )


def _split_into_contiguous_runs(frame_paths):
    """
    frame_paths: sorted-by-frame-number list of file paths.
    Returns a list of runs, each run a list of paths whose frame numbers
    are consecutive (gap == 1). A gap starts a new run.
    """
    if not frame_paths:
        return []
    runs = [[frame_paths[0]]]
    prev_num = _frame_number(frame_paths[0])
    for p in frame_paths[1:]:
        num = _frame_number(p)
        if num == prev_num + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
        prev_num = num
    return runs


def build_sequence_index(dataset_dir=config.DATASET_DIR, seq_len=config.SEQ_LEN,
                          class_names=config.CLASS_NAMES, stride=None, min_run_len=None):
    """
    Walks Dataset/subjectX/<class>/ and returns a list of samples:
        {"frames": [path,...] len==seq_len, "label": int, "subject": "subject1"}

    stride: step between the start of consecutive windows within a run.
        None -> stride = seq_len (non-overlapping windows, default).
        stride < seq_len gives overlapping windows -> more training sequences,
        useful for the minority class (cheat folders are usually smaller).
    min_run_len: runs shorter than this are skipped entirely. Defaults to seq_len
        (i.e. a run must be at least one full sequence long).
    """
    stride = stride or seq_len
    min_run_len = min_run_len or seq_len

    samples = []
    for subj_dir in _list_subject_dirs(dataset_dir):
        subject_name = os.path.basename(subj_dir)
        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(subj_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            frame_paths = sorted(
                glob.glob(os.path.join(class_dir, "*.jpg")) +
                glob.glob(os.path.join(class_dir, "*.png")),
                key=_frame_number,
            )
            for run in _split_into_contiguous_runs(frame_paths):
                if len(run) < min_run_len:
                    continue
                for i in range(0, len(run) - seq_len + 1, stride):
                    chunk = run[i:i + seq_len]
                    samples.append({"frames": chunk, "label": label_idx, "subject": subject_name})
    return samples


def dataset_stats(samples, class_names=config.CLASS_NAMES):
    """Quick class-balance / per-subject report, handy to sanity-check a new dataset drop."""
    by_class = defaultdict(int)
    by_subject_class = defaultdict(lambda: defaultdict(int))
    for s in samples:
        cname = class_names[s["label"]]
        by_class[cname] += 1
        by_subject_class[s["subject"]][cname] += 1
    return {
        "total_sequences": len(samples),
        "by_class": dict(by_class),
        "by_subject": {subj: dict(counts) for subj, counts in by_subject_class.items()},
    }


def _stack_frames(items):
    """Stack a list of per-frame tensors/arrays into one [T, ...] float tensor.

    The original code used torch.tensor(list_of_items, dtype=torch.float32),
    which triggers a UserWarning ("To copy construct from a tensor, it is
    recommended to use sourceTensor.clone().detach()") whenever
    bgr_to_model_tensor() already returns torch tensors (as it appears to,
    based on the warning seen during a real training run), and is also
    slower than a direct stack. This handles both that case and the case
    where it returns numpy arrays.
    """
    tensors = [x if torch.is_tensor(x) else torch.as_tensor(x) for x in items]
    return torch.stack(tensors, dim=0).float()


class CheatSequenceDataset(Dataset):
    """
    Returns per-sample:
        head_crops : FloatTensor [seq_len, 3, H, W]   (good-res head crops)
        env_frames : FloatTensor [seq_len, 3, H, W]   (lower-res environment frames)
        label      : LongTensor scalar (0=Non_cheat, 1=cheat)
        frame_paths: list[str] length seq_len (raw paths, needed by YOLO/pose branches)
    Body-pose keypoints are computed on the fly by the pose model at train time
    (kept out of the dataset so we don't duplicate that heavy computation here).
    """

    def __init__(self, samples=None, dataset_dir=config.DATASET_DIR, seq_len=config.SEQ_LEN):
        self.seq_len = seq_len
        self.samples = samples if samples is not None else build_sequence_index(dataset_dir, seq_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        head_crops, env_frames = [], []
        for p in item["frames"]:
            frame = cv2.imread(p)
            if frame is None:
                raise IOError(f"Could not read frame: {p}")
            head = crop_head(frame)
            env = make_environment_frame(frame)
            head_crops.append(bgr_to_model_tensor(head))
            env_frames.append(bgr_to_model_tensor(env))

        head_crops = _stack_frames(head_crops)   # [T,3,H,W]
        env_frames = _stack_frames(env_frames)   # [T,3,H,W]
        label = torch.tensor(item["label"], dtype=torch.long)
        raw_frame_paths = item["frames"]
        return head_crops, env_frames, label, raw_frame_paths


class CachedFusedSequenceDataset(Dataset):
    """
    Fast-path dataset for the DEFAULT (frozen-branches) training mode.

    Instead of reading raw frames and running them through crop_head /
    YOLO / MediaPipe / eye-gaze every single epoch, this loads the
    already-fused [T, FUSED_FEAT_DIM] embedding straight from the disk
    cache built by precompute_embeddings.py. Branch models never need to
    be loaded at all when training this way.

    Requires precompute_embeddings.py to have been run first for this
    data_dir -- raises a clear FileNotFoundError (naming the missing
    frame) if a sample's cache is incomplete, rather than silently
    returning zeros or crashing deep in a collate function.
    """

    def __init__(self, samples, data_dir):
        self.samples = samples
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        embs = []
        for p in item["frames"]:
            emb = load_cached_embedding(p, self.data_dir)
            if emb is None:
                raise FileNotFoundError(
                    f"No cached embedding for frame '{p}'.\n"
                    f"Run:  python precompute_embeddings.py --data-dir {self.data_dir}\n"
                    f"...before training with the cache enabled (this is the default; "
                    f"pass --no-cache or --finetune-branches to train.py to skip the "
                    f"cache and compute branch embeddings live instead)."
                )
            embs.append(emb)

        fused_seq = torch.from_numpy(np.stack(embs, axis=0)).float()  # [T, FUSED_FEAT_DIM]
        label = torch.tensor(item["label"], dtype=torch.long)
        return fused_seq, label


def train_val_split(samples, val_ratio=0.15, seed=42):
    """
    Splits by SUBJECT, not by individual sequence. Splitting sequences randomly
    would let frames from the same subject/video end up in both train and val
    (leakage -> inflated val accuracy), since adjacent windows overlap or come
    from the same short clip.
    """
    subjects = sorted({s["subject"] for s in samples})
    rnd = random.Random(seed)
    rnd.shuffle(subjects)

    n_val_subjects = max(1, int(len(subjects) * val_ratio))
    val_subjects = set(subjects[:n_val_subjects])

    train_samples = [s for s in samples if s["subject"] not in val_subjects]
    val_samples = [s for s in samples if s["subject"] in val_subjects]
    return train_samples, val_samples
