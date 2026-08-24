"""
train.py
=========
Matches the "Training" flow in the diagram:

    Labeled data (manual) and Labeled data (MIL)
        -> Crop head (good resolution)  -> Eye Gaze Classification Model (MobileOne)
        -> Lower resolution             -> Environment Model (YOLO11s)
        -> Body Pose Model
        -> RNN/LSTM
        -> Prediction

By default this trains ONLY the fusion RNN/LSTM + classifier head on top of
FROZEN branch models (fast, small-data friendly, avoids overfitting the
big backbones). Set --finetune-branches to also fine-tune the branches.

Usage:
    python train.py
    python train.py --epochs 50 --batch-size 4
    python train.py --data-dir Dataset_MIL   # train using the MIL-auto-labeled set instead
    python train.py --resume checkpoints/last.ckpt   # resume from a previous run

Changes vs. the original version:
    - Branch models are correctly switched between .train()/.eval() every
      epoch when --finetune-branches is used (previously stuck in eval()).
    - Frame images are decoded inside the DataLoader worker processes
      (via collate_fn, which PyTorch runs in the worker) instead of the
      main training loop, so disk I/O overlaps with GPU compute when
      --num-workers > 0.
    - cv2.imread() failures now raise a clear error naming the bad path
      instead of silently producing None and crashing deep inside fusion.
    - Gradient clipping, AMP (mixed precision), LR scheduling, early
      stopping, deterministic seeding, and full checkpoint/resume support
      (model + optimizer + scheduler + scaler + epoch) were added.
    - Class weights are computed from the TRAIN split only.
    - tqdm progress bars for both the epoch loop and the batch loop.

NOTE: per-sample looping over branch models (YOLO / pose) is kept as-is
because those branch models are not assumed to support batched inference
in fuse_sequence(). If they do, that loop can be batched for a further
speedup -- see the comment in run_epoch().
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from utils.dataset import CheatSequenceDataset, build_sequence_index, train_val_split, dataset_stats
from utils.fusion import fuse_sequence
from models.eye_gaze_model import load_eye_gaze_model
from models.environment_model import load_environment_model
from models.body_pose_model import load_body_pose_model
from models.sequence_model import CheatSequenceModel


def get_device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_raw_frames(batch):
    """Custom collate: we also need the *raw* uint8 frames (for YOLO / mediapipe),
    which don't stack cleanly like the normalized tensors do.

    NOTE: cv2.imread() happens HERE (not in the main training loop). When the
    DataLoader is created with num_workers > 0, collate_fn runs inside the
    worker subprocess, so this decoding is parallelized across workers and
    overlaps with GPU compute in the main process -- instead of blocking the
    main loop as it did before.
    """
    import cv2

    head_crops = torch.stack([b[0] for b in batch])   # [B,T,3,H,W]
    env_frames = torch.stack([b[1] for b in batch])   # [B,T,3,H,W]
    labels = torch.stack([b[2] for b in batch])        # [B]
    frame_paths = [b[3] for b in batch]                 # list[list[str]] len B, each len T

    raw_frames_bgr = []
    for paths_for_sample in frame_paths:
        imgs = []
        for p in paths_for_sample:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError(
                    f"Failed to read image at '{p}'. File may be missing, "
                    f"corrupted, or not a valid image format."
                )
            imgs.append(img)
        raw_frames_bgr.append(imgs)

    return head_crops, env_frames, labels, frame_paths, raw_frames_bgr


def run_epoch(model, branches, loader, device, optimizer=None, class_weights=None,
              finetune_branches=False, scaler=None, grad_clip_norm=1.0, desc="epoch"):
    """optimizer=None -> eval mode, no gradient updates."""
    eye_gaze_model, environment_model, body_pose_model = branches
    is_train = optimizer is not None

    model.train() if is_train else model.eval()
    # Branch models: only switch to train() when we're both training AND
    # actually fine-tuning them. Otherwise they must stay in eval() at all
    # times (BatchNorm/Dropout should behave consistently as frozen feature
    # extractors), including during the train loop.
    if finetune_branches:
        eye_gaze_model.train() if is_train else eye_gaze_model.eval()
    else:
        eye_gaze_model.eval()
    environment_model.eval()
    body_pose_model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for head_crops, env_frames, labels, frame_paths, raw_frames_bgr in pbar:
        labels = labels.to(device)
        batch_logits = []

        # Process one sample at a time through the branch models (variable-cost
        # ops like YOLO/pose). If fuse_sequence / the branch models support
        # batched inputs, this per-sample loop could be replaced with a single
        # batched call for a meaningful speedup -- left as-is here since we
        # can't assume batching support without seeing fuse_sequence's internals.
        with torch.cuda.amp.autocast(enabled=use_amp):
            for b in range(head_crops.size(0)):
                fused_seq = fuse_sequence(
                    head_crops[b], env_frames[b], raw_frames_bgr[b],
                    eye_gaze_model, environment_model, body_pose_model, device,
                )  # [1, T, FUSED_FEAT_DIM]
                logits = model(fused_seq, mode="classify")  # [1, num_classes]
                batch_logits.append(logits)

            logits = torch.cat(batch_logits, dim=0)  # [B, num_classes]
            loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            trainable_params = [p for g in optimizer.param_groups for p in g["params"]]
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                optimizer.step()

        batch_n = labels.size(0)
        batch_correct = (logits.argmax(dim=1) == labels).sum().item()
        total_loss += loss.item() * batch_n
        total_correct += batch_correct
        total_n += batch_n
        pbar.set_postfix(loss=f"{total_loss / total_n:.4f}", acc=f"{total_correct / total_n:.3f}")

    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=config.DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--finetune-branches", action="store_true",
                         help="Also fine-tune eye-gaze/environment/pose branches (slower, needs more data)")
    parser.add_argument("--stride", type=int, default=None,
                         help="Sliding-window stride within each contiguous run. "
                              "Default = seq_len (no overlap). Use a smaller value "
                              "(e.g. 4-8) to oversample the minority class (usually 'cheat').")
    parser.add_argument("--no-class-weights", action="store_true",
                         help="Disable automatic class-imbalance weighting in the loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--amp", action="store_true", default=True,
                         help="Use automatic mixed precision on CUDA (default: on)")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable mixed precision")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                         help="Stop if val_loss doesn't improve for this many epochs (0 = disabled)")
    parser.add_argument("--checkpoint-dir", default=getattr(config, "CHECKPOINT_DIR", "checkpoints"),
                         help="Where to save resumable checkpoints (model+optimizer+scheduler+epoch)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device} | seed: {args.seed}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- data ---------------------------------------------------------
    all_samples = build_sequence_index(args.data_dir, stride=args.stride)
    if not all_samples:
        raise RuntimeError(f"No sequences found under {args.data_dir}. Check your Dataset/subjectX/<class>/ layout.")

    stats = dataset_stats(all_samples)
    print(f"Dataset: {stats['total_sequences']} sequences total | by class: {stats['by_class']} "
          f"| {len(stats['by_subject'])} subjects")

    train_samples, val_samples = train_val_split(all_samples)
    print(f"Train sequences: {len(train_samples)} | Val sequences: {len(val_samples)} "
          f"(split is by SUBJECT to avoid leakage, not by individual sequence)")

    # class-imbalance handling: computed from the TRAIN split only, so it
    # reflects what the model actually trains on (not the full dataset,
    # which includes the held-out val subjects).
    class_weights = None
    if not args.no_class_weights:
        train_stats = dataset_stats(train_samples)
        counts = [train_stats["by_class"].get(name, 0) for name in config.CLASS_NAMES]
        total = sum(counts)
        if 0 not in counts:
            weights = [total / (len(counts) * c) for c in counts]
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            print(f"Using class weights (train split only) {dict(zip(config.CLASS_NAMES, weights))}")
        else:
            print("Warning: at least one class has 0 train samples; skipping class weighting.")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_raw_frames,
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(
        CheatSequenceDataset(train_samples), batch_size=args.batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        CheatSequenceDataset(val_samples), batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )

    # ---- branch models (Crop head -> Eye Gaze, Lower res -> Environment, Body Pose) ----
    eye_gaze_model = load_eye_gaze_model(device=device)
    environment_model = load_environment_model(device=device)
    body_pose_model = load_body_pose_model(device=device)

    if not args.finetune_branches:
        for p in eye_gaze_model.parameters():
            p.requires_grad = False
        eye_gaze_model.eval()
        print("Branch models frozen (feature-extractor mode). Use --finetune-branches to train them too.")

    # ---- fusion RNN/LSTM + classifier head ----
    model = CheatSequenceModel().to(device)

    trainable_params = list(model.parameters())
    if args.finetune_branches:
        trainable_params += list(eye_gaze_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    branches = (eye_gaze_model, environment_model, body_pose_model)

    # ---- optionally resume ----
    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if args.finetune_branches and ckpt.get("eye_gaze_state_dict") is not None:
            eye_gaze_model.load_state_dict(ckpt["eye_gaze_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch} (best_val_acc={best_val_acc:.3f}, best_val_loss={best_val_loss:.4f})")

    # ---- training loop ----
    epoch_bar = tqdm(range(start_epoch, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        train_loss, train_acc = run_epoch(
            model, branches, train_loader, device, optimizer, class_weights,
            finetune_branches=args.finetune_branches, scaler=scaler,
            grad_clip_norm=args.grad_clip_norm, desc=f"epoch {epoch} [train]",
        )
        val_loss, val_acc = run_epoch(
            model, branches, val_loader, device, optimizer=None, class_weights=class_weights,
            finetune_branches=args.finetune_branches, scaler=None, desc=f"epoch {epoch} [val]",
        )
        scheduler.step(val_loss)

        epoch_bar.write(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        improved = val_loss < best_val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.SEQ_MODEL_WEIGHTS)
            if args.finetune_branches:
                torch.save(eye_gaze_model.state_dict(), config.EYE_GAZE_WEIGHTS)
            epoch_bar.write(f"  -> saved new best weights (val_loss={val_loss:.4f}, val_acc={val_acc:.3f})")
        else:
            epochs_no_improve += 1

        # Full resumable checkpoint every epoch (separate from the "best weights only" files above)
        ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "eye_gaze_state_dict": eye_gaze_model.state_dict() if args.finetune_branches else None,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }, ckpt_path)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            epoch_bar.write(
                f"Early stopping: no val_loss improvement for {args.early_stop_patience} epochs."
            )
            break

    print(f"Training done. Best val_acc={best_val_acc:.3f}, best val_loss={best_val_loss:.4f}. "
          f"Best weights: {config.SEQ_MODEL_WEIGHTS} | Resumable checkpoint: "
          f"{os.path.join(args.checkpoint_dir, 'last.ckpt')}")
    print("Tip: run `python inference.py --video ... --onnx` after exporting "
          "with export_onnx() in inference.py for faster INT8 inference.")


if __name__ == "__main__":
    main()
