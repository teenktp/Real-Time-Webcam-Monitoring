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
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils.dataset import CheatSequenceDataset, build_sequence_index, train_val_split, dataset_stats
from utils.fusion import fuse_sequence
from models.eye_gaze_model import load_eye_gaze_model
from models.environment_model import load_environment_model
from models.body_pose_model import load_body_pose_model
from models.sequence_model import CheatSequenceModel


def get_device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def collate_raw_frames(batch):
    """Custom collate: we also need the *raw* uint8 frames (for YOLO / mediapipe),
    which don't stack cleanly like the normalized tensors do."""
    head_crops = torch.stack([b[0] for b in batch])   # [B,T,3,H,W]
    env_frames = torch.stack([b[1] for b in batch])   # [B,T,3,H,W]
    labels = torch.stack([b[2] for b in batch])        # [B]
    frame_paths = [b[3] for b in batch]                 # list[list[str]] len B, each len T
    return head_crops, env_frames, labels, frame_paths


def run_epoch(model, branches, loader, device, optimizer=None, class_weights=None):
    """optimizer=None -> eval mode, no gradient updates."""
    import cv2

    eye_gaze_model, environment_model, body_pose_model = branches
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for head_crops, env_frames, labels, frame_paths in loader:
        labels = labels.to(device)
        batch_logits = []

        # process one sample at a time through the branch models (variable-cost ops like YOLO/pose)
        for b in range(head_crops.size(0)):
            raw_frames_bgr = [cv2.imread(p) for p in frame_paths[b]]
            fused_seq = fuse_sequence(
                head_crops[b], env_frames[b], raw_frames_bgr,
                eye_gaze_model, environment_model, body_pose_model, device,
            )  # [1, T, FUSED_FEAT_DIM]
            logits = model(fused_seq, mode="classify")  # [1, num_classes]
            batch_logits.append(logits)

        logits = torch.cat(batch_logits, dim=0)  # [B, num_classes]
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_n += labels.size(0)

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
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

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

    # class-imbalance handling: e.g. cheat/Non_cheat folders are rarely balanced
    class_weights = None
    if not args.no_class_weights:
        counts = [stats["by_class"].get(name, 0) for name in config.CLASS_NAMES]
        total = sum(counts)
        if 0 not in counts:
            weights = [total / (len(counts) * c) for c in counts]
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            print(f"Using class weights {dict(zip(config.CLASS_NAMES, weights))} to counter imbalance")

    train_loader = DataLoader(
        CheatSequenceDataset(train_samples), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_raw_frames,
    )
    val_loader = DataLoader(
        CheatSequenceDataset(val_samples), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_raw_frames,
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

    branches = (eye_gaze_model, environment_model, body_pose_model)

    # ---- training loop ----
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, branches, train_loader, device, optimizer, class_weights)
        val_loss, val_acc = run_epoch(model, branches, val_loader, device, optimizer=None, class_weights=class_weights)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.SEQ_MODEL_WEIGHTS)
            if args.finetune_branches:
                torch.save(eye_gaze_model.state_dict(), config.EYE_GAZE_WEIGHTS)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.3f})")

    print(f"Training done. Best val_acc={best_val_acc:.3f}. Checkpoint: {config.SEQ_MODEL_WEIGHTS}")
    print("Tip: run `python inference.py --video ... --onnx` after exporting "
          "with export_onnx() in inference.py for faster INT8 inference.")


if __name__ == "__main__":
    main()
