"""
label_by_mil.py
=================
Matches the "Label by MIL" flow in the diagram:

    Video
        -> Crop head (good resolution)  -> Eye Gaze Classification Model (MobileOne)
        -> Lower resolution              -> Environment Model (YOLO11s)
        -> Body Pose Model
        -> RNN/LSTM
        -> Multi Instance Learning
        -> Labeled data

Takes RAW, unlabeled videos in `raw_videos/`, treats each video as one
"bag" of frame instances, uses the attention-based MIL head to (a) decide
the bag-level (video-level) label and (b) get a per-frame attention score,
then writes out frames into the same folder layout as the screenshot:

    Dataset_MIL/
        subjectX/
            cheat/       subjectX_frameNNNN.jpg
            Non_cheat/   subjectX_frameNNNN.jpg

Only frames whose attention weight clears `--attn-threshold` are kept as
"strong" evidence for that label (reduces label noise from weak instances,
which is the whole point of MIL).

Usage:
    python label_by_mil.py --video raw_videos/subject3.mp4 --subject subject3
    python label_by_mil.py --video-dir raw_videos/            # batch mode, subject name = filename stem
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch

import config
from utils.preprocessing import extract_frames, crop_head, make_environment_frame, bgr_to_model_tensor
from utils.fusion import fuse_sequence
from models.eye_gaze_model import load_eye_gaze_model
from models.environment_model import load_environment_model
from models.body_pose_model import load_body_pose_model
from models.sequence_model import load_sequence_model


def get_device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def sliding_windows(video_path, seq_len=config.SEQ_LEN):
    """
    Yields (window_frames_bgr, window_head_crops, window_env_frames, start_frame_idx)
    for non-overlapping windows of `seq_len` frames sampled at 5fps.
    Each window is one MIL "bag".
    """
    buf_raw, buf_head, buf_env, buf_idx = [], [], [], []
    for frame_idx, frame in extract_frames(video_path, stride=config.FRAME_STRIDE):
        head = crop_head(frame)
        env = make_environment_frame(frame)
        buf_raw.append(frame)
        buf_head.append(bgr_to_model_tensor(head))
        buf_env.append(bgr_to_model_tensor(env))
        buf_idx.append(frame_idx)

        if len(buf_raw) == seq_len:
            head_t = torch.tensor(np.stack(buf_head), dtype=torch.float32)
            env_t = torch.tensor(np.stack(buf_env), dtype=torch.float32)
            yield buf_raw, head_t, env_t, buf_idx
            buf_raw, buf_head, buf_env, buf_idx = [], [], [], []
    # drop the trailing partial window (shorter than seq_len)


def label_video_with_mil(video_path, subject_name, out_dir=config.MIL_LABELED_DIR,
                          attn_threshold=0.08, device=None):
    """
    Runs the full 'Label by MIL' pipeline on one video and writes labeled
    frames to out_dir/subject_name/<cheat|Non_cheat>/subject_frameNNNN.jpg
    """
    device = device or get_device()

    eye_gaze_model = load_eye_gaze_model(device=device).eval()
    environment_model = load_environment_model(device=device)
    body_pose_model = load_body_pose_model(device=device)
    sequence_model = load_sequence_model(device=device).eval()

    saved_counts = {name: 0 for name in config.CLASS_NAMES}

    for raw_frames, head_crops, env_frames, frame_idxs in sliding_windows(video_path):
        fused_seq = fuse_sequence(
            head_crops, env_frames, raw_frames,
            eye_gaze_model, environment_model, body_pose_model, device,
        )  # [1, T, FUSED_FEAT_DIM]

        with torch.no_grad():
            bag_logits, attn_weights = sequence_model(fused_seq, mode="mil")
            bag_probs = torch.softmax(bag_logits, dim=-1).squeeze(0)   # [num_classes]
            attn_weights = attn_weights.squeeze(0).cpu().numpy()        # [T]

        bag_label_idx = int(torch.argmax(bag_probs).item())
        bag_label = config.CLASS_NAMES[bag_label_idx]
        bag_conf = float(bag_probs[bag_label_idx].item())

        class_dir = os.path.join(out_dir, subject_name, bag_label)
        os.makedirs(class_dir, exist_ok=True)

        for frame, f_idx, attn in zip(raw_frames, frame_idxs, attn_weights):
            # keep only frames the MIL head actually attended to strongly ->
            # these are the "instances" that best support the bag label
            if attn < attn_threshold:
                continue
            out_path = os.path.join(class_dir, f"{subject_name}_frame{f_idx:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_counts[bag_label] += 1

        print(f"  window @frame {frame_idxs[0]:04d}-{frame_idxs[-1]:04d}: "
              f"label={bag_label} conf={bag_conf:.3f} "
              f"(kept {int((attn_weights >= attn_threshold).sum())}/{len(attn_weights)} frames)")

    print(f"[{subject_name}] done. Saved -> {saved_counts}")
    return saved_counts


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single raw video")
    group.add_argument("--video-dir", help="Folder of raw videos to label in batch (subject name = filename stem)")
    parser.add_argument("--subject", help="Subject folder name (defaults to video filename stem)")
    parser.add_argument("--out-dir", default=config.MIL_LABELED_DIR)
    parser.add_argument("--attn-threshold", type=float, default=0.08)
    args = parser.parse_args()

    if args.video:
        subject = args.subject or os.path.splitext(os.path.basename(args.video))[0]
        label_video_with_mil(args.video, subject, out_dir=args.out_dir, attn_threshold=args.attn_threshold)
    else:
        videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")) +
                         glob.glob(os.path.join(args.video_dir, "*.avi")))
        if not videos:
            raise RuntimeError(f"No videos found in {args.video_dir}")
        for v in videos:
            subject = os.path.splitext(os.path.basename(v))[0]
            print(f"Labeling {v} as subject '{subject}' ...")
            label_video_with_mil(v, subject, out_dir=args.out_dir, attn_threshold=args.attn_threshold)


if __name__ == "__main__":
    main()
