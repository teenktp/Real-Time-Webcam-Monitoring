"""
utils/fusion.py
=================
Runs the three branches (Eye Gaze / Environment / Body Pose) on a sequence
of frames and concatenates their embeddings -> the tensor that feeds the
RNN/LSTM box in the diagram.
"""

import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@torch.no_grad()
def fuse_sequence(head_crops, env_frames, raw_frames_bgr, eye_gaze_model, environment_model, body_pose_model, device):
    """
    head_crops     : [T, 3, H, W] float tensor (good-res head crops, already normalized)
    env_frames     : [T, 3, H, W] float tensor (lower-res env frames, already normalized)
    raw_frames_bgr : list[np.ndarray] length T, original BGR frames
                      (kept separately because the YOLO / mediapipe wrappers
                      expect raw uint8 images rather than normalized tensors)
    returns: fused_seq [1, T, FUSED_FEAT_DIM]  ready for CheatSequenceModel
    """
    head_crops = head_crops.to(device)
    eye_gaze_emb = eye_gaze_model.embed(head_crops)                    # [T, EYE_GAZE_FEAT_DIM]
    env_emb = environment_model.embed_batch(raw_frames_bgr)            # [T, ENV_FEAT_DIM]
    pose_emb = body_pose_model.embed_batch(raw_frames_bgr)             # [T, POSE_FEAT_DIM]

    fused = torch.cat([eye_gaze_emb, env_emb.to(device), pose_emb.to(device)], dim=-1)  # [T, FUSED_FEAT_DIM]
    return fused.unsqueeze(0)  # [1, T, FUSED_FEAT_DIM]
