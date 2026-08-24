"""
inference.py
=============
Matches the "Inference" flow in the diagram:

    Receive input video of 30fps
        -> Extract frame only 5fps
        -> Model
        -> Prediction

Usage:
    python inference.py --video path/to/video.mp4
    python inference.py --video path/to/video.mp4 --onnx   # use the INT8 ONNX model for faster inference
"""

import argparse
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


def sample_sequence_from_video(video_path, seq_len=config.SEQ_LEN):
    """
    Step 1: 'Receive input video of 30fps' -> 'Extract frame only 5fps'.
    Grabs the first `seq_len` frames sampled at 5fps.
    For a long video in production, slide this window over the whole video
    and run predict_sequence() on each window.
    """
    head_crops, env_frames, raw_frames = [], [], []
    for _, frame in extract_frames(video_path, stride=config.FRAME_STRIDE):
        head = crop_head(frame)
        env = make_environment_frame(frame)
        head_crops.append(bgr_to_model_tensor(head))
        env_frames.append(bgr_to_model_tensor(env))
        raw_frames.append(frame)
        if len(raw_frames) == seq_len:
            break

    if len(raw_frames) < seq_len:
        raise ValueError(
            f"Video too short: only got {len(raw_frames)} sampled frames, need {seq_len}. "
            f"(At 5fps that's ~{seq_len / config.SAMPLE_FPS:.1f}s of video.)"
        )

    head_crops = torch.tensor(np.stack(head_crops), dtype=torch.float32)  # [T,3,H,W]
    env_frames = torch.tensor(np.stack(env_frames), dtype=torch.float32)  # [T,3,H,W]
    return head_crops, env_frames, raw_frames


class CheatInferenceEngine:
    """Loads every branch once and reuses them across many prediction calls."""

    def __init__(self, use_onnx=False, device=None):
        self.device = device or get_device()
        self.use_onnx = use_onnx

        self.eye_gaze_model = load_eye_gaze_model(device=self.device).eval()
        self.environment_model = load_environment_model(device=self.device)
        self.body_pose_model = load_body_pose_model(device=self.device)

        if use_onnx:
            self._load_onnx_session()
        else:
            self.sequence_model = load_sequence_model(device=self.device).eval()

    def _load_onnx_session(self):
        # "If possible try to use INT8 quantization ONNX for faster inference."
        import onnxruntime as ort
        if not __import__("os").path.exists(config.SEQ_MODEL_ONNX):
            raise FileNotFoundError(
                f"ONNX model not found at {config.SEQ_MODEL_ONNX}. "
                f"Run scripts/export_onnx.py first (see export_onnx() below)."
            )
        self.onnx_session = ort.InferenceSession(
            config.SEQ_MODEL_ONNX, providers=["CPUExecutionProvider"]
        )

    @torch.no_grad()
    def predict(self, video_path):
        head_crops, env_frames, raw_frames = sample_sequence_from_video(video_path)
        fused_seq = fuse_sequence(
            head_crops, env_frames, raw_frames,
            self.eye_gaze_model, self.environment_model, self.body_pose_model,
            self.device,
        )

        if self.use_onnx:
            ort_inputs = {"fused_seq": fused_seq.cpu().numpy().astype(np.float32)}
            logits = self.onnx_session.run(None, ort_inputs)[0]
            logits = torch.tensor(logits)
        else:
            logits = self.sequence_model(fused_seq, mode="classify")  # [1, num_classes]

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        return {
            "label": config.CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "probs": {name: float(p) for name, p in zip(config.CLASS_NAMES, probs.tolist())},
        }


def export_onnx(device="cpu"):
    """Optional helper: export the trained sequence model to INT8 ONNX (per the diagram's note)."""
    import os
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model = load_sequence_model(device=device).eval()
    dummy = torch.randn(1, config.SEQ_LEN, config.FUSED_FEAT_DIM)
    fp32_path = config.SEQ_MODEL_ONNX.replace(".onnx", "_fp32.onnx")

    torch.onnx.export(
        model, dummy, fp32_path,
        input_names=["fused_seq"], output_names=["logits"],
        dynamic_axes={"fused_seq": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    quantize_dynamic(fp32_path, config.SEQ_MODEL_ONNX, weight_type=QuantType.QInt8)
    print(f"Saved INT8 ONNX model to {config.SEQ_MODEL_ONNX}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--onnx", action="store_true", help="Use INT8 ONNX model instead of the PyTorch checkpoint")
    args = parser.parse_args()

    engine = CheatInferenceEngine(use_onnx=args.onnx)
    result = engine.predict(args.video)
    print(f"Prediction: {result['label']}  (confidence: {result['confidence']:.3f})")
    print(f"Full probs: {result['probs']}")
