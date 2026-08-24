"""
inference.py
=============
Matches the "Real-Time Deployment" flow in the diagram:

    Receive input video (30fps)
        -> Extract frame only 5fps (Frame sampling)
        -> Model (Trained)
        -> Prediction (Real-time Output)

with a continuous arrow all the way through -- i.e. this keeps producing
fresh predictions throughout the whole feed, not just once.

Usage:
    # Real-time / continuous (the diagram's flow) -- webcam:
    python inference.py --camera 0

    # Real-time / continuous -- pre-recorded video, sliding window across it:
    python inference.py --video path/to/video.mp4

    # Faster INT8 ONNX model instead of the PyTorch checkpoint:
    python inference.py --camera 0 --onnx

    # Original one-shot behavior (single prediction from the first window,
    # handy for a quick sanity check on one clip):
    python inference.py --video path/to/video.mp4 --once

    # More frequent updates (overlapping windows) instead of one prediction
    # per non-overlapping SEQ_LEN-frame window:
    python inference.py --camera 0 --stride 4
"""

import argparse
import os
from collections import deque

import cv2
import numpy as np
import torch

import config
from utils.preprocessing import crop_head, make_environment_frame, bgr_to_model_tensor
from utils.fusion import fuse_sequence
from models.eye_gaze_model import load_eye_gaze_model
from models.environment_model import load_environment_model
from models.body_pose_model import load_body_pose_model
from models.sequence_model import load_sequence_model


def get_device():
    return torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")


def _stack_frames(items):
    """Stack a list of per-frame tensors/arrays into one [T, ...] float tensor.

    Same helper as utils/dataset.py's _stack_frames -- bgr_to_model_tensor()
    was confirmed (via a real bug) to return a numpy.ndarray rather than a
    torch tensor, so this handles either case instead of assuming one.
    """
    tensors = [x if torch.is_tensor(x) else torch.as_tensor(x) for x in items]
    return torch.stack(tensors, dim=0).float()


class CheatInferenceEngine:
    """
    Loads every branch once and reuses them across many prediction calls.

    Two ways to use it:
      - predict(video_path): ONE prediction from the first seq_len sampled
        frames of a video file. Kept for quick single-clip sanity checks
        (this was the original/only behavior before).
      - stream(video_source, ...): the actual "Real-Time Deployment" flow
        from the diagram -- continuously reads frames from a webcam
        (pass an int index) or a video file (pass a path), samples at
        config.SAMPLE_FPS, and yields a fresh prediction every time a new
        window of seq_len frames is ready.
    """

    def __init__(self, use_onnx=False, device=None):
        self.device = device or get_device()
        self.use_onnx = use_onnx

        self.eye_gaze_model = load_eye_gaze_model(device=self.device).eval()
        self.environment_model = load_environment_model(device=self.device).eval()
        self.body_pose_model = load_body_pose_model(device=self.device).eval()

        if use_onnx:
            self._load_onnx_session()
        else:
            self.sequence_model = load_sequence_model(device=self.device).eval()

    def _load_onnx_session(self):
        # "If possible try to use INT8 quantization ONNX for faster inference."
        import onnxruntime as ort
        if not os.path.exists(config.SEQ_MODEL_ONNX):
            raise FileNotFoundError(
                f"ONNX model not found at {config.SEQ_MODEL_ONNX}. "
                f"Run export_onnx() first (see the bottom of this file)."
            )
        self.onnx_session = ort.InferenceSession(
            config.SEQ_MODEL_ONNX, providers=["CPUExecutionProvider"]
        )

    @torch.no_grad()
    def _predict_window(self, head_crops, env_frames, raw_frames):
        """head_crops/env_frames: [T,3,H,W] tensors. raw_frames: list[np.ndarray] len T."""
        fused_seq = fuse_sequence(
            head_crops, env_frames, raw_frames,
            self.eye_gaze_model, self.environment_model, self.body_pose_model,
            self.device,
        )

        if self.use_onnx:
            ort_inputs = {"fused_seq": fused_seq.cpu().numpy().astype(np.float32)}
            logits = self.onnx_session.run(None, ort_inputs)[0]
            logits = torch.from_numpy(logits)
        else:
            logits = self.sequence_model(fused_seq, mode="classify")  # [1, num_classes]

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        return {
            "label": config.CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "probs": {name: float(p) for name, p in zip(config.CLASS_NAMES, probs.tolist())},
        }

    def predict(self, video_path, seq_len=config.SEQ_LEN):
        """ONE prediction from the FIRST seq_len frames sampled at
        config.SAMPLE_FPS (~seq_len/SAMPLE_FPS seconds of video). Good for a
        quick single-clip check; for continuous monitoring of a live feed
        or an entire video, use stream() instead -- that's the actual
        "Real-time Output" behavior in the diagram."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        try:
            head_crops, env_frames, raw_frames = [], [], []
            frame_idx = 0
            while len(raw_frames) < seq_len:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % config.FRAME_STRIDE == 0:
                    head = crop_head(frame)
                    env = make_environment_frame(frame)
                    head_crops.append(bgr_to_model_tensor(head))
                    env_frames.append(bgr_to_model_tensor(env))
                    raw_frames.append(frame)
                frame_idx += 1
        finally:
            cap.release()

        if len(raw_frames) < seq_len:
            raise ValueError(
                f"Video too short: only got {len(raw_frames)} sampled frames, need {seq_len}. "
                f"(At {config.SAMPLE_FPS}fps that's ~{seq_len / config.SAMPLE_FPS:.1f}s of video.)"
            )

        head_crops = _stack_frames(head_crops)
        env_frames = _stack_frames(env_frames)
        return self._predict_window(head_crops, env_frames, raw_frames)

    def stream(self, video_source, window_stride=None, display=True, quiet=False):
        """
        THE diagram's "Real-Time Deployment" flow:
            Receive input video (30fps) -> Extract frame only 5fps -> Model -> Prediction

        video_source: webcam index (int, e.g. 0) or a video file path (str).
            cv2.VideoCapture accepts either directly.
        window_stride: how many NEW sampled (5fps) frames must arrive before
            the next prediction fires.
              None (default) -> config.SEQ_LEN, i.e. non-overlapping windows
              (one prediction per ~SEQ_LEN/SAMPLE_FPS seconds -- cheapest,
              least redundant compute).
              Smaller (e.g. 4) -> more frequent, overlapping-window updates,
              at the cost of re-running the branch models more often.
        display: if True, opens an OpenCV window overlaying the latest
            prediction on the live feed (press 'q' to stop).
        quiet: if True, don't print each prediction to stdout (useful if
            you're only consuming the yielded dicts programmatically).

        This is a generator: iterate it in a `for` loop. Each iteration
        yields one result dict (same shape as predict()'s return value,
        plus "frame_index" and "timestamp_sec") as soon as a window's
        prediction is ready.
        """
        seq_len = config.SEQ_LEN
        window_stride = window_stride or seq_len

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {video_source!r}")

        head_buf = deque(maxlen=seq_len)
        env_buf = deque(maxlen=seq_len)
        raw_buf = deque(maxlen=seq_len)
        frames_since_last_pred = 0
        frame_idx = 0
        sampled_idx = 0
        last_result = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break  # end of video file, or camera disconnected

                # "Extract frame only 5fps": keep every Nth raw (30fps) frame.
                if frame_idx % config.FRAME_STRIDE != 0:
                    frame_idx += 1
                    continue
                frame_idx += 1
                sampled_idx += 1

                head = crop_head(frame)
                env = make_environment_frame(frame)
                head_buf.append(bgr_to_model_tensor(head))
                env_buf.append(bgr_to_model_tensor(env))
                raw_buf.append(frame)
                frames_since_last_pred += 1

                result = None
                if len(raw_buf) == seq_len and frames_since_last_pred >= window_stride:
                    head_crops = _stack_frames(list(head_buf))
                    env_frames = _stack_frames(list(env_buf))
                    result = self._predict_window(head_crops, env_frames, list(raw_buf))
                    result["frame_index"] = sampled_idx
                    result["timestamp_sec"] = sampled_idx / config.SAMPLE_FPS
                    frames_since_last_pred = 0
                    last_result = result
                    if not quiet:
                        print(f"[t={result['timestamp_sec']:6.1f}s] "
                              f"{result['label']:10s} (confidence={result['confidence']:.3f})")

                if display:
                    overlay = frame.copy()
                    if last_result is not None:
                        text = f"{last_result['label']} ({last_result['confidence']:.2f})"
                        color = (0, 0, 255) if last_result["label"] == "cheat" else (0, 200, 0)
                        cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, color, 2, cv2.LINE_AA)
                    else:
                        cv2.putText(overlay, "warming up...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (200, 200, 200), 2, cv2.LINE_AA)
                    cv2.imshow("Cheat Detection (press 'q' to quit)", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if result is not None:
                    yield result
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


def export_onnx(device="cpu"):
    """Optional helper: export the trained sequence model to INT8 ONNX (per the diagram's note)."""
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
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--video", help="Path to an input video file")
    src_group.add_argument("--camera", type=int,
                            help="Webcam index for live real-time monitoring, e.g. 0")
    parser.add_argument("--onnx", action="store_true",
                         help="Use the INT8 ONNX model instead of the PyTorch checkpoint")
    parser.add_argument("--once", action="store_true",
                         help="Predict once from the first window and exit (the ORIGINAL "
                              "behavior). Default now streams predictions continuously, "
                              "matching the Real-Time Deployment diagram.")
    parser.add_argument("--stride", type=int, default=None,
                         help="Sampled-frames between predictions in stream mode. "
                              "Default = SEQ_LEN (one prediction per window, no overlap). "
                              "Use a smaller value (e.g. 4) for more frequent updates.")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable the OpenCV live preview window (e.g. for headless servers)")
    parser.add_argument("--quiet", action="store_true", help="Don't print each prediction to stdout")
    args = parser.parse_args()

    engine = CheatInferenceEngine(use_onnx=args.onnx)
    video_source = args.camera if args.camera is not None else args.video

    if args.once:
        if args.camera is not None:
            raise SystemExit("--once needs a finite clip -- use --video, not --camera, with --once.")
        result = engine.predict(video_source)
        print(f"Prediction: {result['label']}  (confidence: {result['confidence']:.3f})")
        print(f"Full probs: {result['probs']}")
    else:
        print("Starting real-time stream... (press 'q' in the preview window, or Ctrl+C, to stop)")
        for _ in engine.stream(video_source, window_stride=args.stride,
                                display=not args.no_display, quiet=args.quiet):
            pass  # stream() already prints/displays each result as it happens
