"""
models/body_pose_model.py
===========================
"Body Pose Model" branch from the diagram.
Extracts body/upper-body keypoints (leaning, turning, reaching off-screen,
etc.) and projects them into a fixed-size embedding for the RNN.

Requires: pip install mediapipe>=0.10
Also requires a pose landmarker model file (Tasks API), since MediaPipe
retired the old `mp.solutions.pose` Legacy Solution in favor of the Tasks
API. Download one of:

    pose_landmarker_lite.task   (fastest, least accurate)
    pose_landmarker_full.task   (balanced -- default here)
    pose_landmarker_heavy.task  (most accurate, slowest)

from:
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models

and point config.POSE_LANDMARKER_MODEL_PATH (or the POSE_LANDMARKER_MODEL_PATH
env var) at the downloaded .task file. Example:

    wget -O pose_landmarker_full.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

NUM_LANDMARKS = 33           # mediapipe pose landmarks
RAW_DIM = NUM_LANDMARKS * 3   # x, y, visibility per landmark

# Falls back to a local default path / env var if not set in config.py, so this
# doesn't hard-crash on import for projects that haven't added the setting yet.
DEFAULT_MODEL_PATH = os.environ.get(
    "POSE_LANDMARKER_MODEL_PATH",
    getattr(config, "POSE_LANDMARKER_MODEL_PATH", "models/weights/pose_landmarker_full.task"),
)


class BodyPoseModel:
    def __init__(self, feat_dim=config.POSE_FEAT_DIM, device="cpu", model_path=None):
        if not _HAS_MEDIAPIPE:
            raise ImportError("pip install mediapipe  # needed for pose estimation")

        model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found at '{model_path}'.\n"
                f"MediaPipe's old `mp.solutions.pose` API no longer works "
                f"(Legacy Solutions were retired) -- the Tasks API requires "
                f"a downloaded .task model file. Download one from:\n"
                f"  https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models\n"
                f"and set config.POSE_LANDMARKER_MODEL_PATH or the "
                f"POSE_LANDMARKER_MODEL_PATH env var to point at it."
            )

        # IMAGE mode == process each frame independently (equivalent to the
        # old static_image_mode=True). If you want MediaPipe to use temporal
        # smoothing/tracking across a sequence's frames, switch to VIDEO mode
        # and call detect_for_video(image, timestamp_ms) with increasing
        # timestamps per frame instead of detect(image) in _extract_landmarks.
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        self.device = device
        self.proj = nn.Linear(RAW_DIM, feat_dim).to(device)

    # ---- nn.Module-like interface -------------------------------------
    # MediaPipe's PoseLandmarker isn't a torch module and has no train/eval
    # concept of its own (it's a fixed pretrained landmark detector, not
    # something this codebase fine-tunes). We still expose eval()/train()/
    # parameters()/to() here, delegating to self.proj, so callers like
    # train.py's run_epoch() can call `body_pose_model.eval()` the same way
    # they do for the other branches without needing to special-case this
    # wrapper.

    def eval(self):
        self.proj.eval()
        return self

    def train(self, mode=True):
        self.proj.train(mode)
        return self

    def parameters(self):
        return self.proj.parameters()

    def to(self, device):
        self.device = device
        self.proj.to(device)
        return self

    # ---------------------------------------------------------------

    def _extract_landmarks(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return np.zeros(RAW_DIM, dtype=np.float32)

        # pose_landmarks is a list of detected people; we asked for num_poses=1
        landmarks = result.pose_landmarks[0]
        vec = []
        for lm in landmarks:
            vec.extend([lm.x, lm.y, lm.visibility])
        return np.array(vec, dtype=np.float32)

    @torch.no_grad()
    def embed_frame(self, frame_bgr):
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw = self._extract_landmarks(frame_rgb)
        raw_t = torch.tensor(raw, device=self.device).unsqueeze(0)
        return self.proj(raw_t).squeeze(0)  # [feat_dim]

    @torch.no_grad()
    def embed_batch(self, frames_bgr_list):
        embs = [self.embed_frame(f) for f in frames_bgr_list]
        return torch.stack(embs, dim=0)

    def close(self):
        """Release the underlying MediaPipe Tasks landmarker. The Tasks API
        (unlike the old solutions API) recommends explicitly closing task
        objects when done, since they hold native/GPU resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_body_pose_model(device="cpu", model_path=None):
    return BodyPoseModel(device=device, model_path=model_path)
