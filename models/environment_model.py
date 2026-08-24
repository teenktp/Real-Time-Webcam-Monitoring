"""
models/environment_model.py
=============================
"Environment Model : Yolo11s" branch from the diagram.
Runs YOLO11s on the lower-resolution full frame to detect objects/people
around the subject (phone, book, second person, etc.), then turns the
detections into a fixed-size embedding for the RNN.

Requires: pip install ultralytics
"""

import itertools

import numpy as np
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Classes we specifically care about for a cheating-context environment.
# Adjust to match your fine-tuned YOLO11s class list.
SUSPICIOUS_CLASSES = ["cell phone", "book", "person", "laptop", "paper"]


class EnvironmentModel:
    """
    Thin wrapper around a YOLO11s detector that converts detections into a
    fixed-length embedding: [count_per_suspicious_class, mean_confidence,
    projected into feat_dim via a small MLP].

    NOTE: this class is a plain Python wrapper, not an nn.Module (YOLO's
    ultralytics wrapper doesn't compose cleanly as a submodule). It exposes
    eval()/train()/parameters()/to() itself, delegating to whichever
    sub-parts actually hold learnable weights (self.proj, and the
    underlying torch model inside self.yolo), so callers like train.py's
    run_epoch() can treat it the same way they'd treat a real nn.Module
    branch -- e.g. `environment_model.eval()` -- without needing to know
    it's a wrapper internally.
    """

    def __init__(self, weights_path=config.ENV_MODEL_WEIGHTS, feat_dim=config.ENV_FEAT_DIM, device="cpu"):
        if not _HAS_ULTRALYTICS:
            raise ImportError("pip install ultralytics  # needed for YOLO11s")

        # if you have a fine-tuned checkpoint use it, otherwise fall back to base yolo11s.pt
        model_path = weights_path if os.path.exists(weights_path) else "yolo11s.pt"
        self.yolo = YOLO(model_path)
        self.device = device
        self.feat_dim = feat_dim

        raw_dim = len(SUSPICIOUS_CLASSES) + 1  # per-class counts + mean confidence
        self.proj = nn.Linear(raw_dim, feat_dim).to(device)

    # ---- nn.Module-like interface -------------------------------------
    # YOLO() from ultralytics wraps its own torch model at self.yolo.model.
    # We toggle that plus our own self.proj so this branch behaves like a
    # normal model when the training loop calls .eval()/.train() on it,
    # regardless of whether --finetune-branches is fine-tuning it or not.

    def eval(self):
        self.proj.eval()
        if hasattr(self.yolo, "model") and self.yolo.model is not None:
            self.yolo.model.eval()
        return self

    def train(self, mode=True):
        self.proj.train(mode)
        if hasattr(self.yolo, "model") and self.yolo.model is not None:
            self.yolo.model.train(mode)
        return self

    def parameters(self):
        """Trainable params, for optimizers when --finetune-branches also
        covers the environment branch (not used by default in train.py)."""
        yolo_params = []
        if hasattr(self.yolo, "model") and self.yolo.model is not None:
            yolo_params = self.yolo.model.parameters()
        return itertools.chain(self.proj.parameters(), yolo_params)

    def to(self, device):
        self.device = device
        self.proj.to(device)
        if hasattr(self.yolo, "model") and self.yolo.model is not None:
            self.yolo.model.to(device)
        return self

    # ---------------------------------------------------------------

    def _detections_to_vector(self, result):
        counts = {c: 0 for c in SUSPICIOUS_CLASSES}
        confs = []
        names = result.names
        for box in result.boxes:
            cls_name = names[int(box.cls.item())]
            conf = float(box.conf.item())
            confs.append(conf)
            if cls_name in counts:
                counts[cls_name] += 1
        vec = [counts[c] for c in SUSPICIOUS_CLASSES]
        vec.append(float(np.mean(confs)) if confs else 0.0)
        return torch.tensor(vec, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def embed_frame(self, frame_bgr):
        """frame_bgr: single HxWx3 uint8 image (already resized to ENV_INPUT_SIZE)."""
        results = self.yolo.predict(frame_bgr, verbose=False, device=self.device)
        raw_vec = self._detections_to_vector(results[0])
        return self.proj(raw_vec.unsqueeze(0)).squeeze(0)  # [feat_dim]

    @torch.no_grad()
    def embed_batch(self, frames_bgr_list):
        """frames_bgr_list: list of HxWx3 uint8 images -> [N, feat_dim] tensor."""
        embs = [self.embed_frame(f) for f in frames_bgr_list]
        return torch.stack(embs, dim=0)


def load_environment_model(device="cpu"):
    return EnvironmentModel(device=device)
