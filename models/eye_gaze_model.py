"""
models/eye_gaze_model.py
=========================
"Eye Gaze Classification Model : MobileOne" branch from the diagram.
Takes the good-resolution head crop and returns a feature embedding
(and optionally a direct gaze-direction logit if you want to pretrain it).

Requires: pip install timm   (MobileOne is available via timm)
"""

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EyeGazeModel(nn.Module):
    def __init__(self, feat_dim=config.EYE_GAZE_FEAT_DIM, num_gaze_classes=None, pretrained=True):
        """
        feat_dim: size of the embedding fed downstream into the RNN.
        num_gaze_classes: if set, adds a classification head (e.g. looking
            at screen / looking away / looking at neighbor) for standalone
            pretraining of this branch before it's used as a feature extractor.
        """
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("pip install timm  # needed for the MobileOne backbone")

        self.backbone = timm.create_model(
            "mobileone_s0", pretrained=pretrained, num_classes=0  # num_classes=0 -> feature extractor
        )
        backbone_out = self.backbone.num_features
        self.proj = nn.Linear(backbone_out, feat_dim)

        self.num_gaze_classes = num_gaze_classes
        if num_gaze_classes:
            self.head = nn.Linear(feat_dim, num_gaze_classes)

    def forward(self, x, return_logits=False):
        """
        x: [B, 3, H, W] good-resolution head crops (values in [0,1])
        returns: embedding [B, feat_dim]  (+ logits [B, num_gaze_classes] if requested)
        """
        feats = self.backbone(x)
        emb = self.proj(feats)
        if return_logits and self.num_gaze_classes:
            return emb, self.head(emb)
        return emb

    @torch.no_grad()
    def embed(self, x):
        self.eval()
        return self.forward(x)


def load_eye_gaze_model(weights_path=config.EYE_GAZE_WEIGHTS, device="cpu"):
    model = EyeGazeModel(pretrained=(not os.path.exists(weights_path)))
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)
