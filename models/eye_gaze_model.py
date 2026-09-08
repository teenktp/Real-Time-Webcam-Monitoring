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
    """
    IMPORTANT -- this function used to have a serious bug:

    self.proj (the Linear layer mapping the pretrained MobileOne backbone's
    features down to config.EYE_GAZE_FEAT_DIM) is never actually trained in
    this pipeline -- branches are frozen by default and --finetune-branches
    was never used. Previously, if no checkpoint existed at `weights_path`
    (which was ALWAYS the case, since nothing ever saved one), this function
    just constructed a brand new EyeGazeModel() every single call --
    handing out a DIFFERENT randomly-initialized self.proj every time,
    in every process.

    That meant the RNN/classifier (trained against embeddings baked into
    embedding_cache/ by ONE specific precompute_embeddings.py run) was
    being fed embeddings from a COMPLETELY DIFFERENT random projection
    whenever eye-gaze embeddings were computed live instead of from cache
    -- i.e. every time in verify_predictions.py --live and EVERY time in
    inference.py (which always computes live; there's no camera-feed
    cache). The classifier had never seen that projection's output
    distribution, so predictions were effectively noise -- which is
    exactly the "always predicts cheat" / "predictions look inverted"
    behavior that showed up in both verify_predictions.py and the live
    webcam.

    FIX: the first time this is called with no existing checkpoint, save
    the freshly initialized weights to `weights_path` immediately. Every
    later call -- in ANY process, on ANY machine that has this file --
    then loads those EXACT SAME weights instead of a new random init. This
    doesn't make the projection "trained" in any meaningful sense (it's
    still a random projection on top of pretrained backbone features), but
    it makes it CONSISTENT across train/cache/inference, which is the
    actual requirement for a frozen feature extractor.

    ACTION REQUIRED after this fix (see the chat message for full detail):
    the embedding_cache/ and checkpoints/sequence_rnn.pt from BEFORE this
    fix were built against a projection that's now gone (it lived only in
    the memory of whichever process built the cache, and was never saved).
    They no longer match this newly-persisted projection. You must:
      1. Delete/rebuild embedding_cache/ via precompute_embeddings.py
      2. Retrain via train.py (the old sequence_rnn.pt is stale)
      3. Make sure checkpoints/eye_gaze_mobileone.pt (created by this fix)
         is the SAME FILE on every machine you run this project on -- copy
         it or distribute it via GitHub Releases, same as sequence_rnn.pt.
         Do NOT let each machine generate its own on first run, or this
         exact bug comes right back for that machine.
    """
    model = EyeGazeModel(pretrained=(not os.path.exists(weights_path)))
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"[eye_gaze_model] No checkpoint existed at '{weights_path}'. Saved this process's "
              f"freshly-initialized (untrained) weights there so every future load, on every "
              f"machine that has this exact file, is consistent instead of randomly different. "
              f"See load_eye_gaze_model()'s docstring for why this matters and what to do next.")
    return model.to(device)