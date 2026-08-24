"""
models/sequence_model.py
==========================
The "RNN/LSTM" box in the diagram: fuses per-frame embeddings from the
Eye-Gaze, Environment and Body-Pose branches into a per-sequence prediction.

Two heads are provided:
  - ClassifierHead   : plain per-sequence classification (used in Training / Inference)
  - MILAttentionHead : attention-based Multiple-Instance-Learning pooling,
                        used by label_by_mil.py to turn a whole video ("bag")
                        of frame-level instances into one weakly-supervised
                        video-level label + per-frame attention scores.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FusionRNN(nn.Module):
    """LSTM over the concatenated [eye_gaze | environment | pose] embeddings."""

    def __init__(
        self,
        input_dim=config.FUSED_FEAT_DIM,
        hidden_dim=config.RNN_HIDDEN_DIM,
        num_layers=config.RNN_LAYERS,
        input_dropout=config.RNN_INPUT_DROPOUT,
        inter_layer_dropout=config.RNN_INTER_LAYER_DROPOUT,
    ):
        super().__init__()
        # Dropout applied to the fused branch features themselves, before the
        # LSTM ever sees them -- regularizes against any single branch
        # (e.g. eye-gaze) dominating and the model memorizing frame-specific
        # noise from a small training set.
        self.input_dropout = nn.Dropout(input_dropout)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            # nn.LSTM's own `dropout` only applies BETWEEN stacked layers, so
            # it silently does nothing (and warns) when num_layers == 1 --
            # which is the current default (config.RNN_LAYERS = 1). It's
            # wired up correctly here for whenever RNN_LAYERS is increased.
            dropout=inter_layer_dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, fused_seq):
        """
        fused_seq: [B, T, input_dim]
        returns:
          per_step_out : [B, T, hidden_dim]   (used for MIL instance scoring)
          last_hidden  : [B, hidden_dim]      (used for plain classification)
        """
        fused_seq = self.input_dropout(fused_seq)
        out, (h_n, c_n) = self.lstm(fused_seq)
        last_hidden = h_n[-1]  # last layer's final hidden state
        return out, last_hidden


class ClassifierHead(nn.Module):
    """Sequence-level classifier used for Training / Inference (diagram: RNN/LSTM -> Prediction)."""

    def __init__(self, hidden_dim=config.RNN_HIDDEN_DIM, num_classes=config.NUM_CLASSES,
                 dropout=config.CLASSIFIER_DROPOUT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, last_hidden):
        return self.fc(last_hidden)  # [B, num_classes] logits


class MILAttentionHead(nn.Module):
    """
    Attention-based MIL pooling (Ilse et al., 2018 style).
    Learns a weight per time-step (instance) and produces one bag-level
    prediction, which is exactly the "Multi Instance Learning -> Labeled data"
    step in the diagram.
    """

    def __init__(self, hidden_dim=config.RNN_HIDDEN_DIM, num_classes=config.NUM_CLASSES, attn_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, per_step_out):
        """
        per_step_out: [B, T, hidden_dim] instance-level features from the LSTM
        returns:
          bag_logits     : [B, num_classes]  video/bag-level prediction
          attn_weights   : [B, T]            per-frame importance (softmax over T)
        """
        attn_scores = self.attn(per_step_out).squeeze(-1)      # [B, T]
        attn_weights = torch.softmax(attn_scores, dim=1)       # [B, T]
        bag_repr = torch.bmm(attn_weights.unsqueeze(1), per_step_out).squeeze(1)  # [B, hidden_dim]
        bag_logits = self.classifier(bag_repr)                 # [B, num_classes]
        return bag_logits, attn_weights


class CheatSequenceModel(nn.Module):
    """Full model = FusionRNN + one of the heads above, selectable at call time."""

    def __init__(self):
        super().__init__()
        self.rnn = FusionRNN()
        self.classifier_head = ClassifierHead()
        self.mil_head = MILAttentionHead()

    def forward(self, fused_seq, mode="classify"):
        """
        fused_seq: [B, T, FUSED_FEAT_DIM]
        mode="classify" -> (logits [B,C])                used in Training / Inference
        mode="mil"       -> (bag_logits [B,C], attn [B,T]) used in Label by MIL
        """
        per_step_out, last_hidden = self.rnn(fused_seq)
        if mode == "classify":
            return self.classifier_head(last_hidden)
        elif mode == "mil":
            return self.mil_head(per_step_out)
        else:
            raise ValueError(f"Unknown mode: {mode}")


def load_sequence_model(weights_path=config.SEQ_MODEL_WEIGHTS, device="cpu"):
    model = CheatSequenceModel()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)
