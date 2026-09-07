"""
config.py
=========
Central configuration for the cheat-detection pipeline.
Edit paths / hyperparameters here instead of hunting through every script.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(ROOT_DIR, "Dataset")          # e.g. Dataset/subject1/cheat/*.jpg
RAW_VIDEO_DIR = os.path.join(ROOT_DIR, "raw_videos")      # unlabeled videos to be auto-labeled by MIL
MIL_LABELED_DIR = os.path.join(ROOT_DIR, "Dataset_MIL")   # output of label_by_mil.py (same layout as Dataset)
DATASET_AUGMENTED_DIR = os.path.join(ROOT_DIR, "Dataset_Augmented")

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
ONNX_DIR = os.path.join(ROOT_DIR, "onnx_export")
CACHE_DIR = os.path.join(ROOT_DIR, "embedding_cache")   # precomputed per-frame branch embeddings

EYE_GAZE_WEIGHTS = os.path.join(CHECKPOINT_DIR, "eye_gaze_mobileone.pt")
ENV_MODEL_WEIGHTS = os.path.join(CHECKPOINT_DIR, "environment_yolo11s.pt")
BODY_POSE_WEIGHTS = os.path.join(CHECKPOINT_DIR, "body_pose.pt")
SEQ_MODEL_WEIGHTS = os.path.join(CHECKPOINT_DIR, "sequence_rnn.pt")
SEQ_MODEL_ONNX = os.path.join(ONNX_DIR, "sequence_rnn_int8.onnx")

for d in [CHECKPOINT_DIR, ONNX_DIR, MIL_LABELED_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------
INPUT_FPS = 30          # fps of the raw incoming video
SAMPLE_FPS = 5          # fps we actually run the model on (matches diagram: "Extract frame only 5fps")
FRAME_STRIDE = INPUT_FPS // SAMPLE_FPS   # = 6 -> take every 6th frame

# how many sampled frames make up one sequence ("bag" for MIL / one window for the RNN)
SEQ_LEN = 16

# ---------------------------------------------------------------------------
# Crop / resolution settings
# ---------------------------------------------------------------------------
HEAD_CROP_SIZE = (224, 224)     # good-resolution head crop fed to the eye-gaze model
ENV_INPUT_SIZE = (320, 320)     # lower-resolution full-frame fed to the environment (YOLO11s) model

# ---------------------------------------------------------------------------
# Model dims
# ---------------------------------------------------------------------------
EYE_GAZE_FEAT_DIM = 128     # embedding size pulled from MobileOne backbone
ENV_FEAT_DIM = 128          # embedding size pulled from YOLO11s backbone
POSE_FEAT_DIM = 64          # embedding size from body-pose keypoints
FUSED_FEAT_DIM = EYE_GAZE_FEAT_DIM + ENV_FEAT_DIM + POSE_FEAT_DIM

RNN_HIDDEN_DIM = 128
RNN_LAYERS = 1
NUM_CLASSES = 2   # cheat / non_cheat

# Regularization (added to curb the overfitting seen in practice: val_loss
# starts rising again ~epoch 5 while train_loss keeps dropping). Feel free
# to tune these; RNN_INTER_LAYER_DROPOUT only takes effect if RNN_LAYERS > 1
# (nn.LSTM's own `dropout` arg is a no-op / warns for a single layer).
RNN_INPUT_DROPOUT = 0.2        # dropout on the fused [eye_gaze|env|pose] features before the LSTM
RNN_INTER_LAYER_DROPOUT = 0.2  # dropout between stacked LSTM layers (only if RNN_LAYERS > 1)
CLASSIFIER_DROPOUT = 0.3       # dropout inside the classifier head (was hardcoded 0.3, now configurable)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda"   # falls back to "cpu" automatically in code if no GPU

CLASS_NAMES = ["Non_cheat", "cheat"]
