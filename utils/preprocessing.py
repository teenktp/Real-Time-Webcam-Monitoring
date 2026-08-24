"""
utils/preprocessing.py
=======================
Shared preprocessing used by inference, training and MIL labeling:
  - sample a 30fps video down to 5fps
  - crop the head region (good resolution) for the eye-gaze model
  - downscale the full frame (lower resolution) for the environment model

A face detector is used to find the head box. Swap `detect_head_box` for
whatever detector you already have (mediapipe / retinaface / yolo-face...).
"""

import cv2
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Frame extraction: 30fps -> 5fps
# ---------------------------------------------------------------------------
def extract_frames(video_path, stride=config.FRAME_STRIDE):
    """
    Yields (frame_index, frame_bgr) for every `stride`-th frame of the video.
    With INPUT_FPS=30 and SAMPLE_FPS=5, stride=6.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, frame
        idx += 1
    cap.release()


# ---------------------------------------------------------------------------
# Head detection / crop  (good resolution branch)
# ---------------------------------------------------------------------------
_face_detector = None


def _get_face_detector():
    """Lazy-load a lightweight face detector (OpenCV Haar as a placeholder).
    Replace this with a stronger detector (mediapipe / RetinaFace) for production."""
    global _face_detector
    if _face_detector is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_detector = cv2.CascadeClassifier(cascade_path)
    return _face_detector


def detect_head_box(frame_bgr):
    """Returns (x, y, w, h) of the largest detected face, or None."""
    detector = _get_face_detector()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # pick the largest face box
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return tuple(faces[0])


def crop_head(frame_bgr, box=None, margin=0.4, out_size=config.HEAD_CROP_SIZE):
    """
    Crops the head region with a margin and resizes to `out_size`
    (kept at good resolution -> input for the Eye Gaze Classification model).
    Falls back to a centered crop if no face is detected.
    """
    h, w = frame_bgr.shape[:2]
    if box is None:
        box = detect_head_box(frame_bgr)

    if box is None:
        # fallback: centered square crop
        side = min(h, w)
        cx, cy = w // 2, h // 2
        x0, y0 = cx - side // 2, cy - side // 2
        crop = frame_bgr[y0:y0 + side, x0:x0 + side]
    else:
        x, y, bw, bh = box
        mx, my = int(bw * margin), int(bh * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        crop = frame_bgr[y0:y1, x0:x1]

    if crop.size == 0:
        crop = frame_bgr
    return cv2.resize(crop, out_size)


# ---------------------------------------------------------------------------
# Environment frame (lower resolution branch, full scene incl. desk/room)
# ---------------------------------------------------------------------------
def make_environment_frame(frame_bgr, out_size=config.ENV_INPUT_SIZE):
    """Downscales the *full* frame -> input for the Environment (YOLO11s) model."""
    return cv2.resize(frame_bgr, out_size)


# ---------------------------------------------------------------------------
# Convenience: build both branches from a single raw frame
# ---------------------------------------------------------------------------
def preprocess_frame(frame_bgr):
    head_crop = crop_head(frame_bgr)
    env_frame = make_environment_frame(frame_bgr)
    return head_crop, env_frame


def bgr_to_model_tensor(frame_bgr):
    """BGR uint8 HWC -> normalized float32 CHW in [0,1], RGB order."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return chw
