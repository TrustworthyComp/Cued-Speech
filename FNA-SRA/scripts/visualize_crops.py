"""
Visualize hand and lip crops extracted via MediaPipe landmarks.

Reuses the same HandDetector and LipDetector logic as:
  - scripts/vit_extract_hand_feature.py
  - scripts/vit_extract_lip_feature.py

Output modes (controlled by --mode):
  video   : write a side-by-side mosaic video  (default)
  frames  : save one PNG per frame into --out_dir
  both    : do both

Usage:
  python scripts/visualize_crops.py --video path/to/video.mp4
  python scripts/visualize_crops.py --video path/to/video.mp4 --mode frames --out_dir /tmp/crops
  python scripts/visualize_crops.py --video path/to/video.mp4 --max_frames 60
"""

import argparse
import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

# ── copy of landmark constants from the feature scripts ─────────────────────

_WRIST_IDX      = 0
_MIDDLE_MCP_IDX = 9

_OUTER_LIP = [
    61, 185, 40, 39, 37,  0, 267, 269, 270, 409, 291,
   375, 321, 405, 314, 17,  84, 181,  91, 146,
]
_INNER_LIP = [
    78, 191,  80,  81,  82, 13, 312, 311, 310, 415, 308,
   324, 318, 402, 317, 14,  87, 178,  88,  95,
]
_ALL_LIP = _OUTER_LIP + _INNER_LIP

_LEFT_EYE_OUTER  = 33
_RIGHT_EYE_OUTER = 263

CROP_SIZE = 224   # resize all crops to this square for the mosaic


# ── detectors (identical logic to the feature scripts) ───────────────────────

class HandDetector:
    def __init__(self, min_detection_confidence=0.5, padding=0.15):
        self.padding = padding
        self._mp = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
        )

    def _crop_hand(self, frame_rgb, landmarks):
        h, w = frame_rgb.shape[:2]
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])

        wx, wy = pts[_WRIST_IDX]
        mx, my = pts[_MIDDLE_MCP_IDX]
        angle = np.degrees(np.arctan2(mx - wx, wy - my))

        cx, cy = pts.mean(axis=0)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
        rotated = cv2.warpAffine(frame_rgb, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        ones = np.ones((len(pts), 1))
        rot_pts = (M @ np.hstack([pts, ones]).T).T

        x_min, y_min = rot_pts.min(axis=0)
        x_max, y_max = rot_pts.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        pad_x, pad_y = bw * self.padding, bh * self.padding

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        crop = rotated[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def detect(self, frame_bgr):
        """Returns {'Left': np.ndarray|None, 'Right': np.ndarray|None}  (RGB)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mp.process(frame_rgb)
        out = {'Left': None, 'Right': None}
        if not results.multi_hand_landmarks:
            return out
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            out[label] = self._crop_hand(frame_rgb, hand_lms)
        return out

    def close(self):
        self._mp.close()


class LipDetector:
    def __init__(self, min_detection_confidence=0.5, padding=0.20):
        self.padding = padding
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr):
        """Returns np.ndarray (RGB crop) or None."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0].landmark

        lx = lms[_LEFT_EYE_OUTER].x  * w
        ly = lms[_LEFT_EYE_OUTER].y  * h
        rx = lms[_RIGHT_EYE_OUTER].x * w
        ry = lms[_RIGHT_EYE_OUTER].y * h
        angle = np.degrees(np.arctan2(ry - ly, rx - lx))

        eye_cx = (lx + rx) / 2
        eye_cy = (ly + ry) / 2
        M = cv2.getRotationMatrix2D((eye_cx, eye_cy), angle, 1.0)
        rotated = cv2.warpAffine(frame_rgb, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        raw_pts = np.array([[lms[i].x * w, lms[i].y * h] for i in _ALL_LIP])
        ones    = np.ones((len(raw_pts), 1))
        rot_pts = (M @ np.hstack([raw_pts, ones]).T).T

        x_min, y_min = rot_pts.min(axis=0)
        x_max, y_max = rot_pts.max(axis=0)
        lip_w, lip_h = x_max - x_min, y_max - y_min
        pad_x = lip_w * self.padding
        pad_y = lip_h * self.padding
        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        crop = rotated[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def close(self):
        self._mesh.close()


# ── mosaic helpers ────────────────────────────────────────────────────────────

def _placeholder(size=CROP_SIZE, text='N/A'):
    """Grey tile with centered text for missing crops."""
    tile = np.full((size, size, 3), 80, dtype=np.uint8)
    cv2.putText(tile, text, (size // 2 - 20, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return tile


def _resize_square(img_rgb, size=CROP_SIZE):
    """Resize (keeping aspect) then pad to square."""
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def _label(tile, text, color=(255, 255, 100)):
    """Draw a label at the top of a tile (in-place)."""
    cv2.putText(tile, text, (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def build_mosaic(frame_bgr, left_crop, right_crop, lip_crop, frame_idx):
    """
    Build a 2×2 mosaic:
        [ original frame (resized)  |  left hand  ]
        [ lip                       |  right hand ]
    """
    s = CROP_SIZE

    # ── top-left: original frame (resized to s×s) ─────────────────────
    orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    top_left  = _resize_square(orig_rgb, s)
    cv2.putText(top_left, f'frame {frame_idx}', (4, s - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ── top-right: left hand ──────────────────────────────────────────
    if left_crop is not None:
        top_right = _resize_square(left_crop, s)
    else:
        top_right = _placeholder(s, 'Left N/A')
    _label(top_right, 'Left hand')

    # ── bottom-left: lip ──────────────────────────────────────────────
    if lip_crop is not None:
        bot_left = _resize_square(lip_crop, s)
    else:
        bot_left = _placeholder(s, 'Lip N/A')
    _label(bot_left, 'Lip')

    # ── bottom-right: right hand ──────────────────────────────────────
    if right_crop is not None:
        bot_right = _resize_square(right_crop, s)
    else:
        bot_right = _placeholder(s, 'Right N/A')
    _label(bot_right, 'Right hand')

    row1 = np.hstack([top_left,  top_right])
    row2 = np.hstack([bot_left,  bot_right])
    mosaic_rgb = np.vstack([row1, row2])

    # thin grid lines (BGR for output)
    mosaic_bgr = cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR)
    cv2.line(mosaic_bgr, (s, 0), (s, 2*s), (60, 60, 60), 1)
    cv2.line(mosaic_bgr, (0, s), (2*s, s), (60, 60, 60), 1)
    return mosaic_bgr


# ── main ─────────────────────────────────────────────────────────────────────

def process_video(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'[ERROR] Cannot open video: {args.video}')

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames if args.max_frames > 0 else total

    hand_det = HandDetector(
        min_detection_confidence=args.min_detection_confidence,
        padding=args.hand_padding,
    )
    lip_det = LipDetector(
        min_detection_confidence=args.min_detection_confidence,
        padding=args.lip_padding,
    )

    # ── prepare outputs ───────────────────────────────────────────────
    out_video = None
    if args.mode in ('video', 'both'):
        out_path = args.out_video
        if out_path is None:
            base = os.path.splitext(os.path.basename(args.video))[0]
            out_path = f'{base}_crops.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(out_path, fourcc, fps,
                                    (CROP_SIZE * 2, CROP_SIZE * 2))
        print(f'[INFO] Writing mosaic video → {out_path}')

    if args.mode in ('frames', 'both'):
        os.makedirs(args.out_dir, exist_ok=True)
        print(f'[INFO] Saving frame PNGs → {args.out_dir}/')

    # ── frame loop ────────────────────────────────────────────────────
    frame_idx = 0
    saved_count = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_idx >= max_frames:
                break

            hands = hand_det.detect(frame_bgr)
            lip   = lip_det.detect(frame_bgr)

            mosaic = build_mosaic(
                frame_bgr,
                left_crop  = hands['Left'],
                right_crop = hands['Right'],
                lip_crop   = lip,
                frame_idx  = frame_idx,
            )

            if out_video is not None:
                out_video.write(mosaic)

            if args.mode in ('frames', 'both'):
                png_path = os.path.join(args.out_dir, f'frame_{frame_idx:04d}.png')
                cv2.imwrite(png_path, mosaic)
                saved_count += 1

            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f'  processed {frame_idx}/{min(max_frames, total)} frames …')

    finally:
        cap.release()
        hand_det.close()
        lip_det.close()
        if out_video is not None:
            out_video.release()

    print(f'[DONE] {frame_idx} frames processed.')
    if args.mode in ('frames', 'both'):
        print(f'       {saved_count} PNG files saved to {args.out_dir}/')


def get_parser():
    p = argparse.ArgumentParser(
        description='Visualize MediaPipe hand and lip crops from a single video.'
    )
    p.add_argument('--video', required=True,
                   help='Path to input video file')
    p.add_argument('--mode', choices=['video', 'frames', 'both'], default='video',
                   help='Output mode: mosaic video, per-frame PNGs, or both (default: video)')
    p.add_argument('--out_video', default=None,
                   help='Output mosaic video path (default: <video_name>_crops.mp4 in cwd)')
    p.add_argument('--out_dir', default='crop_frames',
                   help='Directory for per-frame PNGs (default: crop_frames/)')
    p.add_argument('--max_frames', type=int, default=0,
                   help='Stop after this many frames (0 = all)')
    p.add_argument('--hand_padding', type=float, default=0.15,
                   help='Padding ratio for hand bbox (default 0.15)')
    p.add_argument('--lip_padding', type=float, default=0.20,
                   help='Padding ratio for lip bbox (default 0.20)')
    p.add_argument('--min_detection_confidence', type=float, default=0.5)
    return p


if __name__ == '__main__':
    args = get_parser().parse_args()
    process_video(args)
