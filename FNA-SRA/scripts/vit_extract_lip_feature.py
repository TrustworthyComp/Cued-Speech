# Dependencies (run once in your conda env):
#   pip install opencv-python-headless mediapipe
#
# Improvements over the coarse version:
#   1. All 40 lip landmarks (outer 20 + inner 20):
#      Outer contour defines the lip boundary; inner contour captures mouth
#      opening/closing – critical for lip-reading in Cued Speech.
#
#   2. Face-pose normalization:
#      The frame is rotated so the eye line is horizontal before the lip crop
#      is taken.  This removes head-tilt variation so the same mouth shape
#      always looks the same regardless of the signer's head pose.
#
#   3. Tighter padding (0.20) – down from 0.40 – because the 40-point bbox
#      already tightly captures the full lip region.
#
# Output:
#   {save_dir}/{model}_lip_feat_mccsd/{signer}/{video_id}{postfix}.npy
#   shape: (T_lip, 1024)  where T_lip = frames with a detected face
#
# Usage example:
#   python scripts/vit_extract_lip_feature.py \
#       --video_root /home/uic2/mccsd_datasets/RawVideo \
#       --save_dir   /home/uic2/mccsd_datasets/Features \
#       --device     cuda:0

import argparse
import os
import os.path as osp
import glob
import tqdm
import torch
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel

import sys
sys.path.append('./')

from utils.s2wrapper import forward as multiscale_forward

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh lip landmark indices (468-point model)
# ---------------------------------------------------------------------------

# Outer lip boundary – 20 points (upper arc + lower arc)
_OUTER_LIP = [
    61, 185, 40, 39, 37,  0, 267, 269, 270, 409, 291,   # upper arc L→R
   375, 321, 405, 314, 17,  84, 181,  91, 146,           # lower arc R→L
]

# Inner lip boundary – 20 points (upper arc + lower arc)
# These track mouth opening/closing – essential for Cued Speech lip-reading.
_INNER_LIP = [
    78, 191,  80,  81,  82, 13, 312, 311, 310, 415, 308,  # inner upper arc
   324, 318, 402, 317, 14,  87, 178,  88,  95,             # inner lower arc
]

_ALL_LIP = _OUTER_LIP + _INNER_LIP   # 40 points total

# Eye landmarks used for face-pose estimation
_LEFT_EYE_OUTER  = 33    # left eye outer corner  (viewer's left)
_RIGHT_EYE_OUTER = 263   # right eye outer corner (viewer's right)


# ---------------------------------------------------------------------------
# ViT feature extractor (identical to vit_extract_feature.py)
# ---------------------------------------------------------------------------

class ViTFeatureReader(object):
    def __init__(
        self,
        model_name='openai/clip-vit-large-patch14',
        cache_dir=None,
        device='cuda:0',
        s2_mode='',
        scales=[1, 2],
        nth_layer=-1,
    ):
        self.s2_mode = s2_mode
        self.device  = device
        self.scales  = scales
        self.nth_layer = nth_layer

        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def _forward(self, inputs):
        return self.model(inputs).hidden_states[self.nth_layer]

    @torch.no_grad()
    def get_feats(self, images):
        """images: list[PIL.Image] → numpy (N, D)"""
        pv = self.image_processor(list(images), return_tensors='pt').to(self.device).pixel_values
        if self.s2_mode == 's2wrapping':
            out = multiscale_forward(self._forward, pv, scales=self.scales, num_prefix_token=1)
        else:
            out = self._forward(pv)
        return out[:, 0].cpu().numpy()   # CLS token


# ---------------------------------------------------------------------------
# Lip detector with face-pose normalization
# ---------------------------------------------------------------------------

class LipDetector:
    """
    Detects and crops the lip region from a BGR frame.

    Face-pose normalization
    -----------------------
    The angle between the two outer eye corners is computed.  The frame is
    rotated to make this angle zero (eyes horizontal) before extracting the
    lip bounding box.  This ensures that the same mouth shape produces a
    consistent crop regardless of the signer's head tilt.

    Landmark coverage
    -----------------
    40 landmarks are used (20 outer + 20 inner lip contour points) so that
    the bounding box tightly encloses the entire visible mouth region,
    including the mouth-opening captured by the inner contour.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        padding: float = 0.20,
    ):
        self.padding = padding
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> Image.Image | None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        lms = results.multi_face_landmarks[0].landmark

        # ── Face-pose normalization ────────────────────────────────────
        # Compute angle from the eye-corner line to horizontal
        lx = lms[_LEFT_EYE_OUTER].x  * w
        ly = lms[_LEFT_EYE_OUTER].y  * h
        rx = lms[_RIGHT_EYE_OUTER].x * w
        ry = lms[_RIGHT_EYE_OUTER].y * h

        angle = np.degrees(np.arctan2(ry - ly, rx - lx))   # deg to horizontal

        # Rotate around the midpoint between eyes
        eye_cx = (lx + rx) / 2
        eye_cy = (ly + ry) / 2
        M = cv2.getRotationMatrix2D((eye_cx, eye_cy), angle, 1.0)
        rotated = cv2.warpAffine(frame_rgb, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        # ── Transform all 40 lip landmarks into rotated frame ─────────
        raw_pts = np.array([[lms[i].x * w, lms[i].y * h] for i in _ALL_LIP])
        ones    = np.ones((len(raw_pts), 1))
        rot_pts = (M @ np.hstack([raw_pts, ones]).T).T   # (40, 2)

        # ── Tight bounding box with padding ───────────────────────────
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
        return Image.fromarray(crop) if crop.size > 0 else None

    def close(self):
        self._mesh.close()


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def read_video_frames_bgr(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser(
        description='Extract lip-region ViT features (40-landmark, pose-normalised) from MCCSD videos.'
    )
    p.add_argument('--video_root', required=True,
                   help='Root dir of raw videos, e.g. /home/uic2/mccsd_datasets/RawVideo')
    p.add_argument('--save_dir', required=True,
                   help='Output directory for .npy feature files')
    p.add_argument('--model_name', default='openai/clip-vit-large-patch14')
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--s2_mode', default='',
                   help='"s2wrapping" to enable multi-scale, or "" to disable')
    p.add_argument('--scales', nargs='+', type=int, default=[])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--nth_layer', type=int, default=-1)
    p.add_argument('--signers', nargs='+', default=['LF', 'HS', 'WT', 'XP'])
    p.add_argument('--padding', type=float, default=0.20,
                   help='BBox padding ratio around lip region (default 0.20)')
    p.add_argument('--min_detection_confidence', type=float, default=0.5)
    p.add_argument('--fallback_full_frame', action='store_true',
                   help='Use full frame when no face is detected (default: skip frame)')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing feature files')
    return p


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def _extract_feats(reader: ViTFeatureReader, crops: list, batch_size: int) -> np.ndarray:
    feats = []
    for j in range(0, len(crops), batch_size):
        feats.append(reader.get_feats(crops[j: j + batch_size]))
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_mccsd_lips(args):
    _model_name  = os.path.split(args.model_name)[-1]
    feat_dir_name = f'{_model_name}_lip_feat_mccsd'

    reader   = ViTFeatureReader(
        model_name=args.model_name, cache_dir=args.cache_dir,
        device=args.device, s2_mode=args.s2_mode,
        scales=args.scales, nth_layer=args.nth_layer,
    )
    detector = LipDetector(
        min_detection_confidence=args.min_detection_confidence,
        padding=args.padding,
    )

    postfix = ''
    if args.s2_mode:
        postfix = f'_{args.s2_mode}'
    if len(args.scales) == 3:
        postfix += '_large'

    try:
        for signer in args.signers:
            signer_dir = osp.join(args.video_root, signer)
            if not osp.isdir(signer_dir):
                print(f'[WARN] Not found: {signer_dir}, skipping.')
                continue

            video_files = sorted(glob.glob(osp.join(signer_dir, '*.mp4')))
            if not video_files:
                print(f'[WARN] No .mp4 in {signer_dir}, skipping.')
                continue

            save_dir = osp.join(args.save_dir, feat_dir_name, signer)
            os.makedirs(save_dir, exist_ok=True)

            for video_path in tqdm.tqdm(video_files, desc=f'[lip][{signer}]'):
                vid = osp.splitext(osp.basename(video_path))[0]
                save_path = osp.join(save_dir, f'{vid}{postfix}.npy')

                if osp.exists(save_path) and not args.force:
                    continue  # resume support

                # ── Step 1: decode frames ─────────────────────────────
                frames_bgr = read_video_frames_bgr(video_path)
                if not frames_bgr:
                    print(f'[WARN] No frames: {video_path}')
                    continue

                # ── Step 2: detect lip ROI per frame ──────────────────
                lip_crops = []
                for frame in frames_bgr:
                    roi = detector.detect(frame)
                    if roi is not None:
                        lip_crops.append(roi)
                    elif args.fallback_full_frame:
                        lip_crops.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )
                    # else: discard frame where no face is visible

                if not lip_crops:
                    print(f'[WARN] No lip frames in {video_path}')
                    continue

                # ── Step 3: ViT features ───────────────────────────────
                feats = _extract_feats(reader, lip_crops, args.batch_size)

                # ── Step 4: save (T_lip, D) ────────────────────────────
                np.save(save_path, feats)

    finally:
        detector.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    process_mccsd_lips(args)


if __name__ == '__main__':
    main()
