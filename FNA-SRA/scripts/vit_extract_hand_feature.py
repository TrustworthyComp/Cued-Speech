# Dependencies (run once in your conda env):
#   pip install opencv-python-headless mediapipe
#
# Improvements over the coarse version:
#   1. Left/right hands detected and saved as SEPARATE feature streams
#      → {video_id}_left{postfix}.npy  (T_L, 1024)
#      → {video_id}_right{postfix}.npy (T_R, 1024)
#      FNA-SRA can load both with different postfixes and concat along time.
#
#   2. Per-hand pose normalization:
#      The wrist→middle-MCP axis is rotated to be vertical before cropping,
#      so the same hand shape always appears in the same orientation regardless
#      of the signer's arm angle.
#
#   3. Tighter padding (0.15) – only the hand region, no large background.
#
# Usage example:
#   python scripts/vit_extract_hand_feature.py \
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

# MediaPipe Hands landmark indices used for pose normalization
_WRIST_IDX      = 0   # wrist
_MIDDLE_MCP_IDX = 9   # middle-finger base knuckle


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
        self.device = device
        self.scales = scales
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
# Hand detector with per-hand pose normalization
# ---------------------------------------------------------------------------

class HandDetector:
    """
    Per-frame hand detection using MediaPipe Hands.

    Returns a dict  {'Left': PIL.Image | None, 'Right': PIL.Image | None}.

    Pose normalization
    ------------------
    For each detected hand the wrist→middle-MCP axis is computed.  The hand
    crop is taken from a copy of the frame rotated so that axis is vertical.
    This makes the representation invariant to global arm/hand orientation.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        padding: float = 0.15,
    ):
        self.padding = padding
        self._mp = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
        )

    # ------------------------------------------------------------------
    def _crop_hand(self, frame_rgb: np.ndarray, landmarks) -> Image.Image | None:
        """
        Given MediaPipe hand landmarks on *frame_rgb*, rotate the frame so
        the wrist→middle-MCP axis is vertical, then crop the hand bbox.
        """
        h, w = frame_rgb.shape[:2]

        # 21 landmark pixel coordinates
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])

        # ── Pose normalization ─────────────────────────────────────────
        wx, wy = pts[_WRIST_IDX]
        mx, my = pts[_MIDDLE_MCP_IDX]
        # angle that rotates the wrist→MCP vector to point straight up
        angle = np.degrees(np.arctan2(mx - wx, wy - my))

        # Rotate around hand centroid
        cx, cy = pts.mean(axis=0)
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
        rotated = cv2.warpAffine(frame_rgb, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)

        # Transform landmark coords to rotated frame
        ones = np.ones((len(pts), 1))
        rot_pts = (M @ np.hstack([pts, ones]).T).T   # (21, 2)

        # ── Tight bounding box with padding ───────────────────────────
        x_min, y_min = rot_pts.min(axis=0)
        x_max, y_max = rot_pts.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        pad_x, pad_y = bw * self.padding, bh * self.padding

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        crop = rotated[y1:y2, x1:x2]
        return Image.fromarray(crop) if crop.size > 0 else None

    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> dict:
        """Returns {'Left': PIL|None, 'Right': PIL|None}."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mp.process(frame_rgb)

        out = {'Left': None, 'Right': None}
        if not results.multi_hand_landmarks:
            return out

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label  # 'Left'/'Right'
            out[label] = self._crop_hand(frame_rgb, hand_lms)

        return out

    def close(self):
        self._mp.close()


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
        description='Extract per-hand ViT features (left/right separated) from MCCSD videos.'
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
    p.add_argument('--padding', type=float, default=0.15,
                   help='BBox padding ratio around each hand (default 0.15)')
    p.add_argument('--min_detection_confidence', type=float, default=0.5)
    p.add_argument('--fallback_full_frame', action='store_true',
                   help='Store full-frame feature when a hand is absent (preserves T alignment)')
    p.add_argument('--left_only', action='store_true',
                   help='Only extract and save left-hand features, skip right hand entirely')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing feature files')
    return p


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_feats(reader: ViTFeatureReader, crops: list, batch_size: int) -> np.ndarray:
    """Run ViT on a list of PIL crops in batches; return (N, D) array."""
    feats = []
    for j in range(0, len(crops), batch_size):
        batch = crops[j: j + batch_size]
        feats.append(reader.get_feats(batch))
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_mccsd_hands(args):
    _model_name = os.path.split(args.model_name)[-1]
    feat_dir_name = f'{_model_name}_hand_feat_mccsd'

    reader   = ViTFeatureReader(
        model_name=args.model_name, cache_dir=args.cache_dir,
        device=args.device, s2_mode=args.s2_mode,
        scales=args.scales, nth_layer=args.nth_layer,
    )
    detector = HandDetector(
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

            for video_path in tqdm.tqdm(video_files, desc=f'[hand][{signer}]'):
                vid = osp.splitext(osp.basename(video_path))[0]

                left_path  = osp.join(save_dir, f'{vid}_left{postfix}.npy')
                right_path = osp.join(save_dir, f'{vid}_right{postfix}.npy')

                if args.left_only:
                    if osp.exists(left_path) and not args.force:
                        continue
                elif osp.exists(left_path) and osp.exists(right_path) and not args.force:
                    continue  # already done – resume support

                # ── Step 1: decode frames ─────────────────────────────
                frames_bgr = read_video_frames_bgr(video_path)
                if not frames_bgr:
                    print(f'[WARN] No frames: {video_path}')
                    continue

                # ── Step 2: detect left / right hand per frame ────────
                left_crops, right_crops = [], []

                for frame in frames_bgr:
                    det = detector.detect(frame)

                    sides = [('Left', left_crops)] if args.left_only else [('Left', left_crops), ('Right', right_crops)]
                    for side, crops_list in sides:
                        roi = det[side]
                        if roi is not None:
                            crops_list.append(roi)
                        elif args.fallback_full_frame:
                            crops_list.append(
                                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            )
                        # else: skip this frame for this hand

                # ── Step 3: ViT features + save ───────────────────────
                pairs = [(left_crops, left_path, 'left')]
                if not args.left_only:
                    pairs.append((right_crops, right_path, 'right'))
                for crops_list, save_path, side in pairs:
                    if not crops_list:
                        print(f'[WARN] No {side} hand frames in {video_path}')
                        continue
                    feats = _extract_feats(reader, crops_list, args.batch_size)
                    np.save(save_path, feats)
                    # feats.shape: (T_side, D)

    finally:
        detector.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    process_mccsd_hands(args)


if __name__ == '__main__':
    main()
