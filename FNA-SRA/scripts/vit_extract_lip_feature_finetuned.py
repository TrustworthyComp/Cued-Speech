# Dependencies (run once in your conda env):
#   pip install opencv-python-headless mediapipe
#
# Extract lip-region ViT features using a **fine-tuned** CLIP ViT-L/14 encoder.
# The fine-tuned weights come from the CTC+Attention training checkpoint
# produced by vit_finetune_lip_ctc_attn.py.
#
# This script mirrors vit_extract_lip_feature.py exactly in its lip-detection
# and feature-saving logic, differing only in how the ViT encoder is loaded:
# it restores the encoder weights from a fine-tuning checkpoint instead of
# downloading the vanilla HuggingFace checkpoint.  The output .npy files
# have the same (T, 1024) shape and directory layout, so they can be used
# directly by the downstream FlanT5SLT training pipeline.
#
# Usage:
#   python scripts/vit_extract_lip_feature_finetuned.py \
#       --video_root /path/to/RawVideo \
#       --save_dir   /path/to/Features \
#       --checkpoint /path/to/finetune_output/best_model.pt \
#       --device     cuda:0

import argparse
import os
import os.path as osp
import glob
import re
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


# ---------------------------------------------------------------------------
# ViT feature extractor  (loads fine-tuned weights from a training checkpoint)
# ---------------------------------------------------------------------------

class FineTunedViTFeatureReader:
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = 'openai/clip-vit-large-patch14',
        cache_dir: str = None,
        device: str = 'cuda:0',
        s2_mode: str = '',
        scales: list = (1, 2),
        nth_layer: int = -1,
    ):
        self.s2_mode = s2_mode
        self.device  = device
        self.scales  = scales
        self.nth_layer = nth_layer

        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        )

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt)

        vit_state = {}
        for key, value in state_dict.items():
            if key.startswith('vit.'):
                vit_state[key[4:]] = value

        missing, unexpected = self.model.load_state_dict(vit_state, strict=False)
        if missing:
            print(f'[INFO] Missing keys when loading ViT weights (expected for '
                  f'non-ViT params in checkpoint):')
            for k in missing[:5]:
                print(f'  {k}')
            if len(missing) > 5:
                print(f'  ... and {len(missing) - 5} more')

        self.model.to(device).eval()

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        total = len(vit_state)
        loaded = total - len(missing)
        print(f'[INFO] Loaded {loaded}/{total} fine-tuned ViT parameters '
              f'from {checkpoint_path}')

    @torch.no_grad()
    def _forward(self, inputs):
        return self.model(inputs).hidden_states[self.nth_layer]

    @torch.no_grad()
    def get_feats(self, images):
        pv = self.image_processor(
            list(images), return_tensors='pt'
        ).to(self.device).pixel_values
        if self.s2_mode == 's2wrapping':
            out = multiscale_forward(
                self._forward, pv, scales=self.scales, num_prefix_token=1
            )
        else:
            out = self._forward(pv)
        return out[:, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# Lip detector with face-pose normalization  (identical to original)
# ---------------------------------------------------------------------------

class LipDetector:
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

    def detect(self, frame_bgr: np.ndarray) -> Image.Image | None:
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
        description='Extract lip-region ViT features using a fine-tuned encoder '
                    'from a CTC+Attention training checkpoint.'
    )
    p.add_argument('--video_root', required=True,
                   help='Root dir of raw videos, e.g. /home/uic2/mccsd_datasets/RawVideo')
    p.add_argument('--sub_video_root', default=None,
                   help='Secondary video root for mccsd_sub signers (YX, YZ). '
                        'Each signer dir (e.g. YX/) is expected directly under this root.')
    p.add_argument('--save_dir', required=True,
                   help='Output directory for .npy feature files')
    p.add_argument('--checkpoint', required=True,
                   help='Path to fine-tuned checkpoint .pt file '
                        '(from vit_finetune_lip_ctc_attn.py)')
    p.add_argument('--model_name', default='openai/clip-vit-large-patch14')
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--s2_mode', default='',
                   help='"s2wrapping" to enable multi-scale, or "" to disable')
    p.add_argument('--scales', nargs='+', type=int, default=[])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--nth_layer', type=int, default=-1)
    p.add_argument('--signers', nargs='+', default=['LF', 'HS', 'WT', 'XP'])
    p.add_argument('--sub_signers', nargs='+', default=['YX', 'YZ'],
                   help='Signer IDs located under --sub_video_root')
    p.add_argument('--padding', type=float, default=0.20,
                   help='BBox padding ratio around lip region (default 0.20)')
    p.add_argument('--min_detection_confidence', type=float, default=0.5)
    p.add_argument('--frame_step', type=int, default=1,
                   help='Process every Nth frame (default 1 = all frames)')
    p.add_argument('--max_frame_size', type=int, default=480,
                   help='Resize video frame longer side ≤ this before '
                        'MediaPipe detection (default 480).')
    p.add_argument('--fallback_full_frame', action='store_true',
                   help='Use full frame when no face is detected (default: skip frame)')
    p.add_argument('--feat_dir_suffix', default='_ft',
                   help='Suffix appended to the feature directory name. '
                        'Default "_ft" yields "..._lip_feat_mccsd_ft/". '
                        'Set to "" to overwrite the original features.')
    return p


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def _extract_feats(reader: FineTunedViTFeatureReader, crops: list, batch_size: int) -> np.ndarray:
    feats = []
    for j in range(0, len(crops), batch_size):
        feats.append(reader.get_feats(crops[j: j + batch_size]))
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_mccsd_lips(args):
    _model_name  = os.path.split(args.model_name)[-1]
    feat_dir_name = f'{_model_name}_lip_feat_mccsd{args.feat_dir_suffix}'

    reader   = FineTunedViTFeatureReader(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
    )
    detector = LipDetector(
        min_detection_confidence=args.min_detection_confidence,
        padding=args.padding,
    )

    root_map = {}
    for s in args.signers:
        root_map[s] = args.video_root
    if args.sub_video_root:
        for s in args.sub_signers:
            root_map[s] = args.sub_video_root
    all_signers = sorted(set(args.signers) | set(args.sub_signers))

    postfix = ''
    if args.s2_mode:
        postfix = f'_{args.s2_mode}'
    if len(args.scales) == 3:
        postfix += '_large'

    try:
        for signer in all_signers:
            root = root_map.get(signer, args.video_root)
            signer_dir = osp.join(root, signer)
            if not osp.isdir(signer_dir):
                print(f'[WARN] Not found: {signer_dir}, skipping.')
                continue

            video_files = sorted(glob.glob(osp.join(signer_dir, '*.mp4')))
            if not video_files:
                print(f'[WARN] No .mp4 in {signer_dir}, skipping.')
                continue

            save_dir = osp.join(args.save_dir, feat_dir_name, signer)
            os.makedirs(save_dir, exist_ok=True)

            for video_path in tqdm.tqdm(video_files, desc=f'[lip_ft][{signer}]'):
                vid = osp.splitext(osp.basename(video_path))[0]
                save_path = osp.join(save_dir, f'{vid}{postfix}.npy')

                if osp.exists(save_path):
                    continue

                frames_bgr = read_video_frames_bgr(video_path)
                if not frames_bgr:
                    print(f'[WARN] No frames: {video_path}')
                    continue

                lip_crops = []
                for i, frame in enumerate(frames_bgr):
                    if i % args.frame_step != 0:
                        continue
                    h, w = frame.shape[:2]
                    scale = args.max_frame_size / max(h, w)
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                           interpolation=cv2.INTER_LINEAR)
                    roi = detector.detect(frame)
                    if roi is not None:
                        lip_crops.append(roi)
                    elif args.fallback_full_frame:
                        lip_crops.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )

                if not lip_crops:
                    print(f'[WARN] No lip frames in {video_path}')
                    continue

                feats = _extract_feats(reader, lip_crops, args.batch_size)

                np.save(save_path, feats)

    finally:
        detector.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    process_mccsd_lips(args)


if __name__ == '__main__':
    main()
