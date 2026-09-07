# Dependencies (run once in your conda env):
#   pip install opencv-python-headless mediapipe
#
# Design goals (fine granularity):
#   1. Body-pose normalization via MediaPipe Pose:
#      The full frame is rotated so that the shoulder line is horizontal before
#      ViT encoding.  This removes torso-lean variation so the same signing
#      posture always appears in a consistent orientation regardless of the
#      signer's body angle relative to the camera.
#
#   2. Upper-body crop:
#      Instead of feeding the entire raw frame (which contains a lot of
#      uninformative background), we crop from the top of the frame down to
#      just below the hips.  The crop is computed from MediaPipe Pose
#      landmarks (shoulders + hips) with configurable padding.
#      Fallback: full frame is used when no body is detected.
#
#   3. Producer-consumer pipeline (CPU ‖ GPU):
#      A thread pool runs video decode + MediaPipe on the CPU while the main
#      thread runs ViT inference on the GPU.  Each worker thread owns a
#      private MediaPipe Pose instance (MediaPipe is not thread-safe).
#      GPU is never idle waiting for CPU.
#
#   4. Frame-level quality filtering:
#      Frames where MediaPipe Pose shoulder visibility is below 0.3 are
#      discarded (or replaced by full frame if --fallback_full_frame).
#
#   5. Resume support:
#      Already-extracted .npy files are skipped automatically.
#
# Output:
#   {save_dir}/{model}_global_feat_mccsd/{signer}/{video_id}{postfix}.npy
#   shape: (T_global, 1024)  where T_global ≤ total_video_frames
#
# Usage example:
#   python scripts/vit_extract_global_feature.py \
#       --video_root /home/uic2/mccsd_datasets/RawVideo \
#       --save_dir   /home/uic2/mccsd_datasets/Global_Features \
#       --device     cuda:1 \
#       --num_workers 4

import argparse
import os
import os.path as osp
import glob
import threading
import tqdm
import torch
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoImageProcessor, CLIPVisionModel

import sys
sys.path.append('./')

from utils.s2wrapper import forward as multiscale_forward

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

# Thread-local storage: each worker thread gets its own MediaPipe Pose instance
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices
# ---------------------------------------------------------------------------

_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24
_NOSE           = 0


# ---------------------------------------------------------------------------
# ViT feature extractor (GPU, single instance shared by main thread only)
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
        self.s2_mode   = s2_mode
        self.device    = device
        self.scales    = scales
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
        return out[:, 0].cpu().numpy()   # CLS token → (N, D)


# ---------------------------------------------------------------------------
# Full-frame processor with body-pose normalization
# ---------------------------------------------------------------------------

class GlobalFrameProcessor:
    """
    Per-frame body-pose normalisation and upper-body crop for global features.

    NOT thread-safe: each thread must create its own instance.
    Use _get_thread_processor() to obtain a per-thread instance.

    Processing pipeline per frame
    -----------------------------
    1. MediaPipe Pose: locate body landmarks.
    2. Compute shoulder-line angle; rotate frame to make shoulders horizontal.
    3. Re-project landmarks; crop upper body (nose→hips + padding).
    4. Return PIL crop for ViT encoding.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        upper_body_padding: float = 0.15,
    ):
        self.padding = upper_body_padding
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    def _rotate_frame(self, frame_rgb, lms):
        h, w = frame_rgb.shape[:2]
        lsx = lms[_LEFT_SHOULDER].x  * w;  lsy = lms[_LEFT_SHOULDER].y  * h
        rsx = lms[_RIGHT_SHOULDER].x * w;  rsy = lms[_RIGHT_SHOULDER].y * h
        angle = np.degrees(np.arctan2(lsy - rsy, lsx - rsx))
        cx, cy = (lsx + rsx) / 2.0, (lsy + rsy) / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(frame_rgb, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        return rotated, M

    def _upper_body_crop(self, rotated, lms, M):
        h, w = rotated.shape[:2]
        key_indices = [_NOSE, _LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]
        raw_pts = np.array([[lms[i].x * w, lms[i].y * h] for i in key_indices], dtype=np.float32)
        ones    = np.ones((len(raw_pts), 1), dtype=np.float32)
        rot_pts = (M @ np.hstack([raw_pts, ones]).T).T

        nose_y  = rot_pts[0, 1]
        sh_y    = rot_pts[1:3, 1].mean()
        hip_y   = rot_pts[3:5, 1].mean()
        body_h  = abs(hip_y - nose_y)
        x_min, x_max = rot_pts[:, 0].min(), rot_pts[:, 0].max()
        body_w  = x_max - x_min

        pad_y = body_h * self.padding;  pad_x = body_w * self.padding
        y1 = max(0, int(min(nose_y, sh_y) - pad_y))
        y2 = min(h, int(hip_y + pad_y))
        x1 = max(0, int(x_min - pad_x))
        x2 = min(w, int(x_max + pad_x))

        crop = rotated[y1:y2, x1:x2]
        return Image.fromarray(crop) if crop.size > 0 else None

    # ------------------------------------------------------------------
    def process(self, frame_bgr, fallback_full_frame=False):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            return Image.fromarray(frame_rgb) if fallback_full_frame else None

        lms    = results.pose_landmarks.landmark
        ls_vis = lms[_LEFT_SHOULDER].visibility
        rs_vis = lms[_RIGHT_SHOULDER].visibility
        if ls_vis < 0.3 and rs_vis < 0.3:
            return Image.fromarray(frame_rgb) if fallback_full_frame else None

        rotated, M = self._rotate_frame(frame_rgb, lms)
        crop = self._upper_body_crop(rotated, lms, M)
        if crop is None:
            return Image.fromarray(frame_rgb) if fallback_full_frame else None
        return crop

    def close(self):
        self._pose.close()


# ---------------------------------------------------------------------------
# Thread-local MediaPipe instance helper
# ---------------------------------------------------------------------------

def _get_thread_processor(detection_conf, tracking_conf, padding):
    """Return a per-thread GlobalFrameProcessor, creating it on first call."""
    if not hasattr(_thread_local, 'processor'):
        _thread_local.processor = GlobalFrameProcessor(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            upper_body_padding=padding,
        )
    return _thread_local.processor


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def read_video_frames_bgr(video_path: str) -> list:
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
# CPU worker: decode + MediaPipe for one video  (runs in thread pool)
# ---------------------------------------------------------------------------

def _prepare_crops(video_path, args):
    """
    Decode frames and run MediaPipe pose normalization on them.

    Returns
    -------
    crops       : list[PIL.Image]  ready for ViT inference
    n_discarded : int              frames dropped due to no-pose detection
    """
    processor = _get_thread_processor(
        args.min_detection_confidence,
        args.min_tracking_confidence,
        args.padding,
    )

    frames_bgr = read_video_frames_bgr(video_path)
    if not frames_bgr:
        return [], 0

    if args.stride > 1:
        frames_bgr = frames_bgr[::args.stride]

    crops, n_discarded = [], 0
    for frame in frames_bgr:
        if args.no_pose_norm:
            crops.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            roi = processor.process(frame, args.fallback_full_frame)
            if roi is not None:
                crops.append(roi)
            else:
                n_discarded += 1

    return crops, n_discarded


# ---------------------------------------------------------------------------
# ViT helper
# ---------------------------------------------------------------------------

def _extract_feats(reader, crops, batch_size):
    feats = []
    for j in range(0, len(crops), batch_size):
        feats.append(reader.get_feats(crops[j: j + batch_size]))
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser(
        description=(
            'Extract full-frame global ViT features (body-pose normalised, '
            'upper-body crop) from MCCSD videos. '
            'Uses a CPU thread pool to overlap MediaPipe with GPU ViT inference.'
        )
    )
    p.add_argument('--video_root', required=True)
    p.add_argument('--save_dir',   required=True)
    p.add_argument('--model_name', default='openai/clip-vit-large-patch14')
    p.add_argument('--cache_dir',  default=None)
    p.add_argument('--device',     default='cuda:0')
    p.add_argument('--s2_mode',    default='',
                   help='"s2wrapping" for multi-scale, or "" to disable')
    p.add_argument('--scales', nargs='+', type=int, default=[])
    p.add_argument('--batch_size', type=int, default=128,
                   help='Frames per ViT forward pass (default 128)')
    p.add_argument('--nth_layer',  type=int, default=-1)
    p.add_argument('--signers', nargs='+', default=['LF', 'HS', 'WT', 'XP'])
    p.add_argument('--padding', type=float, default=0.15,
                   help='BBox padding around upper-body (default 0.15)')
    p.add_argument('--stride', type=int, default=1,
                   help='Process every Nth frame (default 1)')
    p.add_argument('--num_workers', type=int, default=4,
                   help='CPU threads for parallel MediaPipe preprocessing (default 4)')
    p.add_argument('--prefetch', type=int, default=None,
                   help='Max futures in flight at once. Defaults to num_workers.')
    p.add_argument('--min_detection_confidence', type=float, default=0.5)
    p.add_argument('--min_tracking_confidence',  type=float, default=0.5)
    p.add_argument('--fallback_full_frame', action='store_true',
                   help='Use raw full frame when no body pose is detected')
    p.add_argument('--no_pose_norm', action='store_true',
                   help='Disable pose normalization (ablation)')
    return p


# ---------------------------------------------------------------------------
# Main loop  — producer-consumer pipeline
# ---------------------------------------------------------------------------

def process_mccsd_global(args):
    """
    CPU thread pool (producer)  ‖  GPU ViT inference (consumer).

    The thread pool pre-fetches `prefetch` videos ahead: while the GPU is
    running ViT on video k, CPU threads are decoding + running MediaPipe on
    videos k+1 … k+prefetch.  GPU utilisation stays near 100 %.
    """
    _model_name   = os.path.split(args.model_name)[-1]
    feat_dir_name = f'{_model_name}_global_feat_mccsd'

    reader = ViTFeatureReader(
        model_name=args.model_name, cache_dir=args.cache_dir,
        device=args.device, s2_mode=args.s2_mode,
        scales=args.scales, nth_layer=args.nth_layer,
    )

    prefetch = args.prefetch if args.prefetch else args.num_workers

    postfix = ''
    if args.s2_mode:
        postfix = f'_{args.s2_mode}'
    if len(args.scales) == 3:
        postfix += '_large'
    if args.stride > 1:
        postfix += f'_s{args.stride}'
    if args.no_pose_norm:
        postfix += '_raw'

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
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

            # Build the list of (video_path, save_path) pairs that still need work
            todo = []
            for vp in video_files:
                vid       = osp.splitext(osp.basename(vp))[0]
                save_path = osp.join(save_dir, f'{vid}{postfix}.npy')
                if not osp.exists(save_path):
                    todo.append((vp, save_path, vid))

            if not todo:
                print(f'[{signer}] All files already extracted, skipping.')
                continue

            # ── Producer-consumer with sliding window of futures ───────
            # Maintain a deque of (future, save_path, vid, video_path).
            # We keep at most `prefetch` futures in flight simultaneously;
            # this bounds peak memory while keeping the GPU fed.
            pending   = deque()  # (future, save_path, vid, video_path)
            todo_iter = iter(todo)
            skipped   = 0

            def _submit_next():
                """Submit the next video to the thread pool if any remain."""
                try:
                    vp, sp, vid_id = next(todo_iter)
                    f = executor.submit(_prepare_crops, vp, args)
                    pending.append((f, sp, vid_id, vp))
                except StopIteration:
                    pass

            # Seed the pipeline: pre-submit `prefetch` videos
            for _ in range(prefetch):
                _submit_next()

            pbar = tqdm.tqdm(total=len(todo), desc=f'[global][{signer}]')

            while pending:
                # Wait for the oldest future (FIFO order → outputs match inputs)
                future, save_path, vid, video_path = pending.popleft()

                # Immediately submit the next video so CPU stays busy
                _submit_next()

                # Retrieve CPU result (blocks only until this one future is ready)
                crops, n_discarded = future.result()

                if not crops:
                    if not osp.exists(save_path):  # genuine empty, not a resume skip
                        print(f'[WARN] No valid frames in {video_path}, skipping.')
                        skipped += 1
                    pbar.update(1)
                    continue

                if n_discarded > 0:
                    total = len(crops) + n_discarded
                    ratio = n_discarded / total
                    if ratio > 0.3:
                        print(
                            f'[INFO] {vid}: {n_discarded}/{total} frames '
                            f'({ratio:.0%}) had no pose detection.'
                        )

                # ── GPU: ViT inference ─────────────────────────────────
                feats = _extract_feats(reader, crops, args.batch_size)
                np.save(save_path, feats)

                pbar.update(1)

            pbar.close()

            if skipped:
                print(f'[{signer}] Skipped {skipped} videos with no valid frames.')


def main():
    parser = get_parser()
    args   = parser.parse_args()
    process_mccsd_global(args)


if __name__ == '__main__':
    main()
