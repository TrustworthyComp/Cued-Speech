"""
Extract Conformer encoder lip features using finetuned VSR checkpoint.

Usage:
  python scripts/conformer_extract_lip_feature.py \
      --checkpoint /home/uic/fengling/mccsd/auto_avsr-main/exp/mccsd_vsr_ft_0602_1707/last.ckpt \
      --crop-dir /home/uic/fengling/mccsd/fna_sra/vit_finetune_output/lip_crops \
      --save-dir /home/uic/fengling/mccsd/fna_sra/vit_finetune_output/conformer_lip_feat_mccsd \
      --speakers LF HS WT XP YX YZ \
      --device cuda:0
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'auto_avsr-main'))


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, x):
        return self.functional(x)


def get_eval_transform():
    return torch.nn.Sequential(
        FunctionalModule(lambda x: x / 255.0),
        torchvision.transforms.CenterCrop(88),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize(0.421, 0.165),
    )


def load_npy_and_preprocess(path, transform):
    frames_np = np.load(path)
    frames = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    frames = F.interpolate(frames, size=(88, 88), mode='bilinear', align_corners=False)
    frames = transform(frames)
    return frames


def build_model(odim=44):
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    model = E2E(odim=odim, modality='video', ctc_weight=0.3)
    return model


def load_finetuned_weights(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict']

    def strip_prefix(d, prefix='model.'):
        return {k.replace(prefix, ''): v for k, v in d.items() if k.startswith(prefix)}

    model_state = strip_prefix(state_dict, 'model.')

    frontend_keys = {k: v for k, v in model_state.items() if k.startswith('frontend.')}
    proj_keys = {k: v for k, v in model_state.items() if k.startswith('proj_encoder.')}
    encoder_keys = {k: v for k, v in model_state.items() if k.startswith('encoder.')}

    model.frontend.load_state_dict(
        {k.replace('frontend.', ''): v for k, v in frontend_keys.items()}
    )
    model.proj_encoder.load_state_dict(
        {k.replace('proj_encoder.', ''): v for k, v in proj_keys.items()}
    )
    model.encoder.load_state_dict(
        {k.replace('encoder.', ''): v for k, v in encoder_keys.items()}
    )

    print(f'Loaded frontend ({len(frontend_keys)} keys)')
    print(f'Loaded proj_encoder ({len(proj_keys)} keys)')
    print(f'Loaded encoder ({len(encoder_keys)} keys)')


def extract_features(model, frames, device):
    model.eval()
    with torch.inference_mode():
        x = frames.to(device)
        x = model.frontend(x.unsqueeze(0))
        x = model.proj_encoder(x)
        x, _ = model.encoder(x, None)
        x = x.squeeze(0).cpu()
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to finetuned .ckpt file')
    parser.add_argument('--crop-dir', required=True, help='Root dir of lip crop .npy files')
    parser.add_argument('--save-dir', required=True, help='Output dir for extracted features')
    parser.add_argument('--speakers', nargs='+', default=['LF', 'HS', 'WT', 'XP', 'YX', 'YZ'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--debug', action='store_true', help='Process only first 5 files per speaker')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = build_model()
    model.to(device)
    load_finetuned_weights(model, args.checkpoint)
    transform = get_eval_transform()

    total_files = 0
    for speaker in args.speakers:
        speaker_dir = os.path.join(args.crop_dir, speaker)
        if not os.path.isdir(speaker_dir):
            print(f'[SKIP] Speaker dir not found: {speaker_dir}')
            continue

        out_dir = os.path.join(args.save_dir, speaker)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('.npy')])
        if args.debug:
            files = files[:5]

        print(f'\n=== Speaker: {speaker} ({len(files)} files) ===')
        for i, fname in enumerate(files):
            in_path = os.path.join(speaker_dir, fname)
            out_path = os.path.join(out_dir, fname)

            if os.path.exists(out_path):
                continue

            try:
                frames = load_npy_and_preprocess(in_path, transform)
                feat = extract_features(model, frames, device)
                np.save(out_path, feat.numpy())
            except Exception as e:
                print(f'  [ERROR] {fname}: {e}')
                continue

            total_files += 1
            if (i + 1) % 50 == 0 or i == len(files) - 1:
                print(f'  [{i + 1}/{len(files)}] done, shape={feat.shape}')

    print(f'\nDone. Total files processed: {total_files}')
    print(f'Features saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
