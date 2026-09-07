"""
MCCSD VSR Fine-tuning with ResNet18 + Conformer backbone.

Uses the auto_avsr framework (E2E model) with a pretrained checkpoint
(vsr_trlrs3_base.pth), fine-tuning via transfer-encoder mode.

Usage:
  python scripts/train_mccsd_vsr.py \
      --exp-dir ./exp \
      --exp-name mccsd_vsr_ft \
      --root-dir ../auto_avsr-main/mccsd_data \
      --pretrained-model-path ../vsr_trlrs3_base.pth \
      --transfer-encoder \
      --gpus 1 \
      --num-nodes 1 \
      --max-epochs 20 \
      --max-frames 1600 \
      --lr 1e-3 \
      --warmup-epochs 3
"""

import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'auto_avsr-main'))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from mccsd.lightning import MCCSDModelModule
from mccsd.data_module import MCCSDDataModule

torch.backends.cudnn.enabled = False

CROP_DIR = '/home/uic/fengling/mccsd/fna_sra/vit_finetune_output/lip_crops'
DATA_DIR = '/home/uic/fengling/mccsd/auto_avsr-main/mccsd_data'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp-dir', default='./exp', required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--root-dir', default=DATA_DIR)
    parser.add_argument('--crop-dir', default=CROP_DIR)
    parser.add_argument('--train-file', default='train.csv')
    parser.add_argument('--test-file', default='test.csv')
    parser.add_argument('--num-nodes', default=1, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--pretrained-model-path', type=str, default=None)
    parser.add_argument('--transfer-encoder', action='store_true')
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--max-epochs', default=20, type=int)
    parser.add_argument('--max-frames', type=int, default=1600)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.03)
    parser.add_argument('--ctc-weight', type=float, default=0.3)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--train-num-buckets', type=int, default=400)
    parser.add_argument('--decode-snr-target', type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    model = MCCSDModelModule(args)

    dm = MCCSDDataModule(
        crop_dir=args.crop_dir,
        train_csv=os.path.join(args.root_dir, args.train_file),
        test_csv=os.path.join(args.root_dir, args.test_file),
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        train_num_buckets=args.train_num_buckets,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name),
        monitor='monitoring_step',
        mode='max',
        save_last=True,
        filename='{epoch}',
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=False) if args.gpus > 1 else 'auto',
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=0,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model=model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    main()
