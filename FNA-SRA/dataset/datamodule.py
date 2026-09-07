import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.helpers import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        num_workers=None,
        gk_sampler: dict = None,
    ):
        """
        Args:
            batch_size:   验证/测试 DataLoader 的 batch size；
                          当 gk_sampler 启用时，训练 batch size 由 G*K 决定。
            gk_sampler:   若不为 None，则为传给 GKSampler 的参数字典，
                          支持键：G（gloss 数）、K（signer 数）、drop_last、seed。
                          启用后训练集使用 batch_sampler=GKSampler，
                          忽略 batch_size 和 shuffle 参数。
                          示例：{G: 4, K: 4}
        """
        super().__init__()

        self.batch_size = batch_size
        self.gk_sampler_cfg = gk_sampler  # None 表示不启用
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['valid'] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = self._test_dataloader

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )

    def _train_dataloader(self):
        train_ds = self.datasets['train']
        if self.gk_sampler_cfg is not None:
            from dataset.sampler import GKSampler
            # DDP: 让 GKSampler 内部做 rank 分片，PL 不再套 DistributedSampler
            rank, world_size = None, None
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            sampler = GKSampler(
                train_ds,
                **self.gk_sampler_cfg,
                rank=rank,
                world_size=world_size,
            )
            return DataLoader(
                dataset=train_ds,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=train_ds.collate_fn,
            )
        return DataLoader(
            dataset=train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=train_ds.collate_fn,
        )

    def _val_dataloader(self):
        return DataLoader(
            dataset=self.datasets['valid'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=BaseFeeder.collate_fn if self.use_collate else None,
            collate_fn=self.datasets['valid'].collate_fn
        )

    def _test_dataloader(self):
        return DataLoader(
            dataset=self.datasets['test'], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            # collate_fn=BaseFeeder.collate_fn if self.use_collate else None
            collate_fn=self.datasets['test'].collate_fn
        ) 