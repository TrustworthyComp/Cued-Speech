"""
GKSampler — Gloss-K Sampler for cross-signer contrastive learning.

每个 batch 由 G 个 gloss × K 个 signer 的样本构成（batch_size = G * K）。
保证 batch 内每个 anchor 至少有 K-1 个同 gloss 不同 signer 的正样本，
以及来自同 signer 不同 gloss 的负样本，使 triplet/n-tuple 损失真正工作。

DDP 支持：
    GKSampler 内部感知 rank / world_size，每个 rank 只 yield 属于自己的
    那些 batch（以 world_size 为步长交错分配），batch 内 G×K 结构保持完整。
    使用时须在 lightning trainer 中设置 replace_sampler_ddp: false，
    避免 PL 再套一层 DistributedSampler。

用法（单卡）：
    sampler = GKSampler(dataset, G=4, K=4)
    DataLoader(dataset, batch_sampler=sampler, collate_fn=...)

用法（多卡 DDP，在 DataModule._train_dataloader 内）：
    rank       = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    sampler = GKSampler(dataset, G=4, K=4, rank=rank, world_size=world_size)
    DataLoader(dataset, batch_sampler=sampler, collate_fn=...)
"""

import random
from collections import defaultdict
from typing import Iterator, List, Optional

import numpy as np
from torch.utils.data import Sampler


class GKSampler(Sampler):
    """
    Gloss-K Sampler：每个 mini-batch 恰好包含 G 个 gloss，
    每个 gloss 随机选取 K 个不同 signer 的样本索引。

    支持 DDP：通过 rank / world_size 参数在多卡间分配 batch，
    每张卡只 yield 属于自己的 batch，batch 内 G×K 结构不被拆分。

    Args:
        dataset:        torch Dataset，其 .data 属性为 {int: entry_dict}，
                        每条 entry 含 'gloss' 和 'signer' 字段。
        G:              每个 batch 包含的 gloss 数量。
        K:              每个 gloss 选取的 signer 数量（须 ≤ 该 gloss 的 signer 总数）。
        drop_last:      若某个 epoch 最后不足一个完整 batch，是否丢弃。
        seed:           随机种子，None 表示不固定。
        rank:           当前进程的 rank（DDP 时传入，单卡留 None）。
        world_size:     总进程数（DDP 时传入，单卡留 None）。
    """

    def __init__(
        self,
        dataset,
        G: int = 4,
        K: int = 4,
        drop_last: bool = True,
        seed: int = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        super().__init__()
        self.G = G
        self.K = K
        self.drop_last = drop_last
        self.seed = seed

        # DDP 分片参数
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1

        # 构建 gloss -> signer -> [index, ...] 索引表
        self._gloss_signer_idx: dict = defaultdict(lambda: defaultdict(list))
        for idx, entry in dataset.data.items():
            self._gloss_signer_idx[entry['gloss']][entry['signer']].append(idx)

        # 只保留 signer 数量 >= K 的 gloss
        self._eligible_glosses: List[str] = [
            g for g, s_dict in self._gloss_signer_idx.items()
            if len(s_dict) >= K
        ]

        if len(self._eligible_glosses) < G:
            raise ValueError(
                f"GKSampler: 只有 {len(self._eligible_glosses)} 个 gloss 拥有 ≥{K} 个 signer，"
                f"但 G={G}。请减小 G 或 K。"
            )

        # 全局 batch 总数（所有 rank 合计）
        n_batches_global = len(self._eligible_glosses) // G
        # 每个 rank 负责的 batch 数（向下取整，保证均匀）
        self._n_batches_per_rank = n_batches_global // self.world_size

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """返回当前 rank 每个 epoch yield 的 batch 数。"""
        return self._n_batches_per_rank

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)

        glosses = self._eligible_glosses.copy()
        rng.shuffle(glosses)

        n_batches_global = len(glosses) // self.G

        # 生成全部 batch，再按 rank 交错取子集
        all_batches: List[List[int]] = []
        for b in range(n_batches_global):
            batch_glosses = glosses[b * self.G: b * self.G + self.G]

            batch_indices: List[int] = []
            for gloss in batch_glosses:
                s_dict = self._gloss_signer_idx[gloss]
                chosen_signers = rng.sample(list(s_dict.keys()), self.K)
                for signer in chosen_signers:
                    idx = rng.choice(s_dict[signer])
                    batch_indices.append(idx)

            all_batches.append(batch_indices)

        # 当前 rank 取下标为 rank, rank+world_size, rank+2*world_size, ... 的 batch
        for b_idx in range(self.rank, len(all_batches), self.world_size):
            yield all_batches[b_idx]
