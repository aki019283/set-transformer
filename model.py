import torch
from torch import nn
from set_block import AttentionBlock


def canonical_name(name):
    # Historical exports use both spellings for the same four family slots.
    return "Canlander_" + name.removeprefix("Canlendar_") if name.startswith("Canlendar_") else name

class CrossFeatures(nn.Module):
    """Parallel DCN-V2 with two rank-16 cross matrices."""
    def __init__(self, features):
        super().__init__()
        # 两次低秩变换先将 103 维压到 16 维，再映回因子空间，避免学习完整的方阵。
        self.cross = nn.ModuleList([
            nn.Sequential(nn.Linear(features, 16, bias=False), nn.Linear(16, features))
            for _ in range(2)
        ])
        self.deep = nn.Sequential(nn.Linear(features, 64), nn.ReLU(), nn.Dropout(.1),
                                  nn.Linear(64, 32), nn.ReLU())

    def forward(self, x):
        h = x
        for layer in self.cross:
            # 每层都与原始输入 x 逐元素相乘；残差 h 让低阶信号也能传到输出。
            h = h + x * layer(h)
        # 交叉分支保留 103 维，深层分支输出 32 维，沿特征轴拼接。
        return torch.cat((h, self.deep(x)), dim=-1)


class Hybrid(nn.Module):
    """Encode each named factor family once per stock, then pool the set."""
    families = ("calc", "basic", "defense", "Alpha101", "Canlander", "gf125", "GTJA", "TALib131")

    def __init__(self, names, width=16, inducing=4):
        super().__init__()
        groups = {name: [] for name in self.families}
        # 一只股票是一条样本。按列名归入因子族，记录其在输入面板中的真实列号。
        for i, name in enumerate(names):
            name = canonical_name(name)
            family = "basic" if name in {"Size", "Beta", "EP", "Turnover"} else name.split("_")[0]
            groups[family].append(i)
        self.encoders = nn.ModuleList()
        for family, indices in groups.items():
            if not indices:
                raise ValueError(f"Missing family: {family}")
            # 列号随模型保存、迁移设备；它是索引，不参与梯度更新。
            self.register_buffer("indices_" + family, torch.tensor(indices, dtype=torch.long))
            self.encoders.append(nn.Linear(len(indices), width))
        self.cross = CrossFeatures(len(names))
        self.inducing = nn.Parameter(torch.randn(1, inducing, width) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.gather = AttentionBlock(width, 2)
        self.broadcast = AttentionBlock(width, 2)
        self.pre_pool = nn.Sequential(nn.Linear(width, width), nn.ReLU())
        self.pool = AttentionBlock(width, 2)
        self.head = nn.Linear(len(names) + 32 + width, 1)

    def pool_tokens(self, tokens):
        # [B,4,16]：四个可学习查询从八个族向量中汇总信息。
        latent = self.gather(self.inducing.expand(len(tokens), -1, -1), tokens)
        # [B,8,16]：每个因子族再读取这四个摘要，得到包含其他族信息的表示。
        contextual = self.broadcast(tokens, latent)
        # 一个可学习的汇聚查询生成每只股票的 16 维集合摘要。
        return self.pool(self.query.expand(len(tokens), -1, -1), self.pre_pool(contextual))[:, 0]

    def forward(self, x):
        tokens = torch.stack([
            encoder(x.index_select(1, getattr(self, "indices_" + family)))
            for family, encoder in zip(self.families, self.encoders)
        ], dim=1)
        # 103 + 32 + 16 = 151 维联合表示，线性输出每只股票一个排序分数。
        return self.head(torch.cat((self.cross(x), self.pool_tokens(tokens)), dim=1)).squeeze(-1)


