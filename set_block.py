from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, query, context):
        # query 决定输出位置，context 同时提供键和值。注意力始终在单只股票内部计算。
        attended, _ = self.attention(query, context, context, need_weights=False)
        # 注意力残差与前馈残差分别做层归一化，保留查询自身的信息。
        hidden = self.norm1(query + attended)
        return self.norm2(hidden + self.ff(hidden))


