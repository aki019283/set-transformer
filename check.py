import json
from pathlib import Path

import torch
from model import Hybrid


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(42)
    names = json.loads((Path(__file__).parent/"reference/features.json").read_text())
    model = Hybrid(names).eval()
    x = torch.randn(5, len(names))
    score = model(x)
    assert score.shape == (5,)
    torch.testing.assert_close(score[:1], model(x[:1]))
    tokens = torch.randn(5, 8, 16)
    torch.testing.assert_close(model.pool_tokens(tokens), model.pool_tokens(tokens.flip(1)))
    score.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    for branch in (model.cross.cross, model.cross.deep, model.encoders, model.gather, model.broadcast, model.pool):
        assert sum(p.grad.abs().sum().item() for p in branch.parameters()) > 0
    print("Hybrid forward, backward, stock isolation and token pooling checks passed.")
