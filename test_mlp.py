import torch

from fastvideo.v1.layers.mlp import MLP

layer = MLP(input_dim=1024, mlp_hidden_dim=4096, output_dim=1024)

x = torch.randn(1, 1024)
y1 = layer(x)
torch.compiler.set_stance("force_eager")
y2 = layer(x)
# assert_close(y1, y2)
max_diff = torch.max(torch.abs(y1 - y2))
mean_diff = torch.mean(torch.abs(y1 - y2))
print(f"Max diff: {max_diff}, mean diff: {mean_diff}")
