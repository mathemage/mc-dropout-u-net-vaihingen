import torch

print(f"PyTorch version: {torch.__version__}")

# From https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html#torch-tensor-unfold
x = torch.arange(1., 8)
print(f"x: {x}")
print(f"x.unfold(0, 2, 1):\n {x.unfold(0, 2, 1)}")
print(f"x.unfold(0, 2, 2):\n {x.unfold(0, 2, 2)}")