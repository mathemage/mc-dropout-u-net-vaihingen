import torch
import logging

from torch import nn

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

logging.info(f"PyTorch version: {torch.__version__}")

# From https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html#torch-tensor-unfold
x = torch.arange(1., 8)
logging.info(f"x: {x}")
logging.info(f"x.unfold(0, 2, 1):\n {x.unfold(dimension=0, size=2, step=1)}")
logging.info(f"x.unfold(0, 2, 2):\n {x.unfold(dimension=0, size=2, step=2)}")

x_3d = torch.arange(24.)
x_3d = torch.reshape(x_3d, (2, 4, 3))
logging.info(f"x_3d: {x_3d}")
# logging.info(f"x_3d.unfold(1, 2, 1):\n {x_3d.unfold(dimension=1, size=2, step=1)}")

# From https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
unfold = nn.Unfold(kernel_size=(2, 3))
inp = torch.randn(2, 5, 3, 4)
logging.info(f"input: {inp}")
output = unfold(inp)
logging.info(f"output: {output}")
# each patch contains 30 values (2x3=6 vectors, each of 5 channels)
# 4 blocks (2x3 kernels) in total in the 3x4 input
logging.info(f"output.size(): {output.size()}")  # TODO understand

# From https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
unfold_3d = nn.Unfold(kernel_size=(2, 2))(x_3d)
logging.info(f"unfold_3d: {unfold_3d}")  # TODO check here
