import torch
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


logging.info(f"PyTorch version: {torch.__version__}")

# From https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html#torch-tensor-unfold
x = torch.arange(1., 8)
logging.info(f"x: {x}")
logging.info(f"x.unfold(0, 2, 1):\n {x.unfold(0, 2, 1)}")
logging.info(f"x.unfold(0, 2, 2):\n {x.unfold(0, 2, 2)}")
