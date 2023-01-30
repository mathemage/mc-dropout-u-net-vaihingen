import argparse
import logging
from pathlib import Path

import torch

import wandb

wandb.init(project="MCD-U-Net-Vaihingen-test", entity="mathemage")

from torch.utils.data import DataLoader

from utils.data_loading import BasicDataset, CarvanaDataset
from evaluate_on_vaihingen import evaluate
from unet import UNet

test_dir_img = Path('./data/vaihingen/testset/imgs/')
test_dir_mask = Path('./data/vaihingen/testset/masks/')


def test_net(net,
             device,
             batch_size: int = 1,
             img_scale: float = 0.5,
             amp: bool = False):
    # 1. Create dataset
    try:
        test_set = CarvanaDataset(test_dir_img, test_dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        test_set = BasicDataset(test_dir_img, test_dir_mask, img_scale)
    n_test = len(test_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(batch_size=batch_size, img_scale=img_scale, amp=amp))

    logging.info(f'''Starting testing:
        Batch size:      {batch_size}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Begin testing
    histograms = {}
    for tag, value in net.named_parameters():
        tag = tag.replace('/', '.')
        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())

    test_score = evaluate(net, test_loader, device)

    logging.info('Test Dice score: {}'.format(test_score))
    experiment.log({
        'test Dice': test_score,
        **histograms
    })


def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        test_net(net=net,
                 batch_size=args.batch_size,
                 device=device,
                 img_scale=args.scale,
                 amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
