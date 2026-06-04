import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.network import Network


TILE_SIZE = 256
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_config(config_path):
    content = ''
    with open(config_path) as f:
        for line in f:
            content += line.split('//')[0] + '\n'
    return json.loads(content, object_pairs_hook=OrderedDict)


def build_network(opt):
    net_args = opt['model']['which_networks'][0]['args']
    return Network(**net_args)


def load_checkpoint(net, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state, strict=False)
    return net


def to_tensor(img_pil, device):
    """PIL RGB → (1, 3, H, W) float32 tensor in [-1, 1]"""
    arr = np.array(img_pil).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)


def to_uint8(tensor):
    """(1, 3, H, W) tensor in [-1, 1] → (H, W, 3) uint8 numpy"""
    arr = tensor.squeeze(0).cpu().float().numpy().transpose(1, 2, 0)
    return ((arr + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def tile_and_infer(net, cond_tensor, device):
    """
    Split a (1, 3, H, W) conditioning tensor into TILE_SIZE x TILE_SIZE tiles,
    run restoration on each, and stitch back into (1, 3, H, W).
    H and W must be multiples of TILE_SIZE.
    """
    _, _, H, W = cond_tensor.shape
    assert H % TILE_SIZE == 0 and W % TILE_SIZE == 0, \
        f"Image size {H}x{W} must be divisible by {TILE_SIZE}"

    output = torch.zeros_like(cond_tensor)
    rows, cols = H // TILE_SIZE, W // TILE_SIZE

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            x0, x1 = c * TILE_SIZE, (c + 1) * TILE_SIZE
            tile = cond_tensor[:, :, y0:y1, x0:x1]
            pred, _ = net.restoration(tile, sample_num=1)
            output[:, :, y0:y1, x0:x1] = pred

    return output


def main():
    parser = argparse.ArgumentParser(description='Tiling inference for Palette virtual staining.')
    parser.add_argument('--config', required=True, help='Path to finetune config JSON')
    parser.add_argument('--checkpoint', default=None,
                        help='Full path to the .pth checkpoint file, e.g. .../checkpoint/best_Network_ema.pth. '
                             'If omitted, reads path.test_checkpoint from the config.')
    parser.add_argument('--cond_dir', default=None,
                        help='H&E test image folder (overrides config test cond_dir)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save 1024x1024 predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    opt = load_config(args.config)

    checkpoint = args.checkpoint or opt['path'].get('test_checkpoint')
    if not checkpoint:
        raise ValueError("No checkpoint provided. Pass --checkpoint or set path.test_checkpoint in the config.")
    print(f"Checkpoint : {checkpoint}")

    cond_dir = args.cond_dir or opt['datasets']['test']['which_dataset']['args']['cond_dir']
    print(f"H&E source : {cond_dir}")
    print(f"Output dir : {args.output_dir}")

    net = build_network(opt)
    net = load_checkpoint(net, checkpoint, device)
    net = net.to(device)
    net.eval()
    net.set_new_noise_schedule(device=device, phase='test')

    os.makedirs(args.output_dir, exist_ok=True)

    filenames = sorted(f for f in os.listdir(cond_dir) if Path(f).suffix.lower() in IMG_EXTENSIONS)
    print(f"Images found: {len(filenames)}")

    with torch.no_grad():
        for fname in tqdm(filenames, desc='Tiling inference'):
            cond_pil = Image.open(os.path.join(cond_dir, fname)).convert('RGB')
            cond_tensor = to_tensor(cond_pil, device)

            pred_tensor = tile_and_infer(net, cond_tensor, device)

            pred_np = to_uint8(pred_tensor)
            out_path = os.path.join(args.output_dir, Path(fname).stem + '.png')
            Image.fromarray(pred_np).save(out_path)

    print(f"\nDone. {len(filenames)} predictions saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
