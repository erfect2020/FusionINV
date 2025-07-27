import pathlib
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import RunConfig


def load_images(cfg: RunConfig, save_path: Optional[pathlib.Path] = None) -> Tuple[Image.Image, Image.Image]:
    # image_vis = load_size(cfg.vis_image_path)
    # image_ir = load_size(cfg.ir_image_path)

    image_vis = load_size_src(cfg.vis_image_path)
    image_ir = load_size_src(cfg.ir_image_path)
    if save_path is not None:
        Image.fromarray(image_vis).save(save_path / f"in_vis.png")
        Image.fromarray(image_ir).save(save_path / f"in_ir.png")
    return image_vis, image_ir


def load_size_src(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    image = np.array(Image.fromarray(image).resize((672,528)))
    # image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def load_size(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((size, size)))
    return image


def save_generated_masks(model, cfg: RunConfig):
    tensor2im(model.image_vis_mask_32).save(cfg.output_path / f"mask_vis_32.png")
    tensor2im(model.image_ir_mask_32).save(cfg.output_path / f"mask_ir_32.png")
    tensor2im(model.image_vis_mask_64).save(cfg.output_path / f"mask_vis_64.png")
    tensor2im(model.image_ir_mask_64).save(cfg.output_path / f"mask_ir_64.png")


def tensor2im(x) -> Image.Image:
    return Image.fromarray(x.cpu().numpy().astype(np.uint8) * 255)