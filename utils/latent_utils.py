from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from diffusers.training_utils import set_seed

from AllinVIS import AllinVISModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion_vis import invert
from utils.ddpm_inversion_inf2vis import invertinf


def load_latents_or_invert_images(model: AllinVISModel, cfg: RunConfig):
    if cfg.load_latents and cfg.vis_latent_save_path.exists() and cfg.ir_latent_save_path.exists():
        print("Loading existing latents...")
        latents_vis, latents_ir = load_latents(cfg.vis_latent_save_path, cfg.ir_latent_save_path)
        noise_vis, noise_ir = load_noise(cfg.vis_latent_save_path, cfg.ir_latent_save_path)
        print("Done.")
    else:
        print("Inverting images...")
        vis_image, ir_image = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents_vis, latents_ir, noise_vis, noise_ir = invert_images(vis_image=vis_image,
                                                                             ir_image=ir_image,
                                                                             sd_model=model.pipe,
                                                                             cfg=cfg)
        model.enable_edit = True
        print("Done.")
    return latents_vis, latents_ir, noise_vis, noise_ir


def load_latents(vis_latent_save_path: Path, ir_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_vis = torch.load(vis_latent_save_path)
    latents_ir = torch.load(ir_latent_save_path)
    if type(latents_ir) == list:
        latents_vis = [l.to("cuda") for l in latents_vis]
        latents_ir = [l.to("cuda") for l in latents_ir]
    else:
        latents_vis = latents_vis.to("cuda")
        latents_ir = latents_ir.to("cuda")
    return latents_vis, latents_ir


def load_noise(vis_latent_save_path: Path, ir_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_vis = torch.load(vis_latent_save_path.parent / (vis_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_ir = torch.load(ir_latent_save_path.parent / (ir_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_vis = latents_vis.to("cuda")
    latents_ir = latents_ir.to("cuda")
    return latents_vis, latents_ir


def invert_images(sd_model: AllinVISModel, vis_image: Image.Image, ir_image: Image.Image, cfg: RunConfig):
    input_vis = torch.from_numpy(np.array(vis_image)).float() / 127.5 - 1.0
    input_ir = torch.from_numpy(np.array(ir_image)).float() / 127.5 - 1.0
    set_seed(cfg.seed)
    zs_vis, latents_vis, directions = invert(x0=input_vis.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                 pipe=sd_model,
                                 # prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5)

    set_seed(cfg.seed)
    direction_step_size = float(cfg.direction_step_size)
    zs_ir, latents_ir = invertinf(x0=input_ir.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                  vis_direction=directions,
                                  direction_step_size= direction_step_size,
                                       pipe=sd_model,
                                       # prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5)
    # Save the inverted latents and noises
    # torch.save(latents_vis, cfg.latents_path / f"{cfg.vis_image_path.stem}_vis.pt")
    # torch.save(latents_ir, cfg.latents_path / f"{cfg.ir_image_path.stem}_ir.pt")
    # torch.save(zs_vis, cfg.latents_path / f"{cfg.vis_image_path.stem}_vis_ddpm_noise.pt")
    # torch.save(zs_ir, cfg.latents_path / f"{cfg.ir_image_path.stem}_ir_ddpm_noise.pt")
    return latents_vis, latents_ir, zs_vis, zs_ir


def get_init_latents_and_noises(model: AllinVISModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_ir.dim() == 4 and model.latents_vis.dim() == 4 and model.latents_vis.shape[0] > 1:
        model.latents_ir = model.latents_ir[cfg.skip_steps]
        model.latents_vis = model.latents_vis[cfg.skip_steps]
    init_latents = torch.stack([model.latents_vis, model.latents_vis, model.latents_ir])
    zs_fusion = model.zs_vis.clone()
    init_zs = [ zs_fusion[cfg.skip_steps:], model.zs_vis[cfg.skip_steps:], model.zs_ir[cfg.skip_steps:]]
    return init_latents, init_zs
