import torch
from diffusers import DDIMScheduler

from models.stable_diffusion_baseline import FusionINVAttentionStableDiffusionPipeline
from models.unet_2d_condition import FreeUUNet2DConditionModel


def get_stable_diffusion_model() -> CrossImageAttentionStableDiffusionPipeline:
    print("Loading Stable Diffusion model...")
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = FusionINVAttentionStableDiffusionPipeline.from_pretrained("pretrained/stable-diffusion-v1-5",
                                                                      safety_checker=None).to(device)
    pipe.unet = FreeUUNet2DConditionModel.from_pretrained("pretrained/stable-diffusion-v1-5", subfolder="unet").to(device)
    pipe.scheduler = DDIMScheduler.from_config("pretrained/stable-diffusion-v1-5", subfolder="scheduler")

    print("Done.")
    return pipe
