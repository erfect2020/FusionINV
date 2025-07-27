import abc

import torch
from torch import inference_mode
from tqdm import tqdm
from torch.special import expm1
import math
import numpy as np

"""
Inversion code taken from: 
1. The official implementation of Edit-Friendly DDPM Inversion: https://github.com/inbarhub/DDPM_inversion
2. The LEDITS demo: https://huggingface.co/spaces/editing-images/ledits/tree/main
"""

LOW_RESOURCE = True


def invertinf(x0, pipe, vis_direction, direction_step_size, prompt_src="", num_diffusion_steps=100, cfg_scale_src=3.5, eta=1):
    #  inverts a real image according to Algorithm 1 in https://arxiv.org/pdf/2304.06140.pdf,
    #  based on the code in https://github.com/inbarhub/DDPM_inversion
    #  returns wt, zs, wts:
    #  wt - inverted latent
    #  wts - intermediate inverted latents
    #  zs - noise maps
    pipe.scheduler.set_timesteps(num_diffusion_steps)
    with inference_mode():
        w0 = (pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()
    wt, zs, wts = inversion_forward_process(pipe, w0, vis_direction, direction_step_size, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src,
                                            prog_bar=True, num_inference_steps=num_diffusion_steps)
    return zs, wts


def inversion_forward_process(model, x0, vis_direction, direction_step_size,
                              etas=None,
                              prog_bar=False,
                              prompt="",
                              cfg_scale=3.5,
                              num_inference_steps=50, eps=None
                              ):
    if not prompt == "":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    # variance_noise_shape = (
    #     num_inference_steps,
    #     model.unet.in_channels,
    #     model.unet.sample_size,
    #     model.unet.sample_size)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        x0.shape[2],
        x0.shape[3])
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas] * model.scheduler.num_inference_steps
        # xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        xts, xt_noises = ddpm_sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        # xts = ddpm_sample_xts_from_xts1(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)


    # --- Dynamic direction_step_size setup ---
    initial_direction_step_size = direction_step_size # Store original
    if len(timesteps) > 0:
        # max_t_val is the largest t (e.g., 999), min_t_val is smallest (e.g., 1 or 0)
        # timesteps are already sorted from high to low by `reversed(timesteps)` for the loop
        # but for normalization, we need the original range.
        # `model.scheduler.timesteps` are usually [T, T-step, ..., t_small]
        min_t_val = float(timesteps[-1]) # Smallest t in the current schedule
        max_t_val = float(timesteps[0])  # Largest t in the current schedule
    else: # Fallback if timesteps is empty
        max_t_val = float(model.scheduler.config.num_train_timesteps -1)
        min_t_val = 0.0
    
    # For Cosine Annealing
    max_cosine_step_size = initial_direction_step_size
    min_cosine_step_size = 0.0 # Or a small fraction like initial_direction_step_size * 0.1


    for t in op:
        idx = t_to_idx[int(t)]
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx][None]

        with torch.no_grad():
            out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
            if not prompt == "":
                cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)

        if not prompt == "":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            xtm1 = xts[idx + 1][None]
            # pred of x0
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[
                prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (0.5) * noise_pred

            if idx > 0:
                variance_pred_noise = xt_noises[idx + 1] * direction_step_size
                pred_sample_vis_direction = vis_direction[idx] * direction_step_size
            else:
                pred_sample_vis_direction = 0
                variance_pred_noise = 0

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction  + pred_sample_vis_direction

            z = (xtm1 - mu_xt + variance_pred_noise) / (etas[idx] * variance ** 0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx + 1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    return xt, zs, xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size)

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    xts = torch.cat([xts, x0], dim=0)

    return xts


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def q_sample( x_start, times, noise = None):
    noise =  torch.randn_like(x_start)
    log_snr = alpha_cosine_log_snr(times)
    log_snr_padded = right_pad_dims_to(x_start, log_snr)
    alpha, sigma = torch.sqrt(log_snr_padded.sigmoid()), torch.sqrt((-log_snr_padded).sigmoid())
    x_noised =  x_start * alpha + noise * sigma
    return x_noised

def add_noise(
        scheduler,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
) -> torch.Tensor:
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device=original_samples.device)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples, sqrt_one_minus_alpha_prod * noise


def ddpm_sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    # variance_noise_shape = (
    #     num_inference_steps,
    #     model.unet.in_channels,
    #     model.unet.sample_size,
    #     model.unet.sample_size)

    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        x0.shape[2],
        x0.shape[3])

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    xt_noises = torch.zeros(variance_noise_shape).to(x0.device)

    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        noise = torch.randn(x0.shape, device=x0.device)
        noisy_samples, noisy_term = add_noise(model.scheduler, x0, noise, t)
        xts[idx] = noisy_samples
        xt_noises[idx] = noisy_term

    xts = torch.cat([xts, x0], dim=0)
    xt_noises = torch.cat([xt_noises, torch.zeros_like(x0, device=x0.device)], dim=0)

    return xts, xt_noises


def ddpm_sample_xts_from_xts1(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size)

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    idx_to_t = {int(k): v for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    alphas_cumprod = model.scheduler.alphas_cumprod.to(dtype=x0.dtype)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xt_minus_one = xts[idx+1] if idx < len(timesteps)-1 else x0
        margin_t = idx_to_t[idx+1] if idx < len(timesteps)-1 else 0

        # one_minus_betat = alphas_cumprod[t + 1] / alphas_cumprod[margin_t]
        # betat = 1 - one_minus_betat
        # noise = torch.randn(x0.shape, device=x0.device)
        # xt = xt_minus_one * one_minus_betat ** 0.5 + betat ** 0.5 * noise
        # xt_minus_one = xt

        for current_t in range(margin_t,t):
            one_minus_betat = alphas_cumprod[current_t+1] / alphas_cumprod[current_t]
            betat = 1 - one_minus_betat
            noise = torch.randn(x0.shape, device=x0.device)
            xt = xt_minus_one * one_minus_betat ** 0.5 + betat ** 0.5 * noise
            xt_minus_one = xt
        xts[idx] = xt
    xts = torch.cat([xts, x0], dim=0)

    return xts


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                            model_output,
                                            torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
