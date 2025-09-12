from typing import List, Optional, Callable

import torch
import torch.nn.functional as F

from config import RunConfig
from constants import OUT_INDEX, IR_INDEX, VIS_INDEX
from models.stable_diffusion import FusionINVAttentionStableDiffusionPipeline
from utils import attention_utils
from utils.fusion_utils import maskedfusionin, fusion_in, adain, fusiondetails_in, maskedadain
from utils.model_utils import get_stable_diffusion_model
from utils.segmentation import Segmentor


# def null_optimization(self, latents, num_inner_steps, epsilon):
#     uncond_embeddings, cond_embeddings = self.context.chunk(2)
#     uncond_embeddings_list = []
#     latent_cur = latents[-1]
#     bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
#     for i in range(NUM_DDIM_STEPS):
#         uncond_embeddings = uncond_embeddings.clone().detach()
#         uncond_embeddings.requires_grad = True
#         optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
#         latent_prev = latents[len(latents) - i - 2]
#         t = self.model.scheduler.timesteps[i]
#         with torch.no_grad():
#             noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
#         for j in range(num_inner_steps):
#             noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
#             noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
#             latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
#             loss = nnf.mse_loss(latents_prev_rec, latent_prev)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loss_item = loss.item()
#             bar.update()
#             if loss_item < epsilon + i * 2e-5:
#                 break
#         for j in range(j + 1, num_inner_steps):
#             bar.update()
#         uncond_embeddings_list.append(uncond_embeddings[:1].detach())
#         with torch.no_grad():
#             context = torch.cat([uncond_embeddings, cond_embeddings])
#             latent_cur = self.get_noise_pred(latent_cur, t, False, context)
#     bar.close()
#     return uncond_embeddings_list


class AllinVISModel:

    def __init__(self, config: RunConfig, pipe: Optional[CrossImageAttentionStableDiffusionPipeline] = None):
        self.config = config
        self.pipe = get_stable_diffusion_model() if pipe is None else pipe
        self.register_attention_control()
        self.segmentor = Segmentor(prompt=config.prompt, object_nouns=[config.object_noun])
        self.latents_vis, self.latents_ir = None, None
        self.zs_vis, self.zs_ir = None, None
        self.image_vis_mask_32, self.image_vis_mask_64 = None, None
        self.image_ir_mask_32, self.image_ir_mask_64 = None, None
        self.enable_edit = False
        self.step = 0

    # get_adain_callback.attn_weight = None
    def set_latents(self, latents_vis: torch.Tensor, latents_ir: torch.Tensor):
        self.latents_vis = latents_vis
        self.latents_ir = latents_ir

    def set_noise(self, zs_vis: torch.Tensor, zs_ir: torch.Tensor):
        self.zs_vis = zs_vis
        self.zs_ir = zs_ir

    def set_masks(self, masks: List[torch.Tensor]):
        self.image_vis_mask_32, self.image_ir_mask_32, self.image_vis_mask_64, self.image_ir_mask_64 = masks

    def get_adain_callback(self):

        def callback(st: int, timestep: int, latents: torch.FloatTensor) -> Callable:
            self.step = st
            # Compute the masks using prompt mixing self-segmentation and use the masks for AdaIN operation
            if self.config.use_masked_adain and self.step == self.config.adain_range.start:
                masks = self.segmentor.get_object_masks()
                self.set_masks(masks)
            # Apply AdaIN operation using the computed masks
            if self.config.adain_range.start <= self.step < self.config.adain_range.end:
                if self.config.use_masked_adain:
                    latents[0] = maskedadain(latents[0], latents[1], self.image_ir_mask_64, self.image_vis_mask_64)
                    # latents[0] = maskedfusionin(latents[0], latents[1], self.image_ir_mask_64, self.image_vis_mask_64)
                else:
                    # latents[0] = adain(latents[0], latents[1])
                    # latents[2] = fusion_in(latents[2], latents[0])
                    latents[0] = adain(latents[0], latents[1])
                    # print("latens atte ", latents[0].shape, AllinVISModel.get_adain_callback.attn_weight[2].shape)
                    # latents[0] = fusiondetails_in(latents[0], latents[2], AllinVISModel.get_adain_callback.attn_weight[2])

        return callback

    def register_attention_control(self):

        model_self = self

        class AttentionProcessor:

            def __init__(self, place_in_unet: str):
                self.place_in_unet = place_in_unet
                if not hasattr(F, "scaled_dot_product_attention"):
                    raise ImportError("AttnProcessor2_0 requires torch 2.0, to use it, please upgrade torch to 2.0.")

            def __call__(self,
                         attn,
                         hidden_states: torch.Tensor,
                         encoder_hidden_states: Optional[torch.Tensor] = None,
                         attention_mask=None,
                         temb=None,
                         perform_swap: bool = False):

                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                is_cross = encoder_hidden_states is not None
                if not is_cross:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                should_mix = False
                should_ens = True
                config_contrast_strength = model_self.config.contrast_strength

                # Potentially apply our cross image attention operation
                # To do so, we need to be in a self-attention alyer in the decoder part of the denoising network
                if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
                    # print("swap is wasp",  hidden_states.shape, attention_utils.should_mix_keys_and_values_with_msrs(model_self, hidden_states))
                    # if attention_utils.should_mix_keys_and_values_with_fmb(model_self, hidden_states):
                    if attention_utils.should_mix_keys_and_values(model_self, hidden_states):
                        should_mix = True
                        # print("model self step", model_self.step)
                        # if (model_self.step % 2 == 0 and model_self.step < 70) or model_self.step < 65:
                        # if model_self.step < 40:
                        if model_self.step < 40 or (model_self.step % 2 == 0 and model_self.step < 60):
                            # Inject the structure's keys and values
                            # key[OUT_INDEX] = (key[OUT_INDEX] + key[IR_INDEX])/2
                            # value[OUT_INDEX] = (value[OUT_INDEX] + value[IR_INDEX])/2
                            key[OUT_INDEX] = key[IR_INDEX]
                            value[OUT_INDEX] = value[IR_INDEX]
                            # config_contrast_strength = 1.67
                            # if model_self.step < 70:
                            #     query[OUT_INDEX] = query[IR_INDEX] * 1.0
                                # config_contrast_strength = 1.67
                            # print("inf inject", model_self.step)
                            # should_ens = True
                             #* 10
                        else:
                            # pass
                            if model_self.step < 70:
                                # key[OUT_INDEX] = (key[OUT_INDEX] + key[VIS_INDEX])/2
                                # value[OUT_INDEX] = (value[OUT_INDEX] + value[VIS_INDEX])/2
                                key[OUT_INDEX] = key[VIS_INDEX]
                                value[OUT_INDEX] = value[VIS_INDEX]
                                # config_contrast_strength = 0.1
                                # query[OUT_INDEX] = query[VIS_INDEX]
                        # else:
                            # Inject the appearance's keys and values
                            # key[OUT_INDEX] = key[VIS_INDEX]
                            # value[OUT_INDEX] = value[VIS_INDEX]

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Compute the cross attention and apply our contrasting operation
                hidden_states, attn_weight = attention_utils.compute_scaled_dot_product_attention(
                    query, key, value,
                    edit_map=perform_swap and model_self.enable_edit and should_mix and should_ens,
                    is_cross=is_cross,
                    contrast_strength= config_contrast_strength,
                    mask=attention_mask
                )

                # if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit and should_mix:
                #     hidden_states[OUT_INDEX] -= 0.2 * hidden_states[IR_INDEX]

                # Update attention map for segmentation
                if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range.start - 1:
                    model_self.segmentor.update_attention(attn_weight, is_cross)

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor


                return hidden_states

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'ResnetBlock2D':
                pass
            if net_.__class__.__name__ == 'Attention':
                net_.set_processor(AttentionProcessor(place_in_unet + f"_{count + 1}"))
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0 # the belowed code need more review
        sub_nets = self.pipe.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
