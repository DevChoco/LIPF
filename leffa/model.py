import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler

from leffa.diffusion_model.unet_ref import (
    UNet2DConditionModel as ReferenceUNet,
)
from leffa.diffusion_model.unet_gen import (
    UNet2DConditionModel as GenerativeUNet,
)

from collections import OrderedDict  # Add this at the top
import safetensors  # safetensors 지원 추가
import safetensors.torch

logger: logging.Logger = logging.getLogger(__name__)

class LeffaModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 12,  # noisy_image: 4, mask: 1, masked_image: 4, densepose: 3
        height: int = 1024,
        width: int = 768,
        dtype: str = "float16",
    ):
        super().__init__()

        self.height = height
        self.width = width

        self.build_models(
            pretrained_model_name_or_path,
            pretrained_model,
            new_in_channels,
        )

        if dtype == "float16":
            self.half()

    def build_models(
        self,
        pretrained_model_name_or_path: str = "",
        pretrained_model: str = "",
        new_in_channels: int = 12,
    ):
        diffusion_model_type = ""
        if "stable-diffusion-inpainting" in pretrained_model_name_or_path:
            diffusion_model_type = "sd15"
        elif "stable-diffusion-xl-1.0-inpainting-0.1" in pretrained_model_name_or_path:
            diffusion_model_type = "sdxl"

        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            rescale_betas_zero_snr=False if diffusion_model_type == "sd15" else True,
        )
        # VAE
        vae_config, vae_kwargs = AutoencoderKL.load_config(
            pretrained_model_name_or_path,
            subfolder="vae",
            return_unused_kwargs=True,
        )
        self.vae = AutoencoderKL.from_config(vae_config, **vae_kwargs)
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        # Reference UNet
        unet_config, unet_kwargs = ReferenceUNet.load_config(
            pretrained_model_name_or_path,
            subfolder="unet",
            return_unused_kwargs=True,
        )
        self.unet_encoder = ReferenceUNet.from_config(
            unet_config, **unet_kwargs)
        self.unet_encoder.config.addition_embed_type = None
        # Generative UNet
        unet_config, unet_kwargs = GenerativeUNet.load_config(
            pretrained_model_name_or_path,
            subfolder="unet",
            return_unused_kwargs=True,
        )
        self.unet = GenerativeUNet.from_config(unet_config, **unet_kwargs)
        self.unet.config.addition_embed_type = None
        # Change Generative UNet conv_in and conv_out
        unet_conv_in_channel_changed = self.unet.config.in_channels != new_in_channels
        if unet_conv_in_channel_changed:
            self.unet.conv_in = self.replace_conv_in_layer(
                self.unet, new_in_channels)
            self.unet.config.in_channels = new_in_channels
        unet_conv_out_channel_changed = (
            self.unet.config.out_channels != self.vae.config.latent_channels
        )
        if unet_conv_out_channel_changed:
            self.unet.conv_out = self.replace_conv_out_layer(
                self.unet, self.vae.config.latent_channels
            )
            self.unet.config.out_channels = self.vae.config.latent_channels

        unet_encoder_conv_in_channel_changed = (
            self.unet_encoder.config.in_channels != self.vae.config.latent_channels
        )
        if unet_encoder_conv_in_channel_changed:
            self.unet_encoder.conv_in = self.replace_conv_in_layer(
                self.unet_encoder, self.vae.config.latent_channels
            )
            self.unet_encoder.config.in_channels = self.vae.config.latent_channels
        unet_encoder_conv_out_channel_changed = (
            self.unet_encoder.config.out_channels != self.vae.config.latent_channels
        )
        if unet_encoder_conv_out_channel_changed:
            self.unet_encoder.conv_out = self.replace_conv_out_layer(
                self.unet_encoder, self.vae.config.latent_channels
            )
            self.unet_encoder.config.out_channels = self.vae.config.latent_channels

        # Remove Cross Attention
        remove_cross_attention(self.unet)
        remove_cross_attention(self.unet_encoder, model_type="unet_encoder")

        # Load pretrained model (safetensors 및 torch 모두 지원)
        if pretrained_model != "" and pretrained_model is not None:
            try:
                if pretrained_model.endswith('.safetensors'):
                    state_dict = safetensors.torch.load_file(pretrained_model)
                else:
                    state_dict = torch.load(pretrained_model, map_location="cpu", weights_only=False)
                
                # Remove module. prefix
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # **중요: strict=False 사용**
                missing_keys = self.load_state_dict(new_state_dict, strict=False)
                logger.info(f"Loaded pretrained model from {pretrained_model}")
                logger.info(f"Missing keys: {missing_keys.missing_keys}")
                logger.info(f"Unexpected keys: {missing_keys.unexpected_keys}")
                
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
                logger.info("Using default initialization")

    def replace_conv_in_layer(self, unet_model, new_in_channels):
        original_conv_in = unet_model.conv_in
        if original_conv_in.in_channels == new_in_channels:
            return original_conv_in
        new_conv_in = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=original_conv_in.out_channels,
            kernel_size=original_conv_in.kernel_size,
            padding=1,
        )
        new_conv_in.weight.data.zero_()
        new_conv_in.bias.data = original_conv_in.bias.data.clone()
        if original_conv_in.in_channels < new_in_channels:
            new_conv_in.weight.data[:, : original_conv_in.in_channels] = (
                original_conv_in.weight.data
            )
        else:
            new_conv_in.weight.data[:, :new_in_channels] = original_conv_in.weight.data[
                :, :new_in_channels
            ]
        return new_conv_in

    def replace_conv_out_layer(self, unet_model, new_out_channels):
        original_conv_out = unet_model.conv_out
        if original_conv_out.out_channels == new_out_channels:
            return original_conv_out
        new_conv_out = torch.nn.Conv2d(
            in_channels=original_conv_out.in_channels,
            out_channels=new_out_channels,
            kernel_size=original_conv_out.kernel_size,
            padding=1,
        )
        new_conv_out.weight.data.zero_()
        new_conv_out.bias.data[: original_conv_out.out_channels] = (
            original_conv_out.bias.data.clone()
        )
        if original_conv_out.out_channels < new_out_channels:
            new_conv_out.weight.data[: original_conv_out.out_channels] = (
                original_conv_out.weight.data
            )
        else:
            new_conv_out.weight.data[:new_out_channels] = original_conv_out.weight.data[
                :new_out_channels
            ]
        return new_conv_out

    def vae_encode(self, pixel_values):
        pixel_values = pixel_values.to(
            device=self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latent = self.vae.encode(pixel_values).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        return latent


class SkipAttnProcessor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states


def remove_cross_attention(
    unet,
    cross_attn_cls=SkipAttnProcessor,
    self_attn_cls=None,
    cross_attn_dim=None,
    **kwargs,
):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else cross_attn_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    **kwargs,
                )
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    layer_name=name,
                    **kwargs,
                )
        else:
            attn_procs[name] = cross_attn_cls(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                **kwargs,
            )

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self, hidden_size=None, cross_attention_dim=None, layer_name=None, **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.layer_name = layer_name
        self.model_type = kwargs.get("model_type", "none")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
