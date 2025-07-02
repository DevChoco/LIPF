import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PIL import Image, ImageFilter


class LeffaPipeline(object):
    def __init__(
        self,
        model,
        device="cuda",
    ):
        self.vae = model.vae
        self.unet_encoder = model.unet_encoder
        self.unet = model.unet
        self.noise_scheduler = model.noise_scheduler
        self.device = device

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        src_image,
        ref_image,
        mask,
        densepose,
        ref_acceleration=False,
        num_inference_steps=50,
        do_classifier_free_guidance=True,
        guidance_scale=2.5,
        generator=None,
        eta=1.0,
        repaint=False,  # used for virtual try-on
        **kwargs,
    ):
        src_image = src_image.to(device=self.vae.device, dtype=self.vae.dtype)
        ref_image = ref_image.to(device=self.vae.device, dtype=self.vae.dtype)
        mask = mask.to(device=self.vae.device, dtype=self.vae.dtype)
        densepose = densepose.to(device=self.vae.device, dtype=self.vae.dtype)
        masked_image = src_image * (mask < 0.5)

        # 1. VAE encoding
        with torch.no_grad():
            # src_image_latent = self.vae.encode(src_image).latent_dist.sample()
            masked_image_latent = self.vae.encode(
                masked_image).latent_dist.sample()
            ref_image_latent = self.vae.encode(ref_image).latent_dist.sample()
        # src_image_latent = src_image_latent * self.vae.config.scaling_factor
        masked_image_latent = masked_image_latent * self.vae.config.scaling_factor
        ref_image_latent = ref_image_latent * self.vae.config.scaling_factor
        mask_latent = F.interpolate(
            mask, size=masked_image_latent.shape[-2:], mode="nearest")
        densepose_latent = F.interpolate(
            densepose, size=masked_image_latent.shape[-2:], mode="nearest")

        # 2. prepare noise
        noise = torch.randn_like(masked_image_latent)
        self.noise_scheduler.set_timesteps(
            num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        noise = noise * self.noise_scheduler.init_noise_sigma
        latent = noise

        # 3. classifier-free guidance
        if do_classifier_free_guidance:
            # src_image_latent = torch.cat([src_image_latent] * 2)
            masked_image_latent = torch.cat([masked_image_latent] * 2)
            ref_image_latent = torch.cat(
                [torch.zeros_like(ref_image_latent), ref_image_latent])
            mask_latent = torch.cat([mask_latent] * 2)
            densepose_latent = torch.cat([densepose_latent] * 2)

        # 6. Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.noise_scheduler.order
        )

        if ref_acceleration:
            down, reference_features = self.unet_encoder(
                ref_image_latent, timesteps[num_inference_steps//2], encoder_hidden_states=None, return_dict=False
            )
            reference_features = list(reference_features)

        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latent if we are doing classifier free guidance
                _latent_model_input = (
                    torch.cat(
                        [latent] * 2) if do_classifier_free_guidance else latent
                )
                _latent_model_input = self.noise_scheduler.scale_model_input(
                    _latent_model_input, t
                )

                # prepare the input for the inpainting model
                latent_model_input = torch.cat(
                    [
                        _latent_model_input,
                        mask_latent,
                        masked_image_latent,
                        densepose_latent,
                    ],
                    dim=1,
                )

                if not ref_acceleration:
                    down, reference_features = self.unet_encoder(
                        ref_image_latent, t, encoder_hidden_states=None, return_dict=False
                    )
                    reference_features = list(reference_features)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    reference_features=reference_features,
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_scale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_cond,
                        guidance_rescale=guidance_scale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latent = self.noise_scheduler.step(
                    noise_pred, t, latent, **extra_step_kwargs, return_dict=False
                )[0]
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latent
        gen_image = latent_to_image(latent, self.vae)

        if repaint:
            src_image = (src_image / 2 + 0.5).clamp(0, 1)
            src_image = src_image.cpu().permute(0, 2, 3, 1).float().numpy()
            src_image = numpy_to_pil(src_image)
            mask = mask.cpu().permute(0, 2, 3, 1).float().numpy()
            mask = numpy_to_pil(mask)
            mask = [i.convert("RGB") for i in mask]
            gen_image = [
                do_repaint(_src_image, _mask, _gen_image)
                for _src_image, _mask, _gen_image in zip(src_image, mask, gen_image)
            ]

        return (gen_image,)


def latent_to_image(latent, vae):
    latent = 1 / vae.config.scaling_factor * latent
    image = vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = numpy_to_pil(image)
    return image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L")
                      for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def do_repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 100
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled +
        (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg
