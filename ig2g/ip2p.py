# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import os
import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import Tensor, nn
import torch.fft as fft

from jaxtyping import Float
import wandb

os.environ["COMMANDLINE_ARGS"] = '--precision full --no-half'

CONSOLE = Console(width=120)

try:
    from diffusers import (
        DDIMScheduler,
        StableDiffusionInstructPix2PixPipeline,
        DiffusionPipeline,
    )
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"


def get_low_or_high_fft(x, scale, is_low=True):
    # FFT
    x_freq = fft.fftn(x.float(), dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape

    # extract
    if is_low:
        mask = torch.zeros((B, C, H, W), device=x.device)
        crow, ccol = H // 2, W // 2
        mask[..., crow - int(crow * scale):crow + int(crow * scale),
        ccol - int(ccol * scale):ccol + int(ccol * scale)] = 1
    else:
        mask = torch.ones((B, C, H, W), device=x.device)
        crow, ccol = H // 2, W // 2
        mask[..., crow - int(crow * scale):crow + int(crow * scale),
        ccol - int(ccol * scale):ccol + int(ccol * scale)] = 0
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    return x_filtered


def normalize_latent_noise(noise, device, use_outlier_clipping=False, use_scaling=False):
    # Ensure the noise tensor is on the specified device
    noise = noise.to(device)

    # Calculate mean and std of the input tensor for each channel
    mean = noise.mean(dim=(2, 3), keepdim=True)
    std = noise.std(dim=(2, 3), keepdim=True)

    # Standardize the noise tensor
    standardized_noise = (noise - mean) / std

    if use_outlier_clipping:
        # Clip the values to avoid extreme outliers (optional)
        clipped_noise = torch.clamp(standardized_noise, -3, 3)
        standardized_noise = clipped_noise

    if use_scaling:
        # Scale the clipped tensor to the range [-1, 1] (optional)
        min_val = standardized_noise.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_val = standardized_noise.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        scaled_noise = 2 * (standardized_noise - min_val) / (max_val - min_val) - 1
        standardized_noise = scaled_noise

    return standardized_noise


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor


class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000,
                 ip2p_use_full_precision=False, ip2p_params=None) -> None:
        super().__init__()

        if ip2p_params is None:
            ip2p_params = {}
            self.ip2p_params = {}
        else:
            self.ip2p_params = ip2p_params
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        # First, load the entire pipeline with from_pretrained to get all the components
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            IP2P_SOURCE,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)

        freeu_mode = ip2p_params['freeu_mode']
        if freeu_mode == "intermediate":
            # # Get the configuration of the original UNet2DConditionModel
            # unet_config = pipe.unet.config  # Extract the UNet config
            #
            # # Initialize the custom UNet with the same configuration
            # custom_unet = UNet2DIntermediateConditionModel(**unet_config).to(self.device)
            #
            # # Replace the UNet in the pipeline with the custom one
            # pipe.unet = custom_unet
            pass

        # pipe = DiffusionPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        is_freeu = ip2p_params['is_freeu']
        self.freeu_mode = freeu_mode
        self.is_freeu = is_freeu
        s1, s2, b1, b2 = ip2p_params['freeu_s1'], ip2p_params['freeu_s2'], ip2p_params['freeu_b1'], ip2p_params['freeu_b2']
        # FreeU enabled
        if is_freeu:
            pipe.enable_freeu(s1, s2, b1, b2)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        CONSOLE.print("InstructPix2Pix loaded!")

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae

    def edit_image(
            self,
            text_embeddings: Float[Tensor, "N max_length embed_dim"],
            image: Float[Tensor, "BS 3 H W"],
            image_cond: Float[Tensor, "BS 3 H W"],
            rendered_noise: Float[Tensor, "BS 3 H W"],
            guidance_scale: float = 7.5,
            image_guidance_scale: float = 1.5,
            diffusion_steps: int = 20,
            lower_bound: float = 0.70,
            upper_bound: float = 0.98,
            noise_type: str = "None",
            noise_reg: str = "None",
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)

        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            latents_rendered = None
            noise = None
            noise_rendered = None

            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)  # image: (1, 3, 412, 622) | latents: (1, 4, 51, 77)
            latents_0 = latents

            image_cond_latents = self.prepare_image_latents(image_cond)  # image_cond: (1, 3, 412, 622)
            # image_cond_latents: (3, 4, 51, 77)

            # add noise
            if noise_type == 'None' or 'concat' in noise_type:
                noise = torch.randn_like(latents)
            else:
                noise = rendered_noise

            if 'encoded' in noise_type and 'concat' not in noise_type:
                # project noise into latent using autoencoder
                noise_rendered = self.prepare_noise_latents(noise)

                if 'encoded-normalized' in noise_type:
                    # normalize noise
                    if noise_reg == 'outlier':
                        use_outlier_clipping = True
                        use_scaling = False
                    elif noise_reg == 'scaling':
                        use_outlier_clipping = False
                        use_scaling = True
                    elif ('outlier' in noise_reg) and ('scaling' in noise_reg):
                        use_outlier_clipping = True
                        use_scaling = True
                    else:
                        use_outlier_clipping = False
                        use_scaling = False
                    noise_rendered = normalize_latent_noise(noise_rendered, self.device, use_outlier_clipping, use_scaling)

                if self.freeu_mode in ["intermediate", "intermediate-reverse", "cfg"]:
                    noise = torch.randn_like(latents)
                    latents_rendered = self.scheduler.add_noise(latents, noise_rendered, self.scheduler.timesteps[0])
                else:
                    noise = noise_rendered

            elif 'concat' in noise_type:
                noise_latents = self.prepare_noise_latents(rendered_noise)
                image_cond_latents[1, :, :, :] = noise_latents

        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore

        with torch.no_grad():
            if self.ip2p_params['is_noise_calibration']:
                # ### Noise Calibration(Algorithm 1)
                # N = self.ip2p_params['noise_calibration_steps']
                # a_t = self.alphas[T]
                # x_r = image_cond_latents
                # sqrt_one_minus_at = (1 - a_t).sqrt()
                # scale = self.ip2p_params['noise_calibration_scale']
                # for _ in range(N):
                #     # x = a_t.sqrt() * x_r + sqrt_one_minus_at * noise
                #     # x = x.to(dtype=torch.float16)
                #
                #     # e_t_theta = self.model.apply_model(x, t, c, **kwargs)
                #     x = latents.to(dtype=torch.float16)
                #
                #     latent_model_input = torch.cat([x] * 3)
                #     latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                #     # noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                #     # noise_pred_text, noise_pred_image, e_t_theta = noise_pred.chunk(3)
                #     noise_pred_text, e_t_theta, noise_pred_uncond = noise_pred.chunk(3)
                #
                #     x_0_t = (x - sqrt_one_minus_at * e_t_theta) / a_t.sqrt()
                #     e_t = e_t_theta + a_t.sqrt() / sqrt_one_minus_at * (
                #                 get_low_or_high_fft(x_0_t, scale, is_low=False) - get_low_or_high_fft(x_r, scale,
                #                                                                                       is_low=False))
                #
                # # latent_model_input = a_t.sqrt() * x_r + sqrt_one_minus_at * e_t
                # latents = self.scheduler.add_noise(latents, e_t, self.scheduler.timesteps[0])  # type: ignore

                ### Noise Calibration(Algorithm 1)
                x_r = image_cond_latents[0].unsqueeze(0)
                N = self.ip2p_params['noise_calibration_steps']
                t = self.scheduler.timesteps[0]
                # a_t = self.alphas[t]
                a_t = self.alphas[t-1]
                sqrt_one_minus_at = (1 - a_t).sqrt()
                e_t = noise
                scale = self.ip2p_params['noise_calibration_scale']

                for _ in range(N):
                    # x = a_t.sqrt() * x_r + sqrt_one_minus_at * e_t
                    x = self.scheduler.add_noise(latents_0, e_t, self.scheduler.timesteps[0])

                    # e_t_theta = self.model.apply_model(x, t, c, **kwargs)
                    latent_model_input = torch.cat([latents_0] * 3)
                    latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_text, e_t_theta, noise_pred_uncond = noise_pred.chunk(3)
                    x_0_t = (x - sqrt_one_minus_at * e_t_theta) / a_t.sqrt()
                    e_t = e_t_theta + a_t.sqrt() / sqrt_one_minus_at * (
                            get_low_or_high_fft(x_0_t, scale, is_low=False) - get_low_or_high_fft(x_r, scale,
                                                                                                  is_low=False))

                # x = a_t.sqrt() * x_r + sqrt_one_minus_at * e_t
                latents = self.scheduler.add_noise(latents_0, e_t, self.scheduler.timesteps[0])  # type: ignore

        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                # "intermediate": backbone - random noise; skip - rendered noise
                # "intermediate-reverse": backbone - rendered noise; skip - random noise
                if self.freeu_mode in ["intermediate", "intermediate-reverse"]:
                    latent_model_input_rendered = torch.cat([latents_rendered] * 3)
                    latent_model_input_rendered = torch.cat([latent_model_input_rendered, image_cond_latents], dim=1)
                    self.unet.to(self.device)

                    if self.freeu_mode == "intermediate":
                        intermediate_feature = self.unet.forward_intermediate(latent_model_input_rendered, t, encoder_hidden_states=text_embeddings)
                        noise_pred = self.unet.forward_fused(latent_model_input,
                                                             intermediate_feature,
                                                             lambda_intermediate=self.ip2p_params['lambda_intermediate'],
                                                             timestep=t,
                                                             encoder_hidden_states=text_embeddings).sample
                    elif self.freeu_mode == "intermediate-reverse":
                        intermediate_feature = self.unet.forward_intermediate(latent_model_input, t,
                                                                              encoder_hidden_states=text_embeddings)
                        noise_pred = self.unet.forward_fused(latent_model_input_rendered,
                                                             intermediate_feature,
                                                             lambda_intermediate=self.ip2p_params['lambda_intermediate'],
                                                             timestep=t,
                                                             encoder_hidden_states=text_embeddings).sample
                elif self.freeu_mode == "cfg":
                    latent_model_input_rendered = torch.cat([latents_rendered] * 3)
                    latent_model_input_rendered = torch.cat([latent_model_input_rendered, image_cond_latents], dim=1)
                    self.unet.to(self.device)
                    rendered_noise_pred = self.unet(latent_model_input_rendered, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                else:
                    noise_pred = self.unet(latent_model_input.to(torch.float16), t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            if self.freeu_mode == "cfg":
                noise_guidance_scale = self.ip2p_params['noise_guidance_scale']
                noise_guidance_scale2 = self.ip2p_params['noise_guidance_scale2']
                rendered_pred_text, rendered_pred_image, rendered_pred_uncond = rendered_noise_pred.chunk(3)
                noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        + noise_guidance_scale * (rendered_pred_image - noise_pred_uncond)
                        + noise_guidance_scale2 * (rendered_pred_image - noise_pred_image)
                )
            else:
                noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        # Check if latents are in float32, if so, cast them to float16
        if latents.dtype == torch.float32:
            latents = latents.to(torch.float16)

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        # Convert images to the range [-1, 1]
        imgs = 2 * imgs - 1

        # Ensure the input tensor is of the same type as the autoencoder's expected input
        imgs = imgs.to(self.auto_encoder.dtype)

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        # Convert images to the range [-1, 1]
        imgs = 2 * imgs - 1

        # Ensure the input tensor is of the same type as the autoencoder's expected input
        imgs = imgs.to(self.auto_encoder.dtype)

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def prepare_noise_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        # Convert images to the range [-1, 1]
        imgs = 2 * imgs - 1

        # Ensure the input tensor is of the same type as the autoencoder's expected input
        imgs = imgs.to(self.auto_encoder.dtype)

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        return image_latents

    def prepare_image_noise_latents(self, imgs: Float[Tensor, "BS 3 H W"], noise: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        # Convert images to the range [-1, 1]
        imgs = 2 * imgs - 1
        noise = 2 * noise - 1

        # Ensure the input tensor is of the same type as the autoencoder's expected input
        imgs = imgs.to(self.auto_encoder.dtype)
        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        noise = noise.to(self.auto_encoder.dtype)
        noise_latents = self.auto_encoder.encode(noise).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents, noise_latents], dim=0)

        return image_latents

    def decode_latents_to_noise(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        # Decode the latents to get the noise
        with torch.no_grad():
            decoded_images = self.auto_encoder.decode(latents).sample

        # Convert the noise from the range [-1, 1] to the range [0, 1]
        decoded_images = (decoded_images + 1) / 2

        return decoded_images

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
