# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from TensorRT demo diffusion, which has the following license:
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import time

import torch
from diffusion_models import PipelineInfo
from pipeline_stable_diffusion import StableDiffusionPipeline


class Img2ImgXLPipeline(StableDiffusionPipeline):
    """
    Stable Diffusion Img2Img XL pipeline.
    """

    def __init__(self, pipeline_info: PipelineInfo, *args, **kwargs):
        """
        Initializes the Img2Img XL Diffusion pipeline.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of stable diffusion pipeline.
        """
        assert pipeline_info.is_xl_refiner()

        super().__init__(pipeline_info, *args, **kwargs)

        self.requires_aesthetics_score = True

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype
    ):
        if self.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0).to(device=self.device)
        return add_time_ids

    def _infer(
        self,
        prompt,
        negative_prompt,
        init_image,
        image_height,
        image_width,
        denoising_steps=30,
        guidance=5.0,
        seed=None,
        warmup=False,
        return_type="image",
    ):
        assert negative_prompt is None or len(prompt) == len(negative_prompt)

        original_size = (image_height, image_width)
        crops_coords_top_left = (0, 0)
        target_size = (image_height, image_width)

        strength = 0.3
        aesthetic_score = 6.0
        negative_aesthetic_score = 2.5

        self.set_denoising_steps(denoising_steps)
        self.set_random_seed(seed)

        with torch.inference_mode(), torch.autocast("cuda"):
            batch_size = len(prompt)

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Initialize timesteps
            timesteps, t_start = self.initialize_timesteps(self.denoising_steps, strength)

            latent_timestep = timesteps[:1].repeat(batch_size)

            # CLIP text encoder 2
            text_embeddings, pooled_embeddings2 = self.encode_prompt(
                prompt,
                negative_prompt,
                encoder="clip2",
                tokenizer=self.tokenizer2,
                pooled_outputs=True,
                output_hidden_states=True,
            )

            # Time embeddings
            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                aesthetic_score,
                negative_aesthetic_score,
                dtype=text_embeddings.dtype,
            )

            add_time_ids = add_time_ids.repeat(batch_size, 1)

            add_kwargs = {"text_embeds": pooled_embeddings2, "time_ids": add_time_ids}

            # Pre-process input image
            init_image = self.preprocess_images(batch_size, (init_image,))[0]

            # VAE encode init image
            if init_image.shape[1] == 4:
                init_latents = init_image
            else:
                init_latents = self.encode_image(init_image)

            # Add noise to latents using timesteps
            noise = torch.randn(init_latents.shape, device=self.device, dtype=torch.float32, generator=self.generator)
            latents = self.scheduler.add_noise(init_latents, noise, t_start, latent_timestep)

            # UNet denoiser
            latents = self.denoise_latent(
                latents,
                text_embeddings,
                timesteps=timesteps,
                step_offset=t_start,
                denoiser="unetxl",
                guidance=guidance,
                add_kwargs=add_kwargs,
            )

        with torch.inference_mode():
            # VAE decode latent
            if return_type == "latent":
                images = latents
            else:
                images = self.decode_latent(latents / self.vae_scaling_factor)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            if not warmup:
                print("SD-XL Refiner Pipeline")
                self.print_summary(e2e_tic, e2e_toc, batch_size)
                self.save_images(images, "img2img-xl", prompt)

        return images, (e2e_toc - e2e_tic) * 1000.0

    def run(
        self,
        prompt,
        negative_prompt,
        init_image,
        image_height,
        image_width,
        denoising_steps=30,
        guidance=5.0,
        seed=None,
        warmup=False,
        return_type="image",
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            init_image (tuple[torch.Tensor]):
                Image from base pipeline.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            denoising_steps (int):
                Number of denoising steps. More steps usually lead to higher quality image at the expense of slower inference.
            guidance (float):
                Higher guidance scale encourages to generate images that are closely linked to the text prompt.
            seed (int):
                Seed for the random generator
            warmup (bool):
                Indicate if this is a warmup run.
            return_type (str):
                It can be "latent" or "image".
        """

        if self.is_backend_tensorrt():
            import tensorrt as trt
            from trt_utilities import TRT_LOGGER

            with trt.Runtime(TRT_LOGGER):
                return self._infer(
                    prompt,
                    negative_prompt,
                    init_image,
                    image_height,
                    image_width,
                    denoising_steps=denoising_steps,
                    guidance=guidance,
                    seed=seed,
                    warmup=warmup,
                    return_type=return_type,
                )
        else:
            return self._infer(
                prompt,
                negative_prompt,
                init_image,
                image_height,
                image_width,
                denoising_steps=denoising_steps,
                guidance=guidance,
                seed=seed,
                warmup=warmup,
                return_type=return_type,
            )
