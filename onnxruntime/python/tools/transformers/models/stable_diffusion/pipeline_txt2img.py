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


class Txt2ImgPipeline(StableDiffusionPipeline):
    """
    Stable Diffusion Txt2Img pipeline using NVidia TensorRT.
    """

    def __init__(self, pipeline_info: PipelineInfo, **kwargs):
        """
        Initializes the Txt2Img Diffusion pipeline.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of stable diffusion pipeline.
        """
        super().__init__(pipeline_info, **kwargs)

    def _infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        denoising_steps=50,
        guidance=7.5,
        seed=None,
        warmup=False,
        return_type="latents",
    ):
        assert len(prompt) == len(negative_prompt)
        batch_size = len(prompt)

        self.set_denoising_steps(denoising_steps)
        self.set_random_seed(seed)

        with torch.inference_mode(), torch.autocast("cuda"):
            # Pre-initialize latents
            latents = self.initialize_latents(
                batch_size=batch_size,
                unet_channels=4,
                latent_height=(image_height // 8),
                latent_width=(image_width // 8),
            )

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # CLIP text encoder
            text_embeddings = self.encode_prompt(prompt, negative_prompt)

            # UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, guidance=guidance)

            # VAE decode latent
            images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            if not warmup:
                self.print_summary(e2e_tic, e2e_toc, batch_size)
                self.save_images(images, "txt2img", prompt)

            return images, (e2e_toc - e2e_tic) * 1000.0

    def run(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        denoising_steps=30,
        guidance=7.5,
        seed=None,
        warmup=False,
        return_type="images",
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
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
                type of return. The value can be "latents" or "images".
        """
        if self.is_backend_tensorrt():
            import tensorrt as trt
            from trt_utilities import TRT_LOGGER

            with trt.Runtime(TRT_LOGGER):
                return self._infer(
                    prompt,
                    negative_prompt,
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
                image_height,
                image_width,
                denoising_steps=denoising_steps,
                guidance=guidance,
                seed=seed,
                warmup=warmup,
                return_type=return_type,
            )
