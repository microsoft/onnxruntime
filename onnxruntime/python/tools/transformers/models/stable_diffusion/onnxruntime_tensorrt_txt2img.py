# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
#
# Copyright 2023 The HuggingFace Inc. team.
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

"""
Stable diffusion text to image pipeline using ONNX Runtime TensorRT execution provider.
Based on https://github.com/huggingface/diffusers/blob/v0.17.1/examples/community/stable_diffusion_tensorrt_txt2img.py
Modifications: (1) Create ONNX Runtime session (2) Use I/O Binding of ONNX Runtime for inference

Installation instructions
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade transformers diffusers>=0.16.0
pip install --upgrade tensorrt>=8.6.1
pip install --upgrade polygraphy>=0.47.0 onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install onnxruntime-gpu
"""

import logging
import os
from typing import List, Optional, Union

import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler
from diffusion_models import PipelineInfo
from engine_builder_ort_trt import OrtTensorrtEngineBuilder
from ort_utils import StableDiffusionPipelineMixin
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


class OnnxruntimeTensorRTStableDiffusionPipeline(StableDiffusionPipelineMixin, StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using TensorRT execution provider in ONNX Runtime.

    This pipeline inherits from [`StableDiffusionPipeline`]. Check the documentation in super class for most parameters.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        image_height: int = 768,
        image_width: int = 768,
        max_batch_size: int = 16,
        # ONNX export parameters
        onnx_opset: int = 17,
        onnx_dir: str = "onnx_trt",
        # TensorRT engine build parameters
        engine_dir: str = "ORT_TRT",  # use short name here to avoid path exceeds 260 chars in Windows.
        force_engine_rebuild: bool = False,
        enable_cuda_graph: bool = False,
        pipeline_info: Optional[PipelineInfo] = None,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )

        self.vae.forward = self.vae.decode

        self.image_height = image_height
        self.image_width = image_width
        self.onnx_opset = onnx_opset
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild

        # Although cuda graph requires static input shape, engine built with dynamic batch gets better performance in T4.
        # Use static batch could reduce GPU memory footprint.
        self.build_static_batch = enable_cuda_graph

        # TODO: support dynamic image shape.
        self.build_dynamic_shape = False

        self.max_batch_size = max_batch_size
        # Restrict batch size to 4 for larger image dimensions as a walkaround for TensorRT limitation.
        if self.build_dynamic_shape or self.image_height > 512 or self.image_width > 512:
            self.max_batch_size = 4

        self.engines = {}  # loaded in build_engines()
        self.engine_builder = OrtTensorrtEngineBuilder(
            pipeline_info, max_batch_size=max_batch_size, use_cuda_graph=enable_cuda_graph
        )

        self.pipeline_info = pipeline_info
        self.stages = pipeline_info.stages()

    def to(
        self,
        torch_device: Optional[Union[str, torch.device]] = None,
        silence_dtype_warnings: bool = False,
    ):
        super().to(torch_device, silence_dtype_warnings=silence_dtype_warnings)

        self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
        self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)

        # set device
        self.torch_device = self._execution_device
        logger.info(f"Running inference on device: {self.torch_device}")

        self.engines = self.engine_builder.build_engines(
            self.engine_dir,
            None,
            self.onnx_dir,
            self.onnx_opset,
            opt_image_height=self.image_height,
            opt_image_width=self.image_width,
            force_engine_rebuild=self.force_engine_rebuild,
            static_batch=self.build_static_batch,
            static_image_shape=not self.build_dynamic_shape,
            device_id=self.torch_device.index,
        )

        return self

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.

        """
        self.generator = generator
        self.denoising_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.set_timesteps(self.denoising_steps, device=self.torch_device)

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Expected prompt to be of type list or str but got {type(prompt)}")

        if negative_prompt is None:
            negative_prompt = [""] * batch_size

        if negative_prompt is not None and isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        assert len(prompt) == len(negative_prompt)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompt)} is larger than allowed {self.max_batch_size}. If dynamic shape is used, then maximum batch size is 4"
            )

        self.engine_builder.load_resources(self.image_height, self.image_width, batch_size)

        with torch.inference_mode(), torch.autocast("cuda"):
            # CLIP text encoder
            text_embeddings = self.encode_prompt(self.engines["clip"], prompt, negative_prompt)

            # Pre-initialize latents
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                self.image_height,
                self.image_width,
                torch.float32,
                self.torch_device,
                generator,
            )

            # UNet denoiser
            latents = self.denoise_latent(self.engines["unet"], latents, text_embeddings)

            # VAE decode latent
            images = self.decode_latent(self.engines["vae"], latents)

        images, has_nsfw_concept = self.run_safety_checker(images, self.torch_device, text_embeddings.dtype)
        images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


if __name__ == "__main__":
    pipeline_info = PipelineInfo("1.5")
    model_name_or_path = pipeline_info.name()
    scheduler = DDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")

    pipe = OnnxruntimeTensorRTStableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler,
        image_height=512,
        image_width=512,
        max_batch_size=4,
        pipeline_info=pipeline_info,
    )

    # re-use cached folder to save ONNX models and TensorRT Engines
    pipe.set_cached_folder(model_name_or_path, revision="fp16")

    pipe = pipe.to("cuda")

    prompt = "photorealistic new zealand hills"
    image = pipe(prompt).images[0]
    image.save("ort_trt_txt2img_new_zealand_hills.png")
