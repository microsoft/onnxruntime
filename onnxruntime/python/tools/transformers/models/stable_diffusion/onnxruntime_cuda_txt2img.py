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
Stable diffusion text to image pipeline using ONNX Runtime CUDA execution provider.
Based on https://github.com/huggingface/diffusers/blob/v0.17.1/examples/community/stable_diffusion_tensorrt_txt2img.py
Modifications: (1) Create ONNX Runtime session (2) Use I/O Binding of ONNX Runtime for inference

Installation instructions
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade transformers diffusers>=0.16.0
pip install numpy>=1.24.1 onnx>=1.13.0 coloredlogs protobuf==3.20.3 psutil sympy
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
from diffusion_models import CLIP, VAE, PipelineInfo, UNet
from ort_utils import Engines, StableDiffusionPipelineMixin
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


class OnnxruntimeCudaStableDiffusionPipeline(StableDiffusionPipelineMixin, StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using CUDA provider in ONNX Runtime.
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
        # ONNX export parameters
        onnx_opset: int = 14,
        onnx_dir: str = "onnx_ort",
        # Onnxruntime execution provider parameters
        engine_dir: str = "ORT_CUDA",
        force_engine_rebuild: bool = False,
        enable_cuda_graph: bool = False,
        pipeline_info: PipelineInfo = None,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )

        self.vae.forward = self.vae.decode
        self.unet_in_channels = unet.config.in_channels

        self.inpaint = False
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild
        self.enable_cuda_graph = enable_cuda_graph

        self.max_batch_size = 16

        self.models = {}  # loaded in __load_models()
        self.engines = Engines("CUDAExecutionProvider", onnx_opset)

        self.fp16 = False

        self.pipeline_info = pipeline_info

    def load_models(self):
        assert self.pipeline_info.clip_embedding_dim() == self.text_encoder.config.hidden_size

        stages = self.pipeline_info.stages()
        if "clip" in stages:
            self.models["clip"] = CLIP(
                self.pipeline_info,
                self.text_encoder,
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
                clip_skip=0,
            )

        if "unet" in stages:
            self.models["unet"] = UNet(
                self.pipeline_info,
                self.unet,
                device=self.torch_device,
                fp16=False,
                max_batch_size=self.max_batch_size,
                unet_dim=(9 if self.pipeline_info.is_inpaint() else 4),
            )

        if "vae" in stages:
            self.models["vae"] = VAE(
                self.pipeline_info,
                self.vae,
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
            )

    def to(
        self,
        torch_device: Union[str, torch.device],
        torch_dtype: Optional[torch.dtype] = None,
        silence_dtype_warnings: bool = False,
    ):
        self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
        self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)

        # set device
        self.torch_device = torch.device(torch_device)

        # load models
        self.fp16 = torch_dtype == torch.float16
        self.load_models()

        # build engines
        self.engines.build(
            self.models,
            self.engine_dir,
            self.onnx_dir,
            force_engine_rebuild=self.force_engine_rebuild,
            fp16=self.fp16,
            device_id=self.torch_device.index or torch.cuda.current_device(),
            enable_cuda_graph=self.enable_cuda_graph,
        )

        # Load the remaining modules to GPU.
        self.text_encoder = None
        self.vae = None
        self.unet = None
        super().to(torch_device, torch_dtype, silence_dtype_warnings=silence_dtype_warnings)

        self.torch_device = self._execution_device
        logger.info(f"Running inference on device: {self.torch_device}")

        return self

    def __allocate_buffers(self, image_height, image_width, batch_size):
        # Allocate output tensors for I/O bindings
        for model_name, obj in self.models.items():
            self.engines.get_engine(model_name).allocate_buffers(
                obj.get_shape_dict(batch_size, image_height, image_width)
            )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        image_height: int = 512,
        image_width: int = 512,
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

        self.__allocate_buffers(image_height, image_width, batch_size)

        with torch.inference_mode(), torch.autocast("cuda"):
            # CLIP text encoder
            text_embeddings = self.encode_prompt(self.engines.get_engine("clip"), prompt, negative_prompt)

            # Pre-initialize latents
            num_channels_latents = self.unet_in_channels
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                image_height,
                image_width,
                torch.float16 if self.fp16 else torch.float32,
                self.torch_device,
                generator,
            )

            # UNet denoiser
            latents = self.denoise_latent(
                self.engines.get_engine("unet"), latents, text_embeddings, timestep_fp16=self.fp16
            )

            # VAE decode latent
            images = self.decode_latent(self.engines.get_engine("vae"), latents)

        images, has_nsfw_concept = self.run_safety_checker(images, self.torch_device, text_embeddings.dtype)
        images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


def example():
    pipeline_info = PipelineInfo("1.5")
    model_name_or_path = pipeline_info.name()
    scheduler = DDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
    pipe = OnnxruntimeCudaStableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        scheduler=scheduler,
        pipeline_info=pipeline_info,
    )

    # re-use cached folder to save ONNX models
    pipe.set_cached_folder(model_name_or_path, resume_download=True, local_files_only=True)

    pipe = pipe.to("cuda", torch_dtype=torch.float16)

    prompt = "photorealistic new zealand hills"
    image = pipe(prompt).images[0]
    image.save("ort_cuda_txt2img_new_zealand_hills.png")


if __name__ == "__main__":
    example()
