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
from diffusers.utils import DIFFUSERS_CACHE
from huggingface_hub import snapshot_download
from models import CLIP, VAE, UNet
from ort_utils import Engines
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


class OnnxruntimeCudaStableDiffusionPipeline(StableDiffusionPipeline):
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
        onnx_dir: str = "raw_onnx",
        # Onnxruntime execution provider parameters
        engine_dir: str = "onnxruntime_optimized_onnx",
        force_engine_rebuild: bool = False,
        enable_cuda_graph: bool = False,
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

    def __load_models(self):
        self.embedding_dim = self.text_encoder.config.hidden_size

        self.models["clip"] = CLIP(
            self.text_encoder,
            device=self.torch_device,
            max_batch_size=self.max_batch_size,
            embedding_dim=self.embedding_dim,
        )

        self.models["unet"] = UNet(
            self.unet,
            device=self.torch_device,
            fp16=self.fp16,
            max_batch_size=self.max_batch_size,
            embedding_dim=self.embedding_dim,
            unet_dim=(9 if self.inpaint else 4),
        )

        self.models["vae"] = VAE(
            self.vae, device=self.torch_device, max_batch_size=self.max_batch_size, embedding_dim=self.embedding_dim
        )

    @classmethod
    def set_cached_folder(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)

        cls.cached_folder = (
            pretrained_model_name_or_path
            if os.path.isdir(pretrained_model_name_or_path)
            else snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )
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
        self.__load_models()

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

    def __encode_prompt(self, prompt, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        # Tokenize prompt
        text_input_ids = (
            self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings = (
            self.engines.get_engine("clip").infer({"input_ids": text_input_ids})["text_embeddings"].clone()
        )

        # Tokenize negative prompt
        uncond_input_ids = (
            self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        uncond_embeddings = self.engines.get_engine("clip").infer({"input_ids": uncond_input_ids})["text_embeddings"]

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        return text_embeddings

    def __denoise_latent(self, latents, text_embeddings, timesteps=None, mask=None, masked_image_latents=None):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps

        for _step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            timestep_float = timestep.to(torch.float16) if self.fp16 else timestep.to(torch.float32)

            # Predict the noise residual
            noise_pred = self.engines.get_engine("unet").infer(
                {"sample": latent_model_input, "timestep": timestep_float, "encoder_hidden_states": text_embeddings},
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        latents = 1.0 / 0.18215 * latents
        return latents

    def __decode_latent(self, latents):
        images = self.engines.get_engine("vae").infer({"latent": latents})["images"]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).float().numpy()

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
            text_embeddings = self.__encode_prompt(prompt, negative_prompt)

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
            latents = self.__denoise_latent(latents, text_embeddings)

            # VAE decode latent
            images = self.__decode_latent(latents)

        images, has_nsfw_concept = self.run_safety_checker(images, self.torch_device, text_embeddings.dtype)
        images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


if __name__ == "__main__":
    model_name_or_path = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")

    pipe = OnnxruntimeCudaStableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        scheduler=scheduler,
    )

    # re-use cached folder to save ONNX models
    pipe.set_cached_folder(model_name_or_path)

    pipe = pipe.to("cuda", torch_dtype=torch.float16)

    prompt = "photorealistic new zealand hills"
    image = pipe(prompt).images[0]
    image.save("ort_trt_txt2img_new_zealand_hills.png")
