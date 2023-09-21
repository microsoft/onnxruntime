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

import gc
import os
import shutil
from typing import List, Optional, Union

import torch
from cuda import cudart
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import DIFFUSERS_CACHE, logging
from huggingface_hub import snapshot_download
from models import CLIP, VAE, UNet
from ort_utils import OrtCudaSession
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import onnxruntime as ort

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Engine(OrtCudaSession):
    def __init__(self, engine_path, device_id, onnx_path, fp16, input_profile, workspace_size, enable_cuda_graph):
        self.engine_path = engine_path
        self.ort_trt_provider_options = self.get_tensorrt_provider_options(
            input_profile,
            workspace_size,
            fp16,
            device_id,
            enable_cuda_graph,
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        ort_session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=[
                ("TensorrtExecutionProvider", self.ort_trt_provider_options),
            ],
        )

        device = torch.device("cuda", device_id)
        super().__init__(ort_session, device, enable_cuda_graph)

    def get_tensorrt_provider_options(self, input_profile, workspace_size, fp16, device_id, enable_cuda_graph):
        trt_ep_options = {
            "device_id": device_id,
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_timing_cache_enable": True,
            "trt_detailed_build_log": True,
            "trt_engine_cache_path": self.engine_path,
        }

        if enable_cuda_graph:
            trt_ep_options["trt_cuda_graph_enable"] = True

        if workspace_size > 0:
            trt_ep_options["trt_max_workspace_size"] = workspace_size

        if input_profile:
            min_shapes = []
            max_shapes = []
            opt_shapes = []
            for name, profile in input_profile.items():
                assert isinstance(profile, list) and len(profile) == 3
                min_shape = profile[0]
                opt_shape = profile[1]
                max_shape = profile[2]
                assert len(min_shape) == len(opt_shape) and len(opt_shape) == len(max_shape)

                min_shapes.append(f"{name}:" + "x".join([str(x) for x in min_shape]))
                opt_shapes.append(f"{name}:" + "x".join([str(x) for x in opt_shape]))
                max_shapes.append(f"{name}:" + "x".join([str(x) for x in max_shape]))

            trt_ep_options["trt_profile_min_shapes"] = ",".join(min_shapes)
            trt_ep_options["trt_profile_max_shapes"] = ",".join(max_shapes)
            trt_ep_options["trt_profile_opt_shapes"] = ",".join(opt_shapes)

        logger.info("trt_ep_options=%s", trt_ep_options)

        return trt_ep_options


def get_onnx_path(model_name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, model_name + (".opt" if opt else "") + ".onnx")


def get_engine_path(engine_dir, model_name, profile_id):
    return os.path.join(engine_dir, model_name + profile_id)


def has_engine_file(engine_path):
    if os.path.isdir(engine_path):
        children = os.scandir(engine_path)
        for entry in children:
            if entry.is_file() and entry.name.endswith(".engine"):
                return True
    return False


def get_work_space_size(model_name, max_workspace_size):
    gibibyte = 2**30
    workspace_size = 4 * gibibyte if model_name == "clip" else max_workspace_size
    if workspace_size == 0:
        _, free_mem, _ = cudart.cudaMemGetInfo()
        # The following logic are adopted from TensorRT demo diffusion.
        if free_mem > 6 * gibibyte:
            workspace_size = free_mem - 4 * gibibyte
    return workspace_size


def build_engines(
    models,
    engine_dir,
    onnx_dir,
    onnx_opset,
    opt_image_height,
    opt_image_width,
    opt_batch_size=1,
    force_engine_rebuild=False,
    static_batch=False,
    static_image_shape=True,
    max_workspace_size=0,
    device_id=0,
    enable_cuda_graph=False,
):
    if force_engine_rebuild:
        if os.path.isdir(onnx_dir):
            logger.info("Remove existing directory %s since force_engine_rebuild is enabled", onnx_dir)
            shutil.rmtree(onnx_dir)
        if os.path.isdir(engine_dir):
            logger.info("Remove existing directory %s since force_engine_rebuild is enabled", engine_dir)
            shutil.rmtree(engine_dir)

    if not os.path.isdir(engine_dir):
        os.makedirs(engine_dir)

    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)

    # Export models to ONNX
    for model_name, model_obj in models.items():
        profile_id = model_obj.get_profile_id(
            opt_batch_size, opt_image_height, opt_image_width, static_batch, static_image_shape
        )
        engine_path = get_engine_path(engine_dir, model_name, profile_id)
        if not has_engine_file(engine_path):
            onnx_path = get_onnx_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = get_onnx_path(model_name, onnx_dir)
            if not os.path.exists(onnx_opt_path):
                if not os.path.exists(onnx_path):
                    logger.info(f"Exporting model: {onnx_path}")
                    model = model_obj.get_model()
                    with torch.inference_mode(), torch.autocast("cuda"):
                        inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.info("Found cached model: %s", onnx_path)

                # Optimize onnx
                if not os.path.exists(onnx_opt_path):
                    logger.info("Generating optimizing model: %s", onnx_opt_path)
                    model_obj.optimize_trt(onnx_path, onnx_opt_path)
                else:
                    logger.info("Found cached optimized model: %s", onnx_opt_path)

    built_engines = {}
    for model_name, model_obj in models.items():
        profile_id = model_obj.get_profile_id(
            opt_batch_size, opt_image_height, opt_image_width, static_batch, static_image_shape
        )

        engine_path = get_engine_path(engine_dir, model_name, profile_id)
        onnx_opt_path = get_onnx_path(model_name, onnx_dir)

        if not has_engine_file(engine_path):
            logger.info(
                "Building TensorRT engine for %s from %s to %s. It can take a while to complete...",
                model_name,
                onnx_opt_path,
                engine_path,
            )
        else:
            logger.info("Reuse cached TensorRT engine in directory %s", engine_path)

        input_profile = model_obj.get_input_profile(
            opt_batch_size,
            opt_image_height,
            opt_image_width,
            static_batch=static_batch,
            static_image_shape=static_image_shape,
        )

        engine = Engine(
            engine_path,
            device_id,
            onnx_opt_path,
            fp16=True,
            input_profile=input_profile,
            workspace_size=get_work_space_size(model_name, max_workspace_size),
            enable_cuda_graph=enable_cuda_graph,
        )

        built_engines[model_name] = engine

    return built_engines


def run_engine(engine, feed_dict):
    return engine.infer(feed_dict)


class OnnxruntimeTensorRTStableDiffusionPipeline(StableDiffusionPipeline):
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
        onnx_dir: str = "onnx",
        # TensorRT engine build parameters
        engine_dir: str = "onnxruntime_tensorrt_engine",
        force_engine_rebuild: bool = False,
        enable_cuda_graph: bool = False,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker
        )

        self.vae.forward = self.vae.decode

        self.image_height = image_height
        self.image_width = image_width
        self.inpaint = False
        self.onnx_opset = onnx_opset
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild
        self.enable_cuda_graph = enable_cuda_graph

        # Although cuda graph requires static input shape, engine built with dyamic batch gets better performance in T4.
        # Use static batch could reduce GPU memory footprint.
        self.build_static_batch = False

        # TODO: support dynamic image shape.
        self.build_dynamic_shape = False

        self.max_batch_size = max_batch_size
        # Restrict batch size to 4 for larger image dimensions as a walkaround for TensorRT limitation.
        if self.build_dynamic_shape or self.image_height > 512 or self.image_width > 512:
            self.max_batch_size = 4

        self.models = {}  # loaded in __load_models()
        self.engines = {}  # loaded in build_engines()

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
            fp16=True,
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
        torch_device: Optional[Union[str, torch.device]] = None,
        silence_dtype_warnings: bool = False,
    ):
        super().to(torch_device, silence_dtype_warnings=silence_dtype_warnings)

        self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
        self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)

        # set device
        self.torch_device = self._execution_device
        logger.info(f"Running inference on device: {self.torch_device}")

        self.__load_models()

        self.engines = build_engines(
            self.models,
            self.engine_dir,
            self.onnx_dir,
            self.onnx_opset,
            opt_image_height=self.image_height,
            opt_image_width=self.image_width,
            force_engine_rebuild=self.force_engine_rebuild,
            static_batch=self.build_static_batch,
            static_image_shape=not self.build_dynamic_shape,
            device_id=self.torch_device.index,
            enable_cuda_graph=self.enable_cuda_graph,
        )

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
        text_embeddings = run_engine(self.engines["clip"], {"input_ids": text_input_ids})["text_embeddings"].clone()

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

        uncond_embeddings = run_engine(self.engines["clip"], {"input_ids": uncond_input_ids})["text_embeddings"]

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

            # Predict the noise residual
            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            noise_pred = run_engine(
                self.engines["unet"],
                {"sample": latent_model_input, "timestep": timestep_float, "encoder_hidden_states": text_embeddings},
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        latents = 1.0 / 0.18215 * latents
        return latents

    def __decode_latent(self, latents):
        images = run_engine(self.engines["vae"], {"latent": latents})["images"]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).float().numpy()

    def __allocate_buffers(self, image_height, image_width, batch_size):
        # Allocate output tensors for I/O bindings
        for model_name, obj in self.models.items():
            self.engines[model_name].allocate_buffers(obj.get_shape_dict(batch_size, image_height, image_width))

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

        self.__allocate_buffers(self.image_height, self.image_width, batch_size)

        with torch.inference_mode(), torch.autocast("cuda"):
            # CLIP text encoder
            text_embeddings = self.__encode_prompt(prompt, negative_prompt)

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
            latents = self.__denoise_latent(latents, text_embeddings)

            # VAE decode latent
            images = self.__decode_latent(latents)

        images, has_nsfw_concept = self.run_safety_checker(images, self.torch_device, text_embeddings.dtype)
        images = self.numpy_to_pil(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


if __name__ == "__main__":
    model_name_or_path = "runwayml/stable-diffusion-v1-5"

    scheduler = DDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")

    pipe = OnnxruntimeTensorRTStableDiffusionPipeline.from_pretrained(
        model_name_or_path,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler,
        image_height=512,
        image_width=512,
        max_batch_size=4,
    )

    # re-use cached folder to save ONNX models and TensorRT Engines
    pipe.set_cached_folder(model_name_or_path, revision="fp16")

    pipe = pipe.to("cuda")

    prompt = "photorealistic new zealand hills"
    image = pipe(prompt).images[0]
    image.save("ort_trt_txt2img_new_zealand_hills.png")
