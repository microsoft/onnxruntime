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

import gc
import os
import shutil
from typing import List, Optional, Union

import torch
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import DIFFUSERS_CACHE, logging
from huggingface_hub import snapshot_download
from ort_utils import OrtCudaSession
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import onnxruntime as ort
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_clip import ClipOnnxModel
from onnxruntime.transformers.onnx_model_unet import UnetOnnxModel
from onnxruntime.transformers.onnx_model_vae import VaeOnnxModel
from onnxruntime.transformers.optimizer import optimize_by_onnxruntime, optimize_model

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Engine(OrtCudaSession):
    def __init__(self, engine_path, provider, device_id: int = 0, enable_cuda_graph=False):
        self.engine_path = engine_path
        self.provider = provider
        self.provider_options = self.get_cuda_provider_options(device_id, enable_cuda_graph)

        device = torch.device("cuda", device_id)
        ort_session = ort.InferenceSession(
            self.engine_path,
            providers=[
                (provider, self.provider_options),
                "CPUExecutionProvider",
            ],
        )

        super().__init__(ort_session, device, enable_cuda_graph)

    def get_cuda_provider_options(self, device_id: int, enable_cuda_graph: bool):
        return {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cuda_graph": enable_cuda_graph,
        }


class OrtStableDiffusionOptimizer:
    def __init__(self, model_type: str):
        assert model_type in ["vae", "unet", "clip"]
        self.model_type = model_type
        self.model_type_class_mapping = {
            "unet": UnetOnnxModel,
            "vae": VaeOnnxModel,
            "clip": ClipOnnxModel,
        }

    def optimize_by_ort(self, onnx_model):
        import tempfile
        from pathlib import Path

        import onnx

        # Use this step to see the final graph that executed by Onnx Runtime.
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save to a temporary file so that we can load it with Onnx Runtime.
            logger.info("Saving a temporary model to run OnnxRuntime graph optimizations...")
            tmp_model_path = Path(tmp_dir) / "model.onnx"
            onnx_model.save_model_to_file(str(tmp_model_path))
            ort_optimized_model_path = tmp_model_path
            optimize_by_onnxruntime(
                str(tmp_model_path), use_gpu=True, optimized_model_path=str(ort_optimized_model_path)
            )
            model = onnx.load(str(ort_optimized_model_path), load_external_data=True)
            return self.model_type_class_mapping[self.model_type](model)

    def optimize(self, input_fp32_onnx_path, optimized_onnx_path, float16=True):
        """Optimize onnx model using ONNX Runtime transformers optimizer"""
        logger.info(f"Optimize {input_fp32_onnx_path}...")
        fusion_options = FusionOptions(self.model_type)
        if self.model_type in ["unet"] and not float16:
            fusion_options.enable_packed_kv = False
            fusion_options.enable_packed_qkv = False

        m = optimize_model(
            input_fp32_onnx_path,
            model_type=self.model_type,
            num_heads=0,  # will be deduced from graph
            hidden_size=0,  # will be deduced from graph
            opt_level=0,
            optimization_options=fusion_options,
            use_gpu=True,
        )

        if self.model_type == "clip":
            m.prune_graph(outputs=["text_embeddings"])  # remove the pooler_output, and only keep the first output.

        if float16:
            logger.info("Convert to float16 ...")
            m.convert_float_to_float16(
                keep_io_types=False,
                op_block_list=["RandomNormalLike"],
            )

        # Note that ORT 1.15 could not save model larger than 2GB. This only works for float16
        if float16 or (self.model_type != "unet"):
            m = self.optimize_by_ort(m)

        m.get_operator_statistics()
        m.get_fused_operator_statistics()
        m.save_model_to_file(optimized_onnx_path, use_external_data_format=(self.model_type == "unet") and not float16)
        logger.info("%s is optimized: %s", self.model_type, optimized_onnx_path)


class BaseModel:
    def __init__(self, model, name, device="cuda", max_batch_size=16, embedding_dim=768, text_maxlen=77):
        self.model = model
        self.name = name
        self.device = device

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256  # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

        self.model_type = name.lower() if name in ["CLIP", "UNet"] else "vae"
        self.optimizer = OrtStableDiffusionOptimizer(self.model_type)

    def get_model(self):
        return self.model

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, input_fp32_onnx_path, optimized_onnx_path, fp16):
        self.optimizer.optimize(input_fp32_onnx_path, optimized_onnx_path, fp16)

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_image_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_image_shape else self.min_image_shape
        max_image_height = image_height if static_image_shape else self.max_image_shape
        min_image_width = image_width if static_image_shape else self.min_image_shape
        max_image_width = image_width if static_image_shape else self.max_image_shape
        min_latent_height = latent_height if static_image_shape else self.min_latent_shape
        max_latent_height = latent_height if static_image_shape else self.max_latent_shape
        min_latent_width = latent_width if static_image_shape else self.min_latent_shape
        max_latent_width = latent_width if static_image_shape else self.max_latent_shape
        return (
            min_batch,
            max_batch,
            min_image_height,
            max_image_height,
            min_image_width,
            max_image_width,
            min_latent_height,
            max_latent_height,
            min_latent_width,
            max_latent_width,
        )


def get_onnx_path(model_name, onnx_dir):
    return os.path.join(onnx_dir, model_name + ".onnx")


def get_engine_path(engine_dir, model_name, profile_id):
    return os.path.join(engine_dir, model_name + profile_id + ".onnx")


def build_engines(
    models,
    engine_dir,
    onnx_dir,
    onnx_opset,
    force_engine_rebuild: bool = False,
    fp16: bool = True,
    provider: str = "CUDAExecutionProvider",
    device_id: int = 0,
    enable_cuda_graph: bool = False,
):
    profile_id = "_fp16" if fp16 else "_fp32"

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
        onnx_path = get_onnx_path(model_name, onnx_dir)
        onnx_opt_path = get_engine_path(engine_dir, model_name, profile_id)
        if os.path.exists(onnx_opt_path):
            logger.info("Found cached optimized model: %s", onnx_opt_path)
        else:
            if os.path.exists(onnx_path):
                logger.info("Found cached model: %s", onnx_path)
            else:
                logger.info("Exporting model: %s", onnx_path)
                model = model_obj.get_model().to(model_obj.device)
                with torch.inference_mode():
                    inputs = model_obj.get_sample_input(1, 512, 512)
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

            # Optimize onnx
            logger.info("Generating optimized model: %s", onnx_opt_path)
            model_obj.optimize(onnx_path, onnx_opt_path, fp16)

    built_engines = {}
    for model_name in models:
        engine_path = get_engine_path(engine_dir, model_name, profile_id)
        engine = Engine(engine_path, provider, device_id=device_id, enable_cuda_graph=enable_cuda_graph)
        logger.info("%s options for %s: %s", provider, model_name, engine.provider_options)
        built_engines[model_name] = engine

    return built_engines


def run_engine(engine, feed_dict):
    return engine.infer(feed_dict)


class CLIP(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super().__init__(
            model=model,
            name="CLIP",
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        )

    def get_input_names(self):
        return ["input_ids"]

    def get_output_names(self):
        return ["text_embeddings", "pooler_output"]

    def get_dynamic_axes(self):
        return {"input_ids": {0: "B"}, "text_embeddings": {0: "B"}}

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            "input_ids": (batch_size, self.text_maxlen),
            "text_embeddings": (batch_size, self.text_maxlen, self.embedding_dim),
            # "pooler_output": (batch_size, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)


class UNet(BaseModel):
    def __init__(
        self,
        model,
        device="cuda",
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4,
    ):
        super().__init__(
            model=model,
            name="UNet",
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            text_maxlen=text_maxlen,
        )
        self.unet_dim = unet_dim

    def get_input_names(self):
        return ["sample", "timestep", "encoder_hidden_states"]

    def get_output_names(self):
        return ["latent"]

    def get_dynamic_axes(self):
        return {
            "sample": {0: "2B", 2: "H", 3: "W"},
            "encoder_hidden_states": {0: "2B"},
            "latent": {0: "2B", 2: "H", 3: "W"},
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "sample": (2 * batch_size, self.unet_dim, latent_height, latent_width),
            "timestep": [1],
            "encoder_hidden_states": (2 * batch_size, self.text_maxlen, self.embedding_dim),
            "latent": (2 * batch_size, 4, latent_height, latent_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return (
            torch.randn(
                2 * batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device
            ),
            torch.tensor([1.0], dtype=torch.float32, device=self.device),
            torch.randn(2 * batch_size, self.text_maxlen, self.embedding_dim, dtype=torch.float32, device=self.device),
        )


class VAE(BaseModel):
    def __init__(self, model, device, max_batch_size, embedding_dim):
        super().__init__(
            model=model,
            name="VAE Decoder",
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        )

    def get_input_names(self):
        return ["latent"]

    def get_output_names(self):
        return ["images"]

    def get_dynamic_axes(self):
        return {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            "latent": (batch_size, 4, latent_height, latent_width),
            "images": (batch_size, 3, image_height, image_width),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)


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
        self.onnx_opset = onnx_opset
        self.onnx_dir = onnx_dir
        self.engine_dir = engine_dir
        self.force_engine_rebuild = force_engine_rebuild
        self.enable_cuda_graph = enable_cuda_graph

        self.max_batch_size = 16

        self.models = {}  # loaded in __load_models()
        self.engines = {}  # loaded in build_engines()

        self.provider = "CUDAExecutionProvider"
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
        self.__load_models()

        # build engines
        self.fp16 = torch_dtype == torch.float16
        self.engines = build_engines(
            self.models,
            self.engine_dir,
            self.onnx_dir,
            self.onnx_opset,
            force_engine_rebuild=self.force_engine_rebuild,
            fp16=self.fp16,
            provider=self.provider,
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

            timestep_float = timestep.to(torch.float16) if self.fp16 else timestep.to(torch.float32)

            # Predict the noise residual
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
    import torch
    from diffusers import DDIMScheduler

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
