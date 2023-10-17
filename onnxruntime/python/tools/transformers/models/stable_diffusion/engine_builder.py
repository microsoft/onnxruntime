# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from enum import Enum

import torch
from diffusion_models import CLIP, VAE, CLIPWithProj, PipelineInfo, UNet, UNetXL


class EngineType(Enum):
    ORT_CUDA = 0  # ONNX Runtime CUDA Execution Provider
    ORT_TRT = 1  # ONNX Runtime TensorRT Execution Provider
    TRT = 2  # TensorRT
    TORCH = 3  # PyTorch


def get_engine_type(name: str) -> EngineType:
    name_to_type = {
        "ORT_CUDA": EngineType.ORT_CUDA,
        "ORT_TRT": EngineType.ORT_TRT,
        "TRT": EngineType.TRT,
        "TORCH": EngineType.TORCH,
    }
    return name_to_type[name]


class EngineBuilder:
    def __init__(
        self,
        engine_type: EngineType,
        pipeline_info: PipelineInfo,
        device="cuda",
        max_batch_size=16,
        hf_token=None,
        use_cuda_graph=False,
    ):
        """
        Initializes the Engine Builder.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            device (str | torch.device):
                device to run engine
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        self.engine_type = engine_type
        self.pipeline_info = pipeline_info
        self.max_batch_size = max_batch_size
        self.hf_token = hf_token
        self.use_cuda_graph = use_cuda_graph
        self.device = torch.device(device)
        self.torch_device = torch.device(device, torch.cuda.current_device())
        self.stages = pipeline_info.stages()

        # TODO: use custom fp16 for ORT_TRT, and no need to fallback to torch.
        self.vae_torch_fallback = self.pipeline_info.is_xl() and engine_type != EngineType.ORT_CUDA

        # For SD XL, use an VAE that modified to run in fp16 precision without generating NaNs.
        self.custom_fp16_vae = (
            "madebyollin/sdxl-vae-fp16-fix"
            if self.pipeline_info.is_xl() and self.engine_type == EngineType.ORT_CUDA
            else None
        )

        self.models = {}
        self.engines = {}
        self.torch_models = {}

    def teardown(self):
        for engine in self.engines.values():
            del engine
        self.engines = {}

    def get_cached_model_name(self, model_name):
        if self.pipeline_info.is_inpaint():
            model_name += "_inpaint"
        return model_name

    def get_onnx_path(self, model_name, onnx_dir, opt=True):
        engine_name = self.engine_type.name.lower()
        directory_name = self.get_cached_model_name(model_name) + (f".{engine_name}" if opt else "")
        onnx_model_dir = os.path.join(onnx_dir, directory_name)
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, "model.onnx")

    def get_engine_path(self, engine_dir, model_name, profile_id):
        return os.path.join(engine_dir, self.get_cached_model_name(model_name) + profile_id)

    def load_models(self, framework_model_dir: str):
        # Disable torch SDPA since torch 2.0.* cannot export it to ONNX
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            delattr(torch.nn.functional, "scaled_dot_product_attention")

        # For TRT or ORT_TRT, we will export fp16 torch model for UNet.
        # For ORT_CUDA, we export fp32 model first, then optimize to fp16.
        export_fp16_unet = self.engine_type in [EngineType.ORT_TRT, EngineType.TRT]

        if "clip" in self.stages:
            self.models["clip"] = CLIP(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
                clip_skip=0,
            )

        if "clip2" in self.stages:
            self.models["clip2"] = CLIPWithProj(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
                clip_skip=0,
            )

        if "unet" in self.stages:
            self.models["unet"] = UNet(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                fp16=export_fp16_unet,
                max_batch_size=self.max_batch_size,
                unet_dim=(9 if self.pipeline_info.is_inpaint() else 4),
            )

        if "unetxl" in self.stages:
            self.models["unetxl"] = UNetXL(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                fp16=export_fp16_unet,
                max_batch_size=self.max_batch_size,
                unet_dim=4,
                time_dim=(5 if self.pipeline_info.is_xl_refiner() else 6),
            )

        # VAE Decoder
        if "vae" in self.stages:
            self.models["vae"] = VAE(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
                custom_fp16_vae=self.custom_fp16_vae,
            )

            if self.vae_torch_fallback:
                self.torch_models["vae"] = self.models["vae"].load_model(framework_model_dir, self.hf_token)

    def load_resources(self, image_height, image_width, batch_size):
        # Allocate buffers for I/O bindings
        for model_name, obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            self.engines[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.torch_device
            )

    def vae_decode(self, latents):
        if self.vae_torch_fallback:
            if not self.custom_fp16_vae:
                latents = latents.to(dtype=torch.float32)
                self.torch_models["vae"] = self.torch_models["vae"].to(dtype=torch.float32)
            images = self.torch_models["vae"](latents)["sample"]
        else:
            images = self.run_engine("vae", {"latent": latents})["images"]

        return images


def get_engine_paths(work_dir: str, pipeline_info: PipelineInfo, engine_type: EngineType):
    root_dir = work_dir or "."
    short_name = pipeline_info.short_name()

    # When both ORT_CUDA and ORT_TRT/TRT is used, we shall make sub directory for each engine since
    # ORT_CUDA need fp32 torch model, while ORT_TRT/TRT use fp16 torch model.
    onnx_dir = os.path.join(root_dir, engine_type.name, short_name, "onnx")
    engine_dir = os.path.join(root_dir, engine_type.name, short_name, "engine")
    output_dir = os.path.join(root_dir, engine_type.name, short_name, "output")
    timing_cache = os.path.join(root_dir, engine_type.name, "timing_cache")
    framework_model_dir = os.path.join(root_dir, engine_type.name, "torch_model")

    return onnx_dir, engine_dir, output_dir, framework_model_dir, timing_cache
