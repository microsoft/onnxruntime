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
import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List

import controlnet_aux
import cv2
import numpy as np
import torch
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_paths
from PIL import Image


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def arg_parser(description: str):
    return argparse.ArgumentParser(
        description=description, formatter_class=RawTextArgumentDefaultsHelpFormatter, add_help=False
    )


def set_default_arguments(args):
    # set default value for some arguments if not provided
    if args.height is None:
        args.height = PipelineInfo.default_resolution(args.version)

    if args.width is None:
        args.width = PipelineInfo.default_resolution(args.version)

    is_lcm = (args.version == "xl-1.0" and args.lcm) or "lcm" in args.lora_weights
    is_turbo = args.version in ["sd-turbo", "xl-turbo"]
    if args.denoising_steps is None:
        args.denoising_steps = 4 if is_turbo else 8 if is_lcm else (30 if args.version == "xl-1.0" else 50)

    if args.scheduler is None:
        args.scheduler = "LCM" if (is_lcm or is_turbo) else ("EulerA" if args.version == "xl-1.0" else "DDIM")

    if args.guidance is None:
        args.guidance = 0.0 if (is_lcm or is_turbo) else (5.0 if args.version == "xl-1.0" else 7.5)


def parse_arguments(is_xl: bool, parser):
    engines = ["ORT_CUDA", "ORT_TRT", "TRT"]
    parser.add_argument("--help", action="store_true", help="show this help message and exit")

    parser.add_argument(
        "-e",
        "--engine",
        type=str,
        default=engines[0],
        choices=engines,
        help="Backend engine in {engines}. "
        "ORT_CUDA is CUDA execution provider; ORT_TRT is Tensorrt execution provider; TRT is TensorRT",
    )

    supported_versions = PipelineInfo.supported_versions(is_xl)
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="xl-1.0" if is_xl else "1.5",
        choices=supported_versions,
        help="Version of Stable Diffusion" + (" XL." if is_xl else "."),
    )

    parser.add_argument(
        "-h",
        "--height",
        type=int,
        default=None,
        help="Height of image to generate (must be multiple of 8).",
    )
    parser.add_argument(
        "-w", "--width", type=int, default=None, help="Height of image to generate (must be multiple of 8)."
    )

    parser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default=None,
        choices=["DDIM", "EulerA", "UniPC", "LCM"],
        help="Scheduler for diffusion process" + " of base" if is_xl else "",
    )

    parser.add_argument(
        "-wd",
        "--work-dir",
        default=".",
        help="Root Directory to store torch or ONNX models, built engines and output images etc.",
    )

    parser.add_argument("prompt", nargs="*", default=[""], help="Text prompt(s) to guide image generation.")

    parser.add_argument(
        "-n",
        "--negative-prompt",
        nargs="*",
        default=[""],
        help="Optional negative prompt(s) to guide the image generation.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Number of times to repeat the prompt (batch size multiplier).",
    )

    parser.add_argument(
        "-d",
        "--denoising-steps",
        type=int,
        default=None,
        help="Number of denoising steps" + (" in base." if is_xl else "."),
    )

    parser.add_argument(
        "-g",
        "--guidance",
        type=float,
        default=None,
        help="Higher guidance scale encourages to generate images that are closely linked to the text prompt.",
    )

    parser.add_argument(
        "-ls", "--lora-scale", type=float, default=1, help="Scale of LoRA weights, default 1 (must between 0 and 1)"
    )
    parser.add_argument("-lw", "--lora-weights", type=str, default="", help="LoRA weights to apply in the base model")

    if is_xl:
        parser.add_argument(
            "--lcm",
            action="store_true",
            help="Use fine-tuned latent consistency model to replace the UNet in base.",
        )

        parser.add_argument(
            "-rs",
            "--refiner-scheduler",
            type=str,
            default="EulerA",
            choices=["DDIM", "EulerA", "UniPC"],
            help="Scheduler for diffusion process of refiner.",
        )

        parser.add_argument(
            "-rg",
            "--refiner-guidance",
            type=float,
            default=5.0,
            help="Guidance scale used in refiner.",
        )

        parser.add_argument(
            "-rd",
            "--refiner-denoising-steps",
            type=int,
            default=30,
            help="Number of denoising steps in refiner. Note that actual steps is refiner_denoising_steps * strength.",
        )

        parser.add_argument(
            "--strength",
            type=float,
            default=0.3,
            help="A value between 0 and 1. The higher the value less the final image similar to the seed image.",
        )

        parser.add_argument(
            "-r",
            "--enable-refiner",
            action="store_true",
            help="Enable SDXL refiner to refine image from base pipeline.",
        )

    # ONNX export
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=None,
        choices=range(14, 18),
        help="Select ONNX opset version to target for exported models.",
    )
    parser.add_argument(
        "--force-onnx-export", action="store_true", help="Force ONNX export of CLIP, UNET, and VAE models."
    )
    parser.add_argument(
        "--force-onnx-optimize", action="store_true", help="Force ONNX optimizations for CLIP, UNET, and VAE models."
    )

    # Framework model ckpt
    parser.add_argument(
        "--framework-model-dir",
        default="pytorch_model",
        help="Directory for HF saved models. Default is pytorch_model.",
    )
    parser.add_argument("--hf-token", type=str, help="HuggingFace API access token for downloading model checkpoints.")

    # Engine build options.
    parser.add_argument("--force-engine-build", action="store_true", help="Force rebuilding the TensorRT engine.")
    parser.add_argument(
        "-db",
        "--build-dynamic-batch",
        action="store_true",
        help="Build TensorRT engines to support dynamic batch size.",
    )
    parser.add_argument(
        "-ds",
        "--build-dynamic-shape",
        action="store_true",
        help="Build TensorRT engines to support dynamic image sizes.",
    )
    parser.add_argument("--max-batch-size", type=int, default=None, choices=[1, 2, 4, 8, 16, 32], help="Max batch size")

    # Inference related options
    parser.add_argument(
        "-nw", "--num-warmup-runs", type=int, default=5, help="Number of warmup runs before benchmarking performance."
    )
    parser.add_argument("--nvtx-profile", action="store_true", help="Enable NVTX markers for performance profiling.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator to get consistent results.")
    parser.add_argument("-dc", "--disable-cuda-graph", action="store_true", help="Disable cuda graph.")

    group = parser.add_argument_group("Options for ORT_CUDA engine only")
    group.add_argument("--enable-vae-slicing", action="store_true", help="True will feed only one image to VAE once.")

    # TensorRT only options
    group = parser.add_argument_group("Options for TensorRT (--engine=TRT) only")
    group.add_argument("--onnx-refit-dir", help="ONNX models to load the weights from.")
    group.add_argument(
        "--build-enable-refit", action="store_true", help="Enable Refit option in TensorRT engines during build."
    )
    group.add_argument(
        "--build-preview-features", action="store_true", help="Build TensorRT engines with preview features."
    )
    group.add_argument(
        "--build-all-tactics", action="store_true", help="Build TensorRT engines using all tactic sources."
    )

    args = parser.parse_args()
    if args.help:
        parser.print_help()
        sys.exit()

    set_default_arguments(args)

    if (
        args.engine in ["ORT_CUDA", "ORT_TRT"]
        and (args.force_onnx_export or args.force_onnx_optimize)
        and not args.force_engine_build
    ):
        raise ValueError(
            "For ORT_CUDA or ORT_TRT, --force_onnx_export and --force_onnx_optimize are not supported. "
            "Please use --force_engine_build instead."
        )

    # Validate image dimensions
    if args.height % 64 != 0 or args.width % 64 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 64 but specified as: {args.height} and {args.width}."
        )

    if (args.build_dynamic_batch or args.build_dynamic_shape) and not args.disable_cuda_graph:
        print("[I] CUDA Graph is disabled since dynamic input shape is configured.")
        args.disable_cuda_graph = True

    if args.onnx_opset is None:
        args.onnx_opset = 14 if args.engine == "ORT_CUDA" else 17

    if is_xl:
        if args.version == "xl-turbo":
            if args.lcm:
                print("[I] sdxl-turbo cannot use with LCM.")
                args.lcm = False

        assert args.strength > 0.0 and args.strength < 1.0

        assert not (args.lcm and args.lora_weights), "it is not supported to use both lcm unet and Lora together"

    if args.scheduler == "LCM":
        if args.guidance > 2.0:
            print("[I] Use --guidance=0.0 (no more than 2.0) when LCM scheduler is used.")
            args.guidance = 0.0
        if args.denoising_steps > 16:
            print("[I] Use --denoising_steps=8 (no more than 16) when LCM scheduler is used.")
            args.denoising_steps = 8

    print(args)

    return args


def max_batch(args):
    if args.max_batch_size:
        max_batch_size = args.max_batch_size
    else:
        do_classifier_free_guidance = args.guidance > 1.0
        batch_multiplier = 2 if do_classifier_free_guidance else 1
        max_batch_size = 32 // batch_multiplier
        if args.engine != "ORT_CUDA" and (args.build_dynamic_shape or args.height > 512 or args.width > 512):
            max_batch_size = 8 // batch_multiplier
    return max_batch_size


def get_metadata(args, is_xl: bool = False) -> Dict[str, Any]:
    metadata = {
        "command": " ".join(['"' + x + '"' if " " in x else x for x in sys.argv]),
        "args.prompt": args.prompt,
        "args.negative_prompt": args.negative_prompt,
        "args.batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "cuda_graph": not args.disable_cuda_graph,
        "vae_slicing": args.enable_vae_slicing,
        "engine": args.engine,
    }

    if args.lora_weights:
        metadata["lora_weights"] = args.lora_weights
        metadata["lora_scale"] = args.lora_scale

    if args.controlnet_type:
        metadata["controlnet_type"] = args.controlnet_type
        metadata["controlnet_scale"] = args.controlnet_scale

    if is_xl and args.enable_refiner:
        metadata["base.scheduler"] = args.scheduler
        metadata["base.denoising_steps"] = args.denoising_steps
        metadata["base.guidance"] = args.guidance
        metadata["refiner.strength"] = args.strength
        metadata["refiner.scheduler"] = args.refiner_scheduler
        metadata["refiner.denoising_steps"] = args.refiner_denoising_steps
        metadata["refiner.guidance"] = args.refiner_guidance
    else:
        metadata["scheduler"] = args.scheduler
        metadata["denoising_steps"] = args.denoising_steps
        metadata["guidance"] = args.guidance

    # Version of installed python packages
    packages = ""
    for name in [
        "onnxruntime-gpu",
        "torch",
        "tensorrt",
        "transformers",
        "diffusers",
        "onnx",
        "onnx-graphsurgeon",
        "polygraphy",
        "controlnet_aux",
    ]:
        try:
            packages += (" " if packages else "") + f"{name}=={version(name)}"
        except PackageNotFoundError:
            continue
    metadata["packages"] = packages
    metadata["device"] = torch.cuda.get_device_name()
    metadata["torch.version.cuda"] = torch.version.cuda

    return metadata


def repeat_prompt(args):
    if not isinstance(args.prompt, list):
        raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    prompt = args.prompt * args.batch_size

    if not isinstance(args.negative_prompt, list):
        raise ValueError(
            f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}"
        )

    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    return prompt, negative_prompt


def init_pipeline(
    pipeline_class, pipeline_info, engine_type, args, max_batch_size, opt_batch_size, opt_image_height, opt_image_width
):
    onnx_dir, engine_dir, output_dir, framework_model_dir, timing_cache = get_engine_paths(
        work_dir=args.work_dir, pipeline_info=pipeline_info, engine_type=engine_type
    )

    # Initialize demo
    pipeline = pipeline_class(
        pipeline_info,
        scheduler=args.refiner_scheduler if pipeline_info.is_xl_refiner() else args.scheduler,
        output_dir=output_dir,
        hf_token=args.hf_token,
        verbose=False,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size,
        use_cuda_graph=not args.disable_cuda_graph,
        framework_model_dir=framework_model_dir,
        engine_type=engine_type,
    )

    if engine_type == EngineType.ORT_CUDA:
        # Build CUDA EP engines and load pytorch modules
        pipeline.backend.build_engines(
            engine_dir=engine_dir,
            framework_model_dir=framework_model_dir,
            onnx_dir=onnx_dir,
            tmp_dir=os.path.join(args.work_dir or ".", engine_type.name, pipeline_info.short_name(), "tmp"),
            force_engine_rebuild=args.force_engine_build,
            device_id=torch.cuda.current_device(),
        )
    elif engine_type == EngineType.ORT_TRT:
        # Build TensorRT EP engines and load pytorch modules
        pipeline.backend.build_engines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            args.onnx_opset,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            opt_batch_size=opt_batch_size,
            force_engine_rebuild=args.force_engine_build,
            static_batch=not args.build_dynamic_batch,
            static_image_shape=not args.build_dynamic_shape,
            max_workspace_size=0,
            device_id=torch.cuda.current_device(),
            timing_cache=timing_cache,
        )
    elif engine_type == EngineType.TRT:
        # Load TensorRT engines and pytorch modules
        pipeline.backend.load_engines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            args.onnx_opset,
            opt_batch_size=opt_batch_size,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            force_export=args.force_onnx_export,
            force_optimize=args.force_onnx_optimize,
            force_build=args.force_engine_build,
            static_batch=not args.build_dynamic_batch,
            static_shape=not args.build_dynamic_shape,
            enable_refit=args.build_enable_refit,
            enable_preview=args.build_preview_features,
            enable_all_tactics=args.build_all_tactics,
            timing_cache=timing_cache,
            onnx_refit_dir=args.onnx_refit_dir,
        )

    return pipeline


def get_depth_image(image):
    """
    Create depth map for SDXL depth control net.
    """
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation

    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    # The depth map is 384x384 by default, here we interpolate to the default output size.
    # Note that it will be resized to output image size later. May change the size here to avoid interpolate twice.
    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def get_canny_image(image) -> Image.Image:
    """
    Create canny image for SDXL control net.
    """
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


def process_controlnet_images_xl(args) -> List[Image.Image]:
    """
    Process control image for SDXL control net.
    """
    assert len(args.controlnet_image) == 1
    image = Image.open(args.controlnet_image[0]).convert("RGB")

    controlnet_images = []
    if args.controlnet_type[0] == "canny":
        controlnet_images.append(get_canny_image(image))
    elif args.controlnet_type[0] == "depth":
        controlnet_images.append(get_depth_image(image))
    else:
        raise ValueError(f"This controlnet type is not supported for SDXL or Turbo: {args.controlnet_type}.")

    return controlnet_images


def add_controlnet_arguments(parser, is_xl: bool = False):
    """
    Add control net related arguments.
    """
    group = parser.add_argument_group("Options for ControlNet (only supports SD 1.5 or XL).")

    group.add_argument(
        "-ci",
        "--controlnet-image",
        nargs="*",
        type=str,
        default=[],
        help="Path to the input regular RGB image/images for controlnet",
    )
    group.add_argument(
        "-ct",
        "--controlnet-type",
        nargs="*",
        type=str,
        default=[],
        choices=list(PipelineInfo.supported_controlnet("xl-1.0" if is_xl else "1.5").keys()),
        help="A list of controlnet type",
    )
    group.add_argument(
        "-cs",
        "--controlnet-scale",
        nargs="*",
        type=float,
        default=[],
        help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet. Default is 0.5 for SDXL, or 1.0 for SD 1.5",
    )


def process_controlnet_image(controlnet_type: str, image: Image.Image, height, width):
    """
    Process control images of control net v1.1 for Stable Diffusion 1.5.
    """
    control_image = None
    shape = (height, width)
    image = image.convert("RGB")
    if controlnet_type == "canny":
        canny_image = controlnet_aux.CannyDetector()(image)
        control_image = canny_image.resize(shape)
    elif controlnet_type == "normalbae":
        normal_image = controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = normal_image.resize(shape)
    elif controlnet_type == "depth":
        depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = depth_image.resize(shape)
    elif controlnet_type == "mlsd":
        mlsd_image = controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = mlsd_image.resize(shape)
    elif controlnet_type == "openpose":
        openpose_image = controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/Annotators")(image)
        control_image = openpose_image.resize(shape)
    elif controlnet_type == "scribble":
        scribble_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(image, scribble=True)
        control_image = scribble_image.resize(shape)
    elif controlnet_type == "seg":
        seg_image = controlnet_aux.SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")(
            image
        )
        control_image = seg_image.resize(shape)
    else:
        raise ValueError(f"There is no demo image of this controlnet_type: {controlnet_type}")
    return control_image


def process_controlnet_arguments(args):
    """
    Process control net arguments, and returns a list of control images and a tensor of control net scales.
    """
    assert isinstance(args.controlnet_type, list)
    assert isinstance(args.controlnet_scale, list)
    assert isinstance(args.controlnet_image, list)

    if len(args.controlnet_image) != len(args.controlnet_type):
        raise ValueError(
            f"Numbers of controlnet_image {len(args.controlnet_image)} should be equal to number of controlnet_type {len(args.controlnet_type)}."
        )

    if len(args.controlnet_type) == 0:
        return None, None

    if args.version not in ["1.5", "xl-1.0", "xl-turbo"]:
        raise ValueError("This demo only supports ControlNet in Stable Diffusion 1.5, XL or Turbo.")

    is_xl = "xl" in args.version
    if is_xl and len(args.controlnet_type) > 1:
        raise ValueError("This demo only support one ControlNet for Stable Diffusion XL or Turbo.")

    if len(args.controlnet_scale) == 0:
        args.controlnet_scale = [0.5 if is_xl else 1.0] * len(args.controlnet_type)
    elif len(args.controlnet_type) != len(args.controlnet_scale):
        raise ValueError(
            f"Numbers of controlnet_type {len(args.controlnet_type)} should be equal to number of controlnet_scale {len(args.controlnet_scale)}."
        )

    # Convert controlnet scales to tensor
    controlnet_scale = torch.FloatTensor(args.controlnet_scale)

    if is_xl:
        images = process_controlnet_images_xl(args)
    else:
        images = []
        for i, image in enumerate(args.controlnet_image):
            images.append(process_controlnet_image(args.controlnet_type[i], Image.open(image), args.height, args.width))

    return images, controlnet_scale
