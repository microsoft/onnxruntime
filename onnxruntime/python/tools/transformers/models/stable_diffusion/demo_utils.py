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
from io import BytesIO
from typing import Any, Dict, List

import controlnet_aux
import cv2
import numpy as np
import requests
import torch
from diffusers.utils import load_image
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_paths
from PIL import Image


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def arg_parser(description: str):
    return argparse.ArgumentParser(description=description, formatter_class=RawTextArgumentDefaultsHelpFormatter)


def parse_arguments(is_xl: bool, parser):
    engines = ["ORT_CUDA", "ORT_TRT", "TRT"]

    parser.add_argument(
        "--engine",
        type=str,
        default=engines[0],
        choices=engines,
        help="Backend engine in {engines}. "
        "ORT_CUDA is CUDA execution provider; ORT_TRT is Tensorrt execution provider; TRT is TensorRT",
    )

    supported_versions = PipelineInfo.supported_versions(is_xl)
    parser.add_argument(
        "--version",
        type=str,
        default=supported_versions[-1] if is_xl else "1.5",
        choices=supported_versions,
        help="Version of Stable Diffusion" + (" XL." if is_xl else "."),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024 if is_xl else 512,
        help="Height of image to generate (must be multiple of 8).",
    )
    parser.add_argument(
        "--width", type=int, default=1024 if is_xl else 512, help="Height of image to generate (must be multiple of 8)."
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDIM",
        choices=["DDIM", "UniPC", "LCM"] if is_xl else ["DDIM", "EulerA", "UniPC", "LCM"],
        help="Scheduler for diffusion process" + " of base" if is_xl else "",
    )

    parser.add_argument(
        "--work-dir",
        default=".",
        help="Root Directory to store torch or ONNX models, built engines and output images etc.",
    )

    parser.add_argument("prompt", nargs="*", default=[""], help="Text prompt(s) to guide image generation.")

    parser.add_argument(
        "--negative-prompt", nargs="*", default=[""], help="Optional negative prompt(s) to guide the image generation."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Number of times to repeat the prompt (batch size multiplier).",
    )

    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=30 if is_xl else 50,
        help="Number of denoising steps" + (" in base." if is_xl else "."),
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0 if is_xl else 7.5,
        help="Higher guidance scale encourages to generate images that are closely linked to the text prompt.",
    )

    parser.add_argument(
        "--lora-scale", type=float, default=1, help="Scale of LoRA weights, default 1 (must between 0 and 1)"
    )
    parser.add_argument("--lora-weights", type=str, default="", help="LoRA weights to apply in the base model")

    if is_xl:
        parser.add_argument(
            "--lcm",
            action="store_true",
            help="Use fine-tuned latent consistency model to replace the UNet in base.",
        )

        parser.add_argument(
            "--refiner-scheduler",
            type=str,
            default="DDIM",
            choices=["DDIM", "UniPC"],
            help="Scheduler for diffusion process of refiner.",
        )

        parser.add_argument(
            "--refiner-guidance",
            type=float,
            default=5.0,
            help="Guidance scale used in refiner.",
        )

        parser.add_argument(
            "--refiner-steps",
            type=int,
            default=30,
            help="Number of denoising steps in refiner. Note that actual refiner steps is refiner_steps * strength.",
        )

        parser.add_argument(
            "--strength",
            type=float,
            default=0.3,
            help="A value between 0 and 1. The higher the value less the final image similar to the seed image.",
        )

        parser.add_argument(
            "--disable-refiner", action="store_true", help="Disable refiner and only run base for XL pipeline."
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
        "--build-dynamic-batch", action="store_true", help="Build TensorRT engines to support dynamic batch size."
    )
    parser.add_argument(
        "--build-dynamic-shape", action="store_true", help="Build TensorRT engines to support dynamic image sizes."
    )

    # Inference related options
    parser.add_argument(
        "--num-warmup-runs", type=int, default=5, help="Number of warmup runs before benchmarking performance."
    )
    parser.add_argument("--nvtx-profile", action="store_true", help="Enable NVTX markers for performance profiling.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator to get consistent results.")
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable cuda graph.")

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
        if args.lcm and args.scheduler != "LCM":
            print("[I] Use --scheduler=LCM for base since LCM is used.")
            args.scheduler = "LCM"

        assert args.strength > 0.0 and args.strength < 1.0

        assert not (args.lcm and args.lora_weights), "it is not supported to use both lcm unet and Lora together"

    if args.scheduler == "LCM":
        if args.guidance > 1.0:
            print("[I] Use --guidance=1.0 for base since LCM is used.")
            args.guidance = 1.0
        if args.denoising_steps > 16:
            print("[I] Use --denoising_steps=8 (no more than 16) for base since LCM is used.")
            args.denoising_steps = 8

    print(args)

    return args


def max_batch(args):
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

    if is_xl and not args.disable_refiner:
        metadata["base.scheduler"] = args.scheduler
        metadata["base.denoising_steps"] = args.denoising_steps
        metadata["base.guidance"] = args.guidance
        metadata["refiner.strength"] = args.strength
        metadata["refiner.scheduler"] = args.refiner_scheduler
        metadata["refiner.denoising_steps"] = args.refiner_steps
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
    image = None
    if args.controlnet_image:
        image = Image.open(args.controlnet_image[0])
    else:
        # If no image is provided, download an image for demo purpose.
        if args.controlnet_type[0] == "canny":
            image = load_image(
                "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
            )
        elif args.controlnet_type[0] == "depth":
            image = load_image(
                "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
            )

    controlnet_images = []
    if args.controlnet_type[0] == "canny":
        controlnet_images.append(get_canny_image(image))
    elif args.controlnet_type[0] == "depth":
        controlnet_images.append(get_depth_image(image))
    else:
        raise ValueError(f"The controlnet is not supported for SDXL: {args.controlnet_type}")

    return controlnet_images


def add_controlnet_arguments(parser, is_xl: bool = False):
    """
    Add control net related arguments.
    """
    group = parser.add_argument_group("Options for ControlNet (only supports SD 1.5 or XL).")

    group.add_argument(
        "--controlnet-image",
        nargs="*",
        type=str,
        default=[],
        help="Path to the input regular RGB image/images for controlnet",
    )
    group.add_argument(
        "--controlnet-type",
        nargs="*",
        type=str,
        default=[],
        choices=list(PipelineInfo.supported_controlnet("xl-1.0" if is_xl else "1.5").keys()),
        help="A list of controlnet type",
    )
    group.add_argument(
        "--controlnet-scale",
        nargs="*",
        type=float,
        default=[],
        help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet. Default is 0.35 for SDXL, or 1.0 for SD 1.5",
    )


def download_image(url) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def controlnet_demo_images(controlnet_list: List[str], height, width) -> List[Image.Image]:
    """
    Return demo images of control net v1.1 for Stable Diffusion 1.5.
    """
    control_images = []
    shape = (height, width)
    for controlnet in controlnet_list:
        if controlnet == "canny":
            canny_image = download_image(
                "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
            )
            canny_image = controlnet_aux.CannyDetector()(canny_image)
            control_images.append(canny_image.resize(shape))
        elif controlnet == "normalbae":
            normal_image = download_image(
                "https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/images/toy.png"
            )
            normal_image = controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")(normal_image)
            control_images.append(normal_image.resize(shape))
        elif controlnet == "depth":
            depth_image = download_image(
                "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
            )
            depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(depth_image)
            control_images.append(depth_image.resize(shape))
        elif controlnet == "mlsd":
            mlsd_image = download_image(
                "https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png"
            )
            mlsd_image = controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/Annotators")(mlsd_image)
            control_images.append(mlsd_image.resize(shape))
        elif controlnet == "openpose":
            openpose_image = download_image(
                "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"
            )
            openpose_image = controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/Annotators")(openpose_image)
            control_images.append(openpose_image.resize(shape))
        elif controlnet == "scribble":
            scribble_image = download_image(
                "https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png"
            )
            scribble_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(
                scribble_image, scribble=True
            )
            control_images.append(scribble_image.resize(shape))
        elif controlnet == "seg":
            seg_image = download_image(
                "https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/images/house.png"
            )
            seg_image = controlnet_aux.SamDetector.from_pretrained(
                "ybelkada/segment-anything", subfolder="checkpoints"
            )(seg_image)
            control_images.append(seg_image.resize(shape))
        else:
            raise ValueError(f"There is no demo image of this controlnet: {controlnet}")
    return control_images


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
    if args.version not in ["1.5", "xl-1.0"]:
        raise ValueError("This demo only supports ControlNet in Stable Diffusion 1.5 or XL.")

    is_xl = args.version == "xl-1.0"
    if is_xl and len(args.controlnet_type) > 1:
        raise ValueError("This demo only support one ControlNet for Stable Diffusion XL.")

    if len(args.controlnet_image) != 0 and len(args.controlnet_image) != len(args.controlnet_scale):
        raise ValueError(
            f"Numbers of ControlNets {len(args.controlnet_image)} should be equal to number of ControlNet scales {len(args.controlnet_scale)}."
        )

    if len(args.controlnet_type) == 0:
        return None, None

    if len(args.controlnet_scale) == 0:
        args.controlnet_scale = [0.5 if is_xl else 1.0] * len(args.controlnet_type)
    elif len(args.controlnet_type) != len(args.controlnet_scale):
        raise ValueError(
            f"Numbers of ControlNets {len(args.controlnet_type)} should be equal to number of ControlNet scales {len(args.controlnet_scale)}."
        )

    # Convert controlnet scales to tensor
    controlnet_scale = torch.FloatTensor(args.controlnet_scale)

    if is_xl:
        images = process_controlnet_images_xl(args)
    else:
        images = []
        if len(args.controlnet_image) > 0:
            for i, image in enumerate(args.controlnet_image):
                images.append(
                    process_controlnet_image(args.controlnet_type[i], Image.open(image), args.height, args.width)
                )
        else:
            images = controlnet_demo_images(args.controlnet_type, args.height, args.width)

    return images, controlnet_scale
