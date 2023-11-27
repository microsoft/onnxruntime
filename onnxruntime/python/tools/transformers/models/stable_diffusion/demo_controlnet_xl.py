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

import coloredlogs
import controlnet_aux
import torch
from cuda import cudart
from demo_utils import (
    arg_parser,
    download_image,
    get_metadata,
    init_pipeline,
    max_batch,
    parse_arguments,
    repeat_prompt,
)
from diffusers.utils import load_image
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_type
from PIL import Image
from pipeline_txt2img_xl import Txt2ImgXLPipeline


def load_pipeline(args, batch_size):
    # Register TensorRT plugins
    engine_type = get_engine_type(args.engine)
    if engine_type == EngineType.TRT:
        from trt_utilities import init_trt_plugins

        init_trt_plugins()

    max_batch_size = max_batch(args)

    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {batch_size} is larger than allowed {max_batch_size}.")

    # For TensorRT,  performance of engine built with dynamic shape is very sensitive to the range of image size.
    # Here, we reduce the range of image size for TensorRT to trade-off flexibility and performance.
    # This range can cover most frequent shape of landscape (832x1216), portrait (1216x832) or square (1024x1024).
    min_image_size = 832 if args.engine != "ORT_CUDA" else 512
    max_image_size = 1216 if args.engine != "ORT_CUDA" else 2048

    # No VAE decoder in base when it outputs latent instead of image.
    base_info = PipelineInfo(
        args.version,
        use_vae=True,
        min_image_size=min_image_size,
        max_image_size=max_image_size,
        use_lcm=args.lcm,
        do_classifier_free_guidance=(args.guidance > 1.0),
        controlnet=[args.controlnet_type],
        lora_weights=args.lora_weights,
        lora_scale=args.lora_scale,
    )

    # Ideally, the optimized batch size and image size for TRT engine shall align with user's preference. That is to
    # optimize the shape used most frequently. We can let user config it when we develop a UI plugin.
    # In this demo, we optimize batch size 1 and image size 1024x1024 for SD XL dynamic engine.
    # This is mainly for benchmark purpose to simulate the case that we have no knowledge of user's preference.
    opt_batch_size = 1 if args.build_dynamic_batch else batch_size
    opt_image_height = base_info.default_image_size() if args.build_dynamic_shape else args.height
    opt_image_width = base_info.default_image_size() if args.build_dynamic_shape else args.width

    base = init_pipeline(
        Txt2ImgXLPipeline,
        base_info,
        engine_type,
        args,
        max_batch_size,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
    )

    if engine_type == EngineType.TRT:
        max_device_memory = max(base.backend.max_device_memory(), base.backend.max_device_memory())
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        base.backend.activate_engines(shared_device_memory)

    if engine_type == EngineType.ORT_CUDA:
        enable_vae_slicing = args.enable_vae_slicing
        if batch_size > 4 and not enable_vae_slicing:
            print("Updating enable_vae_slicing to be True to avoid cuDNN error for batch size > 4.")
            enable_vae_slicing = True
        if enable_vae_slicing:
            base.backend.enable_vae_slicing()
    return base


def run_pipeline(args, base, prompt, negative_prompt, controlnet_images, controlnet_scales):
    image_height = args.height
    image_width = args.width
    batch_size = len(prompt)
    base.load_resources(image_height, image_width, batch_size)

    def run_base(warmup=False):
        images, base_perf = base.run(
            prompt,
            negative_prompt,
            image_height,
            image_width,
            warmup=warmup,
            denoising_steps=args.denoising_steps,
            guidance=args.guidance,
            seed=args.seed,
            controlnet_images=controlnet_images,
            controlnet_scales=controlnet_scales,
            return_type="image",
        )
        return images, base_perf

    if not args.disable_cuda_graph:
        # inference once to get cuda graph
        _, _ = run_base(warmup=True)

    if args.num_warmup_runs > 0:
        print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        _, _ = run_base(warmup=True)

    print("[I] Running StableDiffusion XL pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images, perf_data = run_base(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    metadata = get_metadata(args, True)
    metadata.update({"base." + key: val for key, val in base.metadata().items()})
    if perf_data:
        metadata.update(perf_data)
    metadata["images"] = len(images)
    print(metadata)
    base.save_images(images, prompt, negative_prompt, metadata)


def run_demo(args, controlnet_images, controlnet_scales):
    """Run Stable Diffusion XL Base to generate an image."""
    prompt, negative_prompt = repeat_prompt(args)
    batch_size = len(prompt)
    base = load_pipeline(args, batch_size)
    run_pipeline(args, base, prompt, negative_prompt, controlnet_images, controlnet_scales)
    base.teardown()


def get_depth_map(image):
    import numpy as np
    import torch
    from PIL import Image
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


if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    parser = arg_parser("Options for Stable Diffusion XL Control Net Demo")
    parser.add_argument(
        "--controlnet-image",
        type=str,
        default=None,
        help="Path to the input image/images already prepared for ControlNet modality. For example: canny edged image for canny ControlNet, not just regular rgb image",
    )
    parser.add_argument(
        "--controlnet-type",
        type=str,
        default="canny",
        choices=list(PipelineInfo.supported_controlnet("xl-1.0").keys()),
        help="Controlnet type",
    )
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=0.4,
        help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet",
    )
    args = parse_arguments(is_xl=True, parser=parser, disable_refiner=True)

    controlnet_scales = torch.FloatTensor([args.controlnet_scale])

    controlnet_images = []
    if args.controlnet_image:
        controlnet_images.append(Image.open(args.controlnet_image))
    elif args.controlnet_type == "canny":
        canny_image = download_image(
            "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        )
        canny_image = controlnet_aux.CannyDetector()(canny_image)
        controlnet_images.append(canny_image.resize((args.height, args.width)))
    elif args.controlnet_type == "depth":
        depth_image = load_image(
            "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
        )
        controlnet_images.append(get_depth_map(depth_image))
    else:
        raise ValueError(f"You should implement the conditional image of this controlnet: {args.controlnet_type}")

    run_demo(args, controlnet_images, controlnet_scales)
