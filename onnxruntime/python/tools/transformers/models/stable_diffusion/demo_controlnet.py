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
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_type
from PIL import Image
from pipeline_txt2img import Txt2ImgPipeline

if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    parser = arg_parser("Options for Stable Diffusion Control Net Demo")
    parser.add_argument(
        "--input-image",
        nargs="+",
        type=str,
        default=[],
        help="Path to the input image/images already prepared for ControlNet modality. For example: canny edged image for canny ControlNet, not just regular rgb image",
    )
    parser.add_argument(
        "--controlnet-type",
        nargs="+",
        type=str,
        default=["canny"],
        choices=list(PipelineInfo.supported_controlnet().keys()),
        help="A list of controlnet type",
    )
    parser.add_argument(
        "--controlnet-scale",
        nargs="+",
        type=float,
        default=[1.0],
        help="The outputs of the controlnet are multiplied by `controlnet_scale` before they are added to the residual in the original unet, can be `None`, `float` or `float` list",
    )
    args = parse_arguments(is_xl=False, parser=parser)

    # Controlnet configuration
    if not isinstance(args.controlnet_type, list):
        raise ValueError(
            f"`--controlnet-type` must be of type `str` or `str` list, but is {type(args.controlnet_type)}"
        )

    # Controlnet configuration
    if not isinstance(args.controlnet_scale, list):
        raise ValueError(
            f"`--controlnet-scale`` must be of type `float` or `float` list, but is {type(args.controlnet_scale)}"
        )

    # Check number of ControlNets to ControlNet scales
    if len(args.controlnet_type) != len(args.controlnet_scale):
        raise ValueError(
            f"Numbers of ControlNets {len(args.controlnet_type)} should be equal to number of ControlNet scales {len(args.controlnet_scale)}."
        )

    assert len(args.controlnet_type) <= 2

    # Convert controlnet scales to tensor
    controlnet_scale = torch.FloatTensor(args.controlnet_scale)

    input_images = []
    if len(args.input_image) > 0:
        for image in args.input_image:
            input_images.append(Image.open(image))
    else:
        for controlnet in args.controlnet_type:
            if controlnet == "canny":
                canny_image = download_image(
                    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
                )
                canny_image = controlnet_aux.CannyDetector()(canny_image)
                input_images.append(canny_image.resize((args.height, args.width)))
            elif controlnet == "normal":
                normal_image = download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-normal/resolve/main/images/toy.png"
                )
                normal_image = controlnet_aux.NormalBaeDetector.from_pretrained("lllyasviel/Annotators")(normal_image)
                input_images.append(normal_image.resize((args.height, args.width)))
            elif controlnet == "depth":
                depth_image = download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png"
                )
                depth_image = controlnet_aux.LeresDetector.from_pretrained("lllyasviel/Annotators")(depth_image)
                input_images.append(depth_image.resize((args.height, args.width)))
            elif controlnet == "mlsd":
                mlsd_image = download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-mlsd/resolve/main/images/room.png"
                )
                mlsd_image = controlnet_aux.MLSDdetector.from_pretrained("lllyasviel/Annotators")(mlsd_image)
                input_images.append(mlsd_image.resize((args.height, args.width)))
            elif controlnet == "openpose":
                openpose_image = download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"
                )
                openpose_image = controlnet_aux.OpenposeDetector.from_pretrained("lllyasviel/Annotators")(
                    openpose_image
                )
                input_images.append(openpose_image.resize((args.height, args.width)))
            elif controlnet == "scribble":
                scribble_image = download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/images/bag.png"
                )
                scribble_image = controlnet_aux.HEDdetector.from_pretrained("lllyasviel/Annotators")(
                    scribble_image, scribble=True
                )
                input_images.append(scribble_image.resize((args.height, args.width)))
            elif controlnet == "seg":
                seg_image = download_image(
                    "https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/images/house.png"
                )
                seg_image = controlnet_aux.SamDetector.from_pretrained(
                    "ybelkada/segment-anything", subfolder="checkpoints"
                )(seg_image)
                input_images.append(seg_image.resize((args.height, args.width)))
            else:
                raise ValueError(f"You should implement the conditional image of this controlnet: {controlnet}")

    prompt, negative_prompt = repeat_prompt(args)

    image_height = args.height
    image_width = args.width

    # Register TensorRT plugins
    engine_type = get_engine_type(args.engine)
    if engine_type == EngineType.TRT:
        from trt_utilities import init_trt_plugins

        init_trt_plugins()

    max_batch_size = max_batch(args)

    batch_size = len(prompt)
    if batch_size > max_batch_size:
        raise ValueError(
            f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4"
        )

    # For TensorRT,  performance of engine built with dynamic shape is very sensitive to the range of image size.
    # Here, we reduce the range of image size for TensorRT to trade-off flexibility and performance.
    # This range can cover common used shape of landscape 512x768, portrait 768x512, or square 512x512 and 768x768.
    min_image_size = 512 if args.engine != "ORT_CUDA" else 256
    max_image_size = 768 if args.engine != "ORT_CUDA" else 1024
    pipeline_info = PipelineInfo(
        args.version,
        min_image_size=min_image_size,
        max_image_size=max_image_size,
        do_classifier_free_guidance=(args.guidance > 1.0),
        controlnet=args.controlnet_type,
        lora_weights=args.lora_weights,
        lora_scale=args.lora_scale,
    )

    # Ideally, the optimized batch size and image size for TRT engine shall align with user's preference. That is to
    # optimize the shape used most frequently. We can let user config it when we develop a UI plugin.
    # In this demo, we optimize batch size 4 and image size 512x512 (or 768x768 for SD 2.0/2.1) for dynamic engine.
    # This is mainly for benchmark purpose to simulate the case that we have no knowledge of user's preference.
    opt_batch_size = 4 if args.build_dynamic_batch else batch_size
    opt_image_height = pipeline_info.default_image_size() if args.build_dynamic_shape else args.height
    opt_image_width = pipeline_info.default_image_size() if args.build_dynamic_shape else args.width

    pipeline = init_pipeline(
        Txt2ImgPipeline,
        pipeline_info,
        engine_type,
        args,
        max_batch_size,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
    )

    if engine_type == EngineType.TRT:
        max_device_memory = max(pipeline.backend.max_device_memory(), pipeline.backend.max_device_memory())
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        pipeline.backend.activate_engines(shared_device_memory)

    if engine_type == EngineType.ORT_CUDA and args.enable_vae_slicing:
        pipeline.backend.enable_vae_slicing()

    pipeline.load_resources(image_height, image_width, batch_size)

    def run_inference(warmup=False):
        return pipeline.run(
            prompt,
            negative_prompt,
            image_height,
            image_width,
            warmup=warmup,
            denoising_steps=args.denoising_steps,
            guidance=args.guidance,
            seed=args.seed,
            controlnet_images=input_images,
            controlnet_scales=controlnet_scale,
            return_type="image",
        )

    if not args.disable_cuda_graph:
        # inference once to get cuda graph
        _, _ = run_inference(warmup=True)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        _, _ = run_inference(warmup=True)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images, perf_data = run_inference(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    metadata = get_metadata(args, False)
    metadata.update(pipeline.metadata())
    if perf_data:
        metadata.update(perf_data)
    metadata["images"] = len(images)
    print(metadata)
    pipeline.save_images(images, prompt, negative_prompt, metadata)

    pipeline.teardown()
