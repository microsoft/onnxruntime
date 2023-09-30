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

import torch
from cuda import cudart
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_type
from pipeline_img2img_xl import Img2ImgXLPipeline
from pipeline_txt2img_xl import Txt2ImgXLPipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Demo", conflict_handler="resolve")
    parser.add_argument(
        "--engine",
        type=str,
        default="ORT_TRT",
        choices=["ORT_TRT", "TRT"],
        help="Backend engine. Default is ORT_TRT, which means OnnxRuntime TensorRT execution provider.",
    )

    parser.add_argument(
        "--version", type=str, default="xl-1.0", choices=["xl-1.0"], help="Version of Stable Diffusion XL"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Height of image to generate (must be multiple of 8). Default is 1024."
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Height of image to generate (must be multiple of 8). Default is 1024."
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDIM",
        choices=["DDIM", "EulerA", "UniPC"],
        help="Scheduler for diffusion process",
    )

    parser.add_argument(
        "--work-dir",
        default="",
        help="Root Directory to store torch or ONNX models, built engines and output images etc",
    )

    parser.add_argument("prompt", nargs="+", help="Text prompt(s) to guide image generation")

    parser.add_argument(
        "--negative-prompt", nargs="*", default=[""], help="Optional negative prompt(s) to guide the image generation."
    )
    parser.add_argument(
        "--repeat-prompt",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Number of times to repeat the prompt (batch size multiplier). Default is 1.",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=30,
        help="Number of denoising steps in each of base and refiner. Default is 30.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="Higher guidance scale encourages to generate images that are closely linked to the text prompt.",
    )

    # ONNX export
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        choices=range(14, 18),
        help="Select ONNX opset version to target for exported models. Default is 17.",
    )
    parser.add_argument(
        "--force-onnx-export", action="store_true", help="Force ONNX export of CLIP, UNET, and VAE models"
    )
    parser.add_argument(
        "--force-onnx-optimize", action="store_true", help="Force ONNX optimizations for CLIP, UNET, and VAE models"
    )

    # Framework model ckpt
    parser.add_argument("--framework-model-dir", default="pytorch_model", help="Directory for HF saved models")
    parser.add_argument("--hf-token", type=str, help="HuggingFace API access token for downloading model checkpoints")

    # Engine build options.
    parser.add_argument("--force-engine-build", action="store_true", help="Force rebuilding the TensorRT engine")
    parser.add_argument(
        "--build-dynamic-batch", action="store_true", help="Build TensorRT engines to support dynamic batch size."
    )
    parser.add_argument(
        "--build-dynamic-shape", action="store_true", help="Build TensorRT engines to support dynamic image sizes."
    )

    # Inference related options
    parser.add_argument(
        "--num-warmup-runs", type=int, default=5, help="Number of warmup runs before benchmarking performance"
    )
    parser.add_argument("--nvtx-profile", action="store_true", help="Enable NVTX markers for performance profiling")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator to get consistent results")
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable cuda graph.")

    # TensorRT only options
    group = parser.add_argument_group("Options for TensorRT (--engine=TRT) only")
    group.add_argument("--onnx-refit-dir", help="ONNX models to load the weights from")
    group.add_argument(
        "--build-enable-refit", action="store_true", help="Enable Refit option in TensorRT engines during build."
    )
    group.add_argument(
        "--build-preview-features", action="store_true", help="Build TensorRT engines with preview features."
    )
    group.add_argument(
        "--build-all-tactics", action="store_true", help="Build TensorRT engines using all tactic sources."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Process prompt
    if not isinstance(args.prompt, list):
        raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    prompt = args.prompt * args.repeat_prompt

    if not isinstance(args.negative_prompt, list):
        raise ValueError(
            f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}"
        )
    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    # Validate image dimensions
    image_height = args.height
    image_width = args.width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}."
        )

    # Register TensorRT plugins
    engine_type = get_engine_type(args.engine)
    if engine_type == EngineType.TRT:
        from trt_utilities import init_trt_plugins

        init_trt_plugins()

    max_batch_size = 16
    if args.build_dynamic_shape or image_height > 512 or image_width > 512:
        max_batch_size = 4

    batch_size = len(prompt)
    if batch_size > max_batch_size:
        raise ValueError(
            f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4"
        )

    use_cuda_graph = not args.disable_cuda_graph
    if use_cuda_graph and (args.build_dynamic_batch or args.build_dynamic_shape):
        raise ValueError(
            "Using CUDA graph requires static dimensions. Do not specify `--build-dynamic-batch` and do not specify `--build-dynamic-shape`"
        )

    def init_pipeline(pipeline_class, pipeline_info, engine_type):
        short_name = pipeline_info.short_name()
        work_dir = args.work_dir or engine_type.name

        onnx_dir = os.path.join(work_dir, short_name, "onnx")
        engine_dir = os.path.join(work_dir, short_name, f"engine_{batch_size}_{image_height}_{image_width}")
        output_dir = os.path.join(work_dir, short_name, "output")
        framework_model_dir = os.path.join(work_dir, "torch_model")
        timing_cache = os.path.join(work_dir, "timing_cache")

        # Initialize demo
        pipeline = pipeline_class(
            pipeline_info,
            scheduler=args.scheduler,
            output_dir=output_dir,
            hf_token=args.hf_token,
            verbose=False,
            nvtx_profile=args.nvtx_profile,
            max_batch_size=max_batch_size,
            use_cuda_graph=use_cuda_graph,
            framework_model_dir=framework_model_dir,
            engine_type=engine_type,
        )

        # Load TensorRT engines and pytorch modules
        if engine_type == EngineType.ORT_TRT:
            pipeline.backend.build_engines(
                engine_dir,
                framework_model_dir,
                onnx_dir,
                args.onnx_opset,
                opt_image_height=image_height,
                opt_image_width=image_width,
                opt_batch_size=len(prompt),
                # force_export=args.force_onnx_export,
                # force_optimize=args.force_onnx_optimize,
                force_engine_rebuild=args.force_engine_build,
                static_batch=not args.build_dynamic_batch,
                static_image_shape=not args.build_dynamic_shape,
                max_workspace_size=0,
                device_id=torch.cuda.current_device(),
            )
        elif engine_type == EngineType.TRT:
            # Load TensorRT engines and pytorch modules
            pipeline.backend.load_engines(
                engine_dir,
                framework_model_dir,
                onnx_dir,
                args.onnx_opset,
                opt_batch_size=len(prompt),
                opt_image_height=image_height,
                opt_image_width=image_width,
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

    base_info = PipelineInfo(args.version)
    base = init_pipeline(Txt2ImgXLPipeline, base_info, engine_type)

    refiner_info = PipelineInfo(args.version, is_sd_xl_refiner=True)
    refiner = init_pipeline(Img2ImgXLPipeline, refiner_info, engine_type)

    if engine_type == EngineType.TRT:
        max_device_memory = max(base.backend.max_device_memory(), refiner.backend.max_device_memory())
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        base.backend.activate_engines(shared_device_memory)
        refiner.backend.activate_engines(shared_device_memory)

    base.load_resources(image_height, image_width, batch_size)
    refiner.load_resources(image_height, image_width, batch_size)

    def run_sd_xl_inference(warmup=False):
        images, time_base = base.run(
            prompt,
            negative_prompt,
            image_height,
            image_width,
            warmup=warmup,
            denoising_steps=args.denoising_steps,
            guidance=args.guidance,
            seed=args.seed,
            return_type="latents",
        )
        images, time_refiner = refiner.run(
            prompt, negative_prompt, images, image_height, image_width, warmup=warmup, seed=args.seed
        )
        return images, time_base + time_refiner

    if use_cuda_graph:
        # inference once to get cuda graph
        images, _ = run_sd_xl_inference(warmup=True)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        images, _ = run_sd_xl_inference(warmup=True)

    print("[I] Running StableDiffusion XL pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    images, pipeline_time = run_sd_xl_inference(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    print("|------------|--------------|")
    print("| {:^10} | {:>9.2f} ms |".format("e2e", pipeline_time))
    print("|------------|--------------|")

    base.teardown()
    refiner.teardown()
