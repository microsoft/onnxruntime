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
from cuda import cudart
from demo_utils import init_pipeline, parse_arguments, repeat_prompt
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_type
from pipeline_txt2img import Txt2ImgPipeline

if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    args = parse_arguments(is_xl=False, description="Options for Stable Diffusion Demo")
    prompt, negative_prompt = repeat_prompt(args)

    image_height = args.height
    image_width = args.width

    # Register TensorRT plugins
    engine_type = get_engine_type(args.engine)
    if engine_type == EngineType.TRT:
        from trt_utilities import init_trt_plugins

        init_trt_plugins()

    max_batch_size = 16
    if engine_type != EngineType.ORT_CUDA and (args.build_dynamic_shape or image_height > 512 or image_width > 512):
        max_batch_size = 4

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
    pipeline_info = PipelineInfo(args.version, min_image_size=min_image_size, max_image_size=max_image_size)

    # Ideally, the optimized batch size and image size for TRT engine shall align with user's preference. That is to
    # optimize the shape used most frequently. We can let user config it when we develop a UI plugin.
    # In this demo, we optimize batch size 1 and image size 512x512 (or 768x768 for SD 2.0/2.1) for dynamic engine.
    # This is mainly for benchmark purpose to simulate the case that we have no knowledge of user's preference.
    opt_batch_size = 1 if args.build_dynamic_batch else batch_size
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
            return_type="image",
        )

    if not args.disable_cuda_graph:
        # inference once to get cuda graph
        _image, _latency = run_inference(warmup=True)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        _image, _latency = run_inference(warmup=True)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    _image, _latency = run_inference(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    pipeline.teardown()
