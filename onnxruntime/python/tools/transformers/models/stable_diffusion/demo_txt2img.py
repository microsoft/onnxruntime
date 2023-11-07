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

    pipeline_info = PipelineInfo(args.version)
    pipeline = init_pipeline(Txt2ImgPipeline, pipeline_info, engine_type, args, max_batch_size, batch_size)

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
