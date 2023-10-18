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
from pipeline_img2img_xl import Img2ImgXLPipeline
from pipeline_txt2img_xl import Txt2ImgXLPipeline


def run_demo():
    """Run Stable Diffusion XL Base + Refiner together (known as ensemble of expert denoisers) to generate an image."""

    args = parse_arguments(is_xl=True, description="Options for Stable Diffusion XL Demo")

    prompt, negative_prompt = repeat_prompt(args)

    # Recommend image size as one of those used in training (see Appendix I in https://arxiv.org/pdf/2307.01952.pdf).
    image_height = args.height
    image_width = args.width

    # Register TensorRT plugins
    engine_type = get_engine_type(args.engine)
    if engine_type == EngineType.TRT:
        from trt_utilities import init_trt_plugins

        init_trt_plugins()

    max_batch_size = 16
    if (engine_type in [EngineType.ORT_TRT, EngineType.TRT]) and (
        args.build_dynamic_shape or image_height > 512 or image_width > 512
    ):
        max_batch_size = 4

    batch_size = len(prompt)
    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {batch_size} is larger than allowed {max_batch_size}.")

    # No VAE decoder in base when it outputs latent instead of image.
    base_info = PipelineInfo(args.version, use_vae=False)
    base = init_pipeline(Txt2ImgXLPipeline, base_info, engine_type, args, max_batch_size, batch_size)

    refiner_info = PipelineInfo(args.version, is_refiner=True)
    refiner = init_pipeline(Img2ImgXLPipeline, refiner_info, engine_type, args, max_batch_size, batch_size)

    if engine_type == EngineType.TRT:
        max_device_memory = max(base.backend.max_device_memory(), refiner.backend.max_device_memory())
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        base.backend.activate_engines(shared_device_memory)
        refiner.backend.activate_engines(shared_device_memory)

    base.load_resources(image_height, image_width, batch_size)
    refiner.load_resources(image_height, image_width, batch_size)

    def run_base_and_refiner(warmup=False):
        images, time_base = base.run(
            prompt,
            negative_prompt,
            image_height,
            image_width,
            warmup=warmup,
            denoising_steps=args.denoising_steps,
            guidance=args.guidance,
            seed=args.seed,
            return_type="latent",
        )

        images, time_refiner = refiner.run(
            prompt,
            negative_prompt,
            images,
            image_height,
            image_width,
            warmup=warmup,
            denoising_steps=args.denoising_steps,
            guidance=args.guidance,
            seed=args.seed,
        )

        return images, time_base + time_refiner

    if not args.disable_cuda_graph:
        # inference once to get cuda graph
        _, _ = run_base_and_refiner(warmup=True)

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        _, _ = run_base_and_refiner(warmup=True)

    print("[I] Running StableDiffusion XL pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    _, latency = run_base_and_refiner(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    base.teardown()

    print("|------------|--------------|")
    print("| {:^10} | {:>9.2f} ms |".format("e2e", latency))
    print("|------------|--------------|")
    refiner.teardown()


if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")
    run_demo()
