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


def load_pipelines(args, batch_size):
    # Register TensorRT plugins
    engine_type = get_engine_type(args.engine)
    if engine_type == EngineType.TRT:
        from trt_utilities import init_trt_plugins

        init_trt_plugins()

    max_batch_size = 16
    if (engine_type in [EngineType.ORT_TRT, EngineType.TRT]) and (
        args.build_dynamic_shape or args.height > 512 or args.width > 512
    ):
        max_batch_size = 4

    if batch_size > max_batch_size:
        raise ValueError(f"Batch size {batch_size} is larger than allowed {max_batch_size}.")

    # For TensorRT,  performance of engine built with dynamic shape is very sensitive to the range of image size.
    # Here, we reduce the range of image size for TensorRT to trade-off flexibility and performance.
    min_image_size = 832 if args.engine != "ORT_CUDA" else 512
    max_image_size = 1216 if args.engine != "ORT_CUDA" else 2048

    # No VAE decoder in base when it outputs latent instead of image.
    base_info = PipelineInfo(args.version, use_vae=False, min_image_size=min_image_size, max_image_size=max_image_size)
    base = init_pipeline(Txt2ImgXLPipeline, base_info, engine_type, args, max_batch_size, batch_size)

    refiner_info = PipelineInfo(
        args.version, is_refiner=True, min_image_size=min_image_size, max_image_size=max_image_size
    )
    refiner = init_pipeline(Img2ImgXLPipeline, refiner_info, engine_type, args, max_batch_size, batch_size)

    if engine_type == EngineType.TRT:
        max_device_memory = max(base.backend.max_device_memory(), refiner.backend.max_device_memory())
        _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        base.backend.activate_engines(shared_device_memory)
        refiner.backend.activate_engines(shared_device_memory)

    if engine_type == EngineType.ORT_CUDA:
        enable_vae_slicing = args.enable_vae_slicing
        if batch_size > 4 and not enable_vae_slicing:
            print("Updating enable_vae_slicing to be True to avoid cuDNN error for batch size > 4.")
            enable_vae_slicing = True
        if enable_vae_slicing:
            refiner.backend.enable_vae_slicing()
    return base, refiner


def run_pipelines(args, base, refiner, prompt, negative_prompt, is_warm_up=False):
    image_height = args.height
    image_width = args.width
    batch_size = len(prompt)
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

    if args.num_warmup_runs > 0:
        print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        _, _ = run_base_and_refiner(warmup=True)

    if is_warm_up:
        return

    print("[I] Running StableDiffusion XL pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    _, latency = run_base_and_refiner(warmup=False)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    print("|------------|--------------|")
    print("| {:^10} | {:>9.2f} ms |".format("e2e", latency))
    print("|------------|--------------|")


def run_demo(args):
    """Run Stable Diffusion XL Base + Refiner together (known as ensemble of expert denoisers) to generate an image."""

    prompt, negative_prompt = repeat_prompt(args)
    batch_size = len(prompt)
    base, refiner = load_pipelines(args, batch_size)
    run_pipelines(args, base, refiner, prompt, negative_prompt)
    base.teardown()
    refiner.teardown()


def run_dynamic_shape_demo(args):
    """Run demo of generating images with different settings with ORT CUDA provider."""
    args.engine = "ORT_CUDA"
    args.disable_cuda_graph = True
    base, refiner = load_pipelines(args, 1)

    prompts = [
        "starry night over Golden Gate Bridge by van gogh",
        "beautiful photograph of Mt. Fuji during cherry blossom",
        "little cute gremlin sitting on a bed, cinematic",
        "cute grey cat with blue eyes, wearing a bowtie, acrylic painting",
        "beautiful Renaissance Revival Estate, Hobbit-House, detailed painting, warm colors, 8k, trending on Artstation",
        "blue owl, big green eyes, portrait, intricate metal design, unreal engine, octane render, realistic",
    ]

    # batch size, height, width, scheduler, steps, prompt
    configs = [
        (1, 832, 1216, "UniPC", 8, prompts[0]),
        (1, 1024, 1024, "DDIM", 24, prompts[1]),
        (1, 1216, 832, "EulerA", 18, prompts[2]),
        (2, 1344, 768, "DDIM", 30, prompts[3]),
        (2, 640, 1536, "UniPC", 18, prompts[4]),
        (2, 1152, 896, "EulerA", 30, prompts[5]),
    ]

    # Warm up (for cudnn convolution algo search) once before serving.
    args.prompt = ["warm up"]
    args.num_warmup_runs = 1
    for batch_size, height, width, _, _, _ in configs:
        args.batch_size = batch_size
        args.height = height
        args.width = width
        print(f"\nWarm up batch_size={batch_size}, height={height}, width={width}")
        prompt, negative_prompt = repeat_prompt(args)
        run_pipelines(args, base, refiner, prompt, negative_prompt, is_warm_up=True)

    # Run pipeline on a list of prompts.
    args.num_warmup_runs = 0
    for batch_size, height, width, scheduler, steps, example_prompt in configs:
        args.prompt = [example_prompt]
        args.batch_size = batch_size
        args.height = height
        args.width = width
        args.scheduler = scheduler
        args.denoising_steps = steps
        base.set_scheduler(scheduler)
        refiner.set_scheduler(scheduler)
        print(
            f"\nbatch_size={batch_size}, height={height}, width={width}, scheduler={scheduler}, steps={steps}, prompt={example_prompt}"
        )
        prompt, negative_prompt = repeat_prompt(args)
        run_pipelines(args, base, refiner, prompt, negative_prompt, is_warm_up=False)

    base.teardown()
    refiner.teardown()


if __name__ == "__main__":
    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    args = parse_arguments(is_xl=True, description="Options for Stable Diffusion XL Demo")
    no_prompt = isinstance(args.prompt, list) and len(args.prompt) == 1 and not args.prompt[0]
    if no_prompt:
        run_dynamic_shape_demo(args)
    else:
        run_demo(args)
