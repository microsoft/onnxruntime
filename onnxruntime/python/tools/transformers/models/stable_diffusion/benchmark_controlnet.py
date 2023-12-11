# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import importlib.util
import time
from statistics import mean

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
)

"""
Benchmark script for SDXL-Turbo with control net for engines like PyTorch or Stable Fast.

Setup for Stable Fast (see https://github.com/chengzeyi/stable-fast/blob/main/README.md for more info):
    git clone https://github.com/chengzeyi/stable-fast.git
    cd stable-fast
    git submodule update --init
    pip3 install torch torchvision torchaudio ninja
    pip3 install -e '.[dev,xformers,triton,transformers,diffusers]' -v
    sudo apt install libgoogle-perftools-dev
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so
"""


def get_canny_image():
    import cv2
    import numpy as np
    from PIL import Image

    # Test Image can be downloaded from https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png
    image = Image.open("input_image_vermeer.png").convert("RGB")

    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def compile_stable_fast(pipeline, enable_cuda_graph=True):
    from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig, compile

    config = CompilationConfig.Default()

    if importlib.util.find_spec("xformers") is not None:
        config.enable_xformers = True

    if importlib.util.find_spec("triton") is not None:
        config.enable_triton = True

    config.enable_cuda_graph = enable_cuda_graph

    pipeline = compile(pipeline, config)
    return pipeline


def compile_torch(pipeline, use_nhwc=False):
    if use_nhwc:
        pipeline.unet.to(memory_format=torch.channels_last)

    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

    if hasattr(pipeline, "controlnet"):
        if use_nhwc:
            pipeline.controlnet.to(memory_format=torch.channels_last)
        pipeline.controlnet = torch.compile(pipeline.controlnet, mode="reduce-overhead", fullgraph=True)
    return pipeline


def load_pipeline(name, engine, use_control_net=False, use_nhwc=False, enable_cuda_graph=True):
    gc.collect()
    torch.cuda.empty_cache()
    before_memory = torch.cuda.memory_allocated()

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")

    if use_control_net:
        assert "xl" in name
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            name,
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        pipeline = DiffusionPipeline.from_pretrained(
            name,
            vae=vae,
            scheduler=scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
    pipeline.safety_checker = None

    gc.collect()
    after_memory = torch.cuda.memory_allocated()
    print(f"Loaded model with {after_memory - before_memory} bytes allocated")

    if engine == "stable_fast":
        pipeline = compile_stable_fast(pipeline, enable_cuda_graph=enable_cuda_graph)
    elif engine == "torch":
        pipeline = compile_torch(pipeline, use_nhwc=use_nhwc)

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def test(pipeline, batch_size=1, steps=4, control_image=None, warmup_runs=3, test_runs=10, seed=123, verbose=False):
    control_net_args = {}
    if hasattr(pipeline, "controlnet"):
        control_net_args = {
            "image": control_image,
            "controlnet_conditioning_scale": 0.5,
        }

    warmup_prompt = "warm up"
    for _ in range(warmup_runs):
        image = pipeline(
            prompt=warmup_prompt,
            num_inference_steps=steps,
            num_images_per_prompt=batch_size,
            guidance_scale=0.0,
            **control_net_args,
        ).images
        assert len(image) == batch_size

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    prompt = "little cute gremlin wearing a jacket, cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed"

    latency_list = []
    image = None
    for _ in range(test_runs):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        image = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            num_images_per_prompt=batch_size,
            guidance_scale=0.0,
            generator=generator,
            **control_net_args,
        ).images[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start_time
        latency_list.append(seconds)

    if verbose:
        print(latency_list)

    return image, latency_list


def arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Stable Diffusion pipeline (optional control net for SDXL)")
    parser.add_argument(
        "--engine",
        type=str,
        default="torch",
        choices=["torch", "stable_fast"],
        help="Backend engine: torch or stable_fast",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="stabilityai/sdxl-turbo",
        help="Stable diffusion model name. Default is stabilityai/sdxl-turbo",
    )

    parser.add_argument(
        "--use_control_net",
        action="store_true",
        help="Use control net diffusers/controlnet-canny-sdxl-1.0",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Denoising steps",
    )

    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=3,
        help="Number of warmup runs before measurement",
    )

    parser.add_argument(
        "--use_nhwc",
        action="store_true",
        help="use channel last format for torch compile",
    )

    parser.add_argument(
        "--enable_cuda_graph",
        action="store_true",
        help="enable cuda graph for stable fast",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print more information",
    )

    args = parser.parse_args()
    return args


def main():
    args = arguments()

    with torch.no_grad():
        pipeline = load_pipeline(
            args.name,
            args.engine,
            use_control_net=args.use_control_net,
            use_nhwc=args.use_nhwc,
            enable_cuda_graph=args.enable_cuda_graph,
        )

        canny_image = get_canny_image()

        if args.engine == "stable_fast":
            from sfast.utils.compute_precision import low_compute_precision

            with low_compute_precision():
                image, latency_list = test(
                    pipeline,
                    args.batch_size,
                    args.steps,
                    control_image=canny_image,
                    warmup_runs=args.warmup_runs,
                    verbose=args.verbose,
                )
        else:
            image, latency_list = test(
                pipeline,
                args.batch_size,
                args.steps,
                control_image=canny_image,
                warmup_runs=args.warmup_runs,
                verbose=args.verbose,
            )

        # Save the first output image to inspect the result.
        if image:
            image.save(
                f"{args.engine}_{args.name.replace('/', '_')}_{args.batch_size}_{args.steps}_c{int(args.use_control_net)}.png"
            )

        result = {
            "engine": args.engine,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "control_net": args.use_control_net,
            "nhwc": args.use_nhwc,
            "enable_cuda_graph": args.enable_cuda_graph,
            "average_latency_in_ms": mean(latency_list) * 1000,
        }
        print(result)


main()
