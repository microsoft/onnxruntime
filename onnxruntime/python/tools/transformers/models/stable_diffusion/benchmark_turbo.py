import gc
import time
from statistics import mean

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, EulerAncestralDiscreteScheduler

"""
Benchmark script for SDXL-Turbo.

Setup of Linux before running Stable Fast:
    git clone https://github.com/chengzeyi/stable-fast.git
    cd stable-fast
    git submodule update --init
    pip3 install torch torchvision torchaudio ninja
    pip3 install -e '.[dev,xformers,triton,transformers,diffusers]' -v
    sudo apt install libgoogle-perftools-dev
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so
"""


def compile_stable_fast(pipeline):
    from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig, compile

    config = CompilationConfig.Default()
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")

    try:
        import triton

        config.enable_triton = True
    except ImportError:
        print("triton not installed, skip")

    config.enable_cuda_graph = True

    pipeline = compile(pipeline, config)
    return pipeline


def compile_torch(pipeline):
    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    return pipeline


def load_pipeline(name, engine):
    gc.collect()
    torch.cuda.empty_cache()
    before_memory = torch.cuda.memory_allocated()

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(name, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
    pipeline = DiffusionPipeline.from_pretrained(
        name,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipeline.safety_checker = None

    gc.collect()
    after_memory = torch.cuda.memory_allocated()
    print(f"Loaded model with {after_memory - before_memory} bytes allocated")

    if engine == "stable_fast":
        pipeline = compile_stable_fast(pipeline)
    elif engine == "torch":
        pipeline = compile_torch(pipeline)

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def test(pipeline, batch_size=1, steps=4, warmup_runs=3, test_runs=10, seed=123, verbose=False, display=False):
    warmup_prompt = "warm up"
    for _ in range(warmup_runs):
        image = pipeline(
            prompt=warmup_prompt,
            num_inference_steps=steps,
            num_images_per_prompt=batch_size,
            guidance_scale=0.0,
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
        ).images[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start_time
        latency_list.append(seconds)

    if verbose:
        print(latency_list)

    print(f"batch_size={batch_size}, steps={steps}, average_latency_in_ms={mean(latency_list) * 1000:.1f}")

    if image:
        image.save(f"stable_fast_xl_turbo_{batch_size}_{steps}.png")

    if display:
        from sfast.utils.term_image import print_image

        print_image(image, max_width=120)


def arguments():
    import argparse

    parser = argparse.ArgumentParser(description="benchmark stable fast")
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
        help="stable diffusion model name like stabilityai/sdxl-turbo",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="unet steps",
    )

    parser.add_argument("--display", action="store_true", help="Display image in terminal.")

    args = parser.parse_args()
    return args


def main():
    args = arguments()

    with torch.no_grad():
        pipeline = load_pipeline(args.name, args.engine)

        if args.engine == "stable_fast":
            from sfast.utils.compute_precision import low_compute_precision

            with low_compute_precision():
                test(pipeline, args.batch_size, args.steps, display=args.display)
        else:
            test(pipeline, args.batch_size, args.steps, display=args.display)


main()
