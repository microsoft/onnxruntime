import time
from statistics import mean

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, EulerAncestralDiscreteScheduler


def load_pipeline(compile=True):
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/sdxl-turbo", subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")

    if compile:
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def test(pipeline, batch_size=1, steps=4, warmup_runs=5, test_runs=10, seed=123):
    warmup_prompt = "warm up"
    for _ in range(warmup_runs):
        image = pipeline(
            prompt=[warmup_prompt] * batch_size,
            negative_prompt=[""] * batch_size,
            num_inference_steps=4,
            guidance_scale=0.0,
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    prompt = "little cute gremlin wearing a jacket, cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed"

    latency_list = []
    image = None
    for _ in range(test_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        image = pipeline(
            prompt=[prompt] * batch_size,
            negative_prompt=[""] * batch_size,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        torch.cuda.synchronize()
        seconds = time.time() - start_time
        latency_list.append(seconds)

    print(latency_list)
    print(f"batch_size={batch_size}, steps={steps}, average_latency_in_ms={mean(latency_list) * 1000:.1f}")

    if image:
        image.save(f"torch_xl_turbo_{batch_size}_{steps}.png")


def main():
    pipeline = load_pipeline()
    for batch_size in (1, 4, 8):
        for steps in (1, 4):
            test(pipeline, batch_size, steps)


main()
