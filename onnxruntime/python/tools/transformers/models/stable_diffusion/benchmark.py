# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import csv
import os
import statistics
import sys
import time

import coloredlogs

# import torch before onnxruntime so that onnxruntime uses the cuDNN in the torch package.
import torch

SD_MODELS = {
    "1.5": "runwayml/stable-diffusion-v1-5",
    "2.0": "stabilityai/stable-diffusion-2",
    "2.1": "stabilityai/stable-diffusion-2-1",
}

PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


def example_prompts():
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "cute grey cat with blue eyes, wearing a bowtie, acrylic painting",
        "a cute magical flying dog, fantasy art drawn by disney concept artists, highly detailed, digital painting",
        "an illustration of a house with large barn with many cute flower pots and beautiful blue sky scenery",
        "one apple sitting on a table, still life, reflective, full color photograph, centered, close-up product",
        "background texture of stones, masterpiece, artistic, stunning photo, award winner photo",
        "new international organic style house, tropical surroundings, architecture, 8k, hdr",
        "beautiful Renaissance Revival Estate, Hobbit-House, detailed painting, warm colors, 8k, trending on Artstation",
        "blue owl, big green eyes, portrait, intricate metal design, unreal engine, octane render, realistic",
        "delicate elvish moonstone necklace on a velvet background, symmetrical intricate motifs, leaves, flowers, 8k",
    ]

    return prompts


class CudaMemoryMonitor:
    def __init__(self, keep_measuring=True):
        self.keep_measuring = keep_measuring

    def measure_gpu_usage(self):
        from py3nvml.py3nvml import (
            NVMLError,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
        )

        max_gpu_usage = []
        gpu_name = []
        try:
            nvmlInit()
            device_count = nvmlDeviceGetCount()
            if not isinstance(device_count, int):
                print(f"nvmlDeviceGetCount result is not integer: {device_count}")
                return None

            max_gpu_usage = [0 for i in range(device_count)]
            gpu_name = [nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(i)) for i in range(device_count)]
            while True:
                for i in range(device_count):
                    info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
                    if isinstance(info, str):
                        print(f"nvmlDeviceGetMemoryInfo returns str: {info}")
                        return None
                    max_gpu_usage[i] = max(max_gpu_usage[i], info.used / 1024**2)
                time.sleep(0.002)  # 2ms
                if not self.keep_measuring:
                    break
            nvmlShutdown()
            return [
                {
                    "device_id": i,
                    "name": gpu_name[i],
                    "max_used_MB": max_gpu_usage[i],
                }
                for i in range(device_count)
            ]
        except NVMLError as error:
            print("Error fetching GPU information using nvml: %s", error)
            return None


class RocmMemoryMonitor:
    def __init__(self, keep_measuring=True):
        self.keep_measuring = keep_measuring
        rocm_smi_path = "/opt/rocm/libexec/rocm_smi"
        if os.path.exists(rocm_smi_path):
            if rocm_smi_path not in sys.path:
                sys.path.append(rocm_smi_path)
        try:
            import rocm_smi

            self.rocm_smi = rocm_smi
            self.rocm_smi.initializeRsmi()
        except ImportError:
            self.rocm_smi = None

    def get_used_memory(self, dev):
        if self.rocm_smi is None:
            return -1
        return self.rocm_smi.getMemInfo(dev, "VRAM")[0] / 1024 / 1024

    def measure_gpu_usage(self):
        device_count = len(self.rocm_smi.listDevices()) if self.rocm_smi is not None else 0
        max_gpu_usage = [0 for i in range(device_count)]
        gpu_name = [f"GPU{i}" for i in range(device_count)]
        while True:
            for i in range(device_count):
                max_gpu_usage[i] = max(max_gpu_usage[i], self.get_used_memory(i))
            time.sleep(0.002)  # 2ms
            if not self.keep_measuring:
                break
        return [
            {
                "device_id": i,
                "name": gpu_name[i],
                "max_used_MB": max_gpu_usage[i],
            }
            for i in range(device_count)
        ]


def measure_gpu_memory(monitor_type, func, start_memory=None):
    if monitor_type is None:
        return None

    monitor = monitor_type(False)
    memory_before_test = monitor.measure_gpu_usage()

    if start_memory is None:
        start_memory = memory_before_test
    if start_memory is None:
        return None
    if func is None:
        return start_memory

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        monitor = monitor_type()
        mem_thread = executor.submit(monitor.measure_gpu_usage)
        try:
            fn_thread = executor.submit(func)
            _ = fn_thread.result()
        finally:
            monitor.keep_measuring = False
            max_usage = mem_thread.result()

        if max_usage is None:
            return None

        print(f"GPU memory usage: before={memory_before_test}  peak={max_usage}")
        if len(start_memory) >= 1 and len(max_usage) >= 1 and len(start_memory) == len(max_usage):
            # When there are multiple GPUs, we will check the one with maximum usage.
            max_used = 0
            for i, memory_before in enumerate(start_memory):
                before = memory_before["max_used_MB"]
                after = max_usage[i]["max_used_MB"]
                used = after - before
                max_used = max(max_used, used)
            return max_used
    return None


def get_ort_pipeline(model_name: str, directory: str, provider, disable_safety_checker: bool):
    from diffusers import DDIMScheduler, OnnxStableDiffusionPipeline

    import onnxruntime

    if directory is not None:
        assert os.path.exists(directory)
        session_options = onnxruntime.SessionOptions()
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            directory,
            provider=provider,
            sess_options=session_options,
        )
    else:
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            model_name,
            revision="onnx",
            provider=provider,
            use_auth_token=True,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if disable_safety_checker:
        pipe.safety_checker = None
        pipe.feature_extractor = None

    return pipe


def get_torch_pipeline(model_name: str, disable_safety_checker: bool, enable_torch_compile: bool, use_xformers: bool):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from torch import channels_last, float16

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=float16).to("cuda")

    pipe.unet.to(memory_format=channels_last)  # in-place operation

    if use_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    if enable_torch_compile:
        pipe.unet = torch.compile(pipe.unet)
        pipe.vae = torch.compile(pipe.vae)
        pipe.text_encoder = torch.compile(pipe.text_encoder)
        print("Torch compiled unet, vae and text_encoder")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    if disable_safety_checker:
        pipe.safety_checker = None
        pipe.feature_extractor = None

    return pipe


def get_image_filename_prefix(engine: str, model_name: str, batch_size: int, disable_safety_checker: bool):
    short_model_name = model_name.split("/")[-1].replace("stable-diffusion-", "sd")
    return f"{engine}_{short_model_name}_b{batch_size}" + ("" if disable_safety_checker else "_safe")


def run_ort_pipeline(
    pipe,
    batch_size: int,
    image_filename_prefix: str,
    height,
    width,
    steps,
    num_prompts,
    batch_count,
    start_memory,
    memory_monitor_type,
):
    from diffusers import OnnxStableDiffusionPipeline

    assert isinstance(pipe, OnnxStableDiffusionPipeline)

    prompts = example_prompts()

    def warmup():
        pipe("warm up", height, width, num_inference_steps=steps, num_images_per_prompt=batch_size)

    # Run warm up, and measure GPU memory of two runs
    # cuDNN/MIOpen The first run has  algo search so it might need more memory)
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    latency_list = []
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        for j in range(batch_count):
            inference_start = time.time()
            images = pipe(
                prompt,
                height,
                width,
                num_inference_steps=steps,
                negative_prompt=None,
                guidance_scale=7.5,
                num_images_per_prompt=batch_size,
            ).images
            inference_end = time.time()
            latency = inference_end - inference_start
            latency_list.append(latency)
            print(f"Inference took {latency:.3f} seconds")
            for k, image in enumerate(images):
                image.save(f"{image_filename_prefix}_{i}_{j}_{k}.jpg")

    from onnxruntime import __version__ as ort_version

    return {
        "engine": "onnxruntime",
        "version": ort_version,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
    }


def run_torch_pipeline(
    pipe,
    batch_size: int,
    image_filename_prefix: str,
    height,
    width,
    steps,
    num_prompts,
    batch_count,
    start_memory,
    memory_monitor_type,
):
    prompts = example_prompts()

    # total 2 runs of warm up, and measure GPU memory for CUDA EP
    def warmup():
        pipe("warm up", height, width, num_inference_steps=steps, num_images_per_prompt=batch_size)

    # Run warm up, and measure GPU memory of two runs (The first run has cuDNN algo search so it might need more memory)
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    torch.set_grad_enabled(False)

    latency_list = []
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        torch.cuda.synchronize()
        for j in range(batch_count):
            inference_start = time.time()
            images = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=7.5,
                negative_prompt=None,
                num_images_per_prompt=batch_size,
                generator=None,  # torch.Generator
            ).images

            torch.cuda.synchronize()
            inference_end = time.time()
            latency = inference_end - inference_start
            latency_list.append(latency)
            print(f"Inference took {latency:.3f} seconds")
            for k, image in enumerate(images):
                image.save(f"{image_filename_prefix}_{i}_{j}_{k}.jpg")

    return {
        "engine": "torch",
        "version": torch.__version__,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
    }


def run_ort(
    model_name: str,
    directory: str,
    provider: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    tuning: bool,
):
    provider_and_options = provider
    if tuning and provider in ["CUDAExecutionProvider", "ROCMExecutionProvider"]:
        provider_and_options = (provider, {"tunable_op_enable": 1, "tunable_op_tuning_enable": 1})

    load_start = time.time()
    pipe = get_ort_pipeline(model_name, directory, provider_and_options, disable_safety_checker)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    image_filename_prefix = get_image_filename_prefix("ort", model_name, batch_size, disable_safety_checker)
    result = run_ort_pipeline(
        pipe,
        batch_size,
        image_filename_prefix,
        height,
        width,
        steps,
        num_prompts,
        batch_count,
        start_memory,
        memory_monitor_type,
    )

    result.update(
        {
            "model_name": model_name,
            "directory": directory,
            "provider": provider.replace("ExecutionProvider", ""),
            "disable_safety_checker": disable_safety_checker,
            "enable_cuda_graph": False,
        }
    )
    return result


def export_and_run_ort(
    model_name: str,
    provider: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    enable_cuda_graph: bool,
):
    assert provider == "CUDAExecutionProvider"

    from diffusers import DDIMScheduler
    from onnxruntime_cuda_txt2img import OnnxruntimeCudaStableDiffusionPipeline

    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    pipe = OnnxruntimeCudaStableDiffusionPipeline.from_pretrained(
        model_name,
        scheduler=scheduler,
        requires_safety_checker=not disable_safety_checker,
        enable_cuda_graph=enable_cuda_graph,
    )

    # re-use cached folder to save ONNX models
    pipe.set_cached_folder(model_name)

    pipe = pipe.to("cuda", torch_dtype=torch.float16)

    def warmup():
        pipe(["warm up"] * batch_size, image_height=height, image_width=width, num_inference_steps=steps)

    # Run warm up, and measure GPU memory of two runs
    # The first run has algo search so it might need more memory
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    # An extra warm up run is needed for cuda graph
    warmup()

    image_filename_prefix = get_image_filename_prefix("ort_cuda", model_name, batch_size, disable_safety_checker)

    latency_list = []
    prompts = example_prompts()
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        for j in range(batch_count):
            inference_start = time.time()
            images = pipe(
                [prompt] * batch_size,
                num_inference_steps=steps,
            ).images
            inference_end = time.time()
            latency = inference_end - inference_start
            latency_list.append(latency)
            print(f"Inference took {latency:.3f} seconds")
            for k, image in enumerate(images):
                image.save(f"{image_filename_prefix}_{i}_{j}_{k}.jpg")

    from onnxruntime import __version__ as ort_version

    return {
        "model_name": model_name,
        "engine": "onnxruntime",
        "version": ort_version,
        "provider": provider.replace("ExecutionProvider", ""),
        "directory": pipe.engine_dir,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
        "disable_safety_checker": disable_safety_checker,
        "enable_cuda_graph": enable_cuda_graph,
    }


def run_ort_trt(
    model_name: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    max_batch_size: int,
    enable_cuda_graph: bool,
):
    from diffusers import DDIMScheduler
    from onnxruntime_tensorrt_txt2img import OnnxruntimeTensorRTStableDiffusionPipeline

    assert batch_size <= max_batch_size

    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipe = OnnxruntimeTensorRTStableDiffusionPipeline.from_pretrained(
        model_name,
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler,
        requires_safety_checker=not disable_safety_checker,
        image_height=height,
        image_width=width,
        max_batch_size=max_batch_size,
        onnx_opset=17,
        enable_cuda_graph=enable_cuda_graph,
    )

    # re-use cached folder to save ONNX models and TensorRT Engines
    pipe.set_cached_folder(model_name, revision="fp16")

    pipe = pipe.to("cuda")

    def warmup():
        pipe(["warm up"] * batch_size, num_inference_steps=steps)

    # Run warm up, and measure GPU memory of two runs
    # The first run has algo search so it might need more memory
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    image_filename_prefix = get_image_filename_prefix("ort_trt", model_name, batch_size, disable_safety_checker)

    latency_list = []
    prompts = example_prompts()
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        for j in range(batch_count):
            inference_start = time.time()
            images = pipe(
                [prompt] * batch_size,
                num_inference_steps=steps,
            ).images
            inference_end = time.time()
            latency = inference_end - inference_start
            latency_list.append(latency)
            print(f"Inference took {latency:.3f} seconds")
            for k, image in enumerate(images):
                image.save(f"{image_filename_prefix}_{i}_{j}_{k}.jpg")

    from tensorrt import __version__ as trt_version

    from onnxruntime import __version__ as ort_version

    return {
        "model_name": model_name,
        "engine": "onnxruntime",
        "version": ort_version,
        "provider": f"tensorrt{trt_version})",
        "directory": pipe.engine_dir,
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
        "disable_safety_checker": disable_safety_checker,
        "enable_cuda_graph": enable_cuda_graph,
    }


def run_tensorrt(
    model_name: str,
    batch_size: int,
    disable_safety_checker: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
    max_batch_size: int,
):
    from diffusers import DDIMScheduler
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

    assert batch_size <= max_batch_size

    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="stable_diffusion_tensorrt_txt2img",
        revision="fp16",
        torch_dtype=torch.float16,
        scheduler=scheduler,
        requires_safety_checker=not disable_safety_checker,
        image_height=height,
        image_width=width,
        max_batch_size=max_batch_size,
    )

    # re-use cached folder to save ONNX models and TensorRT Engines
    pipe.set_cached_folder(model_name, revision="fp16")

    pipe = pipe.to("cuda")

    def warmup():
        pipe(["warm up"] * batch_size, num_inference_steps=steps)

    # Run warm up, and measure GPU memory of two runs
    # The first run has algo search so it might need more memory
    first_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)
    second_run_memory = measure_gpu_memory(memory_monitor_type, warmup, start_memory)

    warmup()

    image_filename_prefix = get_image_filename_prefix("trt", model_name, batch_size, disable_safety_checker)

    latency_list = []
    prompts = example_prompts()
    for i, prompt in enumerate(prompts):
        if i >= num_prompts:
            break
        for j in range(batch_count):
            inference_start = time.time()
            images = pipe(
                [prompt] * batch_size,
                num_inference_steps=steps,
            ).images
            inference_end = time.time()
            latency = inference_end - inference_start
            latency_list.append(latency)
            print(f"Inference took {latency:.3f} seconds")
            for k, image in enumerate(images):
                image.save(f"{image_filename_prefix}_{i}_{j}_{k}.jpg")

    from tensorrt import __version__ as trt_version

    return {
        "engine": "tensorrt",
        "version": trt_version,
        "provider": "default",
        "height": height,
        "width": width,
        "steps": steps,
        "batch_size": batch_size,
        "batch_count": batch_count,
        "num_prompts": num_prompts,
        "average_latency": sum(latency_list) / len(latency_list),
        "median_latency": statistics.median(latency_list),
        "first_run_memory_MB": first_run_memory,
        "second_run_memory_MB": second_run_memory,
        "enable_cuda_graph": False,
    }


def run_torch(
    model_name: str,
    batch_size: int,
    disable_safety_checker: bool,
    enable_torch_compile: bool,
    use_xformers: bool,
    height: int,
    width: int,
    steps: int,
    num_prompts: int,
    batch_count: int,
    start_memory,
    memory_monitor_type,
):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.set_grad_enabled(False)

    load_start = time.time()
    pipe = get_torch_pipeline(model_name, disable_safety_checker, enable_torch_compile, use_xformers)
    load_end = time.time()
    print(f"Model loading took {load_end - load_start} seconds")

    image_filename_prefix = get_image_filename_prefix("torch", model_name, batch_size, disable_safety_checker)

    if not enable_torch_compile:
        with torch.inference_mode():
            result = run_torch_pipeline(
                pipe,
                batch_size,
                image_filename_prefix,
                height,
                width,
                steps,
                num_prompts,
                batch_count,
                start_memory,
                memory_monitor_type,
            )
    else:
        result = run_torch_pipeline(
            pipe,
            batch_size,
            image_filename_prefix,
            height,
            width,
            steps,
            num_prompts,
            batch_count,
            start_memory,
            memory_monitor_type,
        )

    result.update(
        {
            "model_name": model_name,
            "directory": None,
            "provider": "compile" if enable_torch_compile else "xformers" if use_xformers else "default",
            "disable_safety_checker": disable_safety_checker,
            "enable_cuda_graph": False,
        }
    )
    return result


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--engine",
        required=False,
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "torch", "tensorrt"],
        help="Engines to benchmark. Default is onnxruntime.",
    )

    parser.add_argument(
        "-r",
        "--provider",
        required=False,
        type=str,
        default="cuda",
        choices=list(PROVIDERS.keys()),
        help="Provider to benchmark. Default is CUDAExecutionProvider.",
    )

    parser.add_argument(
        "-t",
        "--tuning",
        action="store_true",
        help="Enable TunableOp and tuning. "
        "This will incur longer warmup latency, and is mandatory for some operators of ROCm EP.",
    )

    parser.add_argument(
        "-v",
        "--version",
        required=False,
        type=str,
        choices=list(SD_MODELS.keys()),
        default="1.5",
        help="Stable diffusion version like 1.5, 2.0 or 2.1. Default is 1.5.",
    )

    parser.add_argument(
        "-p",
        "--pipeline",
        required=False,
        type=str,
        default=None,
        help="Directory of saved onnx pipeline. It could be the output directory of optimize_pipeline.py.",
    )

    parser.add_argument(
        "--enable_safety_checker",
        required=False,
        action="store_true",
        help="Enable safety checker",
    )
    parser.set_defaults(enable_safety_checker=False)

    parser.add_argument(
        "--enable_torch_compile",
        required=False,
        action="store_true",
        help="Enable compile unet for PyTorch 2.0",
    )
    parser.set_defaults(enable_torch_compile=False)

    parser.add_argument(
        "--use_xformers",
        required=False,
        action="store_true",
        help="Use xformers for PyTorch",
    )
    parser.set_defaults(use_xformers=False)

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 8, 10, 16, 32],
        help="Number of images per batch. Default is 1.",
    )

    parser.add_argument(
        "--height",
        required=False,
        type=int,
        default=512,
        help="Output image height. Default is 512.",
    )

    parser.add_argument(
        "--width",
        required=False,
        type=int,
        default=512,
        help="Output image width. Default is 512.",
    )

    parser.add_argument(
        "-s",
        "--steps",
        required=False,
        type=int,
        default=50,
        help="Number of steps. Default is 50.",
    )

    parser.add_argument(
        "-n",
        "--num_prompts",
        required=False,
        type=int,
        default=1,
        help="Number of prompts. Default is 1.",
    )

    parser.add_argument(
        "-c",
        "--batch_count",
        required=False,
        type=int,
        choices=range(1, 11),
        default=5,
        help="Number of batches to test. Default is 5.",
    )

    parser.add_argument(
        "-m",
        "--max_trt_batch_size",
        required=False,
        type=int,
        choices=range(1, 16),
        default=4,
        help="Maximum batch size for TensorRT. Change the value may trigger TensorRT engine rebuild. Default is 4.",
    )

    parser.add_argument(
        "-g",
        "--enable_cuda_graph",
        required=False,
        action="store_true",
        help="Enable Cuda Graph. Requires onnxruntime >= 1.16",
    )
    parser.set_defaults(enable_cuda_graph=False)

    args = parser.parse_args()

    return args


def print_loaded_libraries(cuda_related_only=True):
    import psutil

    p = psutil.Process(os.getpid())
    for lib in p.memory_maps():
        if (not cuda_related_only) or any(x in lib.path for x in ("libcu", "libnv", "tensorrt")):
            print(lib.path)


def main():
    args = parse_arguments()
    print(args)

    if args.enable_cuda_graph:
        if not (args.engine == "onnxruntime" and args.provider in ["cuda", "tensorrt"] and args.pipeline is None):
            raise ValueError("The stable diffusion pipeline does not support CUDA graph.")

        from packaging import version

        from onnxruntime import __version__ as ort_version

        if version.parse(ort_version) < version.parse("1.16"):
            raise ValueError(
                "CUDA graph requires ONNX Runtime 1.16. You can install nightly like the following:\n"
                " pip uninstall onnxruntime-gpu\n"
                " pip install ort-nightly-gpu -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/"
            )

    coloredlogs.install(fmt="%(funcName)20s: %(message)s")

    memory_monitor_type = None
    if args.provider in ["cuda", "tensorrt"]:
        memory_monitor_type = CudaMemoryMonitor
    elif args.provider == "rocm":
        memory_monitor_type = RocmMemoryMonitor

    start_memory = measure_gpu_memory(memory_monitor_type, None)
    print("GPU memory used before loading models:", start_memory)

    sd_model = SD_MODELS[args.version]
    provider = PROVIDERS[args.provider]
    if args.engine == "onnxruntime" and args.provider == "tensorrt":
        result = run_ort_trt(
            sd_model,
            args.batch_size,
            not args.enable_safety_checker,
            args.height,
            args.width,
            args.steps,
            args.num_prompts,
            args.batch_count,
            start_memory,
            memory_monitor_type,
            args.max_trt_batch_size,
            args.enable_cuda_graph,
        )
    elif args.engine == "onnxruntime" and provider == "CUDAExecutionProvider" and args.pipeline is None:
        print("Pipeline is not specified. Trying export and optimize onnx models...")
        result = export_and_run_ort(
            sd_model,
            provider,
            args.batch_size,
            not args.enable_safety_checker,
            args.height,
            args.width,
            args.steps,
            args.num_prompts,
            args.batch_count,
            start_memory,
            memory_monitor_type,
            args.enable_cuda_graph,
        )
    elif args.engine == "onnxruntime":
        assert args.pipeline and os.path.isdir(
            args.pipeline
        ), "--pipeline should be specified for the directory of ONNX models"

        if args.version in ["2.1"]:
            # Set a flag to avoid overflow in attention, which causes black image output in SD 2.1 model
            # This shall be done before the first inference run.
            os.environ["ORT_DISABLE_TRT_FLASH_ATTENTION"] = "1"

        result = run_ort(
            sd_model,
            args.pipeline,
            provider,
            args.batch_size,
            not args.enable_safety_checker,
            args.height,
            args.width,
            args.steps,
            args.num_prompts,
            args.batch_count,
            start_memory,
            memory_monitor_type,
            args.tuning,
        )
    elif args.engine == "tensorrt":
        result = run_tensorrt(
            sd_model,
            args.batch_size,
            not args.enable_safety_checker,
            args.height,
            args.width,
            args.steps,
            args.num_prompts,
            args.batch_count,
            start_memory,
            memory_monitor_type,
            args.max_trt_batch_size,
        )
    else:
        result = run_torch(
            sd_model,
            args.batch_size,
            not args.enable_safety_checker,
            args.enable_torch_compile,
            args.use_xformers,
            args.height,
            args.width,
            args.steps,
            args.num_prompts,
            args.batch_count,
            start_memory,
            memory_monitor_type,
        )

    print(result)

    with open("benchmark_result.csv", mode="a", newline="") as csv_file:
        column_names = [
            "model_name",
            "directory",
            "engine",
            "version",
            "provider",
            "disable_safety_checker",
            "height",
            "width",
            "steps",
            "batch_size",
            "batch_count",
            "num_prompts",
            "average_latency",
            "median_latency",
            "first_run_memory_MB",
            "second_run_memory_MB",
            "enable_cuda_graph",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()
        csv_writer.writerow(result)

    # Show loaded DLLs when steps == 1 for debugging purpose.
    if args.steps == 1:
        print_loaded_libraries(args.provider in ["cuda", "tensorrt"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = sys.exc_info()
        print(e.with_traceback(tb[2]))
