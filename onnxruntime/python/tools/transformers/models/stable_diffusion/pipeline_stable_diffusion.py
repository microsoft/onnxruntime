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

import os
import pathlib
import random

import nvtx
import torch
from cuda import cudart
from diffusion_models import PipelineInfo, get_tokenizer
from diffusion_schedulers import DDIMScheduler, EulerAncestralDiscreteScheduler, LCMScheduler, UniPCMultistepScheduler
from engine_builder import EngineType
from engine_builder_ort_cuda import OrtCudaEngineBuilder
from engine_builder_ort_trt import OrtTensorrtEngineBuilder
from engine_builder_tensorrt import TensorrtEngineBuilder


class StableDiffusionPipeline:
    """
    Stable Diffusion pipeline using TensorRT.
    """

    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        scheduler="DDIM",
        device="cuda",
        output_dir=".",
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
        framework_model_dir="pytorch_model",
        engine_type: EngineType = EngineType.ORT_TRT,
    ):
        """
        Initializes the Diffusion pipeline.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of [DDIM, EulerA, UniPC, LCM].
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
            framework_model_dir (str):
                cache directory for framework checkpoints
            engine_type (EngineType)
                backend engine type like ORT_TRT or TRT
        """

        self.pipeline_info = pipeline_info
        self.version = pipeline_info.version

        self.vae_scaling_factor = pipeline_info.vae_scaling_factor()

        self.max_batch_size = max_batch_size

        self.framework_model_dir = framework_model_dir
        self.output_dir = output_dir
        for directory in [self.framework_model_dir, self.output_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        self.hf_token = hf_token
        self.device = device
        self.torch_device = torch.device(device, torch.cuda.current_device())
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        self.stages = pipeline_info.stages()

        self.use_cuda_graph = use_cuda_graph

        self.tokenizer = None
        self.tokenizer2 = None

        self.generator = torch.Generator(device="cuda")
        self.actual_steps = None

        self.current_scheduler = None
        self.set_scheduler(scheduler)

        # backend engine
        self.engine_type = engine_type
        if engine_type == EngineType.TRT:
            self.backend = TensorrtEngineBuilder(pipeline_info, max_batch_size, hf_token, device, use_cuda_graph)
        elif engine_type == EngineType.ORT_TRT:
            self.backend = OrtTensorrtEngineBuilder(pipeline_info, max_batch_size, hf_token, device, use_cuda_graph)
        elif engine_type == EngineType.ORT_CUDA:
            self.backend = OrtCudaEngineBuilder(pipeline_info, max_batch_size, hf_token, device, use_cuda_graph)
        else:
            raise RuntimeError(f"Backend engine type {engine_type.name} is not supported")

        # Load text tokenizer
        if not self.pipeline_info.is_xl_refiner():
            self.tokenizer = get_tokenizer(
                self.pipeline_info, self.framework_model_dir, self.hf_token, subfolder="tokenizer"
            )

        if self.pipeline_info.is_xl():
            self.tokenizer2 = get_tokenizer(
                self.pipeline_info, self.framework_model_dir, self.hf_token, subfolder="tokenizer_2"
            )

        # Create CUDA events
        self.events = {}
        for stage in ["clip", "denoise", "vae", "vae_encoder"]:
            for marker in ["start", "stop"]:
                self.events[stage + "-" + marker] = cudart.cudaEventCreate()[1]

    def is_backend_tensorrt(self):
        return self.engine_type == EngineType.TRT

    def set_scheduler(self, scheduler: str):
        if scheduler == self.current_scheduler:
            return

        # Scheduler options
        sched_opts = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012}
        if self.version in ("2.0", "2.1"):
            sched_opts["prediction_type"] = "v_prediction"
        else:
            sched_opts["prediction_type"] = "epsilon"

        if scheduler == "DDIM":
            self.scheduler = DDIMScheduler(device=self.device, **sched_opts)
        elif scheduler == "EulerA":
            self.scheduler = EulerAncestralDiscreteScheduler(device=self.device, **sched_opts)
        elif scheduler == "UniPC":
            self.scheduler = UniPCMultistepScheduler(device=self.device, **sched_opts)
        elif scheduler == "LCM":
            self.scheduler = LCMScheduler(device=self.device, **sched_opts)
        else:
            raise ValueError("Scheduler should be either DDIM, EulerA, UniPC or LCM")

        self.current_scheduler = scheduler
        self.denoising_steps = None

    def set_denoising_steps(self, denoising_steps: int):
        if not (self.denoising_steps == denoising_steps and isinstance(self.scheduler, DDIMScheduler)):
            self.scheduler.set_timesteps(denoising_steps)
            self.scheduler.configure()
            self.denoising_steps = denoising_steps

    def load_resources(self, image_height, image_width, batch_size):
        # If engine is built with static input shape, call this only once after engine build.
        # Otherwise, it need be called before every inference run.
        self.backend.load_resources(image_height, image_width, batch_size)

    def set_random_seed(self, seed):
        if isinstance(seed, int):
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()

    def get_current_seed(self):
        return self.generator.initial_seed()

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e)

        if self.backend:
            self.backend.teardown()

    def run_engine(self, model_name, feed_dict):
        return self.backend.run_engine(model_name, feed_dict)

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width):
        latents_dtype = torch.float32  # text_embeddings.dtype
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def initialize_timesteps(self, timesteps, strength):
        self.scheduler.set_timesteps(timesteps)
        offset = self.scheduler.steps_offset if hasattr(self.scheduler, "steps_offset") else 0
        init_timestep = int(timesteps * strength) + offset
        init_timestep = min(init_timestep, timesteps)
        t_start = max(timesteps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:].to(self.device)
        return timesteps, t_start

    def preprocess_images(self, batch_size, images=()):
        if self.nvtx_profile:
            nvtx_image_preprocess = nvtx.start_range(message="image_preprocess", color="pink")
        init_images = []
        for i in images:
            image = i.to(self.device).float()
            if image.shape[0] != batch_size:
                image = image.repeat(batch_size, 1, 1, 1)
            init_images.append(image)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_image_preprocess)
        return tuple(init_images)

    def encode_prompt(
        self,
        prompt,
        negative_prompt,
        encoder="clip",
        tokenizer=None,
        pooled_outputs=False,
        output_hidden_states=False,
        force_zeros_for_empty_prompt=False,
        do_classifier_free_guidance=True,
    ):
        if tokenizer is None:
            tokenizer = self.tokenizer

        if self.nvtx_profile:
            nvtx_clip = nvtx.start_range(message="clip", color="green")
        cudart.cudaEventRecord(self.events["clip-start"], 0)

        # Tokenize prompt
        text_input_ids = (
            tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.device)
        )

        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        outputs = self.run_engine(encoder, {"input_ids": text_input_ids})
        text_embeddings = outputs["text_embeddings"].clone()
        if output_hidden_states:
            hidden_states = outputs["hidden_states"].clone()

        # Note: negative prompt embedding is not needed for SD XL when guidance <= 1
        if do_classifier_free_guidance:
            # For SD XL base, handle force_zeros_for_empty_prompt
            is_empty_negative_prompt = all([not i for i in negative_prompt])
            if force_zeros_for_empty_prompt and is_empty_negative_prompt:
                uncond_embeddings = torch.zeros_like(text_embeddings)
                if output_hidden_states:
                    uncond_hidden_states = torch.zeros_like(hidden_states)
            else:
                # Tokenize negative prompt
                uncond_input_ids = (
                    tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    .input_ids.type(torch.int32)
                    .to(self.device)
                )

                outputs = self.run_engine(encoder, {"input_ids": uncond_input_ids})
                uncond_embeddings = outputs["text_embeddings"]
                if output_hidden_states:
                    uncond_hidden_states = outputs["hidden_states"]

            # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        if pooled_outputs:
            pooled_output = text_embeddings

        if output_hidden_states:
            if do_classifier_free_guidance:
                text_embeddings = torch.cat([uncond_hidden_states, hidden_states]).to(dtype=torch.float16)
            else:
                text_embeddings = hidden_states.to(dtype=torch.float16)

        cudart.cudaEventRecord(self.events["clip-stop"], 0)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_clip)

        if pooled_outputs:
            return text_embeddings, pooled_output
        return text_embeddings

    def denoise_latent(
        self,
        latents,
        text_embeddings,
        denoiser="unet",
        timesteps=None,
        step_offset=0,
        mask=None,
        masked_image_latents=None,
        guidance=7.5,
        add_kwargs=None,
    ):
        do_classifier_free_guidance = guidance > 1.0

        cudart.cudaEventRecord(self.events["denoise-start"], 0)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps

        for step_index, timestep in enumerate(timesteps):
            if self.nvtx_profile:
                nvtx_latent_scale = nvtx.start_range(message="latent_scale", color="pink")

            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, step_offset + step_index, timestep
            )

            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_latent_scale)

            # Predict the noise residual
            if self.nvtx_profile:
                nvtx_unet = nvtx.start_range(message="unet", color="blue")

            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            params = {
                "sample": latent_model_input,
                "timestep": timestep_float,
                "encoder_hidden_states": text_embeddings,
            }
            if add_kwargs:
                params.update(add_kwargs)

            noise_pred = self.run_engine(denoiser, params)["latent"]

            if self.nvtx_profile:
                nvtx.end_range(nvtx_unet)

            if self.nvtx_profile:
                nvtx_latent_step = nvtx.start_range(message="latent_step", color="pink")

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            if type(self.scheduler) == UniPCMultistepScheduler:
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            elif type(self.scheduler) == LCMScheduler:
                latents = self.scheduler.step(noise_pred, timestep, latents, generator=self.generator)[0]
            else:
                latents = self.scheduler.step(noise_pred, latents, step_offset + step_index, timestep)

            if self.nvtx_profile:
                nvtx.end_range(nvtx_latent_step)

        cudart.cudaEventRecord(self.events["denoise-stop"], 0)

        # The actual number of steps. It might be different from denoising_steps.
        self.actual_steps = len(timesteps)

        return latents

    def encode_image(self, init_image):
        if self.nvtx_profile:
            nvtx_vae = nvtx.start_range(message="vae_encoder", color="red")
        cudart.cudaEventRecord(self.events["vae_encoder-start"], 0)
        init_latents = self.run_engine("vae_encoder", {"images": init_image})["latent"]
        cudart.cudaEventRecord(self.events["vae_encoder-stop"], 0)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_vae)

        init_latents = self.vae_scaling_factor * init_latents
        return init_latents

    def decode_latent(self, latents):
        if self.nvtx_profile:
            nvtx_vae = nvtx.start_range(message="vae", color="red")
        cudart.cudaEventRecord(self.events["vae-start"], 0)
        images = self.backend.vae_decode(latents)
        cudart.cudaEventRecord(self.events["vae-stop"], 0)
        if self.nvtx_profile:
            nvtx.end_range(nvtx_vae)
        return images

    def print_summary(self, tic, toc, batch_size, vae_enc=False):
        print("|------------|--------------|")
        print("| {:^10} | {:^12} |".format("Module", "Latency"))
        print("|------------|--------------|")
        if vae_enc:
            print(
                "| {:^10} | {:>9.2f} ms |".format(
                    "VAE-Enc",
                    cudart.cudaEventElapsedTime(self.events["vae_encoder-start"], self.events["vae_encoder-stop"])[1],
                )
            )
        print(
            "| {:^10} | {:>9.2f} ms |".format(
                "CLIP", cudart.cudaEventElapsedTime(self.events["clip-start"], self.events["clip-stop"])[1]
            )
        )
        print(
            "| {:^10} | {:>9.2f} ms |".format(
                "UNet x " + str(self.actual_steps),
                cudart.cudaEventElapsedTime(self.events["denoise-start"], self.events["denoise-stop"])[1],
            )
        )
        print(
            "| {:^10} | {:>9.2f} ms |".format(
                "VAE-Dec", cudart.cudaEventElapsedTime(self.events["vae-start"], self.events["vae-stop"])[1]
            )
        )

        print("|------------|--------------|")
        print("| {:^10} | {:>9.2f} ms |".format("Pipeline", (toc - tic) * 1000.0))
        print("|------------|--------------|")
        print(f"Throughput: {batch_size / (toc - tic):.2f} image/s")

    @staticmethod
    def to_pil_image(images):
        images = (
            ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
        )

        from PIL import Image

        return [Image.fromarray(images[i]) for i in range(images.shape[0])]

    def save_images(self, images, pipeline, prompt):
        image_name_prefix = (
            pipeline + "".join(set(["-" + prompt[i].replace(" ", "_")[:10] for i in range(len(prompt))])) + "-"
        )

        images = self.to_pil_image(images)
        random_session_id = str(random.randint(1000, 9999))
        for i, image in enumerate(images):
            seed = str(self.get_current_seed())
            image_path = os.path.join(
                self.output_dir, image_name_prefix + str(i + 1) + "-" + random_session_id + "-" + seed + ".png"
            )
            print(f"Saving image {i+1} / {len(images)} to: {image_path}")

            from PIL import PngImagePlugin

            # TODO: This only saves info of the last pipeline. Save info of both base and refiner pipelines.
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("prompt", prompt[i])
            metadata.add_text("batch_size", str(len(images)))
            metadata.add_text("denoising_steps", str(self.denoising_steps))
            metadata.add_text("actual_steps", str(self.actual_steps))
            metadata.add_text("seed", seed)
            metadata.add_text("scheduler", self.current_scheduler)
            image.save(image_path, "PNG", pnginfo=metadata)
