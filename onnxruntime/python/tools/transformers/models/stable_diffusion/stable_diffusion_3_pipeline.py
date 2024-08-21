#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

from cuda import cudart
from models import (
    get_clip_embedding_dim,
    SD3_CLIPGModel,
    SD3_CLIPLModel,
    SD3_T5XXLModel,
    SD3_MMDiTModel,
    SD3_VAEEncoderModel,
    SD3_VAEDecoderModel
)
import nvtx
import os
import math
import pathlib
import tensorrt as trt
import time
import torch
from utilities import (
    PIPELINE_TYPE,
    TRT_LOGGER,
    Engine,
    save_image,
)
from utils_sd3.other_impls import SD3Tokenizer
from utils_sd3.sd3_impls import SD3LatentFormat, sample_euler

class StableDiffusion3Pipeline:
    """
    Application showcasing the acceleration of Stable Diffusion 3 pipelines using NVidia TensorRT.
    """
    def __init__(
        self,
        version='sd3',
        pipeline_type=PIPELINE_TYPE.TXT2IMG,
        max_batch_size=16,
        shift=1.0,
        cfg_scale=5,
        denoising_steps=50,
        denoising_percentage=0.6,
        input_image=None,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        use_cuda_graph=False,
        framework_model_dir='pytorch_model',
        torch_inference='',
    ):
        """
        Initializes the Stable Diffusion 3 pipeline.

        Args:
            version (str):
                The version of the pipeline. Should be one of ['sd3]
            pipeline_type (PIPELINE_TYPE):
                Type of current pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            shift (float):
                Shift parameter for MMDiT model. Default: 1.0
            cfg_scale (int):
                CFG Scale used for denoising. Default: 5
            denoising_steps (int):
                Number of denoising steps. Default: 1.0
            denoising_percentage (float):
                Denoising percentage. Default: 0.6
            input_image (float):
                Input image for conditioning. Default: None
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
            torch_inference (str):
                Run inference with PyTorch (using specified compilation mode) instead of TensorRT.
        """

        self.max_batch_size = max_batch_size
        self.shift = shift
        self.cfg_scale = cfg_scale
        self.denoising_steps = denoising_steps
        self.input_image = input_image
        self.denoising_percentage = denoising_percentage if input_image is not None else 1.0

        self.framework_model_dir = framework_model_dir
        self.output_dir = output_dir
        for directory in [self.framework_model_dir, self.output_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        self.version = version

        # Pipeline type
        self.pipeline_type = pipeline_type
        self.stages = ['clip_g', 'clip_l', 't5xxl', 'mmdit', 'vae_decoder']
        if input_image is not None:
            self.stages += ['vae_encoder']

        self.config = {}
        self.config['clip_hidden_states'] = True
        self.torch_inference = torch_inference
        if self.torch_inference:
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True
        self.use_cuda_graph = use_cuda_graph

        # initialized in loadEngines()
        self.models = {}
        self.torch_models = {}
        self.engine = {}
        self.shared_device_memory = None

        # initialized in loadResources()
        self.events = {}
        self.generator = None
        self.markers = {}
        self.seed = None
        self.stream = None
        self.tokenizer = None

    def loadResources(self, image_height, image_width, batch_size, seed):
        # Initialize noise generator
        if seed:
            self.seed = seed
            self.generator = torch.Generator(device="cuda").manual_seed(seed)

        # Create CUDA events and stream
        for stage in ['clip_g', 'clip_l', 't5xxl', 'denoise', 'vae_encode', 'vae_decode']:
            self.events[stage] = [cudart.cudaEventCreate()[1], cudart.cudaEventCreate()[1]]
        self.stream = cudart.cudaStreamCreate()[1]

        # Allocate TensorRT I/O buffers
        if not self.torch_inference:
            for model_name, obj in self.models.items():
                if self.torch_fallback[model_name]:
                    continue
                self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)

    def teardown(self):
        for e in self.events.values():
            cudart.cudaEventDestroy(e[0])
            cudart.cudaEventDestroy(e[1])

        for engine in self.engine.values():
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def getOnnxPath(self, model_name, onnx_dir, opt=True, suffix=''):
        onnx_model_dir = os.path.join(onnx_dir, model_name+suffix+('.opt' if opt else ''))
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, 'model.onnx')

    def getEnginePath(self, model_name, engine_dir, enable_refit=False, suffix=''):
        return os.path.join(engine_dir, model_name+suffix+('.refit' if enable_refit else '')+'.trt'+trt.__version__+'.plan')

    def loadEngines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=False,
        static_shape=True,
        enable_all_tactics=False,
        timing_cache=None,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to store the TensorRT engines.
            framework_model_dir (str):
                Directory to store the framework model ckpt.
            onnx_dir (str):
                Directory to store the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to speed up TensorRT build.
        """
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        # Load pipeline models
        models_args = {'version': self.version, 'pipeline': self.pipeline_type, 'device': self.device,
            'hf_token': self.hf_token, 'verbose': self.verbose, 'framework_model_dir': framework_model_dir,
            'max_batch_size': self.max_batch_size}

        # Load text tokenizer
        self.tokenizer = SD3Tokenizer()

        # Load text encoders
        embedding_dim = get_clip_embedding_dim(self.version, self.pipeline_type)
        if 'clip_g' in self.stages:
            self.models['clip_g'] = SD3_CLIPGModel(**models_args, fp16=True, embedding_dim=embedding_dim)

        if 'clip_l' in self.stages:
            self.models['clip_l'] = SD3_CLIPLModel(**models_args, fp16=True, embedding_dim=embedding_dim)

        if 't5xxl' in self.stages:
            self.models['t5xxl'] = SD3_T5XXLModel(**models_args, fp16=True, embedding_dim=embedding_dim)

        # Load MMDiT model
        if 'mmdit' in self.stages:
            self.models['mmdit'] = SD3_MMDiTModel(**models_args, fp16=True, shift=self.shift)

        # Load VAE Encoder model
        if 'vae_encoder' in self.stages:
            self.models['vae_encoder'] = SD3_VAEEncoderModel(**models_args, fp16=True)

        # Load VAE Decoder model
        if 'vae_decoder' in self.stages:
            self.models['vae_decoder'] = SD3_VAEDecoderModel(**models_args, fp16=True)

        # Configure pipeline models to load
        model_names = self.models.keys()
        # Torch fallback
        self.torch_fallback = dict(zip(model_names, [self.torch_inference or model_name in ('clip_g', 'clip_l', 't5xxl') for model_name in model_names]))

        onnx_path = dict(zip(model_names, [self.getOnnxPath(model_name, onnx_dir, opt=False) for model_name in model_names]))
        onnx_opt_path = dict(zip(model_names, [self.getOnnxPath(model_name, onnx_dir) for model_name in model_names]))
        engine_path = dict(zip(model_names, [self.getEnginePath(model_name, engine_dir) for model_name in model_names]))

        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            # Export models to ONNX
            do_export_onnx = not os.path.exists(engine_path[model_name]) and not os.path.exists(onnx_opt_path[model_name])
            if do_export_onnx:
                obj.export_onnx(onnx_path[model_name], onnx_opt_path[model_name], onnx_opset, opt_image_height, opt_image_width, static_shape=static_shape)

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            engine = Engine(engine_path[model_name])
            if not os.path.exists(engine_path[model_name]):
                update_output_names = obj.get_output_names() + obj.extra_output_names if obj.extra_output_names else None
                extra_build_args = {'verbose': self.verbose}
                fp16amp = obj.fp16
                engine.build(onnx_opt_path[model_name],
                    fp16=fp16amp,
                    input_profile=obj.get_input_profile(
                        opt_batch_size, opt_image_height, opt_image_width,
                        static_batch=static_batch, static_shape=static_shape
                    ),
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    update_output_names=update_output_names,
                    **extra_build_args)
            self.engine[model_name] = engine

        # Load TensorRT engines
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name]:
                continue
            self.engine[model_name].load()

        # Load torch models
        for model_name, obj in self.models.items():
            if self.torch_fallback[model_name] or model_name == 'mmdit':
                self.torch_models[model_name] = obj.get_model(torch_inference=self.torch_inference)

    def calculateMaxDeviceMemory(self):
        max_device_memory = 0
        for model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activateEngines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.calculateMaxDeviceMemory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engine.values():
            engine.activate(reuse_device_memory=self.shared_device_memory)

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width):
        return torch.ones(batch_size, unet_channels, latent_height, latent_width, device="cuda") * 0.0609

    def profile_start(self, name, color='blue'):
        if self.nvtx_profile:
            self.markers[name] = nvtx.start_range(message=name, color=color)
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][0], 0)

    def profile_stop(self, name):
        if name in self.events:
            cudart.cudaEventRecord(self.events[name][1], 0)
        if self.nvtx_profile:
            nvtx.end_range(self.markers[name])

    def print_summary(self, denoising_steps, walltime_ms, batch_size):
        print('|-----------------|--------------|')
        print('| {:^15} | {:^12} |'.format('Module', 'Latency'))
        print('|-----------------|--------------|')
        if 'vae_encoder' in self.stages:
            print('| {:^15} | {:>9.2f} ms |'.format('VAE Encoder', cudart.cudaEventElapsedTime(self.events['vae_encode'][0], self.events['vae_encode'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('CLIP-G', cudart.cudaEventElapsedTime(self.events['clip_g'][0], self.events['clip_g'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('CLIP-L', cudart.cudaEventElapsedTime(self.events['clip_l'][0], self.events['clip_l'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('T5XXL', cudart.cudaEventElapsedTime(self.events['t5xxl'][0], self.events['t5xxl'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('MMDiT'+' x '+str(denoising_steps), cudart.cudaEventElapsedTime(self.events['denoise'][0], self.events['denoise'][1])[1]))
        print('| {:^15} | {:>9.2f} ms |'.format('VAE Decoder', cudart.cudaEventElapsedTime(self.events['vae_decode'][0], self.events['vae_decode'][1])[1]))
        print('|-----------------|--------------|')
        print('| {:^15} | {:>9.2f} ms |'.format('Pipeline', walltime_ms))
        print('|-----------------|--------------|')
        print('Throughput: {:.2f} image/s'.format(batch_size*1000./walltime_ms))

    def save_image(self, images, pipeline, prompt, seed):
        # Save image
        image_name_prefix = pipeline+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(len(prompt))]))+'-'+str(seed)+'-'
        save_image(images, self.output_dir, image_name_prefix)

    def encode_prompt(self, prompt, negative_prompt):
        def encode_token_weights(model_name, token_weight_pairs):
            self.profile_start(model_name, color='green')

            tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
            tokens = torch.tensor([tokens], dtype=torch.int64, device=self.device)
            if self.torch_inference or self.torch_fallback[model_name]:
                out, pooled = self.torch_models[model_name](tokens)
            else:
                out = self.runEngine('t5xxl', {'input_ids': tokens})['text_embeddings']
                pooled = None
            
            self.profile_stop(model_name)

            if pooled is not None:
                first_pooled = pooled[0:1].cuda()
            else:
                first_pooled = pooled
            output = [out[0:1]]
            return torch.cat(output, dim=-2).cuda(), first_pooled

        def tokenize(prompt):
            tokens = self.tokenizer.tokenize_with_weights(prompt)
            l_out, l_pooled = encode_token_weights('clip_l', tokens["l"])
            g_out, g_pooled = encode_token_weights('clip_g', tokens["g"])
            t5_out, _ = encode_token_weights('t5xxl', tokens["t5xxl"])
            lg_out = torch.cat([l_out, g_out], dim=-1)
            lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))

            return torch.cat([lg_out, t5_out], dim=-2), torch.cat((l_pooled, g_pooled), dim=-1)

        conditioning = tokenize(prompt[0])
        neg_conditioning = tokenize(negative_prompt[0])
        return conditioning, neg_conditioning
    
    def denoise_latent(self, latent, conditioning, neg_conditioning, model_name='mmdit'):
        def get_noise(latent):
            return torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=self.generator, device="cuda").to(latent.dtype)

        def get_sigmas(sampling, steps):
            start = sampling.timestep(sampling.sigma_max)
            end = sampling.timestep(sampling.sigma_min)
            timesteps = torch.linspace(start, end, steps)
            sigs = []
            for x in range(len(timesteps)):
                ts = timesteps[x]
                sigs.append(sampling.sigma(ts))
            sigs += [0.0]
            return torch.FloatTensor(sigs)

        def max_denoise(sigmas):
            max_sigma = float(self.torch_models[model_name].model_sampling.sigma_max)
            sigma = float(sigmas[0])
            return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

        def fix_cond(cond):
            cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
            return { "c_crossattn": cond, "y": pooled }
        
        def cfg_denoiser(x, timestep, cond, uncond, cond_scale):
            # Run cond and uncond in a batch together
            sample = torch.cat([x, x])
            sigma = torch.cat([timestep, timestep])
            c_crossattn = torch.cat([cond["c_crossattn"], uncond["c_crossattn"]])
            y = torch.cat([cond["y"], uncond["y"]])
            if self.torch_inference:
                with torch.autocast("cuda", dtype=torch.float16):
                    batched = self.torch_models[model_name](sample, sigma, c_crossattn=c_crossattn, y=y)
            else:
                input_dict = {'sample': sample, 'sigma': sigma, 'c_crossattn': c_crossattn, 'y': y}
                batched = self.runEngine(model_name, input_dict)['latent']

            # Then split and apply CFG Scaling
            pos_out, neg_out = batched.chunk(2)
            scaled = neg_out + (pos_out - neg_out) * cond_scale
            return scaled

        self.profile_start('denoise', color='blue')

        latent = latent.half().cuda()
        noise = get_noise(latent).cuda()
        sigmas = get_sigmas(self.torch_models[model_name].model_sampling, self.denoising_steps).cuda()
        sigmas = sigmas[int(self.denoising_steps * (1 - self.denoising_percentage)):]
        conditioning = fix_cond(conditioning)
        neg_conditioning = fix_cond(neg_conditioning)

        noise_scaled = self.torch_models[model_name].model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise(sigmas))
        extra_args = { "cond": conditioning, "uncond": neg_conditioning, "cond_scale": self.cfg_scale }
        latent = sample_euler(cfg_denoiser, noise_scaled, sigmas, extra_args=extra_args)
        latent = SD3LatentFormat().process_out(latent)

        self.profile_stop('denoise')

        return latent

    def encode_image(self):
        self.input_image = self.input_image.to(self.device)
        self.profile_start('vae_encode', color='orange')
        if self.torch_inference:
            with torch.autocast("cuda", dtype=torch.float16):
                latent = self.torch_models['vae_encoder'](self.input_image)
        else:
            latent = self.runEngine('vae_encoder', {'images': self.input_image})['latent']

        latent = SD3LatentFormat().process_in(latent)
        self.profile_stop('vae_encode')
        return latent

    def decode_latent(self, latent):
        self.profile_start('vae_decode', color='red')
        if self.torch_inference:
            with torch.autocast("cuda", dtype=torch.float16):
                image = self.torch_models['vae_decoder'](latent)
        else:
            image = self.runEngine('vae_decoder', {'latent': latent})['images']
        image = image.float()
        self.profile_stop('vae_decode')
        return image

    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        warmup=False,
        save_image=True,
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            warmup (bool):
                Indicate if this is a warmup run.
            save_image (bool):
                Save the generated image (if applicable)
        """
        assert len(prompt) == len(negative_prompt)
        batch_size = len(prompt)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        if self.generator and self.seed:
            self.generator.manual_seed(self.seed)

        with torch.inference_mode(), trt.Runtime(TRT_LOGGER):
            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # Initialize Latents
            latent = self.initialize_latents(batch_size=batch_size,
                unet_channels=16,
                latent_height=latent_height,
                latent_width=latent_width)
            
            # Encode input image
            if self.input_image is not None:
                latent = self.encode_image()

            # Get Conditionings
            conditioning, neg_conditioning = self.encode_prompt(prompt, negative_prompt)

            # Denoise
            latent = self.denoise_latent(latent, conditioning, neg_conditioning)
            
            # Decode Latents
            images = self.decode_latent(latent)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

        walltime_ms = (e2e_toc - e2e_tic) * 1000.
        if not warmup:
            num_inference_steps = int(self.denoising_steps * self.denoising_percentage)
            self.print_summary(num_inference_steps, walltime_ms, batch_size)
            if save_image:
                self.save_image(images, self.pipeline_type.name.lower(), prompt, self.seed)

        return images, walltime_ms

    def run(self, prompt, negative_prompt, height, width, batch_size, batch_count, num_warmup_runs, use_cuda_graph, **kwargs):
        # Process prompt
        if not isinstance(prompt, list):
            raise ValueError(f"`prompt` must be of type `str` list, but is {type(prompt)}")
        prompt = prompt * batch_size

        if not isinstance(negative_prompt, list):
            raise ValueError(f"`--negative-prompt` must be of type `str` list, but is {type(negative_prompt)}")
        if len(negative_prompt) == 1:
            negative_prompt = negative_prompt * batch_size

        num_warmup_runs = max(1, num_warmup_runs) if use_cuda_graph else num_warmup_runs
        if num_warmup_runs > 0:
            print("[I] Warming up ..")
            for _ in range(num_warmup_runs):
                self.infer(prompt, negative_prompt, height, width, warmup=True, **kwargs)

        for _ in range(batch_count):
            print("[I] Running StableDiffusion3 pipeline")
            if self.nvtx_profile:
                cudart.cudaProfilerStart()
            self.infer(prompt, negative_prompt, height, width, warmup=False, **kwargs)
            if self.nvtx_profile:
                cudart.cudaProfilerStop()
