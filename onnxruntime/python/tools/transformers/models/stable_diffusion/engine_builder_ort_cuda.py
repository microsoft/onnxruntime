# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os
import shutil

import torch
from diffusion_models import PipelineInfo
from engine_builder import EngineBuilder, EngineType

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import CudaSession

logger = logging.getLogger(__name__)


class OrtCudaEngine(CudaSession):
    def __init__(self, onnx_path, device_id: int = 0, enable_cuda_graph=False, disable_optimization=False):
        self.onnx_path = onnx_path
        self.provider = "CUDAExecutionProvider"
        self.provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)

        session_options = ort.SessionOptions()
        # When the model has been optimized by onnxruntime, we can disable optimization to save session creation time.
        if disable_optimization:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        logger.info("creating CUDA EP session for %s", onnx_path)
        ort_session = ort.InferenceSession(
            onnx_path,
            session_options,
            providers=[
                (self.provider, self.provider_options),
                "CPUExecutionProvider",
            ],
        )
        logger.info("created CUDA EP session for %s", onnx_path)

        device = torch.device("cuda", device_id)
        super().__init__(ort_session, device, enable_cuda_graph)

    def allocate_buffers(self, shape_dict, device):
        super().allocate_buffers(shape_dict)


class OrtCudaEngineBuilder(EngineBuilder):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        hf_token=None,
        device="cuda",
        use_cuda_graph=False,
    ):
        """
        Initializes the ONNX Runtime TensorRT ExecutionProvider Engine Builder.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            device (str):
                device to run.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        super().__init__(
            EngineType.ORT_CUDA,
            pipeline_info,
            max_batch_size=max_batch_size,
            hf_token=hf_token,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )

    def build_engines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_image_height=512,
        opt_image_width=512,
        opt_batch_size=1,
        force_engine_rebuild=False,
        device_id=0,
        disable_cuda_graph_models=None,
    ):
        self.torch_device = torch.device("cuda", device_id)
        self.load_models(framework_model_dir)

        if force_engine_rebuild:
            if os.path.isdir(onnx_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", onnx_dir)
                shutil.rmtree(onnx_dir)
            if os.path.isdir(engine_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", engine_dir)
                shutil.rmtree(engine_dir)

        if not os.path.isdir(engine_dir):
            os.makedirs(engine_dir)

        if not os.path.isdir(onnx_dir):
            os.makedirs(onnx_dir)

        # Export models to ONNX
        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.get_onnx_path(model_name, engine_dir, opt=True)
            if not os.path.exists(onnx_opt_path):
                if not os.path.exists(onnx_path):
                    logger.info("Exporting model: %s", onnx_path)
                    model = model_obj.load_model(framework_model_dir, self.hf_token)
                    with torch.inference_mode():
                        # For CUDA EP, export FP32 onnx since some graph fusion only supports fp32 graph pattern.
                        inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)

                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.info("Found cached model: %s", onnx_path)

                # Run graph optimization and convert to mixed precision (computation in FP16)
                if not os.path.exists(onnx_opt_path):
                    logger.info("Generating optimized model: %s", onnx_opt_path)
                    model_obj.optimize_ort(onnx_path, onnx_opt_path, to_fp16=True)
                else:
                    logger.info("Found cached optimized model: %s", onnx_opt_path)

        built_engines = {}
        for model_name in self.models:
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            onnx_opt_path = self.get_onnx_path(model_name, engine_dir, opt=True)

            use_cuda_graph = self.use_cuda_graph
            if self.use_cuda_graph and disable_cuda_graph_models and model_name in disable_cuda_graph_models:
                use_cuda_graph = False

            engine = OrtCudaEngine(onnx_opt_path, device_id=device_id, enable_cuda_graph=use_cuda_graph)
            logger.info("%s options for %s: %s", engine.provider, model_name, engine.provider_options)
            built_engines[model_name] = engine

        self.engines = built_engines

        return built_engines

    def run_engine(self, model_name, feed_dict):
        return self.engines[model_name].infer(feed_dict)
