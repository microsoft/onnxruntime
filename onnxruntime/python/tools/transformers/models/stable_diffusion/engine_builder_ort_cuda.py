# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os
import shutil
from typing import List, Optional

import torch
from diffusion_models import PipelineInfo
from engine_builder import EngineBuilder, EngineType
from ort_utils import CudaSession

import onnxruntime as ort

logger = logging.getLogger(__name__)


class OrtCudaEngine(CudaSession):
    def __init__(
        self,
        onnx_path,
        device_id: int = 0,
        enable_cuda_graph: bool = False,
        disable_optimization: bool = False,
    ):
        self.onnx_path = onnx_path
        self.provider = "CUDAExecutionProvider"
        self.provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
        # self.provider_options["enable_skip_layer_norm_strict_mode"] = True

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


class _ModelConfig:
    """
    Configuration of one model (like Clip, UNet etc) on ONNX export and optimization for CUDA provider.
    For example, if you want to use fp32 in layer normalization, set the following:
        force_fp32_ops=["SkipLayerNormalization", "LayerNormalization"]
    """

    def __init__(
        self,
        onnx_opset_version: int,
        use_cuda_graph: bool,
        fp16: bool = True,
        force_fp32_ops: Optional[List[str]] = None,
        optimize_by_ort: bool = True,
    ):
        self.onnx_opset_version = onnx_opset_version
        self.use_cuda_graph = use_cuda_graph
        self.fp16 = fp16
        self.force_fp32_ops = force_fp32_ops
        self.optimize_by_ort = optimize_by_ort


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

        self.model_config = {}

    def _configure(
        self,
        model_name: str,
        onnx_opset_version: int,
        use_cuda_graph: bool,
        fp16: bool = True,
        force_fp32_ops: Optional[List[str]] = None,
        optimize_by_ort: bool = True,
    ):
        self.model_config[model_name] = _ModelConfig(
            onnx_opset_version,
            use_cuda_graph,
            fp16=fp16,
            force_fp32_ops=force_fp32_ops,
            optimize_by_ort=optimize_by_ort,
        )

    def configure_xl(self, onnx_opset_version: int):
        self._configure(
            "clip",
            onnx_opset_version=onnx_opset_version,
            use_cuda_graph=self.use_cuda_graph,
        )
        self._configure(
            "clip2",
            onnx_opset_version=onnx_opset_version,  # TODO: ArgMax-12 is not implemented in CUDA
            use_cuda_graph=False,  # TODO: fix Runtime Error with cuda graph
        )
        self._configure(
            "unetxl",
            onnx_opset_version=onnx_opset_version,
            use_cuda_graph=self.use_cuda_graph,
        )

        self._configure(
            "vae",
            onnx_opset_version=onnx_opset_version,
            use_cuda_graph=self.use_cuda_graph,
        )

    def build_engines(
        self,
        engine_dir: str,
        framework_model_dir: str,
        onnx_dir: str,
        tmp_dir: Optional[str] = None,
        onnx_opset_version: int = 17,
        force_engine_rebuild: bool = False,
        device_id: int = 0,
        save_fp32_intermediate_model=False,
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

        # Add default configuration if missing
        if self.pipeline_info.is_xl():
            self.configure_xl(onnx_opset_version)
        for model_name in self.models:
            if model_name not in self.model_config:
                self.model_config[model_name] = _ModelConfig(onnx_opset_version, self.use_cuda_graph)

        # Load lora only when we need export text encoder or UNet to ONNX.
        load_lora = False
        if self.pipeline_info.lora_weights:
            for model_name in self.models:
                if model_name not in ["clip", "clip2", "unet", "unetxl"]:
                    continue
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)

                suffix = ".fp16" if self.model_config[model_name].fp16 else ".fp32"
                onnx_opt_path = self.get_onnx_path(model_name, engine_dir, opt=True, suffix=suffix)
                if not os.path.exists(onnx_opt_path):
                    if not os.path.exists(onnx_path):
                        load_lora = True

        # Export models to ONNX
        self.disable_torch_spda()
        pipe = self.load_pipeline_with_lora() if load_lora else None

        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
            suffix = ".fp16" if self.model_config[model_name].fp16 else ".fp32"
            onnx_opt_path = self.get_onnx_path(model_name, engine_dir, opt=True, suffix=suffix)
            if not os.path.exists(onnx_opt_path):
                if not os.path.exists(onnx_path):
                    print("----")
                    logger.info("Exporting model: %s", onnx_path)

                    model = self.get_or_load_model(pipe, model_name, model_obj, framework_model_dir)
                    model.to(torch.float32)

                    with torch.inference_mode():
                        # For CUDA EP, export FP32 onnx since some graph fusion only supports fp32 graph pattern.
                        # Export model with sample of batch size 1, image size 512 x 512
                        inputs = model_obj.get_sample_input(1, 512, 512)

                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=self.model_config[model_name].onnx_opset_version,
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

                # Generate fp32 optimized model.
                # If final target is fp16 model, we save fp32 optimized model so that it is easy to tune
                # fp16 conversion. That could save a lot of time in developing.
                use_fp32_intermediate = save_fp32_intermediate_model and self.model_config[model_name].fp16
                onnx_fp32_path = onnx_path
                if use_fp32_intermediate:
                    onnx_fp32_path = self.get_onnx_path(model_name, engine_dir, opt=True, suffix=".fp32")
                    if not os.path.exists(onnx_fp32_path):
                        print("------")
                        logger.info("Generating optimized model: %s", onnx_fp32_path)
                        model_obj.optimize_ort(
                            onnx_path,
                            onnx_fp32_path,
                            to_fp16=False,
                            fp32_op_list=self.model_config[model_name].force_fp32_ops,
                            optimize_by_ort=self.model_config[model_name].optimize_by_ort,
                            tmp_dir=self.get_model_dir(model_name, tmp_dir, opt=False, suffix=".fp32", create=False),
                        )
                    else:
                        logger.info("Found cached optimized model: %s", onnx_fp32_path)

                # Generate the final optimized model.
                if not os.path.exists(onnx_opt_path):
                    print("------")
                    logger.info("Generating optimized model: %s", onnx_opt_path)

                    # When there is fp32 intermediate optimized model, this will just convert model from fp32 to fp16.
                    optimize_by_ort = False if use_fp32_intermediate else self.model_config[model_name].optimize_by_ort

                    model_obj.optimize_ort(
                        onnx_fp32_path,
                        onnx_opt_path,
                        to_fp16=self.model_config[model_name].fp16,
                        fp32_op_list=self.model_config[model_name].force_fp32_ops,
                        optimize_by_ort=optimize_by_ort,
                        optimize_by_fusion=not use_fp32_intermediate,
                        tmp_dir=self.get_model_dir(model_name, tmp_dir, opt=False, suffix=".fp16", create=False),
                    )
                else:
                    logger.info("Found cached optimized model: %s", onnx_opt_path)
        self.enable_torch_spda()

        built_engines = {}
        for model_name in self.models:
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            suffix = ".fp16" if self.model_config[model_name].fp16 else ".fp32"
            onnx_opt_path = self.get_onnx_path(model_name, engine_dir, opt=True, suffix=suffix)

            use_cuda_graph = self.model_config[model_name].use_cuda_graph

            engine = OrtCudaEngine(
                onnx_opt_path,
                device_id=device_id,
                enable_cuda_graph=use_cuda_graph,
                disable_optimization=False,
            )

            logger.info("%s options for %s: %s", engine.provider, model_name, engine.provider_options)
            built_engines[model_name] = engine

        self.engines = built_engines

        return built_engines

    def run_engine(self, model_name, feed_dict):
        return self.engines[model_name].infer(feed_dict)
