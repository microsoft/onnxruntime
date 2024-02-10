# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os
from typing import List, Optional

import onnx
import torch
from diffusion_models import PipelineInfo
from engine_builder import EngineBuilder, EngineType
from packaging import version

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import DmlSession
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime import OrtValue

logger = logging.getLogger(__name__)


class OrtDmlEngine(DmlSession):
    def __init__(
        self,
        onnx_path,
        device_id: int = 0,
        disable_optimization: bool = False,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ):
        self.onnx_path = onnx_path
        self.provider = "DmlExecutionProvider"
        self.provider_options = DmlSession.get_dml_provider_options(device_id)

        session_options = ort.SessionOptions()
        session_options.add_free_dimension_override_by_name("unet_sample_batch", batch_size * 2) # Olive naming
        session_options.add_free_dimension_override_by_name("unet_sample_height", height // 8) # Olive naming
        session_options.add_free_dimension_override_by_name("unet_sample_width", width // 8) # Olive naming
        session_options.add_free_dimension_override_by_name("unet_sample_channels", 4) # Olive naming
        session_options.add_free_dimension_override_by_name("unet_time_batch", 1) # Olive naming
        session_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size * 2) # Olive naming
        session_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77) # Olive naming
        session_options.add_free_dimension_override_by_name("2B", batch_size * 2)
        session_options.add_free_dimension_override_by_name("B", batch_size)
        session_options.add_free_dimension_override_by_name("H", height // 8)
        session_options.add_free_dimension_override_by_name("W", width // 8)
        session_options.add_free_dimension_override_by_name("S", 77)

        # When the model has been optimized by onnxruntime, we can disable optimization to save session creation time.
        if disable_optimization:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        logger.info("creating DML EP session for %s", onnx_path)
        ort_session = ort.InferenceSession(
            onnx_path,
            session_options,
            providers=[
                (self.provider, self.provider_options),
                "CPUExecutionProvider",
            ],
        )
        logger.info("created DML EP session for %s", onnx_path)

        super().__init__(ort_session)

    def allocate_buffers(self, shape_dict, device):
        super().allocate_buffers(shape_dict)


class _ModelConfig:
    """
    Configuration of one model (like Clip, UNet etc) on ONNX export and optimization for DML provider.
    For example, if you want to use fp32 in layer normalization, set the following:
        force_fp32_ops=["SkipLayerNormalization", "LayerNormalization"]
    """

    def __init__(
        self,
        onnx_opset_version: int,
        fp16: bool = True,
        force_fp32_ops: Optional[List[str]] = None,
        optimize_by_ort: bool = True,
    ):
        self.onnx_opset_version = onnx_opset_version
        self.fp16 = fp16
        self.force_fp32_ops = force_fp32_ops
        self.optimize_by_ort = optimize_by_ort


class OrtDmlEngineBuilder(EngineBuilder):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        device="cpu",
    ):
        """
        Initializes the ONNX Runtime TensorRT ExecutionProvider Engine Builder.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            device (str):
                device to run.
        """
        super().__init__(
            EngineType.ORT_DML,
            pipeline_info,
            max_batch_size=max_batch_size,
            device=device,
        )

        self.model_config = {}

    def _configure(
        self,
        model_name: str,
        onnx_opset_version: int,
        fp16: bool = True,
        force_fp32_ops: Optional[List[str]] = None,
        optimize_by_ort: bool = True,
    ):
        self.model_config[model_name] = _ModelConfig(
            onnx_opset_version,
            fp16=fp16,
            force_fp32_ops=force_fp32_ops,
            optimize_by_ort=optimize_by_ort,
        )

    def configure_xl(self, onnx_opset_version: int):
        self._configure(
            "clip",
            onnx_opset_version=onnx_opset_version,
        )
        self._configure(
            "clip2",
            onnx_opset_version=onnx_opset_version,
        )
        self._configure(
            "unetxl",
            onnx_opset_version=onnx_opset_version,
        )

        self._configure(
            "vae",
            onnx_opset_version=onnx_opset_version,
        )

    def optimized_onnx_path(self, engine_dir, model_name):
        suffix = "" if self.model_config[model_name].fp16 else ".fp32"
        return self.get_onnx_path(model_name, engine_dir, opt=True, suffix=suffix)

    def import_diffusers_engine(self, diffusers_onnx_dir: str, engine_dir: str):
        """Import optimized onnx models for diffusers from Olive or optimize_pipeline tools.

        Args:
            diffusers_onnx_dir (str): optimized onnx directory of Olive
            engine_dir (str): the directory to store imported onnx
        """
        if version.parse(ort.__version__) < version.parse("1.16.3"):
            print("Skip importing since onnxruntime-directml version < 1.16.3.")
            return

        for model_name, model_obj in self.models.items():
            onnx_import_path = self.optimized_onnx_path(diffusers_onnx_dir, model_name)
            if not os.path.exists(onnx_import_path):
                print(f"{onnx_import_path} not existed. Skip importing.")
                continue

            onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)
            if os.path.exists(onnx_opt_path):
                print(f"{onnx_opt_path} existed. Skip importing.")
                continue

            if model_name == "vae" and self.pipeline_info.is_xl():
                print(f"Skip importing VAE since it is not fully compatible with float16: {onnx_import_path}.")
                continue

            model = OnnxModel(onnx.load(onnx_import_path, load_external_data=True))

            if model_name in ["clip", "clip2"]:
                hidden_states_per_layer = []
                for output in model.graph().output:
                    if output.name.startswith("hidden_states."):
                        hidden_states_per_layer.append(output.name)
                if hidden_states_per_layer:
                    kept_hidden_states = hidden_states_per_layer[-2 - model_obj.clip_skip]
                    model.rename_graph_output(kept_hidden_states, "hidden_states")

                model.rename_graph_output(
                    "last_hidden_state" if model_name == "clip" else "text_embeds", "text_embeddings"
                )
                model.prune_graph(
                    ["text_embeddings", "hidden_states"] if hidden_states_per_layer else ["text_embeddings"]
                )

                if model_name == "clip2":
                    model.change_graph_input_type(model.find_graph_input("input_ids"), onnx.TensorProto.INT32)

                model.save_model_to_file(onnx_opt_path, use_external_data_format=(model_name == "clip2"))
            elif model_name in ["unet", "unetxl"]:
                model.rename_graph_output("out_sample", "latent")
                model.save_model_to_file(onnx_opt_path, use_external_data_format=True)

            del model
            continue

    def build_engines(
        self,
        engine_dir: str,
        framework_model_dir: str,
        onnx_dir: str,
        tmp_dir: Optional[str] = None,
        onnx_opset_version: int = 17,
        device_id: int = 0,
        save_fp32_intermediate_model: bool = False,
        import_engine_dir: Optional[str] = None,
        batch_size=1,
        height=512,
        width=512,
    ):
        self.load_models(framework_model_dir)
        self.device_id = device_id

        if not os.path.isdir(engine_dir):
            os.makedirs(engine_dir)

        if not os.path.isdir(onnx_dir):
            os.makedirs(onnx_dir)

        # Add default configuration if missing
        if self.pipeline_info.is_xl():
            self.configure_xl(onnx_opset_version)
        for model_name in self.models:
            if model_name not in self.model_config:
                self.model_config[model_name] = _ModelConfig(onnx_opset_version)

        # Import Engine
        if import_engine_dir:
            if self.pipeline_info.is_xl():
                self.import_diffusers_engine(import_engine_dir, engine_dir)
            else:
                print(f"Only support importing SDXL onnx. Ignore --engine-dir {import_engine_dir}")

        # Load lora only when we need export text encoder or UNet to ONNX.
        load_lora = False
        if self.pipeline_info.lora_weights:
            for model_name in self.models:
                if model_name not in ["clip", "clip2", "unet", "unetxl"]:
                    continue
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)
                if not os.path.exists(onnx_opt_path):
                    if not os.path.exists(onnx_path):
                        load_lora = True
                        break

        # Export models to ONNX
        self.disable_torch_spda()
        pipe = self.load_pipeline_with_lora() if load_lora else None

        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)
            if not os.path.exists(onnx_opt_path):
                if not os.path.exists(onnx_path):
                    print("----")
                    logger.info("Exporting model: %s", onnx_path)

                    model = self.get_or_load_model(pipe, model_name, model_obj, framework_model_dir)
                    model = model.to(torch.float32)

                    with torch.inference_mode():
                        # For DML EP, export FP32 onnx since some graph fusion only supports fp32 graph pattern.
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
                    gc.collect()
                else:
                    logger.info("Found cached model: %s", onnx_path)

                custom_fusion_options = {
                    "enable_gelu": True,
                    "enable_layer_norm": True,
                    "enable_attention": True,
                    "use_multi_head_attention": True,
                    "enable_skip_layer_norm": False,
                    "enable_embed_layer_norm": True,
                    "enable_bias_skip_layer_norm": False,
                    "enable_bias_gelu": True,
                    "enable_gelu_approximation": False,
                    "enable_qordered_matmul": False,
                    "enable_shape_inference": True,
                    "enable_gemm_fast_gelu": False,
                    "enable_nhwc_conv": False,
                    "enable_group_norm": True,
                    "enable_bias_splitgelu": False,
                    "enable_packed_qkv": True,
                    "enable_packed_kv": True,
                    "enable_bias_add": False,
                    "group_norm_channels_last": False,
                }

                force_fp16_inputs = {
                    "GroupNorm": [0, 1, 2]
                }

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
                            optimize_by_ort=False,
                            tmp_dir=self.get_model_dir(model_name, tmp_dir, opt=False, suffix=".fp32", create=False),
                            custom_fusion_options=custom_fusion_options,
                            force_fp16_inputs=force_fp16_inputs,
                        )
                    else:
                        logger.info("Found cached optimized model: %s", onnx_fp32_path)

                # Generate the final optimized model.
                if not os.path.exists(onnx_opt_path):
                    print("------")
                    logger.info("Generating optimized model: %s", onnx_opt_path)

                    model_obj.optimize_ort(
                        onnx_fp32_path,
                        onnx_opt_path,
                        to_fp16=self.model_config[model_name].fp16,
                        optimize_by_ort=False,
                        optimize_by_fusion=not use_fp32_intermediate,
                        tmp_dir=self.get_model_dir(model_name, tmp_dir, opt=False, suffix=".ort", create=False),
                        custom_fusion_options=custom_fusion_options,
                        force_fp16_inputs=force_fp16_inputs,
                    )
                else:
                    logger.info("Found cached optimized model: %s", onnx_opt_path)
        self.enable_torch_spda()

        built_engines = {}
        for model_name in self.models:
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)

            engine = OrtDmlEngine(
                onnx_opt_path,
                device_id=device_id,
                disable_optimization=False,
                batch_size=batch_size,
                height=height,
                width=width,
            )

            logger.info("%s options for %s: %s", engine.provider, model_name, engine.provider_options)
            built_engines[model_name] = engine

        self.engines = built_engines

    def run_engine(self, model_name, feed_dict):
        ort_feed_dict = {}

        for input_name in feed_dict:
            numpy_tensor = feed_dict[input_name].numpy()
            if numpy_tensor.shape == ():
                numpy_tensor = numpy_tensor.reshape((1,))

            ort_feed_dict[input_name] = OrtValue.ortvalue_from_numpy(
                numpy_tensor,
                device_type="dml",
                device_id=self.device_id,
            )
        ort_outputs = self.engines[model_name].infer(ort_feed_dict)
        pytorch_outputs = {}
        for output_name in ort_outputs:
            pytorch_outputs[output_name] = torch.from_numpy(ort_outputs[output_name].numpy())

        return pytorch_outputs
