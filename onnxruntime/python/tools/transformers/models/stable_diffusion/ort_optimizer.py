# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
ONNX Model Optimizer for Stable Diffusion
"""

import logging
import tempfile
from pathlib import Path

import onnx
from optimize_pipeline import has_external_data

from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.onnx_model_clip import ClipOnnxModel
from onnxruntime.transformers.onnx_model_unet import UnetOnnxModel
from onnxruntime.transformers.onnx_model_vae import VaeOnnxModel
from onnxruntime.transformers.optimizer import optimize_by_onnxruntime, optimize_model

logger = logging.getLogger(__name__)


class OrtStableDiffusionOptimizer:
    def __init__(self, model_type: str):
        assert model_type in ["vae", "unet", "clip"]
        self.model_type = model_type
        self.model_type_class_mapping = {
            "unet": UnetOnnxModel,
            "vae": VaeOnnxModel,
            "clip": ClipOnnxModel,
        }

    def optimize_by_ort(self, onnx_model, use_external_data_format=False):
        # Use this step to see the final graph that executed by Onnx Runtime.
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save to a temporary file so that we can load it with Onnx Runtime.
            logger.info("Saving a temporary model to run OnnxRuntime graph optimizations...")
            tmp_model_path = Path(tmp_dir) / "model.onnx"
            onnx_model.save_model_to_file(str(tmp_model_path), use_external_data_format=use_external_data_format)
            ort_optimized_model_path = Path(tmp_dir) / "optimized.onnx"
            optimize_by_onnxruntime(
                str(tmp_model_path),
                use_gpu=True,
                optimized_model_path=str(ort_optimized_model_path),
                save_as_external_data=use_external_data_format,
                external_data_filename="optimized.onnx_data",
            )
            model = onnx.load(str(ort_optimized_model_path), load_external_data=True)
            return self.model_type_class_mapping[self.model_type](model)

    def optimize(self, input_fp32_onnx_path, optimized_onnx_path, float16=True, keep_io_types=False, keep_outputs=None):
        """Optimize onnx model using ONNX Runtime transformers optimizer"""
        logger.info(f"Optimize {input_fp32_onnx_path}...")
        fusion_options = FusionOptions(self.model_type)
        if self.model_type in ["unet"] and not float16:
            fusion_options.enable_packed_kv = False
            fusion_options.enable_packed_qkv = False

        use_external_data_format = has_external_data(input_fp32_onnx_path)

        m = optimize_model(
            input_fp32_onnx_path,
            model_type=self.model_type,
            num_heads=0,  # will be deduced from graph
            hidden_size=0,  # will be deduced from graph
            opt_level=0,
            optimization_options=fusion_options,
            use_gpu=True,
        )

        if keep_outputs is None and self.model_type == "clip":
            # remove the pooler_output, and only keep the first output.
            keep_outputs = ["text_embeddings"]

        if keep_outputs:
            m.prune_graph(outputs=keep_outputs)

        if float16:
            logger.info("Convert to float16 ...")
            m.convert_float_to_float16(
                keep_io_types=keep_io_types,
            )

        # Note that ORT < 1.16 could not save model larger than 2GB.
        if float16 or (self.model_type != "unet"):
            m = self.optimize_by_ort(m, use_external_data_format=use_external_data_format)

        m.get_operator_statistics()
        m.get_fused_operator_statistics()
        m.save_model_to_file(optimized_onnx_path, use_external_data_format=use_external_data_format)
        logger.info("%s is optimized: %s", self.model_type, optimized_onnx_path)
