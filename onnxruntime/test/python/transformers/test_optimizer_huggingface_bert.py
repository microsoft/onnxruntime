#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# For live logging, use the following command:
#   pytest -o log_cli=true --log-cli-level=DEBUG test_optimizer_huggingface_bert.py

import shutil
import unittest
from pathlib import Path

import torch
from parity_utilities import find_transformers_source
from transformers.utils import default_cache_path

if find_transformers_source():
    from benchmark_helper import ConfigModifier, OptimizerInfo, Precision
    from compare_bert_results import run_test as bert_parity_test
    from onnx_exporter import export_onnx_model_from_pt
else:
    from onnxruntime.transformers.benchmark_helper import ConfigModifier, OptimizerInfo, Precision
    from onnxruntime.transformers.compare_bert_results import run_test as bert_parity_test
    from onnxruntime.transformers.onnx_exporter import export_onnx_model_from_pt


class TestHuggingfaceBertModelOptimization(unittest.TestCase):
    def run_optimizer_on_model(
        self,
        model_name,
        expected_fusion_result_list,
        inputs_count=1,
        validate_model=True,
        opset_version=16,
        use_external_data_format=False,
        model_type="bert",
    ):
        onnx_dir = Path(".") / "onnx_models" / model_name
        shutil.rmtree(onnx_dir, ignore_errors=True)

        Path(onnx_dir).mkdir(parents=True, exist_ok=True)

        model_fusion_statistics = {}

        input_names = ["input_ids", "attention_mask", "token_type_ids"]

        config_modifier = ConfigModifier(None)
        fusion_options = None
        model_class = "AutoModel"
        with torch.no_grad():
            optimized_model_path, is_valid_onnx_model, _, _ = export_onnx_model_from_pt(
                model_name=model_name,
                opset_version=opset_version,
                use_external_data_format=use_external_data_format,
                model_type=model_type,
                model_class=model_class,
                config_modifier=config_modifier,
                cache_dir=default_cache_path,
                onnx_dir=str(onnx_dir),
                input_names=input_names[:inputs_count],
                use_gpu=False,
                precision=Precision.FLOAT32,
                optimizer_info=OptimizerInfo.BYSCRIPT,
                validate_onnx=True,
                use_raw_attention_mask=True,
                overwrite=True,
                model_fusion_statistics=model_fusion_statistics,
                fusion_options=fusion_options,
            )

        if validate_model:
            self.assertEqual(is_valid_onnx_model, True)

        expected_node_count = {
            "EmbedLayerNormalization": expected_fusion_result_list[0],
            "Attention": expected_fusion_result_list[1],
            "Gelu": expected_fusion_result_list[2],
            "FastGelu": expected_fusion_result_list[3],
            "BiasGelu": expected_fusion_result_list[4],
            "LayerNormalization": expected_fusion_result_list[5],
            "SkipLayerNormalization": expected_fusion_result_list[6],
        }

        node_count = None
        for value in model_fusion_statistics.values():
            node_count = value
        self.assertIsNotNone(node_count)

        actual_node_count = {}
        for op_type in expected_node_count:
            actual_node_count[op_type] = node_count.get(op_type, 0)

        expected = ", ".join(f"{key}: {value}" for key, value in sorted(expected_node_count.items()))
        actual = ", ".join(f"{key}: {value}" for key, value in sorted(actual_node_count.items()))
        self.assertEqual(expected, actual)

        suffix = "_fp32_cpu.onnx"
        assert optimized_model_path.endswith(suffix)
        baseline_model_path = optimized_model_path[: -len(suffix)] + ".onnx"
        for batch_size in [1, 2]:
            for sequence_length in [1, 8]:
                max_abs_diff, case_passed = bert_parity_test(
                    baseline_model_path,
                    optimized_model_path,
                    output_dir=None,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    use_gpu=False,
                    test_cases=1,
                    seed=123,
                    verbose=False,
                    rtol=1e-4,
                    atol=1e-4,
                    input_ids_name=input_names[0],
                    segment_ids_name=input_names[2] if inputs_count > 2 else None,
                    input_mask_name=input_names[1] if inputs_count > 1 else None,
                    mask_type=2,
                    dictionary_size=1024,
                )
                self.assertTrue(
                    case_passed, f"bert parity test failed: {batch_size=} {sequence_length=} {max_abs_diff=}"
                )

    def test_bert(self):
        model_name = "hf-internal-testing/tiny-random-bert"
        self.run_optimizer_on_model(model_name, [1, 5, 0, 0, 5, 0, 10], inputs_count=1)
        self.run_optimizer_on_model(model_name, [1, 5, 0, 0, 5, 0, 10], inputs_count=2)
        self.run_optimizer_on_model(model_name, [1, 5, 0, 0, 5, 0, 10], inputs_count=3)

    def test_roberta(self):
        model_name = "hf-internal-testing/tiny-random-roberta"
        # TODO: EmbedLayerNormalization fusion.
        self.run_optimizer_on_model(model_name, [0, 5, 0, 0, 5, 1, 10], inputs_count=1)
        self.run_optimizer_on_model(model_name, [0, 5, 0, 0, 5, 1, 10], inputs_count=2)

    def test_distillbert(self):
        model_name = "hf-internal-testing/tiny-random-distilbert"
        self.run_optimizer_on_model(model_name, [1, 5, 0, 0, 5, 0, 10], inputs_count=1)
        self.run_optimizer_on_model(model_name, [1, 5, 0, 0, 5, 0, 10], inputs_count=2)

    def test_xlm_roberta(self):
        model_name = "hf-internal-testing/tiny-xlm-roberta"
        # TODO: EmbedLayerNormalization fusion.
        self.run_optimizer_on_model(model_name, [0, 2, 0, 0, 2, 1, 4], inputs_count=1)
        self.run_optimizer_on_model(model_name, [0, 2, 0, 0, 2, 1, 4], inputs_count=2)


if __name__ == "__main__":
    unittest.main()
