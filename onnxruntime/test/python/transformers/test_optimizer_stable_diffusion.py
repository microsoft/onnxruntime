# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import shutil
import unittest

import numpy as np
import pytest
from parity_utilities import find_transformers_source

if find_transformers_source():
    from compare_bert_results import run_test
    from fusion_options import FusionOptions
    from optimizer import optimize_model
else:
    from onnxruntime.transformers.compare_bert_results import run_test
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.optimizer import optimize_model

if find_transformers_source(["models", "stable_diffusion"]):
    from optimize_pipeline import main as optimize_stable_diffusion
else:
    from onnxruntime.transformers.models.stable_diffusion.optimize_pipeline import main as optimize_stable_diffusion


TINY_MODELS = {
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
}


class TestStableDiffusionOptimization(unittest.TestCase):
    def verify_node_count(self, onnx_model, expected_node_count, test_name):
        for op_type, count in expected_node_count.items():
            if len(onnx_model.get_nodes_by_op_type(op_type)) != count:
                print(f"Counters is not expected in test: {test_name}")
                for op, counter in expected_node_count.items():
                    print(f"{op}: {len(onnx_model.get_nodes_by_op_type(op))} expected={counter}")

                self.assertEqual(len(onnx_model.get_nodes_by_op_type(op_type)), count)

    def verify_clip_optimizer(self, clip_onnx_path, optimized_clip_onnx_path, expected_counters, float16=False):
        fusion_options = FusionOptions("clip")
        m = optimize_model(
            clip_onnx_path,
            model_type="clip",
            num_heads=0,
            hidden_size=0,
            opt_level=0,
            optimization_options=fusion_options,
            use_gpu=True,
        )
        self.verify_node_count(m, expected_counters, "test_clip")

        if float16:
            m.convert_float_to_float16(
                keep_io_types=True,
            )
        print(m.get_operator_statistics())
        m.save_model_to_file(optimized_clip_onnx_path)

        threshold = 1e-2 if float16 else 3e-3
        max_abs_diff, passed = run_test(
            clip_onnx_path,
            optimized_clip_onnx_path,
            output_dir=None,
            batch_size=1,
            sequence_length=77,
            use_gpu=True,
            test_cases=10,
            seed=1,
            verbose=False,
            rtol=1e-1,
            atol=threshold,
            input_ids_name="input_ids",
            segment_ids_name=None,
            input_mask_name=None,
            mask_type=0,
        )

        self.assertLess(max_abs_diff, threshold)
        self.assertTrue(passed)

    @pytest.mark.slow
    def test_clip_sd(self):
        save_directory = "tiny-random-stable-diffusion"
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory, ignore_errors=True)

        model_type = "stable-diffusion"
        model_name = TINY_MODELS[model_type]

        from optimum.onnxruntime import ORTStableDiffusionPipeline

        base = ORTStableDiffusionPipeline.from_pretrained(model_name, export=True)
        base.save_pretrained(save_directory)

        clip_onnx_path = os.path.join(save_directory, "text_encoder", "model.onnx")
        optimized_clip_onnx_path = os.path.join(save_directory, "text_encoder", "opt.onnx")
        self.verify_clip_optimizer(
            clip_onnx_path,
            optimized_clip_onnx_path,
            expected_counters={
                "EmbedLayerNormalization": 0,
                "Attention": 5,
                "SkipLayerNormalization": 10,
                "LayerNormalization": 1,
                "Gelu": 0,
                "BiasGelu": 0,
            },
            float16=True,
        )

    @pytest.mark.slow
    def test_clip_sdxl(self):
        save_directory = "tiny-random-stable-diffusion-xl"
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory, ignore_errors=True)

        model_type = "stable-diffusion-xl"
        model_name = TINY_MODELS[model_type]

        from optimum.onnxruntime import ORTStableDiffusionXLPipeline

        base = ORTStableDiffusionXLPipeline.from_pretrained(model_name, export=True)
        base.save_pretrained(save_directory)

        clip_onnx_path = os.path.join(save_directory, "text_encoder", "model.onnx")
        optimized_clip_onnx_path = os.path.join(save_directory, "text_encoder", "opt.onnx")
        self.verify_clip_optimizer(
            clip_onnx_path,
            optimized_clip_onnx_path,
            expected_counters={
                "EmbedLayerNormalization": 0,
                "Attention": 5,
                "SkipLayerNormalization": 10,
                "LayerNormalization": 1,
                "Gelu": 0,
                "BiasGelu": 5,
            },
        )

        clip_onnx_path = os.path.join(save_directory, "text_encoder_2", "model.onnx")
        optimized_clip_onnx_path = os.path.join(save_directory, "text_encoder_2", "opt.onnx")
        self.verify_clip_optimizer(
            clip_onnx_path,
            optimized_clip_onnx_path,
            expected_counters={
                "EmbedLayerNormalization": 0,
                "Attention": 5,
                "SkipLayerNormalization": 10,
                "LayerNormalization": 1,
                "Gelu": 0,
                "BiasGelu": 5,
            },
        )

    @pytest.mark.slow
    def test_optimize_sdxl_fp32(self):
        save_directory = "tiny-random-stable-diffusion-xl"
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory, ignore_errors=True)

        model_type = "stable-diffusion-xl"
        model_name = TINY_MODELS[model_type]

        from optimum.onnxruntime import ORTStableDiffusionXLPipeline

        baseline = ORTStableDiffusionXLPipeline.from_pretrained(model_name, export=True)
        if not os.path.exists(save_directory):
            baseline.save_pretrained(save_directory)

        batch_size, num_images_per_prompt, height, width = 2, 2, 64, 64
        latents = baseline.prepare_latents(
            batch_size * num_images_per_prompt,
            baseline.unet.config["in_channels"],
            height,
            width,
            dtype=np.float32,
            generator=np.random.RandomState(0),
        )

        optimized_directory = "tiny-random-stable-diffusion-xl-optimized"
        argv = [
            "--input",
            save_directory,
            "--output",
            optimized_directory,
            "--disable_group_norm",
            "--disable_bias_splitgelu",
            "--overwrite",
        ]
        optimize_stable_diffusion(argv)

        treatment = ORTStableDiffusionXLPipeline.from_pretrained(optimized_directory, provider="CUDAExecutionProvider")
        inputs = {
            "prompt": ["starry night by van gogh"] * batch_size,
            "num_inference_steps": 3,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
            "output_type": "np",
        }

        ort_outputs_1 = baseline(latents=latents, **inputs)
        ort_outputs_2 = treatment(latents=latents, **inputs)
        self.assertTrue(np.allclose(ort_outputs_1.images[0], ort_outputs_2.images[0], atol=1e-3))

    @pytest.mark.slow
    def test_optimize_sdxl_fp16(self):
        """This tests optimized fp16 pipeline, and result is deterministic for a given seed"""
        save_directory = "tiny-random-stable-diffusion-xl"
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory, ignore_errors=True)

        model_type = "stable-diffusion-xl"
        model_name = TINY_MODELS[model_type]

        from optimum.onnxruntime import ORTStableDiffusionXLPipeline

        baseline = ORTStableDiffusionXLPipeline.from_pretrained(model_name, export=True)
        if not os.path.exists(save_directory):
            baseline.save_pretrained(save_directory)

        optimized_directory = "tiny-random-stable-diffusion-xl-optimized-fp16"
        argv = [
            "--input",
            save_directory,
            "--output",
            optimized_directory,
            "--disable_group_norm",
            "--disable_bias_splitgelu",
            "--float16",
            "--overwrite",
        ]
        optimize_stable_diffusion(argv)

        fp16_pipeline = ORTStableDiffusionXLPipeline.from_pretrained(
            optimized_directory, provider="CUDAExecutionProvider"
        )
        batch_size, num_images_per_prompt, height, width = 1, 1, 64, 64
        inputs = {
            "prompt": ["starry night by van gogh"] * batch_size,
            "num_inference_steps": 3,
            "num_images_per_prompt": num_images_per_prompt,
            "height": height,
            "width": width,
            "guidance_rescale": 0.1,
            "output_type": "latent",
        }

        seed = 123
        np.random.seed(seed)
        ort_outputs_1 = fp16_pipeline(**inputs)

        np.random.seed(seed)
        ort_outputs_2 = fp16_pipeline(**inputs)

        np.random.seed(seed)
        ort_outputs_3 = fp16_pipeline(**inputs)

        self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_2.images[0]))
        self.assertTrue(np.array_equal(ort_outputs_1.images[0], ort_outputs_3.images[0]))


if __name__ == "__main__":
    unittest.main()
