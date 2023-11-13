# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy as np
import pytest
import torch

from onnxruntime import InferenceSession, SessionOptions, get_available_providers


class TestTimestampProcessor(unittest.TestCase):
    def generate_model(self, arguments: str):
        from onnxruntime.transformers.models.whisper.convert_to_onnx import main as whisper_to_onnx

        whisper_to_onnx(arguments.split())

    def generate_dataset(self):
        from datasets import load_dataset
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        input_features = inputs.input_features
        return [input_features, processor]

    def run_timestamp(self, provider: str):
        self.generate_model("-m openai/whisper-tiny --optimize_onnx --precision fp32 -l -e")
        [input_features, processor] = self.generate_dataset()
        model_path = "./onnx_models/whisper-tiny_beamsearch.onnx"
        sess_options = SessionOptions()
        sess_options.log_severity_level = 4
        sess = InferenceSession(model_path, sess_options, providers=[provider])
        input_data = input_features.repeat(1, 1, 1)
        ort_inputs = {
            "input_features": np.float32(input_data.cpu().numpy()),
            "max_length": np.array([128], dtype=np.int32),
            "min_length": np.array([0], dtype=np.int32),
            "num_beams": np.array([1], dtype=np.int32),
            "num_return_sequences": np.array([1], dtype=np.int32),
            "length_penalty": np.array([1.0], dtype=np.float32),
            "repetition_penalty": np.array([1.0], dtype=np.float32),
            "logits_processor": np.array([1], dtype=np.int32),
        }
        ort_out = sess.run(None, ort_inputs)
        ort_out_tensor = torch.from_numpy(ort_out[0])
        ort_transcription = processor.batch_decode(
            ort_out_tensor[0][0].view(1, -1), skip_special_tokens=True, output_offsets=True
        )
        print(ort_transcription)
        expected_transcription = [
            {
                "text": "<|0.00|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|5.44|>",
                "offsets": [
                    {
                        "text": "<|0.00|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|5.44|>",
                        "timestamp": (0.0, 5.44),
                    }
                ],
            }
        ]
        self.assertEqual(ort_transcription, expected_transcription)

    @pytest.mark.slow
    def test_timestamp_cpu(self):
        provider = "CPUExecutionProvider"
        self.run_timestamp(provider)

    @pytest.mark.slow
    def test_timestamp_cuda(self):
        cuda_provider = "CUDAExecutionProvider"
        if cuda_provider in get_available_providers():
            self.run_timestamp(cuda_provider)


if __name__ == "__main__":
    unittest.main()
