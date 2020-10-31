#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# For live logging, use the command: pytest -o log_cli=true --log-cli-level=DEBUG

import unittest
import os
import onnx
import onnxruntime
import pytest

BERT_TEST_MODELS = {
    "bert_keras_squad": ('bert_squad_tensorflow2.1_keras2onnx_opset11', 'TFBertForQuestionAnswering.onnx'),
    "gpt2_past_mask": ('FUSION', 'gpt2_past_mask_one_layer.onnx'),
}


def _get_test_model_path(name):
    sub_dir, file = BERT_TEST_MODELS[name]
    if sub_dir == "FUSION":
        return os.path.join('..', '..', '..', 'test', 'testdata', 'transform', 'fusion', file)
    else:
        return os.path.join('test_data', sub_dir, file)


class TestBertProfiler(unittest.TestCase):
    def run_profile(self, arguments: str):
        from bert_profiler import parse_arguments, run
        args = parse_arguments(arguments.split())
        results = run(args)
        self.assertTrue(len(results) > 1)

    def test_profiler_gpu(self):
        input_model_path = _get_test_model_path('bert_keras_squad')
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            self.run_profile(f'--model {input_model_path} --batch_size 1 --sequence_length 7 --use_gpu')

    def test_profiler_cpu(self):
        input_model_path = _get_test_model_path('bert_keras_squad')
        self.run_profile(f'--model {input_model_path} --batch_size 1 --sequence_length 7 --use_dummy_inputs')


if __name__ == '__main__':
    unittest.main()
