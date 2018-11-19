# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import unittest
import onnx.backend.test

import onnxruntime.backend as c2

pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(c2, __name__)

# Current status
# Ran 930 tests in 0.948s OK (skipped=574)

# We should investigate.

# dimension issue
# Error message is: Input X must be 4-dimensional. X: {1,1,3}
# ONNX Runtime expects a 4-dimension vector for operator ConvTranspose.
# file onnxruntime/core/providers/cpu/nn/conv_transpose.cc
backend_test.exclude(r'(convtranspose)')

# Type not supported
backend_test.exclude(r'(FLOAT16)')

# Operator not supported
backend_test.exclude(r'(test_expand)')
backend_test.exclude(r'(maxpool)')
backend_test.exclude(r'(AvgPool)')
backend_test.exclude(r'(max_one_input)|(max_example_cpu)|(max_two_inputs)')
backend_test.exclude(r'(min_one_input)|(min_example_cpu)|(min_two_inputs)')
backend_test.exclude(r'(mean_example)|(mean_one_input)|(mean_two_inputs)')
backend_test.exclude(r'(sum_example)|(sum_one_input)|(sum_two_inputs)')
backend_test.exclude(r'(BatchNorm)')
backend_test.exclude(r'(GLU)|(PReLU)|(PoissonNLLLLoss)|(Softsign)')
backend_test.exclude(r'(Linear_cpu)')
backend_test.exclude(r'(broadcast)')
backend_test.exclude(r'(addconstant)|(addmm)|(basic)|(lstm)')
backend_test.exclude(r'(_mm)|(non_float)|(test_operator_params_cpu)')
backend_test.exclude(r'(pow_cpu)|(rnn_cpu)|(rnn_single)|(_gru_)')
backend_test.exclude("(precomputed_((pads)|(strides)))")

# Exclude deep learning
backend_test.exclude(r'(test_vgg19|test_vgg)')  # Too long.
backend_test.exclude(r'(alexnet)')  # Too long.
backend_test.exclude(r'(densenet)')  # Too long.
backend_test.exclude(r'(inception)')  # Too long.
backend_test.exclude(r'(resnet)')  # Too long.
backend_test.exclude(r'(shufflenet)')  # Too long.
backend_test.exclude(r'(squeezenet)')  # Too long.
backend_test.exclude(r'(zfnet)')  # Too long.

# Exclude experimental
backend_test.exclude(r'(test_mvn_cpu)')

# import all test cases at global scope to make
# them visible to python.unittest.
globals().update(backend_test.enable_report().test_cases)


if __name__ == '__main__':
    unittest.main()
