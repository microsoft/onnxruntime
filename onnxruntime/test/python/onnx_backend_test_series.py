# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import sys
import os
import platform
import unittest
import onnx
import onnx.backend.test

import numpy as np
import onnxruntime.backend as c2

pytest_plugins = 'onnx.backend.test.report',


class OrtBackendTest(onnx.backend.test.BackendTest):
    def __init__(self, backend, parent_module=None):
        super(OrtBackendTest, self).__init__(backend, parent_module)

    @classmethod
    def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol):
        def assert_similar_array(ref_output, output):
            np.testing.assert_equal(ref_output.dtype, output.dtype)
            if ref_output.dtype == np.object:
                np.testing.assert_array_equal(ref_output, output)
            else:
                np.testing.assert_allclose(ref_output, output, rtol=1e-3, atol=1e-5)            
        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            if isinstance(outputs[i], list):
                for j in range(len(outputs[i])):
                    assert_similar_array(ref_outputs[i][j], outputs[i][j])
            else:
                assert_similar_array(ref_outputs[i], outputs[i])


def create_backend_test(testname=None):
    backend_test = OrtBackendTest(c2, __name__)

    # Type not supported
    backend_test.exclude(r'(FLOAT16)')

    if testname:
        backend_test.include(testname + '.*')
    else:
        # read filters data
        with open(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testdata',
                             'onnx_backend_test_series_filters.jsonc')) as f:
            filters_lines = f.readlines()
        filters_lines = [x.split('//')[0] for x in filters_lines]
        filters = json.loads('\n'.join(filters_lines))

        current_failing_tests = filters['current_failing_tests']

        if platform.architecture()[0] == '32bit':
            current_failing_tests += filters['current_failing_tests_x86']

        if c2.supports_device('DNNL'):
            current_failing_tests += filters['current_failing_tests_DNNL']

        if c2.supports_device('NNAPI'):
            current_failing_tests += filters['current_failing_tests_NNAPI']

        if c2.supports_device('OPENVINO_GPU_FP32') or c2.supports_device('OPENVINO_GPU_FP16'):
            current_failing_tests += filters['current_failing_tests_OPENVINO_GPU']

        if c2.supports_device('OPENVINO_MYRIAD'):
            current_failing_tests += filters['current_failing_tests_OPENVINO_GPU']
            current_failing_tests += filters['current_failing_tests_OPENVINO_MYRIAD']

        if c2.supports_device('OPENVINO_CPU_FP32'):
            current_failing_tests += filters['current_failing_tests_OPENVINO_CPU_FP32']

        if c2.supports_device('MIGRAPHX'):
            current_failing_tests += [
                '^test_constant_pad_cpu', '^test_softmax_axis_1_cpu', '^test_softmax_axis_0_cpu',
                '^test_softmax_default_axis_cpu', '^test_round_cpu', '^test_lrn_default_cpu', '^test_lrn_cpu',
                '^test_logsoftmax_axis_0_cpu', '^test_logsoftmax_axis_1_cpu', '^test_logsoftmax_default_axis_cpu',
                '^test_dynamicquantizelinear_expanded_cpu', '^test_dynamicquantizelinear_max_adjusted_cpu',
                '^test_dynamicquantizelinear_max_adjusted_expanded_cpu', '^test_dynamicquantizelinear_min_adjusted_cpu',
                '^test_dynamicquantizelinear_min_adjusted_expanded_cpu',
                '^test_range_float_type_positive_delta_expanded_cpu',
                '^test_range_int32_type_negative_delta_expanded_cpu',
                '^test_operator_symbolic_override_nested_cpu',
                '^test_negative_log_likelihood_loss',
                '^test_softmax_cross_entropy',
                '^test_greater_equal', '^test_less_equal'
            ]

        # Skip these tests for a "pure" DML onnxruntime python wheel. We keep these tests enabled for instances where both DML and CUDA
        # EPs are available (Windows GPU CI pipeline has this config) - these test will pass because CUDA has higher precendence than DML
        # and the nodes are assigned to only the CUDA EP (which supports these tests)
        if c2.supports_device('DML') and not c2.supports_device('GPU'):
            current_failing_tests += [
                '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_cpu',
                '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded_cpu',
                '^test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_cpu',
                '^test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded_cpu',
                '^test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_cpu',
                '^test_softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded_cpu',
                '^test_asin_example_cpu',
                '^test_dynamicquantizelinear_expanded_cpu',
                '^test_resize_downsample_scales_linear_cpu',
                '^test_resize_downsample_sizes_linear_pytorch_half_pixel_cpu',
                '^test_resize_downsample_sizes_nearest_cpu',
                '^test_resize_upsample_sizes_nearest_cpu',
                '^test_roialign_cpu'
            ]

        filters = current_failing_tests + \
            filters['tests_with_pre_opset7_dependencies'] + \
            filters['unsupported_usages'] + \
            filters['failing_permanently'] + \
            filters['test_with_types_disabled_due_to_binary_size_concerns']

        backend_test.exclude('(' + '|'.join(filters) + ')')
        print('excluded tests:', filters)

    # import all test cases at global scope to make
    # them visible to python.unittest.
    globals().update(backend_test.enable_report().test_cases)

    return backend_test


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='Run the ONNX backend tests using ONNXRuntime.')

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than inidividual
    # test case names.
    parser.add_argument(
        '-t',
        '--test-name',
        dest='testname',
        type=str,
        help="Only run tests that match this value. Matching is regex based, and '.*' is automatically appended")

    # parse just our args. python unittest has its own args and arg parsing, and that runs inside unittest.main()
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    return args


if __name__ == '__main__':
    args = parse_args()

    backend_test = create_backend_test(args.testname)
    unittest.main()
