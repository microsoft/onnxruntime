# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import os
import platform
import unittest
import onnx
import onnx.backend.test

import numpy as np
import onnxruntime.backend as c2

pytest_plugins = 'onnx.backend.test.report',

def GetVersionTag():
    version2tag = {}
    file_path = '/data/onnx/version2tag'
    if os.path.isfile(file_path):
       with open(file_path, 'r') as f:
           for line in f.readlines():
               fields = line.strip().split(':')
               version2tag[fields[0]] = fields[1]
    print ("version2tag map", version2tag)
    if onnx.version.git_version in version2tag:
        return version2tag[onnx.version.git_version]
    else: return "unknown"
        
version_tag = GetVersionTag()
print ("onnx version:", onnx.__version__)
print ("git version:", onnx.version.git_version)
print ("VERSION TAG:", version_tag)

class OrtBackendTest(onnx.backend.test.BackendTest):

    def __init__(self, backend, parent_module=None):
        super(OrtBackendTest, self).__init__(backend, parent_module)

    @classmethod
    def assert_similar_outputs(cls, ref_outputs, outputs, rtol, atol):
        np.testing.assert_equal(len(ref_outputs), len(outputs))
        for i in range(len(outputs)):
            np.testing.assert_equal(ref_outputs[i].dtype, outputs[i].dtype)
            if ref_outputs[i].dtype == np.object:
                np.testing.assert_array_equal(ref_outputs[i], outputs[i])
            else:
                np.testing.assert_allclose(
                    ref_outputs[i],
                    outputs[i],
                    rtol=1e-3,
                    atol=1e-5)


# ORT first supported opset 7, so models with nodes that require versions prior to opset 7 are not supported
def tests_with_pre_opset7_dependencies_filters():
    filters = ('^test_AvgPool1d_cpu.*',
               '^test_AvgPool1d_stride_cpu.*',
               '^test_AvgPool2d_cpu.*',
               '^test_AvgPool2d_stride_cpu.*',
               '^test_AvgPool3d_cpu.*',
               '^test_AvgPool3d_stride1_pad0_gpu_input_cpu.*',
               '^test_AvgPool3d_stride_cpu.*',
               '^test_BatchNorm1d_3d_input_eval_cpu.*',
               '^test_BatchNorm2d_eval_cpu.*',
               '^test_BatchNorm2d_momentum_eval_cpu.*',
               '^test_BatchNorm3d_eval_cpu.*',
               '^test_BatchNorm3d_momentum_eval_cpu.*',
               '^test_GLU_cpu.*',
               '^test_GLU_dim_cpu.*',
               '^test_Linear_cpu.*',
               '^test_PReLU_1d_cpu.*',
               '^test_PReLU_1d_multiparam_cpu.*',
               '^test_PReLU_2d_cpu.*',
               '^test_PReLU_2d_multiparam_cpu.*',
               '^test_PReLU_3d_cpu.*',
               '^test_PReLU_3d_multiparam_cpu.*',
               '^test_PoissonNLLLLoss_no_reduce_cpu.*',
               '^test_Softsign_cpu.*',
               '^test_operator_add_broadcast_cpu.*',
               '^test_operator_add_size1_broadcast_cpu.*',
               '^test_operator_add_size1_right_broadcast_cpu.*',
               '^test_operator_add_size1_singleton_broadcast_cpu.*',
               '^test_operator_addconstant_cpu.*',
               '^test_operator_addmm_cpu.*',
               '^test_operator_basic_cpu.*',
               '^test_operator_mm_cpu.*',
               '^test_operator_non_float_params_cpu.*',
               '^test_operator_params_cpu.*',
               '^test_operator_pow_cpu.*')

    return filters


def unsupported_usages_filters():
    filters = ('^test_convtranspose_1d_cpu.*',  # ConvTransponse supports 4-D only
               '^test_convtranspose_3d_cpu.*')

    return filters


def create_backend_test(testname=None):
    backend_test = OrtBackendTest(c2, __name__)

    # Type not supported
    backend_test.exclude(r'(FLOAT16)')

    if testname:
        backend_test.include(testname + '.*')
    else:
        # Tests that are failing temporarily and should be fixed
        current_failing_tests = ('^test_cast_STRING_to_FLOAT_cpu.*',
                                 '^test_cast_FLOAT_to_STRING_cpu.*',
                                 '^test_dequantizelinear_cpu.*',
                                 '^test_qlinearconv_cpu.*',
                                 '^test_quantizelinear_cpu.*',
                                 '^test_gru_seq_length_cpu.*')
        global version_tag
        if version_tag == 'onnx141' or onnx.__version__ == '1.4.1':
            current_failing_tests = current_failing_tests + ('^test_shrink_cpu.*', '^test_constantofshape_*.*',)
        if version_tag == 'onnx150' or onnx.__version__ == '1.5.0':
            current_failing_tests = current_failing_tests + ('^test_constantofshape_*.*',)

        # Failing for nGraph.
        if c2.supports_device('NGRAPH'):
            current_failing_tests = current_failing_tests + ('|^test_operator_repeat_dim_overflow_cpu.*',)

        filters = current_failing_tests + \
                  tests_with_pre_opset7_dependencies_filters() + \
                  unsupported_usages_filters()

        backend_test.exclude('(' + '|'.join(filters) + ')')
        print ('excluded tests:', filters)

    # import all test cases at global scope to make
    # them visible to python.unittest.
    globals().update(backend_test.enable_report().test_cases)

    return backend_test


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='Run the ONNX backend tests using ONNXRuntime.')

    # Add an argument to match a single test name, by adding the name to the 'include' filter.
    # Using -k with python unittest (https://docs.python.org/3/library/unittest.html#command-line-options)
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than inidividual test case names.
    parser.add_argument('-t', '--test-name', dest='testname', type=str,
                        help="Only run tests that match this value. Matching is regex based, and '.*' is automatically appended")

    # parse just our args. python unittest has its own args and arg parsing, and that runs inside unittest.main()	
    args, left = parser.parse_known_args()
    sys.argv = sys.argv[:1] + left

    return args


if __name__ == '__main__':
    args = parse_args()

    backend_test = create_backend_test(args.testname)
    unittest.main()
