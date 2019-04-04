# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import unittest
import onnx.backend.test

import numpy as np
import onnxruntime.backend as c2
import platform

pytest_plugins = 'onnx.backend.test.report',

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

backend_test = OrtBackendTest(c2, __name__)

# Type not supported
backend_test.exclude(r'(FLOAT16)')

backend_test.exclude(r'('
'^test_cast_DOUBLE_to_FLOAT_cpu.*'
'|^test_cast_FLOAT_to_DOUBLE_cpu.*'
'|^test_cast_FLOAT_to_STRING_cpu.*'
'|^test_cast_STRING_to_FLOAT_cpu.*'
'|^test_convtranspose_1d_cpu.*'
'|^test_convtranspose_3d_cpu.*'
'|^test_constantofshape_*.*'

'|^test_AvgPool1d_cpu.*'
'|^test_AvgPool1d_stride_cpu.*'
'|^test_AvgPool2d_cpu.*'
'|^test_AvgPool2d_stride_cpu.*'
'|^test_AvgPool3d_cpu.*'
'|^test_AvgPool3d_stride1_pad0_gpu_input_cpu.*'
'|^test_AvgPool3d_stride_cpu.*'
'|^test_BatchNorm1d_3d_input_eval_cpu.*'
'|^test_BatchNorm2d_eval_cpu.*'
'|^test_BatchNorm2d_momentum_eval_cpu.*'
'|^test_BatchNorm3d_eval_cpu.*'
'|^test_BatchNorm3d_momentum_eval_cpu.*'
'|^test_GLU_cpu.*'
'|^test_GLU_dim_cpu.*'
'|^test_Linear_cpu.*'
'|^test_PReLU_1d_cpu.*'
'|^test_PReLU_1d_multiparam_cpu.*'
'|^test_PReLU_2d_cpu.*'
'|^test_PReLU_2d_multiparam_cpu.*'
'|^test_PReLU_3d_cpu.*'
'|^test_PReLU_3d_multiparam_cpu.*'
'|^test_PoissonNLLLLoss_no_reduce_cpu.*'
'|^test_Softsign_cpu.*'
'|^test_operator_add_broadcast_cpu.*'
'|^test_operator_add_size1_broadcast_cpu.*'
'|^test_operator_add_size1_right_broadcast_cpu.*'
'|^test_operator_add_size1_singleton_broadcast_cpu.*'
'|^test_operator_addconstant_cpu.*'
'|^test_operator_addmm_cpu.*'
'|^test_operator_basic_cpu.*'
'|^test_operator_mm_cpu.*'
'|^test_operator_non_float_params_cpu.*'
'|^test_operator_params_cpu.*'
'|^test_operator_pow_cpu.*'
'|^test_shrink_cpu.*'
'|^test_vgg19_cpu.*'
'|^test_zfnet512_cpu.*'
'|^test_gru_seq_length_cpu.*'
'|^test_slice_cpu.*'
'|^test_slice_default_axes_cpu.*'
'|^test_slice_default_steps_cpu.*'
'|^test_slice_end_out_of_bounds_cpu.*'
'|^test_slice_neg_cpu.*'
'|^test_slice_neg_steps_cpu.*'
'|^test_slice_start_out_of_bounds_cpu.*'
'|^test_averagepool_2d_ceil_cpu.*'
'|^test_maxpool_2d_ceil_cpu.*'
'|^test_maxpool_2d_dilations_cpu.*'
'|^test_mvn_cpu.*'
'|^test_resize_downsample_linear_cpu.*'
'|^test_resize_downsample_nearest_cpu.*'
'|^test_resize_nearest_cpu.*'
'|^test_resize_upsample_linear_cpu.*'
'|^test_resize_upsample_nearest_cpu.*'
')')

# import all test cases at global scope to make
# them visible to python.unittest.
globals().update(backend_test.enable_report().test_cases)


if __name__ == '__main__':
    unittest.main()
