# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import unittest
import onnx.backend.test

import onnxruntime.backend as c2

pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(c2, __name__)


# Type not supported
backend_test.exclude(r'(FLOAT16)')

backend_test.exclude(r'(test_acosh_cpu.*'
'|test_acosh_example_cpu.*'
'|test_asinh_cpu.*'
'|test_asinh_example_cpu.*'
'|test_atanh_cpu.*'
'|test_atanh_example_cpu.*'
'|test_convtranspose_1d_cpu.*'
'|test_convtranspose_3d_cpu.*'
'|test_cosh_cpu.*'
'|test_cosh_example_cpu.*'
'|test_dynamic_slice_cpu.*'
'|test_dynamic_slice_default_axes_cpu.*'
'|test_dynamic_slice_end_out_of_bounds_cpu.*'
'|test_dynamic_slice_neg_cpu.*'
'|test_dynamic_slice_start_out_of_bounds_cpu.*'
'|test_eyelike_populate_off_main_diagonal_cpu.*'
'|test_eyelike_with_dtype_cpu.*'
'|test_eyelike_without_dtype_cpu.*'
'|test_gru_seq_length_cpu.*'
'|test_maxunpool_export_with_output_shape_cpu.*'
'|test_maxunpool_export_without_output_shape_cpu.*'
'|test_onehot_with_axis_cpu.*'
'|test_onehot_without_axis_cpu.*'
'|test_scan_sum_cpu.*'
'|test_scatter_with_axis_cpu.*'
'|test_scatter_without_axis_cpu.*'
'|test_sign_cpu.*'
'|test_sinh_cpu.*'
'|test_sinh_example_cpu.*'
'|test_AvgPool1d_cpu.*'
'|test_AvgPool1d_stride_cpu.*'
'|test_AvgPool2d_cpu.*'
'|test_AvgPool2d_stride_cpu.*'
'|test_AvgPool3d_cpu.*'
'|test_AvgPool3d_stride1_pad0_gpu_input_cpu.*'
'|test_AvgPool3d_stride_cpu.*'
'|test_BatchNorm1d_3d_input_eval_cpu.*'
'|test_BatchNorm2d_eval_cpu.*'
'|test_BatchNorm2d_momentum_eval_cpu.*'
'|test_BatchNorm3d_eval_cpu.*'
'|test_BatchNorm3d_momentum_eval_cpu.*'
'|test_GLU_cpu.*'
'|test_GLU_dim_cpu.*'
'|test_Linear_cpu.*'
'|test_PReLU_1d_cpu.*'
'|test_PReLU_1d_multiparam_cpu.*'
'|test_PReLU_2d_cpu.*'
'|test_PReLU_2d_multiparam_cpu.*'
'|test_PReLU_3d_cpu.*'
'|test_PReLU_3d_multiparam_cpu.*'
'|test_PoissonNLLLLoss_no_reduce_cpu.*'
'|test_Softsign_cpu.*'
'|test_operator_add_broadcast_cpu.*'
'|test_operator_add_size1_broadcast_cpu.*'
'|test_operator_add_size1_right_broadcast_cpu.*'
'|test_operator_add_size1_singleton_broadcast_cpu.*'
'|test_operator_addconstant_cpu.*'
'|test_operator_addmm_cpu.*'
'|test_operator_basic_cpu.*'
'|test_operator_lstm_cpu.*'
'|test_operator_mm_cpu.*'
'|test_operator_non_float_params_cpu.*'
'|test_operator_params_cpu.*'
'|test_operator_pow_cpu.*'
'|test_operator_rnn_cpu.*'
'|test_operator_rnn_single_layer_cpu.*'
'|test_sign_model_cpu.*'
'|test_mvn.*' #No schema registered for 'MeanVarianceNormalization'!
')')

# import all test cases at global scope to make
# them visible to python.unittest.
globals().update(backend_test.enable_report().test_cases)


if __name__ == '__main__':
    unittest.main()
