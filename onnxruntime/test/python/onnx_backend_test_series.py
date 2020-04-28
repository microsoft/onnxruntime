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
                np.testing.assert_allclose(ref_outputs[i], outputs[i], rtol=1e-3, atol=1e-5)


# ORT first supported opset 7, so models with nodes that require versions prior to opset 7 are not supported
def tests_with_pre_opset7_dependencies_filters():
    filters = [
        '^test_AvgPool1d_cpu', '^test_AvgPool1d_stride_cpu', '^test_AvgPool2d_cpu', '^test_AvgPool2d_stride_cpu',
        '^test_AvgPool3d_cpu', '^test_AvgPool3d_stride1_pad0_gpu_input_cpu', '^test_AvgPool3d_stride_cpu',
        '^test_BatchNorm1d_3d_input_eval_cpu', '^test_BatchNorm2d_eval_cpu', '^test_BatchNorm2d_momentum_eval_cpu',
        '^test_BatchNorm3d_eval_cpu', '^test_BatchNorm3d_momentum_eval_cpu', '^test_GLU_cpu', '^test_GLU_dim_cpu',
        '^test_Linear_cpu', '^test_PReLU_1d_cpu', '^test_PReLU_1d_multiparam_cpu', '^test_PReLU_2d_cpu',
        '^test_PReLU_2d_multiparam_cpu', '^test_PReLU_3d_cpu', '^test_PReLU_3d_multiparam_cpu',
        '^test_PoissonNLLLLoss_no_reduce_cpu', '^test_Softsign_cpu', '^test_operator_add_broadcast_cpu',
        '^test_operator_add_size1_broadcast_cpu', '^test_operator_add_size1_right_broadcast_cpu',
        '^test_operator_add_size1_singleton_broadcast_cpu', '^test_operator_addconstant_cpu',
        '^test_operator_addmm_cpu', '^test_operator_basic_cpu', '^test_operator_mm_cpu',
        '^test_operator_non_float_params_cpu', '^test_operator_params_cpu', '^test_operator_pow_cpu'
    ]

    return filters


def unsupported_usages_filters():
    filters = [
        '^test_convtranspose_1d_cpu',  # ConvTransponse supports 4-D only
        '^test_convtranspose_3d_cpu'
    ]

    return filters


def other_tests_failing_permanently_filters():
    # Numpy float to string has unexpected rounding for some results given numpy default precision is meant to be 8.
    # e.g. 0.296140194 -> '0.2961402' not '0.29614019'. ORT produces the latter with precision set to 8, which
    # doesn't match the expected output that was generated with numpy.
    filters = ['^test_cast_FLOAT_to_STRING_cpu']

    return filters


def test_with_types_disabled_due_to_binary_size_concerns_filters():
    filters = ['^test_bitshift_right_uint16_cpu', '^test_bitshift_left_uint16_cpu']

    return filters


def create_backend_test(testname=None):
    backend_test = OrtBackendTest(c2, __name__)

    # Type not supported
    backend_test.exclude(r'(FLOAT16)')

    if testname:
        backend_test.include(testname + '.*')
    else:
        # Tests that are failing temporarily and should be fixed
        current_failing_tests = [
            '^test_adagrad_cpu',
            '^test_adagrad_multiple_cpu',
            '^test_batchnorm_epsilon_old_cpu',
            '^test_batchnorm_epsilon_training_mode_cpu',
            '^test_batchnorm_example_old_cpu',
            '^test_batchnorm_example_training_mode_cpu',
            '^test_dropout_default_cpu',
            '^test_dropout_random_cpu',
            '^test_einsum_batch_diagonal_cpu',
            '^test_einsum_batch_matmul_cpu',
            '^test_einsum_inner_prod_cpu',
            '^test_einsum_sum_cpu',
            '^test_einsum_transpose_cpu',
            '^test_gathernd_example_int32_batch_dim1_cpu',
            '^test_inverse_batched_cpu',
            '^test_inverse_cpu',
            '^test_max_int16_cpu',
            '^test_max_int8_cpu',
            '^test_max_uint16_cpu',
            '^test_max_uint8_cpu',
            '^test_mean_square_distance_mean_3d_cpu',
            '^test_mean_square_distance_mean_3d_expanded_cpu',
            '^test_mean_square_distance_mean_4d_cpu',
            '^test_mean_square_distance_mean_4d_expanded_cpu',
            '^test_mean_square_distance_mean_cpu',
            '^test_mean_square_distance_mean_expanded_cpu',
            '^test_mean_square_distance_none_cpu',
            '^test_mean_square_distance_none_expanded_cpu',
            '^test_mean_square_distance_none_weights_cpu',
            '^test_mean_square_distance_none_weights_expanded_cpu',
            '^test_mean_square_distance_sum_cpu',
            '^test_mean_square_distance_sum_expanded_cpu',
            '^test_min_int16_cpu',
            '^test_min_int8_cpu',
            '^test_min_uint16_cpu',
            '^test_min_uint8_cpu',
            '^test_momentum_cpu',
            '^test_momentum_multiple_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NC_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_cpu',
            '^test_negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_cpu',
            '^test_nesterov_momentum_cpu',
            '^test_pow_bcast_array_cpu',
            '^test_pow_bcast_scalar_cpu',
            '^test_pow_cpu',
            '^test_pow_example_cpu',
            '^test_pow_types_float32_int32_cpu',
            '^test_pow_types_float32_int64_cpu',
            '^test_pow_types_float32_uint32_cpu',
            '^test_pow_types_float32_uint64_cpu',
            '^test_pow_types_float_cpu',
            '^test_pow_types_int32_float32_cpu',
            '^test_pow_types_int32_int32_cpu',
            '^test_pow_types_int64_float32_cpu',
            '^test_pow_types_int64_int64_cpu',
            '^test_pow_types_int_cpu',
            '^test_softmax_cross_entropy_mean_3d_cpu',
            '^test_softmax_cross_entropy_mean_3d_expanded_cpu',
            '^test_softmax_cross_entropy_mean_cpu',
            '^test_softmax_cross_entropy_mean_expanded_cpu',
            '^test_softmax_cross_entropy_mean_weight_cpu',
            '^test_softmax_cross_entropy_mean_weight_expanded_cpu',
            '^test_softmax_cross_entropy_mean_weight_ignore_index_cpu',
            '^test_softmax_cross_entropy_mean_weight_ignore_index_expanded_cpu',
            '^test_softmax_cross_entropy_none_cpu',
            '^test_softmax_cross_entropy_none_expanded_cpu',
            '^test_softmax_cross_entropy_none_weights_cpu',
            '^test_softmax_cross_entropy_none_weights_expanded_cpu',
            '^test_softmax_cross_entropy_sum_cpu',
            '^test_softmax_cross_entropy_sum_expanded_cpu',
            '^test_unfoldtodepth_with_padding_cpu',
            '^test_unfoldtodepth_with_padding_stride_cpu',
            '^test_unfoldtodepth_without_padding_cpu',
            '^test_gradient_of_add_and_mul_cpu',
            '^test_gradient_of_add_cpu',
            '^test_batchnorm_example_training_mode_cpu',
            '^test_batchnorm_epsilon_training_mode_cpu',
            '^test_maxunpool_export_with_output_shape_cpu', #result mismatch
            '^test_resize_downsample_scales_cubic_align_corners_cpu',  # results mismatch with onnx tests
            '^test_resize_downsample_scales_linear_align_corners_cpu'  # results mismatch with onnx tests
        ]
        if platform.architecture()[0] == '32bit':
            current_failing_tests += ['^test_vgg19', '^test_zfnet512', '^test_bvlc_alexnet_cpu']
        # Example of how to disable tests for a specific provider.
        # if c2.supports_device('NGRAPH'):
        #    current_failing_tests.append('^test_operator_repeat_dim_overflow_cpu')
        if c2.supports_device('NGRAPH'):
            current_failing_tests += [
                '^test_clip.*', '^test_qlinearconv_cpu', '^test_depthtospace_crd.*', '^test_argmax_negative_axis.*',
                '^test_argmin_negative_axis.*', '^test_hardmax_negative_axis.*', '^test_gemm_default_no_bias_cpu',
                '^test_flatten_negative_axis.*', '^test_reduce_[a-z1-9_]*_negative_axes_.*',
                'test_squeeze_negative_axes_cpu', 'test_unsqueeze_negative_axes_cpu', 'test_constant_pad_cpu',
                'test_edge_pad_cpu', 'test_reflect_pad_cpu', '^test_split_zero_size_splits_.*','^test_argmax_keepdims_example_select_last_index_.*', '^test_argmax_no_keepdims_example_select_last_index_.*','^test_argmin_no_keepdims_example_select_last_index_.*','^test_argmin_keepdims_example_select_last_index_.*'
            ]

        if c2.supports_device('DNNL'):
            current_failing_tests += [
                '^test_range_float_type_positive_delta_expanded_cpu',
                '^test_range_int32_type_negative_delta_expanded_cpu', '^test_averagepool_2d_ceil_cpu',
                '^test_maxpool_2d_ceil_cpu', '^test_maxpool_2d_dilations_cpu',
                '^test_maxpool_2d_uint8'
            ]

        if c2.supports_device('NNAPI'):
            current_failing_tests += [
              '^test_maxpool_2d_uint8'
            ]

        if c2.supports_device('OPENVINO_GPU_FP32') or c2.supports_device('OPENVINO_GPU_FP16'):
            current_failing_tests.append('^test_div_cpu')
            # temporarily exclude vgg19 test which comsumes too much memory, run out of memory on Upsquared device.
            # single test pass for vgg19, need furture investigation
            current_failing_tests.append('^test_vgg19_cpu')

        if c2.supports_device('OPENVINO_CPU_FP32'):
            current_failing_tests += [
                '^test_operator_permute2_cpu',
                '^test_operator_repeat_cpu',
                '^test_operator_repeat_dim_overflow_cpu'
            ]
        if c2.supports_device('OPENVINO_GPU_FP32'):
            current_failing_tests += [
                '^test_operator_permute2_cpu',
                '^test_operator_repeat_cpu',
                '^test_operator_repeat_dim_overflow_cpu',
                '^test_add_bcast_cpu',
                '^test_batchnorm_epsilon_cpu',
                '^test_div_bcast_cpu',
                '^test_mul_bcast_cpu',
                '^test_pow_bcast_array_cpu',
                '^test_sub_bcast_cpu',
                '^test_batchnorm_example_cpu',
                '^test_clip_default_inbounds_cpu',
                '^test_resize_upsample_sizes_nearest_ceil_half_pixel_cpu',
                '^test_resize_upsample_sizes_nearest_floor_align_corners_cpu',
                '^test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric_cpu',
                '^test_unique_not_sorted_without_axis_cpu'
            ]


        filters = current_failing_tests + \
                  tests_with_pre_opset7_dependencies_filters() + \
                  unsupported_usages_filters() + \
                  other_tests_failing_permanently_filters() + \
                  test_with_types_disabled_due_to_binary_size_concerns_filters()

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
    # doesn't work as it filters on the test method name (Runner._add_model_test) rather than inidividual test case names.
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
