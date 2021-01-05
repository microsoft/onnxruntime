// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include <string>
#include <set>

namespace onnxruntime {
namespace test {

/*
Following filters are used by "onnxruntime/onnxruntime/test/providers/cpu/model_tests.cc" and 
"onnxruntime/onnxruntime/test/onnx/main.cc". Any broken tests or blacklisted tests from onnx backend test data set
and from model test data set should be added here.
*/

// Permanently exclude following tests because ORT support only opset starting from 7,
// Please make no more changes to the list
static const ORTCHAR_T* immutable_broken_tests[] = {
    ORT_TSTR("AvgPool1d"),
    ORT_TSTR("AvgPool1d_stride"),
    ORT_TSTR("AvgPool2d"),
    ORT_TSTR("AvgPool2d_stride"),
    ORT_TSTR("AvgPool3d"),
    ORT_TSTR("AvgPool3d_stride"),
    ORT_TSTR("AvgPool3d_stride1_pad0_gpu_input"),
    ORT_TSTR("BatchNorm1d_3d_input_eval"),
    ORT_TSTR("BatchNorm2d_eval"),
    ORT_TSTR("BatchNorm2d_momentum_eval"),
    ORT_TSTR("BatchNorm3d_eval"),
    ORT_TSTR("BatchNorm3d_momentum_eval"),
    ORT_TSTR("GLU"),
    ORT_TSTR("GLU_dim"),
    ORT_TSTR("Linear"),
    ORT_TSTR("PReLU_1d"),
    ORT_TSTR("PReLU_1d_multiparam"),
    ORT_TSTR("PReLU_2d"),
    ORT_TSTR("PReLU_2d_multiparam"),
    ORT_TSTR("PReLU_3d"),
    ORT_TSTR("PReLU_3d_multiparam"),
    ORT_TSTR("PoissonNLLLLoss_no_reduce"),
    ORT_TSTR("Softsign"),
    ORT_TSTR("operator_add_broadcast"),
    ORT_TSTR("operator_add_size1_broadcast"),
    ORT_TSTR("operator_add_size1_right_broadcast"),
    ORT_TSTR("operator_add_size1_singleton_broadcast"),
    ORT_TSTR("operator_addconstant"),
    ORT_TSTR("operator_addmm"),
    ORT_TSTR("operator_basic"),
    ORT_TSTR("operator_mm"),
    ORT_TSTR("operator_non_float_params"),
    ORT_TSTR("operator_params"),
    ORT_TSTR("operator_pow"),
};

static const ORTCHAR_T* cuda_flaky_tests[] = {
    ORT_TSTR("fp16_inception_v1"),
    ORT_TSTR("fp16_shufflenet"),
    ORT_TSTR("fp16_tiny_yolov2"),
    ORT_TSTR("candy"),
    ORT_TSTR("tinyyolov3"),
    ORT_TSTR("mlperf_ssd_mobilenet_300"),
    ORT_TSTR("mlperf_ssd_resnet34_1200"),
    ORT_TSTR("tf_inception_v1"),
    ORT_TSTR("faster_rcnn"),
    ORT_TSTR("split_zero_size_splits"),
    ORT_TSTR("convtranspose_3d"),
    ORT_TSTR("fp16_test_tiny_yolov2-Candy"),
    ORT_TSTR("fp16_coreml_FNS-Candy"),
    ORT_TSTR("fp16_test_tiny_yolov2"),
    ORT_TSTR("fp16_test_shufflenet")};

static const ORTCHAR_T* openvino_disabled_tests[] = {ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                     ORT_TSTR("bertsquad"),
                                                     ORT_TSTR("yolov3"),
                                                     ORT_TSTR("LSTM_Seq_lens_unpacked"),
                                                     ORT_TSTR("tinyyolov3"),
                                                     ORT_TSTR("faster_rcnn"),
                                                     ORT_TSTR("mask_rcnn"),
                                                     ORT_TSTR("coreml_FNS-Candy_ImageNet"),
                                                     ORT_TSTR("tf_mobilenet_v2_1.0_224"),
                                                     ORT_TSTR("tf_mobilenet_v2_1.4_224"),
                                                     ORT_TSTR("operator_permute2"),
                                                     ORT_TSTR("operator_repeat"),
                                                     ORT_TSTR("operator_repeat_dim_overflow"),
                                                     ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                     ORT_TSTR("candy"),
                                                     ORT_TSTR("cntk_simple_seg"),
                                                     ORT_TSTR("GPT2_LM_HEAD"),
                                                     ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight"),
                                                     ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded"),
                                                     ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight"),
                                                     ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob"),
                                                     ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded")};

static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                ORT_TSTR("mlperf_ssd_mobilenet_300"), ORT_TSTR("mask_rcnn"),
                                                ORT_TSTR("faster_rcnn"), ORT_TSTR("tf_pnasnet_large"),
                                                ORT_TSTR("zfnet512"), ORT_TSTR("keras2coreml_Dense_ImageNet")};

static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("densenet121"), ORT_TSTR("resnet18v2"),
                                                 ORT_TSTR("resnet34v2"), ORT_TSTR("resnet50v2"),
                                                 ORT_TSTR("resnet101v2"),
                                                 ORT_TSTR("resnet101v2"), ORT_TSTR("vgg19"),
                                                 ORT_TSTR("tf_inception_resnet_v2"), ORT_TSTR("tf_inception_v1"),
                                                 ORT_TSTR("tf_inception_v3"), ORT_TSTR("tf_inception_v4"),
                                                 ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                 ORT_TSTR("tf_mobilenet_v2_1.0_224"),
                                                 ORT_TSTR("tf_mobilenet_v2_1.4_224"), ORT_TSTR("tf_nasnet_large"),
                                                 ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("tf_resnet_v1_50"),
                                                 ORT_TSTR("tf_resnet_v1_101"), ORT_TSTR("tf_resnet_v1_101"),
                                                 ORT_TSTR("tf_resnet_v2_101"), ORT_TSTR("tf_resnet_v2_152"),
                                                 ORT_TSTR("batchnorm_example_training_mode"),
                                                 ORT_TSTR("batchnorm_epsilon_training_mode"),
                                                 ORT_TSTR("mobilenetv2-1.0"),
                                                 ORT_TSTR("candy"),
                                                 ORT_TSTR("range_float_type_positive_delta_expanded"),
                                                 ORT_TSTR("range_int32_type_negative_delta_expanded"),
                                                 ORT_TSTR("averagepool_2d_ceil"),
                                                 ORT_TSTR("maxpool_2d_ceil"),
                                                 ORT_TSTR("maxpool_2d_dilations"),
                                                 ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                 ORT_TSTR("convtranspose_1d"),
                                                 ORT_TSTR("convtranspose_3d"),
                                                 ORT_TSTR("maxpool_2d_uint8")};

static const ORTCHAR_T* tensorrt_disabled_tests[] = {
    ORT_TSTR("udnie"), ORT_TSTR("rain_princess"),
    ORT_TSTR("pointilism"), ORT_TSTR("mosaic"),
    ORT_TSTR("LSTM_Seq_lens_unpacked"),
    ORT_TSTR("cgan"), ORT_TSTR("candy"),
    ORT_TSTR("tinyyolov3"), ORT_TSTR("yolov3"),
    ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mlperf_ssd_mobilenet_300"),
    ORT_TSTR("mask_rcnn"),
    ORT_TSTR("faster_rcnn"),
    ORT_TSTR("fp16_shufflenet"),
    ORT_TSTR("fp16_inception_v1"),
    ORT_TSTR("fp16_tiny_yolov2"),
    ORT_TSTR("tf_inception_v3"),
    ORT_TSTR("tf_mobilenet_v1_1.0_224"),
    ORT_TSTR("tf_mobilenet_v2_1.0_224"),
    ORT_TSTR("tf_mobilenet_v2_1.4_224"),
    ORT_TSTR("tf_resnet_v1_101"),
    ORT_TSTR("tf_resnet_v1_152"),
    ORT_TSTR("tf_resnet_v1_50"),
    ORT_TSTR("tf_resnet_v2_101"),
    ORT_TSTR("tf_resnet_v2_152"),
    ORT_TSTR("tf_resnet_v2_50"),
    ORT_TSTR("convtranspose_1d"),
    ORT_TSTR("convtranspose_3d"),
    ORT_TSTR("conv_with_strides_and_asymmetric_padding"),
    ORT_TSTR("conv_with_strides_padding"),
    ORT_TSTR("size")  //INVALID_ARGUMENT: Cannot find binding of given name: x
};

#if !defined(__amd64__) && !defined(_M_AMD64)
// out of memory
static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("BERT_Squad"),
                                                ORT_TSTR("bvlc_alexnet"),
                                                ORT_TSTR("bvlc_reference_caffenet"),
                                                ORT_TSTR("coreml_VGG16_ImageNet"),
                                                ORT_TSTR("faster_rcnn"),
                                                ORT_TSTR("GPT2"),
                                                ORT_TSTR("GPT2_LM_HEAD"),
                                                ORT_TSTR("keras_lotus_resnet3D"),
                                                ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                ORT_TSTR("mask_rcnn_keras"),
                                                ORT_TSTR("mask_rcnn"),
                                                ORT_TSTR("ssd"),
                                                ORT_TSTR("vgg19"),
                                                ORT_TSTR("zfnet512")};
#endif

struct BrokenTest {
  std::string test_name_;
  std::string reason_;
  std::set<std::string> broken_versions_ = {};  // apply to all versions if empty
  BrokenTest(std::string name, std::string reason) : test_name_(std::move(name)), reason_(std::move(reason)) {
  }

  BrokenTest(std::string name, std::string reason, const std::initializer_list<std::string>& versions)
      : test_name_(std::move(name)), reason_(std::move(reason)), broken_versions_(versions) {
  }

  bool operator<(const struct BrokenTest& test) const {
    return strcmp(test_name_.c_str(), test.test_name_.c_str()) < 0;
  }
};

inline std::set<BrokenTest> GetBrokenTestsForProvider(const std::string& provider_name) {
  // TODO: filter model based on opset
  std::set<BrokenTest> broken_tests = {
      {"mnist", "Input data isn't in valid range"},
      {"BERT_Squad", "test data bug"},
      {"constantofshape_float_ones", "test data bug", {"onnx141", "onnx150"}},
      {"constantofshape_int_zeros", "test data bug", {"onnx141", "onnx150"}},
      {"cast_STRING_to_FLOAT", "Linux CI has old ONNX python package with bad test data", {"onnx141"}},
      // Numpy float to string has unexpected rounding for some results given numpy default precision is meant to be 8.
      // "e.g. 0.296140194 -> '0.2961402' not '0.29614019'. ORT produces the latter with precision set to 8,
      // which doesn't match the expected output that was generated with numpy.
      {"cast_FLOAT_to_STRING", "Numpy float to string has unexpected rounding for some results."},
      {"tf_nasnet_large", "disable temporarily"},
      {"tf_nasnet_mobile", "disable temporarily"},
      {"tf_pnasnet_large", "disable temporarily"},
      {"shrink", "test case is wrong", {"onnx141"}},
      {"maxpool_with_argmax_2d_precomputed_strides", "ShapeInferenceError"},
      {"tf_inception_v2", "result mismatch"},
      {"tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied"},
      {"mxnet_arcface", "Model is an invalid ONNX model"},
      {"unique_not_sorted_without_axis", "Expected data for 'Y' is incorrect and in sorted order."},
      {"cumsum_1d_reverse_exclusive", "only failing linux GPU CI. Likely build error."},
      {"resize_downsample_scales_cubic_align_corners", "results mismatch with onnx tests"},
      {"resize_downsample_scales_linear_align_corners", "results mismatch with onnx tests"},
      {"resize_tf_crop_and_resize", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_ceil_half_pixel", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_floor_align_corners", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", "Bad onnx test output. Needs test fix."},
      {"bitshift_right_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"bitshift_left_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"maxunpool_export_with_output_shape",
       "Invalid output in ONNX test. See https://github.com/onnx/onnx/issues/2398"},
      {"training_dropout", "result differs", {}},               // Temporary, subsequent PR will remove this.
      {"training_dropout_default", "result differs", {}},       // Temporary, subsequent PR will remove this.
      {"training_dropout_default_mask", "result differs", {}},  // Temporary, subsequent PR will remove this.
      {"training_dropout_mask", "result differs", {}},          // Temporary, subsequent PR will remove this.
#ifdef ENABLE_TRAINING
      {"adagrad", "not a registered function/op", {}},                  // Op not registered.
      {"adagrad_multiple", "not a registered function/op", {}},         // Op not registered.
      {"adam", "not a registered function/op", {}},                     // Op not registered.
      {"adam_multiple", "not a registered function/op", {}},            // Op not registered.
      {"gradient_of_add", "not a registered function/op", {}},          // Op not registered.
      {"gradient_of_add_and_mul", "not a registered function/op", {}},  // Op not registered.
      {"momentum", "not a registered function/op", {}},                 // Op not registered.
      {"momentum_multiple", "not a registered function/op", {}},        // Op not registered.
      {"nesterov_momentum", "not a registered function/op", {}},        // Op not registered.
#endif
      {"mask_rcnn_keras", "this model currently has an invalid contrib op version set to 10", {}},
      {"cast_FLOAT_to_BFLOAT16", "onnx generate bfloat tensor as uint16 type", {}},
      {"cast_BFLOAT16_to_FLOAT", "onnx generate bfloat tensor as uint16 type", {}},
      {"sequence_insert_at_back", "onnx currently not supporting loading segment", {}},
      {"sequence_insert_at_front", "onnx currently not supporting loading segment", {}},
      {"loop13_seq", "ORT api does not currently support creating empty sequences (needed for this test)", {}},
  };

  if (provider_name == "nnapi") {
    broken_tests.insert({"scan9_sum", "Error with the extra graph"});
    broken_tests.insert({"scan_sum", "Error with the extra graph"});
    broken_tests.insert({"mvn_expanded", "Failed to find kernel for MemcpyFromHost(1) (node Memcpy_1)"});
    broken_tests.insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"gemm_transposeB", "Temporarily disabled pending investigation"});
    broken_tests.insert({"range_float_type_positive_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"range_int32_type_negative_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});
    broken_tests.insert({"convtranspose_3d", "3d convtranspose not supported yet"});
    broken_tests.insert({"maxpool_2d_uint8", "result mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NC_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_expanded", "shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded", "shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded", "shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded", "shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded", "shape mismatch"});
    // Disable based on George Wu's recommendation.
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded",
         "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NC", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index",
                         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                         "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                         "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"nllloss_NCd1_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_with_weight_reduction_mean", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_with_weight_reduction_mean_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2d3d4d5_mean_weight", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2d3d4d5_mean_weight_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_ii_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ii_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ii_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded", "wait for investigation"});
  }

  if (provider_name == "dml") {
    broken_tests.insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests.insert({"PixelShuffle", "Test requires 6D Reshape, which isn't supported by DirectML"});
    broken_tests.insert({"operator_permute2", "Test requires 6D Transpose, which isn't supported by DirectML"});
    broken_tests.insert({"resize_downsample_linear",
                         "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests.insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests.insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});

    // These tests are temporarily disabled pending investigation
    broken_tests.insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"mxnet_arcface", "Temporarily disabled pending investigation"});
    broken_tests.insert({"yolov3", "Temporarily disabled pending investigation"});
    broken_tests.insert({"tf_inception_v2", "Temporarily disabled pending investigation"});
    broken_tests.insert({"fp16_inception_v1", "Temporarily disabled pending investigation"});
    broken_tests.insert({"candy", "Temporarily disabled pending investigation"});
    broken_tests.insert({"BERT_Squad", "Temporarily disabled pending investigation"});
    broken_tests.insert({"LSTM_Seq_lens_unpacked", "The parameter is incorrect"});

    broken_tests.insert({"resize_downsample_scales_linear",
                         "DML uses half_pixel and this test assumed \"asymmetric\" but does not include \"mode\""});
    broken_tests.insert({"resize_downsample_sizes_linear_pytorch_half_pixel",
                         "DML does not support downsampling by such a large factor - skips input pixels"});
    broken_tests.insert({"resize_downsample_sizes_nearest",
                         "DML uses pixel centers for nearest, rounding 1 value off for the middle column"});
    broken_tests.insert({"resize_upsample_sizes_nearest",
                         "DML uses pixel centers for nearest, which makes more sense (the 3rd row mismatches)"});
    broken_tests.insert({"unsqueeze_three_axes", "DML does not support 6D tensors"});
    broken_tests.insert({"unsqueeze_unsorted_axes", "DMLdoes not support 6D tensors"});

    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded",
                         "DML does not support 5D+ tensors"});
  }

#ifdef DISABLE_CONTRIB_OPS
  broken_tests.insert({"coreml_SqueezeNet_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Permute_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_ReLU_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Padding-Upsampling-Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"fp16_tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Pooling_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Padding_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet_small", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet_large", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_leakyrelu_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_hard_sigmoid_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_elu_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Dense_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Conv2D_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_VGG16_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_Resnet50_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_Inceptionv3_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_FNS-Candy_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_AgeNet_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_ImageNet_large", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_ImageNet_small", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu_default", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_default_axes", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu_example", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_neg failed", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_start_out_of_bounds", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_end_out_of_bounds", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_neg", "This model uses contrib ops."});
  broken_tests.insert({"mvn", "This model uses contrib ops.", {"onnx130"}});
  broken_tests.insert({"cdist_float32_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"bidaf", "This model uses contrib ops."});
  broken_tests.insert({"fp16_test_tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"fp16_coreml_FNS-Candy", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Repeat_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_BiDirectional_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"fp16_coreml_LinearRegression_NYCTaxi", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_GRU_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_SimpleRNN_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Dot_imageNet", "This model uses contrib ops."});
#endif

  return broken_tests;
}

/*
This filter is used by KernelRegistryTests.kernels_registered_for_all_onnx_ops. This test
validates that ort CPU EP has registered kernels for all onnx domain operator schemas.
When ONNX commit is updated ORT may not have an implementation for the latest updates coming in from ONNX
in this case the ops which are updated or newly added should be added here so that the test can still pass.
Before any ORT release all the ops from this filter should be cleared.
*/
static const std::vector<std::string> expected_not_registered_ops = {
    // Following 3 ops are permanently filters since CPU EP does not have kernel for these ops by design.
    "Constant",
    "MemcpyToHost",
    "MemcpyFromHost"};
}  // namespace test
}  // namespace onnxruntime