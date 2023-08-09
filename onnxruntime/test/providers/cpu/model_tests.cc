// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iterator>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_apis.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "asserts.h"
#include <core/platform/path_lib.h>
#include "default_providers.h"
#include <string>
#include <codecvt>
#include <locale>

#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#endif

#ifdef USE_NNAPI
#include "core/providers/nnapi/nnapi_provider_factory.h"
#endif

#ifdef USE_RKNPU
#include "core/providers/rknpu/rknpu_provider_factory.h"
#endif

#ifdef USE_ACL
#include "core/providers/acl/acl_provider_factory.h"
#endif

#ifdef USE_ARMNN
#include "core/providers/armnn/armnn_provider_factory.h"
#endif

// test infrastructure
#include "test/onnx/testenv.h"
#include "test/onnx/TestCase.h"
#include "test/compare_ortvalue.h"
#include "test/onnx/heap_buffer.h"
#include "test/onnx/onnx_model_info.h"
#include "test/onnx/callback.h"
#include "test/onnx/testcase_request.h"

extern std::unique_ptr<Ort::Env> ort_env;

// asserts that the OrtStatus* result of `status_expr` does not indicate an error
// note: this takes ownership of the OrtStatus* result
#define ASSERT_ORT_STATUS_OK(status_expr)                                           \
  do {                                                                              \
    if (OrtStatus* _status = (status_expr); _status != nullptr) {                   \
      std::unique_ptr<OrtStatus, decltype(&OrtApis::ReleaseStatus)> _rel_status{    \
          _status, &OrtApis::ReleaseStatus};                                        \
      FAIL() << "OrtStatus error: " << OrtApis::GetErrorMessage(_rel_status.get()); \
    }                                                                               \
  } while (false)

using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {
// parameter is provider_name + "_" + model_path
class ModelTest : public testing::TestWithParam<std::basic_string<ORTCHAR_T>> {};

namespace {
struct BrokenTest {
  std::string test_name_;
  std::string reason_;
  std::set<std::string> broken_opset_versions_ = {};  // apply to all versions if empty
  BrokenTest(std::string name, std::string reason) : test_name_(std::move(name)), reason_(std::move(reason)) {
  }

  BrokenTest(std::string name, std::string reason, const std::initializer_list<std::string>& opversions)
      : test_name_(std::move(name)), reason_(std::move(reason)), broken_opset_versions_(opversions) {
  }

  bool operator<(const struct BrokenTest& test) const {
    return strcmp(test_name_.c_str(), test.test_name_.c_str()) < 0;
  }
};
}  // namespace
#ifdef GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ModelTest);
#endif

void SkipTest(const std::string& reason = "") {
  GTEST_SKIP() << "Skipping single test " << reason;
}

TEST_P(ModelTest, Run) {
  std::basic_string<ORTCHAR_T> param = GetParam();
  size_t pos = param.find(ORT_TSTR("_"));
  ASSERT_NE(pos, std::string::npos);
  std::string provider_name = ToUTF8String(param.substr(0, pos));
  std::basic_string<ORTCHAR_T> model_path = param.substr(pos + 1);
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 1e-3;

  // when cuda or openvino is enabled, set it to a larger value for resolving random MNIST test failure
  if (model_path.find(ORT_TSTR("_MNIST")) > 0) {
    if (provider_name == "cuda" || provider_name == "openvino") {
      relative_per_sample_tolerance = 1e-2;
    }
  }

  std::unique_ptr<OnnxModelInfo> model_info = std::make_unique<OnnxModelInfo>(model_path.c_str());
  if ((model_info->GetONNXOpSetVersion() < 14 || model_info->GetONNXOpSetVersion() > 17) &&
      provider_name == "tensorrt") {
    // TensorRT can run most of the model tests, but only part of
    // them is enabled here to save CI build time.
    // Besides saving CI build time, TRT isn’t able to support full ONNX ops spec and therefore some testcases will
    // fail. That's one of reasons we skip those testcases and only test latest ONNX opsets.
    SkipTest(" tensorrt: only enable opset 14 to 17 of onnx tests");
    return;
  }

  if ((model_info->GetONNXOpSetVersion() == 10 || model_info->GetONNXOpSetVersion() >= 18) && provider_name == "dnnl") {
    // DNNL can run most of the model tests, but only part of
    // them is enabled here to save CI build time.
    std::ostringstream oss;
    oss << " dnnl doesn't support opset " << model_info->GetONNXOpSetVersion();
    SkipTest(oss.str());
    return;
  }

  if (model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_TRAINING_DOMAIN) ||
      model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_PREVIEW_TRAINING_DOMAIN)) {
    SkipTest("it has the training domain. No pipeline should need to run these tests.");
    return;
  }
  std::set<BrokenTest> broken_tests = {
      {"slice_neg_steps",
       "Type parameter (Tind) bound to different types (tensor(int64) and tensor(int32) in node ()."},
      {"cast_BFLOAT16_to_FLOAT", "Unexpected input data type"},
      {"loop13_seq", "Creation of empty sequences is currently not supported in the test runner"},
      {"sequence_insert_at_front", "shape mismatch, expect {4} got {3}"},
      {"cast_FLOAT_to_BFLOAT16", "expect uint16 got bfloat16"},
      {"mnist", "Input data isn't in valid range"},
      {"BERT_Squad", "test data bug"},
      {"constantofshape_float_ones", "test data bug", {"opset9", "opset10"}},
      {"constantofshape_int_zeros", "test data bug", {"opset9", "opset10"}},
      {"cast_STRING_to_FLOAT", "Linux CI has old ONNX python package with bad test data", {"opset9", "opset10"}},
      // Numpy float to string has unexpected rounding for some results given numpy default precision is meant to be 8.
      // "e.g. 0.296140194 -> '0.2961402' not '0.29614019'. ORT produces the latter with precision set to 8,
      // which doesn't match the expected output that was generated with numpy.
      {"cast_FLOAT_to_STRING", "Numpy float to string has unexpected rounding for some results."},
      {"tf_nasnet_large", "disable temporarily"},
      {"tf_nasnet_mobile", "disable temporarily"},
      {"tf_pnasnet_large", "disable temporarily"},
      {"shrink", "test case is wrong", {"opset9"}},
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
      {"cntk_simple_seg", "Bad onnx test output caused by wrong SAME_UPPER/SAME_LOWER for ConvTranspose"},
      {"training_dropout", "result differs", {}},               // Temporary, subsequent PR will remove this.
      {"training_dropout_default", "result differs", {}},       // Temporary, subsequent PR will remove this.
      {"training_dropout_default_mask", "result differs", {}},  // Temporary, subsequent PR will remove this.
      {"training_dropout_mask", "result differs", {}},          // Temporary, subsequent PR will remove this.
      {"batchnorm_epsilon_training_mode", "training only", {}},
      {"batchnorm_example_training_mode", "training only", {}},
      {"bernoulli", "type error", {}},
      {"bernoulli_double", "type error", {}},
      {"bernoulli_double_expanded", "type error", {}},
      {"bernoulli_expanded", "type error", {}},
      {"bernoulli_seed", "type error", {}},
      {"bernoulli_seed_expanded", "type error", {}},
      {"castlike_BFLOAT16_to_FLOAT", "type error", {}},
      {"castlike_BFLOAT16_to_FLOAT_expanded", "type error", {}},
      {"castlike_FLOAT_to_BFLOAT16", "type error", {}},
      {"castlike_FLOAT_to_BFLOAT16_expanded", "type error", {}},
      {"castlike_FLOAT_to_STRING", "type error", {}},
      {"castlike_FLOAT_to_STRING_expanded", "type error", {}},
      {"convtranspose_autopad_same", "Test data has been corrected in ONNX 1.10.", {"opset13", "opset14"}},
      {"gru_batchwise", "type error", {}},
      {"lstm_batchwise", "type error", {}},
      {"optional_get_element", "type error", {}},
      {"optional_get_element_sequence", "type error", {}},
      {"optional_has_element", "type error", {}},
      {"optional_has_element_empty", "type error", {}},
      {"shape_end_1", "type error", {}},
      {"shape_end_negative_1", "type error", {}},
      {"shape_start_1", "type error", {}},
      {"shape_start_1_end_2", "type error", {}},
      {"shape_start_1_end_negative_1", "type error", {}},
      {"shape_start_negative_1", "type error", {}},
      {"simple_rnn_batchwise", "type error", {}},
      {"mod_float_mixed_sign_example", "fmod attribute must be true for floating point types", {}},
      {"col2im_pads", "result mismatch", {"opset18"}},
#ifdef ENABLE_TRAINING_CORE
      {"adagrad", "not a registered function/op", {}},                  // Op not registered.
      {"adagrad_multiple", "not a registered function/op", {}},         // Op not registered.
      {"adam", "not a registered function/op", {}},                     // Op not registered.
      {"adam_multiple", "not a registered function/op", {}},            // Op not registered.
      {"gradient_of_add", "not a registered function/op", {}},          // Op not registered.
      {"gradient_of_add_and_mul", "not a registered function/op", {}},  // Op not registered.
      {"momentum", "not a registered function/op", {}},                 // Op not registered.
      {"momentum_multiple", "not a registered function/op", {}},        // Op not registered.
      {"nesterov_momentum", "not a registered function/op", {}},        // Op not registered.
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_mean_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_mean_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none_weights_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_sum_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_sum", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
       "type error",
       {"opset12"}},
      {"softmax_cross_entropy_none_weights", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "type error", {"opset12"}},
      {"softmax_cross_entropy_none", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "type error", {"opset12"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_weight", "type error", {"opset12"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "type error", {"opset12"}},
#endif
      {"mask_rcnn_keras", "this model currently has an invalid contrib op version set to 10", {}}};

  // Some EPs may fail to pass some specific testcases.
  // For example TenosrRT EP may fail on FLOAT16 related testcases if GPU doesn't support float16.
  // Instead of list all these testcases, we can use following keyword set to filter out testcases wchich contain
  // specific keyword.
  std::set<std::string> broken_tests_keyword_set = {};

  if (provider_name == "cuda") {
#ifdef _WIN32
    broken_tests.insert({"LSTM_Seq_lens_unpacked", "this test fails with new image since Aug 25."});
    broken_tests.insert({"bidaf", "this test fails with new image since Aug 25."});
#else
    broken_tests.insert({"bidaf", "this test should be recovered when multi-gpu pipeline deprecates NV12", {"opset9"}});
#endif
  }

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
  }

  if (provider_name == "tensorrt") {
    broken_tests.insert({"convtranspose_with_kernel", "It causes segmentation fault"});
    broken_tests.insert({"convtranspose_pad", "It causes segmentation fault"});
    broken_tests.insert({"convtranspose_kernel_shape", "It causes segmentation fault"});
    broken_tests.insert({"dynamicquantizelinear_expanded", "It causes segmentation fault"});
    broken_tests.insert({"dynamicquantizelinear_min_adjusted_expanded", "It causes segmentation fault"});
    broken_tests.insert({"dynamicquantizelinear_max_adjusted_expanded", "It causes segmentation fault"});

    broken_tests.insert({"basic_conv_with_padding",
                         "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                         "engine for fused node"});
    broken_tests.insert({"basic_conv_without_padding",
                         "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                         "engine for fused node"});
    broken_tests.insert({"conv_with_strides_no_padding",
                         "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                         "engine for fused node"});

    broken_tests.insert({"conv_with_autopad_same",
                         "Internal Error (node_of_y: Cannot set more than one input unless network has Q/DQ layers.)"});

    // unsupported tests since opset16
    broken_tests.insert({"sequence_map_add_2_sequences", "not supported by TensorRT EP"});
    broken_tests.insert({"sequence_map_extract_shapes", "not supported by TensorRT EP."});
    broken_tests.insert({"sequence_map_add_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests.insert({"sequence_map_identity_1_sequence", "not supported by TensorRT EP."});
    broken_tests.insert({"sequence_map_identity_2_sequences", "not supported by TensorRT EP."});
    broken_tests.insert({"sequence_map_identity_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests.insert({"leakyrelu_expanded", "not supported by TensorRT EP."});
    broken_tests.insert({"leakyrelu_default_expanded", "not supported by TensorRT EP."});
    broken_tests.insert({"leakyrelu_example_expanded", "not supported by TensorRT EP."});
    broken_tests.insert({"prelu_broadcast_expanded", "not supported by TensorRT EP."});
    broken_tests.insert({"prelu_example_expanded", "not supported by TensorRT EP."});
    broken_tests_keyword_set.insert({"scatternd_add"});
    broken_tests_keyword_set.insert({"scatternd_multiply"});
    broken_tests_keyword_set.insert({"scatter_elements_with_duplicate_indices"});

    // sce op is not supported
    broken_tests_keyword_set.insert({"sce"});

    // TensorRT EP CI uses Nvidia Tesla M60 which doesn't support fp16.
    broken_tests_keyword_set.insert({"FLOAT16"});
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

  std::basic_string<ORTCHAR_T> model_dir;
  (void)GetDirNameFromFilePath(model_path, model_dir);
  std::basic_string<PATH_CHAR_TYPE> test_case_name = GetLastComponent(model_dir);
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0)
    test_case_name = test_case_name.substr(5);
  {
    BrokenTest t = {ToUTF8String(test_case_name), ""};
    auto iter = broken_tests.find(t);
    auto opset_version = model_info->GetNominalOpsetVersion();
    if (iter != broken_tests.end() &&
        (opset_version == TestModelInfo::unknown_version || iter->broken_opset_versions_.empty() ||
         iter->broken_opset_versions_.find(opset_version) != iter->broken_opset_versions_.end())) {
      SkipTest("It's in broken_tests");
      return;
    }

    for (auto iter2 = broken_tests_keyword_set.begin(); iter2 != broken_tests_keyword_set.end(); ++iter2) {
      std::string keyword = *iter2;
      if (ToUTF8String(test_case_name).find(keyword) != std::string::npos) {
        SkipTest("It's in broken_tests_keyword");
        return;
      }
    }
  }

  // TODO(leca): move the parallel run test list to a config file and load it in GetParameterStrings() to make the load process run only once
  std::set<std::string> tests_run_parallel = {"test_resnet18v2",
                                              "test_resnet34v2",
                                              "test_resnet50",
                                              "test_resnet50v2",
                                              "test_resnet101v2",
                                              "test_resnet152v2",
                                              "keras_lotus_resnet3D",
                                              "coreml_Resnet50_ImageNet",
                                              "mlperf_mobilenet",
                                              "mlperf_resnet",
                                              "mlperf_ssd_mobilenet_300",
                                              "mlperf_ssd_resnet34_1200"};
  bool is_single_node = !model_info->GetNodeName().empty();
  std::vector<ExecutionMode> execution_modes = {ExecutionMode::ORT_SEQUENTIAL};
  if (provider_name == "cpu" && !is_single_node)
    execution_modes.push_back(ExecutionMode::ORT_PARALLEL);

  std::vector<bool> use_single_thread{false};
  // Test the model with intra op threadpool disabled
  if (provider_name == "cpu" && is_single_node)
    use_single_thread.push_back(true);

  std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToUTF8String(test_case_name), std::move(model_info),
                                                    per_sample_tolerance, relative_per_sample_tolerance);

#ifndef USE_DNNL
  auto tp = TestEnv::CreateThreadPool(Env::Default());
#endif

  for (bool is_single_thread : use_single_thread) {
    for (ExecutionMode execution_mode : execution_modes) {
      Ort::SessionOptions ortso{};
      if (!is_single_thread) {
        ortso.DisablePerSessionThreads();
      } else {
        ortso.SetIntraOpNumThreads(1);
      }
      ortso.SetExecutionMode(execution_mode);
      ortso.SetLogId(ToUTF8String(test_case_name).c_str());
      ortso.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
      if (provider_name == "cuda") {
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        ASSERT_ORT_STATUS_OK(OrtApis::CreateCUDAProviderOptions(&cuda_options));
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(&OrtApis::ReleaseCUDAProviderOptions)> rel_cuda_options(
            cuda_options, &OrtApis::ReleaseCUDAProviderOptions);
        std::vector<const char*> keys{"device_id"};

        std::vector<const char*> values;
        std::string device_id = Env::Default().GetEnvironmentVar("ONNXRUNTIME_TEST_GPU_DEVICE_ID");
        values.push_back(device_id.empty() ? "0" : device_id.c_str());
        ASSERT_ORT_STATUS_OK(OrtApis::UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), 1));
        ortso.AppendExecutionProvider_CUDA_V2(*cuda_options);
      } else if (provider_name == "rocm") {
        OrtROCMProviderOptions ep_options;
        ortso.AppendExecutionProvider_ROCM(ep_options);
      }
#ifdef USE_DNNL
      else if (provider_name == "dnnl") {
        OrtDnnlProviderOptions* ep_option;
        ASSERT_ORT_STATUS_OK(OrtApis::CreateDnnlProviderOptions(&ep_option));
        std::unique_ptr<OrtDnnlProviderOptions, decltype(&OrtApis::ReleaseDnnlProviderOptions)>
            rel_dnnl_options(ep_option, &OrtApis::ReleaseDnnlProviderOptions);
        ep_option->use_arena = 0;
        ASSERT_ORT_STATUS_OK(OrtApis::SessionOptionsAppendExecutionProvider_Dnnl(ortso, ep_option));
      }
#endif
      else if (provider_name == "tensorrt") {
        if (test_case_name.find(ORT_TSTR("FLOAT16")) != std::string::npos) {
          OrtTensorRTProviderOptionsV2 params{0, 0, nullptr, 1000, 1, 1 << 30,
                                              1,  // enable fp16
                                              0, nullptr, 0, 0, 0, 0, 0, nullptr, 0, nullptr, 0, 0, 0, 0, 0, 0, 0, 0,
                                              3, -1, nullptr, nullptr, nullptr, nullptr, nullptr, 0};

          ortso.AppendExecutionProvider_TensorRT_V2(params);
        } else {
          OrtTensorRTProviderOptionsV2* ep_option = nullptr;
          ASSERT_ORT_STATUS_OK(OrtApis::CreateTensorRTProviderOptions(&ep_option));
          std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(&OrtApis::ReleaseTensorRTProviderOptions)>
              rel_cuda_options(ep_option, &OrtApis::ReleaseTensorRTProviderOptions);
          ortso.AppendExecutionProvider_TensorRT_V2(*ep_option);
        }
        // Enable CUDA fallback
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        ASSERT_ORT_STATUS_OK(OrtApis::CreateCUDAProviderOptions(&cuda_options));
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(&OrtApis::ReleaseCUDAProviderOptions)> rel_cuda_options(
            cuda_options, &OrtApis::ReleaseCUDAProviderOptions);
        ortso.AppendExecutionProvider_CUDA_V2(*cuda_options);
      } else if (provider_name == "migraphx") {
        OrtMIGraphXProviderOptions ep_options;
        ortso.AppendExecutionProvider_MIGraphX(ep_options);
      } else if (provider_name == "openvino") {
        OrtOpenVINOProviderOptions ep_options;
        ortso.AppendExecutionProvider_OpenVINO(ep_options);
      }
#ifdef USE_NNAPI
      else if (provider_name == "nnapi") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_Nnapi(ortso, 0));
      }
#endif
#ifdef USE_RKNPU
      else if (provider_name == "rknpu") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_Rknpu(ortso));
      }
#endif
#ifdef USE_ACL
      else if (provider_name == "acl") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_ACL(ortso, 0));
      }
#endif
#ifdef USE_ARMNN
      else if (provider_name == "armnn") {
        ASSERT_ORT_STATUS_OK(OrtSessionOptionsAppendExecutionProvider_ArmNN(ortso));
      }
#endif
      OrtSession* ort_session;
      OrtStatus* ort_st = OrtApis::CreateSession(*ort_env, model_path.c_str(), ortso, &ort_session);
      if (ort_st != nullptr) {
        OrtErrorCode error_code = OrtApis::GetErrorCode(ort_st);
        if (error_code == ORT_NOT_IMPLEMENTED) {
          OrtApis::ReleaseStatus(ort_st);
          continue;
        }
        FAIL() << OrtApis::GetErrorMessage(ort_st);
      }
      std::unique_ptr<OrtSession, decltype(&OrtApis::ReleaseSession)> rel_ort_session(ort_session,
                                                                                      &OrtApis::ReleaseSession);
      const size_t data_count = l->GetDataCount();
#ifndef USE_DNNL  // potential crash for DNNL pipeline
      if (data_count > 1 && tests_run_parallel.find(l->GetTestCaseName()) != tests_run_parallel.end()) {
        LOGS_DEFAULT(ERROR) << "Parallel test for " << l->GetTestCaseName();  // TODO(leca): change level to INFO or even delete the log once verified parallel test working
        std::shared_ptr<TestCaseResult> results = TestCaseRequestContext::Run(tp.get(), *l, *ort_env, ortso, data_count, 1 /*repeat_count*/);
        for (EXECUTE_RESULT res : results->GetExcutionResult()) {
          EXPECT_EQ(res, EXECUTE_RESULT::SUCCESS) << "is_single_thread:" << is_single_thread << ", execution_mode:" << execution_mode << ", provider_name:"
                                                  << provider_name << ", test name:" << results->GetName() << ", result: " << res;
        }
        continue;
      }
#endif  // !USE_DNNL
      // TODO(leca): leverage TestCaseRequestContext::Run() to make it short
      auto default_allocator = std::make_unique<MockedOrtAllocator>();

      for (size_t task_id = 0; task_id != data_count; ++task_id) {
        onnxruntime::test::HeapBuffer holder;
        std::unordered_map<std::string, Ort::Value> feeds;
        l->LoadTestData(task_id, holder, feeds, true);
        size_t output_count;
        ASSERT_ORT_STATUS_OK(OrtApis::SessionGetOutputCount(ort_session, &output_count));
        // Create output feed
        std::vector<char*> output_names(output_count);
        for (size_t i = 0; i != output_count; ++i) {
          ASSERT_ORT_STATUS_OK(
              OrtApis::SessionGetOutputName(ort_session, i, default_allocator.get(), &output_names[i]));
        }

        std::vector<const char*> input_names;
        std::vector<OrtValue*> input_values;
        std::vector<OrtValue*> output_values(output_count);
        {
          for (auto& p : feeds) {
            input_names.push_back(p.first.c_str());
            input_values.push_back(p.second);
          }
          ort_st = OrtApis::Run(ort_session, nullptr, input_names.data(), input_values.data(), input_values.size(),
                                output_names.data(), output_names.size(), output_values.data());
          if (ort_st != nullptr) {
            OrtErrorCode error_code = OrtApis::GetErrorCode(ort_st);
            if (error_code == ORT_NOT_IMPLEMENTED) {
              OrtApis::ReleaseStatus(ort_st);
              for (char* p : output_names) {
                default_allocator->Free(p);
              }
              for (OrtValue* v : output_values) {
                OrtApis::ReleaseValue(v);
              }
            }
            FAIL() << OrtApis::GetErrorMessage(ort_st);
          }
        }

        bool post_procesing = false;
        Status status;
        l->GetPerSampleTolerance(&per_sample_tolerance);
        l->GetRelativePerSampleTolerance(&relative_per_sample_tolerance);
        l->GetPostProcessing(&post_procesing);

        // TODO: if there are no output value files, just skip the validation
        std::unordered_map<std::string, Ort::Value> expected_output_values;
        l->LoadTestData(task_id, holder, expected_output_values, false);

        std::unordered_map<std::string, OrtValue*> name_fetch_output_map;
        std::unordered_map<std::string, const ONNX_NAMESPACE::ValueInfoProto*> name_output_value_info_proto;
        size_t i = 0;
        for (auto& output_name : output_names) {
          // p_fetches is filled in the order of output_names.
          name_fetch_output_map[output_name] = output_values[i];
          const ONNX_NAMESPACE::ValueInfoProto* infoProto = l->GetOutputInfoFromModel(i);
          if (infoProto != nullptr)
            name_output_value_info_proto.insert(std::make_pair(infoProto->name(), infoProto));
          i++;
        }

        for (auto& output : expected_output_values) {
          const OrtValue* expected_output_value = output.second;
          const std::string& output_name = output.first;
          auto iter = name_fetch_output_map.find(output_name);
          ASSERT_NE(iter, name_fetch_output_map.end());

          OrtValue* actual_output_value = iter->second;
          std::pair<COMPARE_RESULT, std::string> ret =
              CompareOrtValue(*actual_output_value, *expected_output_value, per_sample_tolerance,
                              relative_per_sample_tolerance, post_procesing);
          COMPARE_RESULT compare_result = ret.first;
          ASSERT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;

          const ONNX_NAMESPACE::ValueInfoProto* v = name_output_value_info_proto[output_name];
          if (v == nullptr)
            continue;
          ret = VerifyValueInfo(*v, actual_output_value);
          compare_result = ret.first;
          ASSERT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;

          if (compare_result != COMPARE_RESULT::SUCCESS) {
            break;
          }
        }
        for (char* p : output_names) {
          default_allocator->Free(p);
        }
        for (OrtValue* v : output_values) {
          OrtApis::ReleaseValue(v);
        }
      }
    }
  }
}

// TODO: all providers
::std::vector<::std::basic_string<ORTCHAR_T>> GetParameterStrings() {
  std::vector<const ORTCHAR_T*> provider_names;
  provider_names.push_back(ORT_TSTR("cpu"));

#ifdef USE_TENSORRT
  provider_names.push_back(ORT_TSTR("tensorrt"));
#endif
#ifdef USE_MIGRAPHX
  provider_names.push_back(ORT_TSTR("migraphx"));
#endif
#ifdef USE_OPENVINO
  provider_names.push_back(ORT_TSTR("openvino"));
#endif
#ifdef USE_CUDA
  provider_names.push_back(ORT_TSTR("cuda"));
#endif
#ifdef USE_ROCM
  provider_names.push_back(ORT_TSTR("rocm"));
#endif
#ifdef USE_DNNL
  provider_names.push_back(ORT_TSTR("dnnl"));
#endif
// For any non-Android system, NNAPI will only be used for ort model converter
#if defined(USE_NNAPI) && defined(__ANDROID__)
  provider_names.push_back(ORT_TSTR("nnapi"));
#endif
#ifdef USE_RKNPU
  provider_names.push_back(ORT_TSTR("rknpu"));
#endif
#ifdef USE_ACL
  provider_names.push_back(ORT_TSTR("acl"));
#endif
#ifdef USE_ARMNN
  provider_names.push_back(ORT_TSTR("armnn"));
#endif
#ifdef USE_DML
  provider_names.push_back(ORT_TSTR("dml"));
#endif
  std::vector<std::basic_string<ORTCHAR_T>> v;
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

  static const ORTCHAR_T* cuda_flaky_tests[] = {ORT_TSTR("fp16_inception_v1"),
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
                                                ORT_TSTR("fp16_test_shufflenet"),
                                                ORT_TSTR("keras2coreml_SimpleRNN_ImageNet")};
  static const ORTCHAR_T* openvino_disabled_tests[] = {
      ORT_TSTR("tf_mobilenet_v1_1.0_224"),
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
      ORT_TSTR("mlperf_ssd_mobilenet_300"),
      ORT_TSTR("fp16_coreml_FNS-Candy"),
      ORT_TSTR("fp16_test_tiny_yolov2"),
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
      ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded"),
      // models from model zoo
      ORT_TSTR("Tiny YOLOv3"),
      ORT_TSTR("BERT-Squad"),
      ORT_TSTR("YOLOv3"),
      ORT_TSTR("Candy"),
      ORT_TSTR("SSD"),
      ORT_TSTR("ResNet101_DUC_HDC-12"),
      ORT_TSTR("YOLOv3-12")};
  static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                  ORT_TSTR("mlperf_ssd_mobilenet_300"),
                                                  ORT_TSTR("mask_rcnn"),
                                                  ORT_TSTR("faster_rcnn"),
                                                  ORT_TSTR("tf_pnasnet_large"),
                                                  ORT_TSTR("zfnet512"),
                                                  ORT_TSTR("keras2coreml_Dense_ImageNet")};
  static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("densenet121"),
                                                   ORT_TSTR("resnet18v2"),
                                                   ORT_TSTR("resnet34v2"),
                                                   ORT_TSTR("resnet50v2"),
                                                   ORT_TSTR("resnet101v2"),
                                                   ORT_TSTR("resnet101v2"),
                                                   ORT_TSTR("vgg19"),
                                                   ORT_TSTR("tf_inception_resnet_v2"),
                                                   ORT_TSTR("tf_inception_v1"),
                                                   ORT_TSTR("tf_inception_v3"),
                                                   ORT_TSTR("tf_inception_v4"),
                                                   ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                   ORT_TSTR("tf_mobilenet_v2_1.0_224"),
                                                   ORT_TSTR("tf_mobilenet_v2_1.4_224"),
                                                   ORT_TSTR("tf_nasnet_large"),
                                                   ORT_TSTR("tf_pnasnet_large"),
                                                   ORT_TSTR("tf_resnet_v1_50"),
                                                   ORT_TSTR("tf_resnet_v1_101"),
                                                   ORT_TSTR("tf_resnet_v1_101"),
                                                   ORT_TSTR("tf_resnet_v2_101"),
                                                   ORT_TSTR("tf_resnet_v2_152"),
                                                   ORT_TSTR("batchnorm_example_training_mode"),
                                                   ORT_TSTR("batchnorm_epsilon_training_mode"),
                                                   ORT_TSTR("mobilenetv2-1.0"),
                                                   ORT_TSTR("shufflenet"),
                                                   ORT_TSTR("candy"),
                                                   ORT_TSTR("range_float_type_positive_delta_expanded"),
                                                   ORT_TSTR("range_int32_type_negative_delta_expanded"),
                                                   ORT_TSTR("averagepool_2d_ceil"),
                                                   ORT_TSTR("maxpool_2d_ceil"),
                                                   ORT_TSTR("maxpool_2d_dilations"),
                                                   ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                   ORT_TSTR("convtranspose_1d"),
                                                   ORT_TSTR("convtranspose_3d"),
                                                   ORT_TSTR("maxpool_2d_uint8"),
                                                   ORT_TSTR("mul_uint8"),
                                                   ORT_TSTR("div_uint8")};
  static const ORTCHAR_T* tensorrt_disabled_tests[] = {
      ORT_TSTR("udnie"),
      ORT_TSTR("rain_princess"),
      ORT_TSTR("pointilism"),
      ORT_TSTR("mosaic"),
      ORT_TSTR("LSTM_Seq_lens_unpacked"),
      ORT_TSTR("cgan"),
      ORT_TSTR("candy"),
      ORT_TSTR("tinyyolov3"),
      ORT_TSTR("yolov3"),
      ORT_TSTR("mlperf_ssd_resnet34_1200"),
      ORT_TSTR("mlperf_ssd_mobilenet_300"),
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
      ORT_TSTR("size")  // INVALID_ARGUMENT: Cannot find binding of given name: x
  };
  for (const ORTCHAR_T* provider_name : provider_names) {
    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests),
                                                                        std::end(immutable_broken_tests));
    if (CompareCString(provider_name, ORT_TSTR("cuda")) == 0) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("dml")) == 0) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("dnnl")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("tensorrt")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(tensorrt_disabled_tests), std::end(tensorrt_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("openvino")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(openvino_disabled_tests), std::end(openvino_disabled_tests));
    }

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
                                                    ORT_TSTR("zfnet512"),
                                                    ORT_TSTR("ResNet101_DUC_HDC"),
                                                    ORT_TSTR("ResNet101_DUC_HDC-12"),
                                                    ORT_TSTR("FCN ResNet-101"),
                                                    ORT_TSTR("SSD")};
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif
    // fp16 models have different outputs with different kinds of hardware. We need to disable all fp16 models
    all_disabled_tests.insert(ORT_TSTR("fp16_shufflenet"));
    all_disabled_tests.insert(ORT_TSTR("fp16_inception_v1"));
    all_disabled_tests.insert(ORT_TSTR("fp16_tiny_yolov2"));
    std::vector<std::basic_string<ORTCHAR_T>> paths;
#if defined(NDEBUG) || defined(RUN_MODELTEST_IN_DEBUG_MODE)
#ifdef _WIN32
    paths.push_back(ORT_TSTR("..\\models"));
#else
    paths.push_back(ORT_TSTR("../models"));
#endif
#endif

// TENSORRT/OpenVino has too many test failures in the single node tests
#if !defined(USE_OPENVINO)
#if !defined(_WIN32)
    paths.push_back(ORT_TSTR("/data/onnx"));
#else
    paths.push_back(ORT_TSTR("c:\\local\\data\\onnx"));
#endif
#endif

    while (!paths.empty()) {
      std::basic_string<ORTCHAR_T> node_data_root_path = paths.back();
      paths.pop_back();
      std::basic_string<ORTCHAR_T> my_dir_name = GetLastComponent(node_data_root_path);
      ORT_TRY {
        LoopDir(node_data_root_path, [&](const ORTCHAR_T* filename, OrtFileType f_type) -> bool {
          if (filename[0] == ORT_TSTR('.'))
            return true;
          if (f_type == OrtFileType::TYPE_DIR) {
            std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename);
            paths.push_back(p);
            return true;
          }
          std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
          if (!HasExtensionOf(filename_str, ORT_TSTR("onnx")))
            return true;
          std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
          if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0)
            test_case_name = test_case_name.substr(5);
          if (all_disabled_tests.find(test_case_name) != all_disabled_tests.end())
            return true;

#ifdef DISABLE_ML_OPS
          auto starts_with = [](const std::basic_string<PATH_CHAR_TYPE>& find_in,
                                const std::basic_string<PATH_CHAR_TYPE>& find_what) {
            return find_in.compare(0, find_what.size(), find_what) == 0;
          };
          if (starts_with(test_case_name, ORT_TSTR("XGBoost_")) || starts_with(test_case_name, ORT_TSTR("coreml_")) ||
              starts_with(test_case_name, ORT_TSTR("scikit_")) || starts_with(test_case_name, ORT_TSTR("libsvm_"))) {
            return true;
          }
#endif
          std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename_str);
          std::basic_string<PATH_CHAR_TYPE> r = provider_name;
          r.append(ORT_TSTR("_")).append(p);
          v.emplace_back(r);
          return true;
        });
      }
      ORT_CATCH(const std::exception&) {
      }  // ignore non-exist dir
    }
  }
  return v;
}

auto ExpandModelName = [](const ::testing::TestParamInfo<ModelTest::ParamType>& info) {
  // use info.param here to generate the test suffix
  std::basic_string<ORTCHAR_T> name = info.param;

  // the original name here is the combination of provider name and model path name
  // remove the trailing 'xxxxxxx/model.onnx' of name
  if (name.size() > 11 && name.substr(name.size() - 11) == ORT_TSTR("/model.onnx")) {
    name = name.substr(0, info.param.size() - 11);
  }
  // remove the trailing 'xxxxxx.onnx' of name
  else if (name.size() > 5 && name.substr(name.size() - 5) == ORT_TSTR(".onnx")) {
    name = name.substr(0, info.param.size() - 5);
  }

  // Note: test name only accepts '_' and alphanumeric
  // replace '/' or '\' with '_'
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '\\', '_');

  // in case there's whitespace in directory name
  std::replace(name.begin(), name.end(), ' ', '_');

  // Note: test name only accepts '_' and alphanumeric
  // remove '.', '-', ':'
  char chars[] = ".-:()";
  for (unsigned int i = 0; i < strlen(chars); ++i) {
    name.erase(std::remove(name.begin(), name.end(), chars[i]), name.end());
  }
#ifdef _WIN32
  // Note: The return value of INSTANTIATE_TEST_SUITE_P accepts std::basic_string<char...>.
  // Need conversion of wchar_t to char.
  return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(name);
#else
  return name;
#endif
};

// The optional last argument is a function or functor that generates custom test name suffixes based on the test
// parameters. Specify the last argument to make test name more meaningful and clear instead of just the sequential
// number.
INSTANTIATE_TEST_SUITE_P(ModelTests, ModelTest, testing::ValuesIn(GetParameterStrings()), ExpandModelName);

}  // namespace test
}  // namespace onnxruntime
