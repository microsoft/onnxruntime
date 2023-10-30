#include "DisabledTests.h"

namespace onnxruntime {
namespace test {

std::unique_ptr<std::set<BrokenTest>> GetBrokenTests(const std::string& provider_name) {
  auto broken_tests = std::make_unique<std::set<BrokenTest>>(std::initializer_list<BrokenTest>{
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
      {"gridsample_volumetric_nearest_align_corners_0", "result differs", {}},
      {"gridsample_volumetric_nearest_align_corners_1", "result differs", {}},
      {"reduce_l1_empty_set", "unknown version", {}},
      {"reduce_l1_empty_set_expanded", "unknown version", {}},
      {"reduce_l2_empty_set", "unknown version", {}},
      {"reduce_l2_empty_set_expanded", "unknown version", {}},
      {"reduce_log_sum_empty_set", "unknown version", {}},
      {"reduce_log_sum_empty_set_expanded", "unknown version", {}},
      {"reduce_log_sum_exp_empty_set", "unknown version", {}},
      {"reduce_log_sum_exp_empty_set_expanded", "unknown version", {}},
      {"reduce_prod_empty_set", "unknown version", {}},
      {"reduce_sum_empty_set", "unknown version", {}},
      {"reduce_sum_square_empty_set", "unknown version", {}},
      {"reduce_sum_square_empty_set_expanded", "unknown version", {}},
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
      {"mask_rcnn_keras", "this model currently has an invalid contrib op version set to 10", {}}});

  // Some EPs may fail to pass some specific testcases.
  // For example TenosrRT EP may fail on FLOAT16 related testcases if GPU doesn't support float16.
  // Instead of list all these testcases, we can use following keyword set to filter out testcases wchich contain
  // specific keyword.
  // std::set<std::string> broken_tests_keyword_set = {};

  if (provider_name == "cuda") {
#ifdef _WIN32
    broken_tests->insert({"LSTM_Seq_lens_unpacked", "this test fails with new image since Aug 25."});
    broken_tests->insert({"bidaf", "this test fails with new image since Aug 25."});
    broken_tests->insert({"Candy", "Flaky test, need to investigate", {"opset9"}});
#else
    broken_tests->insert({"bidaf", "this test should be recovered when multi-gpu pipeline deprecates NV12", {"opset9"}});
#endif
  }

  if (provider_name == "nnapi") {
    broken_tests->insert({"scan9_sum", "Error with the extra graph"});
    broken_tests->insert({"scan_sum", "Error with the extra graph"});
    broken_tests->insert({"mvn_expanded", "Failed to find kernel for MemcpyFromHost(1) (node Memcpy_1)"});
    broken_tests->insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"gemm_transposeB", "Temporarily disabled pending investigation"});
    broken_tests->insert({"range_float_type_positive_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"range_int32_type_negative_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"convtranspose_1d", "1d convtranspose not supported yet"});
    broken_tests->insert({"convtranspose_3d", "3d convtranspose not supported yet"});
    broken_tests->insert({"maxpool_2d_uint8", "result mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NC_expanded", "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded", "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded", "shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded", "shape mismatch"});
    // Disable based on George Wu's recommendation.
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded",
         "shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NC", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                          "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob_expanded",
                          "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "Shape mismatch"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_none_weights_log_prob_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_expanded", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_log_prob", "Shape mismatch"});
    broken_tests->insert({"softmax_cross_entropy_sum_log_prob_expanded", "Shape mismatch"});
  }

  if (provider_name == "tensorrt") {
    broken_tests->insert({"convtranspose_with_kernel", "It causes segmentation fault"});
    broken_tests->insert({"convtranspose_pad", "It causes segmentation fault"});
    broken_tests->insert({"convtranspose_kernel_shape", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_expanded", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "It causes segmentation fault"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "It causes segmentation fault"});

    broken_tests->insert({"basic_conv_with_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});
    broken_tests->insert({"basic_conv_without_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});
    broken_tests->insert({"conv_with_strides_no_padding",
                          "Cannot set more than one input unless network has Q/DQ layers. TensorRT EP could not build "
                          "engine for fused node"});

    broken_tests->insert({"conv_with_autopad_same",
                          "Internal Error (node_of_y: Cannot set more than one input unless network has Q/DQ layers.)"});

    // unsupported tests since opset16
    broken_tests->insert({"sequence_map_add_2_sequences", "not supported by TensorRT EP"});
    broken_tests->insert({"sequence_map_extract_shapes", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_add_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_1_sequence", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_2_sequences", "not supported by TensorRT EP."});
    broken_tests->insert({"sequence_map_identity_1_sequence_1_tensor", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_default_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"leakyrelu_example_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"prelu_broadcast_expanded", "not supported by TensorRT EP."});
    broken_tests->insert({"prelu_example_expanded", "not supported by TensorRT EP."});
  }

  if (provider_name == "dml") {
    broken_tests->insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests->insert({"PixelShuffle", "Test requires 6D Reshape, which isn't supported by DirectML"});
    broken_tests->insert({"operator_permute2", "Test requires 6D Transpose, which isn't supported by DirectML"});
    broken_tests->insert({"resize_downsample_linear",
                          "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests->insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests->insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});

    // These tests are temporarily disabled pending investigation
    broken_tests->insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests->insert({"mxnet_arcface", "Temporarily disabled pending investigation"});
    broken_tests->insert({"yolov3", "Temporarily disabled pending investigation"});
    broken_tests->insert({"tf_inception_v2", "Temporarily disabled pending investigation"});
    broken_tests->insert({"fp16_inception_v1", "Temporarily disabled pending investigation"});
    broken_tests->insert({"candy", "Temporarily disabled pending investigation"});
    broken_tests->insert({"BERT_Squad", "Temporarily disabled pending investigation"});
    broken_tests->insert({"LSTM_Seq_lens_unpacked", "The parameter is incorrect"});
    broken_tests->insert({"mlperf_ssd_resnet34_1200", "The parameter is incorrect"});

    broken_tests->insert({"resize_downsample_scales_linear",
                          "DML uses half_pixel and this test assumed \"asymmetric\" but does not include \"mode\""});
    broken_tests->insert({"resize_downsample_sizes_linear_pytorch_half_pixel",
                          "DML does not support downsampling by such a large factor - skips input pixels"});
    broken_tests->insert({"resize_downsample_sizes_nearest",
                          "DML uses pixel centers for nearest, rounding 1 value off for the middle column"});
    broken_tests->insert({"resize_upsample_sizes_nearest",
                          "DML uses pixel centers for nearest, which makes more sense (the 3rd row mismatches)"});
    broken_tests->insert({"unsqueeze_three_axes", "DML does not support 6D tensors"});
    broken_tests->insert({"unsqueeze_unsorted_axes", "DMLdoes not support 6D tensors"});

    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob",
                          "DML does not support 5D+ tensors"});
    broken_tests->insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded",
                          "DML does not support 5D+ tensors"});
  }

  if (provider_name == "qnn") {
    broken_tests->insert({"gemm_default_no_bias", "result differs"});
    broken_tests->insert({"resize_downsample_scales_linear", "result differs"});
    broken_tests->insert({"resize_downsample_scales_linear_antialias", "result differs"});
    broken_tests->insert({"resize_downsample_sizes_linear_antialias", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_expanded", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_NCd1_mean_weight_negative_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean", "result differs"});
    broken_tests->insert({"sce_mean_3d", "result differs"});
    broken_tests->insert({"sce_mean_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_expanded", "result differs"});
    broken_tests->insert({"sce_mean_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_4d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_expanded", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_no_weight_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight", "result differs"});
    broken_tests->insert({"sce_mean_weight_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_3d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_4d_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_ii_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_mean_weight_log_prob", "result differs"});
    broken_tests->insert({"sce_mean_weight_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_none", "result differs"});
    broken_tests->insert({"sce_none_expanded", "result differs"});
    broken_tests->insert({"sce_none_log_prob", "result differs"});
    broken_tests->insert({"sce_none_log_prob_expanded", "result differs"});
    broken_tests->insert({"sce_sum", "result differs"});
    broken_tests->insert({"sce_sum_expanded", "result differs"});
    broken_tests->insert({"sce_sum_log_prob", "result differs"});
    broken_tests->insert({"sce_sum_log_prob_expanded", "result differs"});
    broken_tests->insert({"gridsample_reflection_padding", "result differs"});
    broken_tests->insert({"spacetodepth", "result differs"});
  }

#ifdef DISABLE_CONTRIB_OPS
  broken_tests->insert({"coreml_SqueezeNet_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Permute_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_ReLU_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Padding-Upsampling-Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"fp16_tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Pooling_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Padding_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet_small", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet_large", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_linear_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_leakyrelu_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_hard_sigmoid_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_elu_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Dense_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Conv2D_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_VGG16_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_Resnet50_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_Inceptionv3_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_FNS-Candy_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"coreml_AgeNet_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_ImageNet_large", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_ImageNet_small", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_thresholdedrelu_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu_default", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_default_axes", "This model uses contrib ops."});
  broken_tests->insert({"thresholdedrelu_example", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_neg failed", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_start_out_of_bounds", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_end_out_of_bounds", "This model uses contrib ops."});
  broken_tests->insert({"dynamic_slice_neg", "This model uses contrib ops."});
  broken_tests->insert({"mvn", "This model uses contrib ops.", {"onnx130"}});
  broken_tests->insert({"cdist_float32_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float32_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests->insert({"cdist_float64_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"bidaf", "This model uses contrib ops."});
  broken_tests->insert({"fp16_test_tiny_yolov2", "This model uses contrib ops."});
  broken_tests->insert({"fp16_coreml_FNS-Candy", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Repeat_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_BiDirectional_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"fp16_coreml_LinearRegression_NYCTaxi", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_GRU_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_SimpleRNN_ImageNet", "This model uses contrib ops."});
  broken_tests->insert({"keras2coreml_Dot_imageNet", "This model uses contrib ops."});
#endif
  return broken_tests;
}

// Some EPs may fail to pass some specific testcases.
// For example TenosrRT EP may fail on FLOAT16 related testcases if GPU doesn't support float16.
// Instead of list all these testcases, we can use following keyword set to filter out testcases wchich contain
// specific keyword.
std::unique_ptr<std::set<std::string>> GetBrokenTestsKeyWordSet(const std::string& provider_name) {
  auto broken_tests_keyword_set = std::make_unique<std::set<std::string>>();
  if (provider_name == "tensorrt") {
    broken_tests_keyword_set->insert({"scatternd_add"});
    broken_tests_keyword_set->insert({"scatternd_multiply"});
    broken_tests_keyword_set->insert({"scatter_elements_with_duplicate_indices"});

    // sce op is not supported
    broken_tests_keyword_set->insert({"sce"});

    // TensorRT EP CI uses Nvidia Tesla M60 which doesn't support fp16.
    broken_tests_keyword_set->insert({"FLOAT16"});
  }
  return broken_tests_keyword_set;
}

std::unique_ptr<std::unordered_set<std::basic_string<ORTCHAR_T>>> GetAllDisabledTests(const std::basic_string_view<ORTCHAR_T>& provider_name) {
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
    static const ORTCHAR_T* qnn_disabled_tests[] = {
        ORT_TSTR("nllloss_NCd1d2d3_none_no_weight_negative_ii"),
        ORT_TSTR("nllloss_NCd1d2d3_none_no_weight_negative_ii_expanded"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii_expanded"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii_log_prob"),
        ORT_TSTR("sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded"),
        ORT_TSTR("gather_negative_indices"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded"),
        ORT_TSTR("nllloss_NCd1d2_with_weight"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_expanded"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum_expanded"),
        ORT_TSTR("nllloss_NCd1d2_with_weight_reduction_sum_ii"),
        ORT_TSTR("nllloss_NCd1_weight_ii_expanded"),
        ORT_TSTR("nllloss_NCd1_ii_expanded"),
        ORT_TSTR("nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded"),
        ORT_TSTR("sce_none_weights"),
        ORT_TSTR("sce_none_weights_log_prob"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii_log_prob"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii_log_prob_expanded"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii"),
        ORT_TSTR("sce_NCd1d2d3_sum_weight_high_ii_expanded"),
        ORT_TSTR("sce_none_weights_log_prob_expanded"),
        ORT_TSTR("sce_none_weights_expanded")};
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

    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests),
                                                                        std::end(immutable_broken_tests));
    if (provider_name == provider_name_cuda) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    } else if (provider_name == provider_name_dml) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    } else if (provider_name == provider_name_dnnl) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
    } else if (provider_name == provider_name_tensorrt) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(tensorrt_disabled_tests), std::end(tensorrt_disabled_tests));
    } else if (provider_name == provider_name_openvino) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(openvino_disabled_tests), std::end(openvino_disabled_tests));
    } else if (provider_name == provider_name_qnnl) {
      all_disabled_tests.insert(std::begin(qnn_disabled_tests), std::end(qnn_disabled_tests));
      all_disabled_tests.insert(std::begin(float8_tests), std::end(float8_tests));
    }

#if !defined(__amd64__) && !defined(_M_AMD64)
    // out of memory
    static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("BERT_Squad"),
                                                    ORT_TSTR("bvlc_alexnet"),
                                                    ORT_TSTR("bvlc_reference_caffenet"),
                                                    ORT_TSTR("coreml_VGG16_ImageNet"),
                                                    ORT_TSTR("VGG 16-fp32"),
                                                    ORT_TSTR("VGG 19-caffe2"),
                                                    ORT_TSTR("VGG 19-bn"),
                                                    ORT_TSTR("VGG 16-bn"),
                                                    ORT_TSTR("VGG 19"),
                                                    ORT_TSTR("VGG 16"),
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
    return std::make_unique<std::unordered_set<std::basic_string<ORTCHAR_T>>>(all_disabled_tests);
}

}  // namespace test
}  // namespace onnxruntime
