// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>
#include <unordered_map>

#include "core/common/span_utils.h"
#include "core/common/float16.h"
#include "core/framework/int4.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/unittest_util/qdq_test_utils.h"

#include "gtest/gtest.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif  // #if defined(_MSC_VER)

struct QDQOpKeys {
  const char* quantize_linear;
  const char* dequantize_linear;
};

constexpr QDQOpKeys GetQDQOpKeys(bool use_contrib_qdq) {
  if (use_contrib_qdq) {
    return {"com.microsoft.QuantizeLinear", "com.microsoft.DequantizeLinear"};
  }
  return {"QuantizeLinear", "DequantizeLinear"};
}

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

// Input1   Input2
//   |        |
//    \      DQ
//     \    /
//     MatMul
//       |
//     output
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQMatMulNotConverted_NonConstDQ(const std::vector<int64_t>& input1_shape,
                                   const std::vector<int64_t>& input2_shape,
                                   const int64_t axis,
                                   const int64_t block_size,
                                   int64_t accuracy_level,
                                   std::unique_ptr<IExecutionProvider> ep = nullptr) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* input2_arg = builder.MakeInput(input2_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);

    auto scale_shape = std::vector<int64_t>{input2_shape};
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;
    auto* scale_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {input2_arg, scale_arg, zp_arg}, {dq_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {input2_arg, scale_arg}, {dq_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {input1_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn,
                    {},
                    ep ? std::move(ep) : nullptr);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_NonConstDQ) {
  // DQ contrib op schema is not updated to support blocked quantization
  // Rejection doesn't depend on type/zp/accuracy_level — keep representative combos only.
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_NonConstDQ_Cuda) {
  // DQ contrib op schema is not updated to support blocked quantization
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
}

//        Input2
//           |
//    DQ     /
//     \    /
//     MatMul
//       |
//     output
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQMatMulNotConverted_FirstDQInput(const std::vector<int64_t>& weight_shape,
                                     const std::vector<int64_t>& input2_shape,
                                     const int64_t axis,
                                     const int64_t block_size,
                                     int64_t accuracy_level,
                                     std::unique_ptr<IExecutionProvider> ep = nullptr) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* input2_arg = builder.MakeInput(input2_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);

    std::vector<int64_t> scale_shape = weight_shape;
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;
    auto* scale_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {dq_output, input2_arg}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn,
                    {},
                    ep ? std::move(ep) : nullptr);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_FirstDQInput) {
  // DQ contrib op schema is not updated to support blocked quantization
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_FirstDQInput_Cuda) {
  // DQ contrib op schema is not updated to support blocked quantization
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
}

// Input1
//   |
//    \      DQ
//     \    /
//     MatMul
//       |
//     output
template <typename T, bool use_zp>
void RunDQMatMulNotConverted_TypeShapeMismatch(const std::vector<int64_t>& input1_shape,
                                               const std::vector<int64_t>& weight_shape,
                                               const int64_t axis,
                                               const int64_t block_size,
                                               int64_t accuracy_level,
                                               std::unique_ptr<IExecutionProvider> ep = nullptr) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();
    NodeArg* weight_arg = nullptr;

    // add DQ
    if constexpr (std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>) {
      weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    } else {
      weight_arg = builder.MakeInitializer(weight_shape,
                                           std::numeric_limits<T>::min(),
                                           std::numeric_limits<T>::max());
    }

    auto* dq_output = builder.MakeIntermediate();
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);

    std::vector<int64_t> scale_shape = weight_shape;
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;
    auto* scale_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      NodeArg* zp_arg;
      if constexpr (std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>) {
        zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      } else {
        zp_arg = builder.MakeInitializer<T>(scale_shape, 0, 2);
      }

      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn,
                    {},
                    ep ? std::move(ep) : nullptr);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_TypeMismatch) {
  // int8/uint8 are now converted (8-bit support added), so only 16-bit and 32-bit remain as type mismatches
  RunDQMatMulNotConverted_TypeShapeMismatch<int16_t, true>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<int16_t, false>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<uint16_t, true>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<uint16_t, false>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<int32_t, false>({12, 37}, {37, 12}, 0, 16, 0);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_ShapeMismatch) {
  // One representative type combo per rejection scenario (type doesn't affect rejection logic).
  // block size too small
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 8, 0);
  // block size not 2's power
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 17, 0);
  // not axis 0
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 37}, 1, 16, 0);
  // not rank 2
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({2, 12, 37}, {2, 37, 12}, 0, 16, 0);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_ShapeMismatch_Cuda) {
  // One representative type combo per rejection scenario.
  // block size too small
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 8, 0, DefaultCudaExecutionProvider());
  // block size not 2's power
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 17, 0, DefaultCudaExecutionProvider());
  // not axis 0
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 37}, 1, 16, 0, DefaultCudaExecutionProvider());
  // not rank 2
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({2, 12, 37}, {2, 37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
}

// Two MatMul nodes sharing the SAME weight and scale initializers (tied embedding pattern).
// Regression test for issue #28306: the second DQ->MatMul fusion used to crash with
// "Missing required scale" because the first fusion consumed the shared initializer.
// Both DQ nodes should be rejected from fusion when weight or scale is shared.
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQMatMulNotConverted_SharedWeight(const std::vector<int64_t>& input_shape,
                                     const std::vector<int64_t>& weight_shape,
                                     const int64_t block_size,
                                     int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput(input_shape, -100.0f, 100.0f);
    auto* input2_arg = builder.MakeInput(input_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);

    auto scale_shape = std::vector<int64_t>{weight_shape};
    scale_shape[0] = (scale_shape[0] + block_size - 1) / block_size;

    // Both DQ nodes share the SAME weight and scale initializers (tied embedding).
    auto* shared_weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* shared_scales_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);

    auto* dq1_output = builder.MakeIntermediate();
    auto* dq2_output = builder.MakeIntermediate();

    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg, zp_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg, zp_arg}, {dq2_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg}, {dq2_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {input1_arg, dq1_output}, {output_arg});
    // Use a second graph output so the second MatMul is not pruned as dead code.
    auto* output2_arg = builder.MakeOutput();
    builder.AddNode("MatMul", {input2_arg, dq2_output}, {output2_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    // Fusion must NOT happen: shared initializers prevent safe fusion.
    EXPECT_EQ(op_to_count["MatMul"], 2);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_SharedWeight) {
  RunDQMatMulNotConverted_SharedWeight<UInt4x2, false>({12, 12}, {12, 37}, 16, 0);
  RunDQMatMulNotConverted_SharedWeight<Int4x2, false>({12, 12}, {12, 37}, 16, 0);
  RunDQMatMulNotConverted_SharedWeight<UInt4x2, true>({12, 12}, {12, 37}, 16, 0);
  RunDQMatMulNotConverted_SharedWeight<Int4x2, true>({12, 12}, {12, 37}, 16, 0);
}

// Two Gemm nodes sharing the SAME weight and scale initializers (tied embedding pattern).
// Regression test for the Gemm path of issue #28306: both DQ->Gemm fusions should be
// rejected when weight or scale is shared. Bias initializers are unshared (safe to share,
// but using distinct ones here keeps the test focused on weight/scale sharing).
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQGemmNotConverted_SharedWeight(const std::vector<int64_t>& input_shape,
                                   const std::vector<int64_t>& weight_shape,
                                   const int64_t block_size,
                                   int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput(input_shape, -100.0f, 100.0f);
    auto* input2_arg = builder.MakeInput(input_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);

    auto scale_shape = std::vector<int64_t>{weight_shape};
    scale_shape[0] = (scale_shape[0] + block_size - 1) / block_size;

    // Both DQ nodes share the SAME weight and scale initializers (tied embedding).
    auto* shared_weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* shared_scales_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);

    auto* dq1_output = builder.MakeIntermediate();
    auto* dq2_output = builder.MakeIntermediate();

    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg, zp_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg, zp_arg}, {dq2_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {shared_weight_arg, shared_scales_arg}, {dq2_output}, "", &attrs);
    }

    // Each Gemm has its own unshared bias initializer.
    int64_t N = weight_shape[1];
    auto* bias1_arg = builder.MakeInitializer<float>({N}, std::vector<float>(static_cast<size_t>(N), 0.5f));
    auto* bias2_arg = builder.MakeInitializer<float>({N}, std::vector<float>(static_cast<size_t>(N), 0.5f));

    NodeAttributes gemm_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("alpha", 1.0f), gemm_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("transA", static_cast<int64_t>(0)), gemm_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("transB", static_cast<int64_t>(0)), gemm_attrs);

    builder.AddNode("Gemm", {input1_arg, dq1_output, bias1_arg}, {output_arg}, "", &gemm_attrs);
    // Use a second graph output so the second Gemm is not pruned as dead code.
    auto* output2_arg = builder.MakeOutput();
    builder.AddNode("Gemm", {input2_arg, dq2_output, bias2_arg}, {output2_arg}, "", &gemm_attrs);
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    // Fusion must NOT happen: shared initializers prevent safe fusion.
    EXPECT_EQ(op_to_count["Gemm"], 2);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQGemmNotConvertedToMatMulNBits_SharedWeight) {
  RunDQGemmNotConverted_SharedWeight<UInt4x2, false>({12, 12}, {12, 37}, 16, 0);
  RunDQGemmNotConverted_SharedWeight<Int4x2, false>({12, 12}, {12, 37}, 16, 0);
  RunDQGemmNotConverted_SharedWeight<UInt4x2, true>({12, 12}, {12, 37}, 16, 0);
  RunDQGemmNotConverted_SharedWeight<Int4x2, true>({12, 12}, {12, 37}, 16, 0);
}

//  Input1
//    |      DQ
//     \    /
//     MatMul
//       |      DQ
//        \    /
//        MatMul
//          |
//        output
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQMatMulConverted(const std::vector<int64_t>& input1_shape,
                     const std::vector<int64_t>& weight1_shape,
                     const std::vector<int64_t>& weight2_shape,
                     const int64_t axis,
                     const int64_t block_size,
                     int64_t accuracy_level,
                     std::unique_ptr<IExecutionProvider> ep = nullptr) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    // add DQ
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
    auto scale1_shape = std::vector<int64_t>{weight1_shape};
    auto scale2_shape = std::vector<int64_t>{weight2_shape};
    scale1_shape[axis] = (scale1_shape[axis] + block_size - 1) / block_size;
    scale2_shape[axis] = (scale2_shape[axis] + block_size - 1) / block_size;

    auto* weight1_arg = builder.MakeInitializer(weight1_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* weight2_arg = builder.MakeInitializer(weight2_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq1_output = builder.MakeIntermediate();
    auto* dq2_output = builder.MakeIntermediate();
    auto* matmul1_output = builder.MakeIntermediate();

    auto* scales1_arg = builder.MakeInitializer(scale1_shape, 8.0f, 12.0f);
    auto* scales2_arg = builder.MakeInitializer(scale2_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto* zp1_arg = builder.MakeInitializer(scale1_shape, T(0, 0), T(2, 0));
      auto* zp2_arg = builder.MakeInitializer(scale2_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {weight1_arg, scales1_arg, zp1_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {weight2_arg, scales2_arg, zp2_arg}, {dq2_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight1_arg, scales1_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {weight2_arg, scales2_arg}, {dq2_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {input_arg, dq1_output}, {matmul1_output});
    builder.AddNode("MatMul", {matmul1_output, dq2_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 2);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    2e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn,
                    {},
                    ep ? std::move(ep) : nullptr);
}

TEST(QDQTransformerTests, DQMatMulConvertedToMatMulNBits) {
  // DQ contrib op schema is not updated to support blocked quantization
  RunDQMatMulConverted<Int4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulConverted<Int4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulConverted<UInt4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulConverted<UInt4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulConverted<Int4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulConverted<Int4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulConverted<UInt4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulConverted<UInt4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1);
}

// 8-bit DQ -> MatMul conversion to MatMulNBits(bits=8)
//  Input1
//    |      DQ(int8/uint8)
//     \    /
//     MatMul
//       |      DQ(int8/uint8)
//        \    /
//        MatMul
//          |
//        output
template <typename T, bool use_zp>
void RunDQMatMulConverted_8bit(const std::vector<int64_t>& input1_shape,
                               const std::vector<int64_t>& weight1_shape,
                               const std::vector<int64_t>& weight2_shape,
                               const int64_t axis,
                               const int64_t block_size,
                               int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    // add DQ
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
    auto scale1_shape = std::vector<int64_t>{weight1_shape};
    auto scale2_shape = std::vector<int64_t>{weight2_shape};
    scale1_shape[axis] = (scale1_shape[axis] + block_size - 1) / block_size;
    scale2_shape[axis] = (scale2_shape[axis] + block_size - 1) / block_size;

    auto* weight1_arg = builder.MakeInitializer<T>(weight1_shape,
                                                   std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());
    auto* weight2_arg = builder.MakeInitializer<T>(weight2_shape,
                                                   std::numeric_limits<T>::min(),
                                                   std::numeric_limits<T>::max());
    auto* dq1_output = builder.MakeIntermediate();
    auto* dq2_output = builder.MakeIntermediate();
    auto* matmul1_output = builder.MakeIntermediate();

    auto* scales1_arg = builder.MakeInitializer(scale1_shape, 0.01f, 0.05f);
    auto* scales2_arg = builder.MakeInitializer(scale2_shape, 0.01f, 0.05f);
    if constexpr (use_zp) {
      auto* zp1_arg = builder.MakeInitializer<T>(scale1_shape,
                                                 static_cast<T>(0), static_cast<T>(2));
      auto* zp2_arg = builder.MakeInitializer<T>(scale2_shape,
                                                 static_cast<T>(0), static_cast<T>(2));
      builder.AddNode("DequantizeLinear", {weight1_arg, scales1_arg, zp1_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {weight2_arg, scales2_arg, zp2_arg}, {dq2_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight1_arg, scales1_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {weight2_arg, scales2_arg}, {dq2_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {input_arg, dq1_output}, {matmul1_output});
    builder.AddNode("MatMul", {matmul1_output, dq2_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 2);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-4 /*per_sample_tolerance*/,
                    1e-4 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

// 2-bit DQ -> MatMul conversion to MatMulNBits(bits=2)
//  Input1
//    |      DQ(int2/uint2)
//     \    /
//     MatMul
//       |      DQ(int2/uint2)
//        \    /
//        MatMul
//          |
//        output
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int2x4> || std::is_same_v<T, UInt2x4>, void>::type
RunDQMatMulConverted_2bit(const std::vector<int64_t>& input1_shape,
                          const std::vector<int64_t>& weight1_shape,
                          const std::vector<int64_t>& weight2_shape,
                          const int64_t axis,
                          const int64_t block_size,
                          int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
    auto scale1_shape = std::vector<int64_t>{weight1_shape};
    auto scale2_shape = std::vector<int64_t>{weight2_shape};
    scale1_shape[axis] = (scale1_shape[axis] + block_size - 1) / block_size;
    scale2_shape[axis] = (scale2_shape[axis] + block_size - 1) / block_size;

    auto* weight1_arg = builder.MakeInitializer(weight1_shape,
                                                T(T::min_val, T::min_val, T::min_val, T::min_val),
                                                T(T::max_val, T::max_val, T::max_val, T::max_val));
    auto* weight2_arg = builder.MakeInitializer(weight2_shape,
                                                T(T::min_val, T::min_val, T::min_val, T::min_val),
                                                T(T::max_val, T::max_val, T::max_val, T::max_val));
    auto* dq1_output = builder.MakeIntermediate();
    auto* dq2_output = builder.MakeIntermediate();
    auto* matmul1_output = builder.MakeIntermediate();

    auto* scales1_arg = builder.MakeInitializer(scale1_shape, 8.0f, 12.0f);
    auto* scales2_arg = builder.MakeInitializer(scale2_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto* zp1_arg = builder.MakeInitializer(scale1_shape, T(0, 0, 0, 0), T(1, 1, 1, 1));
      auto* zp2_arg = builder.MakeInitializer(scale2_shape, T(0, 0, 0, 0), T(1, 1, 1, 1));
      builder.AddNode("DequantizeLinear", {weight1_arg, scales1_arg, zp1_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {weight2_arg, scales2_arg, zp2_arg}, {dq2_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight1_arg, scales1_arg}, {dq1_output}, "", &attrs);
      builder.AddNode("DequantizeLinear", {weight2_arg, scales2_arg}, {dq2_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {input_arg, dq1_output}, {matmul1_output});
    builder.AddNode("MatMul", {matmul1_output, dq2_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 2);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    25 /*opset_version*/,
                    1e-4 /*per_sample_tolerance*/,
                    1e-4 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulConvertedToMatMulNBits_2bit) {
  // 2-bit int2/uint2 DQ weights should be fused to MatMulNBits(bits=2)
  RunDQMatMulConverted_2bit<Int2x4, true>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
  RunDQMatMulConverted_2bit<Int2x4, false>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
  RunDQMatMulConverted_2bit<UInt2x4, true>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
  RunDQMatMulConverted_2bit<UInt2x4, false>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
}

TEST(QDQTransformerTests, DQMatMulConvertedToMatMulNBits_8bit) {
  // 8-bit int8/uint8 DQ weights should be fused to MatMulNBits(bits=8)
  RunDQMatMulConverted_8bit<int8_t, true>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
  RunDQMatMulConverted_8bit<int8_t, false>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
  RunDQMatMulConverted_8bit<uint8_t, true>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
  RunDQMatMulConverted_8bit<uint8_t, false>({12, 32}, {32, 16}, {16, 12}, 0, 16, 0);
  // block_size=32
  RunDQMatMulConverted_8bit<int8_t, true>({12, 32}, {32, 16}, {16, 12}, 0, 32, 0);
  RunDQMatMulConverted_8bit<uint8_t, true>({12, 32}, {32, 16}, {16, 12}, 0, 32, 0);
}

TEST(QDQTransformerTests, DQMatMulConvertedToMatMulNBits_Cuda) {
  // DQ contrib op schema is not updated to support blocked quantization
  RunDQMatMulConverted<Int4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulConverted<Int4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulConverted<UInt4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulConverted<UInt4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulConverted<Int4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulConverted<Int4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulConverted<UInt4x2, true>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulConverted<UInt4x2, false>({12, 12}, {12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
}

// DQ(fp16) -> MatMul fusion test
// Pattern: DQ(int4, fp16_scale) -> MatMul(fp16)
// For FP16 models on CPU EP, CPU EP doesn't claim FP16 MatMul during partitioning
// (no FP16 MatMul kernel on CPU), so the node's EP is empty "".
// The DQ->MatMul fusion should still match and fuse to MatMulNBits.
//
//  Input1(fp16)     DQ(int4->fp16)
//     \               /
//     MatMul(fp16)
//          |
//       output(fp16)
//
// After optimization:
//  Input1(fp16)  ->  MatMulNBits(fp16)  ->  output(fp16)
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQMatMulFP16Converted(const std::vector<int64_t>& input1_shape,
                         const std::vector<int64_t>& weight_shape,
                         const int64_t axis,
                         const int64_t block_size,
                         int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<MLFloat16>(input1_shape, MLFloat16(-1.0f), MLFloat16(1.0f));
    auto* output_arg = builder.MakeOutput();

    // DQ with fp16 scales
    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);

    std::vector<int64_t> scale_shape = weight_shape;
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;

    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* scale_arg = builder.MakeInitializer<MLFloat16>(scale_shape,
                                                         MLFloat16(0.01f), MLFloat16(0.05f));
    auto* dq_output = builder.MakeIntermediate();
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
    }

    // MatMul (fp16)
    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-2 /*per_sample_tolerance*/,
                    1e-2 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulFP16ConvertedToMatMulNBits) {
  // DQ(int4, fp16_scale) -> MatMul(fp16) should be fused to MatMulNBits
  RunDQMatMulFP16Converted<Int4x2, true>({12, 32}, {32, 16}, 0, 16, 0);
  RunDQMatMulFP16Converted<Int4x2, false>({12, 32}, {32, 16}, 0, 16, 0);
  RunDQMatMulFP16Converted<UInt4x2, true>({12, 32}, {32, 16}, 0, 16, 0);
  RunDQMatMulFP16Converted<UInt4x2, false>({12, 32}, {32, 16}, 0, 16, 0);
}

// Per-tensor DQ -> MatMul conversion to MatMulNBits
// DQ has scalar scale (and optional scalar zero-point), no block_size attribute.
//  Input1
//    |      DQ(per-tensor)
//     \    /
//     MatMul
//       |
//     output
template <typename T, bool use_zp>
void RunDQMatMulPerTensorConverted(const std::vector<int64_t>& input1_shape,
                                   const std::vector<int64_t>& weight_shape,
                                   int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq_output = builder.MakeIntermediate();

    // Scalar scale (per-tensor)
    auto* scale_arg = builder.MakeInitializer<float>({}, {10.0f});
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer<T>({}, std::vector<T>{T(1, 0)});
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output});
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output});
    }

    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    0.01 /*per_sample_tolerance - higher due to blockwise accumulation reordering*/,
                    5e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulPerTensorConvertedToMatMulNBits) {
  // Per-tensor: cover both types and a non-divisible K case.
  RunDQMatMulPerTensorConverted<Int4x2, true>({12, 32}, {32, 16}, 0);
  RunDQMatMulPerTensorConverted<UInt4x2, false>({12, 37}, {37, 16}, 0);
}

// Per-channel (axis=1) DQ -> MatMul conversion to MatMulNBits
// DQ has 1D scale shape [N], axis=1, no block_size attribute.
//  Input1
//    |      DQ(per-channel axis=1)
//     \    /
//     MatMul
//       |
//     output
template <typename T, bool use_zp>
void RunDQMatMulPerChannelConverted(const std::vector<int64_t>& input1_shape,
                                    const std::vector<int64_t>& weight_shape,
                                    int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq_output = builder.MakeIntermediate();

    int64_t N = weight_shape[1];
    // 1D scale shape [N] for per-channel (axis=1)
    auto* scale_arg = builder.MakeInitializer<float>({N}, 8.0f, 12.0f);

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(1)), attrs);

    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(std::vector<int64_t>{N}, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &attrs);
    }

    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulPerChannelConvertedToMatMulNBits) {
  RunDQMatMulPerChannelConverted<Int4x2, true>({12, 37}, {37, 16}, 0);
}

// Negative test: per-axis axis=0 with 1D scale should NOT fuse
template <typename T>
void RunDQMatMulPerAxisAxis0NotConverted(const std::vector<int64_t>& input1_shape,
                                         const std::vector<int64_t>& weight_shape) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq_output = builder.MakeIntermediate();

    int64_t K = weight_shape[0];
    // 1D scale shape [K] for per-axis axis=0 — should NOT match
    auto* scale_arg = builder.MakeInitializer<float>({K}, 8.0f, 12.0f);

    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), attrs);

    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &attrs);
    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  std::function<void(SessionOptions&)> add_session_options_fn = [](SessionOptions& sess_opts) {
    std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel, "0");
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulPerAxisAxis0NotConvertedToMatMulNBits) {
  RunDQMatMulPerAxisAxis0NotConverted<Int4x2>({12, 32}, {32, 16});
}

// Per-tensor DQ -> MatMul with configurable block_size session option
template <typename T, bool use_zp>
void RunDQMatMulPerTensorWithBlockSize(const std::vector<int64_t>& input1_shape,
                                       const std::vector<int64_t>& weight_shape,
                                       int64_t block_size_option) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq_output = builder.MakeIntermediate();

    auto* scale_arg = builder.MakeInitializer<float>({}, {10.0f});
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer<T>({}, std::vector<T>{T(1, 0)});
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output});
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output});
    }

    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);

    // Verify the MatMulNBits node has the expected block_size attribute
    for (const auto& node : session.GetGraph().Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        auto& attrs = node.GetAttributes();
        auto bs_iter = attrs.find("block_size");
        ASSERT_NE(bs_iter, attrs.end());
        int64_t expected_bs = block_size_option > 0 ? block_size_option : 32;  // default is 32
        EXPECT_EQ(bs_iter->second.i(), expected_bs);
      }
    }
  };

  std::function<void(SessionOptions&)> add_session_options_fn =
      [block_size_option](SessionOptions& sess_opts) {
        std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel, "0");
        std::ignore = sess_opts.config_options.AddConfigEntry(
            kOrtSessionOptionsQDQMatMulNBitsBlockSize,
            std::to_string(block_size_option).c_str());
      };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulPerTensorWithBlockSizeOption) {
  // Default block_size (0 -> 32)
  RunDQMatMulPerTensorWithBlockSize<Int4x2, false>({12, 32}, {32, 16}, 0);
  // Explicit block_size=16
  RunDQMatMulPerTensorWithBlockSize<Int4x2, true>({12, 32}, {32, 16}, 16);
}

// UINT8 per-tensor DQ -> MatMul -> MatMulNBits
// Tests shapes from real models including small dimensions (N=1, N=8).
template <bool use_zp>
void RunDQMatMulPerTensorUint8Converted(const std::vector<int64_t>& input1_shape,
                                        const std::vector<int64_t>& weight_shape,
                                        int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    auto* weight_arg = builder.MakeInitializer<uint8_t>(weight_shape, uint8_t(0), uint8_t(255));
    auto* dq_output = builder.MakeIntermediate();

    // Scalar scale (per-tensor)
    auto* scale_arg = builder.MakeInitializer<float>({}, {0.05f});
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer<uint8_t>({}, {uint8_t(128)});
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output});
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output});
    }

    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    0.01 /*per_sample_tolerance - higher due to blockwise accumulation reordering*/,
                    5e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQMatMulPerTensorUint8ConvertedToMatMulNBits) {
  RunDQMatMulPerTensorUint8Converted<true>({12, 96}, {96, 8}, 0);
}

// ---------------------------------------------------------------------------
// DQ -> Gemm tests for MatMulNBits fusion
// ---------------------------------------------------------------------------

//  Input1
//    |      DQ (4-bit weight)
//     \    /
//      Gemm
//        |
//      output
// Gemm has no bias, equivalent to MatMul. Should fuse to MatMulNBits.
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQGemmConvertedNoBias(const std::vector<int64_t>& input1_shape,
                         const std::vector<int64_t>& weight_shape,
                         const int64_t axis,
                         const int64_t block_size,
                         int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
    std::vector<int64_t> scale_shape = weight_shape;
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;

    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq_output = builder.MakeIntermediate();
    auto* scales_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {weight_arg, scales_arg, zp_arg}, {dq_output}, "", &dq_attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scales_arg}, {dq_output}, "", &dq_attrs);
    }

    builder.AddNode("Gemm", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["Gemm"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    2e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQGemmConvertedToMatMulNBits_NoBias) {
  RunDQGemmConvertedNoBias<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
}

//  Input1
//    |      DQ (4-bit weight)    bias (float)
//     \    /                    /
//      Gemm
//        |
//      output
// Gemm has a direct (non-DQ) float bias. Should fuse to MatMulNBits with bias at input 5.
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQGemmConvertedWithBias(const std::vector<int64_t>& input1_shape,
                           const std::vector<int64_t>& weight_shape,
                           const int64_t axis,
                           const int64_t block_size,
                           int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
    std::vector<int64_t> scale_shape = weight_shape;
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;

    int64_t N = weight_shape[1];
    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq_output = builder.MakeIntermediate();
    auto* scales_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {weight_arg, scales_arg, zp_arg}, {dq_output}, "", &dq_attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scales_arg}, {dq_output}, "", &dq_attrs);
    }

    auto* bias_arg = builder.MakeInitializer<float>({N}, std::vector<float>(static_cast<size_t>(N), 0.5f));
    builder.AddNode("Gemm", {input_arg, dq_output, bias_arg}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["Gemm"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    2e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQGemmConvertedToMatMulNBits_WithBias) {
  RunDQGemmConvertedWithBias<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
}

//  Input1
//    |      DQ (4-bit weight)    DQ (bias)
//     \    /                    /
//      Gemm
//        |
//      output
// Gemm has a bias from DQ. Weight DQ fused into MatMulNBits, bias DQ stays alive,
// bias DQ output wired to MatMulNBits input 5.
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQGemmConvertedWithDQBias(const std::vector<int64_t>& input1_shape,
                             const std::vector<int64_t>& weight_shape,
                             const int64_t axis,
                             const int64_t block_size,
                             int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    // Weight DQ
    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
    std::vector<int64_t> scale_shape = weight_shape;
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;

    int64_t N = weight_shape[1];
    auto* weight_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq_output = builder.MakeIntermediate();
    auto* scales_arg = builder.MakeInitializer(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto* zp_arg = builder.MakeInitializer(scale_shape, T(0, 0), T(2, 0));
      builder.AddNode("DequantizeLinear", {weight_arg, scales_arg, zp_arg}, {dq_output}, "", &dq_attrs);
    } else {
      builder.AddNode("DequantizeLinear", {weight_arg, scales_arg}, {dq_output}, "", &dq_attrs);
    }

    // Bias DQ (int8 quantized bias -> float)
    auto* bias_quantized = builder.MakeInitializer<int8_t>({N}, std::vector<int8_t>(static_cast<size_t>(N), 5));
    auto* bias_scale = builder.MakeInitializer<float>({}, std::vector<float>{0.1f});
    auto* bias_zp = builder.MakeInitializer<int8_t>({}, std::vector<int8_t>{0});
    auto* bias_dq_output = builder.MakeIntermediate();
    builder.AddNode("DequantizeLinear", {bias_quantized, bias_scale, bias_zp}, {bias_dq_output});

    builder.AddNode("Gemm", {input_arg, dq_output, bias_dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["Gemm"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    // Weight DQ removed, bias DQ stays
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  std::function<void(SessionOptions&)> add_session_options_fn{};
  if (accuracy_level >= 0) {
    add_session_options_fn = [accuracy_level](SessionOptions& sess_opts) {
      std::ignore = sess_opts.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                            std::to_string(accuracy_level).c_str());
    };
  }

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    2e-5 /*relative_per_sample_tolerance*/,
                    nullptr,
                    add_session_options_fn);
}

TEST(QDQTransformerTests, DQGemmConvertedToMatMulNBits_WithDQBias) {
  RunDQGemmConvertedWithDQBias<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
}

// Negative test: DQ -> Gemm with transB=1 should NOT be fused.
TEST(QDQTransformerTests, DQGemmNotConvertedToMatMulNBits_TransB) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({12, 37}, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    // With transB=1, Gemm transposes B at runtime: weight shape [N,K]=[12,37], transposed to [K,N]=[37,12].
    // DQ weight shape is [12,37] (N=12, K=37 after transpose).
    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", static_cast<int64_t>(16)), dq_attrs);
    auto* weight_arg = builder.MakeInitializer<Int4x2>({12, 37}, Int4x2(Int4x2::min_val, 0), Int4x2(Int4x2::max_val, 0));
    auto* scales_arg = builder.MakeInitializer<float>({1, 37}, 8.0f, 12.0f);
    auto* dq_output = builder.MakeIntermediate();
    builder.AddNode("DequantizeLinear", {weight_arg, scales_arg}, {dq_output}, "", &dq_attrs);

    NodeAttributes gemm_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("transB", static_cast<int64_t>(1)), gemm_attrs);
    builder.AddNode("Gemm", {input_arg, dq_output}, {output_arg}, "", &gemm_attrs);
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["Gemm"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 0);
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5, 2e-5);
}

// Negative test: DQ -> Gemm with alpha != 1.0 should NOT be fused.
TEST(QDQTransformerTests, DQGemmNotConvertedToMatMulNBits_Alpha) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({12, 37}, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", static_cast<int64_t>(16)), dq_attrs);
    auto* weight_arg = builder.MakeInitializer<Int4x2>({37, 12}, Int4x2(Int4x2::min_val, 0), Int4x2(Int4x2::max_val, 0));
    auto* scales_arg = builder.MakeInitializer<float>({3, 12}, 8.0f, 12.0f);
    auto* dq_output = builder.MakeIntermediate();
    builder.AddNode("DequantizeLinear", {weight_arg, scales_arg}, {dq_output}, "", &dq_attrs);

    NodeAttributes gemm_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("alpha", 2.0f), gemm_attrs);
    builder.AddNode("Gemm", {input_arg, dq_output}, {output_arg}, "", &gemm_attrs);
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["Gemm"], 1);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 0);
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1e-5, 2e-5);
}

// ---------------------------------------------------------------------------
// Cross-session pre-pack sharing for the DEFAULT DQ->MatMulNBits path
// ---------------------------------------------------------------------------
// DQMatMulToMatMulNBitsAction (in the QDQ selector/action transformer) runs without the
// session.enable_dq_matmulnbits_fusion flag and synthesizes the MatMulNBits B/scales/zp initializers
// with names that are NOT stable across sessions. It tags the generated B weight with a sharing
// identity that SessionState treats as the enrollment signal opting the buffer into the cross-session
// container; the actual sharing is keyed by the packed-bytes hash (only byte-identical packed buffers
// are reused, exactly like the AddInitializer path), so packings that differ by compute type/options
// are never falsely shared.

// Packs uint4 nibble values (row-major, 2 per byte) into UInt4x2 storage.
static std::vector<UInt4x2> PackUint4Nibbles(const std::vector<uint8_t>& values) {
  const size_t num_pairs = UInt4x2::CalcNumInt4Pairs(values.size());
  std::vector<UInt4x2> packed(num_pairs);
  for (size_t i = 0; i < values.size(); i += 2) {
    const uint8_t lo = values[i] & 0x0F;
    const uint8_t hi = (i + 1 < values.size()) ? (values[i + 1] & 0x0F) : 0;
    packed[i / 2] = UInt4x2(lo, hi);
  }
  return packed;
}

// Builds a default-path model: a constant UINT4 weight [K, N] block-quantized along axis 0 feeding a
// DequantizeLinear whose output is the second input to a single MatMul. The QDQ selector/action
// transformer converts this into a MatMulNBits. Explicit weight/scale/zp give a deterministic identity.
static void BuildDefaultPathDQMatMul(ModelTestBuilder& builder, int64_t M, int64_t N, int64_t K,
                                     int64_t block_size, const std::vector<uint8_t>& weight,
                                     const std::vector<float>& scale, const std::vector<uint8_t>& zp) {
  const int64_t num_blocks = (K + block_size - 1) / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  auto* weight_arg = builder.MakeInitializer<UInt4x2>({K, N}, PackUint4Nibbles(weight));
  auto* scale_arg = builder.MakeInitializer<float>({num_blocks, N}, scale);
  auto* zp_arg = builder.MakeInitializer<UInt4x2>({num_blocks, N}, PackUint4Nibbles(zp));

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();
  builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);

  builder.AddNode("MatMul", {input_a, dq_output}, {output});
}

// Serializes a default-path DQ->MatMul model built from explicit quantization data.
static void SerializeDefaultPathModel(int64_t M, int64_t N, int64_t K, int64_t block_size,
                                      const std::vector<uint8_t>& weight, const std::vector<float>& scale,
                                      const std::vector<uint8_t>& zp, std::string& model_bytes) {
  const std::unordered_map<std::string, int> domain_to_version{{"", 21}, {kMSDomain, 1}};
  Model model("dq_matmul_default_share", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
              std::vector<ONNX_NAMESPACE::FunctionProto>(), DefaultLoggingManager().DefaultLogger());
  ModelTestBuilder builder(model.MainGraph());
  BuildDefaultPathDQMatMul(builder, M, N, K, block_size, weight, scale, zp);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_bytes));
}

// Loads the model on the CPU EP with the given shared container and DEFAULT options (no fusion flag).
// Reports whether a MatMulNBits was produced, the sharing identity tagged onto its B weight, and how
// many pre-packed weights this session served from the container.
static void RunDefaultPathSession(const std::string& model_bytes, PrepackedWeightsContainer& container,
                                  bool& produced_matmulnbits, std::string& b_tag, size_t& used_shared_count,
                                  int accuracy_level = -1) {
  SessionOptions so;
  // This test exercises prepack-weight sharing, not parallel execution. Cap the intra-op thread pool
  // to a single thread so we don't spin up one worker per core: under AddressSanitizer each thread adds
  // fake-stack and thread-local allocator overhead, which on a high-core CI runner multiplies across the
  // sessions every test creates (the sibling SessionStatePrepackingTest caps it for the same reason).
  so.intra_op_param.thread_pool_size = 1;
  if (accuracy_level >= 0) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQMatMulNBitsAccuracyLevel,
                                                      std::to_string(accuracy_level).c_str()));
  }
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.AddPrePackedWeightsContainer(&container));
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  produced_matmulnbits = false;
  b_tag.clear();
  const Graph& graph = session.GetGraph();
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "MatMulNBits") {
      produced_matmulnbits = true;
      const std::string& b_name = node.InputDefs()[1]->Name();  // input 1 == quantized B
      if (const std::string* id = graph.GetSharedPrepackInitializerId(b_name); id != nullptr) {
        b_tag = *id;
      }
      break;
    }
  }
  used_shared_count = session.GetSessionState().GetUsedSharedPrePackedWeightCounter();
}

// Verifies the default DQ->MatMulNBits path tags its generated B weight with a stable, content-derived
// enrollment identity: identical quantization data yields the SAME identity, while different zero points
// yield a DIFFERENT identity. (The tag only enrolls the buffer for sharing; the container keys by the
// packed-bytes hash. A stable, content-distinct tag keeps enrollment deterministic across sessions.)
TEST(QDQTransformerTests, DefaultPath_TagsGeneratedWeightWithStableContentIdentity) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  const int64_t num_blocks = K / block_size;

  std::vector<uint8_t> weight(static_cast<size_t>(K * N));
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = static_cast<uint8_t>(i % 16);
  }
  std::vector<float> scale(static_cast<size_t>(num_blocks * N));
  for (size_t i = 0; i < scale.size(); ++i) {
    scale[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
  }
  std::vector<uint8_t> zp_a(static_cast<size_t>(num_blocks * N), 3);
  std::vector<uint8_t> zp_b(zp_a.size(), 5);

  auto tag_for = [&](const std::vector<uint8_t>& zp) -> std::string {
    std::string model_bytes;
    SerializeDefaultPathModel(M, N, K, block_size, weight, scale, zp, model_bytes);
    PrepackedWeightsContainer container;
    bool produced = false;
    std::string tag;
    size_t used = 0;
    RunDefaultPathSession(model_bytes, container, produced, tag, used);
    EXPECT_TRUE(produced) << "DQ -> MatMulNBits conversion did not run on the default path";
    return tag;
  };

  const std::string id_a1 = tag_for(zp_a);
  const std::string id_a2 = tag_for(zp_a);
  const std::string id_b = tag_for(zp_b);

  ASSERT_FALSE(id_a1.empty()) << "generated B weight was not tagged for cross-session sharing";
  EXPECT_EQ(id_a1, id_a2);  // stable: identical quantization data -> identical identity
  EXPECT_NE(id_a1, id_b);   // collision-safe: different zero points -> different identity
}

// End-to-end: two sessions converting the same model via the default path share the MatMulNBits B
// pre-packed buffer through a common container (no session option). A model whose quantized weight
// differs packs to different bytes -> different container key, so it must not reuse the buffer. (A
// zero-point-only difference is intentionally NOT used here: on the CompFp32 path the zero points are
// applied at compute time and left out of the packed B, so two such models pack identically and would
// correctly share -- packed-bytes keying only ever reuses byte-identical buffers.)
TEST(QDQTransformerTests, DefaultPath_SharesWeightAcrossSessionsViaTag) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  const int64_t num_blocks = K / block_size;

  std::vector<uint8_t> weight(static_cast<size_t>(K * N));
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = static_cast<uint8_t>(i % 16);
  }
  // A different quantized weight -> different packed B on every compute type (unlike a zp-only change).
  std::vector<uint8_t> weight_other(weight.size());
  for (size_t i = 0; i < weight_other.size(); ++i) {
    weight_other[i] = static_cast<uint8_t>((i + 7) % 16);
  }
  std::vector<float> scale(static_cast<size_t>(num_blocks * N));
  for (size_t i = 0; i < scale.size(); ++i) {
    scale[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
  }
  std::vector<uint8_t> zp(static_cast<size_t>(num_blocks * N), 3);

  std::string model_a, model_other;
  SerializeDefaultPathModel(M, N, K, block_size, weight, scale, zp, model_a);
  SerializeDefaultPathModel(M, N, K, block_size, weight_other, scale, zp, model_other);

  PrepackedWeightsContainer container;
  bool produced1 = false, produced2 = false, produced_other = false;
  std::string tag1, tag2, tag_other;
  size_t used1 = 0, used2 = 0, used_other = 0;

  RunDefaultPathSession(model_a, container, produced1, tag1, used1);
  ASSERT_TRUE(produced1) << "DQ -> MatMulNBits conversion did not run on the default path";
  if (container.GetNumberOfElements() == 0) {
    GTEST_SKIP() << "MatMulNBits B was not pre-packed on this platform";
  }
  EXPECT_EQ(used1, static_cast<size_t>(0));  // first session: nothing to share yet

  // Second session over the SAME model reuses the tagged B from the container.
  RunDefaultPathSession(model_a, container, produced2, tag2, used2);
  ASSERT_TRUE(produced2);
  EXPECT_GT(used2, static_cast<size_t>(0));

  // A model with a different quantized weight packs to different bytes -> different key, so it must NOT
  // reuse the buffer (on any compute type).
  RunDefaultPathSession(model_other, container, produced_other, tag_other, used_other);
  ASSERT_TRUE(produced_other);
  EXPECT_EQ(used_other, static_cast<size_t>(0));
}

// accuracy_level participates in the enrollment identity, so the same weights requested at different
// accuracy levels get distinct identities. Whether the two sessions then share the packed buffer is
// platform-dependent (level 4 may pack as CompInt8 -- different bytes, no share -- or fall back to the
// same CompFp32 packing as level 0 and benignly reuse the byte-identical buffer); packed-bytes keying
// makes either outcome safe, so this asserts the identity is distinct, not a fixed sharing count.
TEST(QDQTransformerTests, DefaultPath_DifferentAccuracyLevelGetsDistinctIdentity) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  const int64_t num_blocks = K / block_size;

  std::vector<uint8_t> weight(static_cast<size_t>(K * N));
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = static_cast<uint8_t>(i % 16);
  }
  std::vector<float> scale(static_cast<size_t>(num_blocks * N));
  for (size_t i = 0; i < scale.size(); ++i) {
    scale[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
  }
  std::vector<uint8_t> zp(static_cast<size_t>(num_blocks * N), 3);

  std::string model_bytes;
  SerializeDefaultPathModel(M, N, K, block_size, weight, scale, zp, model_bytes);

  PrepackedWeightsContainer container;
  bool produced0 = false, produced4 = false;
  std::string tag0, tag4;
  size_t used0 = 0, used4 = 0;

  RunDefaultPathSession(model_bytes, container, produced0, tag0, used0, /*accuracy_level*/ 0);
  ASSERT_TRUE(produced0) << "DQ -> MatMulNBits conversion did not run on the default path";

  // Same model/weights, different accuracy level, sharing the same container.
  RunDefaultPathSession(model_bytes, container, produced4, tag4, used4, /*accuracy_level*/ 4);
  ASSERT_TRUE(produced4);

  ASSERT_FALSE(tag0.empty());
  ASSERT_FALSE(tag4.empty());
  EXPECT_NE(tag0, tag4);  // accuracy_level participates in the enrollment identity
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
