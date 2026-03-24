// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "core/common/span_utils.h"
#include "core/common/float16.h"
#include "core/framework/int4.h"
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

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
