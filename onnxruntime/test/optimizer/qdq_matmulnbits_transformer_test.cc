// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "core/common/span_utils.h"
#include "core/framework/int4.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"

#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"

#include "qdq_test_utils.h"

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
                                   bool use_contrib_qdq) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* input2_arg = builder.MakeInput(input2_shape, T(T::min_val, 0), T(T::min_val, 0));
    auto* output_arg = builder.MakeOutput();

    // add DQ
    auto* dq_output = builder.MakeIntermediate();
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);

    auto scale_shape = std::vector<int64_t>{input2_shape};
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;
    auto scales = builder.rand_gen_.Uniform(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      auto zero_points = builder.rand_gen_.Uniform<T>(scale_shape, T(0, 0), T(2, 0));
      builder.AddDequantizeLinearNode(input2_arg, scales, zero_points, dq_output, &attrs, use_contrib_qdq);
    } else {
      builder.AddDequantizeLinearNode(input2_arg, scales, dq_output, &attrs, use_contrib_qdq);
    }

    builder.AddNode("MatMul", {input1_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["MatMul"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    19 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_NonConstDQ) {
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, true);
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
                                               bool use_contrib_qdq) {
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

    auto scale_shape = std::vector<int64_t>{weight_shape};
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;
    auto scales = builder.rand_gen_.Uniform<float>(scale_shape, 8.0f, 12.0f);
    if constexpr (use_zp) {
      std::vector<T> zero_points;
      if constexpr (std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>) {
        zero_points = builder.rand_gen_.Uniform<T>(scale_shape, T(0, 0), T(2, 0));
      } else {
        zero_points = builder.rand_gen_.Uniform<T>(scale_shape, static_cast<T>(0), static_cast<T>(2));
      }

      builder.AddDequantizeLinearNode(weight_arg, scales, zero_points, dq_output, &attrs, use_contrib_qdq);
    } else {
      builder.AddDequantizeLinearNode(weight_arg, scales, dq_output, &attrs, use_contrib_qdq);
    }

    builder.AddNode("MatMul", {input_arg, dq_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["MatMul"], 1);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    19 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_TypeMismatch) {
  RunDQMatMulNotConverted_TypeShapeMismatch<int8_t, true>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<int8_t, false>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<uint8_t, true>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<uint8_t, false>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<int16_t, true>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<int16_t, false>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<uint16_t, true>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<uint16_t, false>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<int32_t, true>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<int32_t, false>({12, 37}, {37, 12}, 0, 16, true);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_ShapeMismatch) {
  // block size too small
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 8, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 8, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 8, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 8, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 8, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 8, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 8, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 8, true);
  // block size not 2's power
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 17, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 17, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 17, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 17, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 17, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 17, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 17, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 17, true);
  // not axis 0
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 37}, 1, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 37}, 1, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 37}, 1, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 37}, 1, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 37}, 1, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 37}, 1, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 37}, 1, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 37}, 1, 16, true);
  // not rank 2
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12, 2}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12, 2}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12, 2}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12, 2}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12, 2}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12, 2}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12, 2}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12, 2}, 0, 16, true);
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
                     const std::vector<int64_t>& weight_shape,
                     const int64_t axis,
                     const int64_t block_size,
                     bool use_contrib_qdq) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -100.0f, 100.0f);
    auto* output_arg = builder.MakeOutput();

    // add DQ
    NodeAttributes attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), attrs);
    auto scale_shape = std::vector<int64_t>{weight_shape};
    scale_shape[axis] = (scale_shape[axis] + block_size - 1) / block_size;

    auto* weight1_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* weight2_arg = builder.MakeInitializer(weight_shape, T(T::min_val, 0), T(T::max_val, 0));
    auto* dq1_output = builder.MakeIntermediate();
    auto* dq2_output = builder.MakeIntermediate();
    auto* matmul1_output = builder.MakeIntermediate();

    auto scales1 = builder.rand_gen_.Uniform(scale_shape, 8.0f, 12.0f);
    auto scales2 = builder.rand_gen_.Uniform(scale_shape, 8.0f, 12.0f);
    Node* dp1_node = nullptr;
    if constexpr (use_zp) {
      auto zero_points1 = builder.rand_gen_.Uniform<T>(scale_shape, T(0, 0), T(2, 0));
      auto zero_points2 = builder.rand_gen_.Uniform<T>(scale_shape, T(0, 0), T(2, 0));
      builder.AddDequantizeLinearNode(weight1_arg, scales1, zero_points1, dq1_output, &attrs, use_contrib_qdq);
      builder.AddDequantizeLinearNode(weight2_arg, scales2, zero_points2, dq2_output, &attrs, use_contrib_qdq);
    } else {
      builder.AddDequantizeLinearNode(weight1_arg, scales1, dq1_output, &attrs, use_contrib_qdq);
      builder.AddDequantizeLinearNode(weight2_arg, scales2, dq2_output, &attrs, use_contrib_qdq);
    }

    builder.AddNode("MatMul", {input_arg, dq1_output}, {matmul1_output});
    builder.AddNode("MatMul", {matmul1_output, dq2_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 2);
    EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
    EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    19 /*opset_version*/,
                    1e-5 /*per_sample_tolerance*/,
                    1e-5 /*relative_per_sample_tolerance*/);
}

TEST(QDQTransformerTests, DQMatMulConvertedToMatMulNBits) {
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, true);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, false);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, true);
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
