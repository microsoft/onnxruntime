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
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, -1);
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, -1);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, -1);
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, -1);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_NonConstDQ_Cuda) {
  // DQ contrib op schema is not updated to support blocked quantization
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
  ;
  RunDQMatMulNotConverted_NonConstDQ<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, true>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_NonConstDQ<Int4x2, false>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
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

    auto scale_shape = std::vector<int64_t>{weight_shape};
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
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 1);
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4);
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, -1);
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, -1);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, -1);
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, -1);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_FirstDQInput_Cuda) {
  // DQ contrib op schema is not updated to support blocked quantization
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, 4, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, true>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
  ;
  RunDQMatMulNotConverted_FirstDQInput<UInt4x2, false>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, true>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_FirstDQInput<Int4x2, false>({12, 37}, {37, 12}, 0, 16, -1, DefaultCudaExecutionProvider());
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

    auto scale_shape = std::vector<int64_t>{weight_shape};
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
  // DQ contrib op schema is not updated to support blocked quantization
  // block size too small
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 8, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 8, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 8, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 8, 0);
  // block size not 2's power
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 17, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 17, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 17, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 17, 0);
  // not axis 0
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 37}, 1, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 37}, 1, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 37}, 1, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 37}, 1, 16, 0);
  // not rank 2
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({2, 12, 37}, {2, 37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({2, 12, 37}, {2, 37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({2, 12, 37}, {2, 37, 12}, 0, 16, 0);
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({2, 12, 37}, {2, 37, 12}, 0, 16, 0);
}

TEST(QDQTransformerTests, DQMatMulNotConvertedToMatMulNBits_ShapeMismatch_Cuda) {
  // DQ contrib op schema is not updated to support blocked quantization
  // block size too small
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 8, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 8, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 8, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 8, 0, DefaultCudaExecutionProvider());
  // block size not 2's power
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 12}, 0, 17, 0, DefaultCudaExecutionProvider());
  ;
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 12}, 0, 17, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 12}, 0, 17, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 12}, 0, 17, 0, DefaultCudaExecutionProvider());
  // not axis 0
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({12, 37}, {37, 37}, 1, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({12, 37}, {37, 37}, 1, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({12, 37}, {37, 37}, 1, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({12, 37}, {37, 37}, 1, 16, 0, DefaultCudaExecutionProvider());
  // not rank 2
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, true>({2, 12, 37}, {2, 37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<UInt4x2, false>({2, 12, 37}, {2, 37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, true>({2, 12, 37}, {2, 37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
  RunDQMatMulNotConverted_TypeShapeMismatch<Int4x2, false>({2, 12, 37}, {2, 37, 12}, 0, 16, 0, DefaultCudaExecutionProvider());
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

// Cast-aware DQ->MatMul fusion tests
// Pattern: DQ(int4->fp16) -> Cast(fp16->fp32) -> MatMul(fp32)
// The Cast between DQ and MatMul on input B should be handled by the
// DQCastMatMulToMatMulNBits selector-action pair.
// MatMulNBits always operates in the DQ scale dtype (fp16).
// The action always inserts Cast on input A and Cast on output.
// ORT's redundant cast elimination optimizer cleans up unnecessary casts.
//
// Input1(fp32)      DQ(int4->fp16)
//   |                    |
//    \             Cast(fp16->fp32)
//     \              /
//      MatMul(fp32)
//          |
//        output(fp32)
//
// After optimization:
// Input1(fp32) -> Cast(fp32->fp16) -> MatMulNBits(fp16) -> Cast(fp16->fp32) -> output(fp32)
template <typename T, bool use_zp>
typename std::enable_if<std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>, void>::type
RunDQCastMatMulConverted(const std::vector<int64_t>& input1_shape,
                         const std::vector<int64_t>& weight_shape,
                         const int64_t axis,
                         const int64_t block_size,
                         int64_t accuracy_level) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput(input1_shape, -1.0f, 1.0f);
    auto* output_arg = builder.MakeOutput();

    // DQ with fp16 scales
    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", axis), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);

    auto scale_shape = std::vector<int64_t>{weight_shape};
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

    // Cast fp16 -> fp32
    auto* cast_output = builder.MakeIntermediate();
    NodeAttributes cast_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("to",
                                                 static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT)),
                            cast_attrs);
    builder.AddNode("Cast", {dq_output}, {cast_output}, "", &cast_attrs);

    // MatMul
    builder.AddNode("MatMul", {input_arg, cast_output}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    const QDQOpKeys qdq_keys = GetQDQOpKeys(false);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    // B-side Cast removed. New Cast(fp32->fp16) on A and Cast(fp16->fp32) on output.
    EXPECT_EQ(op_to_count["Cast"], 2);
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

TEST(QDQTransformerTests, DQCastMatMulConvertedToMatMulNBits) {
  // DQ(int4->fp16) -> Cast(fp16->fp32) -> MatMul should be fused to MatMulNBits
  RunDQCastMatMulConverted<Int4x2, true>({12, 32}, {32, 16}, 0, 16, 0);
  RunDQCastMatMulConverted<Int4x2, false>({12, 32}, {32, 16}, 0, 16, 0);
  RunDQCastMatMulConverted<UInt4x2, true>({12, 32}, {32, 16}, 0, 16, 0);
  RunDQCastMatMulConverted<UInt4x2, false>({12, 32}, {32, 16}, 0, 16, 0);
}

#endif  // !defined(DISABLE_CONTRIB_OPS)

}  // namespace test
}  // namespace onnxruntime
