// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <filesystem>

namespace ONNX_NAMESPACE {
class TensorProto;
}

namespace onnxruntime {

class Node;
class Path;

namespace QDQ {

constexpr const char* QOpName = "QuantizeLinear";
constexpr const char* DQOpName = "DequantizeLinear";

enum InputIndex : int {
  INPUT_ID = 0,
  SCALE_ID = 1,
  ZERO_POINT_ID = 2,
  TOTAL_COUNT = 3,
};

using GetConstantInitializerFn = std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>;

// Check if Q/DQ pair is supported in extended level QDQ transformers. It requires:
// 1. Q/DQ doesn't have optional input.
// 2. scale and zero point is constant scalar
// 3. Q and DQ have same scale and zero point
bool IsQDQPairSupported(
    const Node& q_node, const Node& dq_node,
    const GetConstantInitializerFn& get_const_initializer,
    const std::filesystem::path& model_path,
    bool check_op_type = true);

// Check if a DQ -> Q sequence represents a conversion in quantization data type.
// Example of uint8 to uint16:
//     Dequantize (uint8 to float) -> Quantize (float to uint16)
// Requires:
// 1. Q/DQ doesn't have optional input.
// 2. scale and zero-point are constant scalars.
// 3. Q and DQ have the same scale *type* and different zero-point *types*.
bool IsDQQConversion(
    const Node& dq_node, const Node& q_node,
    const GetConstantInitializerFn& get_const_initializer,
    const std::filesystem::path& model_path);

// Check if DQ is supported in extended level QDQ transformers. It requires:
// 1. DQ doesn't have optional input.
// 2. scale and zero point is constant scalar
bool IsDQSupported(const Node& dq_node, const GetConstantInitializerFn& get_const_initializer);

// Check if Q or DQ node's scale and zero point inputs are constant scalars.
// If the zero point input does not exist, it is assumed to have a default scalar value.
// `zero_point_exists` will indicate if it does exist.
bool QOrDQNodeHasConstantScalarScaleAndZeroPoint(
    const Node& q_or_dq_node,
    const GetConstantInitializerFn& get_const_initializer,
    bool& zero_point_exists);

// Checks that the y_scale/x_scale input to the QuantizeLinear/DequantizeLinear node is a positive scalar.
bool IsQOrDQScalePositiveConstantScalar(const Node& q_or_dq_node, const GetConstantInitializerFn& get_const_initializer,
                                        const std::filesystem::path& model_path);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// Check Q node op type, version, and domain.
bool MatchQNode(const Node& node);

// Check DQ node op type, version, and domain.
bool MatchDQNode(const Node& node);
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

// 1. Clip can be removed iff the codoamin of QuantizeLinear remain unchanged.
// 2. To remain unchanged, y=QuantizeLinear(Clip(x)) must span the full range of values that can be represented by the
//    integer type of y. We can use this precondition to eval QuantizeLinear backward to to get the domain.
// 3. Indicates the domain of QuantizeLinear is strict subset of the codomain of Clip. We can use this to test if a
//    removal is valid or not.
// 4. Due to rounding effect, we can be get the upperbound and lowerbound of min or max
//    - Which one to use?
//      upperbound of min and lowerbound of max
//    - Why?
//      We want the codomain to be unchanged, so as long as the domain genreate a codomain that fill the integer value
//      range will be fine.

// The quantization formula is y = saturate((x / y_scale) + y_zero_point)
// For (x / y_scale), it rounds to the nearest even. So the allowed quantize limits before saturate need to be taken
// care of.
//
// The following struct provides a wrapper to compute the domain from codomain (Q output dtype).
template <typename T>
struct QuantizeDomain {
  inline static float MinUpper(float scale, int zp) {
    int64_t codomain_min = std::numeric_limits<T>::lowest();
    int64_t biased = codomain_min - zp;
    float before_round = static_cast<float>(biased) + 0.5f;  // move to upperbound
    if (biased % 2 == 1) {                                   // cannot be exact ?.5 because of rounding to even
      before_round = std::nextafterf(before_round, static_cast<float>(biased));
    }
    return before_round * scale;
  }

  inline static float MaxLower(float scale, int zp) {
    int64_t codomain_max = std::numeric_limits<T>::max();
    int64_t biased = codomain_max - zp;
    float before_round = static_cast<float>(biased) - 0.5f;  // move to lowerbound
    if (biased % 2 == 1) {                                   // cannot be exact ?.5 because of rounding to even
      before_round = std::nextafterf(before_round, static_cast<float>(biased));
    }
    return before_round * scale;
  }
};

}  // namespace QDQ
}  // namespace onnxruntime
