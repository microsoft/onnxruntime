// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>

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
    const Path& model_path);

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
    const Path& model_path);

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

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// Check Q node op type, version, and domain.
bool MatchQNode(const Node& node);

// Check DQ node op type, version, and domain.
bool MatchDQNode(const Node& node);
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
}  // namespace QDQ
}  // namespace onnxruntime
