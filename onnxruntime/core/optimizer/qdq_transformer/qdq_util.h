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

static constexpr const char* QOpName = "QuantizeLinear";
static constexpr const char* DQOpName = "DequantizeLinear";

enum InputIndex : int {
  INPUT_ID = 0,
  SCALE_ID = 1,
  ZERO_POINT_ID = 2,
  TOTAL_COUNT = 3,
};

// Check if Q/DQ pair is supported in the QDQ transformer. It requires:
// 1. Q/DQ doesn't have optional input.
// 2. scale and zero point is constant scalar
// 3. Q and DQ have same scale and zero point
bool IsQDQPairSupported(
    const Node& q_node, const Node& dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer,
    const Path& model_path);

// Check if DQ is supported in the QDQ transformer. It requires:
// 1. DQ doesn't have optional input.
// 2. scale and zero point is constant scalar
bool IsDQSupported(
    const Node& dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer);

}  // namespace QDQ
}  // namespace onnxruntime
