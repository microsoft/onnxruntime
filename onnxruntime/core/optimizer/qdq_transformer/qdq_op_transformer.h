// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

static const char* const QOPTypeName = "QuantizeLinear";
static const char* const DQOPTypeName = "DequantizeLinear";

class Node;
class Graph;

class QDQOperatorTransformer {
 public:
  QDQOperatorTransformer(Node& node, Graph& graph) : node_(node), graph_(graph) {}
  virtual ~QDQOperatorTransformer() {}
  bool Transform(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes);

  /* Determine whether to keep node_ itself or not.
           For operators that support int8, keep node_ and only change its input and output.
           Otherwise, node_ will be removed and replaced by a QLinear* version.
  */
  virtual bool KeepNode() const {
    return false;
  }

 protected:
  // A general check QDQ and Op fusion. It requires:
  //   1. All inputs of Op come from DequantizeLinear
  //   2. All outputs of Op flows to QuantizeLinear
  //   3. Op outputs are not graph outputs
  // Override this function if it is not case for certain op
  virtual bool Check(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) const;

  // Implement the fusion process
  // Prerequisites are handled in Transform. You can assume it is valid to fuse when implementing.
  virtual bool TransformImpl(const std::vector<const Node*>& dq_nodes, const std::vector<const Node*>& q_nodes) = 0;

  Node& node_;
  Graph& graph_;

 private:
  void FillQDQOptionalZeroPoint(const std::vector<const Node*>& parents);

  static const ONNX_NAMESPACE::TensorProto optional_zero_point_int8_;
  static const ONNX_NAMESPACE::TensorProto optional_zero_point_uint8_;
};
}  // namespace onnxruntime
