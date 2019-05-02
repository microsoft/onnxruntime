// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

/**
@class ConstantFolding

Transformer that traverses the graph top-down and performs constant folding, i.e.,
it statically computes parts of the graph that rely only on constant initializers.
*/
class ConstantFolding : public GraphTransformer {
 public:
  ConstantFolding(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept :
    GraphTransformer("ConstantFolding", compatible_execution_providers) {}

 private:
  /** Constant folding will not be applied to nodes whose op_type is included in this set.
      All non-deterministic operators should be included in this set. */
  const std::unordered_set<std::string> excluded_op_types_ =
      {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"};

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;

  /** Create a TensorProto that has the same value as the given MLValue 
  and the same type and dimensions as the given NodeArg. */
  void BuildTensorProtoForInitializer(const MLValue& mlvalue,
                                      const NodeArg& constant_node_arg,
                                      ONNX_NAMESPACE::TensorProto& tensorproto) const;
};

}  // namespace onnxruntime
