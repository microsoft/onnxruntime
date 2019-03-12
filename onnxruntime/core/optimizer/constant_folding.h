// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

/**
@class ConstantFolding

Rewrite rule that performs constant folding to the graph.
The rule gets applied to nodes that have only initializers as inputs. It statically computes these 
nodes and replaces their output with an initializer that corresponds to the result of the computation.
*/
class ConstantFolding : public RewriteRule {
 public:
  ConstantFolding() noexcept : RewriteRule("ConstantFolding", "Constant folding") {}

 private:
  /** Constant folding will not be applied to nodes whose op_type is included in this set.
      All non-deterministic operators should be included in this set. */
  const std::unordered_set<std::string> excluded_op_types_ =
      {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Multinomial"};

  bool SatisfyCondition(const Graph& graph, const Node& node) override;

  bool OpTypeCondition(const Node& node) override;

  Status Apply(Graph& graph, Node& node, bool& modified, bool& deleted) override;

  /** Create a TensorProto that has the same value as the given MLValue 
  and the same type and dimensions as the given NodeArg. */
  void BuildTensorProtoForInitializer(const MLValue& mlvalue,
                                      const NodeArg& constant_node_arg,
                                      ONNX_NAMESPACE::TensorProto& tensorproto);
};

}  // namespace onnxruntime
