// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GistEncode
*/
class GistEncodeDecode : public RewriteRule {
 public:
  int operator_type;

  static constexpr const char* GIST_PAIR_NODE_NAME_BASE = "gist";

  static constexpr int GIST_PACK1_FACTOR = 8;

  mutable int priority_generator_ = INT32_MAX;

  // map stores GIST signature - source operator type to destination operator type(s)
  typedef std::vector<std::string> vector_t;
  const std::unordered_map<std::string, vector_t> PATTERN_MAP = {
      {"Softmax", {"SoftmaxGrad"}},
      {"Transpose", {"Transpose"}},
      {"Reshape", {"Reshape"}},
      {"Add", {"LayerNormalizationGrad"}},
      {"Dropout", {"Transpose", "Reshape", "DropoutGrad"}},
      {"LayerNormalization", {"Reshape", "Shape", "LayerNormalizationGrad"}},
      {"MatMul", {"Shape"}},
      {"Relu", {"ReluGrad", "Shape", "Reshape"}}};

  GistEncodeDecode() noexcept : RewriteRule("GistEncodeDecode") {}
  GistEncodeDecode(int op_type, std::string compr_type) noexcept : RewriteRule("GistEncodeDecode"), operator_type(op_type), compression_type_(std::move(compr_type)) {}

 private:
  int GenerateDecodePriority() const { return priority_generator_--; };
  std::vector<std::string> TargetOpTypes() const noexcept override;
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;
  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
  bool AddEncodeDecode(Graph& graph, Node& curr_node, std::string compression_type, const logging::Logger& logger) const;

  std::string compression_type_;
};

}  // namespace onnxruntime
