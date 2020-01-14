// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
@Class GistEncode

Rewrite rule that encode when Relu is found.

It is attempted to be triggered only on nodes with op type "Relu".
*/
class GistEncodeDecode : public RewriteRule {
 public:
  static constexpr const char* GIST_ENCODER_NODE_NAME_BASE = "gist_encode";
  static constexpr const char* GIST_DECODER_NODE_NAME_BASE = "gist_decode";

  GistEncodeDecode() noexcept : RewriteRule("GistEncodeDecode") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Relu"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const override;

  Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const override;
  bool AddEncodeDecode(Graph& graph, Node& curr_node, std::string compression_type) const;
};

}