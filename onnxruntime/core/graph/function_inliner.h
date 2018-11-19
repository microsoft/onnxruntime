// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/graph/function.h"
#include "core/graph/rewrite_rule.h"

namespace onnxruntime {
class Node;
}  // namespace onnxruntime

namespace onnxruntime {

// A function-inlining rewrite-rule.
class FunctionInliner : public onnxruntime::RewriteRule {
 public:
  FunctionInliner(const std::string& name, const std::string& desc)
      : RewriteRule(name, desc) {}

  Status Apply(onnxruntime::GraphEditor /*graph_editor*/, onnxruntime::Node* /*node*/, bool* /*modified*/) override {
    return Status::OK();
  }
};

}  // namespace onnxruntime
