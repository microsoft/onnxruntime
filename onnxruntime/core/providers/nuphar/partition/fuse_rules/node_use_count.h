// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "fuse_rule_base.h"
#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "core/providers/nuphar/partition/fuse_rules/cut_config.h"

namespace onnxruntime {
namespace nuphar {

class RuleNodeUseCount : public FuseRule {
 public:
  RuleNodeUseCount(const codegen::OrtGraphStats* graph_stats)
      : FuseRule(),
        graph_stats_(graph_stats),
        uses_valid_for_cut_(FuseCutConfig::node_uses_valid_for_cut),
        uses_threshold_(FuseCutConfig::node_uses_cut_threshold) {}
  virtual ~RuleNodeUseCount() = default;
  virtual Status Fuse(const onnxruntime::GraphViewer& graph,
                      IsOpTypeSupportedFunc is_op_type_supported_func,
                      std::set<NodeIndex>& claimed_nodes,
                      std::vector<std::unique_ptr<ComputeCapability>>& result) override;

 private:
  void AddToSubgraph(const onnxruntime::GraphViewer& graph,
                     std::unique_ptr<IndexedSubGraph>& subgraph,
                     std::set<NodeIndex>& claimed_nodes,
                     int* acc_uses,
                     std::vector<std::unique_ptr<ComputeCapability>>& result);

  bool CanCut(const onnxruntime::Node* node) const;

  const onnxruntime::codegen::OrtGraphStats* graph_stats_;

  // a magic number for cutting subgraph: any node that is used at least
  // this number of times is valid for cut.
  int uses_valid_for_cut_;

  int uses_threshold_;
};

}  // namespace nuphar
}  // namespace onnxruntime
