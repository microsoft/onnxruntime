// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/framework/compute_capability.h"
#include "core/providers/nuphar/partition/partitioner.h"

#include <functional>
#include <unordered_set>
#include <vector>

namespace onnxruntime {
namespace nuphar {

using IsOpTypeSupportedFunc = std::function<bool(const Node& node)>;

// GraphPartitioner partitions Ort graph and generates FuseNodes.
class GraphPartitioner : public Partitioner {
 public:
  GraphPartitioner(IsOpTypeSupportedFunc is_op_type_supported_func)
      : Partitioner(), is_op_type_supported_func_(is_op_type_supported_func) {}

  Status Partition(const onnxruntime::GraphViewer& graph,
                   int& fused_count,
                   std::vector<std::unique_ptr<ComputeCapability>>& result);

 private:
  IsOpTypeSupportedFunc is_op_type_supported_func_;

  bool IsNodeSupported(const Node& node) const override;

  void HandleSubgraph(const onnxruntime::GraphViewer& graph) override;

  void CreateNewPartition(const Node& node, const std::vector<NodeIndex>& immedidate_rejected_partitions) override;

  // FORCE_ONE_SUBGRAPH is a marco to generate single subgraph partition
  // It is mainly for debug and reproducing older version
#ifdef FORCE_ONE_SUBGRAPH
  bool ForcePartition(
      const onnxruntime::GraphViewer& /*graph*/,
      const Node& node,
      const std::vector<NodeIndex>& candiates,
      const std::vector<NodeIndex>& immedidate_rejected_partitions) override;
#endif
};

}  // namespace nuphar
}  // namespace onnxruntime
