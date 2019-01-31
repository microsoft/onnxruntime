// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/node_index_info.h"

#include "core/framework/mlvalue_name_idx_map.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/node_arg.h"

namespace onnxruntime {

NodeIndexInfo::NodeIndexInfo(const GraphViewer& graph_viewer, const MLValueNameIdxMap& mlvalue_idx_map)
    : max_mlvalue_idx_{mlvalue_idx_map.MaxIdx()} {
  std::size_t total_def_count{};

  bool include_missing_optional_defs = true;

  for (const auto& node : graph_viewer.Nodes()) {
    node.ForEachDef(
        [&](const onnxruntime::NodeArg& /*arg*/, bool /*is_input*/) {
          ++total_def_count;
        },
        include_missing_optional_defs);
  }

  // init all to kInvalidEntry
  node_offsets_.resize(graph_viewer.MaxNodeIndex(), kInvalidEntry);
  node_values_.resize(total_def_count, kInvalidEntry);
  int cur_idx = 0;

  for (auto& node : graph_viewer.Nodes()) {
    node_offsets_[node.Index()] = cur_idx;

    node.ForEachDef(
        [&](const onnxruntime::NodeArg& node_arg, bool /*is_input*/) {
          auto& name = node_arg.Name();
          if (node_arg.Exists()) {
            int index;
            Status status = mlvalue_idx_map.GetIdx(name, index);
            ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
            node_values_[cur_idx] = index;
          }
          // else it's a missing optional input or output so leave the -1

          ++cur_idx;
        },
        include_missing_optional_defs);
  }
}

}  // namespace onnxruntime
