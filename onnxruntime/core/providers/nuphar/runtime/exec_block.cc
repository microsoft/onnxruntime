// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/runtime/exec_block.h"

// all execution block headers
#include "core/providers/nuphar/runtime/sequential/basic.h"
#include "core/providers/nuphar/runtime/sequential/loop.h"

#include "core/providers/nuphar/common/nuphar_subgraph.h"

namespace onnxruntime {
namespace nuphar {

void CreateExecBlock(std::vector<std::unique_ptr<ExecBlock>>& exec_blocks,
                     const NupharFuncInfo* func_info,
                     const nuphar::NupharSubgraphUnit& subgraph,
                     bool /*enable_tiling*/) {
  if (subgraph.IsSingleNode() && subgraph.nodes.front()->OpType() == "Scan") {
    exec_blocks.push_back(
        std::move(std::make_unique<LoopExecBlock>(func_info, "nuphar_exec_" + subgraph.UniqueId())));
  } else {
    exec_blocks.push_back(
        std::move(std::make_unique<BasicExecBlock>(func_info, "nuphar_exec_" + subgraph.UniqueId())));
  }
}

}  // namespace nuphar
}  // namespace onnxruntime
