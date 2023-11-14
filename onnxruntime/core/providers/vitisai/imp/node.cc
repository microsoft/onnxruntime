// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/node.h"
#include "vaip/vai_assert.h"

#include "attr_proto.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_arg.h"
#include "vaip/node_arg.h"

namespace vaip {

vaip_core::DllSafe<std::vector<NodeInput>> node_get_inputs(const Node& node) {
  auto input_defs = node.InputDefs();
  auto ret = std::vector<NodeInput>(input_defs.size());
  int index = 0;
  for (auto input : input_defs) {
    ret[index].node_arg = input;
    ret[index].node = nullptr;
    index = index + 1;
  }
  for (auto iter = node.InputEdgesBegin(); iter != node.InputEdgesEnd();
       ++iter) {
    auto dst_idx = static_cast<size_t>(iter->GetDstArgIndex());
    if (dst_idx < ret.size()) {
      // ignore implicit nodes.
      ret[dst_idx].node = &iter->GetNode();
    }
  }
  return vaip_core::DllSafe(ret);
}

vaip_core::DllSafe<std::vector<const NodeArg*>> node_get_output_node_args(const Node& node) {
  auto outputs = node.OutputDefs();
  auto size = outputs.size();
  auto ret = std::vector<const NodeArg*>(size);
  for (auto i = 0u; i < size; ++i) {
    auto output = outputs[i];
    ret[i] = output;
    assert(output != nullptr);
    vai_assert(output->Exists(), std::string("output must exists. name=" + output->Name()));
  }
  return vaip_core::DllSafe(ret);
}

vaip_core::DllSafe<std::vector<int64_t>> node_get_output_shape(const Node& node, int index) {
  auto outputs = node.OutputDefs();
  assert((size_t)index < outputs.size());
  return node_arg_get_shape_i64(*outputs[index]);
}

}  // namespace vaip
