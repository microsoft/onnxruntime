// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/node.h"
#include "./vai_assert.h"

#include "attr_proto.h"
#include "vaip/node_arg.h"
#include "core/providers/shared_library/provider_api.h"

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
    assert(output != nullptr);
    // Optional Outputs
    // Some operators have outputs that are optional. When an actual output parameter of an operator is not specified, the operator implementation MAY forgo computing values for such outputs.
    // There are two ways to leave an optional input or output unspecified: the first, available only for trailing inputs and outputs, is to simply not provide that input; the second method is to use an empty string in place of an input or output name.
    // so optional output maybe output != null && output->Exists() return false
    // Our processing : nullptr means optional output , and clinet code needs to handle nullptr
    if (output->Exists()) {
      ret[i] = output;
    } else {
      ret[i] = nullptr;
    }
  }
  return vaip_core::DllSafe(ret);
}
}  // namespace vaip
