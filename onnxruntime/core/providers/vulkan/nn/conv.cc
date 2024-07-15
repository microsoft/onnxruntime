// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/nn/conv.h"

namespace onnxruntime {
namespace vulkan {
Status ConvKernel::CreateNcnnKernel(const GraphViewer* /*graph_viewer*/, ValueIndexes& value_indexes) {
  ncnn::ParamDict params;

  // const auto& node = Node();
  // const auto& input_defs = node.InputDefs();

  // figure out op_type

  return VulkanKernel::SetupNcnnLayer(value_indexes, params);
}

}  // namespace vulkan
}  // namespace onnxruntime
