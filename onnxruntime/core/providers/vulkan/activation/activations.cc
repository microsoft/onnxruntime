
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/activation/activations.h"

#include "include/ncnn/layer/vulkan/sigmoid_vulkan.h"

#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
namespace vulkan {

bool ClipKernel::IsSupported(bool use_kompute, const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                             const logging::Logger& logger) {
  if (use_kompute) {
    return false;
  }

  if (node.SinceVersion() < 11) {
    // min/max are attributes
    return true;
  }

  const auto& input_defs = node.InputDefs();
  const auto num_inputs = input_defs.size();
  const NodeArg* min = num_inputs > 1 ? input_defs[1] : nullptr;
  const NodeArg* max = num_inputs > 2 ? input_defs[2] : nullptr;

  // min/max if provided must be constant. we can use the default values if they're missing.
  bool supported =
      (min == nullptr || min->Exists() == false || graph_viewer.GetConstantInitializer(min->Name()) != nullptr) &&
      (max == nullptr || max->Exists() == false || graph_viewer.GetConstantInitializer(max->Name()) != nullptr);

  if (!supported) {
    LOGS(logger, VERBOSE) << "Clip min/max must be constant";
  }

  return supported;
}
}  // namespace vulkan
}  // namespace onnxruntime
