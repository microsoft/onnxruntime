// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#ifdef __APPLE__
#include <sys/utsname.h>
#include <TargetConditionals.h>
#endif

#include "helper.h"
#include <core/graph/graph_viewer.h>

#include "core/providers/common.h"
#include "core/providers/coreml/model/host_utils.h"
#include "op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

class ModelBuilder;
struct OpBuilderInputParams;

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger) {
  const auto* shape_proto = node_arg.Shape();
  if (!shape_proto) {
    LOGS(logger, WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // We already checked the shape has no dynamic dimension
  for (const auto& dim : shape_proto->dim()) {
    shape.push_back(dim.dim_value());
  }

  return true;
}

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const logging::Logger& logger) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    OpBuilderInputParams input_params(graph_viewer);
    return op_builder->IsOpSupported(node, input_params, logger);
  } else {
    return false;
  }
}

bool IsInputSupported(const NodeArg& input, const std::string& parent_name, const logging::Logger& logger) {
  const auto& input_name = input.Name();
  const auto* shape_proto = input.Shape();
  // We do not support input with no shape
  if (!shape_proto) {
    LOGS(logger, VERBOSE) << "Input [" << input_name << "] of [" << parent_name
                          << "] has not shape";
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    // For now we do not support dynamic shape
    if (!dim.has_dim_value()) {
      LOGS(logger, WARNING) << "Dynamic shape is not supported for now, for input:" << input_name;
      return false;
    }

    // For some undocumented reason, Apple CoreML framework will fail loading the model if the model
    // input has dimension > 16384
    // See this issue, https://github.com/apple/coremltools/issues/1003
    if (dim.dim_value() > 16384) {
      LOGS(logger, WARNING) << "CoreML does not support input dim > 16384, input:" << input_name
                            << ", actual dim: " << dim.dim_value();
      return false;
    }
  }

  return true;
}

std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer, const logging::Logger& logger) {
  std::vector<std::vector<size_t>> supported_node_groups;
  if (!util::HasRequiredBaseOS()) {
    LOGS(logger, WARNING) << "All ops will fallback to CPU EP, because we do not have supported OS";
    return supported_node_groups;
  }

  for (const auto* input : graph_viewer.GetInputs()) {
    if (!IsInputSupported(*input, "graph", logger)) {
      return supported_node_groups;
    }
  }

  std::vector<size_t> supported_node_group;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    auto node_idx = node_indices[i];
    const auto* node(graph_viewer.GetNode(node_idx));
    bool supported = IsNodeSupported(*node, graph_viewer, logger);
    LOGS(logger, VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << node_idx
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_node_group.push_back(node_idx);
    } else {
      if (!supported_node_group.empty()) {
        supported_node_groups.push_back(supported_node_group);
        supported_node_group.clear();
      }
    }
  }

  if (!supported_node_group.empty()) {
    supported_node_groups.push_back(supported_node_group);
  }

  return supported_node_groups;
}

bool HasNeuralEngine(const logging::Logger& logger) {
  bool has_neural_engine = false;

#ifdef __APPLE__
  struct utsname system_info;
  uname(&system_info);
  LOGS(logger, VERBOSE) << "Current Apple hardware info: " << system_info.machine;

#if TARGET_OS_IPHONE
  // utsname.machine has device identifier. For example, identifier for iPhone Xs is "iPhone11,2".
  // Since Neural Engine is only available for use on A12 and later, major device version in the
  // identifier is checked for these models:
  // A12: iPhone XS (11,2), iPad Mini - 5th Gen (11,1)
  // A12X: iPad Pro - 3rd Gen (8,1)
  // For more information, see https://www.theiphonewiki.com/wiki/Models
  size_t str_len = strlen(system_info.machine);
  if (str_len > 4 && strncmp("iPad", system_info.machine, 4) == 0) {
    const int major_version = atoi(system_info.machine + 4);
    has_neural_engine = major_version >= 8;  // There are no device between iPad 8 and 11.
  } else if (str_len > 6 && strncmp("iPhone", system_info.machine, 6) == 0) {
    const int major_version = atoi(system_info.machine + 6);
    has_neural_engine = major_version >= 11;
  }
#elif TARGET_OS_OSX && TARGET_CPU_ARM64
  // Only Mac with arm64 CPU (Apple Silicon) has ANE.
  has_neural_engine = true;
#endif  // #if TARGET_OS_IPHONE
#else
  // In this case, we are running the EP on non-apple platform, which means we are running the model
  // conversion with CoreML EP enabled, for this we always assume the target system has Neural Engine
  LOGS(logger, VERBOSE) << "HasNeuralEngine running on non-Apple hardware for model conversion only";
  has_neural_engine = true;
#endif  // #ifdef __APPLE__

  return has_neural_engine;
}

}  // namespace coreml
}  // namespace onnxruntime