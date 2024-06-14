// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"

#include <algorithm>
#include <vector>

#ifdef __APPLE__
#include <sys/utsname.h>
#include <TargetConditionals.h>
#endif

#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/builders/op_builder.h"
#include "core/providers/coreml/coreml_provider_factory.h"  // for COREMLFlags
#include "core/providers/coreml/model/host_utils.h"
#include "core/providers/coreml/shape_utils.h"

namespace onnxruntime {
namespace coreml {

OpBuilderInputParams MakeOpBuilderParams(const GraphViewer& graph_viewer,
                                         int32_t coreml_version,
                                         uint32_t coreml_flags) {
  return OpBuilderInputParams{graph_viewer,
                              coreml_version,
                              (coreml_flags & COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES) != 0,
                              (coreml_flags & COREML_FLAG_CREATE_MLPROGRAM) != 0};
}

const IOpBuilder* GetOpBuilder(const Node& node) {
  const auto& op_builders = GetOpBuilders();
  const auto it = op_builders.find(node.OpType());
  if (it != op_builders.cend()) {
    return it->second;
  }

  return nullptr;
}

bool IsNodeSupported(const Node& node, const OpBuilderInputParams& input_params, const logging::Logger& logger) {
  const auto* op_builder = GetOpBuilder(node);
  if (op_builder) {
    return op_builder->IsOpSupported(node, input_params, logger);
  } else {
    return false;
  }
}

bool IsInputSupported(const Node& node, const NodeArg& input,
                      const OpBuilderInputParams& input_params, const logging::Logger& logger) {
  if (!input.Exists()) {
    // optional input that is not provided
    return true;
  }

  const auto& input_name = input.Name();
  std::vector<int64_t> shape;
  // We do not support input with no shape
  if (!GetShape(input, shape, logger)) {
    LOGS(logger, VERBOSE) << MakeString("Input [", input_name, "] of Node [", node.Name(), "] type [", node.OpType(),
                                        "] has no shape");
    return false;
  }

  if (input_params.only_allow_static_input_shapes && !IsStaticShape(shape)) {
    LOGS(logger, VERBOSE) << "CoreML EP is set to only allow static input shapes. Input has a dynamic shape. Input: "
                          << input_name << ", shape: " << Shape2String(shape);
    return false;
  }

  for (const auto dim : shape) {
    // For some undocumented reason, Apple CoreML framework will fail loading the model if the model
    // input has dimension > 16384
    // See this issue, https://github.com/apple/coremltools/issues/1003
    // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf has maximum texture widths which may be the
    // root cause.
    if (dim > 16384) {
      LOGS(logger, WARNING) << "CoreML does not support input dim > 16384. Input:" << input_name
                            << ", shape: " << Shape2String(shape);
      return false;
    }

    if (dim == 0) {
      if (node.OpType() == "Resize" && &input == node.InputDefs()[1]) {
        // one special case. Resize 'roi' input was originally a required input but is rarely used.
        // ROI is not supported in the CoreML implementation so we will ignore the value, but is often added
        // (at least in the unit tests) as an initializer with shape {0}.
      } else {
        LOGS(logger, WARNING) << "CoreML does not support shapes with dimension values of 0. Input:" << input_name
                              << ", shape: " << Shape2String(shape);
        return false;
      }
    }
  }

  // Limit input shape rank to 5.
  // CoreML doesn't currently support shapes with rank greater that 5.
  // https://github.com/apple/coremltools/issues/832
  if (shape.size() > 5) {
    LOGS(logger, VERBOSE) << "CoreML EP doesn't allow input shapes with rank greater than 5. Input: "
                          << input_name << ", shape: " << Shape2String(shape);
    return false;
  }

  return true;
}

std::unordered_set<const Node*> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                  const OpBuilderInputParams& input_params,
                                                  const logging::Logger& logger) {
  std::unordered_set<const Node*> supported_nodes{};

  for (const auto& node : graph_viewer.Nodes()) {
    const bool supported = IsNodeSupported(node, input_params, logger);
    LOGS(logger, VERBOSE) << "Operator type: [" << node.OpType()
                          << "] index: [" << node.Index()
                          << "] name: [" << node.Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_nodes.insert(&node);
    }
  }

  return supported_nodes;
}

bool CheckIsConstantInitializer(const NodeArg& node_arg, const GraphViewer& graph_viewer,
                                const logging::Logger& logger, std::string_view input_description) {
  if (graph_viewer.GetConstantInitializer(node_arg.Name()) == nullptr) {
    LOGS(logger, VERBOSE) << input_description << " (NodeArg name: '" << node_arg.Name()
                          << "') is not a constant initializer tensor";
    return false;
  }
  return true;
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
  size_t str_len = strnlen(system_info.machine, onnxruntime::kMaxStrLen);
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
  LOGS(logger, INFO) << "HasNeuralEngine running on non-Apple hardware. "
                        "Returning true to enable model conversion and local testing of CoreML EP implementation. "
                        "No CoreML model will be compiled or run.";
  has_neural_engine = true;
#endif  // #ifdef __APPLE__

  return has_neural_engine;
}

}  // namespace coreml
}  // namespace onnxruntime
