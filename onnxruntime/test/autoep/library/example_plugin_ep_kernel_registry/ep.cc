// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <gsl/span>
#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ep_factory.h"
#include "../plugin_ep_utils.h"

ExampleKernelEp::ExampleKernelEp(ExampleKernelEpFactory& factory, const Config& config, const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      factory_{factory},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      name_{factory.GetEpName()},
      config_{config},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;

  // This is not a compiling EP, so don't need the following
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;

  IGNORE_ORTSTATUS(ort_api_.Logger_LogMessage(&logger_,
                                              OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                              ("ExampleKernelEp has been created with name " + name_).c_str(),
                                              ORT_FILE, __LINE__, __FUNCTION__));
}

ExampleKernelEp::~ExampleKernelEp() = default;

/*static*/
const char* ORT_API_CALL ExampleKernelEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const ExampleKernelEp*>(this_ptr);
  return ep->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                           OrtEpGraphSupportInfo* graph_support_info) noexcept {
  try {
    ExampleKernelEp* ep = static_cast<ExampleKernelEp*>(this_ptr);

    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

    if (all_nodes.empty()) {
      return nullptr;  // No nodes to process
    }

    // Collect candidate nodes that this EP may support.
    std::vector<Ort::ConstNode> candidate_nodes;

    for (const auto& node : all_nodes) {
      std::string op_type = node.GetOperatorType();

      if (op_type == "Relu" || op_type == "Squeeze" || op_type == "If" || op_type == "Loop" || op_type == "Scan") {
        candidate_nodes.push_back(node);
      } else if (op_type == "Mul" || op_type == "Sub") {
        std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();

        // Note: ONNX shape inference should ensure Mul/Sub has two inputs.
        std::optional<std::vector<int64_t>> input_0_shape = GetTensorShape(inputs[0]);
        std::optional<std::vector<int64_t>> input_1_shape = GetTensorShape(inputs[1]);

        if (!input_0_shape.has_value() || !input_1_shape.has_value()) {
          continue;  // Unable to get input shapes (non-tensor).
        }

        if (!AreShapesStaticAndEqual(*input_0_shape, *input_1_shape)) {
          continue;  // Don't support broadcasting and dynamic dimensions.
        }

        candidate_nodes.push_back(node);
      }
    }

    // Mark candidate nodes as supported if we have a registered kernel.
    for (const auto& node : candidate_nodes) {
      const OrtKernelDef* kernel_def = nullptr;
      RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def));

      if (kernel_def != nullptr) {
        RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddSingleNode(graph_support_info, node));
      }
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEp::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr,
    _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept {
  ExampleKernelEp* ep = static_cast<ExampleKernelEp*>(this_ptr);

  *kernel_registry = nullptr;

  // Get the cached kernel registry from parent factory to avoid recreating the kernel registry for every EP instance.
  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}
