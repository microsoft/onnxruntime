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
  GetProfiler = GetProfilerImpl;

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

// ---------------------------------------------------------------------------
// ExampleKernelEpProfiler — demonstrates how a plugin EP reports profiling events.
// ---------------------------------------------------------------------------

namespace {

struct ExampleKernelEpProfiler : OrtEpProfilerImpl {
  const OrtEpApi& ep_api;
  bool is_profiling = false;

  explicit ExampleKernelEpProfiler(const OrtEpApi& api) : OrtEpProfilerImpl{}, ep_api(api) {
    ort_version_supported = ORT_API_VERSION;
    Release = ReleaseImpl;
    StartProfiling = StartProfilingImpl;
    EndProfiling = EndProfilingImpl;
    Start = nullptr;  // Optional — not needed for this example.
    Stop = nullptr;
  }

  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
    delete static_cast<ExampleKernelEpProfiler*>(this_ptr);
  }

  static bool ORT_API_CALL StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                              int64_t /*profiling_start_time_ns*/) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    self->is_profiling = true;
    return true;
  }

  static OrtStatus* ORT_API_CALL EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                  int64_t /*start_time_ns*/,
                                                  OrtEpProfilingEventsContainer* events_container) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    self->is_profiling = false;

    // Report a single summary event so the integration test can verify EP events appear in the profile.
    const char* arg_keys[] = {"ep_name"};
    const char* arg_values[] = {"ExampleKernelEp"};

    OrtEpProfilingEvent event{};
    event.ort_version_supported = ORT_API_VERSION;
    event.category = OrtEpProfilingEventCategory_SESSION;
    event.process_id = 0;
    event.thread_id = 0;
    event.event_name = "ExampleKernelEp_profiling_event";
    event.timestamp_us = 0;
    event.duration_us = 0;
    event.arg_keys = arg_keys;
    event.arg_values = arg_values;
    event.num_args = 1;

    return self->ep_api.EpProfilingEventsContainer_AddEvents(events_container, &event, 1);
  }
};

}  // namespace

/*static*/
OrtStatus* ORT_API_CALL ExampleKernelEp::GetProfilerImpl(OrtEp* this_ptr,
                                                         OrtEpProfilerImpl** profiler) noexcept {
  try {
    ExampleKernelEp* ep = static_cast<ExampleKernelEp*>(this_ptr);
    *profiler = new ExampleKernelEpProfiler(ep->ep_api_);
    return nullptr;
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}
