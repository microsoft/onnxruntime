// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <gsl/span>
#include <array>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
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
  int64_t profiling_start_time_ns = 0;

  // Stack-based timer for tracking operator execution.
  struct PendingOp {
    uint64_t start_offset_us;  // microseconds since profiling start (from Start's id param)
    std::chrono::high_resolution_clock::time_point wall_start;
  };

  struct CompletedOp {
    uint64_t start_offset_us;
    int64_t duration_us;
  };

  std::vector<PendingOp> pending_ops;
  std::vector<CompletedOp> completed_ops;

  explicit ExampleKernelEpProfiler(const OrtEpApi& api) : OrtEpProfilerImpl{}, ep_api(api) {
    ort_version_supported = ORT_API_VERSION;
    Release = ReleaseImpl;
    StartProfiling = StartProfilingImpl;
    EndProfiling = EndProfilingImpl;
    Start = StartImpl;
    Stop = StopImpl;
  }

  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept {
    delete static_cast<ExampleKernelEpProfiler*>(this_ptr);
  }

  // A single mutex is sufficient for this example profiler implementation.
  // It protects access to shared mutable state (is_profiling, profiling_start_time_ns,
  // pending_ops, completed_ops) across Start/Stop/StartProfiling/EndProfiling calls.
  static std::mutex& GetProfilerMutex() {
    static std::mutex m;
    return m;
  }

  static bool ORT_API_CALL StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                              int64_t profiling_start_time_ns) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    std::lock_guard<std::mutex> lock(GetProfilerMutex());
    self->is_profiling = true;
    self->profiling_start_time_ns = profiling_start_time_ns;
    self->pending_ops.clear();
    self->completed_ops.clear();
    return true;
  }

  static void ORT_API_CALL StartImpl(OrtEpProfilerImpl* this_ptr, uint64_t id) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    std::lock_guard<std::mutex> lock(GetProfilerMutex());
    if (!self->is_profiling) return;
    self->pending_ops.push_back({id, std::chrono::high_resolution_clock::now()});
  }

  static void ORT_API_CALL StopImpl(OrtEpProfilerImpl* this_ptr, uint64_t /*id*/) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    std::lock_guard<std::mutex> lock(GetProfilerMutex());
    if (!self->is_profiling || self->pending_ops.empty()) return;
    auto pending = self->pending_ops.back();
    self->pending_ops.pop_back();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now() - pending.wall_start)
                        .count();
    self->completed_ops.push_back({pending.start_offset_us, duration});
  }

  static OrtStatus* ORT_API_CALL EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                  int64_t start_time_ns,
                                                  OrtEpProfilingEventsContainer* events_container) noexcept {
    auto* self = static_cast<ExampleKernelEpProfiler*>(this_ptr);
    std::lock_guard<std::mutex> lock(GetProfilerMutex());
    self->is_profiling = false;

    // Build per-op NODE events from completed ops.
    std::vector<std::string> event_names;
    std::vector<OrtEpProfilingEvent> events;
    event_names.reserve(self->completed_ops.size() + 1);
    events.reserve(self->completed_ops.size() + 1);

    for (size_t i = 0; i < self->completed_ops.size(); ++i) {
      const auto& op = self->completed_ops[i];
      event_names.push_back("ep_op_" + std::to_string(i));

      OrtEpProfilingEvent ev{};
      ev.ort_version_supported = ORT_API_VERSION;
      ev.category = OrtEpProfilingEventCategory_NODE;
      ev.process_id = 0;
      ev.thread_id = 0;
      ev.event_name = event_names.back().c_str();
      // Use the same time base as ORT's profiling::EventRecord:
      // microseconds relative to profiling_start_time_.
      ev.timestamp_us = static_cast<int64_t>(op.start_offset_us);
      ev.duration_us = op.duration_us;
      ev.num_args = 0;
      ev.arg_keys = nullptr;
      ev.arg_values = nullptr;
      events.push_back(ev);
    }

    // Also emit a summary SESSION event (preserves backward compatibility with existing test).
    const char* arg_keys[] = {"ep_name"};
    const char* arg_values[] = {"ExampleKernelEp"};

    event_names.push_back("ExampleKernelEp_profiling_event");

    OrtEpProfilingEvent summary{};
    summary.ort_version_supported = ORT_API_VERSION;
    summary.category = OrtEpProfilingEventCategory_SESSION;
    summary.process_id = 0;
    summary.thread_id = 0;
    summary.event_name = event_names.back().c_str();
    summary.timestamp_us = 0;
    summary.duration_us = 0;
    summary.arg_keys = arg_keys;
    summary.arg_values = arg_values;
    summary.num_args = 1;
    events.push_back(summary);

    OrtStatus* status = self->ep_api.EpProfilingEventsContainer_AddEvents(
        events_container, events.data(), events.size());

    self->completed_ops.clear();
    self->pending_ops.clear();

    return status;
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
