// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include "allocator.h"
#include "factory.h"
#include "sync_stream.h"
#include "telemetry.h"

#include "core/framework/run_options.h"
#include "core/framework/kernel_registry.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/plugin_ep/ep_kernel_registration.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/quantization/quantize_linear.h"

#include "ep/get_capability_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

using onnxruntime::ep::Api;

namespace {

// Sets `supported` to false for nodes whose WebGPU kernel exists but does not support the node's configuration.
OrtStatus* CheckNodeConfigSupported(const Ort::ConstNode& node, bool& supported) {
  supported = true;
  const auto value_exists = [](const auto& values, size_t idx) {
    return values.size() > idx && values[idx] != nullptr;
  };

  // Check for Attention
  if (node.GetOperatorType() == "Attention" && node.GetDomain() == kMSDomain) {
    const auto& inputs = node.GetInputs();
    const auto& outputs = node.GetOutputs();

    // Current implementation does not support mask_index(input[3]), past(input[4]) and past_seq_len(input[6])
    // Current implementation does not support present(output[1])
    if (value_exists(inputs, 3) || value_exists(inputs, 4) || value_exists(inputs, 6) ||
        value_exists(outputs, 1)) {
      supported = false;
      return nullptr;
    }

    // If attribute past_present_share_buffer is set, fallback to CPU
    for (const auto& attr : node.GetAttributes()) {
      if (attr.GetName() == "past_present_share_buffer") {
        int64_t val = 0;
        RETURN_IF_ERROR(attr.GetValue(val));
        if (val != 0) {
          supported = false;
        }
        break;
      }
    }
    return nullptr;
  }

  if (node.GetOperatorType() == "QuantizeLinear" && node.GetDomain() == kOnnxDomain) {
    for (const auto& attr : node.GetAttributes()) {
      if (attr.GetName() == "precision") {
        int64_t precision = 0;
        RETURN_IF_ERROR(attr.GetValue(precision));
        supported = IsQuantizeLinearPrecisionSupported(precision);
        break;
      }
    }
  }

  return nullptr;
}

}  // namespace

// Constructor
Ep::Ep(std::unique_ptr<IExecutionProvider> impl, Factory& factory, const OrtLogger& logger, const Config& config)
    : onnxruntime::ep::adapter::Ep{std::move(impl), config.cpu_allocator, config.device_allocator},
      factory_{factory},
      logger_{logger},
      config_{config},
      device_memory_info_{onnxruntime::ep::adapter::CreateApiMemoryInfo(config_.device_allocator->Info())} {
  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  Compile = nullptr;  // Per-kernel EP does not use Compile
  ReleaseNodeComputeInfos = nullptr;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  SetDynamicOptions = nullptr;  // Not implemented
  OnRunStart = OnRunStartImpl;
  OnRunEnd = OnRunEndImpl;
  CreateAllocator = CreateAllocatorImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  GetDefaultMemoryDevice = GetDefaultMemoryDeviceImpl;
  GetCompiledModelCompatibilityInfo = nullptr;  // Not a compiled EP
  IsConcurrentRunSupported = IsConcurrentRunSupportedImpl;
  IsGraphCaptureEnabled = IsGraphCaptureEnabledImpl;
  IsGraphCaptured = IsGraphCapturedImpl;
  ReplayGraph = ReplayGraphImpl;
  ReleaseCapturedGraph = ReleaseCapturedGraphImpl;
  GetGraphCaptureNodeAssignmentPolicy = GetGraphCaptureNodeAssignmentPolicyImpl;
}

Ep::~Ep() {
  if (run_counters_.HasData()) {
    LogRunSummary(/*is_final=*/true);
  }
}

void Ep::MaybeLogRunSummary() noexcept {
  const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - run_summary_last_emit_time_)
                              .count();
  if (run_summary_scheduler_.IsDue(elapsed_us)) {
    LogRunSummary(/*is_final=*/false);
  }
}

void Ep::LogRunSummary(bool is_final) noexcept {
  telemetry::RunSummaryInfo info;
  info.ep_instance_id = config_.telemetry_instance_id;
  info.is_final = is_final;
  info.runs = run_counters_.runs;
  info.run_start_failures = run_counters_.run_start_failures;
  info.run_end_failures = run_counters_.run_end_failures;
  info.first_run_duration_us = first_run_duration_us_;
  info.total_run_duration_us = run_counters_.total_run_duration_us;
  info.max_run_duration_us = run_counters_.max_run_duration_us;
  info.graph_replays = run_counters_.graph_replays;
  info.graph_replay_failures = run_counters_.graph_replay_failures;
  try {
    const int context_id = EpImpl()->GetDeviceId();
    info.program_cache_size = WebGpuContextFactory::GetContext(context_id).NumCachedPrograms();
  } catch (...) {
  }
  telemetry::LogRunSummary(info);

  run_counters_ = {};
  run_summary_scheduler_.OnEmitted();
  run_summary_last_emit_time_ = std::chrono::steady_clock::now();
}

// OrtEp interface implementations
const char* ORT_API_CALL Ep::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->factory_.GetName(&ep->factory_);
}

OrtStatus* ORT_API_CALL Ep::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                              OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN

  auto* ep_adapter = static_cast<Ep*>(this_ptr);
  auto& ep = *static_cast<WebGpuExecutionProvider*>(ep_adapter->EpImpl());
  Ort::ConstGraph ort_graph{graph};

  // Get all nodes in the graph
  std::vector<Ort::ConstNode> all_nodes = ort_graph.GetNodes();

  if (all_nodes.empty()) {
    return nullptr;  // No nodes to process
  }

  const bool collect_telemetry = telemetry::IsEnabled();
  telemetry::GraphPartitionInfo partition_info;
  telemetry::OpTypeCounter no_kernel_op_types;
  telemetry::OpTypeCounter unsupported_config_op_types;

  std::vector<const OrtNode*> candidate_nodes;
  std::vector<const OrtNode*> tentative_candidate_nodes;

  // For each node, check if we have a registered kernel for it
  for (const auto& node : all_nodes) {
    std::string ep_name = node.GetEpName();

    if (ep_name == kWebGpuExecutionProvider) {
      ++partition_info.preassigned_nodes;
      candidate_nodes.push_back(node);
      continue;
    }

    // Reject nodes already assigned to a different (non-CPU) EP
    if (!ep_name.empty() && ep_name != kCpuExecutionProvider) {
      ++partition_info.other_ep_nodes;
      continue;
    }

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def));

    if (kernel_def == nullptr) {
      LOGS(ep.GetEpLogger(), INFO) << "webgpu kernel not found in registries for Op type: "
                                   << node.GetOperatorType() << " node name: " << node.GetName();
      ++partition_info.no_kernel_nodes;
      if (collect_telemetry) {
        no_kernel_op_types.Add(node.GetDomain(), node.GetOperatorType());
      }
      continue;
    }

    auto cpu_node_names = ep.GetForceCpuNodeNames();
    if (std::find(cpu_node_names.begin(),
                  cpu_node_names.end(),
                  node.GetName()) != cpu_node_names.end()) {
      LOGS(ep.GetEpLogger(), INFO) << "Force CPU execution for node: " << node.GetName();
      ++partition_info.forced_cpu_nodes;
      continue;
    }

    // Check if the node is really supported by webgpu EP
    bool config_supported = true;
    RETURN_IF_ERROR(CheckNodeConfigSupported(node, config_supported));
    if (!config_supported) {
      ++partition_info.unsupported_config_nodes;
      if (collect_telemetry) {
        unsupported_config_op_types.Add(node.GetDomain(), node.GetOperatorType());
      }
      continue;
    }

    candidate_nodes.push_back(node);
    tentative_candidate_nodes.push_back(node);
  }

  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  RETURN_IF_ERROR(onnxruntime::ep::GetCpuPreferredNodes(*ort_graph,
                                                        *graph_support_info,
                                                        ep_adapter->GetOrtLogger(),
                                                        tentative_candidate_nodes,
                                                        cpu_preferred_nodes));

  for (const auto& node : candidate_nodes) {
    if (cpu_preferred_nodes.count(node) == 0) {
      RETURN_IF_ERROR(Api().ep.EpGraphSupportInfo_AddSingleNode(graph_support_info, node));
      ++partition_info.claimed_nodes;
    }
  }

  if (collect_telemetry) {
    // Bound the op type lists well below the telemetry string limit.
    constexpr size_t kMaxReportedOpTypes = 32;
    const std::string no_kernel_summary = no_kernel_op_types.ToString(kMaxReportedOpTypes);
    const std::string unsupported_config_summary = unsupported_config_op_types.ToString(kMaxReportedOpTypes);
    partition_info.ep_instance_id = ep_adapter->config_.telemetry_instance_id;
    partition_info.pass_index = ep_adapter->graph_partition_pass_count_;
    partition_info.total_nodes = static_cast<uint32_t>(all_nodes.size());
    partition_info.cpu_preferred_nodes = static_cast<uint32_t>(cpu_preferred_nodes.size());
    partition_info.no_kernel_op_types = no_kernel_summary;
    partition_info.no_kernel_op_type_count = static_cast<uint32_t>(no_kernel_op_types.DistinctCount());
    partition_info.unsupported_config_op_types = unsupported_config_summary;
    telemetry::LogGraphPartition(partition_info);
  }
  ++ep_adapter->graph_partition_pass_count_;

  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr,
    _Outptr_result_maybenull_ const OrtKernelRegistry** kernel_registry) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN

  *kernel_registry = nullptr;

  // For the WebGPU EP, delegate to the CreateKernelRegistry function
  // which properly constructs a registry using only public APIs
  auto* ep = static_cast<Ep*>(this_ptr);

  auto& webgpu_ep = *static_cast<WebGpuExecutionProvider*>(ep->EpImpl());

  *kernel_registry = *webgpu_ep.GetKernelRegistryImpl();
  return nullptr;

  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetPreferredDataLayoutImpl(_In_ OrtEp* this_ptr,
                                                       _Out_ OrtEpDataLayout* preferred_data_layout) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // Delegate to the underlying WebGPU EP's GetPreferredLayout()
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  *preferred_data_layout = static_cast<OrtEpDataLayout>(ep->EpImpl()->GetPreferredLayout());
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::ShouldConvertDataLayoutForOpImpl(_In_ OrtEp* this_ptr,
                                                             _In_z_ const char* domain,
                                                             _In_z_ const char* op_type,
                                                             _In_ OrtEpDataLayout target_data_layout,
                                                             _Outptr_ int* should_convert) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // DataLayout enum values map 1:1 to OrtEpDataLayout (NCHW=0, NHWC=1)
  auto* ep = static_cast<Ep*>(this_ptr);
  auto result = ep->EpImpl()->ShouldConvertDataLayoutForOp(domain, op_type,
                                                           static_cast<DataLayout>(target_data_layout));
  if (result.has_value()) {
    *should_convert = result.value() ? 1 : 0;
  } else {
    *should_convert = -1;
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::OnRunStartImpl(_In_ OrtEp* this_ptr,
                                           _In_ const OrtRunOptions* run_options) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  onnxruntime::RunOptions options{};
  // currently only option "gpu_graph_id" is used
  auto graph_annotation_str = Api().ort.GetRunConfigEntry(run_options, kOrtRunOptionsConfigCudaGraphAnnotation);
  if (graph_annotation_str != nullptr) {
    auto status = options.config_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, graph_annotation_str);
    if (!status.IsOK()) {
      return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                    status.ErrorMessage().c_str());
    }
  }
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->OnRunStart(options);
  if (!status.IsOK()) {
    ++ep->run_counters_.run_start_failures;
    ep->run_start_time_.reset();
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  ep->run_start_time_ = std::chrono::steady_clock::now();
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::OnRunEndImpl(_In_ OrtEp* this_ptr,
                                         _In_ const OrtRunOptions* /*run_options*/,
                                         _In_ bool sync_stream) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->OnRunEnd(sync_stream, {});
  if (ep->run_start_time_.has_value()) {
    const int64_t duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - *ep->run_start_time_)
                                    .count();
    ep->run_start_time_.reset();
    if (ep->first_run_duration_us_ == 0) {
      ep->first_run_duration_us_ = duration_us;
    }
    ep->run_counters_.AddRun(duration_us, status.IsOK());
  }
  ep->MaybeLogRunSummary();
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::IsConcurrentRunSupportedImpl(_In_ OrtEp* /*this_ptr*/, _Out_ bool* is_concurrent_run_supported) noexcept {
  *is_concurrent_run_supported = false;
  return nullptr;
}

bool ORT_API_CALL Ep::IsGraphCaptureEnabledImpl(_In_ const OrtEp* this_ptr) noexcept {
  auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->EpImpl()->IsGraphCaptureEnabled();
}

bool ORT_API_CALL Ep::IsGraphCapturedImpl(_In_ const OrtEp* this_ptr, _In_ int graph_annotation_id) noexcept {
  auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->EpImpl()->IsGraphCaptured(graph_annotation_id);
}

OrtStatus* ORT_API_CALL Ep::ReplayGraphImpl(_In_ OrtEp* this_ptr, _In_ int graph_annotation_id) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->ReplayGraph(graph_annotation_id);
  ep->run_counters_.AddGraphReplay(status.IsOK());
  ep->MaybeLogRunSummary();
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::ReleaseCapturedGraphImpl(_In_ OrtEp* this_ptr, _In_ int graph_annotation_id) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  auto status = ep->EpImpl()->ReleaseCapturedGraph(graph_annotation_id);
  if (!status.IsOK()) {
    return Api().ort.CreateStatus(static_cast<OrtErrorCode>(status.Code()),
                                  status.ErrorMessage().c_str());
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtGraphCaptureNodeAssignmentPolicy ORT_API_CALL Ep::GetGraphCaptureNodeAssignmentPolicyImpl(
    _In_ const OrtEp* this_ptr) noexcept {
  auto* ep = static_cast<const Ep*>(this_ptr);
  return ep->EpImpl()->GetGraphCaptureNodeAssignmentPolicy();
}

OrtStatus* ORT_API_CALL Ep::CreateSyncStreamForDeviceImpl(
    OrtEp* this_ptr, const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep_adapter = static_cast<Ep*>(this_ptr);
  auto& ep = *static_cast<WebGpuExecutionProvider*>(ep_adapter->EpImpl());
  ORT_ENFORCE(Api().ep.MemoryDevice_GetDeviceType(memory_device) == OrtMemoryInfoDeviceType_GPU &&
                  static_cast<int64_t>(Api().ep.MemoryDevice_GetDeviceId(memory_device)) == ep.GetDeviceId(),
              "Unsupported memory device for WebGPU Session stream.");
  *stream = CreateWebGpuSyncStream(ep);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

OrtStatus* ORT_API_CALL Ep::GetDefaultMemoryDeviceImpl(
    const OrtEp* this_ptr, const OrtMemoryDevice** memory_device) noexcept {
  const auto* ep = static_cast<const Ep*>(this_ptr);
  *memory_device = Api().ep.MemoryInfo_GetMemoryDevice(ep->device_memory_info_);
  return nullptr;
}

OrtStatus* ORT_API_CALL Ep::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                _In_ const OrtMemoryInfo* memory_info,
                                                _Outptr_result_maybenull_ OrtAllocator** allocator) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto* ep = static_cast<Ep*>(this_ptr);
  Ort::ConstMemoryInfo ort_memory_info{memory_info};
  // Wrap the existing Session allocator, unlike Factory::CreateAllocatorImpl's Env path.
  // The wrapper retains the allocator implementation, not the EP borrowed by its getters.
  if (ort_memory_info.GetAllocatorType() == OrtReadOnlyAllocator) {
    *allocator = CreateWebGpuSessionAllocator(ep->config_.initializer_allocator);
  } else {
    *allocator = CreateWebGpuSessionAllocator(ep->config_.device_allocator);
  }
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
