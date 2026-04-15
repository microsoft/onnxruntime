// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep.h"
#include "cuda_ep_factory.h"
#include "cuda_stream_plugin.h"
#include "cuda_graph_plugin.h"
#include "core/providers/cuda/plugin/cuda_kernel_adapter.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/framework/allocator.h"
#include "ep/get_capability_utils.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "core/graph/constants.h"

namespace onnxruntime {
namespace cuda_plugin {

namespace {

std::unique_ptr<IExecutionProvider> CreateCudaPluginProvider(std::string_view ep_name, const OrtEp* ort_ep) {
  return std::make_unique<::onnxruntime::CUDAExecutionProvider>(std::string{ep_name}, ort_ep);
}

AllocatorPtr CreateCudaPluginTempSpaceAllocator(int device_id) {
  return std::make_shared<::onnxruntime::CUDAAllocator>(device_id, ::onnxruntime::CUDA);
}

AllocatorPtr CreateCudaPluginTempSpaceCpuAllocator() {
  return ::onnxruntime::CPUAllocator::DefaultInstance();
}

void DestroyCudaStreamForDevice(cudaStream_t stream, int device_id);

cudaStream_t CreateCudaStreamForDevice(int device_id) {
  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);
  cudaStream_t stream = nullptr;

  try {
    PL_CUDA_CALL_THROW(cudaSetDevice(device_id));
    PL_CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (restore_prev_device) {
      PL_CUDA_CALL_THROW(cudaSetDevice(prev_device));
    }
  } catch (...) {
    if (stream != nullptr) {
      DestroyCudaStreamForDevice(stream, device_id);
    }
    if (restore_prev_device) {
      static_cast<void>(cudaSetDevice(prev_device));
    }
    throw;
  }

  return stream;
}

void DestroyCudaStreamForDevice(cudaStream_t stream, int device_id) {
  if (stream == nullptr) {
    return;
  }

  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);
  static_cast<void>(cudaSetDevice(device_id));
  static_cast<void>(cudaStreamDestroy(stream));
  if (restore_prev_device) {
    static_cast<void>(cudaSetDevice(prev_device));
  }
}

}  // namespace

struct CudaEp::PerThreadContext {
  explicit PerThreadContext(int device_id)
      : device_id(device_id),
        graph_stream(CreateCudaStreamForDevice(device_id)),
        cuda_graph(graph_stream) {
  }

  ~PerThreadContext() {
    // Destroy captured graph execs before destroying the stream they replay on.
    cuda_graph.Reset();
    DestroyCudaStreamForDevice(graph_stream, device_id);
    graph_stream = nullptr;
  }

  int device_id;
  cudaStream_t graph_stream = nullptr;
  CudaGraphManager cuda_graph;
  size_t pre_capture_free_mem = 0;
};

CudaEp::CudaEp(CudaEpFactory& factory, const Config& config, const OrtLogger& logger)
    : onnxruntime::ep::adapter::Ep{CreateCudaPluginProvider(factory.GetEpName(), static_cast<const OrtEp*>(this)),
                                   CreateCudaPluginTempSpaceCpuAllocator(),
                                   CreateCudaPluginTempSpaceAllocator(config.device_id)},
      factory_(factory),
      name_(factory.GetEpName()),
      config_(config),
      logger_(logger) {
  ort_version_supported = ORT_API_VERSION;

  // Set function pointers for kernel-registry-based EP
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  GetKernelRegistry = GetKernelRegistryImpl;
  GetPreferredDataLayout = GetPreferredDataLayoutImpl;
  ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
  CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
  Sync = SyncImpl;
  IsConcurrentRunSupported = IsConcurrentRunSupportedImpl;
  OnRunStart = config_.enable_cuda_graph ? OnRunStartImpl : nullptr;
  OnRunEnd = config_.enable_cuda_graph ? OnRunEndImpl : nullptr;

  // Not a compile-based EP
  Compile = nullptr;
  ReleaseNodeComputeInfos = nullptr;

  // Graph capture/replay — always set so ORT can query capabilities
  IsGraphCaptureEnabled = IsGraphCaptureEnabledImpl;
  IsGraphCaptured = IsGraphCapturedImpl;
  ReplayGraph = ReplayGraphImpl;
  GetGraphCaptureNodeAssignmentPolicy = GetGraphCaptureNodeAssignmentPolicyImpl;

  const OrtApi& ort_api = factory_.GetOrtApi();
  Ort::Status log_status(ort_api.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_INFO,
                                                   "CUDA Plugin EP created",
                                                   ORT_FILE, __LINE__, __FUNCTION__));

  // Store per-EP runtime configuration inside the adapter-wrapped execution
  // provider itself. Migrated kernels retrieve a shared config object at
  // compute time via GetCudaKernelAdapterRuntimeConfigForProvider().
  // Adding a new config field only requires updating
  // CudaKernelAdapterRuntimeConfig, CudaEp::Config, and the struct-initializer
  // below — no function-signature change.
  onnxruntime::cuda::detail::CudaKernelAdapterRuntimeConfig adapter_config;
  adapter_config.use_tf32 = config_.use_tf32;
  adapter_config.skip_layer_norm_strict_mode = config_.enable_skip_layer_norm_strict_mode;
  adapter_config.cudnn_conv_algo = config_.cudnn_conv_algo;
  adapter_config.cudnn_conv_use_max_workspace = config_.cudnn_conv_use_max_workspace;
  adapter_config.cudnn_conv1d_pad_to_nc1d = config_.cudnn_conv1d_pad_to_nc1d;
  adapter_config.fuse_conv_bias = config_.fuse_conv_bias;
  adapter_config.sdpa_kernel = config_.sdpa_kernel;
  adapter_config.device_id = config_.device_id;
  onnxruntime::cuda::SetCudaKernelAdapterRuntimeConfigForProvider(
      static_cast<const void*>(EpImpl()), adapter_config);

  // CUDA graph streams are created lazily per thread by PerThreadContext.
}

CudaEp::~CudaEp() {
  std::lock_guard<std::mutex> lock(per_thread_contexts_mutex_);
  for (const auto& cache_weak : per_thread_context_caches_) {
    auto cache = cache_weak.lock();
    if (!cache) {
      continue;
    }
    ORT_IGNORE_RETURN_VALUE(cache->erase(this));
  }
  per_thread_context_caches_.clear();
}

/*static*/
const char* ORT_API_CALL CudaEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const CudaEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetCapabilityImpl(
    OrtEp* this_ptr, const OrtGraph* ort_graph,
    OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  const OrtEpApi& ep_api = ep->factory_.GetEpApi();

  Ort::ConstGraph graph{ort_graph};
  std::vector<Ort::ConstNode> all_nodes = graph.GetNodes();

  if (all_nodes.empty()) {
    return nullptr;
  }

  // Three-phase filtering determines which graph nodes run on this EP:
  // Phase 1: Collect tentative nodes that have a registered CUDA kernel.
  // Phase 2: Filter out CPU-preferred nodes (cheap ops where device-to-host
  //          copy overhead would exceed the compute benefit).
  // Phase 3: Register remaining nodes as supported by this EP.

  // Phase 1: Collect tentative nodes — those for which we have a registered kernel.
  std::vector<const OrtNode*> candidate_nodes;
  candidate_nodes.reserve(all_nodes.size());
  std::vector<const OrtNode*> tentative_nodes;
  tentative_nodes.reserve(all_nodes.size());

  for (const auto& node : all_nodes) {
    std::string ep_name = node.GetEpName();
    if (!ep_name.empty()) {
      if (ep_name == ep->name_) {
        candidate_nodes.push_back(node);
      }
      continue;
    }

    const OrtKernelDef* kernel_def = nullptr;
    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(
        graph_support_info, node, &kernel_def));

    if (kernel_def != nullptr) {
      candidate_nodes.push_back(node);
      tentative_nodes.push_back(node);
    }
  }

  // Phase 2: Filter out CPU-preferred nodes (e.g., Shape, NonZero, small compute ops
  // that would be cheaper on CPU than incurring device-to-host copy overhead).
  std::unordered_set<const OrtNode*> cpu_preferred_nodes;
  RETURN_IF_ERROR(ep::GetCpuPreferredNodes(
      *ort_graph, *graph_support_info, ep->logger_,
      gsl::span<const OrtNode* const>(tentative_nodes.data(), tentative_nodes.size()),
      cpu_preferred_nodes));

  // Phase 3: Add final supported nodes (tentative minus CPU-preferred).
  // Resource budget enforcement is handled by the host after GetCapability returns.

  for (const OrtNode* ort_node : candidate_nodes) {
    if (cpu_preferred_nodes.count(ort_node) != 0) {
      continue;
    }

    RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_AddSingleNode(
        graph_support_info, ort_node));
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetKernelRegistryImpl(
    OrtEp* this_ptr,
    const OrtKernelRegistry** kernel_registry) noexcept {
  auto* ep = static_cast<CudaEp*>(this_ptr);
  *kernel_registry = nullptr;

  RETURN_IF_ERROR(ep->factory_.GetKernelRegistryForEp(*ep, kernel_registry));
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::GetPreferredDataLayoutImpl(
    OrtEp* this_ptr, OrtEpDataLayout* preferred_data_layout) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
#ifdef ENABLE_CUDA_NHWC_OPS
  *preferred_data_layout = ep->config_.prefer_nhwc ? OrtEpDataLayout_NHWC : OrtEpDataLayout_NCHW;
#else
  ORT_UNUSED_PARAMETER(ep);
  *preferred_data_layout = OrtEpDataLayout_NCHW;
#endif
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::ShouldConvertDataLayoutForOpImpl(
    OrtEp* this_ptr, const char* domain, const char* op_type,
    OrtEpDataLayout target_data_layout, int* should_convert) noexcept {
  ORT_UNUSED_PARAMETER(this_ptr);

  if (should_convert == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "should_convert must not be null.");
  }

  const char* safe_domain = domain != nullptr ? domain : "";
  const char* safe_op_type = op_type != nullptr ? op_type : "";

#ifndef ENABLE_CUDA_NHWC_OPS
  ORT_UNUSED_PARAMETER(safe_domain);
  ORT_UNUSED_PARAMETER(safe_op_type);
  ORT_UNUSED_PARAMETER(target_data_layout);
  *should_convert = 0;  // NHWC kernels are not compiled into this plugin build.
  return nullptr;
#else

  // Only convert to NHWC; for any other target layout, let ORT decide.
  if (target_data_layout != OrtEpDataLayout_NHWC) {
    *should_convert = -1;  // Let ORT decide
    return nullptr;
  }

  // ONNX domain ops that have NHWC kernel registrations.
  static const std::unordered_set<std::string_view> cuda_nhwc_onnx_ops{
      "BatchNormalization",
      "Conv",
      "ConvTranspose",
      "GlobalMaxPool",
      "MaxPool",
      "GlobalAveragePool",
      "AveragePool",
      "GridSample",
      "DepthToSpace",
      "SpaceToDepth",
      "LRN",
  };

  // Check ONNX domain (empty string) or MS domain (com.microsoft)
  bool is_onnx_domain = (safe_domain[0] == '\0');
  bool is_ms_domain = (std::strcmp(safe_domain, "com.microsoft") == 0);

  if (is_onnx_domain && cuda_nhwc_onnx_ops.count(safe_op_type) > 0) {
    *should_convert = 1;  // Convert
    return nullptr;
  }

  if (is_ms_domain && std::strcmp(safe_op_type, "GridSample") == 0) {
    *should_convert = 1;  // Convert
    return nullptr;
  }

  *should_convert = 0;  // Explicitly decline conversion for unsupported NHWC ops.
  return nullptr;
#endif
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::CreateSyncStreamForDeviceImpl(
    OrtEp* this_ptr, const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  const OrtEpApi& ep_api = ep->factory_.GetEpApi();

  auto mem_type = ep_api.MemoryDevice_GetMemoryType(memory_device);
  if (mem_type != OrtDeviceMemoryType_DEFAULT) {
    std::string error = "Invalid OrtMemoryDevice. Expected OrtDeviceMemoryType_DEFAULT(0). Got ";
    error += std::to_string(mem_type);
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
  }

  int device_id = ep_api.MemoryDevice_GetDeviceId(memory_device);
  if (device_id != ep->config_.device_id) {
    std::string error = "Invalid OrtMemoryDevice. Expected CUDA device ordinal ";
    error += std::to_string(ep->config_.device_id);
    error += " for this EP instance. Got ";
    error += std::to_string(device_id);
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, error.c_str());
  }

  auto cuda_stream = std::make_unique<CudaSyncStream>(ep->factory_, device_id, this_ptr);

  if (ep->config_.enable_cuda_graph) {
    // When CUDA graph capture is enabled, all operations on this thread must go
    // through the thread's graph stream so capture/replay sees the same stream
    // as kernels.
    RETURN_IF_ERROR(cuda_stream->InitHandlesWithExternalStream(ep->GetPerThreadContext().graph_stream));
  } else {
    RETURN_IF_ERROR(cuda_stream->InitHandles());
  }

  *stream = cuda_stream.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::SyncImpl(OrtEp* this_ptr) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);

  int prev_device = -1;
  const bool restore_prev_device = TryGetCurrentCudaDevice(prev_device);

  Ort::Status status = StatusFromCudaError(cudaSetDevice(ep->config_.device_id));
  if (status.IsOK()) {
    status = StatusFromCudaError(cudaDeviceSynchronize());
  }

  if (restore_prev_device) {
    Ort::Status restore_status = StatusFromCudaError(cudaSetDevice(prev_device));
    if (status.IsOK()) {
      status = std::move(restore_status);
    }
  }

  return status.release();

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::IsConcurrentRunSupportedImpl(
    OrtEp* this_ptr, bool* is_supported) noexcept {
  if (is_supported == nullptr) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "is_supported must not be null.");
  }

  ORT_UNUSED_PARAMETER(this_ptr);
  *is_supported = true;
  return nullptr;
}

// --- CUDA Graph callback implementations ---

const std::shared_ptr<CudaEp::PerThreadContextMap>& CudaEp::PerThreadContextCache() {
  thread_local const std::shared_ptr<PerThreadContextMap> per_thread_context_cache =
      std::make_shared<PerThreadContextMap>();
  return per_thread_context_cache;
}

CudaEp::PerThreadContext& CudaEp::GetPerThreadContext() const {
  const auto& per_thread_context_cache = PerThreadContextCache();

  auto cached_context_it = per_thread_context_cache->find(this);
  if (cached_context_it != per_thread_context_cache->end()) {
    return *cached_context_it->second;
  }

  auto context = std::make_shared<PerThreadContext>(config_.device_id);
  PerThreadContext& context_ref = *context;
  {
    std::lock_guard<std::mutex> lock(per_thread_contexts_mutex_);
    for (auto it = per_thread_context_caches_.begin(); it != per_thread_context_caches_.end();) {
      if (it->expired()) {
        it = per_thread_context_caches_.erase(it);
      } else {
        ++it;
      }
    }
    ORT_IGNORE_RETURN_VALUE(per_thread_context_caches_.insert(per_thread_context_cache));
  }

  auto insert_result = per_thread_context_cache->emplace(this, std::move(context));
  ORT_ENFORCE(insert_result.second);
  return context_ref;
}

CudaGraphAnnotation_t CudaEp::GetGraphAnnotationId(const OrtRunOptions* run_options) const {
  if (run_options == nullptr) {
    return kCudaGraphAnnotationDefault;
  }
  const char* value = factory_.GetOrtApi().GetRunConfigEntry(run_options, "gpu_graph_id");
  if (value == nullptr) {
    return kCudaGraphAnnotationDefault;
  }
  try {
    return std::stoi(value);
  } catch (const std::exception&) {
    return kCudaGraphAnnotationDefault;
  }
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunStartImpl(
    OrtEp* this_ptr, const OrtRunOptions* run_options) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  if (!ep->config_.enable_cuda_graph) {
    return nullptr;
  }

  // Recover from any previous failed run on this thread before deciding whether
  // this run will enter capture mode.
  IsThreadCapturingCudaGraph() = false;

  auto& context = ep->GetPerThreadContext();
  CudaGraphAnnotation_t id = ep->GetGraphAnnotationId(run_options);
  if (!context.cuda_graph.IsGraphCaptured(id) &&
      context.cuda_graph.IsGraphCaptureAllowed(id, ep->config_.min_num_runs_before_cuda_graph_capture)) {
    // Keep the current CUDA device aligned with the graph stream for the full
    // capture window. Kernel Compute() skips cudaSetDevice() while capturing.
    PL_CUDA_CALL_THROW(cudaSetDevice(ep->config_.device_id));

    // Record free GPU memory before capture for allocation-during-capture detection.
    context.pre_capture_free_mem = 0;
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      context.pre_capture_free_mem = free_mem;
    }
    context.cuda_graph.CaptureBegin(id);
    IsThreadCapturingCudaGraph() = true;
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::OnRunEndImpl(
    OrtEp* this_ptr, const OrtRunOptions* run_options, bool sync_stream) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  if (!ep->config_.enable_cuda_graph) {
    return nullptr;
  }

  // Always clear the flag before leaving this callback so a failed capture or
  // failed replay cannot poison later runs on the same thread.
  IsThreadCapturingCudaGraph() = false;

  auto& context = ep->GetPerThreadContext();
  CudaGraphAnnotation_t id = ep->GetGraphAnnotationId(run_options);
  bool replayed_graph = false;
  if (!context.cuda_graph.IsGraphCaptured(id)) {
    if (context.cuda_graph.IsGraphCaptureAllowed(id, ep->config_.min_num_runs_before_cuda_graph_capture)) {
      context.cuda_graph.CaptureEnd(id);

      // Check if GPU memory was allocated during capture (which would make the
      // captured graph invalid since CUDA graph replay cannot reproduce allocations).
      if (context.pre_capture_free_mem > 0) {
        size_t post_free_mem = 0;
        size_t total_mem = 0;
        if (cudaMemGetInfo(&post_free_mem, &total_mem) == cudaSuccess) {
          if (post_free_mem < context.pre_capture_free_mem) {
            Ort::Status log_status(ep->factory_.GetOrtApi().Logger_LogMessage(
                &ep->logger_, ORT_LOGGING_LEVEL_WARNING,
                "GPU memory was allocated during CUDA graph capture. "
                "Graph replay may produce incorrect results. Consider increasing "
                "min_num_runs_before_cuda_graph_capture to allow allocations to stabilize.",
                ORT_FILE, __LINE__, __FUNCTION__));
          }
        }
        context.pre_capture_free_mem = 0;
      }

      // CUDA work issued to a capturing stream doesn't actually run on the GPU,
      // so replay the captured graph to actually execute the work.
      OrtStatus* status = context.cuda_graph.Replay(id, sync_stream);
      if (status != nullptr) {
        return status;
      }
      replayed_graph = true;
    } else {
      context.cuda_graph.IncrementRegularRunCount(id);
    }
  }

  if (sync_stream && !replayed_graph) {
    PL_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(context.graph_stream));
  }
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

/*static*/
bool ORT_API_CALL CudaEp::IsGraphCaptureEnabledImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
  return ep->config_.enable_cuda_graph;
}

/*static*/
bool ORT_API_CALL CudaEp::IsGraphCapturedImpl(const OrtEp* this_ptr, int graph_annotation_id) noexcept {
  const auto* ep = static_cast<const CudaEp*>(this_ptr);
  if (!ep->config_.enable_cuda_graph) {
    return false;
  }

  try {
    return ep->GetPerThreadContext().cuda_graph.IsGraphCaptured(graph_annotation_id);
  } catch (...) {
    return false;
  }
}

/*static*/
OrtStatus* ORT_API_CALL CudaEp::ReplayGraphImpl(OrtEp* this_ptr, int graph_annotation_id) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  auto* ep = static_cast<CudaEp*>(this_ptr);
  if (!ep->config_.enable_cuda_graph) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL, "ReplayGraph called but CUDA graph manager is not initialized");
  }
  PL_CUDA_CALL_THROW(cudaSetDevice(ep->config_.device_id));
  return ep->GetPerThreadContext().cuda_graph.Replay(graph_annotation_id);

  EXCEPTION_TO_STATUS_END
}

/*static*/
OrtGraphCaptureNodeAssignmentPolicy ORT_API_CALL CudaEp::GetGraphCaptureNodeAssignmentPolicyImpl(
    const OrtEp* /*this_ptr*/) noexcept {
  return OrtGraphCaptureNodeAssignmentPolicy_ALLOW_CPU_FOR_SHAPES;
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
