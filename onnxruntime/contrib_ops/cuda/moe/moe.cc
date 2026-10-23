// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/common/string_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "contrib_ops/cpu/moe/moe_cpu_offload.h"
#include "contrib_ops/cuda/moe/moe.h"
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
#include "contrib_ops/cuda/moe/kernel_pilot_moe_expert_selection_cuda.h"
#include "core/framework/kernel_pilot.h"
#include "core/providers/cuda/cuda_external_data_loader.h"
#endif
#include "contrib_ops/cuda/moe/moe_kernels.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <mutex>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
class CudaDeviceGuard {
 public:
  Status SetDevice(int device_id) {
    CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous_device_));
    if (previous_device_ != device_id) {
      CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
      restore_ = true;
    }
    return Status::OK();
  }

  ~CudaDeviceGuard() {
    if (restore_) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device_)));
    }
  }

 private:
  int previous_device_{0};
  bool restore_{false};
};

void LogSwigluFusionRemapOnce() {
  static std::once_flag log_warning;
  std::call_once(log_warning, []() {
    LOGS_DEFAULT(WARNING) << "MoE swiglu_fusion is 0 with no fc3_experts_weights; assuming interleaved "
                             "SwiGLU layout for backward compatibility.";
  });
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
constexpr size_t kMaxAsyncCpuRows = 256;

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    CUDA_CALL_THROW(cudaGetDevice(&previous_device_id_));
    if (previous_device_id_ != device_id) {
      CUDA_CALL_THROW(cudaSetDevice(device_id));
      restore_device_ = true;
    }
  }

  ~ScopedCudaDevice() {
    if (restore_device_) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device_id_)));
    }
  }

 private:
  int previous_device_id_{-1};
  bool restore_device_{false};
};
#endif
}  // namespace

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
template <typename T>
struct MoE<T>::CpuTask {
  static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));
  IAllocatorUniquePtr<void> device_buffer;
  IAllocatorUniquePtr<T> host_input;
  IAllocatorUniquePtr<T> host_output;
  IAllocatorUniquePtr<int> host_ids;
  IAllocatorUniquePtr<float> host_scales;
  IAllocatorUniquePtr<uint32_t> host_count;
  IAllocatorUniquePtr<void> completion_storage;
  std::atomic<uint32_t>* completion{nullptr};
  CUdeviceptr device_completion{0};
  uint32_t* device_count{nullptr};
  int* device_rows{nullptr};
  int* device_ids{nullptr};
  float* device_scales{nullptr};
  T* device_input{nullptr};
  T* device_output{nullptr};
  cudaEvent_t routes_ready{nullptr};
  cudaEvent_t input_ready{nullptr};
  size_t capacity{0};
  size_t num_rows{0};
  size_t hidden_size{0};
  size_t inter_size{0};
  size_t num_experts{0};
  size_t k{0};
  ::onnxruntime::contrib::MoeCpuOffloadParameters parameters{};
  concurrency::ThreadPool* thread_pool{nullptr};
  Status status;

  ~CpuTask() {
    for (cudaEvent_t event : {routes_ready, input_ready}) {
      if (event != nullptr) {
        ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(event)));
      }
    }
    if (completion != nullptr) {
      completion->~atomic();
    }
  }
};
#endif

#define REGISTER_KERNEL_TYPED(T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                    \
      MoE, kMSDomain, 1, T, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create()).MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), MoE<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
MoE<T>::MoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  if constexpr (kCpuOffloadSupported) {
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    const auto cpu_offload_experts = op_kernel_info.GetConfigOptions().GetConfigOrDefault(
        kOrtSessionOptionsConfigMoeCpuOffloadExperts, "0");
    int64_t cpu_offload_expert_count = -1;
    ORT_ENFORCE(TryParseStringWithClassicLocale(cpu_offload_experts, cpu_offload_expert_count) &&
                    cpu_offload_expert_count >= 0,
                kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                " must be a non-negative integer. Received: \"", cpu_offload_experts, "\".");
    cpu_offload_enabled_ = cpu_offload_expert_count > 0;
    int64_t on_demand_threshold = 0;
    ORT_ENFORCE(TryParseStringWithClassicLocale(
                    op_kernel_info.GetConfigOptions().GetConfigOrDefault(
                        kOrtSessionOptionsConfigMoeCpuOffloadOnDemandThreshold, "0"),
                    on_demand_threshold) &&
                    on_demand_threshold == 0,
                kOrtSessionOptionsConfigMoeCpuOffloadOnDemandThreshold, " supports CUDA QMoE only.");
    if (cpu_offload_enabled_) {
      kernel_pilot_initializer_.Register(
          this, [this](KernelPilot* pilot) { return InitializeKernelPilot(pilot); });
      device_id_ = GetDeviceId();
      const ScopedCudaDevice device_guard(device_id_);
      const Status resource_status = [&]() -> Status {
        CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&input_copy_stream_, cudaStreamNonBlocking));
        CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&swap_d2h_stream_, cudaStreamNonBlocking));
        CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&swap_h2d_stream_, cudaStreamNonBlocking));
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&last_expert_use_event_, cudaEventDisableTiming));
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&swap_cpu_ready_event_, cudaEventDisableTiming));
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&swap_transfer_complete_event_, cudaEventDisableTiming));
        CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&usage_complete_event_, cudaEventDisableTiming));
        // CUDA 11.7+ selects v2 memory operations, which do not use the deprecated v1 capability flag.
        int supported = 1;
#if CUDA_VERSION < 11070
        CUdevice device;
        ORT_RETURN_IF_ERROR(CU_CALL(cuDeviceGet(&device, device_id_)));
        ORT_RETURN_IF_ERROR(CU_CALL(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, device)));
#endif
        async_cpu_supported_ = supported != 0;
        if (!async_cpu_supported_) {
          LOGS_DEFAULT(WARNING) << "CUDA stream memory operations are unavailable; MoE CPU offload uses host waits.";
        }
        return CUDA_CALL(cudaEventCreateWithFlags(&swap_publication_complete_event_, cudaEventDisableTiming));
      }();
      if (!resource_status.IsOK()) {
        ReleaseSwapResources();
        ORT_THROW_IF_ERROR(resource_status);
      }
    }
#endif
    cuda_allocator_ = op_kernel_info.GetAllocator(OrtMemTypeDefault);
  }
}

template <typename T>
MoE<T>::~MoE() {
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  ReleaseUsageResources();
  ReleaseSwapResources();
#endif
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
template <typename T>
void MoE<T>::ReleaseSwapResources() noexcept {
  CudaDeviceGuard device_guard;
  if (cpu_offload_enabled_) {
    ORT_IGNORE_RETURN_VALUE(device_guard.SetDevice(device_id_));
  }
  if (swap_d2h_stream_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(swap_d2h_stream_)));
  }
  if (swap_h2d_stream_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(swap_h2d_stream_)));
  }
  if (swap_publication_complete_event_ != nullptr && swap_phase_ == SwapPhase::PublicationInFlight) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventSynchronize(swap_publication_complete_event_)));
  }
  swap_pinned_buffer_.reset();
  swap_outgoing_pinned_buffer_.reset();
  swap_cuda_staging_.reset();
  swap_cuda_layout_buffer_.reset();
  for (cudaEvent_t event : {last_expert_use_event_, swap_cpu_ready_event_,
                            swap_transfer_complete_event_, swap_publication_complete_event_, usage_complete_event_}) {
    if (event != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(event)));
    }
  }
  if (swap_h2d_stream_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(swap_h2d_stream_)));
  }
  if (swap_d2h_stream_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(swap_d2h_stream_)));
  }
  if (input_copy_stream_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(input_copy_stream_)));
  }
  swap_publication_complete_event_ = nullptr;
  usage_complete_event_ = nullptr;
  swap_transfer_complete_event_ = nullptr;
  swap_cpu_ready_event_ = nullptr;
  last_expert_use_event_ = nullptr;
  swap_h2d_stream_ = nullptr;
  swap_d2h_stream_ = nullptr;
  input_copy_stream_ = nullptr;
}
#endif

template <typename T>
Status MoE<T>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr,
                       bool& is_packed, PrePackedWeights* prepacked_weights) {
  is_packed = false;
  ORT_UNUSED_PARAMETER(prepacked_weights);
  if constexpr (!kCpuOffloadSupported) {
    return Status::OK();
  }

  if (!cpu_offload_enabled_ || input_idx < 2 || input_idx > 7) {
    return Status::OK();
  }

  auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
  ORT_RETURN_IF(packed.present, "MoE input ", input_idx, " was prepacked more than once.");
  packed.shape = tensor.Shape();
  packed.bytes = tensor.SizeInBytes();
  packed.present = true;

  ORT_RETURN_IF_NOT(packed.bytes % sizeof(T) == 0,
                    "FP16/BF16 MoE input ", input_idx, " has an invalid byte size.");
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  const OrtValue* initializer = nullptr;
  ORT_RETURN_IF_NOT(tensor.Location().device.Type() == OrtDevice::CPU &&
                        Info().TryGetConstantInput(input_idx, &initializer) && initializer != nullptr,
                    "MoE CPU offload requires a CPU constant initializer for input ", input_idx, ".");
  // Retain the mmap/initializer owner, not a complete CPU weight copy.
  packed.initializer = *initializer;
#endif

  is_packed = true;
  return Status::OK();
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
template <typename T>
Status MoE<T>::InitializeKernelPilot(KernelPilot* pilot) {
  if constexpr (!kCpuOffloadSupported) {
    return Status::OK();
  }
  if (!cpu_offload_enabled_) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(pilot, "FP16/BF16 MoE CPU offload requires a KernelPilot.");
  gsl::span<const int> cuda_experts;
  ORT_RETURN_IF_ERROR(pilot->GetMoeCudaExperts(cuda_experts));
  ORT_RETURN_IF_ERROR(InitializeCudaExpertWeights(cuda_experts));
  ORT_RETURN_IF_ERROR(pilot->AttachMoeExpertUsage(this));
  return pilot->AttachMoeExpertCache(this);
}

template <typename T>
Status MoE<T>::PrepareUsage(OpKernelContext* context, KernelPilot* pilot, size_t expert_count,
                            size_t num_rows, size_t hidden_size, size_t inter_size, bool fused_swiglu,
                            bool prepare_cpu_task) const {
  ORT_RETURN_IF_ERROR(pilot->CompleteMoeUsage());
  ORT_RETURN_IF_NOT(pilot->MoeExpertCounters().size() == expert_count, "MoE counters were not initialized.");
  const size_t usage_bytes = SafeInt<size_t>(expert_count) * (sizeof(double) + sizeof(float) + sizeof(int64_t));
  if (!host_usage_) {
    auto host_usage = AllocateBufferOnCPUPinned<void>(usage_bytes);
    auto device_usage = IAllocator::MakeUniquePtr<void>(cuda_allocator_, usage_bytes, true);
    ORT_RETURN_IF_NOT(host_usage && device_usage, "Failed to allocate MoE GPU counter buffers.");
    host_usage_ = std::move(host_usage);
    device_usage_ = std::move(device_usage);
  }
  std::memcpy(host_usage_.get(), pilot->MoeExpertCounters().data(), expert_count * sizeof(double));
  usage_stream_ = Stream(context);
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device_usage_.get(), host_usage_.get(), expert_count * sizeof(double),
                                       cudaMemcpyHostToDevice, usage_stream_));
  usage_pending_ = true;
  usage_recorded_ = false;
  cpu_callback_scheduled_ = false;

  if (!prepare_cpu_task) {
    return Status::OK();
  }
  if (!cpu_task_) {
    auto cpu_task = std::make_unique<CpuTask>();
    auto& task = *cpu_task;
    task.host_count = AllocateBufferOnCPUPinned<uint32_t>(1);
    task.completion_storage = AllocateBufferOnCPUPinned<void>(sizeof(std::atomic<uint32_t>));
    ORT_RETURN_IF_NOT(task.host_count && task.completion_storage, "Failed to allocate MoE CPU notification buffers.");
    task.completion = new (task.completion_storage.get()) std::atomic<uint32_t>(0);
    void* device_completion = nullptr;
    CUDA_RETURN_IF_ERROR(cudaHostGetDevicePointer(&device_completion, task.completion_storage.get(), 0));
    task.device_completion = reinterpret_cast<CUdeviceptr>(device_completion);
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&task.routes_ready, cudaEventDisableTiming));
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&task.input_ready, cudaEventDisableTiming));
    cpu_task_ = std::move(cpu_task);
  }
  if (!cpu_worker_.joinable()) {
    cpu_worker_ = std::thread([this]() { RunCpuWorker(); });
  }
  auto& task = *cpu_task_;
  if (num_rows > task.capacity) {
    const size_t expanded_rows = SafeInt<size_t>(num_rows) * k_;
    const size_t elements = SafeInt<size_t>(num_rows) * hidden_size;
    const size_t bytes = SafeInt<size_t>(sizeof(uint32_t)) + SafeInt<size_t>(num_rows) * sizeof(int) +
                         SafeInt<size_t>(expanded_rows) * (sizeof(int) + sizeof(float)) +
                         SafeInt<size_t>(elements) * sizeof(T) * 2;
    auto device_buffer = IAllocator::MakeUniquePtr<void>(cuda_allocator_, bytes, true);
    auto host_input = AllocateBufferOnCPUPinned<T>(elements);
    auto host_output = AllocateBufferOnCPUPinned<T>(elements);
    auto host_ids = AllocateBufferOnCPUPinned<int>(expanded_rows);
    auto host_scales = AllocateBufferOnCPUPinned<float>(expanded_rows);
    ORT_RETURN_IF_NOT(device_buffer && host_input && host_output && host_ids && host_scales,
                      "Failed to allocate compact MoE CPU routing buffers.");
    task.device_buffer = std::move(device_buffer);
    task.host_input = std::move(host_input);
    task.host_output = std::move(host_output);
    task.host_ids = std::move(host_ids);
    task.host_scales = std::move(host_scales);
    auto* buffer = static_cast<char*>(task.device_buffer.get());
    task.device_count = reinterpret_cast<uint32_t*>(buffer);
    task.device_rows = reinterpret_cast<int*>(buffer + sizeof(uint32_t));
    task.device_ids = task.device_rows + num_rows;
    task.device_scales = reinterpret_cast<float*>(task.device_ids + expanded_rows);
    task.device_input = reinterpret_cast<T*>(task.device_scales + expanded_rows);
    task.device_output = task.device_input + elements;
    task.capacity = num_rows;
  }
  task.num_rows = num_rows;
  task.hidden_size = hidden_size;
  task.inter_size = inter_size;
  task.num_experts = expert_count;
  task.k = static_cast<size_t>(k_);
  task.thread_pool = context->GetOperatorThreadPool();
  task.completion->store(0, std::memory_order_release);
  task.status = Status::OK();
  using CudaActivation = onnxruntime::llm::kernels::cutlass_kernels::ActivationType;
  using CpuActivation = ::onnxruntime::contrib::ActivationType;
  CpuActivation activation = CpuActivation::Identity;
  switch (activation_type_) {
    case CudaActivation::Relu:
      activation = CpuActivation::Relu;
      break;
    case CudaActivation::Gelu:
      activation = CpuActivation::Gelu;
      break;
    case CudaActivation::Silu:
      activation = CpuActivation::Silu;
      break;
    case CudaActivation::Swiglu:
      activation = CpuActivation::SwiGLU;
      break;
    case CudaActivation::Identity:
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unsupported MoE CPU activation.");
  }
  task.parameters = {activation, activation_alpha_, activation_beta_, swiglu_limit_, fused_swiglu};
  return Status::OK();
}

template <typename T>
void CUDART_CB MoE<T>::NotifyCpuWorker(void* context) {
  auto& owner = *static_cast<const MoE<T>*>(context);
  {
    std::lock_guard<std::mutex> lock(owner.cpu_worker_mutex_);
    if (*owner.cpu_task_->host_count == 0) {
      owner.cpu_task_->completion->store(1, std::memory_order_release);
      owner.cpu_task_finished_ = true;
    } else {
      owner.cpu_task_ready_ = true;
    }
  }
  owner.cpu_worker_condition_.notify_all();
}

template <typename T>
Status MoE<T>::ComputeCpuTask() const {
  auto& task = *cpu_task_;
  const size_t rows = *task.host_count;
  ORT_RETURN_IF(rows > task.num_rows, "Invalid compact MoE CPU row count.");
  if (rows == 0) {
    task.completion->store(1, std::memory_order_release);
    return Status::OK();
  }
  const size_t elements = rows * task.hidden_size;
  const size_t routes = rows * task.k;
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(task.host_input.get(), task.device_input, elements * sizeof(T),
                                       cudaMemcpyDeviceToHost, input_copy_stream_));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(task.host_ids.get(), task.device_ids, routes * sizeof(int),
                                       cudaMemcpyDeviceToHost, input_copy_stream_));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(task.host_scales.get(), task.device_scales, routes * sizeof(float),
                                       cudaMemcpyDeviceToHost, input_copy_stream_));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(task.input_ready, input_copy_stream_));
  CUDA_RETURN_IF_ERROR(cudaEventSynchronize(task.input_ready));
  const auto bias = [&](int index) {
    return gsl::make_span(packed_inputs_[static_cast<size_t>(index)].cpu_data);
  };
  if constexpr (std::is_same_v<T, MLFloat16>) {
    const auto weights = [&](int index) {
      return gsl::make_span(packed_inputs_[static_cast<size_t>(index)].cpu_gemm_data);
    };
    ORT_RETURN_IF_ERROR(::onnxruntime::contrib::ComputeMoeCpuOffloadedExpertsFp16(
        gsl::make_span(task.host_input.get(), elements), gsl::make_span(task.host_ids.get(), routes),
        gsl::make_span(task.host_scales.get(), routes), cpu_expert_map_,
        weights(2), bias(3), weights(4), bias(5), rows, task.hidden_size, task.inter_size,
        task.num_experts, task.k, task.parameters, gsl::make_span(task.host_output.get(), elements), task.thread_pool));
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    const auto weights = [&](int index) {
      return gsl::make_span(packed_inputs_[static_cast<size_t>(index)].cpu_gemm_float_data);
    };
    ORT_RETURN_IF_ERROR(::onnxruntime::contrib::ComputeMoeCpuOffloadedExpertsBFloat16(
        gsl::make_span(task.host_input.get(), elements), gsl::make_span(task.host_ids.get(), routes),
        gsl::make_span(task.host_scales.get(), routes), cpu_expert_map_,
        weights(2), bias(3), weights(4), bias(5), rows, task.hidden_size, task.inter_size,
        task.num_experts, task.k, task.parameters, gsl::make_span(task.host_output.get(), elements), task.thread_pool));
  }
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(task.device_output, task.host_output.get(), elements * sizeof(T),
                                       cudaMemcpyHostToDevice, input_copy_stream_));
  ORT_RETURN_IF_ERROR(CU_CALL(cuStreamWriteValue32(reinterpret_cast<CUstream>(input_copy_stream_),
                                                   task.device_completion, 1, 0)));
  return CUDA_CALL(cudaEventRecord(task.input_ready, input_copy_stream_));
}

template <typename T>
void MoE<T>::RunCpuWorker() const {
  const Status device_status = CUDA_CALL(cudaSetDevice(device_id_));
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(cpu_worker_mutex_);
      cpu_worker_condition_.wait(lock, [&]() { return stop_cpu_worker_ || cpu_task_ready_; });
      if (!cpu_task_ready_) {
        return;
      }
      cpu_task_ready_ = false;
    }
    Status status = device_status;
    ORT_TRY {
      if (status.IsOK()) {
        status = ComputeCpuTask();
      }
    }
    ORT_CATCH(const std::exception& error) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Asynchronous MoE CPU execution: ", error.what());
      });
    }
    ORT_CATCH(...) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION, "Unknown asynchronous MoE CPU execution exception.");
    }
    if (!status.IsOK()) {
      // Release the device wait on failure; EndRun propagates the error, never a successful zero result.
      cpu_task_->completion->store(2, std::memory_order_release);
    }
    {
      std::lock_guard<std::mutex> lock(cpu_worker_mutex_);
      cpu_task_->status = std::move(status);
      cpu_task_finished_ = true;
    }
    cpu_worker_condition_.notify_all();
  }
}

template <typename T>
Status MoE<T>::CompleteUsage(KernelPilotMoeExpertUsageSnapshot& snapshot) {
  auto finish = gsl::finally([&]() { usage_pending_ = false; });
  CudaDeviceGuard device_guard;
  ORT_RETURN_IF_ERROR(device_guard.SetDevice(device_id_));
  if (cpu_callback_scheduled_) {
    std::unique_lock<std::mutex> lock(cpu_worker_mutex_);
    cpu_worker_condition_.wait(lock, [&]() { return cpu_task_finished_; });
  }
  if (usage_recorded_) {
    const cudaError_t ready = cudaEventQuery(usage_complete_event_);
    if (ready == cudaErrorNotReady) {
      CUDA_RETURN_IF_ERROR(cudaEventSynchronize(usage_complete_event_));
    } else {
      CUDA_RETURN_IF_ERROR(ready);
    }
  } else {
    if (cpu_callback_scheduled_) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(input_copy_stream_));
    }
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(usage_stream_));
  }
  if (cpu_callback_scheduled_) {
    ORT_RETURN_IF_ERROR(cpu_task_->status);
  }
  const size_t experts = expert_map_.size();
  snapshot.counters = gsl::make_span(static_cast<const double*>(host_usage_.get()), experts);
  snapshot.selected_probabilities = gsl::make_span(
      reinterpret_cast<const float*>(static_cast<const char*>(host_usage_.get()) +
                                     experts * (sizeof(double) + sizeof(int64_t))),
      experts);
  snapshot.first_selected_routes = gsl::make_span(
      reinterpret_cast<const int64_t*>(static_cast<const double*>(host_usage_.get()) + experts), experts);
  return Status::OK();
}

template <typename T>
void MoE<T>::ReleaseUsageResources() noexcept {
  CudaDeviceGuard device_guard;
  if (cpu_offload_enabled_) {
    ORT_IGNORE_RETURN_VALUE(device_guard.SetDevice(device_id_));
  }
  if (usage_pending_) {
    KernelPilotMoeExpertUsageSnapshot snapshot;
    ORT_IGNORE_RETURN_VALUE(CompleteUsage(snapshot));
  }
  {
    std::lock_guard<std::mutex> lock(cpu_worker_mutex_);
    stop_cpu_worker_ = true;
  }
  cpu_worker_condition_.notify_all();
  if (cpu_worker_.joinable()) {
    cpu_worker_.join();
  }
  if (input_copy_stream_ != nullptr) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(input_copy_stream_)));
  }
  cpu_task_.reset();
}

template <typename T>
Status MoE<T>::InitializeCudaExpertWeights(gsl::span<const int> cuda_experts) {
  if constexpr (!kCpuOffloadSupported) {
    ORT_UNUSED_PARAMETER(cuda_experts);
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(packed_inputs_[2].present && packed_inputs_[4].present,
                    "FP16/BF16 MoE CPU offload requires constant FC1 and FC2 weights.");
  for (int input_idx : {3, 5, 6, 7}) {
    const auto& input_defs = Info().node().InputDefs();
    const bool connected = static_cast<size_t>(input_idx) < input_defs.size() &&
                           input_defs[static_cast<size_t>(input_idx)]->Exists();
    ORT_RETURN_IF(connected && !packed_inputs_[static_cast<size_t>(input_idx)].present,
                  "FP16/BF16 MoE CPU offload requires optional expert input ", input_idx,
                  " to be constant or absent.");
  }
  ORT_RETURN_IF(packed_inputs_[6].present || packed_inputs_[7].present,
                "FP16/BF16 MoE CPU offload does not yet support separate FC3 weights or bias.");
  ORT_RETURN_IF(activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
                    swiglu_fusion_ == 2,
                "FP16/BF16 MoE CPU offload does not support chunked SwiGLU.");

  const auto& fc1_shape = packed_inputs_[2].shape;
  const auto& fc2_shape = packed_inputs_[4].shape;
  ORT_RETURN_IF_NOT(fc1_shape.NumDimensions() == 3 && fc2_shape.NumDimensions() == 3 &&
                        fc1_shape[0] > 0 && fc2_shape[0] == fc1_shape[0],
                    "FP16/BF16 MoE FC1 and FC2 weights must be rank 3 with matching positive expert dimensions.");
  const size_t num_experts = static_cast<size_t>(fc1_shape[0]);
  const auto& input_defs = Info().node().InputDefs();
  const auto* input_shape = input_defs[0]->Shape();
  ORT_RETURN_IF_NOT(input_shape != nullptr && input_shape->dim_size() > 0 &&
                        input_shape->dim(input_shape->dim_size() - 1).has_dim_value(),
                    "FP16/BF16 MoE CPU offload requires a static input hidden dimension.");
  const int64_t hidden_size = input_shape->dim(input_shape->dim_size() - 1).dim_value();
  const int64_t fc2_elements_per_expert = fc2_shape.SizeFromDimension(1);
  ORT_RETURN_IF_NOT(hidden_size > 0 && fc2_elements_per_expert % hidden_size == 0,
                    "FP16/BF16 MoE FC2 weights have an invalid shape for hidden size ", hidden_size, ".");
  const int64_t inter_size = fc2_elements_per_expert / hidden_size;
  const bool is_fused_swiglu =
      activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu &&
      swiglu_fusion_ != 2 && !packed_inputs_[6].present;
  const int64_t fc1_output_size = is_fused_swiglu ? 2 * inter_size : inter_size;
  const bool legacy_shape =
      (hidden_size != inter_size && fc2_shape[1] == inter_size) ||
      (hidden_size == inter_size && is_fused_swiglu && fc1_shape[1] == hidden_size);
  const std::array<int64_t, 8> logical_output_sizes{
      0, 0, fc1_output_size, fc1_output_size, hidden_size, hidden_size, 0, 0};
  for (const auto& [bias_idx, weight_idx] : {std::pair{3, 2}, std::pair{5, 4}}) {
    const auto& bias = packed_inputs_[static_cast<size_t>(bias_idx)];
    const auto& weight = packed_inputs_[static_cast<size_t>(weight_idx)];
    ORT_RETURN_IF(bias.present &&
                      (weight.shape.NumDimensions() != 3 ||
                       bias.shape.NumDimensions() != 2 ||
                       bias.shape[0] != static_cast<int64_t>(num_experts) ||
                       bias.shape[1] != logical_output_sizes[static_cast<size_t>(bias_idx)]),
                  "FP16/BF16 MoE input ", bias_idx, " has an invalid bias shape.");
  }
  cuda_experts_.assign(cuda_experts.begin(), cuda_experts.end());
  expert_map_.assign(num_experts, -1);
  cpu_experts_.clear();
  cpu_expert_map_.assign(num_experts, -1);
  for (size_t index = 0; index < cuda_experts_.size(); ++index) {
    const int expert = cuda_experts_[index];
    ORT_RETURN_IF(expert < 0 || static_cast<size_t>(expert) >= num_experts,
                  "KernelPilot selected an invalid CUDA MoE expert: ", expert);
    expert_map_[static_cast<size_t>(expert)] = static_cast<int>(index);
  }
  for (size_t expert = 0; expert < num_experts; ++expert) {
    if (expert_map_[expert] < 0) {
      cpu_expert_map_[expert] = static_cast<int>(cpu_experts_.size());
      cpu_experts_.push_back(static_cast<int>(expert));
    }
  }

  const auto* provider = static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider());
  auto loader = provider->GetExternalDataLoader();
  for (int input_idx : {2, 3, 4, 5}) {
    auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
    if (!packed.present) {
      continue;
    }
    ORT_RETURN_IF_NOT(packed.shape.NumDimensions() >= 1 &&
                          packed.shape[0] == static_cast<int64_t>(num_experts) &&
                          packed.bytes % num_experts == 0,
                      "FP16/BF16 MoE input ", input_idx, " has an invalid expert-major layout.");
    const size_t expert_bytes = packed.bytes / num_experts;
    packed.expert_bytes = expert_bytes;
    packed.swap_offset = swap_staging_bytes_;
    swap_staging_bytes_ = SafeInt<size_t>(swap_staging_bytes_) + expert_bytes;
    const size_t cuda_bytes = SafeInt<size_t>(cuda_experts_.size()) * expert_bytes;
    const auto& source = packed.initializer.template Get<Tensor>();
    if (cuda_bytes > 0) {
      packed.cuda_data = IAllocator::MakeUniquePtr<void>(cuda_allocator_, cuda_bytes, true);
      ORT_RETURN_IF_NOT(packed.cuda_data, "Failed to allocate CUDA storage for MoE input ", input_idx, ".");
      if (loader) {
        TensorShape compact_shape = packed.shape;
        compact_shape[0] = static_cast<int64_t>(cuda_experts_.size());
        auto compact = Tensor::Create(source.DataType(), compact_shape, packed.cuda_data.get(), cuda_allocator_->Info());
        ORT_RETURN_IF_ERROR(static_cast<const onnxruntime::cuda::ExternalDataLoader*>(loader.get())
                                ->LoadTensorFromHost(source, cuda_experts_, *compact));
      } else {
        for (size_t index = 0; index < cuda_experts_.size(); ++index) {
          CUDA_RETURN_IF_ERROR(cudaMemcpy(
              static_cast<char*>(packed.cuda_data.get()) + index * expert_bytes,
              static_cast<const char*>(source.DataRaw()) +
                  static_cast<size_t>(cuda_experts_[index]) * expert_bytes,
              expert_bytes, cudaMemcpyHostToDevice));
        }
      }
    }
    const size_t expert_element_count = packed.expert_bytes / sizeof(T);
    const T* source_data = source.template Data<T>();
    if (input_idx == 2 || input_idx == 4) {
      const size_t output_size = static_cast<size_t>(
          logical_output_sizes[static_cast<size_t>(input_idx)]);
      const size_t input_size = static_cast<size_t>(input_idx == 2 ? hidden_size : inter_size);
      ORT_RETURN_IF_NOT(
          packed.shape.SizeFromDimension(1) == static_cast<int64_t>(expert_element_count) &&
              (legacy_shape ? packed.shape[1] == static_cast<int64_t>(input_size)
                            : packed.shape[1] == static_cast<int64_t>(output_size)),
          "FP16/BF16 MoE input ", input_idx, " has an invalid ",
          legacy_shape ? "legacy" : "standard", " expert weight shape.");
      packed.gemm_input_size = input_size;
      packed.gemm_output_size = output_size;
      if constexpr (std::is_same_v<T, MLFloat16>) {
        packed.cpu_gemm_data.resize(cpu_experts_.size() * expert_element_count);
        for (size_t cpu_slot = 0; cpu_slot < cpu_experts_.size(); ++cpu_slot) {
          const size_t expert = static_cast<size_t>(cpu_experts_[cpu_slot]);
          const T* source = source_data + expert * expert_element_count;
          T* destination = packed.cpu_gemm_data.data() + cpu_slot * expert_element_count;
          for (size_t output = 0; output < output_size; ++output) {
            for (size_t input = 0; input < input_size; ++input) {
              destination[input * output_size + output] = source[output * input_size + input];
            }
          }
        }
      } else if constexpr (std::is_same_v<T, BFloat16>) {
        packed.cpu_gemm_float_data.resize(cpu_experts_.size() * expert_element_count);
        for (size_t cpu_slot = 0; cpu_slot < cpu_experts_.size(); ++cpu_slot) {
          const size_t expert = static_cast<size_t>(cpu_experts_[cpu_slot]);
          const T* source = source_data + expert * expert_element_count;
          float* destination = packed.cpu_gemm_float_data.data() + cpu_slot * expert_element_count;
          for (size_t output = 0; output < output_size; ++output) {
            for (size_t input = 0; input < input_size; ++input) {
              destination[input * output_size + output] = source[output * input_size + input].ToFloat();
            }
          }
        }
      }
    } else {
      std::vector<T> compact_cpu_data(cpu_experts_.size() * expert_element_count);
      for (size_t cpu_slot = 0; cpu_slot < cpu_experts_.size(); ++cpu_slot) {
        const size_t expert = static_cast<size_t>(cpu_experts_[cpu_slot]);
        std::copy_n(source_data + expert * expert_element_count,
                    expert_element_count,
                    compact_cpu_data.data() + cpu_slot * expert_element_count);
      }
      packed.cpu_data.swap(compact_cpu_data);
    }
    packed.initializer = OrtValue{};
  }

  if (!cuda_experts_.empty()) {
    device_expert_map_ =
        IAllocator::MakeUniquePtr<void>(cuda_allocator_, expert_map_.size() * sizeof(int), true);
    ORT_RETURN_IF_NOT(device_expert_map_, "Failed to allocate the CUDA MoE expert map.");
    CUDA_RETURN_IF_ERROR(cudaMemcpy(device_expert_map_.get(), expert_map_.data(),
                                    expert_map_.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  return Status::OK();
}

template <typename T>
bool MoE<T>::HasPendingSwap() const noexcept {
  return swap_phase_ != SwapPhase::Idle;
}

template <typename T>
Status MoE<T>::ReclaimCompletedSwap() {
  if (swap_phase_ != SwapPhase::PublicationInFlight) {
    return Status::OK();
  }
  CudaDeviceGuard device_guard;
  ORT_RETURN_IF_ERROR(device_guard.SetDevice(device_id_));
  const cudaError_t result = cudaEventQuery(swap_publication_complete_event_);
  if (result == cudaErrorNotReady) {
    return Status::OK();
  }
  CUDA_RETURN_IF_ERROR(result);
  swap_cuda_expert_ = -1;
  swap_cpu_expert_ = -1;
  swap_pinned_buffer_.reset();
  swap_outgoing_pinned_buffer_.reset();
  swap_cuda_staging_.reset();
  swap_cuda_layout_buffer_.reset();
  swap_phase_ = SwapPhase::Idle;
  return Status::OK();
}

template <typename T>
void MoE<T>::StageIncomingExpert() noexcept {
  auto* staging = static_cast<char*>(swap_pinned_buffer_.get());
  for (int input_idx : {2, 3, 4, 5}) {
    auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
    if (!packed.present) {
      continue;
    }
    const size_t expert_elements = packed.expert_bytes / sizeof(T);
    auto* incoming = reinterpret_cast<T*>(staging + packed.swap_offset);
    if (!packed.cpu_gemm_data.empty()) {
      std::memcpy(incoming, packed.cpu_gemm_data.data() + swap_cpu_slot_ * expert_elements,
                  packed.expert_bytes);
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      if (!packed.cpu_gemm_float_data.empty()) {
        const auto* source = packed.cpu_gemm_float_data.data() + swap_cpu_slot_ * expert_elements;
        for (size_t i = 0; i < expert_elements; ++i) {
          incoming[i] = T(source[i]);
        }
      } else {
        std::memcpy(incoming, packed.cpu_data.data() + swap_cpu_slot_ * expert_elements, packed.expert_bytes);
      }
    } else {
      std::memcpy(incoming, packed.cpu_data.data() + swap_cpu_slot_ * expert_elements, packed.expert_bytes);
    }
  }
}

template <typename T>
Status MoE<T>::StartSwap(int cuda_expert_id, int cpu_expert_id) {
  ORT_RETURN_IF(swap_phase_ != SwapPhase::Idle, "An MoE expert swap is already pending.");
  CudaDeviceGuard device_guard;
  ORT_RETURN_IF_ERROR(device_guard.SetDevice(device_id_));
  const auto cuda_expert =
      std::find(cuda_experts_.begin(), cuda_experts_.end(), cuda_expert_id);
  ORT_RETURN_IF(cuda_expert == cuda_experts_.end(),
                "MoE swap tried to evict a non-resident CUDA expert: ", cuda_expert_id);
  ORT_RETURN_IF(cpu_expert_id < 0 ||
                    static_cast<size_t>(cpu_expert_id) >= expert_map_.size() ||
                    cpu_expert_map_[static_cast<size_t>(cpu_expert_id)] < 0,
                "MoE swap tried to stage an invalid CPU expert: ", cpu_expert_id);
  ORT_RETURN_IF_NOT(swap_staging_bytes_ > 0, "MoE swap has no expert data to transfer.");

  if (!swap_pinned_buffer_) {
    swap_pinned_buffer_ = AllocateBufferOnCPUPinned<void>(swap_staging_bytes_);
  }
  if (!swap_cuda_staging_) {
    swap_cuda_staging_ =
        IAllocator::MakeUniquePtr<void>(cuda_allocator_, swap_staging_bytes_, true);
  }
  if (!swap_outgoing_pinned_buffer_) {
    swap_outgoing_pinned_buffer_ = AllocateBufferOnCPUPinned<void>(swap_staging_bytes_);
  }
  if (!swap_cuda_layout_buffer_) {
    swap_cuda_layout_buffer_ =
        IAllocator::MakeUniquePtr<void>(cuda_allocator_, swap_staging_bytes_, true);
  }
  ORT_RETURN_IF_NOT(swap_pinned_buffer_ && swap_outgoing_pinned_buffer_ &&
                        swap_cuda_staging_ && swap_cuda_layout_buffer_,
                    "Failed to allocate MoE expert swap staging buffers.");
  swap_cuda_expert_ = cuda_expert_id;
  swap_cpu_expert_ = cpu_expert_id;
  swap_cuda_slot_ = static_cast<size_t>(std::distance(cuda_experts_.begin(), cuda_expert));
  swap_cpu_slot_ = static_cast<size_t>(cpu_expert_map_[static_cast<size_t>(cpu_expert_id)]);
  swap_phase_ = SwapPhase::TransferInFlight;

  const auto schedule = [&]() -> Status {
    // CUDA transposes both directions; no host callback blocks the copy streams
    // while converting a complete expert between CUDA and MLAS layouts.
    StageIncomingExpert();
    CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(swap_d2h_stream_, last_expert_use_event_, 0));
    for (int input_idx : {2, 3, 4, 5}) {
      const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
      if (!packed.present) {
        continue;
      }
      const void* outgoing =
          static_cast<const char*>(packed.cuda_data.get()) + swap_cuda_slot_ * packed.expert_bytes;
      if (packed.gemm_input_size != 0) {
        using CudaT = typename OrtToCudaType<T>::type;
        auto* layout = static_cast<char*>(swap_cuda_layout_buffer_.get()) + packed.swap_offset;
        if constexpr (kCpuOffloadSupported) {
          LaunchTransposeMoeExpertWeights(static_cast<const CudaT*>(outgoing),
                                          reinterpret_cast<CudaT*>(layout),
                                          packed.gemm_output_size, packed.gemm_input_size, swap_d2h_stream_);
        }
        outgoing = layout;
      }
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          static_cast<char*>(swap_outgoing_pinned_buffer_.get()) + packed.swap_offset, outgoing,
          packed.expert_bytes, cudaMemcpyDeviceToHost, swap_d2h_stream_));
    }
    CUDA_RETURN_IF_ERROR(cudaEventRecord(swap_cpu_ready_event_, swap_d2h_stream_));
    CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(swap_h2d_stream_, swap_cpu_ready_event_, 0));
    for (int input_idx : {2, 3, 4, 5}) {
      const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
      if (!packed.present) {
        continue;
      }
      auto* destination = static_cast<char*>(swap_cuda_staging_.get()) + packed.swap_offset;
      auto* layout = static_cast<char*>(swap_cuda_layout_buffer_.get()) + packed.swap_offset;
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
          packed.gemm_input_size != 0 ? layout : destination,
          static_cast<const char*>(swap_pinned_buffer_.get()) + packed.swap_offset,
          packed.expert_bytes, cudaMemcpyHostToDevice, swap_h2d_stream_));
      if (packed.gemm_input_size != 0) {
        using CudaT = typename OrtToCudaType<T>::type;
        if constexpr (kCpuOffloadSupported) {
          LaunchTransposeMoeExpertWeights(reinterpret_cast<const CudaT*>(layout),
                                          reinterpret_cast<CudaT*>(destination),
                                          packed.gemm_input_size, packed.gemm_output_size, swap_h2d_stream_);
        }
      }
    }
    return CUDA_CALL(cudaEventRecord(swap_transfer_complete_event_, swap_h2d_stream_));
  };
  const Status status = schedule();
  if (!status.IsOK()) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(swap_d2h_stream_)));
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(swap_h2d_stream_)));
    swap_phase_ = SwapPhase::Idle;
  }
  return status;
}

template <typename T>
Status MoE<T>::PrepareExpertSwapForInvocation(cudaStream_t stream, KernelPilot* pilot) const {
  if (swap_phase_ != SwapPhase::TransferInFlight) {
    return Status::OK();
  }
  const cudaError_t result = cudaEventQuery(swap_transfer_complete_event_);
  if (result == cudaErrorNotReady) {
    return Status::OK();
  }
  CUDA_RETURN_IF_ERROR(result);

  for (int input_idx : {2, 3, 4, 5}) {
    const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
    if (!packed.present) {
      continue;
    }
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        static_cast<char*>(packed.cuda_data.get()) + swap_cuda_slot_ * packed.expert_bytes,
        static_cast<const char*>(swap_cuda_staging_.get()) + packed.swap_offset,
        packed.expert_bytes, cudaMemcpyDeviceToDevice, stream));
  }
  for (int input_idx : {2, 3, 4, 5}) {
    auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
    if (!packed.present) {
      continue;
    }
    const size_t expert_elements = packed.expert_bytes / sizeof(T);
    const auto* outgoing = reinterpret_cast<const T*>(
        static_cast<const char*>(swap_outgoing_pinned_buffer_.get()) + packed.swap_offset);
    if (!packed.cpu_gemm_data.empty()) {
      std::copy_n(outgoing, expert_elements,
                  packed.cpu_gemm_data.data() + swap_cpu_slot_ * expert_elements);
    } else if (!packed.cpu_gemm_float_data.empty()) {
      if constexpr (std::is_same_v<T, BFloat16>) {
        auto* destination = packed.cpu_gemm_float_data.data() + swap_cpu_slot_ * expert_elements;
        for (size_t i = 0; i < expert_elements; ++i) {
          destination[i] = outgoing[i].ToFloat();
        }
      }
    } else {
      std::copy_n(outgoing, expert_elements,
                  packed.cpu_data.data() + swap_cpu_slot_ * expert_elements);
    }
  }

  cuda_experts_[swap_cuda_slot_] = swap_cpu_expert_;
  expert_map_[static_cast<size_t>(swap_cuda_expert_)] = -1;
  expert_map_[static_cast<size_t>(swap_cpu_expert_)] = static_cast<int>(swap_cuda_slot_);
  cpu_experts_[swap_cpu_slot_] = swap_cuda_expert_;
  cpu_expert_map_[static_cast<size_t>(swap_cuda_expert_)] = static_cast<int>(swap_cpu_slot_);
  cpu_expert_map_[static_cast<size_t>(swap_cpu_expert_)] = -1;
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device_expert_map_.get(), expert_map_.data(),
                                       expert_map_.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaEventRecord(swap_publication_complete_event_, stream));
  ORT_RETURN_IF_ERROR(pilot->PublishMoeExpertSwap(swap_cuda_expert_, swap_cpu_expert_));
  swap_phase_ = SwapPhase::PublicationInFlight;
  return Status::OK();
}
#endif

template <typename T>
Status MoE<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const auto input_if_not_packed = [context, this](int index) {
    return packed_inputs_[index].present ? nullptr : context->Input<Tensor>(index);
  };
  const Tensor* fc1_experts_weights = input_if_not_packed(2);
  const Tensor* fc1_experts_bias_optional = input_if_not_packed(3);
  const Tensor* fc2_experts_weights = input_if_not_packed(4);
  const Tensor* fc2_experts_bias_optional = input_if_not_packed(5);
  const Tensor* fc3_experts_weights_optional = input_if_not_packed(6);
  const Tensor* fc3_experts_bias_optional = input_if_not_packed(7);
  const bool use_packed_weights =
      kCpuOffloadSupported && packed_inputs_[2].present && packed_inputs_[4].present;

  const TensorShape* fc1_experts_weights_shape =
      use_packed_weights ? &packed_inputs_[2].shape : &fc1_experts_weights->Shape();
  const TensorShape* fc2_experts_weights_shape =
      use_packed_weights ? &packed_inputs_[4].shape : &fc2_experts_weights->Shape();
  const TensorShape* fc3_experts_weights_shape =
      use_packed_weights
          ? (packed_inputs_[6].present ? &packed_inputs_[6].shape : nullptr)
          : (fc3_experts_weights_optional != nullptr ? &fc3_experts_weights_optional->Shape() : nullptr);

  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;

  // Backward compatibility: the published gpt-oss-20b model (and any model exported by ORT < 1.27)
  // hard-coded the interleaved SwiGLU fusion layout and did not emit a swiglu_fusion attribute, so it
  // falls back to the default of 0 ("not fused"). When the activation is SwiGLU, swiglu_fusion is 0,
  // and there is no separate FC3 weight, the gate and value projections are actually pre-fused into FC1
  // (interleaved layout). Treat this as swiglu_fusion == 1 so those legacy models keep working.
  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == ActivationType::Swiglu && swiglu_fusion == 0 &&
      fc3_experts_weights_shape == nullptr) {
    swiglu_fusion = 1;
    LogSwigluFusionRemapOnce();
  }

  bool is_fused_swiglu = (activation_type_ == ActivationType::Swiglu) &&
                         (swiglu_fusion != 0) &&
                         (fc3_experts_weights_shape == nullptr);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights_shape, use_packed_weights ? nullptr : fc1_experts_bias_optional, nullptr, nullptr,
      fc2_experts_weights_shape, use_packed_weights ? nullptr : fc2_experts_bias_optional, nullptr, nullptr,
      fc3_experts_weights_shape, use_packed_weights ? nullptr : fc3_experts_bias_optional, nullptr, nullptr,
      1, is_fused_swiglu, 0));
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= moe_params.num_experts,
                    "MoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", moe_params.num_experts);

  using CudaT = typename OrtToCudaType<T>::type;

  void* stream_obj = GetComputeStream(context);
  cudaStream_t stream = Stream(context);
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  KernelPilot* moe_pilot = nullptr;
  bool use_probability_weighted_counters = false;
  if (cpu_offload_enabled_) {
    moe_pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(moe_pilot, "FP16/BF16 MoE CPU offload requires a KernelPilot.");
    use_probability_weighted_counters = moe_pilot->MoeExpertCounterProbabilityWeight() > 0.0;
    ORT_RETURN_IF_ERROR(moe_pilot->CompleteMoeUsage());
    ORT_RETURN_IF_ERROR(PrepareExpertSwapForInvocation(stream, moe_pilot));
  }
#endif

  auto& device_prop = GetDeviceProp();
  int sm = device_prop.major * 10 + device_prop.minor;

  // SM90 TMA WS kernels only support f16/bf16, not float32.
  // Force SM80 path for float32 to use legacy kernels.
  if constexpr (std::is_same_v<T, float>) {
    if (sm >= 90) {
      sm = 80;
    }
  }

  // Validate minimum dimensions for CUTLASS kernels.
  // SM >= 90 TMA WarpSpecialized: smallest tile is 128x16x128B (N=16 for FP16). K < tile_K handled by TMA.
  // SM < 90 Ampere GemmGrouped: smallest instantiated tile N=128, but CUTLASS predicates N < tile_N.
  // Alignment of dimensions to 128 bits is enforced separately in moe_kernels.cu.
  {
    constexpr int min_dim = 16;
    ORT_RETURN_IF(moe_params.hidden_size < min_dim,
                  "MoE CUDA kernel requires hidden_size >= ", min_dim,
                  " for SM", sm, ", got ", moe_params.hidden_size);
    ORT_RETURN_IF(moe_params.inter_size < min_dim,
                  "MoE CUDA kernel requires inter_size >= ", min_dim,
                  " for SM", sm, ", got ", moe_params.inter_size);
  }

  using onnxruntime::llm::kernels::cutlass_kernels::ActivationType;
  ActivationType kernel_activation_type = activation_type_;
  if (activation_type_ == ActivationType::Silu && fc3_experts_weights_optional != nullptr) {
    // Mixtral case: SiLU activation with separate FC3.
    // Kernel supports SwiGLU which is Linear * SiLU(Gate).
    // We map Mixtral to SwiGLU by packing weights as [FC3, FC1] (Linear, Gate).
    kernel_activation_type = ActivationType::Swiglu;
  }

  onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunner<CudaT, CudaT> moe_runner(sm,
                                                                                          kernel_activation_type,
                                                                                          normalize_routing_weights_,
                                                                                          use_sparse_mixer_);

  constexpr bool use_awq = false;
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};
  const int cuda_runner_num_experts = static_cast<int>(cuda_experts_.size());
  const bool run_cuda_experts = !cpu_offload_enabled_ || cuda_runner_num_experts > 0;
  bool compute_cpu_output = false;
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  const bool run_cpu_experts =
      cpu_offload_enabled_ && cuda_runner_num_experts < static_cast<int>(moe_params.num_experts);
  const bool async_cpu = run_cpu_experts && async_cpu_supported_;
  const bool synchronous_cpu = run_cpu_experts && !async_cpu;
  const size_t cpu_task_rows =
      async_cpu ? std::min(static_cast<size_t>(moe_params.num_rows), kMaxAsyncCpuRows)
                : static_cast<size_t>(moe_params.num_rows);
#endif
  const int workspace_num_experts =
      cpu_offload_enabled_
          ? std::max(static_cast<int>(k_), cuda_runner_num_experts)
          : static_cast<int>(moe_params.num_experts);

  if (run_cuda_experts) {
    if (onnxruntime::llm::common::getEnvForceDeterministicMOE()) {
      auto tactics = moe_runner.getTactics();
      if (!tactics.empty()) {
        moe_runner.setTactic(tactics[0], tactics[0]);
      }
    } else {
      std::lock_guard<std::mutex> profiler_lock(mGemmProfilerMutex);
      AllocatorPtr allocator;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
      mGemmProfiler.setAllocator(std::move(allocator));
      mGemmProfiler.setProfilerParams(workspace_num_experts, static_cast<int>(this->k_),
                                      static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size),
                                      static_cast<int64_t>(this->block_size_), kernel_activation_type,
                                      false, true, parallelism_config, sm);

      // Profiling launches grouped-GEMM kernels, records/synchronizes CUDA events, and
      // allocates/frees scratch from the temp allocator on the compute stream. All of these are
      // illegal while that stream is being captured into a CUDA graph; performing them corrupts the
      // capture. During capture we therefore skip profiling and reuse a config cached from an earlier
      // non-capturing run, falling back to the default tactic when nothing is cached.
      const bool stream_is_capturing = onnxruntime::llm::common::isCapturing(stream);

      onnxruntime::llm::nvinfer::DataType dtype = onnxruntime::llm::nvinfer::DataType::kFLOAT;
      if constexpr (std::is_same_v<CudaT, half>) {
        dtype = onnxruntime::llm::nvinfer::DataType::kHALF;
      } else if constexpr (std::is_same_v<CudaT, __nv_bfloat16>) {
        dtype = onnxruntime::llm::nvinfer::DataType::kBF16;
      }

      using onnxruntime::llm::kernels::cutlass_kernels::MoeGemmId;
      using onnxruntime::llm::kernels::weight_only::GemmDims;

      // GEMM 1
      MoeGemmId id1(static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.hidden_size), dtype, MoeGemmId::GemmType::Gemm1);
      if (!stream_is_capturing) {
        // profileTactics caches per (GemmId, M bucket); calling it every forward lets decode
        // (small M) and prefill (large M) each profile and select their own best tile shape.
        GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                      static_cast<int64_t>(moe_params.inter_size), static_cast<int64_t>(moe_params.hidden_size));
        mGemmProfiler.profileTactics(&moe_runner, dims, id1, stream);
      }
      auto config1 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), id1);

      // GEMM 2
      MoeGemmId id2(static_cast<int>(moe_params.hidden_size), static_cast<int>(moe_params.inter_size), dtype, MoeGemmId::GemmType::Gemm2);
      if (!stream_is_capturing) {
        GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                      static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size));
        mGemmProfiler.profileTactics(&moe_runner, dims, id2, stream);
      }
      auto config2 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), id2);

      // Capture-safe fallback: if profiling was skipped (graph capture) and no tuned config was
      // cached from a prior non-capturing run, use the runner's default tactic instead of leaving
      // the config unset.
      if (!config1 || !config2) {
        auto tactics = moe_runner.getTactics();
        if (!tactics.empty()) {
          if (!config1) {
            config1 = tactics[0];
          }
          if (!config2) {
            config2 = tactics[0];
          }
        }
      }

      moe_runner.setTactic(config1, config2);
    }
  }

  size_t ws_size = run_cuda_experts
                       ? moe_runner.getWorkspaceSize(
                             static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
                             static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(workspace_num_experts),
                             static_cast<size_t>(k_), kernel_activation_type, parallelism_config, use_awq)
                       : 0;

  // Scratch buffer for workspace + expert_scales + expert_indices + permutation_map.
  // Use checked arithmetic: these byte counts derive adjacent pointer offsets inside one allocation.
  size_t expanded_rows = SafeInt<size_t>(moe_params.num_rows) * SafeInt<size_t>(k_);
  size_t scales_bytes = expanded_rows * sizeof(float);
  size_t indices_bytes = expanded_rows * sizeof(int);
  size_t permutation_bytes = expanded_rows * sizeof(int);
  size_t total_scratch_bytes = SafeInt<size_t>(ws_size) + scales_bytes + indices_bytes + permutation_bytes;

  auto work_space = GetScratchBuffer<void>(total_scratch_bytes, stream_obj);
  char* workspace_ptr = reinterpret_cast<char*>(work_space.get());
  float* expert_scales = reinterpret_cast<float*>(workspace_ptr + ws_size);
  int* expert_indices = reinterpret_cast<int*>(workspace_ptr + ws_size + scales_bytes);
  int* unpermuted_row_to_permuted_row = reinterpret_cast<int*>(workspace_ptr + ws_size + scales_bytes + indices_bytes);

  ORT_RETURN_IF(cpu_offload_enabled_ && use_sparse_mixer_,
                "FP16/BF16 MoE CPU offload does not support sparse_mixer.");

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  IAllocatorUniquePtr<T> host_input;
  IAllocatorUniquePtr<int> host_expert_indices;
  IAllocatorUniquePtr<float> host_expert_scales;
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t input_copy_ready = nullptr;
  cudaEvent_t routing_copy_ready = nullptr;
  bool input_copy_complete = false;
  bool routing_copy_pending = false;
  bool routing_copy_recorded = false;
  bool routing_copy_complete = false;
  auto release_input_copy_events = gsl::finally([&]() {
    if (routing_copy_pending && !routing_copy_complete) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(routing_copy_recorded
                                            ? cudaEventSynchronize(routing_copy_ready)
                                            : cudaStreamSynchronize(stream)));
    }
    if (synchronous_cpu && !input_copy_complete && input_copy_stream_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(input_copy_stream_)));
    }
    if (routing_copy_ready != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(routing_copy_ready)));
    }
    if (input_copy_ready != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(input_copy_ready)));
    }
    if (input_ready != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaEventDestroy(input_ready)));
    }
  });
  if (cpu_offload_enabled_) {
    ORT_RETURN_IF_ERROR(PrepareUsage(context, moe_pilot, static_cast<size_t>(moe_params.num_experts),
                                     cpu_task_rows,
                                     static_cast<size_t>(moe_params.hidden_size),
                                     static_cast<size_t>(moe_params.inter_size), is_fused_swiglu, async_cpu));
    ORT_RETURN_IF_ERROR(moe_pilot->Moe().BeginInvocation(static_cast<size_t>(moe_params.num_experts)));
  }
  if (synchronous_cpu) {
    host_expert_indices = AllocateBufferOnCPUPinned<int>(expanded_rows);
    ORT_RETURN_IF_NOT(host_expert_indices,
                      "Failed to allocate a pinned routing buffer for FP16/BF16 MoE CPU offload.");
    CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&routing_copy_ready, cudaEventDisableTiming));
    if (run_cpu_experts || use_probability_weighted_counters) {
      host_expert_scales = AllocateBufferOnCPUPinned<float>(expanded_rows);
      ORT_RETURN_IF_NOT(host_expert_scales,
                        "Failed to allocate a pinned routing probability buffer for FP16/BF16 MoE CPU offload.");
    }
    if (run_cpu_experts) {
      const size_t host_element_count =
          static_cast<size_t>(SafeInt<int64_t>(moe_params.num_rows) * moe_params.hidden_size);
      host_input = AllocateBufferOnCPUPinned<T>(host_element_count);
      ORT_RETURN_IF_NOT(host_input,
                        "Failed to allocate pinned host buffers for FP16/BF16 MoE CPU execution.");

      CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
      CUDA_RETURN_IF_ERROR(cudaEventCreateWithFlags(&input_copy_ready, cudaEventDisableTiming));
      {
        std::lock_guard<std::mutex> input_copy_lock(input_copy_mutex_);
        CUDA_RETURN_IF_ERROR(cudaEventRecord(input_ready, stream));
        CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(input_copy_stream_, input_ready, 0));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_input.get(), input->DataRaw(),
                                             input->SizeInBytes(), cudaMemcpyDeviceToHost,
                                             input_copy_stream_));
        CUDA_RETURN_IF_ERROR(cudaEventRecord(input_copy_ready, input_copy_stream_));
      }
    }
  }
#endif

  // Perform Softmax + TopK
  bool is_fp16 = input->IsDataType<MLFloat16>();
  bool is_bf16 = input->IsDataType<BFloat16>();

  if (use_sparse_mixer_) {
    ORT_ENFORCE(k_ == 2, "Sparse mixer only supports k=2");
    ORT_ENFORCE(moe_params.num_experts == 8 || moe_params.num_experts == 16,
                "Sparse mixer only supports 8 or 16 experts, got ", moe_params.num_experts);

    if (is_fp16) {
      LaunchSparseMixerTop2(
          reinterpret_cast<const half*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          unpermuted_row_to_permuted_row,  // source_rows
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          stream);
    } else if (is_bf16) {
      LaunchSparseMixerTop2(
          reinterpret_cast<const __nv_bfloat16*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          unpermuted_row_to_permuted_row,
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          stream);
    } else {
      LaunchSparseMixerTop2(
          reinterpret_cast<const float*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          unpermuted_row_to_permuted_row,
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          stream);
    }
  } else {
    // Standard Softmax + TopK
    if (is_fp16) {
      LaunchSoftmaxTopK(
          reinterpret_cast<const half*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_),
          normalize_routing_weights_,
          stream);
    } else if (is_bf16) {
      LaunchSoftmaxTopK(
          reinterpret_cast<const __nv_bfloat16*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_),
          normalize_routing_weights_,
          stream);
    } else {
      LaunchSoftmaxTopK(
          reinterpret_cast<const float*>(router_probs->DataRaw()),
          expert_scales,
          expert_indices,
          static_cast<int>(moe_params.num_rows),
          static_cast<int>(moe_params.num_experts),
          static_cast<int>(k_),
          normalize_routing_weights_,
          stream);
    }
  }

  Tensor* output = context->Output(0, input->Shape());
  const int* runner_expert_indices = expert_indices;
  IAllocatorUniquePtr<void> remapped_expert_indices_buffer;
  IAllocatorUniquePtr<void> cpu_output_device_buffer;

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  IAllocatorUniquePtr<void> cpu_input_snapshot;
  const auto* cpu_input_data = reinterpret_cast<const CudaT*>(input->DataRaw());
  if (async_cpu && run_cuda_experts && input->DataRaw() == output->DataRaw()) {
    // CUDA writes every output row before later CPU chunks have packed their input.
    cpu_input_snapshot = GetScratchBuffer<void>(input->SizeInBytes(), stream_obj);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_input_snapshot.get(), input->DataRaw(),
                                         input->SizeInBytes(), cudaMemcpyDeviceToDevice, stream));
    cpu_input_data = static_cast<const CudaT*>(cpu_input_snapshot.get());
  }

  auto schedule_async_cpu_chunk = [&](size_t row_offset, size_t row_count) -> Status {
    auto& task = *cpu_task_;
    task.num_rows = row_count;
    task.completion->store(0, std::memory_order_release);
    task.status = Status::OK();
    if constexpr (kCpuOffloadSupported) {
      const size_t route_offset = row_offset * task.k;
      LaunchPackMoeCpuRows(
          cpu_input_data + row_offset * task.hidden_size,
          expert_indices + route_offset, expert_scales + route_offset,
          static_cast<const int*>(device_expert_map_.get()),
          reinterpret_cast<CudaT*>(task.device_input), task.device_ids, task.device_scales,
          task.device_rows, task.device_count, reinterpret_cast<uint32_t*>(task.device_completion),
          task.num_rows, task.hidden_size, task.k, stream);
    }
    CUDA_RETURN_IF_ERROR(cudaEventRecord(task.routes_ready, stream));
    CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(input_copy_stream_, task.routes_ready, 0));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(task.host_count.get(), task.device_count, sizeof(uint32_t),
                                         cudaMemcpyDeviceToHost, input_copy_stream_));
    {
      std::lock_guard<std::mutex> lock(cpu_worker_mutex_);
      cpu_task_finished_ = false;
    }
    CUDA_RETURN_IF_ERROR(cudaLaunchHostFunc(input_copy_stream_, NotifyCpuWorker, const_cast<MoE<T>*>(this)));
    cpu_callback_scheduled_ = true;
    return Status::OK();
  };

  auto complete_async_cpu_chunk = [&](size_t row_offset) -> Status {
    auto& task = *cpu_task_;
    {
      std::unique_lock<std::mutex> lock(cpu_worker_mutex_);
      cpu_worker_condition_.wait(lock, [&]() { return cpu_task_finished_; });
      ORT_RETURN_IF_ERROR(task.status);
    }
    if (*task.host_count != 0) {
      CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, task.input_ready, 0));
    }
    if constexpr (kCpuOffloadSupported) {
      LaunchScatterAddMoeCpuOutput(
          reinterpret_cast<const CudaT*>(task.device_output),
          reinterpret_cast<CudaT*>(output->MutableDataRaw()) + row_offset * task.hidden_size,
          task.device_rows, task.device_count, task.num_rows, task.hidden_size, stream);
    }
    return Status::OK();
  };

  if (cpu_offload_enabled_) {
    ORT_RETURN_IF_NOT(use_packed_weights && expert_map_.size() == static_cast<size_t>(moe_params.num_experts),
                      "FP16/BF16 MoE CPU offload weights were not initialized.");
    auto* device_counters = static_cast<double*>(device_usage_.get());
    auto* first_routes = reinterpret_cast<int64_t*>(device_counters + moe_params.num_experts);
    auto* probabilities = reinterpret_cast<float*>(first_routes + moe_params.num_experts);
    LaunchUpdateMoeExpertCounters(expert_indices, expert_scales, expanded_rows, device_counters,
                                  probabilities, first_routes, static_cast<size_t>(moe_params.num_experts),
                                  moe_pilot->MoeExpertCounterAlpha(), moe_pilot->MoeExpertCounterProbabilityWeight(), stream);
    if (async_cpu) {
      ORT_RETURN_IF_ERROR(schedule_async_cpu_chunk(0, cpu_task_rows));
    }
    if (synchronous_cpu) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_expert_indices.get(), expert_indices,
                                           indices_bytes, cudaMemcpyDeviceToHost, stream));
      routing_copy_pending = true;
      if (run_cpu_experts || use_probability_weighted_counters) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_expert_scales.get(), expert_scales,
                                             scales_bytes, cudaMemcpyDeviceToHost, stream));
      }
      if (run_cpu_experts) {
        // One event covers routing and the input DMA, including the MayInplace dependency.
        CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(stream, input_copy_ready, 0));
      }
      CUDA_RETURN_IF_ERROR(cudaEventRecord(routing_copy_ready, stream));
      routing_copy_recorded = true;
    }

    if (!cuda_experts_.empty()) {
      remapped_expert_indices_buffer = GetScratchBuffer<void>(indices_bytes, stream_obj);
      auto* remapped_expert_indices =
          static_cast<int*>(remapped_expert_indices_buffer.get());
      LaunchRemapMoeExpertIndices(expert_indices, remapped_expert_indices,
                                  static_cast<const int*>(device_expert_map_.get()),
                                  expanded_rows, stream);
      runner_expert_indices = remapped_expert_indices;
    }
  }
#endif

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (routing_snapshot_ && !cpu_offload_enabled_) {
    auto* pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(pilot, "MoE expert tracking is enabled but its collector is unavailable.");
    ORT_RETURN_IF_ERROR(routing_snapshot_->BeginInvocation(pilot->Moe(), static_cast<size_t>(moe_params.num_experts)));
    ORT_RETURN_IF_ERROR(routing_snapshot_->Capture(expert_indices, expanded_rows, stream));
  }
#endif

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params{};

  // =============================================================================
  // WEIGHT PACKING
  // =============================================================================
  // Prepare buffers for CutlassMoeFCRunner.
  // For standard MoE, we copy weights directly.
  // For SwiGLU with separate gates (e.g. Mixtral), we interleave FC1 and FC3 weights.
  // =============================================================================

  if (run_cuda_experts) {
    // Calculate buffer sizes
    size_t fc1_block_size = static_cast<size_t>(moe_params.inter_size) * static_cast<size_t>(moe_params.hidden_size);
    int E = cpu_offload_enabled_ ? cuda_runner_num_experts : static_cast<int>(moe_params.num_experts);

    std::array<IAllocatorUniquePtr<void>, 8> runtime_cuda_inputs;
    const auto packed_cuda_data = [&](int input_idx, const Tensor* tensor) -> const CudaT* {
      if constexpr (kCpuOffloadSupported) {
        const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
        if (packed.present) {
          return static_cast<const CudaT*>(packed.cuda_data.get());
        }
        if (tensor->Location().device.Type() == OrtDevice::CPU) {
          auto& staged = runtime_cuda_inputs[static_cast<size_t>(input_idx)];
          staged = GetScratchBuffer<void>(tensor->SizeInBytes(), stream_obj);
          CUDA_CALL_THROW(cudaMemcpyAsync(staged.get(), tensor->DataRaw(), tensor->SizeInBytes(),
                                          cudaMemcpyHostToDevice, stream));
          return static_cast<const CudaT*>(staged.get());
        }
      }
      return reinterpret_cast<const CudaT*>(tensor->DataRaw());
    };

    // FC1 Handling
    const CudaT* fc1_input_ptr = packed_cuda_data(2, fc1_experts_weights);
    const CudaT* fc1_processed_ptr = fc1_input_ptr;
    IAllocatorUniquePtr<void> fc1_processed_buffer;

    if (fc3_experts_weights_shape != nullptr) {
      const CudaT* fc3_input_ptr = packed_cuda_data(6, fc3_experts_weights_optional);
      size_t fc1_total_size = E * 2 * fc1_block_size * sizeof(CudaT);
      fc1_processed_buffer = GetScratchBuffer<void>(fc1_total_size, stream_obj);
      CudaT* fc1_fc3_processed_ptr = reinterpret_cast<CudaT*>(fc1_processed_buffer.get());
      fc1_processed_ptr = fc1_fc3_processed_ptr;

      for (int e = 0; e < E; ++e) {
        CudaT* dest_fc1 = fc1_fc3_processed_ptr + e * 2 * fc1_block_size;
        CudaT* dest_fc3 = fc1_fc3_processed_ptr + e * 2 * fc1_block_size + fc1_block_size;
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc1, fc1_input_ptr + e * fc1_block_size,
                                             fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc3, fc3_input_ptr + e * fc1_block_size,
                                             fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
      }
    }

    const CudaT* fc2_processed_ptr = packed_cuda_data(4, fc2_experts_weights);
    const CudaT* fc1_bias_ptr =
        use_packed_weights
            ? (packed_inputs_[3].present ? static_cast<const CudaT*>(packed_inputs_[3].cuda_data.get()) : nullptr)
            : (fc1_experts_bias_optional == nullptr ? nullptr : packed_cuda_data(3, fc1_experts_bias_optional));
    const CudaT* fc2_bias_ptr =
        use_packed_weights
            ? (packed_inputs_[5].present ? static_cast<const CudaT*>(packed_inputs_[5].cuda_data.get()) : nullptr)
            : (fc2_experts_bias_optional == nullptr ? nullptr : packed_cuda_data(5, fc2_experts_bias_optional));

    moe_runner.runMoe(
        reinterpret_cast<const CudaT*>(input->template Data<T>()),
        nullptr,
        runner_expert_indices,
        expert_scales,
        fc1_processed_ptr,
        fc1_bias_ptr,
        kernel_activation_type,
        fc2_processed_ptr,
        fc2_bias_ptr,
        quant_params,
        static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.hidden_size),
        static_cast<int>(moe_params.inter_size), E,
        static_cast<int>(k_),
        workspace_ptr,
        reinterpret_cast<void*>(output->template MutableData<T>()),
        unpermuted_row_to_permuted_row,
        parallelism_config,
        [&]() {
          onnxruntime::llm::kernels::cutlass_kernels::ActivationParams params(kernel_activation_type);
          params.alpha = activation_alpha_;
          params.beta = activation_beta_;
          params.swiglu_fusion = swiglu_fusion;
          params.limit = swiglu_limit_;
          return params;
        }(),
        onnxruntime::llm::kernels::cutlass_kernels::FusedRoutingParams{},
        stream);
  }

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (synchronous_cpu) {
    // The CUDA path is already enqueued above, so mixed execution overlaps its kernels with the host GEMMs.
    ORT_RETURN_IF_NOT(moe_pilot, "FP16/BF16 MoE CPU offload requires a KernelPilot.");
    CUDA_RETURN_IF_ERROR(cudaEventSynchronize(routing_copy_ready));
    routing_copy_complete = true;
    input_copy_complete = run_cpu_experts;
    if (use_probability_weighted_counters) {
      ORT_RETURN_IF_ERROR(moe_pilot->Moe().CollectWeighted(
          gsl::make_span(host_expert_indices.get(), expanded_rows),
          gsl::make_span(host_expert_scales.get(), expanded_rows)));
    } else {
      ORT_RETURN_IF_ERROR(
          moe_pilot->Moe().Collect(gsl::make_span(host_expert_indices.get(), expanded_rows)));
    }

    if (run_cpu_experts) {
      compute_cpu_output = std::any_of(host_expert_indices.get(), host_expert_indices.get() + expanded_rows,
                                       [&](int expert) { return cpu_expert_map_[static_cast<size_t>(expert)] >= 0; });
    }
    if (compute_cpu_output) {
      auto cpu_activation_type = ::onnxruntime::contrib::ActivationType::Identity;
      switch (activation_type_) {
        case ActivationType::Relu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Relu;
          break;
        case ActivationType::Gelu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Gelu;
          break;
        case ActivationType::Silu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Silu;
          break;
        case ActivationType::Identity:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::Identity;
          break;
        case ActivationType::Swiglu:
          cpu_activation_type = ::onnxruntime::contrib::ActivationType::SwiGLU;
          break;
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                                 "Unsupported FP16/BF16 MoE CPU-offload activation.");
      }

      auto optional_host_data = [&](int input_idx) -> gsl::span<const T> {
        const auto& packed = packed_inputs_[static_cast<size_t>(input_idx)];
        return packed.present ? gsl::make_span(packed.cpu_data) : gsl::span<const T>{};
      };

      const size_t host_element_count =
          static_cast<size_t>(SafeInt<int64_t>(moe_params.num_rows) * moe_params.hidden_size);
      auto host_cpu_output = AllocateBufferOnCPUPinned<T>(host_element_count);
      ORT_RETURN_IF_NOT(host_cpu_output, "Failed to allocate the pinned FP16/BF16 MoE CPU output buffer.");
      const ::onnxruntime::contrib::MoeCpuOffloadParameters cpu_parameters{
          cpu_activation_type, activation_alpha_, activation_beta_, swiglu_limit_, is_fused_swiglu};
      if constexpr (std::is_same_v<T, MLFloat16>) {
        auto host_data = [&](int input_idx) {
          return gsl::make_span(packed_inputs_[static_cast<size_t>(input_idx)].cpu_gemm_data);
        };
        ORT_RETURN_IF_ERROR(::onnxruntime::contrib::ComputeMoeCpuOffloadedExpertsFp16(
            gsl::make_span(host_input.get(), host_element_count),
            gsl::make_span(host_expert_indices.get(), expanded_rows),
            gsl::make_span(host_expert_scales.get(), expanded_rows), cpu_expert_map_,
            host_data(2), optional_host_data(3), host_data(4), optional_host_data(5),
            moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
            cpu_parameters, gsl::make_span(host_cpu_output.get(), host_element_count),
            context->GetOperatorThreadPool()));
      } else if constexpr (std::is_same_v<T, BFloat16>) {
        auto host_data = [&](int input_idx) {
          return gsl::make_span(packed_inputs_[static_cast<size_t>(input_idx)].cpu_gemm_float_data);
        };
        ORT_RETURN_IF_ERROR(::onnxruntime::contrib::ComputeMoeCpuOffloadedExpertsBFloat16(
            gsl::make_span(host_input.get(), host_element_count),
            gsl::make_span(host_expert_indices.get(), expanded_rows),
            gsl::make_span(host_expert_scales.get(), expanded_rows), cpu_expert_map_,
            host_data(2), optional_host_data(3), host_data(4), optional_host_data(5),
            moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
            cpu_parameters, gsl::make_span(host_cpu_output.get(), host_element_count),
            context->GetOperatorThreadPool()));
      }

      void* cpu_output_destination = output->MutableDataRaw();
      if (!cuda_experts_.empty()) {
        cpu_output_device_buffer = GetScratchBuffer<void>(input->SizeInBytes(), stream_obj);
        cpu_output_destination = cpu_output_device_buffer.get();
      }
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cpu_output_destination, host_cpu_output.get(),
                                           input->SizeInBytes(), cudaMemcpyHostToDevice, stream));
      AddDeferredReleaseCPUPtr(host_cpu_output.release(), stream_obj);
    }
  }
  if (async_cpu) {
    size_t row_offset = 0;
    while (row_offset < static_cast<size_t>(moe_params.num_rows)) {
      if (!run_cuda_experts) {
        const size_t chunk_bytes =
            SafeInt<size_t>(cpu_task_->num_rows) * cpu_task_->hidden_size * sizeof(T);
        auto* chunk_output = reinterpret_cast<CudaT*>(output->MutableDataRaw()) +
                             row_offset * cpu_task_->hidden_size;
        CUDA_RETURN_IF_ERROR(cudaMemsetAsync(chunk_output, 0, chunk_bytes, stream));
      }
      ORT_RETURN_IF_ERROR(complete_async_cpu_chunk(row_offset));
      row_offset += cpu_task_->num_rows;
      if (row_offset < static_cast<size_t>(moe_params.num_rows)) {
        const size_t row_count =
            std::min(static_cast<size_t>(moe_params.num_rows) - row_offset, kMaxAsyncCpuRows);
        ORT_RETURN_IF_ERROR(schedule_async_cpu_chunk(row_offset, row_count));
      }
    }
  }
#endif

  if (compute_cpu_output && !cuda_experts_.empty()) {
    if constexpr (std::is_same_v<T, MLFloat16>) {
      LaunchAddMoeFp16Output(
          reinterpret_cast<half*>(output->MutableDataRaw()),
          static_cast<const half*>(cpu_output_device_buffer.get()),
          static_cast<size_t>(output->Shape().Size()), stream);
    } else if constexpr (std::is_same_v<T, BFloat16>) {
      LaunchAddMoeBf16Output(
          reinterpret_cast<__nv_bfloat16*>(output->MutableDataRaw()),
          static_cast<const __nv_bfloat16*>(cpu_output_device_buffer.get()),
          static_cast<size_t>(output->Shape().Size()), stream);
    }
  }

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  if (routing_snapshot_ && !cpu_offload_enabled_) {
    ORT_RETURN_IF_ERROR(routing_snapshot_->Consume());
  }
  if (cpu_offload_enabled_) {
    const size_t usage_bytes = SafeInt<size_t>(moe_params.num_experts) *
                               (sizeof(double) + sizeof(int64_t) + sizeof(float));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(host_usage_.get(), device_usage_.get(), usage_bytes,
                                         cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(cudaEventRecord(usage_complete_event_, stream));
    usage_recorded_ = true;
    CUDA_RETURN_IF_ERROR(cudaEventRecord(last_expert_use_event_, stream));
  }
#endif

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
