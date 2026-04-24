// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// cuda_kernel_adapter.h — Compatibility shim for migrating CUDA kernels to the
// plugin EP architecture.
//
// This header provides:
//   - CudaKernel base class (scratch buffers, CUDA handles, etc.)
//   - Error-return macros (CUDA_RETURN_IF_ERROR, etc.)
//   - Type mapping helpers (ToCudaType)
//   - Math/compute shims (HalfGemmOptions, CublasMathModeSetter)
//   - Self-registering BuildKernelCreateInfo<> macros via PluginKernelCollector
//   - CUDAExecutionProvider shim class
//   - CPU provider shims for the plugin build

#pragma once

#include "core/common/status.h"
#include "core/common/narrow.h"
#include "core/common/float16.h"
#include "core/common/float8.h"
#include "core/framework/float4.h"
#include "core/framework/allocator.h"
#include "core/framework/stream_handles.h"
#include "core/framework/tensor_shape.h"
#include "core/util/math.h"
#include <gsl/gsl>

#include <cublas_v2.h>
#include <cudnn.h>
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

// ===================================================================
// Macros will be defined later to override core definitions.
// ===================================================================

#include "core/providers/cuda/plugin/cuda_stream_plugin.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

// Forward-declare CudaStream so adapter overloads accepting CudaStream* compile
// without pulling in cuda_stream_handle.h (which depends on CUDAExecutionProvider).
namespace onnxruntime {
struct CudaStream;

// Lightweight Stream shim for plugin build: wraps a raw cudaStream_t as a
// framework-compatible Stream* that can be passed to _impl.cu functions which
// call stream->GetHandle().  Stack-allocated; does NOT own the stream.
// Only available in .cc translation units (not .cu) since Stream is incomplete in NVCC context.
#ifndef __CUDACC__
struct PluginStreamShim : public onnxruntime::Stream {
  explicit PluginStreamShim(void* cuda_stream_handle)
      : onnxruntime::Stream(cuda_stream_handle,
                            OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                      OrtDevice::VendorIds::NVIDIA, 0)) {}
};

class OrtStreamAdapter {
 public:
  explicit OrtStreamAdapter(void* cuda_stream_handle)
      : plugin_stream_shim_(cuda_stream_handle), stream_(&plugin_stream_shim_) {}

  onnxruntime::Stream* get() const { return stream_; }
  operator onnxruntime::Stream*() const { return stream_; }

 private:
  PluginStreamShim plugin_stream_shim_;
  onnxruntime::Stream* stream_;
};
#else
class OrtStreamAdapter {
 public:
  explicit OrtStreamAdapter(void* cuda_stream_handle)
      : stream_(static_cast<onnxruntime::Stream*>(cuda_stream_handle)) {}

  onnxruntime::Stream* get() const { return stream_; }
  operator onnxruntime::Stream*() const { return stream_; }

 private:
  onnxruntime::Stream* stream_;
};
#endif
}  // namespace onnxruntime

// ===================================================================
// Section 1: Include path selection
// ===================================================================

#include "core/graph/constants.h"
#include "ep/adapters.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

// Forward declaration of GetEnvironmentVar for plugin builds on Windows.
// Defined in provider_api_shims.cc; mirrors the provider_api.h declaration
// which is a no-op when BUILD_CUDA_EP_AS_PLUGIN is set.
std::string GetEnvironmentVar(const std::string& var_name);
}  // namespace onnxruntime

namespace onnxruntime {
namespace cuda {

#ifndef CUDA_STREAM_FROM_CTX
// Helper for kernels that need a cudaStream_t from OpKernelContext in plugin build.
#define CUDA_STREAM_FROM_CTX(ctx) static_cast<cudaStream_t>(GetComputeStream(ctx))
#endif

// Forward declare the template for kernel registration macros to specialize
// inside the onnxruntime::cuda namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

// Tensor creation helper to replace deprecated Tensor::Create
inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
namespace cuda {

// Forward declare the template for kernel registration macros to specialize
// inside the onnxruntime::contrib::cuda namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda
}  // namespace contrib
#endif
}  // namespace onnxruntime

// ===================================================================
// Section 2: Error-return macros (redefined for all plugin paths)
// ===================================================================

// Redefine error macros for ported code to use our adapter-specific Status translation
#undef CUDA_RETURN_IF_ERROR
#define CUDA_RETURN_IF_ERROR(expr)                                                                                                                             \
  {                                                                                                                                                            \
    cudaError_t _err = (expr);                                                                                                                                 \
    if (_err != cudaSuccess) {                                                                                                                                 \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    }                                                                                                                                                          \
  }

#undef CUBLAS_RETURN_IF_ERROR
#define CUBLAS_RETURN_IF_ERROR(expr)                                                                                                                                              \
  {                                                                                                                                                                               \
    cublasStatus_t _status = (expr);                                                                                                                                              \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                                                                                                                       \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuBLAS error: ") + std::to_string(static_cast<int>(_status))); \
    }                                                                                                                                                                             \
  }

#undef CUDNN_RETURN_IF_ERROR
#define CUDNN_RETURN_IF_ERROR(expr)                                                                                                                                 \
  {                                                                                                                                                                 \
    cudnnStatus_t _status = (expr);                                                                                                                                 \
    if (_status != CUDNN_STATUS_SUCCESS) {                                                                                                                          \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuDNN error: ") + cudnnGetErrorString(_status)); \
    }                                                                                                                                                               \
  }

#undef CUFFT_RETURN_IF_ERROR
#define CUFFT_RETURN_IF_ERROR(expr)                                                                                                                                 \
  {                                                                                                                                                                 \
    cufftResult _status = (expr);                                                                                                                                   \
    if (_status != CUFFT_SUCCESS) {                                                                                                                                 \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuFFT error: ") + std::to_string((int)_status)); \
    }                                                                                                                                                               \
  }

// ===================================================================
// Section 3: Self-registering kernel collector using BuildKernelCreateInfo<>
//
// Each ONNX_OPERATOR_*_KERNEL_EX macro expansion both:
//   1. Creates a BuildKernelCreateInfo<CLASS>() template specialization
//      (identical to the framework's macro output, but using adapter types)
//   2. Auto-registers the BuildKernelCreateInfoFn pointer into a global
//      PluginKernelCollector singleton
//
// At registration time, the factory iterates the collector and calls
// adapter::KernelRegistry::Register(build_fn()) for each compiled kernel.
// Only ops whose .cc files are actually compiled get registered.
// ===================================================================

#include <vector>

namespace onnxruntime {
namespace cuda {

/// Singleton collector for BuildKernelCreateInfoFn pointers.
/// Each compiled kernel .cc file's macro expansion auto-registers here.
///
/// Thread-safety: Instance() uses a function-local static (C++11 §6.7/4:
/// constructed exactly once, even under concurrent first-access). Add()
/// is guarded by a mutex for formal correctness across translation units,
/// though in practice all calls occur during static initialization.
class PluginKernelCollector {
 public:
  static PluginKernelCollector& Instance() {
    static PluginKernelCollector instance;
    return instance;
  }

  void Add(BuildKernelCreateInfoFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back(fn);
  }
  std::vector<BuildKernelCreateInfoFn> Entries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
  }

 private:
  std::vector<BuildKernelCreateInfoFn> entries_;
  mutable std::mutex mutex_;
};

}  // namespace cuda
}  // namespace onnxruntime

// --- Macro overrides: produce BuildKernelCreateInfo<> AND auto-register ---
//
// These macros mirror the framework's ONNX_OPERATOR_*_KERNEL_EX definitions
// from core/framework/op_kernel.h, but additionally register each
// BuildKernelCreateInfoFn into PluginKernelCollector at static init time.
#define ORT_ADAPTER_CONCAT_IMPL(x, y) x##y
#define ORT_ADAPTER_CONCAT(x, y) ORT_ADAPTER_CONCAT_IMPL(x, y)

// The provider parameter are not used in below macros since we are hardcoding the provider to cuda plugin.
#define CUDA_PLUGIN_EP ::onnxruntime::kCudaPluginExecutionProvider

#undef ONNX_OPERATOR_KERNEL_EX
#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)                           \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                                \
  template <>                                                                                        \
  KernelCreateInfo                                                                                   \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {            \
    return KernelCreateInfo(                                                                         \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(), \
        static_cast<KernelCreatePtrFn>(                                                              \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {   \
              out = std::make_unique<__VA_ARGS__>(info);                                             \
              return Status::OK();                                                                   \
            }));                                                                                     \
  }                                                                                                  \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_, __COUNTER__) =                 \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                   \
           &BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>),    \
       true);

#undef ONNX_OPERATOR_VERSIONED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, provider, builder, ...)                 \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name);                      \
  template <>                                                                                                     \
  KernelCreateInfo                                                                                                \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name)>() {  \
    return KernelCreateInfo(                                                                                      \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(CUDA_PLUGIN_EP).Build(), \
        static_cast<KernelCreatePtrFn>(                                                                           \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                \
              out = std::make_unique<__VA_ARGS__>(info);                                                          \
              return Status::OK();                                                                                \
            }));                                                                                                  \
  }                                                                                                               \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_, __COUNTER__) =                              \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(                                      \
               provider, domain, startver, endver, name)>),                                                       \
       true);

#undef ONNX_OPERATOR_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...)                        \
  class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name);                             \
  template <>                                                                                                 \
  KernelCreateInfo                                                                                            \
  BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>() {         \
    return KernelCreateInfo(                                                                                  \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(),          \
        static_cast<KernelCreatePtrFn>(                                                                       \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {            \
              out = std::make_unique<__VA_ARGS__>(info);                                                      \
              return Status::OK();                                                                            \
            }));                                                                                              \
  }                                                                                                           \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type##_, __COUNTER__) =                 \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                            \
           &BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>), \
       true);

#undef ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, ...)     \
  class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name);          \
  template <>                                                                                                     \
  KernelCreateInfo                                                                                                \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(                                          \
      provider, domain, startver, endver, type, name)>() {                                                        \
    return KernelCreateInfo(                                                                                      \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(CUDA_PLUGIN_EP).Build(), \
        static_cast<KernelCreatePtrFn>(                                                                           \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                \
              out = std::make_unique<__VA_ARGS__>(info);                                                          \
              return Status::OK();                                                                                \
            }));                                                                                                  \
  }                                                                                                               \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type##_, __COUNTER__) =                     \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(                                \
               provider, domain, startver, endver, type, name)>),                                                 \
       true);

#undef ONNX_OPERATOR_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(name, domain, ver, type1, type2, provider, builder, ...)                        \
  class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name);                             \
  template <>                                                                                                             \
  KernelCreateInfo                                                                                                        \
  BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name)>() {         \
    return KernelCreateInfo(                                                                                              \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(),                      \
        static_cast<KernelCreatePtrFn>(                                                                                   \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                        \
              out = std::make_unique<__VA_ARGS__>(info);                                                                  \
              return Status::OK();                                                                                        \
            }));                                                                                                          \
  }                                                                                                                       \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type1##_##type2##_, __COUNTER__) =                  \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                        \
           &BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, name)>), \
       true);

#undef ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(name, domain, startver, endver, type1, type2,                    \
                                                    provider, builder, ...)                                          \
  class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type1, type2, name); \
  template <>                                                                                                        \
  KernelCreateInfo                                                                                                   \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(                                         \
      provider, domain, startver, endver, type1, type2, name)>() {                                                   \
    return KernelCreateInfo(                                                                                         \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(CUDA_PLUGIN_EP).Build(),    \
        static_cast<KernelCreatePtrFn>(                                                                              \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                   \
              out = std::make_unique<__VA_ARGS__>(info);                                                             \
              return Status::OK();                                                                                   \
            }));                                                                                                     \
  }                                                                                                                  \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type1##_##type2##_, __COUNTER__) =             \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                   \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(                               \
               provider, domain, startver, endver, type1, type2, name)>),                                            \
       true);

#undef ONNX_OPERATOR_THREE_TYPED_KERNEL_EX
#define ONNX_OPERATOR_THREE_TYPED_KERNEL_EX(name, domain, ver, type1, type2, type3, provider, builder, ...)                \
  class ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, type3, name);                     \
  template <>                                                                                                              \
  KernelCreateInfo                                                                                                         \
  BuildKernelCreateInfo<ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type1, type2, type3, name)>() { \
    return KernelCreateInfo(                                                                                               \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(CUDA_PLUGIN_EP).Build(),                       \
        static_cast<KernelCreatePtrFn>(                                                                                    \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {                         \
              out = std::make_unique<__VA_ARGS__>(info);                                                                   \
              return Status::OK();                                                                                         \
            }));                                                                                                           \
  }                                                                                                                        \
  static const bool ORT_ADAPTER_CONCAT(ORT_ADAPTER_AUTOREG_##name##_##type1##_##type2##_##type3##_, __COUNTER__) =         \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                                         \
           &BuildKernelCreateInfo<ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME(                                             \
               provider, domain, ver, type1, type2, type3, name)>),                                                        \
       true);

// ===================================================================
// Section 4: Logging shim (adapter path only)
// LOGS_DEFAULT is re-routed through ep::adapter::LoggingManager, which
// holds the ORT default logger set up in CudaEpFactory::CudaEpFactory.
// All severity levels (including ERROR/WARNING) are forwarded to the
// ORT logger; no log output is suppressed.
// ===================================================================

// Explicit function instantiation — called once per unique class in each .cc file
#define ONNX_OPERATOR_TYPED_KERNEL_COMPUTE_INSTANTIATION(cls) template Status cls::ComputeInternal(OpKernelContext* context) const;

// The plugin utilizes ep::adapter::LoggingManager for LOGS_DEFAULT,
// which is initialized in CudaEpFactory::CudaEpFactory.

#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace onnxruntime {
namespace cuda {

// ===================================================================
// Section 5: Runtime configuration for migrated kernels
// Fields are written once during CudaEp construction and owned by the
// shim CUDAExecutionProvider. Plugin kernels cache a shared_ptr to this
// config object during construction so Compute() does not need to rely on
// raw provider-pointer casts.
// ===================================================================

namespace detail {
struct CudaKernelAdapterRuntimeConfig {
  bool use_tf32 = true;
  bool skip_layer_norm_strict_mode = false;
  int cudnn_conv_algo = 0;
  bool cudnn_conv_use_max_workspace = true;
  bool cudnn_conv1d_pad_to_nc1d = false;
  bool fuse_conv_bias = false;
  int sdpa_kernel = 0;
  int device_id = 0;
  cudaDeviceProp device_prop{};
  onnxruntime::AttentionKernelOptions attention_kernel_options;
};
template <typename T>
struct SizeOf {
  static constexpr size_t value = sizeof(T);
};
template <>
struct SizeOf<void> {
  static constexpr size_t value = 0;
};

[[nodiscard]] inline bool TryBytesForCount(size_t count_or_bytes, size_t element_size, size_t& bytes) {
  if (element_size == 0) {
    // `element_size == 0` is the sentinel for the `T = void` path.
    // In that mode callers already pass a raw byte count to helpers like
    // GetScratchBuffer<void>(workspace_bytes, ...), so no multiplication is needed.
    bytes = count_or_bytes;
    return true;
  }

  if (count_or_bytes > (std::numeric_limits<size_t>::max() / element_size)) {
    return false;
  }

  bytes = count_or_bytes * element_size;
  return true;
}

template <typename T>
IConstantBuffer<T>* GetConstOnesBufferForDevice(int device_id) {
  static std::mutex mutex;
  static std::unordered_map<int, std::unique_ptr<IConstantBuffer<T>>> buffers;
  std::lock_guard<std::mutex> lock(mutex);
  auto& buffer = buffers[device_id];
  if (!buffer) {
    buffer = CreateConstantOnes<T>();
  }
  return buffer.get();
}

struct DefaultCudaHandles {
  cublasHandle_t cublas = nullptr;
  cudnnHandle_t cudnn = nullptr;
  cublasLtHandle_t cublas_lt = nullptr;

  ~DefaultCudaHandles() {
    if (cublas != nullptr) {
      cublasDestroy(cublas);
    }
    if (cudnn != nullptr) {
      cudnnDestroy(cudnn);
    }
    if (cublas_lt != nullptr) {
      cublasLtDestroy(cublas_lt);
    }
  }
};

inline DefaultCudaHandles& GetDefaultCudaHandlesForDevice(int device_id) {
  // Fallback handles are only used for code paths that need cuBLAS/cuDNN
  // without an active CudaSyncStream. Keep them thread-local so they are not
  // shared across callers that may use the libraries concurrently.
  thread_local std::unordered_map<int, DefaultCudaHandles> handles_by_device;
  auto [it, inserted] = handles_by_device.try_emplace(device_id);
  if (inserted) {
    int prev_device = -1;
    const cudaError_t get_device_result = cudaGetDevice(&prev_device);
    PL_CUDA_CALL_THROW(cudaSetDevice(device_id));
    if (cublasCreate(&it->second.cublas) != CUBLAS_STATUS_SUCCESS) {
      if (get_device_result == cudaSuccess) {
        cudaSetDevice(prev_device);
      }
      handles_by_device.erase(it);
      ORT_THROW("Failed to create default cuBLAS handle for CUDA plugin device ", device_id);
    }
    if (cudnnCreate(&it->second.cudnn) != CUDNN_STATUS_SUCCESS) {
      cublasDestroy(it->second.cublas);
      it->second.cublas = nullptr;
      if (get_device_result == cudaSuccess) {
        cudaSetDevice(prev_device);
      }
      handles_by_device.erase(it);
      ORT_THROW("Failed to create default cuDNN handle for CUDA plugin device ", device_id);
    }
    if (cublasLtCreate(&it->second.cublas_lt) != CUBLAS_STATUS_SUCCESS) {
      cudnnDestroy(it->second.cudnn);
      it->second.cudnn = nullptr;
      cublasDestroy(it->second.cublas);
      it->second.cublas = nullptr;
      if (get_device_result == cudaSuccess) {
        cudaSetDevice(prev_device);
      }
      handles_by_device.erase(it);
      ORT_THROW("Failed to create default cuBLASLt handle for CUDA plugin device ", device_id);
    }
    if (get_device_result == cudaSuccess) {
      PL_CUDA_CALL_THROW(cudaSetDevice(prev_device));
    }
  }

  return it->second;
}

inline const cudaDeviceProp& GetDevicePropForDevice(int device_id) {
  static std::mutex mutex;
  static std::unordered_map<int, std::unique_ptr<cudaDeviceProp>> props;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = props.find(device_id);
  if (it == props.end()) {
    auto prop = std::make_unique<cudaDeviceProp>();
    const cudaError_t result = cudaGetDeviceProperties(prop.get(), device_id);
    if (result != cudaSuccess) {
      ORT_THROW("Failed to query CUDA device properties for device ", device_id, ": ", cudaGetErrorString(result));
    }
    it = props.emplace(device_id, std::move(prop)).first;
  }
  return *it->second;
}
}  // namespace detail
}  // namespace cuda

// ===================================================================
// Section 6: CUDAExecutionProvider shim
// Provides the minimal API surface that migrated kernels expect
// (GetCudnnConvAlgo, UseTF32, GetDeviceProp, etc.) without the full
// CUDAExecutionProvider class from onnxruntime/core/providers/cuda/.
//
// In the plugin build this shim is wrapped by adapter::Ep. Plugin kernels
// should prefer the CudaKernel base-class accessors for runtime settings
// instead of re-casting info.GetExecutionProvider() inside Compute().
// ===================================================================

// Shim for CUDAExecutionProvider required by conv.cc, einsum, and others
class CUDAExecutionProvider : public onnxruntime::IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const std::string& name, const OrtEp* ort_ep = nullptr)
      : onnxruntime::IExecutionProvider{name}, ort_ep_{ort_ep} {}

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override {
    return std::make_unique<onnxruntime::GPUDataTransfer>();
  }

  const OrtEp* GetOrtEp() const override {
    return ort_ep_;
  }

  std::shared_ptr<cuda::detail::CudaKernelAdapterRuntimeConfig> GetRuntimeConfig() const {
    return config_;
  }

  int GetCudnnConvAlgo() const {
    return config_->cudnn_conv_algo;
  }
  bool GetCudnnConvUseMaxWorkspace() const {
    return config_->cudnn_conv_use_max_workspace;
  }
  bool GetCudnnConv1dPadToNc1d() const {
    return config_->cudnn_conv1d_pad_to_nc1d;
  }
  bool UseTF32() const {
    return config_->use_tf32;
  }
  bool IsFuseConvBias() const {
    return config_->fuse_conv_bias;
  }
  const onnxruntime::AttentionKernelOptions* GetAttentionKernelOptions() const {
    config_->attention_kernel_options.InitializeOnce(config_->sdpa_kernel, true, true);
    return &config_->attention_kernel_options;
  }
  const cudaDeviceProp& GetDeviceProp() const {
    return config_->device_prop;
  }

 private:
  const OrtEp* ort_ep_ = nullptr;
  std::shared_ptr<cuda::detail::CudaKernelAdapterRuntimeConfig> config_ =
      std::make_shared<cuda::detail::CudaKernelAdapterRuntimeConfig>();
};

namespace cuda {
namespace detail {

inline std::shared_ptr<CudaKernelAdapterRuntimeConfig> GetCudaKernelAdapterRuntimeConfigForProvider(const void* provider) {
  return static_cast<const CUDAExecutionProvider*>(provider)->GetRuntimeConfig();
}

}  // namespace detail

// Populate the per-provider adapter config from a pre-filled initializer struct.
// Callers (e.g. CudaEp constructor) construct a detail::CudaKernelAdapterRuntimeConfig,
// fill every field they care about, then call this function. Adding a new config
// field only requires updating the struct and the call site — no signature change.
inline void SetCudaKernelAdapterRuntimeConfigForProvider(
    const void* provider, const detail::CudaKernelAdapterRuntimeConfig& init_config) {
  auto config = detail::GetCudaKernelAdapterRuntimeConfigForProvider(provider);
  // AttentionKernelOptions contains std::once_flag (not copyable), so assign
  // the plain-data fields individually rather than relying on operator=.
  config->use_tf32 = init_config.use_tf32;
  config->skip_layer_norm_strict_mode = init_config.skip_layer_norm_strict_mode;
  config->cudnn_conv_algo = init_config.cudnn_conv_algo;
  config->cudnn_conv_use_max_workspace = init_config.cudnn_conv_use_max_workspace;
  config->cudnn_conv1d_pad_to_nc1d = init_config.cudnn_conv1d_pad_to_nc1d;
  config->fuse_conv_bias = init_config.fuse_conv_bias;
  config->sdpa_kernel = init_config.sdpa_kernel;
  config->device_id = init_config.device_id;
  PL_CUDA_CALL_THROW(cudaGetDeviceProperties(&config->device_prop, config->device_id));
}

inline bool GetCudaKernelAdapterSkipLayerNormStrictMode(const void* provider) {
  const auto config = detail::GetCudaKernelAdapterRuntimeConfigForProvider(provider);
  return config->skip_layer_norm_strict_mode;
}

// Global aliases and shims
using Status = onnxruntime::common::Status;
using MLFloat16 = onnxruntime::MLFloat16;
using BFloat16 = onnxruntime::BFloat16;
using Float8E4M3FN = onnxruntime::Float8E4M3FN;
using Float8E4M3FNUZ = onnxruntime::Float8E4M3FNUZ;
using Float8E5M2 = onnxruntime::Float8E5M2;
using Float8E5M2FNUZ = onnxruntime::Float8E5M2FNUZ;

// Type mapping for CUDA
template <typename T>
struct ToCudaType {
  typedef T MappedType;
  static MappedType FromFloat(float f) { return static_cast<MappedType>(f); }
};

template <>
struct ToCudaType<MLFloat16> {
  typedef half MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = onnxruntime::math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

#ifdef __CUDACC__
template <>
struct ToCudaType<BFloat16> {
  typedef nv_bfloat16 MappedType;
  static MappedType FromFloat(float f) {
    return nv_bfloat16(f);
  }
};

// Forward declare templates from common.cuh to allow specialization
// Match signatures from common.cuh exactly (no default parameters)
template <typename T, bool detect_positive, bool detect_negative>
struct _IsInf;
template <typename T>
struct _IsNan;

namespace bf16_isinf_nan {
template <typename T>
struct IsInfTyped;
template <>
struct IsInfTyped<nv_bfloat16> {
  static __device__ __inline__ bool IsInf(nv_bfloat16 a) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(&a);
    return (val & 0x7F80) == 0x7F80 && (val & 0x007F) == 0x0000;
  }
  static __device__ __inline__ bool IsInfPos(nv_bfloat16 a) {
    return *reinterpret_cast<const uint16_t*>(&a) == 0x7F80;
  }
  static __device__ __inline__ bool IsInfNeg(nv_bfloat16 a) {
    return *reinterpret_cast<const uint16_t*>(&a) == 0xFF80;
  }
};
}  // namespace bf16_isinf_nan

// Specialize for nv_bfloat16 to avoid ambiguity with isnan/isinf overloads
template <>
struct _IsNan<nv_bfloat16> {
  __device__ __inline__ bool operator()(nv_bfloat16 a) const {
    uint16_t val = *reinterpret_cast<const uint16_t*>(&a);
    return (val & 0x7F80) == 0x7F80 && (val & 0x007F) != 0x0000;
  }
};

template <bool detect_positive, bool detect_negative>
struct _IsInf<nv_bfloat16, detect_positive, detect_negative> {
  __device__ __inline__ bool operator()(nv_bfloat16 a) const {
    if constexpr (detect_positive && detect_negative) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInf(a);
    } else if constexpr (detect_positive) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInfPos(a);
    } else if constexpr (detect_negative) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInfNeg(a);
    } else {
      return false;
    }
  }
};
#endif

// ===================================================================
// Section 6b: CPU provider shims for the plugin build
// Inline implementations of CPU utility functions that CUDA kernels
// reference (e.g., OneHot validation, GatherElements shape checks).
// These are normally provided by onnxruntime_providers but the plugin
// does not link against it.
//
// We temporarily close namespace cuda so these shims live directly in
// namespace onnxruntime, where unqualified lookup from onnxruntime::cuda
// will find them, and where onnxruntime::GatherElements resolves correctly.
// ===================================================================

}  // namespace cuda

// Shim for ValidateInputs from core/providers/cpu/tensor/onehot.h
inline Status ValidateInputs(const Tensor* depth, const Tensor* values) {
  if (!depth->Shape().IsScalar()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid argument for depth; it's not a scalar.");
  }
  if (!(values->Shape().NumDimensions() == 1 && values->Shape().Size() == 2)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid argument for values; either it's rank is more than 1"
                  " or it has more than 2 elements");
  }
  return Status::OK();
}

// Shim for PrepareOutputShape from core/providers/cpu/tensor/onehot.h
inline Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis,
                                 int64_t& prefix_dim_size, int64_t& suffix_dim_size,
                                 TensorShapeVector& output_shape) {
  const auto& indices_shape = indices->Shape();
  const auto indices_dims = indices_shape.GetDims();
  const auto indices_num_dims = indices_shape.NumDimensions();
  output_shape = indices_shape.AsShapeVector();
  const auto output_rank = static_cast<int64_t>(indices_num_dims) + 1;
  auto true_axis = HandleNegativeAxis(axis, output_rank);
  output_shape.insert(output_shape.begin() + true_axis, depth_val);
  prefix_dim_size = 1;
  for (int64_t i = 0; i < true_axis; ++i) {
    prefix_dim_size *= indices_dims[narrow<size_t>(i)];
  }
  suffix_dim_size = indices_shape.Size() / prefix_dim_size;
  return Status::OK();
}

// Shim for GatherElements::ValidateInputShapes from
// core/providers/cpu/tensor/gather_elements.h
class GatherElements {
 public:
  static Status ValidateInputShapes(const TensorShape& input_data_shape,
                                    const TensorShape& indices_shape,
                                    int64_t axis) {
    int64_t input_data_rank = static_cast<int64_t>(input_data_shape.NumDimensions());
    int64_t indices_rank = static_cast<int64_t>(indices_shape.NumDimensions());
    if (input_data_rank < 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GatherElements op: Cannot operate on scalar input");
    if (input_data_rank != indices_rank)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'");
    for (int64_t i = 0; i < indices_rank; ++i) {
      if (i != axis) {
        if (indices_shape[narrow<size_t>(i)] < 0 ||
            indices_shape[narrow<size_t>(i)] > input_data_shape[narrow<size_t>(i)])
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "GatherElements op: 'indices' shape should have values within bounds of 'data' shape. "
                                 "Invalid value in indices shape is: ",
                                 indices_shape[narrow<size_t>(i)]);
      }
    }
    return Status::OK();
  }
};

namespace cuda {  // re-open onnxruntime::cuda

// ===================================================================
// Section 7: CudaKernel base class
// Base class for all migrated CUDA kernels. Provides scratch-buffer
// management, CUDA handle access (cuBLAS, cuDNN), device property
// queries, and the CudaAsyncBuffer helper for host→device transfers.
// ===================================================================

// Additional adapter logic for CudaKernel

class CudaKernel : public OpKernel {
 public:
  explicit CudaKernel(const OpKernelInfo& info) : OpKernel(info) {
    const auto* provider = info.GetExecutionProvider();
    runtime_config_ = detail::GetCudaKernelAdapterRuntimeConfigForProvider(provider);
    use_tf32_ = runtime_config_->use_tf32;
    device_id_ = runtime_config_->device_id;
    device_prop_ = runtime_config_->device_prop;
  }
  virtual ~CudaKernel() = default;
  Status Compute(OpKernelContext* ctx) const {
    // Ensure the correct CUDA device is active for this kernel.
    // Worker threads default to device 0; sessions on device > 0 need an
    // explicit cudaSetDevice. Skip during CUDA graph capture because
    // cudaSetDevice is not allowed on a capturing stream.
    if (!IsThreadCapturingCudaGraph()) {
      int current_device = -1;
      PL_CUDA_CALL_THROW(cudaGetDevice(&current_device));
      if (current_device != device_id_) {
        PL_CUDA_CALL_THROW(cudaSetDevice(device_id_));
      }
    }
    Status s = ComputeInternal(ctx);
    if (s.IsOK()) {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    return s;
  }
  virtual Status ComputeInternal(OpKernelContext* ctx) const = 0;

  inline cudaStream_t DefaultCudaStream() const { return Stream(static_cast<OpKernelContext*>(nullptr)); }
  inline cublasHandle_t DefaultCublasHandle() const { return detail::GetDefaultCudaHandlesForDevice(device_id_).cublas; }
  inline cudnnHandle_t DefaultCudnnHandle() const { return detail::GetDefaultCudaHandlesForDevice(device_id_).cudnn; }
  inline cublasLtHandle_t DefaultCublasLtHandle() const { return detail::GetDefaultCudaHandlesForDevice(device_id_).cublas_lt; }

  inline Status CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst, onnxruntime::Stream& stream) const {
    if (src.Shape().Size() == 0) return Status::OK();
    if (cudaMemcpyAsync(dst.MutableDataRaw(), src.DataRaw(), src.SizeInBytes(), cudaMemcpyDeviceToDevice, (cudaStream_t)stream.GetHandle()) != cudaSuccess) {
      return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
    }
    return Status::OK();
  }

  cudaStream_t Stream(OpKernelContext* ctx) const {
    if (!ctx) return nullptr;
    return static_cast<cudaStream_t>(ctx->GetGPUComputeStream());
  }

  // Returns an opaque stream pointer for passing to GetScratchBuffer/AddDeferredReleaseCPUPtr/CopyToGpu.
  // Returns void* for dual-build compatibility: framework wraps Stream*, plugin wraps cudaStream_t.
  inline void* GetComputeStream(OpKernelContext* ctx) const {
    return ctx->GetGPUComputeStream();
  }

  inline onnxruntime::OrtStreamAdapter GetOrtStream(OpKernelContext* ctx) const {
    return onnxruntime::OrtStreamAdapter(GetComputeStream(ctx));
  }

  static cudnnHandle_t GetCudnnHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCudnnHandle() : nullptr;
  }
  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::CudaStream* stream) {
    return stream ? GetCudnnHandle(static_cast<cudaStream_t>(reinterpret_cast<onnxruntime::Stream*>(stream)->GetHandle())) : nullptr;
  }
  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::Stream* stream) {
    return stream ? GetCudnnHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cudnnHandle_t GetCudnnHandle(OpKernelContext* ctx) const {
    auto stream = Stream(ctx);
    auto handle = GetCudnnHandle(stream);
    if (handle != nullptr) {
      return handle;
    }

    handle = DefaultCudnnHandle();
    if (stream != nullptr) {
      CUDNN_CALL_THROW(cudnnSetStream(handle, stream));
    }
    return handle;
  }

  static cublasHandle_t GetCublasHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCublasHandle() : nullptr;
  }
  static inline cublasHandle_t GetCublasHandle(onnxruntime::CudaStream* stream) {
    return stream ? GetCublasHandle(static_cast<cudaStream_t>(reinterpret_cast<onnxruntime::Stream*>(stream)->GetHandle())) : nullptr;
  }
  static inline cublasHandle_t GetCublasHandle(onnxruntime::Stream* stream) {
    return stream ? GetCublasHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cublasHandle_t GetCublasHandle(OpKernelContext* ctx) const {
    auto stream = Stream(ctx);
    auto handle = GetCublasHandle(stream);
    if (handle != nullptr) {
      return handle;
    }

    handle = DefaultCublasHandle();
    if (stream != nullptr) {
      CUBLAS_CALL_THROW(cublasSetStream(handle, stream));
    }
    return handle;
  }

  static cublasLtHandle_t GetCublasLtHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCublasLtHandle() : nullptr;
  }
  static inline cublasLtHandle_t GetCublasLtHandle(onnxruntime::CudaStream* stream) {
    return stream ? GetCublasLtHandle(static_cast<cudaStream_t>(reinterpret_cast<onnxruntime::Stream*>(stream)->GetHandle())) : nullptr;
  }
  static inline cublasLtHandle_t GetCublasLtHandle(onnxruntime::Stream* stream) {
    return stream ? GetCublasLtHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cublasLtHandle_t GetCublasLtHandle(OpKernelContext* ctx) const {
    auto handle = GetCublasLtHandle(Stream(ctx));
    return handle != nullptr ? handle : DefaultCublasLtHandle();
  }

  const cudaDeviceProp& GetDeviceProp() const {
    // Some migrated kernels size their launches from device properties. If the
    // per-provider cache was not populated for this kernel instance, fall back
    // to a direct lookup instead of returning an all-zero struct.
    if (device_prop_.maxThreadsPerMultiProcessor == 0 || device_prop_.multiProcessorCount == 0) {
      return detail::GetDevicePropForDevice(device_id_);
    }

    return device_prop_;
  }
  int GetCudnnConvAlgo() const { return runtime_config_->cudnn_conv_algo; }
  bool GetCudnnConvUseMaxWorkspace() const { return runtime_config_->cudnn_conv_use_max_workspace; }
  bool GetCudnnConv1dPadToNc1d() const { return runtime_config_->cudnn_conv1d_pad_to_nc1d; }
  bool UseTF32() const { return use_tf32_; }
  bool IsFuseConvBias() const { return runtime_config_->fuse_conv_bias; }
  bool IsArchAvailable(int arch) const { return GetDeviceProp().major >= arch; }
  // Delegate to the base OpKernel::Info() which holds a safe copy of OpKernelInfo.
  // Do NOT store a reference to the constructor parameter — it becomes dangling.
  const OpKernelInfo& Info() const { return OpKernel::Info(); }
  const onnxruntime::AttentionKernelOptions* GetAttentionKernelOptions() const {
    runtime_config_->attention_kernel_options.InitializeOnce(runtime_config_->sdpa_kernel, true, true);
    return &runtime_config_->attention_kernel_options;
  }

  // Stub for GetTuningContext — tunable ops are not supported in the plugin.
  struct PluginTuningContextStub {
    bool IsTunableOpEnabled() const { return false; }
  };
  PluginTuningContextStub* GetTuningContext() const {
    static PluginTuningContextStub stub;
    return &stub;
  }

  // GetConstOnes: returns a device buffer of constant ones.
  // Delegates to IConstantBuffer from cuda_utils.h (compiled in cuda_utils.cu).
  template <typename T>
  const T* GetConstOnes(size_t count, cudaStream_t stream) const {
    auto* buf = detail::GetConstOnesBufferForDevice<T>(device_id_);
    return buf->GetBuffer(stream, count);
  }

  template <typename T>
  using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;
  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t cnt, void* s) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    size_t sz = 0;
    if (!detail::TryBytesForCount(cnt, detail::SizeOf<T>::value, sz)) {
      ORT_THROW("CUDA scratch buffer allocation size overflow for ", cnt, " elements");
    }

    void* p = nullptr;
    cudaError_t alloc_result = cudaSuccess;
    bool used_async_alloc = false;
    if (s) {
      // Note: stream-ordered allocations (cudaMallocAsync/cudaFreeAsync) rely on CUDA Memory Pools,
      // which are not supported on NVIDIA GPUs with Multi-Instance GPU (MIG) enabled.
      // On such instances, this will return cudaErrorNotSupported.
      alloc_result = cudaMallocAsync(&p, sz, static_cast<cudaStream_t>(s));
      used_async_alloc = (alloc_result == cudaSuccess);
      if (!used_async_alloc && (alloc_result == cudaErrorNotSupported || alloc_result == cudaErrorInvalidValue)) {
        cudaGetLastError();  // Clear the thread-local error state
        alloc_result = cudaMalloc(&p, sz);
      }
    } else {
      alloc_result = cudaMalloc(&p, sz);
    }

    if (alloc_result != cudaSuccess) {
      ORT_THROW("CUDA scratch buffer allocation failed for ", sz, " bytes: ", cudaGetErrorString(alloc_result));
    }

    return IAllocatorUniquePtr<T>(static_cast<T*>(p), [s, used_async_alloc](T* ptr) {
      if (ptr) {
        // Guard: only attempt async free if the stream is still registered.
        // CudaSyncStream::~CudaSyncStream guarantees UnregisterStream() is
        // called before cudaStreamDestroy(), so a non-null lookup here means
        // the raw cudaStream_t handle is still valid.
        if (used_async_alloc && s &&
            cuda_plugin::CudaSyncStream::FromCudaStream(static_cast<cudaStream_t>(s)) != nullptr) {
          // As noted above, cudaFreeAsync may also return cudaErrorNotSupported on MIG-enabled instances.
          cudaError_t free_result = cudaFreeAsync(ptr, static_cast<cudaStream_t>(s));
          if (free_result == cudaSuccess) {
            return;
          }
          cudaGetLastError();  // Clear any error set by cudaFreeAsync
        }

        // Fall back to synchronous free if async free is unsupported or if the
        // stream is no longer registered. cudaFree is valid for allocations
        // returned by cudaMallocAsync and avoids using a stale stream handle.
        cudaFree(ptr);
      }
    });
  }
  template <typename T>
  inline IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t cnt) const {
    return GetScratchBuffer<T>(cnt, nullptr);
  }
  inline void AddDeferredReleaseCPUPtr(void* p, void* s) const {
    if (!p) return;
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(static_cast<cudaStream_t>(s));
    if (sync) {
      sync->EnqueueDeferredCPUBuffer(p);
      return;
    }

    if (s != nullptr) {
      cudaError_t sync_result = cudaStreamSynchronize(static_cast<cudaStream_t>(s));
      if (sync_result != cudaSuccess) {
        // If the raw stream handle is already invalid during teardown, prefer a
        // bounded leak over freeing pinned memory that could still be in use by
        // an in-flight async copy.
        LOGS_DEFAULT(WARNING) << "AddDeferredReleaseCPUPtr: cudaStreamSynchronize failed ("
                              << cudaGetErrorString(sync_result)
                              << "); leaking pinned buffer to avoid use-after-free";
        return;
      }
    }

    cudaFreeHost(p);
  }
  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t cnt) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    size_t sz = 0;
    if (!detail::TryBytesForCount(cnt, detail::SizeOf<T>::value, sz)) {
      ORT_THROW("CUDA pinned CPU buffer allocation size overflow for ", cnt, " elements");
    }
    void* p = nullptr;
    if (cudaHostAlloc(&p, sz, cudaHostAllocDefault) != cudaSuccess) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    return IAllocatorUniquePtr<T>(static_cast<T*>(p), [](T* ptr) { if (ptr) cudaFreeHost(ptr); });
  }

  template <typename T>
  class CudaAsyncBuffer {
   public:
    CudaAsyncBuffer(const CudaKernel* ok) : gpu_(nullptr, [](T*) {}), count_(0), op_kernel_(ok) {}
    CudaAsyncBuffer(const CudaKernel* ok, size_t n) : CudaAsyncBuffer(ok) { AllocCpuPtr(n); }
    CudaAsyncBuffer(const CudaKernel* ok, const T& v, size_t n) : CudaAsyncBuffer(ok, n) {
      T* p = CpuPtr();
      for (size_t i = 0; i != n; ++i) *p++ = v;
    }
    CudaAsyncBuffer(const CudaKernel* ok, gsl::span<T const> vec) : CudaAsyncBuffer(ok, vec.size()) {
      size_t bytes = 0;
      if (!detail::TryBytesForCount(vec.size(), sizeof(T), bytes)) {
        ORT_THROW("CUDA async buffer host copy size overflow for ", vec.size(), " elements");
      }
      memcpy(CpuPtr(), vec.data(), bytes);
    }
    void AllocCpuPtr(size_t n) {
      cpu_ = op_kernel_->AllocateBufferOnCPUPinned<T>(n);
      if (!cpu_) throw std::runtime_error("alloc fail");
      count_ = n;
    }
    Status CopyToGpu(void* s) {
      if (cpu_) {
        gpu_ = op_kernel_->GetScratchBuffer<T>(count_, s);
        size_t bytes = 0;
        if (!detail::TryBytesForCount(count_, sizeof(T), bytes)) {
          ORT_THROW("CUDA async buffer copy size overflow for ", count_, " elements");
        }
        if (cudaMemcpyAsync(gpu_.get(), cpu_.get(), bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(s)) != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
        op_kernel_->AddDeferredReleaseCPUPtr(cpu_.release(), s);
      }
      return Status::OK();
    }
    T* CpuPtr() const { return cpu_.get(); }
    gsl::span<T> CpuSpan() const { return gsl::span<T>(CpuPtr(), count_); }
    T* GpuPtr() const { return gpu_.get(); }
    size_t count() const { return count_; }

   protected:
    IAllocatorUniquePtr<T> gpu_;
    std::unique_ptr<T, std::function<void(T*)>> cpu_{nullptr, [](T*) {}};
    size_t count_;
    const CudaKernel* op_kernel_;
  };

 private:
  std::shared_ptr<detail::CudaKernelAdapterRuntimeConfig> runtime_config_;
  cudaDeviceProp device_prop_{};
  bool use_tf32_ = true;
  int device_id_ = 0;
};

// ===================================================================
// Section 8: Compute helper shims (HalfGemmOptions, CublasMathModeSetter)
// ===================================================================

// Shims for HalfGemmOptions and CublasMathModeSetter required by fpgeneric.h
class HalfGemmOptions {
 public:
  static const HalfGemmOptions* GetInstance() {
    static HalfGemmOptions instance;
    return &instance;
  }
  cublasMath_t GetMathMode() const { return CUBLAS_DEFAULT_MATH; }
  bool IsCompute16F() const { return false; }
#if defined(CUBLAS_COMPUTE_32F)
  cublasComputeType_t GetComputeType() const { return CUBLAS_COMPUTE_32F; }
#else
  cudaDataType_t GetComputeType() const { return CUDA_R_32F; }
#endif
};

class CublasMathModeSetter {
 public:
  CublasMathModeSetter(const cudaDeviceProp& prop, cublasHandle_t handle, cublasMath_t mode) : handle_(handle) {
    enable_ = (mode == CUBLAS_TF32_TENSOR_OP_MATH ? prop.major >= 8 : true);
    if (enable_) {
      cublasGetMathMode(handle, &mode_);
      enable_ = (mode_ != mode);
      if (enable_) {
        cublasSetMathMode(handle, mode);
      }
    }
  }

  ~CublasMathModeSetter() {
    if (enable_) {
      cublasSetMathMode(handle_, mode_);
    }
  }

 private:
  cublasHandle_t handle_;
  cublasMath_t mode_ = CUBLAS_DEFAULT_MATH;
  bool enable_;
};

}  // namespace cuda

// Global aliases for convenience
using MLFloat16 = onnxruntime::MLFloat16;
using BFloat16 = onnxruntime::BFloat16;
using Float8E4M3FN = onnxruntime::Float8E4M3FN;
using Float8E4M3FNUZ = onnxruntime::Float8E4M3FNUZ;
using Float8E5M2 = onnxruntime::Float8E5M2;
using Float8E5M2FNUZ = onnxruntime::Float8E5M2FNUZ;

}  // namespace onnxruntime
