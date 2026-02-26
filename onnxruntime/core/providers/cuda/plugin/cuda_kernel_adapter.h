// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// cuda_kernel_adapter.h — Compatibility shim for migrating CUDA kernels to the
// plugin EP architecture.
//
// This header supports three compilation modes:
//
//   1. ORT_CUDA_PLUGIN_USE_ADAPTER: kernels are compiled against the real
//      ORT framework types (OpKernel, OpKernelContext, etc.) and use the
//      adapter-path registration macros that feed into PluginRegistry.
//
//   2. BUILD_CUDA_EP_AS_PLUGIN (without ORT_CUDA_PLUGIN_USE_ADAPTER): the
//      legacy plugin path where kernels use provider_api.h types.  Kernel
//      registration macros are no-opped because registration happens via
//      generated .inc tables.
//
//   3. Neither defined: the standard in-tree build.
//
// In modes 1 and 2 this header also provides:
//   - CudaKernel base class (scratch buffers, CUDA handles, etc.)
//   - Error-return macros (CUDA_RETURN_IF_ERROR, etc.)
//   - Type mapping helpers (ToCudaType)
//   - Math/compute shims (HalfGemmOptions, CublasMathModeSetter)

#pragma once

#include "core/common/status.h"
#include "core/common/narrow.h"
#include "core/common/float16.h"
#include "core/common/float8.h"
#include "core/framework/float4.h"
#include "core/framework/allocator.h"
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
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

// Forward-declare CudaStream so adapter overloads accepting CudaStream* compile
// without pulling in cuda_stream_handle.h (which depends on CUDAExecutionProvider).
namespace onnxruntime {
struct CudaStream;
}  // namespace onnxruntime

// ===================================================================
// Section 1: Include path selection
// ===================================================================

#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
// Legacy path: Define SHARED_PROVIDER before including provider_api.h.
// When adapters.h is active, provider_api.h becomes a no-op (via its own guard),
// and we skip this entire block.
#ifndef SHARED_PROVIDER
#define SHARED_PROVIDER 1
#endif

// Include provider_api.h FIRST. This provides the "plugin" versions of ORT types.
#include "core/providers/shared_library/provider_api.h"

// Shadowing to avoid conflicts if core headers are indirectly included
#define kOnnxDomain __kOnnxDomain_ignore
#define kMSDomain __kMSDomain_ignore
#define kPytorchAtenDomain __kPytorchAtenDomain_ignore
#define kNGraphDomain __kNGraphDomain_ignore
#define kCudaExecutionProvider __kCudaExecutionProvider_ignore
#define kCpuExecutionProvider __kCpuExecutionProvider_ignore
#define kAzureExecutionProvider __kAzureExecutionProvider_ignore

// Include framework's cuda_common.h for math utilities and CUDA types.
// We avoid op_kernel.h as it brings in too many conflicting types.
#include "core/providers/cuda/cuda_common.h"

#undef kOnnxDomain
#undef kMSDomain
#undef kPytorchAtenDomain
#undef kNGraphDomain
#undef kCudaExecutionProvider
#undef kCpuExecutionProvider
#undef kAzureExecutionProvider

namespace onnxruntime {
namespace cuda {
using Tensor = ::onnxruntime::Tensor;
using ::onnxruntime::HandleNegativeAxis;
// Tensor creation helper to replace deprecated Tensor::Create
inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return ::onnxruntime::Tensor::Create(type, shape, std::move(allocator));
}
}  // namespace cuda
}  // namespace onnxruntime
#else  // ORT_CUDA_PLUGIN_USE_ADAPTER
// Adapter path: use real framework types but provide local templates and
// namespace aliases to allow registration macros to work in nested namespaces.
#include <map>
#include "core/graph/constants.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

// Tensor creation helper to replace deprecated Tensor::Create
inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

// Local template for registration macros to specialize.
// This allows specializing BuildKernelCreateInfo within onnxruntime::cuda.
template <typename T>
::onnxruntime::KernelCreateInfo BuildKernelCreateInfo();

// Aliases for types used by kernels and registration macros.
using FuncManager = ::onnxruntime::FuncManager;
using KernelCreatePtrFn = ::onnxruntime::KernelCreatePtrFn;
using OpKernel = ::onnxruntime::OpKernel;
using OpKernelContext = ::onnxruntime::OpKernelContext;
using OpKernelInfo = ::onnxruntime::OpKernelInfo;
using MLDataType = ::onnxruntime::MLDataType;
using DataTypeImpl = ::onnxruntime::DataTypeImpl;
using Status = ::onnxruntime::common::Status;
using KernelCreateInfo = ::onnxruntime::KernelCreateInfo;
using KernelDefBuilder = ::onnxruntime::KernelDefBuilder;
using Stream = ::onnxruntime::Stream;
using AllocatorPtr = ::onnxruntime::AllocatorPtr;
using TensorShape = ::onnxruntime::TensorShape;
using Tensor = ::onnxruntime::Tensor;
using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
namespace cuda {

inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

// Local template for registration macros to specialize.
template <typename T>
::onnxruntime::KernelCreateInfo BuildKernelCreateInfo();

using FuncManager = ::onnxruntime::FuncManager;
using KernelCreatePtrFn = ::onnxruntime::KernelCreatePtrFn;
using OpKernel = ::onnxruntime::OpKernel;
using OpKernelContext = ::onnxruntime::OpKernelContext;
using OpKernelInfo = ::onnxruntime::OpKernelInfo;
using MLDataType = ::onnxruntime::MLDataType;
using DataTypeImpl = ::onnxruntime::DataTypeImpl;
using Status = ::onnxruntime::common::Status;
using KernelCreateInfo = ::onnxruntime::KernelCreateInfo;
using KernelDefBuilder = ::onnxruntime::KernelDefBuilder;
using Stream = ::onnxruntime::Stream;
using AllocatorPtr = ::onnxruntime::AllocatorPtr;
using TensorShape = ::onnxruntime::TensorShape;
using Tensor = ::onnxruntime::Tensor;
using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda
}  // namespace contrib
#endif
}  // namespace onnxruntime

#endif  // !ORT_CUDA_PLUGIN_USE_ADAPTER

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

// ===================================================================
// Section 3: Adapter-path kernel registration and PluginRegistry
// ===================================================================

#ifdef ORT_CUDA_PLUGIN_USE_ADAPTER
namespace onnxruntime {
namespace cuda {

/// OrtKernelImpl wrapper that bridges an onnxruntime::OpKernel to the plugin EP
/// kernel interface. Used by the adapter-path registration macros.
struct AdapterKernelImpl : public OrtKernelImpl {
  std::unique_ptr<onnxruntime::OpKernel> kernel;

  explicit AdapterKernelImpl(std::unique_ptr<onnxruntime::OpKernel> k) : kernel(std::move(k)) {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
    auto* self = static_cast<AdapterKernelImpl*>(this_ptr);
    auto* adapter_ctx = reinterpret_cast<onnxruntime::OpKernelContext*>(context);
    Status status = self->kernel->Compute(adapter_ctx);
    if (!status.IsOK()) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, status.ErrorMessage().c_str());
    }
    return nullptr;
  }

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<AdapterKernelImpl*>(this_ptr);
  }
};

// KernelFactory: a plain function-pointer type for creating kernels.
// Non-capturing lambdas in macros decay to this type automatically.
// Storing KernelFactory values (not the function pointers themselves) in a
// std::map provides stable object addresses, which can safely roundtrip through
// void* (object pointer <-> void* is guaranteed valid in C++).
using KernelFactory = ::onnxruntime::OpKernel* (*)(const ::onnxruntime::OpKernelInfo&);

// GenericCreateKernel: ORT plugin registry callback.
// context is a const KernelFactory* (pointer to a KernelFactory stored in the
// PluginRegistry map). Casting an object pointer to/from void* is valid C++.
inline OrtStatus* ORT_API_CALL GenericCreateKernel(void* context, const OrtKernelInfo* info, OrtKernelImpl** out) noexcept {
  const auto* factory = static_cast<const KernelFactory*>(context);
  const auto& adapter_info = *reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  auto* kernel_raw = (*factory)(adapter_info);
  if (!kernel_raw) return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to create kernel");
  *out = new AdapterKernelImpl(std::unique_ptr<onnxruntime::OpKernel>(kernel_raw));
  return nullptr;
}

// PluginRegistry: singleton mapping (op, domain, since, end_version) to a KernelFactory.
//
// Design choices:
//   - std::map gives O(log n) lookup vs the previous O(n) linear scan.
//   - std::map values have stable addresses (iterators are not invalidated by
//     insertions), so &entry.second is a valid long-lived void* context for
//     GenericCreateKernel.
//   - Version range (since, end) is part of the key, so different opset
//     versions of the same op can map to different kernel implementations.
//   - end == 0 means "open-ended" (no maximum version), matching the convention
//     used by kAdapterRegistrations.
class PluginRegistry {
 public:
  struct EntryKey {
    std::string op;
    std::string domain;
    int since{0};
    int end{0};  // 0 = open-ended
    bool operator<(const EntryKey& o) const {
      return std::tie(op, domain, since, end) < std::tie(o.op, o.domain, o.since, o.end);
    }
  };

  static PluginRegistry& Instance() {
    static PluginRegistry instance;
    return instance;
  }

  // Registers a factory and returns a stable pointer to the stored value.
  // The returned pointer is usable as a GenericCreateKernel context.
  const KernelFactory* Add(const std::string& op, const char* domain, int since, int end_ver,
                           KernelFactory fn) {
    auto& slot = fns_[EntryKey{op, domain ? domain : "", since, end_ver}];
    slot = fn;
    return &slot;
  }

  const std::map<EntryKey, KernelFactory>& AllEntries() const { return fns_; }

 private:
  std::map<EntryKey, KernelFactory> fns_;
};
}  // namespace cuda
}  // namespace onnxruntime

// Local template for registration macros to specialize.
template <typename T>
::onnxruntime::KernelCreateInfo BuildKernelCreateInfo();

#define ORT_ADAPTER_CONCAT_IMPL(x, y) x##y
#define ORT_ADAPTER_CONCAT(x, y) ORT_ADAPTER_CONCAT_IMPL(x, y)

#undef ONNX_OPERATOR_KERNEL_EX
#define ONNX_OPERATOR_KERNEL_EX(op, domain, since, ep, builder, ...)                                                                        \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(ep, domain, since, op);                                                                             \
  template <>                                                                                                                               \
  ::onnxruntime::KernelCreateInfo BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(ep, domain, since, op)>() {                         \
    return ::onnxruntime::KernelCreateInfo(                                                                                                 \
        (builder).SetName(#op).SetDomain(domain).SinceVersion(since).Build(),                                                               \
        [](::onnxruntime::FuncManager&, const ::onnxruntime::OpKernelInfo& info, std::unique_ptr<::onnxruntime::OpKernel>& out) -> Status { \
          out = std::make_unique<__VA_ARGS__>(info);                                                                                        \
          return Status::OK();                                                                                                              \
        });                                                                                                                                 \
  }                                                                                                                                         \
  static bool ORT_ADAPTER_CONCAT(_reg_##op##_, __COUNTER__) = []() {                                                                        \
    ::onnxruntime::cuda::PluginRegistry::Instance().Add(                                                                                    \
        #op, domain, since, 0,                                                                                                              \
        [](const ::onnxruntime::OpKernelInfo& info) -> ::onnxruntime::OpKernel* { return new __VA_ARGS__(info); });                         \
    return true;                                                                                                                            \
  }();

#undef ONNX_OPERATOR_VERSIONED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(op, domain, since, end, ep, builder, ...)                                                         \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(ep, domain, since, end, op);                                                              \
  template <>                                                                                                                               \
  ::onnxruntime::KernelCreateInfo BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(ep, domain, since, end, op)>() {          \
    return ::onnxruntime::KernelCreateInfo(                                                                                                 \
        (builder).SetName(#op).SetDomain(domain).SinceVersion(since, end).Build(),                                                          \
        [](::onnxruntime::FuncManager&, const ::onnxruntime::OpKernelInfo& info, std::unique_ptr<::onnxruntime::OpKernel>& out) -> Status { \
          out = std::make_unique<__VA_ARGS__>(info);                                                                                        \
          return Status::OK();                                                                                                              \
        });                                                                                                                                 \
  }                                                                                                                                         \
  static bool ORT_ADAPTER_CONCAT(_reg_##op##_, __COUNTER__) = []() {                                                                        \
    ::onnxruntime::cuda::PluginRegistry::Instance().Add(                                                                                    \
        #op, domain, since, end,                                                                                                            \
        [](const ::onnxruntime::OpKernelInfo& info) -> ::onnxruntime::OpKernel* { return new __VA_ARGS__(info); });                         \
    return true;                                                                                                                            \
  }();

#undef ONNX_OPERATOR_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TYPED_KERNEL_EX(op, domain, since, type, ep, builder, ...)                                                            \
  class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(ep, domain, since, type, op);                                                                 \
  template <>                                                                                                                               \
  ::onnxruntime::KernelCreateInfo BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(ep, domain, since, type, op)>() {             \
    return ::onnxruntime::KernelCreateInfo(                                                                                                 \
        (builder).SetName(#op).SetDomain(domain).SinceVersion(since).Build(),                                                               \
        [](::onnxruntime::FuncManager&, const ::onnxruntime::OpKernelInfo& info, std::unique_ptr<::onnxruntime::OpKernel>& out) -> Status { \
          out = std::make_unique<__VA_ARGS__>(info);                                                                                        \
          return Status::OK();                                                                                                              \
        });                                                                                                                                 \
  }                                                                                                                                         \
  static bool ORT_ADAPTER_CONCAT(_reg_##op##_, __COUNTER__) = []() {                                                                        \
    ::onnxruntime::cuda::PluginRegistry::Instance().Add(                                                                                    \
        #op, domain, since, 0,                                                                                                              \
        [](const ::onnxruntime::OpKernelInfo& info) -> ::onnxruntime::OpKernel* { return new __VA_ARGS__(info); });                         \
    return true;                                                                                                                            \
  }();

#undef ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(op, domain, since, end, type, ep, builder, ...)                                                \
  class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(ep, domain, since, end, type, op);                                                     \
  template <>                                                                                                                                  \
  ::onnxruntime::KernelCreateInfo BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(ep, domain, since, end, type, op)>() { \
    return ::onnxruntime::KernelCreateInfo(                                                                                                    \
        (builder).SetName(#op).SetDomain(domain).SinceVersion(since, end).Build(),                                                             \
        [](::onnxruntime::FuncManager&, const ::onnxruntime::OpKernelInfo& info, std::unique_ptr<::onnxruntime::OpKernel>& out) -> Status {    \
          out = std::make_unique<__VA_ARGS__>(info);                                                                                           \
          return Status::OK();                                                                                                                 \
        });                                                                                                                                    \
  }                                                                                                                                            \
  static bool ORT_ADAPTER_CONCAT(_reg_##op##_, __COUNTER__) = []() {                                                                           \
    ::onnxruntime::cuda::PluginRegistry::Instance().Add(                                                                                       \
        #op, domain, since, end,                                                                                                               \
        [](const ::onnxruntime::OpKernelInfo& info) -> ::onnxruntime::OpKernel* { return new __VA_ARGS__(info); });                            \
    return true;                                                                                                                               \
  }();

// ===================================================================
// Section 3b: Legacy plugin path — no-op all registration macros
// (registration is driven by generated .inc files instead)
// ===================================================================

#elif defined(BUILD_CUDA_EP_AS_PLUGIN)

#undef ONNX_OPERATOR_KERNEL_EX
#define ONNX_OPERATOR_KERNEL_EX(...)
#undef ONNX_OPERATOR_VERSIONED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(...)
#undef ONNX_OPERATOR_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TYPED_KERNEL_EX(...)
#undef ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(...)
#undef ONNX_OPERATOR_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(...)
#undef ONNX_OPERATOR_THREE_TYPED_KERNEL_EX
#define ONNX_OPERATOR_THREE_TYPED_KERNEL_EX(...)
#undef ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(...)

#endif  // BUILD_CUDA_EP_AS_PLUGIN && !ORT_CUDA_PLUGIN_USE_ADAPTER

// ===================================================================
// Section 4: Logging shim (adapter path only)
// Replaces LOGS_DEFAULT with a no-op stream to avoid pulling in the
// full ORT logging framework inside the plugin shared library.
// ===================================================================

// Explicit function instantiation — called once per unique class in each .cc file
#define ONNX_OPERATOR_TYPED_KERNEL_COMPUTE_INSTANTIATION(cls) template Status cls::ComputeInternal(OpKernelContext* context) const;

#ifdef ORT_CUDA_PLUGIN_USE_ADAPTER

#undef CREATE_MESSAGE
#undef LOGS
#undef LOGS_DEFAULT
#undef ORT_LOG_MESSAGE

namespace onnxruntime {
namespace cuda {
struct PluginNoOpLogStream {
  template <typename T>
  PluginNoOpLogStream& operator<<(const T&) { return *this; }
};
}  // namespace cuda
}  // namespace onnxruntime

#ifndef LOGS_DEFAULT
#define LOGS_DEFAULT(severity) ::onnxruntime::cuda::PluginNoOpLogStream()
#endif

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace onnxruntime {
namespace cuda {

// ===================================================================
// Section 5: Runtime configuration for migrated kernels
// Stored as atomics so SetCudaKernelAdapterRuntimeConfig() can be
// called from CudaEp's constructor on any thread.
// ===================================================================

namespace detail {
struct CudaKernelAdapterRuntimeConfig {
  std::atomic<bool> use_tf32{true};
  std::atomic<bool> skip_layer_norm_strict_mode{false};
  std::atomic<int> device_id{0};
  std::atomic<int> cudnn_conv_algo{0};
  std::atomic<bool> cudnn_conv1d_pad_to_nc1d{false};
};
inline CudaKernelAdapterRuntimeConfig& GetCudaKernelAdapterRuntimeConfig() {
  static CudaKernelAdapterRuntimeConfig config;
  return config;
}
template <typename T>
struct SizeOf {
  static constexpr size_t value = sizeof(T);
};
template <>
struct SizeOf<void> {
  static constexpr size_t value = 0;
};
inline size_t BytesForCount(size_t count_or_bytes, size_t element_size) {
  if (element_size == 0) return count_or_bytes;
  if (count_or_bytes > (std::numeric_limits<size_t>::max() / element_size)) return 0;
  return count_or_bytes * element_size;
}
}  // namespace detail
}  // namespace cuda

// ===================================================================
// Section 6: CUDAExecutionProvider shim
// Provides the minimal API surface that migrated kernels expect
// (GetCudnnConvAlgo, UseTF32, GetDeviceProp, etc.) without the full
// CUDAExecutionProvider class from onnxruntime/core/providers/cuda/.
// ===================================================================

// Shim for CUDAExecutionProvider required by conv.cc, einsum, and others
class CUDAExecutionProvider : public onnxruntime::IExecutionProvider {
 public:
  explicit CUDAExecutionProvider(const std::string& name) : onnxruntime::IExecutionProvider{name} {}
  int GetCudnnConvAlgo() const {
    return cuda::detail::GetCudaKernelAdapterRuntimeConfig().cudnn_conv_algo.load(std::memory_order_relaxed);
  }
  bool GetCudnnConv1dPadToNc1d() const {
    return cuda::detail::GetCudaKernelAdapterRuntimeConfig().cudnn_conv1d_pad_to_nc1d.load(std::memory_order_relaxed);
  }
  bool UseTF32() const {
    return cuda::detail::GetCudaKernelAdapterRuntimeConfig().use_tf32.load(std::memory_order_relaxed);
  }
  bool IsFuseConvBias() const {
    return false;
  }
  const cudaDeviceProp& GetDeviceProp() const {
    static cudaDeviceProp prop;
    static std::once_flag flag;
    std::call_once(flag, []() {
      int device_id = cuda::detail::GetCudaKernelAdapterRuntimeConfig().device_id.load(std::memory_order_relaxed);
      if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        std::memset(&prop, 0, sizeof(prop));
        prop.major = -1;
      }
    });
    return prop;
  }
};

namespace cuda {

inline void SetCudaKernelAdapterRuntimeConfig(bool use_tf32, int device_id, bool skip_layer_norm_strict_mode = false,
                                              int cudnn_conv_algo = 0, bool cudnn_conv1d_pad_to_nc1d = false) {
  auto& config = detail::GetCudaKernelAdapterRuntimeConfig();
  config.use_tf32.store(use_tf32, std::memory_order_relaxed);
  config.skip_layer_norm_strict_mode.store(skip_layer_norm_strict_mode, std::memory_order_relaxed);
  config.device_id.store(device_id, std::memory_order_relaxed);
  config.cudnn_conv_algo.store(cudnn_conv_algo, std::memory_order_relaxed);
  config.cudnn_conv1d_pad_to_nc1d.store(cudnn_conv1d_pad_to_nc1d, std::memory_order_relaxed);
}

inline bool GetCudaKernelAdapterSkipLayerNormStrictMode() {
  const auto& config = detail::GetCudaKernelAdapterRuntimeConfig();
  return config.skip_layer_norm_strict_mode.load(std::memory_order_relaxed);
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

#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
// Legacy path: Shims for OpKernel-related types using provider_api.h's versions.
// When adapter is active, these are provided by the namespace aliases in adapters.h.
using Tensor = onnxruntime::Tensor;
using OpKernelContext = onnxruntime::OpKernelContext;
using OpKernelInfo = onnxruntime::OpKernelInfo;
using OpKernel = onnxruntime::OpKernel;

// Guard critical adapter patterns used by Stage 4 kernels.
static_assert(std::is_same_v<decltype(std::declval<OpKernelContext&>().Output(1, std::declval<const TensorShape&>())), Tensor*>,
              "OpKernelContext::Output(index, shape) must support arbitrary output indices.");
static_assert(std::is_same_v<decltype(std::declval<const OpKernelInfo&>().GetAttr<std::string>(std::declval<const std::string&>(), std::declval<std::string*>())),
                             Status>,
              "OpKernelInfo::GetAttr<std::string> must be available.");
static_assert(std::is_same_v<decltype(std::declval<const OpKernelInfo&>().GetAttrs<int64_t>(std::declval<const std::string&>(), std::declval<std::vector<int64_t>&>())),
                             Status>,
              "OpKernelInfo::GetAttrs<int64_t> must be available.");
#endif  // !ORT_CUDA_PLUGIN_USE_ADAPTER

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

#ifdef ORT_CUDA_PLUGIN_USE_ADAPTER

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

#endif  // ORT_CUDA_PLUGIN_USE_ADAPTER

namespace cuda {  // re-open onnxruntime::cuda

// ===================================================================
// Section 7: CudaKernel base class
// Base class for all migrated CUDA kernels. Provides scratch-buffer
// management, CUDA handle access (cuBLAS, cuDNN), device property
// queries, and the CudaAsyncBuffer helper for host→device transfers.
// ===================================================================

// Additional adapter logic for CudaKernel

class CudaKernel : public onnxruntime::OpKernel {
 public:
  explicit CudaKernel(const onnxruntime::OpKernelInfo& info) : onnxruntime::OpKernel(info), info_(info) {
    const auto& config = detail::GetCudaKernelAdapterRuntimeConfig();
    use_tf32_ = config.use_tf32.load(std::memory_order_relaxed);
    device_id_ = config.device_id.load(std::memory_order_relaxed);
    int cur = device_id_;
    if (cudaGetDevice(&cur) == cudaSuccess) device_id_ = cur;
    if (cudaGetDeviceProperties(&device_prop_, device_id_) != cudaSuccess) {
      std::memset(&device_prop_, 0, sizeof(device_prop_));
      device_prop_.major = -1;
    }
  }
  virtual ~CudaKernel() = default;
  Status Compute(onnxruntime::OpKernelContext* ctx) const {
    Status s = ComputeInternal(ctx);
    if (s.IsOK()) {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    return s;
  }
  virtual Status ComputeInternal(onnxruntime::OpKernelContext* ctx) const = 0;

  inline cudaStream_t DefaultCudaStream() const { return Stream(static_cast<onnxruntime::OpKernelContext*>(nullptr)); }
  inline cublasHandle_t DefaultCublasHandle() const { return GetCublasHandle(static_cast<cudaStream_t>(nullptr)); }
  inline cudnnHandle_t DefaultCudnnHandle() const { return GetCudnnHandle(static_cast<cudaStream_t>(nullptr)); }

  inline Status CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst, onnxruntime::Stream& stream) const {
    if (src.Shape().Size() == 0) return Status::OK();
    if (cudaMemcpyAsync(dst.MutableDataRaw(), src.DataRaw(), src.SizeInBytes(), cudaMemcpyDeviceToDevice, (cudaStream_t)stream.GetHandle()) != cudaSuccess) {
      return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
    }
    return Status::OK();
  }

  cudaStream_t Stream(onnxruntime::OpKernelContext* ctx) const {
    if (!ctx) return nullptr;
    // Map onnxruntime::OpKernelContext* (plugin version) to OrtKernelContext* and use Ort::KernelContext to get GPU stream.
    return static_cast<cudaStream_t>(Ort::KernelContext(reinterpret_cast<OrtKernelContext*>(ctx)).GetGPUComputeStream());
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
  cudnnHandle_t GetCudnnHandle(onnxruntime::OpKernelContext* ctx) const { return GetCudnnHandle(Stream(ctx)); }

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
  cublasHandle_t GetCublasHandle(onnxruntime::OpKernelContext* ctx) const { return GetCublasHandle(Stream(ctx)); }

  const cudaDeviceProp& GetDeviceProp() const { return device_prop_; }
  bool UseTF32() const { return use_tf32_; }
  bool IsArchAvailable(int arch) const { return device_prop_.major >= arch; }
  const onnxruntime::OpKernelInfo& Info() const { return info_; }
  const onnxruntime::AttentionKernelOptions* GetAttentionKernelOptions() const {
    static onnxruntime::AttentionKernelOptions options;
    return &options;
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
    static std::unique_ptr<IConstantBuffer<T>> buf;
    static std::once_flag flag;
    std::call_once(flag, []() { buf = CreateConstantOnes<T>(); });
    return buf->GetBuffer(stream, count);
  }

  template <typename T>
  using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;
  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t cnt, void* s) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    size_t sz = detail::BytesForCount(cnt, detail::SizeOf<T>::value);
    void* p = nullptr;
    if (cudaMalloc(&p, sz) != cudaSuccess) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    return IAllocatorUniquePtr<T>(static_cast<T*>(p), [s](T* ptr) {
      if (ptr) {
        if (s) {
          cudaFreeAsync(ptr, static_cast<cudaStream_t>(s));
        } else {
          cudaFree(ptr);
        }
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
    cudaFreeHost(p);
  }
  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t cnt) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    size_t sz = detail::BytesForCount(cnt, detail::SizeOf<T>::value);
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
    CudaAsyncBuffer(const CudaKernel* ok, gsl::span<T const> vec) : CudaAsyncBuffer(ok, vec.size()) { memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T)); }
    void AllocCpuPtr(size_t n) {
      cpu_ = op_kernel_->AllocateBufferOnCPUPinned<T>(n);
      if (!cpu_) throw std::runtime_error("alloc fail");
      count_ = n;
    }
    Status CopyToGpu(void* s) {
      if (cpu_) {
        gpu_ = op_kernel_->GetScratchBuffer<T>(count_, s);
        if (cudaMemcpyAsync(gpu_.get(), cpu_.get(), count_ * sizeof(T), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(s)) != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
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
  const onnxruntime::OpKernelInfo& info_;
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

#endif  // ORT_CUDA_PLUGIN_USE_ADAPTER
