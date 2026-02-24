// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
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

// Macros will be defined later to override core definitions.

#include "core/providers/cuda/plugin/cuda_stream_plugin.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

// Define SHARED_PROVIDER before including provider_api.h
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

// Disable static kernel registration logic when compiling kernels for plugin.
// Typed registration macros can optionally be repurposed to explicit template
// instantiation for selected translation units to avoid per-file plugin
// instantiation boilerplate.
#undef ONNX_OPERATOR_KERNEL_EX
#define ONNX_OPERATOR_KERNEL_EX(...)
#undef ONNX_OPERATOR_VERSIONED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(...)

#if defined(ORT_CUDA_PLUGIN_INSTANTIATE_TYPED_KERNEL_FROM_REGISTRATION)
#undef ONNX_OPERATOR_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...) template class __VA_ARGS__;
#undef ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, ...) template class __VA_ARGS__;
#undef ONNX_OPERATOR_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(name, domain, ver, type1, type2, provider, builder, ...) template class __VA_ARGS__;
#undef ONNX_OPERATOR_THREE_TYPED_KERNEL_EX
#define ONNX_OPERATOR_THREE_TYPED_KERNEL_EX(name, domain, ver, type1, type2, type3, provider, builder, ...) template class __VA_ARGS__;
#undef ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(name, domain, startver, endver, type1, type2, provider, builder, ...) template class __VA_ARGS__;
#else
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
#endif

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

namespace detail {
struct CudaKernelAdapterRuntimeConfig {
  std::atomic<bool> use_tf32{true};
  std::atomic<bool> skip_layer_norm_strict_mode{false};
  std::atomic<int> device_id{0};
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

inline void SetCudaKernelAdapterRuntimeConfig(bool use_tf32, int device_id, bool skip_layer_norm_strict_mode = false) {
  auto& config = detail::GetCudaKernelAdapterRuntimeConfig();
  config.use_tf32.store(use_tf32, std::memory_order_relaxed);
  config.skip_layer_norm_strict_mode.store(skip_layer_norm_strict_mode, std::memory_order_relaxed);
  config.device_id.store(device_id, std::memory_order_relaxed);
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

// Shims for OpKernel-related types using provider_api.h's versions
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

  cudaStream_t Stream(onnxruntime::OpKernelContext* ctx) const {
    if (!ctx) return nullptr;
    // Map onnxruntime::OpKernelContext* (plugin version) to OrtKernelContext* and use Ort::KernelContext to get GPU stream.
    return static_cast<cudaStream_t>(Ort::KernelContext(reinterpret_cast<OrtKernelContext*>(ctx)).GetGPUComputeStream());
  }

  static cudnnHandle_t GetCudnnHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCudnnHandle() : nullptr;
  }
  cudnnHandle_t GetCudnnHandle(onnxruntime::OpKernelContext* ctx) const { return GetCudnnHandle(Stream(ctx)); }

  static cublasHandle_t GetCublasHandle(cudaStream_t s) {
    auto* sync = cuda_plugin::CudaSyncStream::FromCudaStream(s);
    return sync ? sync->GetCublasHandle() : nullptr;
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
