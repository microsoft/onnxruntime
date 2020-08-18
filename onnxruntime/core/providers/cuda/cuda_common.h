// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cuda_pch.h"
#include "core/common/status.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph_viewer.h"
#include "shared_inc/cuda_call.h"
#include "cuda_execution_provider.h"
#include "shared_inc/fast_divmod.h"
#include "core/util/math.h"
#include "cuda_fwd.h"

namespace onnxruntime {
namespace cuda {

#define CUDA_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(CUDA_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error executing ", #expr))

#define CUBLAS_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(CUBLAS_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUBLAS error executing ", #expr))

#define CUSPARSE_RETURN_IF_ERROR(expr)           \
  ORT_RETURN_IF_ERROR(CUSPARSE_CALL(expr)        \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUSPARSE error executing ", #expr))

#define CURAND_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(CURAND_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CURAND error executing ", #expr))

#define CUDNN_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(CUDNN_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDNN error executing ", #expr))

#define CUDNN2_RETURN_IF_ERROR(expr, m)          \
  ORT_RETURN_IF_ERROR(CUDNN_CALL2(expr, m)       \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDNN2 error executing ", #expr))

#define CUFFT_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(CUFFT_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUFFT error executing ", #expr))

// -----------------------------------------------------------------------
// Base class for CUDA kernels
// -----------------------------------------------------------------------
class CudaKernel : public OpKernel {
 public:
  explicit CudaKernel(const OpKernelInfo& info)
      : OpKernel(info),
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<CUDAExecutionProvider*>(static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()))) {
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);
    // use this to precisely locate the node where CUDA failure comes from
    //  if (cudaSuccess != cudaDeviceSynchronize())
    //    __debugbreak();

    if (s.IsOK()) {
      auto err = cudaGetLastError();
      if (err != cudaSuccess) {
        s = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
      }
    }

    return s;
  }

  virtual Status ComputeInternal(OpKernelContext* p_op_kernel_context) const = 0;

  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
    AllocatorPtr allocator = provider_->GetAllocator(CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
    if (!allocator)
      return nullptr;
    return IAllocator::MakeUniquePtr<T>(allocator, count_or_bytes);
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    return provider_->GetScratchBuffer<T>(count_or_bytes);
  }

  inline void AddDeferredReleaseCPUPtr(void* p) const {
    provider_->AddDeferredReleaseCPUPtr(p);
  }

  const cudaDeviceProp& GetDeviceProp() const { return provider_->GetDeviceProp(); };

  // To support cudaMemcpyAsync, the cpu memory should be allocated in pinned memory
  // and it can only be released after the copy has finished
  template <typename T>
  class CudaAsyncBuffer {
   public:
    CudaAsyncBuffer(const CudaKernel* op_kernel) : gpu_copy_(nullptr), count_(0), op_kernel_(op_kernel) {}

    CudaAsyncBuffer(const CudaKernel* op_kernel, size_t count) : CudaAsyncBuffer(op_kernel) {
      AllocCpuPtr(count);
    }

    CudaAsyncBuffer(const CudaKernel* op_kernel, const T& value, size_t count)
        : CudaAsyncBuffer(op_kernel, count) {
      T* p = CpuPtr();
      for (size_t i = 0; i != count; ++i) {
        *p++ = value;
      }
    }

    CudaAsyncBuffer(const CudaKernel* op_kernel, const std::vector<T>& vec) : CudaAsyncBuffer(op_kernel, vec.size()) {
      memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T));
    }

    void AllocCpuPtr(size_t count) {
      cpu_pinned_copy_ = op_kernel_->AllocateBufferOnCPUPinned<T>(count);
      if (cpu_pinned_copy_ == nullptr)
        throw std::runtime_error("alloc failed");
      count_ = count;
    }

    Status CopyToGpu() {
      if (cpu_pinned_copy_) {
        gpu_copy_ = op_kernel_->GetScratchBuffer<T>(count_);
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(gpu_copy_.get(), cpu_pinned_copy_.get(), count_ * sizeof(T), cudaMemcpyHostToDevice));
        op_kernel_->AddDeferredReleaseCPUPtr(cpu_pinned_copy_.release());
      }
      return Status::OK();
    }

    T* CpuPtr() const {
      return cpu_pinned_copy_.get();
    }

    gsl::span<T> CpuSpan() const {
      return gsl::span<T>(CpuPtr(), count_);
    }

    T* GpuPtr() const {
      return gpu_copy_.get();
    }

    size_t count() const {
      return count_;
    }

   protected:
    IAllocatorUniquePtr<T> gpu_copy_;
    IAllocatorUniquePtr<T> cpu_pinned_copy_;
    size_t count_;
    const CudaKernel* op_kernel_;
  };

  inline cublasHandle_t CublasHandle() const {
    return provider_->PerThreadCublasHandle();
  }

  inline cudnnHandle_t CudnnHandle() const {
    return provider_->PerThreadCudnnHandle();
  }

 protected:
  inline curandGenerator_t CurandGenerator() const {
    return provider_->PerThreadCurandGenerator();
  }

  template <typename T>
  inline const T* GetConstOnes(size_t count) const {
    return provider_->template GetConstOnes<T>(count);
  }

  inline Status CopyTensor(const Tensor& src, Tensor& dst) const {
    return Info().GetDataTransferManager().CopyTensor(src, dst);
  }

  inline int GetDeviceId() const { return provider_->GetDeviceId(); }

 private:
  CUDAExecutionProvider* provider_;
};

// Type mapping for MLFloat16 to half
template <typename T>
class ToCudaType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) {
    return static_cast<T>(f);
  }
};

template <>
class ToCudaType<MLFloat16> {
 public:
  typedef half MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

inline bool CalculateFdmStrides(gsl::span<fast_divmod> p, const std::vector<int64_t>& dims) {
  int stride = 1;
  if (dims.empty() || p.size() < dims.size())
    return false;
  auto rank = p.size();
  for (size_t i = 0; i < rank; i++) {
    p[rank - 1 - i] = fast_divmod(stride);
    if (i < dims.size() - 1) {
      stride *= static_cast<int>(dims[dims.size() - 1 - i]);
    }
  }
  return true;
}

class CublasMathModeSetter {
 public:
  CublasMathModeSetter(const cudaDeviceProp& prop,cublasHandle_t handle, cublasMath_t mode) : prop_(prop), handle_(handle) {
    if (prop_.major >= 7) {
      cublasGetMathMode(handle, &mode_);
      cublasSetMathMode(handle, mode);
    }
  }
  ~CublasMathModeSetter() {
    if (prop_.major >= 7) {
      cublasSetMathMode(handle_, mode_);
    }
  }

 private:
  const cudaDeviceProp& prop_;
  cublasHandle_t handle_;
  cublasMath_t mode_;
};

}  // namespace cuda
}  // namespace onnxruntime
