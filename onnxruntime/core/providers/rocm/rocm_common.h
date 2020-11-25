// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "rocm_pch.h"
#include "core/common/status.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph_viewer.h"
#include "shared_inc/rocm_call.h"
#include "rocm_execution_provider.h"
#include "shared_inc/fast_divmod.h"
#include "core/util/math.h"
#include "rocm_fwd.h"

namespace onnxruntime {
namespace rocm {

#define HIP_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(HIP_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIP error executing ", #expr))

#define ROCBLAS_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(ROCBLAS_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ROCBLAS error executing ", #expr))

#define HIPSPARSE_RETURN_IF_ERROR(expr)           \
  ORT_RETURN_IF_ERROR(HIPSPARSE_CALL(expr)        \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPSPARSE error executing ", #expr))

#define HIPRAND_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(HIPRAND_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPRAND error executing ", #expr))

#define MIOPEN_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(MIOPEN_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MIOPEN error executing ", #expr))

#define MIOPEN2_RETURN_IF_ERROR(expr, m)          \
  ORT_RETURN_IF_ERROR(MIOPEN_CALL2(expr, m)       \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MIOPEN2 error executing ", #expr))

#define HIPFFT_RETURN_IF_ERROR(expr)              \
  ORT_RETURN_IF_ERROR(HIPFFT_CALL(expr)           \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPFFT error executing ", #expr))

// -----------------------------------------------------------------------
// Base class for HIP kernels
// -----------------------------------------------------------------------
class RocmKernel : public OpKernel {
 public:
  explicit RocmKernel(const OpKernelInfo& info)
      : OpKernel(info),
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<ROCMExecutionProvider*>(static_cast<const ROCMExecutionProvider*>(info.GetExecutionProvider()))) {
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);

    if (s.IsOK()) {
      auto err = hipGetLastError();
      if (err != hipSuccess) {
        s = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIP error ", hipGetErrorName(err), ":", hipGetErrorString(err));
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

  const hipDeviceProp_t& GetDeviceProp() const { return provider_->GetDeviceProp(); };

  // To support hipMemcpyAsync, the cpu memory should be allocated in pinned memory
  // and it can only be released after the copy has finished
  template <typename T>
  class RocmAsyncBuffer {
   public:
    RocmAsyncBuffer(const RocmKernel* op_kernel) : gpu_copy_(nullptr), count_(0), op_kernel_(op_kernel) {}

    RocmAsyncBuffer(const RocmKernel* op_kernel, size_t count) : RocmAsyncBuffer(op_kernel) {
      AllocCpuPtr(count);
    }

    RocmAsyncBuffer(const RocmKernel* op_kernel, const T& value, size_t count)
        : RocmAsyncBuffer(op_kernel, count) {
      T* p = CpuPtr();
      for (size_t i = 0; i != count; ++i) {
        *p++ = value;
      }
    }

    RocmAsyncBuffer(const RocmKernel* op_kernel, const std::vector<T>& vec) : RocmAsyncBuffer(op_kernel, vec.size()) {
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
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(gpu_copy_.get(), cpu_pinned_copy_.get(), count_ * sizeof(T), hipMemcpyHostToDevice));
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
    const RocmKernel* op_kernel_;
  };

  inline rocblas_handle RocblasHandle() const {
    return provider_->PerThreadRocblasHandle();
  }

  inline miopenHandle_t MiopenHandle() const {
    return provider_->PerThreadMiopenHandle();
  }

 protected:

  template <typename T>
  inline const T* GetConstOnes(size_t count) const {
    return provider_->template GetConstOnes<T>(count);
  }

  inline Status CopyTensor(const Tensor& src, Tensor& dst) const {
    return Info().GetDataTransferManager().CopyTensor(src, dst);
  }

  inline int GetDeviceId() const { return provider_->GetDeviceId(); }

 private:
  ROCMExecutionProvider* provider_;
};

// Type mapping for MLFloat16 to half
template <typename T>
class ToHipType {
 public:
  typedef T MappedType;
  static MappedType FromFloat(float f) {
    return static_cast<T>(f);
  }
};

template <>
class ToHipType<MLFloat16> {
 public:
  typedef __half MappedType;
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

}  // namespace rocm
}  // namespace onnxruntime
