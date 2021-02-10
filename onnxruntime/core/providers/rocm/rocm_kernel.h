// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_fwd.h"

namespace onnxruntime {
namespace rocm {

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
    AllocatorPtr allocator = provider_->GetAllocator(DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
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

  inline hipStream_t Stream() const { return static_cast<hipStream_t>(provider_->GetComputeStream()); }

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
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(gpu_copy_.get(), cpu_pinned_copy_.get(), count_ * sizeof(T), hipMemcpyHostToDevice, op_kernel_->Stream()));
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

}  // namespace rocm
}  // namespace onnxruntime
