// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/backward_guard.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_fwd.h"
#include "core/providers/rocm/rocm_stream_handle.h"

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
    Status s;
    auto is_backward_pass = Info().GetAttrOrDefault<int64_t>("__backwardpass", 0);
    if (is_backward_pass) {
      BackwardPassGuard guard;
      s = ComputeInternal(p_op_kernel_context);
    } else {
      s = ComputeInternal(p_op_kernel_context);
    }
    // use this to precisely locate the node where ROCM failure comes from
    //  if (hipSuccess != hipDeviceSynchronize())
    //    __debugbreak();

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
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, onnxruntime::Stream* stream) const {
    if (count_or_bytes == 0) return nullptr;
    return IAllocator::MakeUniquePtr<T>(Info().GetAllocator(OrtMemType::OrtMemTypeDefault), count_or_bytes, false, stream, WaitRocmNotificationOnDevice);
  }

  // Different from GetScratchBuffer which use IAllocator::Alloc() to allocate memory,
  // this GetTransientScratchBuffer will call IAllocator::Reserve() to allocate memory.
  // IAllocator::Reserve() optionally implement some allocation logic that by-passes any arena-based
  // logic (or similar for different allocator) that may be housed in the Alloc() implementation.
  template <typename T>
  inline IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0) return nullptr;
    return IAllocator::MakeUniquePtr<T>(Info().GetAllocator(OrtMemType::OrtMemTypeDefault), count_or_bytes, true);
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
    if (count_or_bytes == 0) return nullptr;
    return IAllocator::MakeUniquePtr<T>(Info().GetAllocator(OrtMemType::OrtMemTypeCPU), count_or_bytes);
  }

  inline void AddDeferredReleaseCPUPtr(void* p, onnxruntime::Stream* ort_stream) const {
    ORT_ENFORCE(ort_stream->GetDevice().Type() == OrtDevice::GPU);
    auto* rocm_ep_stream = static_cast<RocmStream*>(ort_stream);
    rocm_ep_stream->EnqueDeferredCPUBuffer(p);
  }

  const hipDeviceProp_t& GetDeviceProp() const { return provider_->GetDeviceProp(); }

  inline hipStream_t Stream(OpKernelContext* ctx) const {
    auto* stream = ctx->GetComputeStream();
    return stream ? static_cast<hipStream_t>(stream->GetHandle()) : nullptr;
  }

  tunable::RocmTuningContext* GetTuningContext() const {
    return static_cast<tunable::RocmTuningContext*>(provider_->GetTuningContext());
  }

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

    RocmAsyncBuffer(const RocmKernel* op_kernel, gsl::span<const T> vec) : RocmAsyncBuffer(op_kernel, vec.size()) {
      memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T));
    }

    void AllocCpuPtr(size_t count) {
      cpu_pinned_copy_ = op_kernel_->AllocateBufferOnCPUPinned<T>(count);
      if (cpu_pinned_copy_ == nullptr)
        throw std::runtime_error("alloc failed");
      count_ = count;
    }

    Status CopyToGpu(onnxruntime::Stream* stream) {
      if (cpu_pinned_copy_) {
        gpu_copy_ = op_kernel_->GetScratchBuffer<T>(count_, stream);
        hipStream_t rocm_stream = stream ? static_cast<hipStream_t>(stream->GetHandle()) : nullptr;
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(gpu_copy_.get(), cpu_pinned_copy_.get(), count_ * sizeof(T), hipMemcpyHostToDevice,
                                           rocm_stream));
        op_kernel_->AddDeferredReleaseCPUPtr(cpu_pinned_copy_.release(), stream);
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

  static inline rocblas_handle GetRocblasHandle(onnxruntime::RocmStream* stream) {
    return stream->rocblas_handle_;
  }

  inline rocblas_handle GetRocblasHandle(OpKernelContext* ctx) const {
    return GetRocblasHandle(static_cast<RocmStream*>(ctx->GetComputeStream()));
  }

  static inline miopenHandle_t GetMiopenHandle(onnxruntime::RocmStream* stream) {
    return stream->miopen_handle_;
  }

  inline miopenHandle_t GetMiopenHandle(OpKernelContext* ctx) const {
    return GetMiopenHandle(static_cast<RocmStream*>(ctx->GetComputeStream()));
  }

 protected:
  template <typename T>
  inline const T* GetConstOnes(size_t count, hipStream_t stream) const {
    return provider_->template GetConstOnes<T>(count, stream);
  }

  inline Status CopyTensor(const Tensor& src, Tensor& dst, onnxruntime::Stream& stream) const {
    auto* gpu_data_transfer = Info().GetDataTransferManager().GetDataTransfer(src.Location().device, dst.Location().device);
    return gpu_data_transfer->CopyTensorAsync(src, dst, stream);
  }

  inline int GetDeviceId() const { return provider_->GetDeviceId(); }

 private:
  ROCMExecutionProvider* provider_;
};

}  // namespace rocm
}  // namespace onnxruntime
