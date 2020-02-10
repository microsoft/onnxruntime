// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <hipblas.h>
//#include <hiprand/hiprand.h>

#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/framework/data_transfer_manager.h"
#include "core/graph/graph_viewer.h"
#include "core/util/math.h"

#include "core/providers/hip/fast_divmod.h"
#include "core/providers/hip/hip_call.h"
#include "core/providers/hip/hip_execution_provider.h"


namespace onnxruntime {
namespace hip {

#define HIP_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(HIP_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIP error executing ", #expr))

#define HIPBLAS_RETURN_IF_ERROR(expr)             \
  ORT_RETURN_IF_ERROR(HIPBLAS_CALL(expr)          \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "HIPBLAS error executing ", #expr))
/*
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
*/

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

typedef __half half;

// -----------------------------------------------------------------------
// Base class for HIP kernels
// -----------------------------------------------------------------------
class HipKernel : public OpKernel {
 public:
  explicit HipKernel(const OpKernelInfo& info)
      : OpKernel(info),
        // Is this OK to have a non-const execution provider?
        provider_(const_cast<HIPExecutionProvider*>(dynamic_cast<const HIPExecutionProvider*>(info.GetExecutionProvider()))) {
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override {
    auto s = ComputeInternal(p_op_kernel_context);
    // use this to precisely locate the node where HIP failure comes from
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

  // To support hipMemcpyAsync, the cpu memory should be allocated in pinned memory
  // and it can only be released after the copy has finished
  template <typename T>
  class HipAsyncBuffer {
   public:
    HipAsyncBuffer(const HipKernel* op_kernel) : gpu_copy_(nullptr), count_(0), op_kernel_(op_kernel) {}

    HipAsyncBuffer(const HipKernel* op_kernel, size_t count) : HipAsyncBuffer(op_kernel) {
      AllocCpuPtr(count);
    }

    HipAsyncBuffer(const HipKernel* op_kernel, const T& value, size_t count)
        : HipAsyncBuffer(op_kernel, count) {
      T* p = CpuPtr();
      for (size_t i = 0; i != count; ++i) {
        *p++ = value;
      }
    }

    HipAsyncBuffer(const HipKernel* op_kernel, const std::vector<T>& vec) : HipAsyncBuffer(op_kernel, vec.size()) {
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
    const HipKernel* op_kernel_;
  };

 protected:
  inline hipblasHandle_t HipblasHandle() const {
    return provider_->PerThreadHipblasHandle();
  }

  // inline hipdnnHandle_t CudnnHandle() const {
  //   return provider_->PerThreadCudnnHandle();
  // }
  // inline hiprandGenerator_t CurandGenerator() const {
  //   return provider_->PerThreadCurandGenerator();
  // }

  // template <typename T>
  // inline const T* GetConstOnes(size_t count) const {
  //   return provider_->template GetConstOnes<T>(count);
  // }

  inline Status CopyTensor(const Tensor& src, Tensor& dst) const {
    return Info().GetDataTransferManager().CopyTensor(src, dst);
  }

  inline int GetDeviceId() const { return provider_->GetDeviceId(); }

 private:
  HIPExecutionProvider* provider_;
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

struct DeviceProp {
  static const std::vector<hipDeviceProp_t>& GetCachedDeviceProps() {
    std::call_once(s_cachedDevicePropsInitFlag, [=] {
      int numDevices;
      // must wait GPU idle, otherwise hipGetDeviceProperties might fail
      HIP_CALL_THROW(hipDeviceSynchronize());
      HIP_CALL_THROW(hipGetDeviceCount(&numDevices));
      s_cachedDeviceProps.resize(numDevices);
      for (int i = 0; i < numDevices; i++)
        HIP_CALL_THROW(hipGetDeviceProperties(&s_cachedDeviceProps[i], i));
    });

    return s_cachedDeviceProps;
  }

  static size_t GetCurrentDeviceId() {
    int deviceId;
    hipGetDevice(&deviceId);
    return (size_t)deviceId;
  }

  // get device properties of current device
  static const hipDeviceProp_t& GetDeviceProps() {
    const auto& cachedDevicesProps = GetCachedDeviceProps();
    return cachedDevicesProps[GetCurrentDeviceId()];
  }

 private:
  static std::vector<hipDeviceProp_t> s_cachedDeviceProps;
  static std::once_flag s_cachedDevicePropsInitFlag;
};

}  // namespace hip
}  // namespace onnxruntime
