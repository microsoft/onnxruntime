// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <deque>

#include <hipblas.h>

#include "core/platform/ort_mutex.h"
#include "core/framework/allocator.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

const int CPU_ALLOCATOR_DEVICE_ID = 0;

struct HIPExecutionProviderInfo {
  OrtDevice::DeviceId device_id{0};
};

// Logical device representation.
class HIPExecutionProvider : public IExecutionProvider {
 public:
  explicit HIPExecutionProvider(const HIPExecutionProviderInfo& info);
  ~HIPExecutionProvider();

  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  
  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  int GetDeviceId() const { return device_id_; }

  Status Sync() const override;
  Status OnRunStart() override;
  Status OnRunEnd() override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes) const {
    if (count_or_bytes == 0)
      return nullptr;

    return IAllocator::MakeUniquePtr<T>(GetAllocator(device_id_, OrtMemTypeDefault), count_or_bytes);
  }

  hipblasHandle_t PerThreadHipblasHandle() {
    return GetPerThreadContext().HipblasHandle();
  }

  // cudnnHandle_t PerThreadCudnnHandle() {
  //   return GetPerThreadContext().CudnnHandle();
  // }
  // curandGenerator_t PerThreadCurandGenerator() {
  //   return GetPerThreadContext().CurandGenerator();
  // }

  void AddDeferredReleaseCPUPtr(void* p);
  const hipDeviceProp_t& GetDeviceProp() { return prop_; };

private:
  OrtDevice::DeviceId device_id_;
  AllocatorPtr allocator_;
  hipDeviceProp_t prop_;

  struct DeferredReleaseCPUPtrs {
    bool recorded = false;
    std::vector<void*> cpu_ptrs;
  };
  std::unordered_map<hipEvent_t, DeferredReleaseCPUPtrs> deferred_release_cpu_ptr_;
  OrtMutex deferred_release_cpu_ptr_mutex_;

  class PerThreadContext final {
   public:
    PerThreadContext(OrtDevice::DeviceId device_id);
    ~PerThreadContext();

    hipblasHandle_t HipblasHandle() const {
      return hipblas_handle_;
    }

    // cudnnHandle_t CudnnHandle() const {
    //   return cudnn_handle_;
    // }

    // curandGenerator_t CurandGenerator() const {
    //   return curand_generator_;
    // }

    hipEvent_t& GetCurrentDeferredReleaseEvent() {
      return current_deferred_release_event_;
    }

    // template <typename T>
    // const T* GetConstOnes(size_t count) {
    //   if (std::is_same<T, float>::value) {
    //     if (!constant_ones_float_) {
    //       constant_ones_float_ = hip::CreateConstantOnes<float>();
    //     }
    //     return reinterpret_cast<const T*>(constant_ones_float_->GetBuffer(count));
    //   } else if (std::is_same<T, double>::value) {
    //     if (!constant_ones_double_) {
    //       constant_ones_double_ = hip::CreateConstantOnes<double>();
    //     }
    //     return reinterpret_cast<const T*>(constant_ones_double_->GetBuffer(count));
    //   } else if (std::is_same<T, half>::value) {
    //     if (!constant_ones_half_) {
    //       constant_ones_half_ = hip::CreateConstantOnes<half>();
    //     }
    //     return reinterpret_cast<const T*>(constant_ones_half_->GetBuffer(count));
    //   } else {
    //     return nullptr;
    //   }
    // }

    AllocatorPtr GetAllocator() const {
      return allocator_;
    }

   private:
    hipblasHandle_t hipblas_handle_ = nullptr;
    // cudnnHandle_t cudnn_handle_ = nullptr;
    // curandGenerator_t curand_generator_ = nullptr;

    // deferred release for temporary CPU pinned memory used in hipMemcpyAsync
    // note that hipEvent will be assigned at OnRunEnd() when PerThreadContext destory
    // so the ownership is passed to deferred_release_cpu_ptr_
    hipEvent_t current_deferred_release_event_ = nullptr;

    // std::unique_ptr<hip::IConstantBuffer<float>> constant_ones_float_;
    // std::unique_ptr<hip::IConstantBuffer<double>> constant_ones_double_;
    // std::unique_ptr<hip::IConstantBuffer<half>> constant_ones_half_;

    AllocatorPtr allocator_;
  };

  // thread local context during execution
  using PerThreadContextMap = std::unordered_map<const HIPExecutionProvider*, std::shared_ptr<PerThreadContext>>;
  static thread_local std::unique_ptr<PerThreadContextMap> per_thread_context_map_;

  // reuse thread local context
  mutable std::deque<std::shared_ptr<PerThreadContext>> retired_context_pool_;
  mutable OrtMutex context_pool_mutex_;

  PerThreadContext& GetPerThreadContext() const;
  void ReleasePerThreadStuffs() const;
};

}
