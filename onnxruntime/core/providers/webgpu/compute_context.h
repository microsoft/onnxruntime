// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>

#include "core/providers/webgpu/webgpu_external_header.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_provider.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

class Tensor;

namespace webgpu {

class WebGpuContext;
class BufferManager;

//
// Class ComputeContextBase is designed to provide basic context information
// for running a compute shader program.
//
// An instance of ComputeContextBase does not depend on OpKernelContext, which needs an execution frame to be created.
//
class ComputeContextBase {
 public:
  // Nested accessor class to provide controlled access to BufferManager
  class BufferManagerAccessor {
    // access to BufferManager is limited to class WebGpuContext.
    // This ensures no access to BufferManager from other classes, avoiding
    // potential misuse.
    friend class WebGpuContext;

   private:
    static const webgpu::BufferManager& Get(const ComputeContextBase& context);
  };

  ComputeContextBase(WebGpuContext& webgpu_context,
                     const WebGpuExecutionProvider& ep,
                     const OpKernel& op_kernel);

  ~ComputeContextBase() = default;

  //
  // Get the node name.
  //
  inline decltype(auto) NodeName() const {
    return op_kernel_.Node().Name();
  }

  Status CreateUnmappedGPUTensor(AllocatorPtr alloc, MLDataType data_type, const TensorShape& shape,
                                 std::unique_ptr<Tensor>& tensor) const;

  //
  // Get the operator type.
  //
  inline decltype(auto) OpType() const {
    return op_kernel_.Node().OpType();
  }

  //
  // Get various information from the WebGPU context.
  //

  inline const wgpu::AdapterInfo& AdapterInfo() const {
    return webgpu_context_.AdapterInfo();
  }
  inline const wgpu::Limits& DeviceLimits() const {
    return webgpu_context_.DeviceLimits();
  }
  inline bool HasFeature(wgpu::FeatureName feature) const {
    return webgpu_context_.DeviceHasFeature(feature);
  }
#if !defined(__wasm__)
  inline const wgpu::AdapterPropertiesSubgroupMatrixConfigs& SubgroupMatrixConfigs() const {
    return webgpu_context_.SubgroupMatrixConfigs();
  }
#endif

  //
  // Get Split-K configuration.
  //
  inline const SplitKConfig& GetSplitKConfig() const {
    return webgpu_context_.GetSplitKConfig();
  }

  //
  // Get whether graph capture is enabled.
  //
  inline bool IsGraphCaptureEnabled() const {
    return ep_.IsGraphCaptureEnabled();
  }

  //
  // Get the logger.
  //
  inline const logging::Logger& Logger() const {
    return *ep_.GetLogger();
  }

  //
  // Run a compute shader program.
  //
  inline Status RunProgram(const ProgramBase& program) {
    return webgpu_context_.Run(*this, program);
  }

 protected:
  WebGpuContext& webgpu_context_;
  const WebGpuExecutionProvider& ep_;
  const OpKernel& op_kernel_;
};

//
// Class ComputeContext provides all information a `ComputeContextBase` provides, and also
// access to `OpKernelContext` for input and output tensors.
//
class ComputeContext final : public ComputeContextBase {
 public:
  ComputeContext(WebGpuContext& webgpu_context,
                 const WebGpuExecutionProvider& ep,
                 const OpKernel& op_kernel,
                 OpKernelContext& kernel_context);

  ~ComputeContext() = default;

  //
  // Get the kernel context.
  //
  inline OpKernelContext& KernelContext() {
    return kernel_context_;
  }

  //
  // Get input tensor.
  //
  template <typename T = onnxruntime::Tensor>
  inline const T* Input(int index) const {
    return kernel_context_.Input<T>(index);
  }

  //
  // Get input count.
  //
  inline int InputCount() const {
    return kernel_context_.InputCount();
  }

  //
  // Set output tensor.
  //
  template <typename TensorShapeType>
  inline Tensor* Output(int index, TensorShapeType&& shape) {
    return kernel_context_.Output(index, std::forward<TensorShapeType>(shape));
  }

  //
  // Get output count.
  //
  inline int OutputCount() const {
    return kernel_context_.OutputCount();
  }

  //
  // Create CPU tensor.
  //
  // This method creates a tensor of the given data type and shape, using the CPU allocator.
  // The tensor owns the underlying CPU memory buffer.
  //
  template <typename TensorShapeType>
  Tensor CreateCPUTensor(MLDataType data_type, TensorShapeType&& shape) {
    AllocatorPtr allocator;
    ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceCPUAllocator(&allocator));
    return {data_type, std::forward<TensorShapeType>(shape), allocator};
  }

  //
  // Create GPU tensor.
  //
  // This method creates a tensor of the given data type and shape, using the WebGPU allocator.
  // The tensor owns the underlying WebGPU storage buffer.
  //
  template <typename TensorShapeType>
  Tensor CreateGPUTensor(MLDataType data_type, TensorShapeType&& shape) {
    AllocatorPtr allocator;
    ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceAllocator(&allocator));
    return {data_type, std::forward<TensorShapeType>(shape), allocator};
  }

  //
  // Copy data from a tensor to another tensor.
  //
  // This method assumes that both tensors have the same data size.
  //
  inline Status CopyTensor(const Tensor& src, Tensor& dst) {
    return op_kernel_.Info().GetDataTransferManager().CopyTensor(src, dst);
  }

 private:
  OpKernelContext& kernel_context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
