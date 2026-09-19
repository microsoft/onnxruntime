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
    static CommandRecordingState& GetRecording(const ComputeContextBase& context);
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
  inline const wgpu::AdapterPropertiesSubgroupMatrixConfigs& SubgroupMatrixConfigs() const {
    return webgpu_context_.SubgroupMatrixConfigs();
  }

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
  // Get the multi rotary cache concatenation offset (0 = disabled).
  //
  inline uint32_t MultiRotaryCacheConcatOffset() const {
    return ep_.MultiRotaryCacheConcatOffset();
  }

  //
  // Get the KV cache quantization bit width (0 = disabled, 4 = TurboQuant, 8 = symmetric block quantization).
  //
  inline uint32_t KvCacheQuantizationBits() const {
    return ep_.KvCacheQuantizationBits();
  }

  //
  // Get whether KV cache quantization is enabled.
  //
  inline bool KvCacheQuantizationEnabled() const {
    return ep_.KvCacheQuantizationEnabled();
  }

  //
  // Get whether MatMulNBits dot products accumulate in f32 rather than in the output element type.
  //
  inline bool EnableMatmulFp32Accumulation() const {
    return ep_.EnableMatmulFp32Accumulation();
  }

  //
  // Get the logger.
  //
#if defined(ORT_USE_EP_API_ADAPTERS)
  inline const onnxruntime::ep::adapter::Logger& Logger() const {
    return ep_.GetEpLogger();
  }
#else
  inline const logging::Logger& Logger() const {
    return *ep_.GetLogger();
  }
#endif

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
  // In the plugin, the temp-space allocator is the existing Session device allocator,
  // not an Env shared allocator or a new allocator created for each tensor.
  //
  template <typename TensorShapeType>
  Tensor CreateGPUTensor(MLDataType data_type, TensorShapeType&& shape) {
    AllocatorPtr allocator;
    ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceAllocator(&allocator));
#if defined(ORT_USE_EP_API_ADAPTERS)
    TensorShape tensor_shape{std::forward<TensorShapeType>(shape)};
    const size_t bytes = Tensor::CalculateTensorStorageSize(data_type, tensor_shape);
    // Keep scratch allocation associated with the kernel's Session stream. Ordinary caches
    // return zeroed buffers; capture caches retain their allocation-time initialization policy.
    auto buffer = IAllocator::MakeUniquePtr<void>(
        allocator, bytes, false, reinterpret_cast<Stream*>(kernel_context_.GetSyncStream()));
    Tensor tensor(data_type, tensor_shape, buffer.get(), allocator);
    buffer.release();
    return tensor;
#else
    return {data_type, std::forward<TensorShapeType>(shape), allocator};
#endif
  }

  //
  // Copy data from a tensor to another tensor.
  //
  // This method assumes that both tensors have the same data size.
  //
  inline Status CopyTensor(const Tensor& src, Tensor& dst) {
    return op_kernel_.Info().GetDataTransferManager().CopyTensor(src, dst);
  }

  //
  // Fill a GPU tensor with zeros.
  //
  inline void FillZero(Tensor& dst) {
    auto& recording = ep_.Recording();
    std::lock_guard<std::recursive_mutex> lock{recording.mutex};
    ORT_THROW_IF_ERROR(webgpu_context_.EncodeDeferredDispatches(recording));
    webgpu_context_.EndComputePass(recording);
    auto& command_encoder = webgpu_context_.GetCommandEncoder(recording);
    WGPUBuffer buffer = reinterpret_cast<WGPUBuffer>(dst.MutableDataRaw());
    command_encoder.ClearBuffer(buffer, 0, dst.SizeInBytes());
  }

 private:
  OpKernelContext& kernel_context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
