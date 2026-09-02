// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <memory>
#include <utility>

#include "core/providers/webgpu/webgpu_external_header.h"
#include "core/common/safeint.h"
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
  // Get the multi rotary cache concatenation offset (0 = disabled).
  //
  inline uint32_t MultiRotaryCacheConcatOffset() const {
    return ep_.MultiRotaryCacheConcatOffset();
  }

  //
  // Get the KV cache quantization bits (0 = disabled, 4 = 4-bit).
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
  // Get the logger.
  //
  inline const logging::Logger& Logger() const {
#if defined(ORT_USE_EP_API_ADAPTERS)
    return ep_.GetEpLogger();
#else
    return *ep_.GetLogger();
#endif
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

  // Creates a non-owning GPU tensor view over a planned workspace region. The WebGPU buffer handle
  // remains the tensor's allocation base; the region offset is preserved separately for bind-group
  // construction.
  template <typename TensorShapeType>
  Tensor CreateGPUTensorFromWorkspace(MLDataType data_type, TensorShapeType&& shape,
                                      const WorkspaceBufferRegion& workspace) {
    TensorShape tensor_shape{std::forward<TensorShapeType>(shape)};
    const size_t required_bytes = Tensor::CalculateTensorStorageSize(data_type, tensor_shape);
    constexpr size_t kWebGpuBufferAlignment = 16;
    const size_t binding_bytes = static_cast<size_t>(
        (SafeInt<size_t>(required_bytes) + kWebGpuBufferAlignment - 1) /
        kWebGpuBufferAlignment * kWebGpuBufferAlignment);
    ORT_ENFORCE(workspace.buffer != nullptr || required_bytes == 0,
                "A non-empty workspace tensor requires a buffer.");
    ORT_ENFORCE(workspace.offset_bytes <=
                    std::numeric_limits<size_t>::max() - binding_bytes,
                "Workspace tensor range overflows size_t.");
    ORT_ENFORCE(binding_bytes <= workspace.size_bytes,
                "Workspace tensor requires ", binding_bytes,
                " binding bytes after WebGPU alignment but the region has ",
                workspace.size_bytes, " bytes.");
    ORT_ENFORCE(workspace.offset_bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                "Workspace tensor offset exceeds ptrdiff_t.");
    const uint64_t offset_alignment = DeviceLimits().minStorageBufferOffsetAlignment;
    ORT_ENFORCE(offset_alignment == 0 || workspace.offset_bytes % offset_alignment == 0,
                "Workspace tensor offset ", workspace.offset_bytes,
                " is not aligned to WebGPU's ", offset_alignment, "-byte requirement.");

    AllocatorPtr allocator;
    ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceAllocator(&allocator));
    return {data_type, tensor_shape, workspace.buffer, allocator->Info(),
            static_cast<ptrdiff_t>(workspace.offset_bytes)};
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
    webgpu_context_.EndComputePass();
    auto& command_encoder = webgpu_context_.GetCommandEncoder();
    ORT_ENFORCE(dst.ByteOffset() >= 0, "WebGPU tensor buffer offset must not be negative.");
    WGPUBuffer buffer = reinterpret_cast<WGPUBuffer>(dst.MutableDataRawBase());
    constexpr uint64_t kClearBufferAlignment = 4;
    const uint64_t clear_size =
        (SafeInt<uint64_t>(dst.SizeInBytes()) + kClearBufferAlignment - 1) /
        kClearBufferAlignment * kClearBufferAlignment;
    command_encoder.ClearBuffer(buffer, static_cast<uint64_t>(dst.ByteOffset()), clear_size);
  }

 private:
  OpKernelContext& kernel_context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
