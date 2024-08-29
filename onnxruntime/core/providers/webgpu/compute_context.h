// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include <utility>

#include "core/framework/execution_provider.h"

#include "core/providers/webgpu/program.h"

namespace onnxruntime {

class Tensor;
class OpKernelContext;

namespace webgpu {

class WebGpuContext;

class ComputeContext {
 public:
  ComputeContext(OpKernelContext& kernel_context);

  virtual ~ComputeContext() = default;

  //
  // Get various information from the context.
  //

  const wgpu::AdapterInfo& AdapterInfo() const;
  const wgpu::Limits& DeviceLimits() const;

  //
  // Get input tensor.
  //
  template <typename T = onnxruntime::Tensor>
  const T* Input(int index) const {
    return kernel_context_.Input<T>(index);
  }

  //
  // Get input count.
  //
  int InputCount() const;

  //
  // Set output tensor.
  //
  template <typename TensorShapeType>
  Tensor* Output(int index, TensorShapeType&& shape) {
    return kernel_context_.Output(index, std::forward<TensorShapeType>(shape));
  }

  //
  // Get output count.
  //
  int OutputCount() const;

  //
  // Create CPU tensor.
  //
  template <typename TensorShapeType>
  Tensor CreateCPUTensor(MLDataType data_type, TensorShapeType&& shape) {
    AllocatorPtr allocator;
    ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceCPUAllocator(&allocator));
    return {data_type, std::forward<TensorShapeType>(shape)..., allocator};
  }

  //
  // Create GPU tensor.
  //
  template <typename TensorShapeType>
  Tensor CreateGPUTensor(MLDataType data_type, TensorShapeType&& shape) {
    AllocatorPtr allocator;
    ORT_THROW_IF_ERROR(kernel_context_.GetTempSpaceAllocator(&allocator));
    return {data_type, std::forward<TensorShapeType>(shape)..., allocator};
  }

  //
  // Run a compute shader program.
  //
  Status RunProgram(const ProgramBase& program);

 protected:
  WebGpuContext& webgpu_context_;
  OpKernelContext& kernel_context_;
};

}  // namespace webgpu
}  // namespace onnxruntime
