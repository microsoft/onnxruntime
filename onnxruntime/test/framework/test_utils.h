// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/ml_value.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif
#ifdef USE_TENSORRT 
#include "core/providers/tensorrt/tensorrt_execution_provider.h"
#endif

namespace onnxruntime {
namespace test {
// Doesn't work with ExecutionProviders class and KernelRegistryManager
IExecutionProvider* TestCPUExecutionProvider();

#ifdef USE_CUDA
// Doesn't work with ExecutionProviders class and KernelRegistryManager
IExecutionProvider* TestCudaExecutionProvider();
#endif

#ifdef USE_TENSORRT
// Doesn't work with ExecutionProviders class and KernelRegistryManager
IExecutionProvider* TestTensorrtExecutionProvider();
#endif

template <typename T>
void CreateMLValue(AllocatorPtr alloc, const std::vector<int64_t>& dims, const std::vector<T>& value,
                   OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              alloc);
  if (value.size() > 0) {
    memcpy(p_tensor->MutableData<T>(), &value[0], element_type->Size() * shape.Size());
  }
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

template <typename T>
void AllocateMLValue(AllocatorPtr alloc, const std::vector<int64_t>& dims, OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}
}  // namespace test
}  // namespace onnxruntime
