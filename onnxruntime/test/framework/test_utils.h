// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <map>
#include <string>

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/ml_value.h"

#include "gsl/gsl"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif
#ifdef USE_ROCM
#include "core/providers/rocm/rocm_execution_provider.h"
#endif
#ifdef USE_NNAPI
#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"
#endif
#ifdef USE_RKNPU
#include "core/providers/rknpu/rknpu_execution_provider.h"
#endif
#ifdef USE_COREML
#include "core/providers/coreml/coreml_execution_provider.h"
#endif

namespace onnxruntime {
class Graph;

namespace test {
// Doesn't work with ExecutionProviders class and KernelRegistryManager
IExecutionProvider* TestCPUExecutionProvider();

#ifdef USE_CUDA
// Doesn't work with ExecutionProviders class and KernelRegistryManager
IExecutionProvider* TestCudaExecutionProvider();
#endif

#ifdef USE_ROCM
IExecutionProvider* TestRocmExecutionProvider();
#endif

#ifdef USE_TENSORRT
// Doesn't work with ExecutionProviders class and KernelRegistryManager
IExecutionProvider* TestTensorrtExecutionProvider();
#endif

#ifdef USE_OPENVINO
IExecutionProvider* TestOpenVINOExecutionProvider();
#endif

#ifdef USE_NNAPI
IExecutionProvider* TestNnapiExecutionProvider();
#endif

#ifdef USE_RKNPU
IExecutionProvider* TestRknpuExecutionProvider();
#endif

#ifdef USE_COREML
IExecutionProvider* TestCoreMLExecutionProvider(uint32_t coreml_flags);
#endif

template <typename T>
inline void CopyVectorToTensor(const std::vector<T>& value, Tensor& tensor) {
  gsl::copy(gsl::make_span(value), tensor.MutableDataAsSpan<T>());
}

// vector<bool> is specialized so we need to handle it separately
template <>
inline void CopyVectorToTensor<bool>(const std::vector<bool>& value, Tensor& tensor) {
  auto output_span = tensor.MutableDataAsSpan<bool>();
  for (size_t i = 0, end = value.size(); i < end; ++i) {
    output_span[i] = value[i];
  }
}

template <typename T>
void CreateMLValue(AllocatorPtr alloc, const std::vector<int64_t>& dims, const std::vector<T>& value,
                   OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                                      shape,
                                                                      alloc);
  if (value.size() > 0) {
    CopyVectorToTensor(value, *p_tensor);
  }

  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

// Lifetime of data_buffer should be managed by the caller.
template <typename T>
void CreateMLValue(const std::vector<int64_t>& dims, T* data_buffer, const OrtMemoryInfo& info,
                   OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                                      shape,
                                                                      data_buffer,
                                                                      info);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

template <typename T>
void AllocateMLValue(AllocatorPtr alloc, const std::vector<int64_t>& dims, OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type,
                                                                      shape,
                                                                      alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
std::map<std::string, int> CountOpsInGraph(const Graph& graph, bool recurse_into_subgraphs = true);

}  // namespace test
}  // namespace onnxruntime
