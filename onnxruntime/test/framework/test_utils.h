// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <map>
#include <string>

#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/ort_value.h"

#include "core/common/gsl.h"

#ifdef USE_CUDA
#include "core/providers/providers.h"
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
  Tensor::InitOrtValue(element_type, shape, std::move(alloc), *p_mlvalue);

  if (!value.empty()) {
    Tensor& tensor = *p_mlvalue->GetMutable<Tensor>();
    CopyVectorToTensor(value, tensor);
  }
}

// Lifetime of data_buffer should be managed by the caller.
template <typename T>
void CreateMLValue(gsl::span<const int64_t> dims, T* data_buffer, const OrtMemoryInfo& info,
                   OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  Tensor::InitOrtValue(element_type, shape, data_buffer, info, *p_mlvalue);
}

template <typename T>
void AllocateMLValue(AllocatorPtr alloc, const std::vector<int64_t>& dims, OrtValue* p_mlvalue) {
  TensorShape shape(dims);
  auto element_type = DataTypeImpl::GetType<T>();
  Tensor::InitOrtValue(element_type, shape, std::move(alloc), *p_mlvalue);
}

using OpCountMap = std::map<std::string, int>;

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
OpCountMap CountOpsInGraph(const Graph& graph, bool recurse_into_subgraphs = true);

// Gets the op count from the OpCountMap.
// Can be called with a const OpCountMap, unlike OpCountMap::operator[].
inline int OpCount(const OpCountMap& op_count_map, const std::string& op_type) {
  if (auto it = op_count_map.find(op_type); it != op_count_map.end()) {
    return it->second;
  }
  return 0;
}

#if !defined(DISABLE_SPARSE_TENSORS)
void SparseIndicesChecker(const ONNX_NAMESPACE::TensorProto& indices_proto, gsl::span<const int64_t> expected_indicies);
#endif  // DISABLE_SPARSE_TENSORS

}  // namespace test
}  // namespace onnxruntime
