// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/framework_common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/ort_value.h"
#include "core/providers/cpu/cpu_execution_provider.h"

#include <memory>
#include <string>
#include <vector>

namespace onnxruntime {
class Graph;

namespace test {

// If set to All: verify the entire graph is taken by ep
// If set to Some: verify that at least one node is assigned to ep
// If set to None: verify that no nodes is assigned to ep (typically for an expected failure path test case)
enum class ExpectedEPNodeAssignment { None,
                                      Some,
                                      All,
};

// The struct to hold some verification params for RunAndVerifyOutputsWithEP
struct EPVerificationParams {
  ExpectedEPNodeAssignment ep_node_assignment = ExpectedEPNodeAssignment::Some;

  // Some EP may use different rounding than ORT CPU EP, which may cause a bigger abs error than
  // the default of 1e-5f, especially for scenarios such as [Q -> Quantized op -> DQ]
  // Set this only if this is necessary
  float fp32_abs_err = 1e-5f;

  // optional graph verification function
  const std::function<void(const Graph&)>* graph_verifier{nullptr};
};

// Return number of nodes in the Graph and any subgraphs that are assigned to the specified execution provider
int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type);

// Run the model using the CPU EP to get expected output, comparing to the output when the 'execution_provider'
// is enabled. requires that at least one node is assigned to 'execution_provider'
void RunAndVerifyOutputsWithEP(const ORTCHAR_T* model_path,
                               const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params = EPVerificationParams());

// A helper function that takes in model_data
void RunAndVerifyOutputsWithEP(const std::string& model_data,
                               const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params = EPVerificationParams());

// Check equality of two shapes. Can successfully complete only if rank are equal and all dimensions are equal.
// The way we define dimension equality is that:
// 1. if both dimensions are symbolic, they are equal if their names are equal.
// 2. if both dimensions are not symbolic, they are equal if their values are equal.
// 3. if one dimension is symbolic and the other is not, they are not equal.
void CheckShapeEquality(const ONNX_NAMESPACE::TensorShapeProto* shape1,
                        const ONNX_NAMESPACE::TensorShapeProto* shape2);

// Create OrtValue on CPU copying from provided inputs.
template <typename T>
void CreateInputOrtValueOnCPU(gsl::span<const int64_t> dims, const std::vector<T>& value,
                              OrtValue* p_ortvalue, AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.CreatePreferredAllocators()[0];

  TensorShape shape(dims);
  assert(shape.Size() == static_cast<int64_t>(value.size()));
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  if (value.size() > 0 && !alloc) {  // using CPU allocator
    memcpy(p_tensor->MutableDataRaw(), value.data(), p_tensor->SizeInBytes());
  }

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

}  // namespace test
}  // namespace onnxruntime
