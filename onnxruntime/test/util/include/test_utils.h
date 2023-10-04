// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>
#include <string>
#include <variant>
#include <vector>

#include "core/common/gsl.h"
#include "core/framework/execution_provider.h"
#include "core/framework/framework_common.h"
#include "core/framework/ort_value.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/onnxruntime_c_api.h"

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

// Verify equality of two output tensors.
void VerifyOutput(const std::string& output_name,
                  const Tensor& expected_tensor,
                  const Tensor& tensor,
                  float fp32_abs_err);

// Return number of nodes in the Graph and any subgraphs that are assigned to the specified execution provider
int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type);

// Verify the assignment of nodes to the EP specified by `provider_type`.
void VerifyEPNodeAssignment(const Graph& graph, const std::string& provider_type,
                            ExpectedEPNodeAssignment assignment);

using ModelPathOrBytes = std::variant<std::basic_string_view<ORTCHAR_T>,
                                      gsl::span<const std::byte>>;

// Run the model using the CPU EP to get expected output, comparing to the output when the 'execution_provider'
// is enabled.
void RunAndVerifyOutputsWithEP(ModelPathOrBytes model_path_or_bytes,
                               std::string_view log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params = EPVerificationParams());

// Tests model loading only.
// This can be used to test EPs in builds where only loading (and not running) of a model is supported.
// Calls `check_graph` on the graph of the loaded model.
void TestModelLoad(ModelPathOrBytes model_path_or_bytes,
                   std::unique_ptr<IExecutionProvider> execution_provider,
                   const std::function<void(const Graph&)>& check_graph);

// Tests model loading only.
// The check graph function verifies the expected EP node assignment.
inline void TestModelLoad(ModelPathOrBytes model_path_or_bytes,
                          std::unique_ptr<IExecutionProvider> execution_provider,
                          ExpectedEPNodeAssignment expected_node_assignment) {
  auto check_node_assignment =
      [provider_type = execution_provider->Type(), expected_node_assignment](const Graph& graph) {
        VerifyEPNodeAssignment(graph, provider_type, expected_node_assignment);
      };
  TestModelLoad(model_path_or_bytes, std::move(execution_provider), check_node_assignment);
}

// Check equality of two shapes. Can successfully complete only if rank are equal and all dimensions are equal.
// The way we define dimension equality is that:
// 1. if both dimensions are symbolic, they are equal if their names are equal.
// 2. if both dimensions are not symbolic, they are equal if their values are equal.
// 3. if one dimension is symbolic and the other is not, they are not equal.
void CheckShapeEquality(const ONNX_NAMESPACE::TensorShapeProto* shape1,
                        const ONNX_NAMESPACE::TensorShapeProto* shape2);

// Create OrtValue on CPU copying from provided inputs.
template <typename T>
OrtValue CreateInputOrtValueOnCPU(gsl::span<const int64_t> dims, gsl::span<const T> value,
                                  AllocatorPtr alloc = nullptr) {
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

  OrtValue ort_value;
  ort_value.Init(p_tensor.release(),
                 DataTypeImpl::GetType<Tensor>(),
                 DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return ort_value;
}

}  // namespace test
}  // namespace onnxruntime
