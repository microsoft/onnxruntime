// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/common/logging/logging.h"

#include "core/common/common.h"
#include "core/graph/function_ir.h"

namespace onnxruntime {

using NamedFunctionMap = std::multimap<std::string, std::unique_ptr<FunctionIR> >;
using SparseTensorNamesSet = std::unordered_set<std::reference_wrapper<const std::string>,
                                                std::hash<std::string>, std::equal_to<std::string>>;
    /**
@class GraphContext
Class representing the context for a onnx graph, including:
1. the functions 
2. the initializers
*/
class GraphContext {
 public:
  const FunctionIR& GetMainFunction() const;

  FunctionIR* GetMutableMainFunction();

  const FunctionIR& GetFunction(const std::string& name, const std::string& domain) const;

  FunctionIR* GetMutableFunction(const std::string& name, const std::string& domain);

  Status AddFunction(const std::string& name, const std::string& domain, std::unique_ptr<FunctionIR> p_function);

#if !defined(ORT_MINIMAL_BUILD)
  /** Replaces the initializer tensor with the same name as the given initializer tensor.
  The replacement initializer tensor must have the same type and shape as the existing initializer tensor.

  Note: This currently has linear time complexity. There is room for improvement but it would likely require changes to
  how initializer tensors are stored and tracked.
  */
  Status ReplaceInitializedTensor(const ONNX_NAMESPACE::TensorProto& new_initializer);
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  /** Add an initializer tensor to the Graph. */
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto);
#endif
  /** Remove the initializer tensor with the provided name from the Graph. */
  void RemoveInitializedTensor(const std::string& tensor_name);

  void CleanAllInitializedTensors() noexcept;

  const ONNX_NAMESPACE::TensorProto* GetInitializer(const std::string& name) const;

  const InitializedTensorSet& GetAllInitializedTensors() const noexcept {
    return name_to_initial_tensor_;
  }

  bool IsInitializedTensor(const std::string& name) const {
    return name_to_initial_tensor_.count(name) > 0;
  }

  const SparseTensorNamesSet& GetSparseTensorNames() const {
    return sparse_tensor_names_;
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  /** Check if a given name is a sparse initializer's name in the model
   * we currently convert sparse_initializer field in the model into dense Tensor instances.
   * However, we sometimes want to check if this initializer was stored as sparse in the model.
   */
  bool IsSparseInitializer(const std::string& name) const;
#endif

  GraphContext(ONNX_NAMESPACE::GraphProto* graph_proto, const Path& model_path, Graph* graph, Version ir_version, bool is_subgraph, const logging::Logger& logger);

  //TODO!!!
  // temporary solution for serialization, need to revisit later.
  friend class Graph; 

private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphContext);

  //TODO: re-consider the initializer tensor managenent,
  // for example, whether we can delay the loading of initializer
  // currently we just refer to the TensorProto* in graph proto
  InitializedTensorSet name_to_initial_tensor_;

  // TODO: I really don't like the solution that using graph_proto_ as storage.
  // All the information in the graph proto are duplciated to the IR, except the initializer buffers
  // And the "ToProto" method in Model is defined as return a copy of ModelProto, which means
  // there is no sense to keep a cached graph proto in our IR. It just increase the complexity.
  // Will try to fix it later
  ONNX_NAMESPACE::GraphProto* graph_proto_; // it is hold by the ModelProto in Model

  std::unordered_set<std::reference_wrapper<const std::string>,
                     std::hash<std::string>, std::equal_to<std::string>>
      sparse_tensor_names_;

  NamedFunctionMap name_to_functions_;
};
}  // namespace onnxruntime
