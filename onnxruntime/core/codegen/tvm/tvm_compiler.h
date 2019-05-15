// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>
#include <tvm/build_module.h>
#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/graph/function.h"
#include "core/graph/constants.h"
#include "core/graph/graph_viewer.h"
namespace onnxruntime {

//TODO: this is just initial design to represent TVM Graph, to make the basic test work.
//We may need to revisit it later to finialize it.
struct TVMGraph {
  struct TensorDescriptor {
    tvm::Tensor tvm_tensor_;
    DLContext ctx_;
    DLDataType dtype_;

   public:
    TensorDescriptor(MLDataType type, onnxruntime::ProviderType execution_provider_type, tvm::Tensor tvm_tensor);

    TensorDescriptor() = default;
  };
  std::vector<TensorDescriptor> inputs_;
  std::vector<TensorDescriptor> outputs_;
};

//TODO: compile a onnxruntime graph to tvm's tensor expression is a common logic for all hardwares
//onnxruntime framework should provide this functionality to executionp providers.
//We will need to register how to compiler it for each node. A detail design is needed.
//Here for testing we just provide the functionality that compile add 1D tensors.
TVMGraph CompileToTVM(const onnxruntime::Graph& graph, onnxruntime::ProviderType execution_provider_type);
}  // namespace onnxruntime
