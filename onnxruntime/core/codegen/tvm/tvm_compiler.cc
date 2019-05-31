// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <tvm/tvm.h>
#include <tvm/build_module.h>
#include "core/codegen/tvm/tvm_compiler.h"
namespace onnxruntime {

TVMGraph::TensorDescriptor::TensorDescriptor(MLDataType type, onnxruntime::ProviderType execution_provider_type, tvm::Tensor tvm_tensor) : tvm_tensor_(tvm_tensor) {
  if (execution_provider_type == onnxruntime::kCpuExecutionProvider) {
    ctx_.device_type = DLDeviceType::kDLCPU;
    ctx_.device_id = 0;
  } else {
    ORT_NOT_IMPLEMENTED("Non-cpu execution provider not supported on TVM now.");
  }

  if (DataTypeImpl::GetTensorType<double>() == type) {
    dtype_.code = kDLFloat;
    dtype_.bits = 64;
    dtype_.lanes = 1;
  } else {
    ORT_NOT_IMPLEMENTED("Non-double type not supported on TVM now.");
  }
}

class IdGenerator {
 public:
  IdGenerator() {}
  int GetNext() {
    return cur_++;
  }

 private:
  int cur_{0};
};

// This is a special compiler step for the test case that sum two 1-D tensors
static void Compile1DAddToTVM(const onnxruntime::Node& node, std::unordered_map<std::string, TVMGraph::TensorDescriptor>& tvm_tensors, onnxruntime::ProviderType execution_provider_type, IdGenerator& generator) {
  ORT_ENFORCE(node.OpType() == "Add");
  tvm::Array<tvm::Expr> shape;
  shape.push_back(tvm::var("n1"));

  tvm::Tensor t1;
  tvm::Tensor t2;
  auto it = tvm_tensors.find(node.InputDefs()[0]->Name());
  if (it == tvm_tensors.end()) {
    tvm_tensors[node.InputDefs()[0]->Name()] = TVMGraph::TensorDescriptor(
        DataTypeImpl::TypeFromProto(*node.InputDefs()[0]->TypeAsProto()),
        execution_provider_type,
        tvm::placeholder(shape, tvm::Float(64), "T" + std::to_string(generator.GetNext())));
  }
  t1 = tvm_tensors[node.InputDefs()[0]->Name()].tvm_tensor_;
  it = tvm_tensors.find(node.InputDefs()[1]->Name());
  if (it == tvm_tensors.end()) {
    tvm_tensors[node.InputDefs()[1]->Name()] = TVMGraph::TensorDescriptor(
        DataTypeImpl::TypeFromProto(*node.InputDefs()[1]->TypeAsProto()),
        execution_provider_type,
        tvm::placeholder(shape, tvm::Float(64), "T" + std::to_string(generator.GetNext())));
  }
  t2 = tvm_tensors[node.InputDefs()[1]->Name()].tvm_tensor_;

  tvm_tensors[node.OutputDefs()[0]->Name()] = TVMGraph::TensorDescriptor(
      DataTypeImpl::TypeFromProto(*node.InputDefs()[1]->TypeAsProto()),
      execution_provider_type,
      tvm::compute(
          t1->shape, [&t1, &t2](tvm::Expr i) {
            return t1[i] + t2[i];
          },
          "T" + std::to_string(generator.GetNext())));
}

TVMGraph CompileToTVM(const onnxruntime::Graph& graph, onnxruntime::ProviderType execution_provider_type) {
  TVMGraph result;
  std::unordered_map<std::string, TVMGraph::TensorDescriptor> tvm_tensors;
  IdGenerator generator;
  for (auto& node : graph.Nodes()) {
    Compile1DAddToTVM(node, tvm_tensors, execution_provider_type, generator);
  }

  for (auto& input : graph.GetInputs()) {
    result.inputs_.push_back(tvm_tensors[input->Name()]);
  }

  // check initializer
  for (auto& initializer : graph.GetAllInitializedTensors()) {
    result.inputs_.push_back(tvm_tensors[initializer.first]);
  }

  auto& output = graph.GetOutputs()[0];
  result.outputs_.push_back(tvm_tensors[output->Name()]);
  return result;
}
}  // namespace onnxruntime
