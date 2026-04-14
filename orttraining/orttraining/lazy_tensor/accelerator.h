// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace lazytensor {

// Type of JIT compilation result.
struct CompiledObject {
  // Callable to execute the computation represented by torch::jit::Graph.
  // It processes tensors across ORT and Pytorch and invokes "sess".
  std::function<std::vector<c10::IValue>(at::ArrayRef<c10::IValue>&)> code;
  // Session used in the "code" above.
  std::unique_ptr<onnxruntime::InferenceSession> sess;
};

// Custom JIT engine called by Pytorch.
class Accelerator {
 public:
  Accelerator(const torch::jit::Node* node)
      : subgraph_(node->g(torch::jit::attr::Subgraph)),
        input_types_(subgraph_->inputs().size()),
        output_types_(subgraph_->outputs().size()) {}
  // Execute a call to the torch::jit::Graph represented by "subgraph_".
  // This function could compile the graph and cache the result
  // for repeated uses.
  void Run(torch::jit::Stack& stack);
  // Determine if this node can be translated to ONNX.
  static bool Supported(const torch::jit::Node* node);

 private:
  // This function runs the "subgraph_" using PyTorch JIT executor
  // to get expected doutput schema. No ORT is involved.
  void ExampleRun(at::ArrayRef<c10::IValue> inputs);
  // This function calls "OrtRun" and "PytorchRun" to execute the graph
  // and compare their results. It may fail if their results are different
  // types or shapes.
  void DebugRun(torch::jit::Stack& stack);
  // Execute the graph represented by "subgraph_" using ORT.
  // Inputs are popped out from stack and outputs are pushed to stack.
  void OrtRun(torch::jit::Stack& stack);
  // Similar to "OrtRun" but uses Pytorch as executor.
  void PytorchRun(torch::jit::Stack& stack);
  // Create callable to execute "subgraph_" given "args" as inputs.
  // This calllable is cached for repeated uses.
  CompiledObject Compile(
      torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& args);
  // The graph to be compiled and executed by ORT.
  std::shared_ptr<torch::jit::Graph> subgraph_;
  // Previously compiled results.
  std::unordered_map<torch::jit::CompleteArgumentSpec, CompiledObject> cache_;
  // Types of the inputs (typed to IValue) we got when compile the subgraph.
  // Since the subgraph is compiled for these type, feeding
  // inputs with different types may fail.
  std::vector<c10::TypePtr> input_types_;
  // Types of the outputs (typed to IValue) by running the subgraph with
  // torch::jit::GraphExecutor.
  std::vector<c10::TypePtr> output_types_;
};
}  // namespace lazytensor
}  // namespace onnxruntime
