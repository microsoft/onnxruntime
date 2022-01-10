#include "accelerator.h"
#include <stack>
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <ATen/core/functional.h>

namespace py = pybind11;
namespace aten = torch::jit::aten;
namespace prim = torch::jit::prim;

bool Accelerator::supported(const torch::jit::Node* node) {
  switch (node->kind()) {
    case aten::relu:
    case aten::mul:
    case aten::gt:
    case aten::eq:
    case prim::Constant:
    case aten::threshold_backward:
      std::cout << "[compiler.cc] Support " << *node  << std::endl;
      return true;
    default:
      std::cout << "[compiler.cc] Not support " << *node << std::endl;
      return false;
  }
}

void Accelerator::run(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<torch::jit::Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, num_inputs);

  std::cout << "JIT sub-graph: " << std::endl;
  std::cout << *subgraph_ << std::endl;
  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  torch::jit::CompleteArgumentSpec spec{false, at::ArrayRef<c10::IValue>(inputs)};
  if (cache_.find(spec) == cache_.end()) {
    cache_[spec] = compile(inputs);
  }

  // Run the compiled function!
  auto outputs = cache_[spec](inputs);

  torch::jit::drop(stack, num_inputs);

  for (auto& output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(c10::IValue(var));
  }
}

CompiledCode Accelerator::compile(
    at::ArrayRef<c10::IValue>& inputs) {
  // First we run through some checks to make sure the inputs are Tensors and
  // that the implied semantics are pointwise.

  TORCH_CHECK(inputs.size(), "Need at least one input.");
  for (const auto& input : inputs) {
    TORCH_CHECK(input.isTensor(), "Compiler can only handle Tensor inputs.");
  }
  auto size = inputs[0].toTensor().numel();
  for (const auto& input : inputs) {
    TORCH_CHECK(
        input.toTensor().numel() == size,
        "Compiler can only handle pointwise operations without broadcasting.");
  }

  {
    std::cout << "[ext] gil_scoped_acquire" << std::endl;
    pybind11::gil_scoped_acquire guard{};
    std::cout << "[ext] create to_onnx" << std::endl;
    pybind11::function to_onnx =
        pybind11::reinterpret_borrow<pybind11::function>(   // cast from 'object' to 'function - use `borrow` (copy) or `steal` (move)
            pybind11::module::import("torch.onnx.utils").attr("_optimize_graph")  // import method "min_rosen" from python "module"
        );
    std::cout << "[ext] print JIT graph in Python:" << std::endl;
    pybind11::print(subgraph_);
    std::cout << "[ext] call to_onnx" << std::endl;
    auto result = to_onnx(subgraph_, ::torch::onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK);
    std::cout << "[ext] print ONNX graph:" << std::endl;
    pybind11::print(subgraph_);
  }

  // This function wraps the function pointer we bound our assembly to
  // Adheres to the CompiledCode interface defined in compiler.h
  auto compiled_func = [this](at::ArrayRef<c10::IValue>& inputs) {
    // __value__ is the symbol of arena[__value__] tensor.
    std::map<torch::jit::Value*, c10::IValue> arena;
    for (auto value : subgraph_->inputs()) {
      arena[value] = inputs[value->offset()];
    }

    for (auto node : subgraph_->nodes()) {
      switch (node->kind()) {
        case aten::relu: {
          std::cout << "[compiler.cc] see aten::relu" << std::endl;
          auto x = arena[node->inputs()[0]].toTensor().contiguous();
          auto y = at::relu(x);
          arena[node->outputs()[0]] = y;
          break;
        }
        case aten::mul: {
          std::cout << "[compiler.cc] see aten::mul" << std::endl;
          auto x = arena[node->inputs()[0]].toTensor().contiguous();
          auto y = arena[node->inputs()[1]].toTensor().contiguous();
          auto z = at::mul(x, y);
          arena[node->outputs()[0]] = z;
          break;
        }
        case aten::gt: {
          std::cout << "[compiler.cc] see aten::gt" << std::endl;
          auto x = arena[node->inputs()[0]].toTensor().contiguous();
          int y = arena[node->inputs()[1]].toInt();
          arena[node->outputs()[0]] = at::gt(x, y);
          break;
        }
        case aten::eq: {
          std::cout << "[compiler.cc] see aten::eq" << std::endl;
          auto x = arena[node->inputs()[0]].toTensor().contiguous();
          auto y = arena[node->inputs()[0]].toTensor().contiguous();
          auto z = at::eq(x, y);
          arena[node->outputs()[0]] = z;
          break;
        }
        case aten::type_as: {
          std::cout << "[compiler.cc] see aten::type_as" << std::endl;
          auto x = arena[node->inputs()[0]].toTensor().contiguous();
          auto y = arena[node->inputs()[1]].toTensor();
          auto z = x.to(y.options());
          arena[node->outputs()[0]] = z;
          break;
        }
        case prim::Constant: {
          std::cout << "[compiler.cc] see prim::Constant" << std::endl;
          arena[node->outputs()[0]] = torch::jit::toIValue(node->outputs()[0]).value();
          break;
        }
        case aten::threshold_backward: {
          std::cout << "[compiler.cc] see aten::threshold_backward" << std::endl;
          auto x = arena[node->inputs()[0]].toTensor().contiguous();
          auto y = arena[node->inputs()[1]].toTensor();
          //auto z = arena[node->inputs()[2]].to<at::Scalar>();
          auto z = arena[node->inputs()[2]].toTensor().contiguous();
          auto z_data = z.data_ptr<int64_t>();
          std::cout << "To call\n";
          auto w = at::_ops::threshold_backward::call(x, y, at::Scalar(*z_data));
          std::cout << "Done call\n";
          arena[node->outputs()[0]] = w;
          break;
        }
      }
    }

    std::vector<c10::IValue> outputs;
    for (auto value : subgraph_->outputs()) {
      outputs.push_back(arena[value]);
    }

    return outputs;
  };

  return compiled_func;
}
