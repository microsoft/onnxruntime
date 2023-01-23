// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/lazy_tensor/accelerator.h"
// C++
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
// Pytorch.
#include <torch/csrc/onnx/onnx.h>
#include <torch/torch.h>
// ORT friends.
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/execution_providers.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "python/onnxruntime_pybind_state_common.h"
// Lazy tensor specific.
#include "orttraining/lazy_tensor/bridge.h"
#include "orttraining/lazy_tensor/cuda_tool.h"
#include "orttraining/lazy_tensor/debug.h"
#include "orttraining/lazy_tensor/flags.h"

namespace onnxruntime {

namespace python {
Environment& GetTrainingORTEnv();
}

namespace lazytensor {

namespace py = pybind11;
namespace aten = torch::jit::aten;
namespace prim = torch::jit::prim;

bool IsFusable(const torch::jit::Node* node) {
  // This function checks the fusion restriction inside
  // mergeNodeIntoGroup(...). When selecting onnx-supported aten ops,
  // we need to call this function to make sure they are fusable
  // in mergeNodeIntoGroup(...).

  // Not all inputs are fusable. For example, the fuser
  // expects all inputs are tensors with a few exceptions.
  // This flag denotes if we find an unsupported input.
  bool found_not_fusable = false;
  for (auto input : node->inputs()) {
    if (input->type()->isSubtypeOf(*c10::TensorType::get())) {
      continue;
    } else if (
        (input->type()->isSubtypeOf(*c10::FloatType::get()) &&
         input->node()->kind() != torch::jit::prim::Constant) ||
        (node->kind() == torch::jit::aten::_grad_sum_to_size &&
         input->type()->isSubtypeOf(*c10::ListType::ofInts()))) {
      continue;
    } else if (
        input->type()->isSubtypeOf(*c10::IntType::get()) &&
        input->node()->kind() != torch::jit::prim::Constant) {
      continue;
    } else{
      if (input->node()->kind() == torch::jit::prim::Constant) {
        continue;
      }
      found_not_fusable = true;
    }
  }
  return !found_not_fusable;
}

bool Accelerator::Supported(const torch::jit::Node* node) {
  if (!node) {
    return false;
  }

  switch (node->kind()) {
    // TODO(wechi): add as many ops as possible.
    case aten::embedding:
    case aten::tanh:
    case aten::slice:
    case aten::bmm:
    case aten::gelu:
    case aten::native_layer_norm:
    case aten::native_dropout:
    case aten::expand:
    case aten::add:
    case aten::convolution:
    case aten::reshape:
    case aten::max_pool2d_with_indices:
    case aten::_log_softmax:
    case aten::relu:
    case aten::mul:
    case aten::sub:
    case aten::div:
    case aten::gt:
    case aten::lt:
    case aten::eq:
    case aten::sqrt:
    case aten::permute:
    case aten::mm:
    case aten::ne:
    case aten::abs:
    case aten::max:
    case aten::min: {
      if (DumpAtenOpHistory()) {
        std::cout << "Supported op: "
                  << ToString(*node) << std::endl;
      }
      return IsFusable(node);
    }
    default: {
      if (DumpAtenOpHistory()) {
        std::cout << "Unsupported op: "
                  << ToString(*node) << std::endl;
        // To check sub-graph in specific symbol such as prim::TensorExprGroup,
        // uncomment and extend the following code.
        //
        // if (node->kind() == prim::TensorExprGroup || node->kind() == prim::FallbackGraph) {
        //   auto subgraph = node->g(torch::jit::attr::Subgraph);
        //   std::cout << "Node's subgraph: " << *subgraph;
        // }
      }

      return false;
    }
  }
}

void Accelerator::OrtRun(torch::jit::Stack& stack) {
#ifdef USE_CUDA
  NvtxRange range(__func__);
#endif
  // Uncomment the following if you want to see the
  // sub-graph in Nsys profiling result. This is useful
  // for debugging.
  //
  // NvtxRange range_graph(subgraph_->toString(true));
  if (DumpGraph()) {
    std::cout << "[ORT,Graph]\n"
              << subgraph_->toString(true);
  }

  // Get these inputs from the stack.
  at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, subgraph_->inputs().size());
  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  // Compile a callable to execute "subgraph_" on the inputs.
  // If such input schema appears before, we can reuse a cached compiled callable.
  torch::jit::CompleteArgumentSpec spec{false, inputs};
  if (cache_.find(spec) == cache_.end()) {
    cache_.emplace(spec, Compile(spec, inputs));
  }

  if (DumpInputsOutputs()) {
    std::cout << "[ORT,Input] " << ToString(inputs) << std::endl;
  }

  // Run the compiled function!
  auto outputs = cache_[spec].code(inputs);

  // Discard used inputs.
  torch::jit::drop(stack, inputs.size());

  // Return results to caller.
  for (auto& output : outputs) {
    stack.push_back(output);
  }

  if (DumpInputsOutputs()) {
    at::ArrayRef<c10::IValue> outputs = torch::jit::last(stack, subgraph_->outputs().size());
    std::cout << "[ORT,Output] " << ToString(outputs) << std::endl;
  }
}

void Accelerator::PytorchRun(torch::jit::Stack& stack) {
  DynamicSettings::GetInstance().SetOnnxFusionFlag(false);
#ifdef USE_CUDA
  NvtxRange range(__func__);
#endif
  if (DumpGraph()) {
    std::cout << "[Pytorch,Graph]\n"
              << subgraph_->toString(true);
  }
  if (DumpInputsOutputs()) {
    at::ArrayRef<c10::IValue> inputs = torch::jit::last(
        stack, subgraph_->inputs().size());
    std::cout << "[PyTorch,Input] " << ToString(inputs) << std::endl;
  }

  torch::jit::GraphExecutor executor(subgraph_, "");
  executor.run(stack);

  if (DumpInputsOutputs()) {
    at::ArrayRef<c10::IValue> outputs = torch::jit::last(
        stack, subgraph_->outputs().size());
    std::cout << "[PyTorch,Output] " << ToString(outputs) << std::endl;
  }
  DynamicSettings::GetInstance().SetOnnxFusionFlag(true);
}

void Accelerator::DebugRun(torch::jit::Stack& stack) {
#ifdef USE_CUDA
  NvtxRange range(__func__);
#endif
  torch::jit::Stack copy;
  copy = stack;
  OrtRun(stack);
  PytorchRun(copy);
  ORT_ENFORCE(CompareStack(stack, copy),
              "ORT and Pytorch must generate the same results "
              "but tensor types, shapes or content are different. "
              "Use, e.g., LORT_RELATIVE_TOLERANCE=1e-3 and "
              "LORT_ABSOLUTE_TOLERANCE=1e-4 "
              "to increase the content tolerance, if "
              "the difference is due to numerical errors.");
}

void Accelerator::Run(torch::jit::Stack& stack) {
  const auto run_type = RunType();
  if (run_type == "debug") {
    // Run both ORT and Pytorch to execute the subgraph
    // and compare their output types and shapes.
    DebugRun(stack);
  } else if (run_type == "ort") {
    OrtRun(stack);
  } else if (run_type == "pytorch") {
    PytorchRun(stack);
  } else {
    ORT_THROW("Unknown run type: ", run_type);
  }
}

static void CheckArgs(
    const at::ArrayRef<c10::IValue>& inputs) {
  // TODO(wechi): remove this check.
#ifdef USE_CUDA
  NvtxRange range(__func__);
#endif
  TORCH_CHECK(inputs.size(), "Need at least one input.");
  for (const auto& input : inputs) {
    TORCH_CHECK(input.isTensor() || input.isScalar(), "Compiler can only handle Tensor or Scalar inputs.");
  }
}

// Store input types in sub-graph so that
// ONNX exporter can use them. Input types
// are required when executing ONNX model
// in ORT.
// TODO(wechi): Allow ORT to accept models without
// input types. Then, we can remove this function.
static void SetArgTypes(
    const at::ArrayRef<c10::IValue>& inputs,
    std::shared_ptr<torch::jit::Graph> graph) {
  TORCH_CHECK(graph->inputs().size() == inputs.size(),
              "Number of provided inputs must match captured sub-graph's schema.");
  for (size_t i = 0; i < graph->inputs().size(); ++i) {
    auto input_symbol = graph->inputs()[i];
    auto input_value = inputs[i];
    if (!input_value.isTensor()) {
      // The allowed IR components in ONNX exporter and Pytorch
      // are a little different. I am not confident to fill
      // types other than tensor, because of the ambiguous scalar
      // representations in Pytorch.
      continue;
    }
    input_symbol->setType(input_value.type());
  }
}

// ONNX exporter is written in Python, so
// this function may call some Python functions.
// Be aware of GIL issue.
// The returned value is the path to exported
// ONNX file.
static std::string ExportToOnnx(
    std::shared_ptr<torch::jit::Graph> graph,
    const at::ArrayRef<c10::IValue>& args) {
#ifdef USE_CUDA
  NvtxRange range(__func__);
#endif
  // ONNX exporter modifies the graph in-place, so we
  // need to clone it to avoid interaction between
  // Pytorch's JIT mechanism and ONNX graph.
  std::shared_ptr<torch::jit::Graph> new_subgraph(graph->copyUnique().release());
  // Acquire GIL since Python is not multi-threading.
  pybind11::gil_scoped_acquire guard{};
  // Retrieve Python exporter function.
  pybind11::function export_to_onnx =
      pybind11::reinterpret_borrow<pybind11::function>(
          pybind11::module::import("onnxruntime.training.experimental.exporter")
              .attr("_export_jit_graph_to_onnx_model_proto"));
  // Fill types up. The sub-graphp from LazyTensor doesn't
  // contain input shapes.
  SetArgTypes(args, new_subgraph);
  // Execute Python function.
  auto result = export_to_onnx(new_subgraph, ::torch::onnx::OperatorExportTypes::ONNX);
  return result.cast<std::string>();
}

// Create an empty session object.
// Models will be loaded later.
static std::unique_ptr<onnxruntime::InferenceSession> CreateSession() {
#ifdef USE_CUDA
  NvtxRange range(__func__);
#endif
  // Environment shared by all sessions.
  static onnxruntime::Environment& pybind_default_env = onnxruntime::python::GetTrainingORTEnv();
  // All sessions use the same config.
  static onnxruntime::SessionOptions sess_opts;
  return std::make_unique<onnxruntime::InferenceSession>(sess_opts, pybind_default_env);
}

static OrtDevice CheckAndGetTensorDevice(const at::ArrayRef<c10::IValue>& values) {
  // This memory info must be shared by all tensors;
  // for example, all tensors on CPU or all on a specific GPU.
  // When all values are not tensors, we assume CPU device.
  // c10::Device's index is default to -1.
  c10::Device unique_tensor_device(c10::DeviceType::CPU);
  bool assigned = false;
  for (auto value : values) {
    if (!value.isTensor()) {
      continue;
    }
    auto tensor = value.toTensor();
    if (assigned) {
      // A device has been recorded, so we compare
      // it with the current tensor's device.
      TORCH_CHECK(unique_tensor_device == tensor.device(),
                  "All tensors must be on the same device.");
    } else {
      // Record the 1st tensor device.
      unique_tensor_device = tensor.device();
      assigned = true;
    }
  }
  return CreateOrtDevice(unique_tensor_device);
}

// Initialize empty session with ONNX model.
static void InitializeSession(
    const OrtDevice device,
    const std::string& serialized_model,
    onnxruntime::InferenceSession& sess) {
  // Add EPs.
#ifdef USE_CUDA
  NvtxRange range(__func__);
  // When CUDA is enabled, some CUDA-only graph graph fusions are enabled.
  // If we don't add CUDA EP, ONNX Runtime may throw even when running MNIST.
  // Information needed to construct CUDA execution providers.
  // Note that CUDA is enabled by setting LTC_TS_CUDA=1 when running LazyTensor.
  if (device.Type() == OrtDevice::GPU) {
    ORT_THROW_IF_ERROR(sess.RegisterExecutionProvider(
        CUDAExecutionProviderPool::GetInstance().GetExecutionProvider(device.Id())));
  }
#endif
  ORT_THROW_IF_ERROR(sess.Load(serialized_model.data(), serialized_model.size()));
  ORT_THROW_IF_ERROR(sess.Initialize());
}

void Accelerator::ExampleRun(at::ArrayRef<c10::IValue> inputs) {
#ifdef USE_CUDA
  NvtxRange range(__func__);
#endif
  torch::jit::Stack stack;
  for (auto input : inputs) {
    stack.push_back(input);
  }

  // Run graph and store input and output types.
  // 1. Store input types.
  // This check prevent some unexpected modification on input_types_.
  ORT_ENFORCE(input_types_.size() == inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    c10::TypePtr type = inputs.at(i).type();
    // LazyTensor should only capture graph with numerical inputs and outputs.
    // If this assumption is broken, please use Accelerator::Supported to filter
    /// out unsupported types and operators.
    ORT_ENFORCE(type->isSubtypeOf(*c10::TensorType::get()) ||
                    type->isSubtypeOf(*c10::NumberType::get()),
                "ONNX only support tensor, float, int, bool as graph's input types");
    input_types_.at(i) = type;
  }

  // 2. Run graph.
  torch::jit::GraphExecutor executor(subgraph_, "");
  executor.run(stack);

  // 3. Store output types.
  at::ArrayRef<c10::IValue> outputs = torch::jit::last(stack, subgraph_->outputs().size());
  ORT_ENFORCE(output_types_.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    c10::TypePtr type = outputs.at(i).type();
    ORT_ENFORCE(type->isSubtypeOf(*c10::TensorType::get()) ||
                    type->isSubtypeOf(*c10::NumberType::get()),
                "ONNX only support tensor, float, int, bool as graph's output types. But got ",
                type->str());
    output_types_.at(i) = type;
  }
}

CompiledObject Accelerator::Compile(
    torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& args) {
  CheckArgs(args);
  DynamicSettings::GetInstance().SetOnnxFusionFlag(false);
  ExampleRun(args);
  DynamicSettings::GetInstance().SetOnnxFusionFlag(true);
  // Storage of compilation.
  CompiledObject compiled;
  // Create an empty session.
  compiled.sess = CreateSession();
  // Let's get the empty session and initialize it.
  onnxruntime::InferenceSession& sess = *compiled.sess;
  // Export subgraph_ to ONNX.
  // The exporter should never fail. If it does, please modify
  // Accelerator::Supported to filter out unsupported operators.
  const std::string serialized_model = ExportToOnnx(subgraph_, args);
  // Memory info for all tensors.
  // Assume all inputs are on the same device.
  OrtDevice shared_device = CheckAndGetTensorDevice(args);
  // Load ONNX model into session, register
  // EPs and finally initialize session.
  InitializeSession(shared_device, serialized_model, sess);

  onnxruntime::RunOptions run_options;
  std::vector<std::string> feed_names;
  std::vector<std::string> fetch_names;

  for (auto node_arg : *sess.GetModelInputs().second) {
    feed_names.push_back(node_arg->Name());
  }
  for (auto node_arg : *sess.GetModelOutputs().second) {
    fetch_names.push_back(node_arg->Name());
  }

  // Duplicate device info for putting output tensors on the shared device.
  std::vector<OrtDevice> fetches_device_info(fetch_names.size(), shared_device);

  // Create a callable which feeds inputs to ORT
  // session's Run(...) and returns outputs.
  auto code = [this, run_options,
               feed_names, fetch_names,
               fetches_device_info, &sess](at::ArrayRef<c10::IValue>& args) {
    // Inputs of ORT session.
    std::vector<OrtValue> feeds;
    // Outputs of ORT session.
    std::vector<OrtValue> fetches;

    {
#ifdef USE_CUDA
      NvtxRange range("Prepare inputs");
#endif
      // Prepare inputs.
      const auto num_inputs = subgraph_->inputs().size();
      for (size_t i = 0; i < num_inputs; ++i) {
        // The value can be either tensor or scalar.
        // Scalar is a tensor with empty shape vector.
        // Create ORT tensor from Pytorch tensor without copy.
        if (args.at(i).isScalar()) {
          // Scalar.
          // ORT_ENFORCE(subgraph_->inputs().at(i)->type()->kind() == c10::TypeKind::TensorType);
          feeds.push_back(CreateOrtScalarValue(args.at(i).toScalar()));
        } else if (args.at(i).isTensor()) {
          // Tensor.
          ORT_ENFORCE(subgraph_->inputs().at(i)->type()->kind() == c10::TypeKind::TensorType);
          feeds.push_back(CreateOrtTensorValue(args.at(i).toTensor()));
        } else {
          // Looks like LTC only passes scalars and tensors into backend, so we don't care
          // other types for now.
          ORT_THROW("Only tensor inputs are supported.");
        }
      }
    }

    {
#ifdef USE_CUDA
      NvtxRange range("Call sess.Run");
#endif
      // Inputs are ready. Let's run ORT.
      ORT_THROW_IF_ERROR(sess.Run(
          run_options,
          feed_names, feeds,
          fetch_names, &fetches, &fetches_device_info));
    }

    std::vector<c10::IValue> outputs;
    {
#ifdef USE_CUDA
      NvtxRange range("Convert outputs");
#endif
      // Convert ORT output to Pytorch format.
      for (size_t i = 0; i < fetches.size(); ++i) {
        // Get the expected type of the i-th output.
        const c10::TypePtr type = output_types_.at(i);
        // Convert ORTValue to IValue.
        if (type->isSubtypeOf(*c10::TensorType::get())) {
          ORT_ENFORCE(fetches.at(i).IsTensor(), "Only ORT tensor can be translated to Pytorch tensor.");
          auto value = CreateC10IvalueTensor(fetches.at(i));
          auto expected_scalar_type = output_types_.at(i)->cast<c10::TensorType>()->scalarType().value();
          outputs.push_back(value.toTensor().to(expected_scalar_type));
        } else if (type->isSubtypeOf(*c10::NumberType::get())) {
          // ORT represents scalar as tensor without shape.
          ORT_ENFORCE(fetches.at(i).IsTensor(), "Only ORT tensor can be translated to Pytorch scalar.");
          auto value = CreateC10IvalueScalar(fetches.at(i));
          outputs.push_back(value);
        } else {
          ORT_ENFORCE(false, "Unsupported c10::Type ", type->str());
        }
      }
    }

    return outputs;
  };

  compiled.code = code;
  return compiled;
}
}  // namespace lazytensor
}  // namespace onnxruntime
