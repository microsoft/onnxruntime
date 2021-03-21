// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/eager/ort_kernel_invoker.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/graph/model.h"
#include "core/framework/op_kernel.h"
#include "core/session/ort_env.h"

namespace onnxruntime {

std::once_flag init_flag;
std::unique_ptr<Environment> ort_env;

ORTInvoker::ORTInvoker(std::unique_ptr<IExecutionProvider> execution_provider)
  : execution_provider_(std::move(execution_provider)) {
  std::call_once(init_flag, [&]{
    std::string logger_id{"ORTInvoker"};
    auto logging_manager = onnxruntime::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
      logging::Severity::kVERBOSE, false,
      logging::LoggingManager::InstanceType::Default,
      &logger_id);
    Environment::Create(std::move(logging_manager), ort_env);
  });
}

common::Status ORTInvoker::Invoke(const std::string& op_name,
                                  //optional inputs / outputs?
                                  const std::vector<OrtValue>& inputs,
                                  std::vector<OrtValue>& outputs,
                                  const NodeAttributes* attributes,
                                  const std::string domain,
                                  const int /*version*/) {
  //create a graph
  Model model("test", false, ort_env->GetLoggingManager()->DefaultLogger());

  std::vector<onnxruntime::NodeArg*> input_args;
  std::vector<onnxruntime::NodeArg*> output_args;

  Graph& graph = model.MainGraph();
  std::unordered_map<std::string, OrtValue> initializer_map;
  size_t i = 0;

  for (auto input : inputs) {
    std::string name = "I" + std::to_string(i++);
    const Tensor& input_tensor = input.Get<Tensor>();
    ONNX_NAMESPACE::TypeProto input_tensor_type;
    input_tensor_type.mutable_tensor_type()->set_elem_type(input_tensor.GetElementType());
    auto& arg = graph.GetOrCreateNodeArg(name, &input_tensor_type);
    input_args.push_back(&arg);
    initializer_map[name] = input;
  }

  for (i = 0; i < outputs.size(); ++i) {
    auto& arg = graph.GetOrCreateNodeArg("O" + std::to_string(i), nullptr);
    output_args.push_back(&arg);
  }

  auto& node = graph.AddNode("node1", op_name, "eager mode node", input_args, output_args, attributes, domain);
  ORT_RETURN_IF_ERROR(graph.Resolve());

  if (!execution_provider_) {
    ORT_THROW("Execution provider is nullptr");
  }

  node.SetExecutionProviderType(execution_provider_->Type());
  std::vector<const Node*> frame_nodes{&node};

  OptimizerExecutionFrame::Info info({&node}, initializer_map, graph.ModelPath(), *execution_provider_);
  auto kernel = info.CreateKernel(&node);
  if (!kernel) {
    ORT_THROW("Could not find kernel");
  }

  std::vector<int> fetch_mlvalue_idxs;
  for (const auto* node_out : node.OutputDefs()) {
    fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
  }

  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs, outputs);
  OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, ort_env->GetLoggingManager()->DefaultLogger());
  ORT_RETURN_IF_ERROR(kernel->Compute(&op_kernel_context));

  return frame.GetOutputs(outputs);
}

}  // namespace onnxruntime
