#include <iostream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/providers/cpu/wasm_execution_provider.h"
#include "example.h"

namespace onnxruntime {
namespace logging {

class MockSink : public ISink {
 public:
  MockSink() {}

  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override {
    std::ostringstream msg;
    msg << message.Message() << "\n";
    std::clog << msg.str();
  }
};

}  // namespace logging
}  // namespace onnxruntime

using namespace onnxruntime;

void Example::Run(const emscripten::val& model_jsarray) {
  std::vector<uint8_t> model_data = emscripten::vecFromJSArray<uint8_t>(model_jsarray);

  const std::string default_logger_id{"wasm_test"};
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::MockSink()}, logging::Severity::kWARNING, false,
      logging::LoggingManager::InstanceType::Default, &default_logger_id, -1);

  std::shared_ptr<Model> model;
  IOnnxRuntimeOpSchemaRegistryList schema_registry;
  auto status = Model::LoadFromBytes(static_cast<int>(model_data.size()),
                                     static_cast<void*>(model_data.data()),
                                     model,
                                     &schema_registry,
                                     logging::LoggingManager::DefaultLogger());
  if (!status.IsOK()) {
      std::cout << status.ErrorMessage() << std::endl;
      return;
  }

  auto& graph = model->MainGraph();

  const int tensor_dim = 5;
  const int input_num = 2;
  ONNX_NAMESPACE::TensorProto initializer_tensor[input_num];
  InitializedTensorSet initialized_tensor_set;

  const std::string input_names[] = {"A", "B"};
  for (int i = 0; i < input_num; i++) {
    initializer_tensor[i].set_name(input_names[i]);
    initializer_tensor[i].add_dims(tensor_dim);
    initializer_tensor[i].set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (int j = 0; j < tensor_dim; j++) {
      initializer_tensor[i].add_float_data((i + 1) * j);
    }
    initialized_tensor_set[input_names[i]] = &initializer_tensor[i];
  }

  std::vector<const Node*> nodes;
  for (auto& node : graph.Nodes()) {
    nodes.push_back(&node);
  }

  std::unique_ptr<WasmExecutionProvider> wasm_execution_provider =
      std::make_unique<WasmExecutionProvider>(WasmExecutionProviderInfo{false});
  OptimizerExecutionFrame::Info info(nodes, initialized_tensor_set, graph.ModelPath(), *wasm_execution_provider.get());
  std::vector<int> fetch_mlvalue_idxs{info.GetMLValueIndex("C")};
  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

  for (auto& node : graph.Nodes()) {
    auto kernel = info.CreateKernel(&node);
    OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, logging::LoggingManager::DefaultLogger());

    auto status = kernel->Compute(&op_kernel_context);
    if (!status.IsOK()) {
      std::cout << status.ErrorMessage() << std::endl;
    }
  }

  std::vector<OrtValue> fetches;
  frame.GetOutputs(fetches);
  auto& out_tensor = fetches[0].Get<Tensor>();
  const std::vector<float> found(out_tensor.template Data<float>(), out_tensor.template Data<float>() + tensor_dim);
  std::cout << "[";
  for (int i = 0; i < tensor_dim; i++) {
    std::cout << found[i] << ", " << std::endl;
  }
  std::cout << "]" << std::endl;
}
