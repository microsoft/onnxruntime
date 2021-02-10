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

void Example::Run(int multiplier) {
  const std::string default_logger_id{"wasm_test"};
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::MockSink()}, logging::Severity::kWARNING, false,
      logging::LoggingManager::InstanceType::Default, &default_logger_id, -1);

  Model model("Wasm Test",
              false,
              ModelMetaData(),
              PathString(),
              IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 12}},
              {},
              logging::LoggingManager::DefaultLogger());
  auto& graph = model.MainGraph();

  const int tensor_dim = 10;
  const int input_num = 2;
  ONNX_NAMESPACE::TensorProto initializer_tensor[input_num];
  std::vector<std::unique_ptr<NodeArg>> inputs(input_num);
  std::vector<std::unique_ptr<NodeArg>> outputs(1);
  InitializedTensorSet initialized_tensor_set;

  ONNX_NAMESPACE::TypeProto tensor;
  tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(tensor_dim);

  for (int i = 0; i < input_num; i++) {
    std::string name("input_" + std::to_string(i));
    inputs[i] = std::make_unique<NodeArg>(name, &tensor);

    initializer_tensor[i].set_name(inputs[i]->Name());
    initializer_tensor[i].add_dims(tensor_dim);
    initializer_tensor[i].set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    for (int j = 0; j < tensor_dim; j++) {
      initializer_tensor[i].add_int32_data((i + 1) * j);
    }
    initialized_tensor_set[name] = &initializer_tensor[i];
  }
  outputs[0] = std::make_unique<NodeArg>("out", &tensor);

  std::vector<NodeArg*> tmp_inputs{inputs[0].get(), inputs[1].get()};
  std::vector<NodeArg*> tmp_outputs{outputs[0].get()};

  graph.AddNode("Test_Add", "Add", "Test Add", tmp_inputs, tmp_outputs);
  graph.Resolve();

  std::vector<const Node*> nodes;
  for (auto& node : graph.Nodes()) {
    nodes.push_back(&node);
  }

  std::unique_ptr<WasmExecutionProvider> wasm_execution_provider =
      std::make_unique<WasmExecutionProvider>(WasmExecutionProviderInfo{false});
  OptimizerExecutionFrame::Info info(nodes, initialized_tensor_set, graph.ModelPath(), *wasm_execution_provider.get());
  std::vector<int> fetch_mlvalue_idxs{info.GetMLValueIndex("out")};
  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

  for (auto& node : graph.Nodes()) {
    auto kernel = info.CreateKernel(&node);
    OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, logging::LoggingManager::DefaultLogger());

    auto st = kernel->Compute(&op_kernel_context);
    if (!st.IsOK()) {
      std::cout << st.ErrorMessage() << std::endl;
    }

    std::vector<OrtValue> fetches;
    frame.GetOutputs(fetches);
    auto& tensor = fetches[0].Get<Tensor>();
    const std::vector<int32_t> found(tensor.template Data<int32_t>(), tensor.template Data<int32_t>() + tensor_dim);
    std::cout << "[";
    for (int j = 0; j < tensor_dim; j++) {
      std::cout << (multiplier * j) << ", " << std::endl;
    }
    std::cout << "]" << std::endl;
  }
}
