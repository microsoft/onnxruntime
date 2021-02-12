#include <iostream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/providers/cpu/cpu_execution_provider.h"
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

Example::Example() : cpu_execution_provider_{new CPUExecutionProvider{CPUExecutionProviderInfo{false}}} {
  const std::string default_logger_id{"wasm_test"};
  logging_manager_ = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::MockSink()}, logging::Severity::kWARNING, false,
      logging::LoggingManager::InstanceType::Default, &default_logger_id, -1);

  std::cout << "Example constructor" << std::endl;
};

bool Example::Load(const emscripten::val& model_jsarray) {
  std::vector<uint8_t> model_data = emscripten::vecFromJSArray<uint8_t>(model_jsarray);

  IOnnxRuntimeOpSchemaRegistryList schema_registry;
  auto status = Model::LoadFromBytes(static_cast<int>(model_data.size()),
                                     static_cast<void*>(model_data.data()),
                                     model_,
                                     &schema_registry,
                                     logging::LoggingManager::DefaultLogger());
  if (!status.IsOK()) {
      std::cout << status.ErrorMessage() << std::endl;
      return false;
  }

  auto& graph = model_->MainGraph();
  for (auto& node : graph.Nodes()) {
    nodes_.push_back(&node);
  }

  return true;
}

bool Example::Run() {
  auto& graph = model_->MainGraph();

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

  OptimizerExecutionFrame::Info info(nodes_, initialized_tensor_set, graph.ModelPath(), *cpu_execution_provider_.get());
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

  return true;
}
