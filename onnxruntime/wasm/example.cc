#include <iostream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_frame.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/session_state.h"
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

void Example::Run() {
  std::cout << "Starting an example..." << std::endl;

  const std::string default_logger_id{"wasm_test"};
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<logging::ISink>{new logging::MockSink()}, logging::Severity::kWARNING, false,
      logging::LoggingManager::InstanceType::Default, &default_logger_id, /*default_max_vlog_level*/ -1);

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

  auto& node = graph.AddNode("Test_Concat", "Concat", "Test Concat", tmp_inputs, tmp_outputs);
  node.AddAttribute("axis", 1ll);
  node.SetExecutionProviderType(kCpuExecutionProvider);

  graph.Resolve();

  std::unique_ptr<WasmExecutionProvider> wasm_execution_provider =
      std::make_unique<WasmExecutionProvider>(WasmExecutionProviderInfo{false});
  DataTransferManager data_transfer_mgr;
  data_transfer_mgr.RegisterDataTransfer(std::make_unique<CPUDataTransfer>());

  std::shared_ptr<KernelRegistry> kernel_registry = wasm_execution_provider->GetKernelRegistry();

  profiling::Profiler profiler;
  ExecutionProviders execution_providers;
  execution_providers.Add(kCpuExecutionProvider, std::move(wasm_execution_provider));

  KernelRegistryManager kernel_registry_manager;
  kernel_registry_manager.RegisterKernelRegistry(kernel_registry);
  auto status = kernel_registry_manager.RegisterKernels(execution_providers);

  SessionState session_state(graph,
                             execution_providers,
                             false,
                             nullptr,
                             nullptr,
                             data_transfer_mgr,
                             logging::LoggingManager::DefaultLogger(),
                             profiler);
  session_state.FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager, {}, nullptr, false);

  ExecutionFrame execution_frame(std::vector<int>{},
                                 std::vector<OrtValue>{},
                                 std::vector<int>{},
                                 std::vector<OrtValue>{},
                                 std::unordered_map<size_t, IExecutor::CustomAllocator>{},
                                 session_state);

  for (auto& node : graph.Nodes()) {
    const auto& kci = session_state.GetNodeKernelCreateInfo(node.Index());
    onnxruntime::ProviderType exec_provider_name = node.GetExecutionProviderType();
    const IExecutionProvider& exec_provider = *execution_providers.Get(exec_provider_name);

    auto op_kernel = kernel_registry_manager.CreateKernel(node, exec_provider, session_state, kci);
    if (op_kernel != nullptr) {
      std::cout << "Op name: " << op_kernel->KernelDef().OpName() << std::endl
                << "Domain:  " << op_kernel->KernelDef().Domain() << std::endl
                << "Index:   " << op_kernel->Node().Index() << std::endl;
    }

    OpKernelContext op_kernel_context(&execution_frame,
                                      op_kernel.get(),
                                      nullptr,
                                      logging::LoggingManager::DefaultLogger());

    auto status = op_kernel->Compute(&op_kernel_context);
    if (!status.IsOK()) {
      std::cout << "Can't compute" << std::endl;
    }
  }

  std::cout << "Done" << std::endl;
}
