// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "core/framework/execution_providers.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/finalize_session_state.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/util/thread_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "asserts.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;
using namespace std;
namespace onnxruntime {

namespace test {
class TestOpKernel : public OpKernel {
 public:
  TestOpKernel(const OpKernelInfo& p) : OpKernel(p) {
  }
  Status Compute(OpKernelContext* context) const override {
    ORT_UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const override {
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(done);
    return Status::OK();
  }
};
class SessionStateAddGetKernelTest : public testing::TestWithParam<int> {};

TEST_P(SessionStateAddGetKernelTest, AddGetKernelTest) {
  OrtThreadPoolParams to;
  to.thread_pool_size = GetParam();
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
  ONNX_OPERATOR_SCHEMA(Variable)
      .SetDoc("Input variable.")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");

  onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ExecutionProviders execution_providers;
  DataTransferManager dtm;
  profiling::Profiler profiler;
  SessionState s(graph, execution_providers, true, tp.get(), nullptr, dtm,
                 DefaultLoggingManager().DefaultLogger(), profiler);

  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  onnxruntime::NodeArg output_arg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  onnxruntime::Node& node = graph.AddNode("node_1", "Variable", "node 1.", inputs, outputs);
  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  auto kernel_def = KernelDefBuilder().SetName("Variable").Provider(kCpuExecutionProvider).SinceVersion(1, 10).Build();
  auto cpu_execution_provider = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));

  OpKernelInfo p_info(node, *kernel_def, *cpu_execution_provider, s.GetConstantInitializedTensors(),
                      s.GetOrtValueNameIdxMap(), s.GetFuncMgr(), s.GetDataTransferMgr());
  unique_ptr<TestOpKernel> p_kernel;
  p_kernel.reset(new TestOpKernel(p_info));
  size_t orig_num_outputs = p_kernel->Node().OutputDefs().size();
  std::cout << "node_idx: " << node.Index() << std::endl;

  ASSERT_STATUS_OK(execution_providers.Add(kCpuExecutionProvider, std::move(cpu_execution_provider)));
  KernelRegistryManager kernel_registry_manager;
  status = kernel_registry_manager.RegisterKernels(execution_providers);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  node.SetExecutionProviderType(kCpuExecutionProvider);
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ASSERT_STATUS_OK(kernel_registry->Register(KernelCreateInfo(
      std::move(kernel_def), [](const OpKernelInfo& info) -> OpKernel* { return new TestOpKernel(info); })));
  kernel_registry_manager.RegisterKernelRegistry(kernel_registry);
  s.CreateGraphInfo();
  ASSERT_STATUS_OK(s.CreateKernels(kernel_registry_manager));
  auto test_kernel = s.GetKernel(node.Index());
  std::cout << "orig: " << orig_num_outputs << " new: " << test_kernel->Node().OutputDefs().size() << std::endl;
  EXPECT_EQ(orig_num_outputs, test_kernel->Node().OutputDefs().size());
}

INSTANTIATE_TEST_SUITE_P(SessionStateTests, SessionStateAddGetKernelTest, testing::Values(0, 1));

namespace {
class TestParam {
 public:
  int ir_version;
  bool enable_mem_pattern;
  int thread_count;
};
TestParam param_list[] = {{3, true, 0}, {4, true, 0}, {3, false, 0}, {4, false, 0}, {3, true, 1}, {4, true, 1}, {3, false, 1}, {4, false, 1}};
}  // namespace
class SessionStateTestP : public testing::TestWithParam<TestParam> {};
// Test that we separate out constant and non-constant initializers correctly
TEST_P(SessionStateTestP, TestInitializerProcessing) {
  const TestParam& param = GetParam();
  OrtThreadPoolParams to;
  to.thread_pool_size = to.thread_pool_size;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);

  std::basic_ostringstream<ORTCHAR_T> oss;
  oss << ORT_TSTR("testdata/optional_inputs_ir") << param.ir_version << ORT_TSTR(".onnx");
  Status status;
  std::shared_ptr<Model> model;
  ASSERT_TRUE((status = Model::Load(oss.str(), model, nullptr, DefaultLoggingManager().DefaultLogger())).IsOK())
      << status;
  Graph& graph = model->MainGraph();
  // take a copy as this gets cleared during session state initialization
  InitializedTensorSet initializers = graph.GetAllInitializedTensors();

  ExecutionProviders execution_providers;
  CPUExecutionProviderInfo epi{false};
  status =
      execution_providers.Add(onnxruntime::kCpuExecutionProvider, onnxruntime::make_unique<CPUExecutionProvider>(epi));
  ASSERT_TRUE(status.IsOK()) << status;

  KernelRegistryManager krm;
  status = krm.RegisterKernels(execution_providers);
  ASSERT_TRUE(status.IsOK()) << status;

  DataTransferManager dtm;
  profiling::Profiler profiler;
  SessionState session_state(graph, execution_providers, param.enable_mem_pattern, tp.get(), nullptr, dtm,
                             DefaultLoggingManager().DefaultLogger(), profiler);

  GraphPartitioner partitioner(krm, execution_providers);
  status = partitioner.Partition(graph, session_state.ExportDll(), session_state.GetMutableFuncMgr());
  ASSERT_TRUE(status.IsOK()) << status;

  session_state.CreateGraphInfo();
  ASSERT_STATUS_OK(FinalizeSessionState(session_state, oss.str(), krm, nullptr, SessionOptions()));

  const auto& initialized_tensors = session_state.GetInitializedTensors();
  const auto& const_initialized_tensors = session_state.GetConstantInitializedTensors();

  ASSERT_EQ(initializers.size(), initialized_tensors.size())
      << "SessionState should have an entry for all initializers in Graph.";

  if (param.ir_version < 4) {
    ASSERT_EQ(initialized_tensors.size(), const_initialized_tensors.size())
        << "All initializers should be considered constant if IR version < 4.";
  } else {
    const auto& name_to_idx = session_state.GetOrtValueNameIdxMap();

    for (const auto& entry : initializers) {
      int idx;
      ASSERT_STATUS_OK(name_to_idx.GetIdx(entry.first, idx));

      bool found = initialized_tensors.find(idx) != initialized_tensors.cend();
      ASSERT_TRUE(found) << "Missing entry for " << entry.first << " in session state initialized tensors";

      if (graph_utils::IsConstantInitializer(graph, entry.first, false)) {
        found = const_initialized_tensors.find(idx) != const_initialized_tensors.cend();
        ASSERT_TRUE(found) << "Missing entry for " << entry.first << " in session state const initialized tensors";
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SessionStateTests, SessionStateTestP, testing::ValuesIn(param_list));

class PrePackingTestOpKernel : public OpKernel {
 public:
  PrePackingTestOpKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override {
    ORT_UNUSED_PARAMETER(context);
    return Status::OK();
  }

  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override {
    ORT_UNUSED_PARAMETER(tensor);
    ORT_UNUSED_PARAMETER(input_idx);
    is_packed = true;
    return Status::OK();
  }
};

class SessionStatePrepackingTest : public testing::TestWithParam<bool> {};
TEST_P(SessionStatePrepackingTest, PrePackingTest) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
  ONNX_OPERATOR_SCHEMA(PrePackingTest)
      .SetDoc("Faking Node for PrePacking")
      .Input(0, "Input_0", "input 0", "tensor(float)")
      .Input(1, "Input_1", "input 1", "tensor(float)")
      .Output(0, "output_0", "docstr for output_0.", "tensor(float)");

  onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
  // construct graph
  auto& graph = model.MainGraph();

  // node creation and placement
  TypeProto type;
  type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  std::vector<onnxruntime::NodeArg*> inputs;
  onnxruntime::NodeArg input_0_arg("node_0_input_0", &type);
  onnxruntime::NodeArg input_1_arg("node_0_input_1", &type);
  inputs.push_back(&input_0_arg);
  inputs.push_back(&input_1_arg);

  std::vector<onnxruntime::NodeArg*> outputs;
  onnxruntime::NodeArg output_arg("node_0_output_0", &type);
  outputs.push_back(&output_arg);

  onnxruntime::Node& node = graph.AddNode("node_0", "PrePackingTest", "node 0", inputs, outputs);
  node.SetExecutionProviderType(kCpuExecutionProvider);

  // add an initializer
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("node_0_input_1");
  graph.AddInitializedTensor(tensor);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());

  ExecutionProviders execution_providers;
  auto cpu_execution_provider = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  execution_providers.Add(kCpuExecutionProvider, std::move(cpu_execution_provider));

  DataTransferManager dtm;
  profiling::Profiler profiler;
  SessionState session_state(graph,
                             execution_providers,
                             true, /*enable_mem_pattern*/
                             tp.get(),
                             nullptr, /*inter_op_thread_pool*/
                             dtm,
                             DefaultLoggingManager().DefaultLogger(),
                             profiler);

  KernelRegistryManager kernel_registry_manager;
  status = kernel_registry_manager.RegisterKernels(execution_providers);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  auto kernel_def = KernelDefBuilder().SetName("PrePackingTest").Provider(kCpuExecutionProvider).SinceVersion(1).Build();
  ASSERT_STATUS_OK(kernel_registry->Register(
      KernelCreateInfo(std::move(kernel_def),
                       [](const OpKernelInfo& info) -> OpKernel* { return new PrePackingTestOpKernel(info); })));
  kernel_registry_manager.RegisterKernelRegistry(kernel_registry);

  SessionOptions sess_options;
  sess_options.use_prepacking = GetParam();
  ASSERT_STATUS_OK(FinalizeSessionState(session_state,
                                        std::basic_string<PATH_CHAR_TYPE>() /*graph_loc*/,
                                        kernel_registry_manager,
                                        nullptr /*parent_node*/,
                                        sess_options));
  const auto& const_initialized_tensors = session_state.GetConstantInitializedTensors();
  // check prepacking
  ASSERT_EQ(const_initialized_tensors.size(), size_t(sess_options.use_prepacking ? 0 : 1));
}

INSTANTIATE_TEST_SUITE_P(SessionStateTests, SessionStatePrepackingTest, testing::Values(true, false));

}  // namespace test
}  // namespace onnxruntime
