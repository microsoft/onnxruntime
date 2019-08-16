// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "core/framework/execution_providers.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/session_state_initializer.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace std;

namespace onnxruntime {
namespace test {
class TestOpKernel : public OpKernel {
 public:
  TestOpKernel(const OpKernelInfo& p) : OpKernel(p) {}
  Status Compute(OpKernelContext* context) const {
    ORT_UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const {
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(done);
    return Status::OK();
  }
};

TEST(SessionStateTest, AddGetKernelTest) {
  concurrency::ThreadPool tp{"test", 1};
  ONNX_OPERATOR_SCHEMA(Variable)
      .SetDoc("Input variable.")
      .Output(0, "output_1", "docstr for output_1.", "tensor(int32)");
  ExecutionProviders execution_providers;
  SessionState s{execution_providers, true, &tp};

  onnxruntime::Model model("graph_1");
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;
  TypeProto output_type;
  output_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  onnxruntime::NodeArg output_arg("node_1_out_1", &output_type);
  outputs.push_back(&output_arg);
  onnxruntime::Node& node = graph.AddNode("node_1", "Variable", "node 1.", inputs, outputs);
  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK());
  KernelDef kernel_def;
  CPUExecutionProvider execution_provider{CPUExecutionProviderInfo{"CPUExecutionProvider"}};

  OpKernelInfo p_info(node, kernel_def, execution_provider, s.GetConstantInitializedTensors(),
                      s.GetOrtValueNameIdxMap(), s.GetFuncMgr(), s.GetDataTransferMgr());
  unique_ptr<TestOpKernel> p_kernel;
  p_kernel.reset(new TestOpKernel(p_info));
  size_t orig_num_outputs = p_kernel->Node().OutputDefs().size();
  std::cout << "node_idx: " << node.Index() << std::endl;

  s.SetGraphViewer(std::make_unique<GraphViewer>(graph));
  s.AddKernel(node.Index(), std::move(p_kernel));
  auto test_kernel = s.GetKernel(node.Index());
  std::cout << "orig: " << orig_num_outputs << " new: " << test_kernel->Node().OutputDefs().size() << std::endl;
  EXPECT_EQ(orig_num_outputs, test_kernel->Node().OutputDefs().size());
}

namespace {
class TestParam {
 public:
  int ir_version;
  bool enable_mem_pattern;
};
TestParam param_list[] = {{3, true}, {4, true}, {3, false}, {4, false}};
}  // namespace
class SessionStateTestP : public testing::TestWithParam<TestParam> {
};
// Test that we separate out constant and non-constant initializers correctly
TEST_P(SessionStateTestP, TestInitializerProcessing) {
  const TestParam& param = GetParam();
  concurrency::ThreadPool tp{"test", 1};

  std::string model_path = "testdata/optional_inputs_ir" + std::to_string(param.ir_version) + ".onnx";
  Status status;
  std::shared_ptr<Model> model;
  ASSERT_TRUE((status = Model::Load(model_path, model)).IsOK()) << status;
  Graph& graph = model->MainGraph();
  // take a copy as this gets cleared during session state initialization
  InitializedTensorSet initializers = graph.GetAllInitializedTensors();

  ExecutionProviders execution_providers;
  CPUExecutionProviderInfo epi{false};
  status = execution_providers.Add(onnxruntime::kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(epi));
  ASSERT_TRUE(status.IsOK()) << status;

  KernelRegistryManager krm;
  status = krm.RegisterKernels(execution_providers);
  ASSERT_TRUE(status.IsOK()) << status;

  SessionState session_state(execution_providers, param.enable_mem_pattern, &tp);
  SessionStateInitializer session_initializer(param.enable_mem_pattern, ToWideString(model_path), graph,
                                              session_state, execution_providers, krm);

  GraphPartitioner partitioner(krm, execution_providers);
  status = partitioner.Partition(graph, session_state.ExportDll(), session_state.GetMutableFuncMgr());
  ASSERT_TRUE(status.IsOK()) << status;

  status = session_initializer.CreatePlan(nullptr, nullptr, true);
  ASSERT_TRUE(status.IsOK()) << status;

  status = session_initializer.InitializeAndSave(nullptr);
  ASSERT_TRUE(status.IsOK()) << status;

  const auto& initialized_tensors = session_state.GetInitializedTensors();
  const auto& const_initialized_tensors = session_state.GetConstantInitializedTensors();

  ASSERT_EQ(initializers.size(), initialized_tensors.size())
      << "SessionState should have an entry for all initializers in Graph.";

  if (param.ir_version < 4) {
    ASSERT_EQ(initialized_tensors.size(), const_initialized_tensors.size())
        << "All initializers should be considered constant if IR version < 4.";
  } else {
    const auto& name_to_idx = session_state.GetOrtValueNameIdxMap();

    for (auto entry : initializers) {
      int idx;
      name_to_idx.GetIdx(entry.first, idx);

      bool found = initialized_tensors.find(idx) != initialized_tensors.cend();
      ASSERT_TRUE(found) << "Missing entry for " << entry.first << " in session state initialized tensors";

      if (graph_utils::IsConstantInitializer(graph, entry.first, false)) {
        found = const_initialized_tensors.find(idx) != const_initialized_tensors.cend();
        ASSERT_TRUE(found) << "Missing entry for " << entry.first << " in session state const initialized tensors";
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(SessionStateTests, SessionStateTestP,
                        testing::ValuesIn(param_list));
}  // namespace test
}  // namespace onnxruntime
