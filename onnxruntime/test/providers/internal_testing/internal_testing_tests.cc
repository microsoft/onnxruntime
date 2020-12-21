// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/framework/utils.h"
#include "core/session/inference_session.h"

#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/providers/internal_testing/internal_testing_execution_provider.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {

namespace test {

static void CreateSession(const SessionOptions& so, std::unique_ptr<InferenceSessionWrapper>& session,
                          const ORTCHAR_T* model_path = ORT_TSTR("testdata/mnist.onnx"),  // arbitrary test model
                          bool enable_custom_ep = true,
                          const std::unordered_set<std::string>* override_supported_ops = nullptr) {
  session = onnxruntime::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  // set supported ops to ops that are ideally found consecutively in the model.
  // we can say the EP potentially handles them all, but can also test removing handling of one or more ops
  // at runtime to simulate a lower spec device where not all ops can be handled. this allows us to test
  // that we can revert ops back to the CPU implementation successfully
  const std::unordered_set<std::string> default_supported_ops{"Conv", "Add", "Relu", "MaxPool"};
  const std::unordered_set<std::string>* supported_ops = override_supported_ops ? override_supported_ops
                                                                                : &default_supported_ops;

  if (enable_custom_ep) {
    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
        onnxruntime::make_unique<InternalTestingExecutionProvider>(*supported_ops)));
  }

  ASSERT_STATUS_OK(session->Load(model_path));
  ASSERT_STATUS_OK(session->Initialize());
}

static void ExecuteMnist(InferenceSessionWrapper& session, bool custom_ep_enabled) {
  // validate that we can execute the model. the dummy internal testing EP just creates empty output so the
  // values in the output aren't relevant. all we care about is that we can execute the model and produce output.
  OrtValue ml_value_x;
  TensorShape input_shape{1, 1, 28, 28};
  std::vector<float> input(input_shape.Size(), 1.f);

  CreateMLValue<float>(input_shape.GetDims(), input.data(), OrtMemoryInfo(), &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("Input3", ml_value_x));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Plus214_Output_0");
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));

  if (custom_ep_enabled) {
    // check that the output is all zeros. the dummy EP produces output of the correct shape with all zeros, so any
    // downstream operations should still result in zeros for this model
    // OR it should equal the bias in the final Add operation, which is in the Parameter194 initializer
    const auto& t = fetches[0].Get<Tensor>();
    const auto data = t.DataAsSpan<float>();

    int idx = 0;
    const auto& session_state = session.GetSessionState();
    ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx("Parameter194", idx));
    const auto& initializer = session_state.GetConstantInitializedTensors().at(idx);
    const auto expected = initializer.Get<Tensor>().DataAsSpan<float>();

    ASSERT_THAT(data, ::testing::ContainerEq(expected));
  }
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(InternalTestingEP, TestSaveAndLoadOrtModel) {
  const ORTCHAR_T* ort_model_path = ORT_TSTR("testdata/mnist.test_output.ort");

  //
  // First load the onnx format model and save as an ORT model.
  // This should preserve the nodes the custom EP can handle.
  //
  std::unique_ptr<InferenceSessionWrapper> session;
  SessionOptions so;
  so.optimized_model_filepath = ort_model_path;

  CreateSession(so, session);
  // this graph should include the original nodes that the custom EP will take at runtime
  auto num_nodes = session->GetGraph().NumberOfNodes();

  //
  // Second, load the ORT format model with just the CPU EP to make sure it can be executed. This tests that the
  // fallback to the CPU EP kernel hashes works.
  //
  std::unique_ptr<InferenceSessionWrapper> session2;

  so.optimized_model_filepath.clear();
  bool enable_custom_ep = false;

  CreateSession(so, session2, ort_model_path, enable_custom_ep);
  const auto& graph1 = session2->GetGraph();
  // model should have all the original nodes and we should be able to execute with the fallback to CPU EP
  ASSERT_EQ(graph1.NumberOfNodes(), num_nodes);
  ExecuteMnist(*session2, enable_custom_ep);
  session2 = nullptr;

  //
  // Finally, load the ORT format model with the custom EP enabled. This tests that we support runtime compilation
  // for the ORT format model.
  //
  enable_custom_ep = true;
  CreateSession(so, session2, ort_model_path, enable_custom_ep);
  const auto& graph2 = session2->GetGraph();
  // model should be able to be loaded, and we should compile using custom ep. that will result in one node for the
  // custom EP (with Conv/Add/Relu/MaxPool), one for a reshape, and one for the fused MatMul+Add.
  ASSERT_EQ(graph2.NumberOfNodes(), 3);
  ExecuteMnist(*session2, enable_custom_ep);
}

TEST(InternalTestingEP, PreventSaveOfModelWithCompiledOps) {
  const ORTCHAR_T* ort_model_path = ORT_TSTR("testdata/mnist.ort");

  // make sure we can't save a model with compiled ops. input/output model format doesn't matter
  SessionOptions so;
  so.optimized_model_filepath = ORT_TSTR("invalid_model.ort");

  auto session = onnxruntime::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  const std::unordered_set<std::string> supported_ops{"Conv", "Add", "Relu", "MaxPool"};
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(
      onnxruntime::make_unique<InternalTestingExecutionProvider>(supported_ops)));

  ASSERT_STATUS_OK(session->Load(ort_model_path));
  auto status = session->Initialize();
  ASSERT_FALSE(status.IsOK()) << "Initialize should have failed when trying to save model with compiled kernels";
  ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Unable to serialize model as it contains compiled nodes"));
}
#endif  // !defined(ORT_MINIMAL_BUILD)

// test to validate a minimal build
TEST(InternalTestingEP, TestLoadOrtModel) {
  const ORTCHAR_T* ort_model_path = ORT_TSTR("testdata/mnist.ort");

  std::unique_ptr<InferenceSessionWrapper> session;
  bool enable_custom_ep = true;

  CreateSession(SessionOptions{}, session, ort_model_path, enable_custom_ep);
  ExecuteMnist(*session, enable_custom_ep);
}

// test that is the custom EP cannot take all nodes due to device limitations
// that we fallback to the CPU implementations and can execute the model
TEST(InternalTestingEP, TestLoadOrtModelWithReducedOpCoverage) {
  const ORTCHAR_T* ort_model_path = ORT_TSTR("testdata/mnist.ort");
  const std::unordered_set<std::string> supported_ops{"Conv", "Add", "Relu" /*, "MaxPool"*/};

  std::unique_ptr<InferenceSessionWrapper> session;
  bool enable_custom_ep = true;

  CreateSession(SessionOptions{}, session, ort_model_path, enable_custom_ep, &supported_ops);

  const auto& graph = session->GetGraph();
  // Conv+Add gets fused by level 1 optimizer into single node. The 'Conv'/'Add'/'Relu' nodes should be compiled and
  // handled by the custom EP. fallback to CPU for MaxPool.
  ASSERT_EQ(graph.NumberOfNodes(), 6);
  const auto& func_mgr = session->GetSessionState().GetFuncMgr();
  NodeComputeInfo* compute_func = nullptr;

  // the generated op type should have a hash for the model based on the model path
  const std::string expected_op_type_prefix = "InternalTestingEP_9611636968429821767_";
  int compiled_node_num = 0;

  for (const auto& node : graph.Nodes()) {
    EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
        << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
    if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
      EXPECT_EQ(node.OpType(), expected_op_type_prefix + std::to_string(compiled_node_num++));
    }
  }

  ExecuteMnist(*session, enable_custom_ep);
}

// count nodes assigned to the test EP and make sure they all have valid compute funcs
static int CountAndValidateAssignedNodes(const Graph& current_graph,
                                         const std::unordered_set<std::string>& supported_ops,
                                         const FuncManager& func_mgr) {
  int count = 0;

  for (const auto& node : current_graph.Nodes()) {
    EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
        << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
    if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
      NodeComputeInfo* compute_func = nullptr;
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
      ++count;
    }

    if (node.ContainsSubgraph()) {
      for (const auto& entry : node.GetSubgraphs()) {
        count += CountAndValidateAssignedNodes(*entry, supported_ops, func_mgr);
      }
    }
  }

  return count;
}

// Test model that contains a subgraph. This model has a Loop and an If so multiple layers of nested subgraphs.
// There are Add nodes in the Loop and If subgraphs so we should see the custom EP taking nodes at both these levels.
TEST(InternalTestingEP, TestModelWithSubgraph) {
  const ORTCHAR_T* ort_model_path = ORT_TSTR("testdata/ort_github_issue_4031.onnx.ort");
  const std::unordered_set<std::string> supported_ops{"Add"};

  std::unique_ptr<InferenceSessionWrapper> session;
  bool enable_custom_ep = true;

  CreateSession(SessionOptions{}, session, ort_model_path, enable_custom_ep, &supported_ops);

  const auto& graph = session->GetGraph();
  const auto& func_mgr = session->GetSessionState().GetFuncMgr();

  int num_replaced_nodes = CountAndValidateAssignedNodes(graph, supported_ops, func_mgr);

  // One Add node in the Loop. One Add node in each branch of the If inside the Loop body
  ASSERT_EQ(num_replaced_nodes, 3);

  OrtValue ml_value;

  // this is a bit of a hack. the correct output is the input value + 2, so if we start with -2 the result is 0.
  // the output from fused nodes using the testing EP is always 0, so we should match the expected output this way
  // as we replace all the Add nodes with something that returns 0.
  // RunAndVerifyOutputsWithEP checks that nodes are assigned to the EP so we know it's being used to execute the model
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, {-2.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));
  // compare outputs from CPU EP vs custom EP
  RunAndVerifyOutputsWithEP(ort_model_path,
                            "InternalTestingEP.TestModelWithSubgraph",
                            onnxruntime::make_unique<InternalTestingExecutionProvider>(supported_ops),
                            feeds);
}

}  // namespace test
}  // namespace onnxruntime
