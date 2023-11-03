// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(REDUCED_OPS_BUILD)  // may not work with excluded op kernel implementations

#include "core/common/logging/logging.h"
#include "core/common/span_utils.h"
#include "core/framework/utils.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_env.h"

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

// defined in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

using namespace onnxruntime::internal_testing_ep;

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

static Status CreateSession(const SessionOptions& so, std::unique_ptr<InferenceSessionWrapper>& session,
                            const ORTCHAR_T* model_path = ORT_MODEL_FOLDER "mnist.onnx",  // arbitrary test model
                            bool enable_custom_ep = true,
                            const std::unordered_set<std::string>* override_supported_ops = nullptr) {
  session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  // set supported ops to ops that are ideally found consecutively in the model.
  // we can say the EP potentially handles them all, but can also test removing handling of one or more ops
  // at runtime to simulate a lower spec device where not all ops can be handled. this allows us to test
  // that we can revert ops back to the CPU implementation successfully
  const std::unordered_set<std::string> default_supported_ops{"Conv", "Add", "Relu", "MaxPool"};
  const std::unordered_set<std::string>* supported_ops = override_supported_ops ? override_supported_ops
                                                                                : &default_supported_ops;

  if (enable_custom_ep) {
    ORT_RETURN_IF_ERROR(session->RegisterExecutionProvider(
        std::make_unique<InternalTestingExecutionProvider>(*supported_ops)));
  }

  ORT_RETURN_IF_ERROR(session->Load(model_path));
  ORT_RETURN_IF_ERROR(session->Initialize());
  return Status::OK();
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

    ASSERT_TRUE(SpanEq(data, expected));
  }
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(InternalTestingEP, TestSaveAndLoadOrtModel) {
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "mnist.internal_testing_ep.test_output.ort";

  //
  // First load the onnx format model and save as an ORT model.
  // This should preserve the nodes the custom EP can handle.
  //
  std::unique_ptr<InferenceSessionWrapper> session;
  SessionOptions so;
  so.optimized_model_filepath = ort_model_path;

  ASSERT_STATUS_OK(CreateSession(so, session));
  // this graph should include the original nodes that the custom EP will take at runtime
  auto num_nodes = session->GetGraph().NumberOfNodes();

  //
  // Second, load the ORT format model with just the CPU EP to make sure it can be executed. This tests that the
  // fallback to the CPU EP works.
  //
  std::unique_ptr<InferenceSessionWrapper> session2;

  so.optimized_model_filepath.clear();
  bool enable_custom_ep = false;

  ASSERT_STATUS_OK(CreateSession(so, session2, ort_model_path, enable_custom_ep));
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
  ASSERT_STATUS_OK(CreateSession(so, session2, ort_model_path, enable_custom_ep));
  const auto& graph2 = session2->GetGraph();
  // model should be able to be loaded, and we should compile using custom ep. that will result in one node for the
  // custom EP (with Conv/Add/Relu/MaxPool), one for a reshape, and one for the fused MatMul+Add.
  ASSERT_EQ(graph2.NumberOfNodes(), 3);
  ExecuteMnist(*session2, enable_custom_ep);
}

TEST(InternalTestingEP, PreventSaveOfModelWithCompiledOps) {
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "mnist.internal_testing_ep.ort";

  // make sure we can't save a model with compiled ops. input/output model format doesn't matter
  SessionOptions so;
  so.optimized_model_filepath = ORT_TSTR("invalid_model.ort");

  auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  const std::unordered_set<std::string> supported_ops{"Conv", "Add", "Relu", "MaxPool"};
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(
      std::make_unique<InternalTestingExecutionProvider>(supported_ops)));

  ASSERT_STATUS_OK(session->Load(ort_model_path));
  auto status = session->Initialize();
  ASSERT_FALSE(status.IsOK()) << "Initialize should have failed when trying to save model with compiled kernels";
  ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Unable to serialize model as it contains compiled nodes"));
}

// the internal NHWC operators are only included as part of contrib ops currently. as the EP requests the NHWC
// version of the ONNX operator when matching a static kernel, those are required.
#if !defined(DISABLE_CONTRIB_OPS)
TEST(InternalTestingEP, TestMixOfStaticAndCompiledKernels) {
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "transform/fusion/conv_relu_opset12.onnx";

  SessionOptions so;
  InferenceSessionWrapper session(so, GetEnvironment());

  const std::unordered_set<std::string> supported_ops{"Conv", "Add", "Relu", "MaxPool"};
  auto ep = std::make_unique<InternalTestingExecutionProvider>(supported_ops,
                                                               std::unordered_set<std::string>{},
                                                               DataLayout::NHWC);
  ep->EnableStaticKernels();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));

  ASSERT_STATUS_OK(session.Load(ort_model_path));
  ASSERT_STATUS_OK(session.Initialize());

  TensorShape input_shape_x{1, 1, 7, 7};
  TensorShape input_shape_w{1, 1, 1, 1};
  std::vector<float> input_x(input_shape_x.Size(), 1.f);
  std::vector<float> input_w(input_shape_w.Size(), 1.f);
  OrtValue ml_value_x;
  OrtValue ml_value_w;
  CreateMLValue<float>(input_shape_x.GetDims(), input_x.data(), OrtMemoryInfo(), &ml_value_x);
  CreateMLValue<float>(input_shape_w.GetDims(), input_w.data(), OrtMemoryInfo(), &ml_value_w);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("W", ml_value_w));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Z");
  std::vector<OrtValue> fetches;

  auto status = session.Run(feeds, output_names, &fetches);
  // Error message should come from the Conv implementation with the statically registered kernel
  ASSERT_THAT(status.ErrorMessage(),
              ::testing::HasSubstr("Non-zero status code returned while running Conv node. Name:'Conv' "
                                   "Status Message: TODO: add NHWC implementation here."));
}

TEST(InternalTestingEP, TestNhwcConversionOfStaticKernels) {
  // the internal NHWC domain supports opset 11 and later
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "squeezenet/model_opset11.onnx";

  SessionOptions so;
  // set this if you want to manually inspect the optimized model
  // so.optimized_model_filepath = ORT_MODEL_FOLDER "squeezenet/model.test_output.onnx";
  InferenceSessionWrapper session(so, GetEnvironment());

  const std::unordered_set<std::string> supported_ops{"Conv", "Clip"};
  auto ep = std::make_unique<InternalTestingExecutionProvider>(supported_ops,
                                                               std::unordered_set<std::string>{},
                                                               DataLayout::NHWC);
  ep->EnableStaticKernels();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));

  ASSERT_STATUS_OK(session.Load(ort_model_path));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();

  // all Conv nodes should have been converted to NHWC versions and
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      ASSERT_EQ(node.Domain(), kMSInternalNHWCDomain);
    }
  }

  TensorShape input_shape_x{1, 3, 224, 224};
  std::vector<float> input_x(input_shape_x.Size(), 1.f);
  OrtValue ml_value_x;
  CreateMLValue<float>(input_shape_x.GetDims(), input_x.data(), OrtMemoryInfo(), &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("data_0", ml_value_x));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("softmaxout_1");
  std::vector<OrtValue> fetches;

  auto status = session.Run(feeds, output_names, &fetches);
  ASSERT_THAT(status.ErrorMessage(),
              ::testing::HasSubstr("Non-zero status code returned while running Conv node. Name:'Conv' "
                                   "Status Message: TODO: add NHWC implementation here."));
}

// This test can be deprecated now as the code logic has been changed so the model is not applicable
// TEST(InternalTestingEP, TestRegisterAllocatorHandlesUsageInMultipleSessions) {
//}

// make sure allocators returned by SessionState::GetAllocator are valid when IExecutionProvider::ReplaceAllocator
// is used. if something is off InferenceSession::Initialize will fail.
TEST(InternalTestingEP, TestReplaceAllocatorDoesntBreakDueToLocalAllocatorStorage) {
  OrtMemoryInfo mem_info("Replacement", OrtAllocatorType::OrtDeviceAllocator);
  AllocatorPtr replacement_alloc = std::make_shared<CPUAllocator>(mem_info);
  OrtEnv& env = *(OrtEnv*)(*ort_env);

  ASSERT_STATUS_OK(env.RegisterAllocator(replacement_alloc));

  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1"));
  InferenceSessionWrapper session(so, env.GetEnvironment());

  const std::unordered_set<std::string> supported_ops{"Conv", "Clip"};

  std::vector<std::shared_ptr<IExecutionProvider>> eps{
      std::make_shared<InternalTestingExecutionProvider>(supported_ops, std::unordered_set<std::string>{},
                                                         DataLayout::NHWC),
      std::make_shared<CPUExecutionProvider>(CPUExecutionProviderInfo{})};

  for (const auto& ep : eps) {
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(ep));
  }

  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "squeezenet/model.onnx";
  ASSERT_STATUS_OK(session.Load(ort_model_path));
  ASSERT_STATUS_OK(session.Initialize());

  ASSERT_EQ(replacement_alloc, session.GetAllocator(OrtMemoryInfo())) << "Allocators registered from Env should have the highest priority";
}

#endif  // !defined(DISABLE_CONTRIB_OPS)
#endif  // !defined(ORT_MINIMAL_BUILD)

// test to validate a minimal build
TEST(InternalTestingEP, TestLoadOrtModel) {
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "mnist.internal_testing_ep.ort";

  std::unique_ptr<InferenceSessionWrapper> session;
  bool enable_custom_ep = true;

  ASSERT_STATUS_OK(CreateSession(SessionOptions{}, session, ort_model_path, enable_custom_ep));
  ExecuteMnist(*session, enable_custom_ep);
}

// test that if the custom EP cannot take all nodes due to device limitations
// that we fallback to the CPU implementations and can execute the model
TEST(InternalTestingEP, TestLoadOrtModelWithReducedOpCoverage) {
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "mnist.internal_testing_ep.ort";
  const std::unordered_set<std::string> supported_ops{"Conv", "Add", "Relu" /*, "MaxPool"*/};

  std::unique_ptr<InferenceSessionWrapper> session;
  bool enable_custom_ep = true;

  ASSERT_STATUS_OK(CreateSession(SessionOptions{}, session, ort_model_path, enable_custom_ep, &supported_ops));

  const auto& graph = session->GetGraph();
  // Conv+Add gets fused by level 1 optimizer into single node. The 'Conv'/'Add'/'Relu' nodes should be compiled and
  // handled by the custom EP. fallback to CPU for MaxPool.
  ASSERT_EQ(graph.NumberOfNodes(), 6);
  auto& func_mgr = const_cast<SessionState&>(session->GetSessionState()).GetMutableFuncMgr();
  const NodeComputeInfo* compute_func = nullptr;

  // the generated op type should have a hash for the model based on the model path
  const std::string expected_op_type_prefix = "InternalTestingEP_9611636968429821767_";

  for (const auto& node : graph.Nodes()) {
    EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
        << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
    if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
      EXPECT_THAT(node.OpType(), ::testing::StartsWith(expected_op_type_prefix));
    }
  }

  ExecuteMnist(*session, enable_custom_ep);
}

// count nodes assigned to the test EP and make sure they all have valid compute funcs
static int CountAndValidateAssignedNodes(const Graph& current_graph,
                                         const std::unordered_set<std::string>& supported_ops,
                                         FuncManager& func_mgr) {
  int count = 0;

  for (const auto& node : current_graph.Nodes()) {
    EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
        << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
    if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
      const NodeComputeInfo* compute_func = nullptr;
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
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "ort_github_issue_4031.onnx.ort";
  const std::unordered_set<std::string> supported_ops{"Add"};

  std::unique_ptr<InferenceSessionWrapper> session;
  bool enable_custom_ep = true;

  ASSERT_STATUS_OK(CreateSession(SessionOptions{}, session, ort_model_path, enable_custom_ep, &supported_ops));

  const auto& graph = session->GetGraph();
  auto& func_mgr = const_cast<SessionState&>(session->GetSessionState()).GetMutableFuncMgr();

  int num_replaced_nodes = CountAndValidateAssignedNodes(graph, supported_ops, func_mgr);

  // One Add node in the Loop. One Add node in each branch of the If inside the Loop body
  ASSERT_EQ(num_replaced_nodes, 3);

  OrtValue ml_value;

  // this is a bit of a hack. the correct output is the input value + 2, so if we start with -2 the result is 0.
  // the output from fused nodes using the testing EP is always 0, so we should match the expected output this way
  // as we replace all the Add nodes with something that returns 0.
  // RunAndVerifyOutputsWithEP checks that nodes are assigned to the EP so we know it's being used to execute the model
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], {1}, {-2.f},
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("state_var_in", ml_value));
  // compare outputs from CPU EP vs custom EP
  RunAndVerifyOutputsWithEP(ort_model_path,
                            "InternalTestingEP.TestModelWithSubgraph",
                            std::make_unique<InternalTestingExecutionProvider>(supported_ops),
                            feeds);
}

// A custom InternalTestingEP extension
// This is to testing execution fall back to CPU EP if Compile fails, for ORT format
// This EP will take an additional compile_failure_ops
// If in Compile() any nodes in the partition is also in compile_failure_ops
// The Compile will fail
class CompileFailureTestExecutionProvider : public InternalTestingExecutionProvider {
 public:
  CompileFailureTestExecutionProvider(const std::unordered_set<std::string>& supported_ops,
                                      const std::unordered_set<std::string>& compile_failure_ops);
  virtual ~CompileFailureTestExecutionProvider() = default;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

 private:
  std::unordered_set<std::string> compile_failure_ops_;
};

CompileFailureTestExecutionProvider::CompileFailureTestExecutionProvider(
    const std::unordered_set<std::string>& supported_ops,
    const std::unordered_set<std::string>& compile_failure_ops)
    : InternalTestingExecutionProvider(supported_ops),
      compile_failure_ops_(compile_failure_ops) {}

Status CompileFailureTestExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                                                    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes) {
    // If any nodes in this partition is also in compile_failure_ops_, the Compile will fail
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);
    for (const auto& node : graph_viewer.Nodes()) {
      if (compile_failure_ops_.find(node.OpType()) != compile_failure_ops_.end()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "CompileFailureTestExecutionProvider::Compile failed for node: ", node.Name());
      }
    }
  }

  return InternalTestingExecutionProvider::Compile(fused_nodes, node_compute_funcs);
}

TEST(InternalTestingEP, TestOrtModelWithCompileFailure) {
  // In the test file, there are 2 Conv and 1 Gemm nodes, all disconnected
  // So we should have 3 partitions be taken by InternalTestingExecutionProvider/CompileFailureTestExecutionProvider
  // But CompileFailureTestExecutionProvider will fail the Compile for partition contains "Gemm" node
  // Post layout transformations we cannot revert back if compile fails because
  // the layout transformation for this EP is already done at this stage and reverting
  // can result in more failures.
  // This is to test the model initialization fails if compile fails.
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "mnist.internal_testing_ep.ort";

  const std::unordered_set<std::string>& supported_ops{"Conv", "Gemm"};
  const std::unordered_set<std::string>& compile_failure_ops{"Gemm"};

  // Use InternalTestingExecutionProvider
  // We should have 3 partitions taken by the EP
  // 2 Conv and 1 Gemm
  {
    InferenceSessionWrapper session(SessionOptions(), GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_unique<InternalTestingExecutionProvider>(supported_ops)));
    ASSERT_STATUS_OK(session.Load(ort_model_path));
    ASSERT_STATUS_OK(session.Initialize());

    int num_replaced_nodes = CountAndValidateAssignedNodes(
        session.GetGraph(), supported_ops, const_cast<SessionState&>(session.GetSessionState()).GetMutableFuncMgr());

    ASSERT_EQ(num_replaced_nodes, 3);
  }

  // Use CompileFailureTestExecutionProvider which will fail Compile on "Gemm"
  {
    InferenceSessionWrapper session(SessionOptions(), GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(
        std::make_unique<CompileFailureTestExecutionProvider>(supported_ops, compile_failure_ops)));
    ASSERT_STATUS_OK(session.Load(ort_model_path));
    ASSERT_STATUS_NOT_OK(session.Initialize());
  }
}
}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(REDUCED_OPS_BUILD)
