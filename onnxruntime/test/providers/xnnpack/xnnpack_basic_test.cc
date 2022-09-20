// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <string>

#include "core/common/logging/logging.h"
#include "core/framework/utils.h"
#include "core/graph/graph.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

#if !defined(ORT_MINIMAL_BUILD)
// if this is a full build we need the provider test utils
#include "test/optimizer/qdq_test_utils.h"
#endif

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// test uses ONNX model so can't be run in a minimal build.
// TODO: When we need XNNPACK in a minimal build we should add an ORT format version of the model
#if !defined(ORT_MINIMAL_BUILD)

// use a snippet from a production model that has NHWC input/output, and Conv nodes with possible Clip and Relu fusion.
// xnnpack should be able to take all the Conv nodes, and fuse the Conv+Clip and Conv+Relu nodes.
// That should also mean the Transpose nodes at the start and end of the model can be removed as xnnpack will be
// handling all other nodes in the model, and the xnnpack nodes will have NHWC input and output.
TEST(XnnpackEP, TestNhwcConvReluClipFusion) {
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "nhwc_conv_clip_relu.onnx";

  RandomValueGenerator generator;
  TensorShape input_shape_x{1, 16, 16, 192};
  std::vector<float> input_x = generator.Uniform<float>(input_shape_x.GetDims(), -128, 128);

  OrtValue ml_value_x;
  CreateMLValue<float>(input_shape_x.GetDims(), input_x.data(), OrtMemoryInfo(), &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("model_input", ml_value_x));

  std::function<void(const Graph&)> verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 3) << "Transpose nodes should have been removed, and "
                                           "Conv+Relu and Conv+Clip should have been fused, leaving 3 nodes.";
    auto node_iter = graph.Nodes().begin();
    auto check_node = [](const Node& node, const std::string& fusion_type) {
      const auto& attr = node.GetAttributes();
      auto activation = attr.find("activation");
      ASSERT_NE(activation, attr.cend()) << "Fused node should have activation attribute";
      ASSERT_EQ(activation->second.s(), fusion_type);
    };

    // check 2nd and 3rd nodes.
    // the first node is the Conv that does not get fused (created after first call to GetCapability)
    // the 2nd and 3rd nodes are the fused nodes (created after second call to GetCapability)
    ++node_iter;
    check_node(*node_iter, "Clip");
    ++node_iter;
    check_node(*node_iter, "Relu");
  };

  EPVerificationParams params;
  params.ep_node_assignment = ExpectedEPNodeAssignment::All;
  params.fp32_abs_err = 0.0002f;
  params.graph_verifier = &verify;

  auto ep = DefaultXnnpackExecutionProvider();
  RunAndVerifyOutputsWithEP(ort_model_path, "TestNhwcConvReluClipFusion", std::move(ep), feeds, params);
}

// test we can share the cpu ep allocator with the xnnpack EP
TEST(XnnpackEP, TestAllocatorSharing) {
  auto init_session = [](std::vector<std::shared_ptr<IExecutionProvider>>& eps,
                         InferenceSessionWrapper& session) {
    for (const auto& ep : eps) {
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(ep));
    }

    const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "nhwc_conv_clip_relu.onnx";
    ASSERT_STATUS_OK(session.Load(ort_model_path));
    ASSERT_STATUS_OK(session.Initialize());
  };

  // create 2 sessions
  SessionOptions so;
  InferenceSessionWrapper session1(so, GetEnvironment());
  InferenceSessionWrapper session2(so, GetEnvironment());

  // and use the same EP instances in both
  std::vector<std::shared_ptr<IExecutionProvider>> eps{
      std::make_shared<XnnpackExecutionProvider>(XnnpackExecutionProviderInfo{}),
      std::make_shared<CPUExecutionProvider>(CPUExecutionProviderInfo{}, true /* delay allocator creation to allow sharing */)};

  // check RegisterAllocator is implemented properly and supports calls from multiple inference sessions
  init_session(eps, session1);
  init_session(eps, session2);

  // check that allocator sharing worked. the internal testing EP should be using the CPU EP allocator
  ASSERT_EQ(eps[0]->GetAllocator(0, OrtMemType::OrtMemTypeDefault).get(),
            eps[1]->GetAllocator(0, OrtMemType::OrtMemTypeDefault).get())
      << "EPs do not have the same default allocator";
}

TEST(XnnpackEP, TestAddEpUsingPublicApi) {
  {
    // C++ API test
    Ort::SessionOptions so;
    onnxruntime::ProviderOptions options;
    // no real options currently but set a value to make sure it's passed through. requires manual validation.
    options["one"] = "two";
    so.AppendExecutionProvider("XNNPACK", options);

    const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "nhwc_conv_clip_relu.onnx";
    Ort::Session session(*ort_env, ort_model_path, so);

    // dirty hack to access the underlying InferenceSession but don't know a better way.
    const OrtSession* ort_session = session;
    const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);

    bool have_xnnpack_ep = false;

    for (const auto& provider : s->GetRegisteredProviderTypes()) {
      if (provider == kXnnpackExecutionProvider) {
        have_xnnpack_ep = true;
        break;
      }
    }

    ASSERT_TRUE(have_xnnpack_ep) << "Xnnpack EP was not found in registered providers for session.";
  }

  {
    // C API test
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtSessionOptions* so{nullptr};
    ASSERT_ORTSTATUS_OK(api, CreateSessionOptions(&so));

    // add with provider options. manually check the ProviderOptions instance passed through to
    // OrtSessionOptionsAppendExecutionProvider_Xnnpack is correct.
    const char* keys[1] = {"one"};
    const char* values[1] = {"two"};
    ASSERT_ORTSTATUS_OK(api, SessionOptionsAppendExecutionProvider(so, "XNNPACK", keys, values, 1));
    api->ReleaseSessionOptions(so);
  }
}

static void RunModelTest(
    const GetQDQTestCaseFn& build_test_case,
    const char* test_description,
    const EPVerificationParams& params = EPVerificationParams()) {
  onnxruntime::Model model(test_description, false, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  RunAndVerifyOutputsWithEP(model_data, "XnnpackEP.TestQDQModel",
                            DefaultXnnpackExecutionProvider(),
                            helper.feeds_, params);
}

static void RunModelTestWithPath(const ORTCHAR_T* ort_model_path, const char* graph_name, float scale_factor = 1.0f) {
  std::function<void(const Graph&)> verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 5) << "Transpose*2 + dq +q +qlinearconv "
                                           "leaving 5 nodes.";
  };
  EPVerificationParams params;
  params.ep_node_assignment = ExpectedEPNodeAssignment::Some;
  // Xnnpack has higher precision than CPU_S8S8,
  // we can either give a higher tolerance,or disable Graph_Optimizations for cpu-ep
  params.fp32_abs_err = 1.8f / scale_factor;
  params.graph_verifier = &verify;

  RandomValueGenerator generator;
  TensorShape input_shape_x{1, 3, 24, 24};
  std::vector<float> input_x = generator.Uniform<float>(input_shape_x.GetDims(),
                                                        -128 * scale_factor, 128 * scale_factor);
  OrtValue ml_value_x;
  CreateMLValue<float>(input_shape_x.GetDims(), input_x.data(), OrtMemoryInfo(), &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("input", ml_value_x));

  auto ep = DefaultXnnpackExecutionProvider();
  RunAndVerifyOutputsWithEP(ort_model_path, graph_name, std::move(ep), feeds, params);
}

TEST(XnnpackEP, TestQDQConvU8U8) {
  RunModelTest(BuildQDQConvTestCase<uint8_t /* InputType */,
                                    uint8_t /* WeightType */,
                                    int32_t /* BiasType */,
                                    uint8_t /* OutputType */>(
                   {1, 1, 5, 5} /* input_shape */,
                   {1, 1, 3, 3} /* weights_shape */),
               "xnnpack_qdq_test_graph_conv_u8u8",
               {ExpectedEPNodeAssignment::Some});  // two transpose nodes would be added before and after

  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "conv_qdq_u8u8.onnx";
  RunModelTestWithPath(ort_model_path, "xnnpack_qdq_test_graph_conv_u8u8");
}

TEST(XnnpackEP, TestQDQConvS8S8) {
  RunModelTest(BuildQDQConvTestCase<int8_t /* InputType */,
                                    int8_t /* WeightType */,
                                    int32_t /* BiasType */,
                                    int8_t /* OutputType */>(
                   {1, 1, 5, 5} /* input_shape */,
                   {1, 1, 3, 3} /* weights_shape */),
               "xnnpack_qdq_test_graph_conv_s8s8",
               {ExpectedEPNodeAssignment::Some, 0.2f});

  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "conv_qdq_s8s8.onnx";
  RunModelTestWithPath(ort_model_path, "xnnpack_qdq_test_graph_conv_s8s8", 0.7f);
}

TEST(XnnpackEP, TestQDQConvS8S8_per_channel) {
  const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "conv_qdq_s8s8_perchannel.onnx";
  RunModelTestWithPath(ort_model_path, "xnnpack_qdq_test_graph_conv_s8s8_perchannel", 0.2f);
}

TEST(XnnpackEP, TestAveragePool) {
  const std::vector<int64_t> input_shape = {1, 2, 3, 3};
  auto modelBuilder = [&input_shape](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();
    Node& pool_node = builder.AddNode("AveragePool", {input_arg}, {output_arg});
    std::vector<int64_t> pads((input_shape.size() - 2) * 2, 1);
    pool_node.AddAttribute("pads", pads);
    std::vector<int64_t> kernel_shape(input_shape.size() - 2, 3);
    pool_node.AddAttribute("kernel_shape", kernel_shape);
  };
  RunModelTest(modelBuilder, "xnnpack_test_graph_averagepool",
               {
                   ExpectedEPNodeAssignment::Some,
                   1e-2f /* fp32_abs_err */,
               });
}

TEST(XnnpackEP, TestQDQAveragePool) {
  RunModelTest(BuildQDQAveragePoolTestCase<uint8_t /* InputType */,
                                           uint8_t /* OutputType */>(
                   {1, 1, 30, 30} /* input_shape */, static_cast<int64_t>(1)),
               "xnnpack_qdq_test_graph_averagepool",
               {
                   ExpectedEPNodeAssignment::Some,
                   1e-2f /* fp32_abs_err */,
               });
}

TEST(XnnpackEP, TestMaxPool) {
  const std::vector<int64_t> input_shape = {1, 2, 13, 13};
  auto modelBuilder = [&input_shape](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();
    Node& pool_node = builder.AddNode("MaxPool", {input_arg}, {output_arg});
    std::vector<int64_t> pads((input_shape.size() - 2) * 2, 1);
    pool_node.AddAttribute("pads", pads);
    std::vector<int64_t> kernel_shape(input_shape.size() - 2, 3);
    pool_node.AddAttribute("kernel_shape", kernel_shape);
  };
  RunModelTest(modelBuilder, "xnnpack_test_graph_maxpool",
               {
                   ExpectedEPNodeAssignment::Some,
                   1e-2f /* fp32_abs_err */,
               });
}

TEST(XnnpackEP, TestQDQMaxPool_u8) {
  RunModelTest(BuildQDQMaxPoolTestCase<uint8_t /* InputType */,
                                       uint8_t /* OutputType */>(
                   {1, 1, 30, 30} /* input_shape */, true),
               "xnnpack_qdq_test_graph_maxpool_u8",
               {
                   ExpectedEPNodeAssignment::Some,
                   1e-2f /* fp32_abs_err */,
               });
}

TEST(XnnpackEP, TestQDQMaxPool_s8) {
  std::function<void(const Graph&)> verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 5) << "Transpose *2 +dq + q +pool"
                                           " leaving 5 nodes.";
  };

  RunModelTest(BuildQDQMaxPoolTestCase<int8_t /* InputType */,
                                       int8_t /* OutputType */>(
                   {1, 1, 30, 30} /* input_shape */, true),
               "xnnpack_qdq_test_graph_maxpool_s8",
               {ExpectedEPNodeAssignment::Some,
                1e-2f /* fp32_abs_err */, &verify});
  verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 7) << "Transpose *2 +dq*2 + q*2 +pool"
                                           " leaving 7 nodes.";
  };
  RunModelTest(BuildQDQMaxPoolTestCase<int8_t /* InputType */,
                                       int8_t /* OutputType */>(
                   {1, 1, 30, 30} /* input_shape */, false),
               "xnnpack_qdq_test_graph_maxpool_s8",
               {ExpectedEPNodeAssignment::Some,
                1e-2f /* fp32_abs_err */, &verify});
}

// xnnpack only support the last dim as reduced axis,
// we are expected that the other reduce axis would be handled by CPUEP
TEST(XnnpackEP, TestQDQSoftMax_axisZero_v13) {
  RunModelTest(BuildQDQSoftMaxTestCase<uint8_t, uint8_t>(
                   {1, 2, 3, 32} /* input_shape */,
                   static_cast<int64_t>(0) /* axis */,
                   1.f / 256 /* output_scales */,
                   0 /* output_zp */),
               "xnnpack_qdq_test_graph_softmax",
               {ExpectedEPNodeAssignment::None});
}

TEST(XnnpackEP, TestSoftMax_axisLast) {
  const std::vector<int64_t> input_shape = {1, 2, 3, 5};
  int64_t axis = input_shape.size() - 1;
  auto modelCreater = [input_shape, axis](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape,
                                               std::numeric_limits<float>::min(),
                                               std::numeric_limits<float>::max());

    auto* output_arg = builder.MakeOutput();

    // add SoftMax
    Node& softmax_node = builder.AddNode("Softmax", {input_arg}, {output_arg});
    softmax_node.AddAttribute("axis", axis);
  };
  RunModelTest(modelCreater,
               "xnnpack_test_graph_softmax",
               {ExpectedEPNodeAssignment::All});
}

TEST(XnnpackEP, TestQDQSoftMax_axisLast) {
  RunModelTest(BuildQDQSoftMaxTestCase<uint8_t, uint8_t>(
                   {1, 2, 3, 5} /* input_shape */,
                   static_cast<int64_t>(3) /* axis */,
                   1.f / 256 /* output_scales */,
                   0 /* output_zp */),
               "xnnpack_qdq_test_graph_softmax",
               {ExpectedEPNodeAssignment::All});
}
#endif

}  // namespace test
}  // namespace onnxruntime
