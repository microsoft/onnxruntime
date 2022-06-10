// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "core/common/logging/logging.h"
#include "core/framework/utils.h"
#include "core/graph/graph.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

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
  OrtValue ml_value_w;
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
      std::make_shared<XnnpackExecutionProvider>(XnnpackExecutionProviderInfo{true}),
      std::make_shared<CPUExecutionProvider>(CPUExecutionProviderInfo{})};

  // check RegisterAllocator is implemented properly and supports calls from multiple inference sessions
  init_session(eps, session1);
  init_session(eps, session2);

  // check that allocator sharing worked. the internal testing EP should be using the CPU EP allocator
  ASSERT_EQ(eps[0]->GetAllocator(0, OrtMemType::OrtMemTypeDefault).get(),
            eps[1]->GetAllocator(0, OrtMemType::OrtMemTypeDefault).get())
      << "EPs do not have the same default allocator";
}

#endif

}  // namespace test
}  // namespace onnxruntime
