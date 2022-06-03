// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "core/common/logging/logging.h"
#include "core/framework/utils.h"
#include "core/graph/graph.h"
#include "core/providers/xnnpack/xnnpack_execution_provider.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
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
#endif

}  // namespace test
}  // namespace onnxruntime
