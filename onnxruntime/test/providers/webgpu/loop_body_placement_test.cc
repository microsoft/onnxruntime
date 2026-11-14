// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/tensor.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr int64_t kSteps = 3;
constexpr int64_t kWidth = 2;

void AddValueInfo(ONNX_NAMESPACE::ValueInfoProto& info, const std::string& name, int32_t elem_type,
                  const std::vector<int64_t>& dims) {
  info.set_name(name);
  auto* tensor_type = info.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);
  for (int64_t dim : dims) {
    tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
  }
}

ONNX_NAMESPACE::NodeProto& AddNode(ONNX_NAMESPACE::GraphProto& graph, const std::string& op_type,
                                   const std::vector<std::string>& inputs, const std::vector<std::string>& outputs) {
  auto& node = *graph.add_node();
  node.set_name(op_type + "_" + outputs[0]);
  node.set_op_type(op_type);
  for (const auto& input : inputs) {
    node.add_input(input);
  }
  for (const auto& output : outputs) {
    node.add_output(output);
  }
  return node;
}

// A model that runs a Scan over the rows of `X * X`, summing them in its state `s` (`s = s + x`) and emitting
// `y = s * s` per row, or `y = sqrt(s)` with `sqrt_output`. The Mul in front gives the Scan an input produced on the
// GPU.
std::string BuildScanModel(int32_t elem_type, bool sqrt_output = false) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  ONNX_NAMESPACE::GraphProto body;
  body.set_name("body");
  AddValueInfo(*body.add_input(), "s", elem_type, {kWidth});
  AddValueInfo(*body.add_input(), "x", elem_type, {kWidth});
  AddNode(body, "Add", {"s", "x"}, {"s_next"});
  if (sqrt_output) {
    AddNode(body, "Sqrt", {"s_next"}, {"y"});
  } else {
    AddNode(body, "Mul", {"s_next", "s_next"}, {"y"});
  }
  AddValueInfo(*body.add_output(), "s_next", elem_type, {kWidth});
  AddValueInfo(*body.add_output(), "y", elem_type, {kWidth});

  auto& graph = *model.mutable_graph();
  graph.set_name("main");
  AddValueInfo(*graph.add_input(), "X", elem_type, {kSteps, kWidth});
  AddValueInfo(*graph.add_input(), "S0", elem_type, {kWidth});
  AddNode(graph, "Mul", {"X", "X"}, {"X2"});
  auto& scan = AddNode(graph, "Scan", {"S0", "X2"}, {"S", "Y"});
  auto* num_scan_inputs = scan.add_attribute();
  num_scan_inputs->set_name("num_scan_inputs");
  num_scan_inputs->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  num_scan_inputs->set_i(1);
  auto* body_attribute = scan.add_attribute();
  body_attribute->set_name("body");
  body_attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
  *body_attribute->mutable_g() = body;
  AddValueInfo(*graph.add_output(), "S", elem_type, {kWidth});
  AddValueInfo(*graph.add_output(), "Y", elem_type, {kSteps, kWidth});

  std::string bytes;
  EXPECT_TRUE(model.SerializeToString(&bytes));
  return bytes;
}

// Loads `model_bytes` with the WebGPU EP and returns the EP of each node in the Scan's body.
std::vector<std::string> LoadAndGetBodyPlacement(InferenceSessionWrapper& session, const std::string& model_bytes) {
  std::vector<std::string> placement;
  EXPECT_STATUS_OK(session.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
  EXPECT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  EXPECT_STATUS_OK(session.Initialize());
  for (const auto& node : session.GetGraph().Nodes()) {
    if (node.OpType() != "Scan") {
      continue;
    }
    EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider);
    for (const auto& body : node.GetSubgraphs()) {
      for (const auto& body_node : body->Nodes()) {
        placement.push_back(body_node.GetExecutionProviderType());
      }
    }
  }
  return placement;
}

}  // namespace

// The CPU EP runs Scan, and runs its body once per step. A body on the GPU would copy its state and slice to the GPU
// and its outputs back at every step, so the WebGPU EP leaves the body to the CPU EP.
TEST(WebGpuLoopBodyPlacementTest, ScanBodyRunsOnCpu) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  SessionOptions so;
  InferenceSessionWrapper session{so, GetEnvironment()};
  const auto placement = LoadAndGetBodyPlacement(session, BuildScanModel(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  ASSERT_EQ(placement.size(), 2u);
  for (const auto& ep : placement) {
    EXPECT_EQ(ep, kCpuExecutionProvider);
  }

  // The Mul in front of the Scan stays on the GPU.
  for (const auto& node : session.GetGraph().Nodes()) {
    if (node.OpType() == "Mul") {
      EXPECT_EQ(node.GetExecutionProviderType(), kWebGpuExecutionProvider);
    }
  }

  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtValue x;
  OrtValue s0;
  CreateMLValue<float>(allocator, {kSteps, kWidth}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, &x);
  CreateMLValue<float>(allocator, {kWidth}, {0.f, 1.f}, &s0);
  std::vector<OrtValue> fetches;
  const std::vector<std::string> output_names{"S", "Y"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, NameMLValMap{{"X", x}, {"S0", s0}}, output_names, &fetches));

  // Rows of X * X: [1, 4], [9, 16], [25, 36]. States: [1, 5], [10, 21], [35, 57].
  const std::vector<float> expected_s{35.f, 57.f};
  const std::vector<float> expected_y{1.f, 25.f, 100.f, 441.f, 1225.f, 3249.f};
  const auto s = fetches[0].Get<Tensor>().DataAsSpan<float>();
  const auto y = fetches[1].Get<Tensor>().DataAsSpan<float>();
  EXPECT_EQ(std::vector<float>(s.begin(), s.end()), expected_s);
  EXPECT_EQ(std::vector<float>(y.begin(), y.end()), expected_y);
}

// The CPU EP has no float16 Sqrt kernel, so this body stays on the WebGPU EP: placement does not depend on what the
// CPU EP might be able to run after inserting casts.
TEST(WebGpuLoopBodyPlacementTest, ScanBodyWithoutCpuKernelsStaysOnWebGpu) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  SessionOptions so;
  InferenceSessionWrapper session{so, GetEnvironment()};
  const auto placement =
      LoadAndGetBodyPlacement(session, BuildScanModel(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, true));
  ASSERT_EQ(placement.size(), 2u);
  for (const auto& ep : placement) {
    EXPECT_EQ(ep, kWebGpuExecutionProvider);
  }
}

}  // namespace test
}  // namespace onnxruntime
