// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

// These tests are only applicable for the CUDA EP for now
#ifdef USE_CUDA
TEST(GatherLastToken, MultiTokenFloat) {
  OpTester test("GatherLastToken", 1, kMSDomain);

  std::vector<float> X_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  test.AddInput<float>("X", {2, 2, 2}, X_data);

  std::vector<float> Y_data = {3.f, 4.f, 3.f, 4.f};
  test.AddOutput<float>("Y", {2, 1, 2}, Y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(GatherLastToken, SingleTokenFloat_NoInputBufferReuse) {
  OpTester test("GatherLastToken", 1, kMSDomain);

  // The input buffer will not be re-used for the output as
  // the input will be a graph input
  std::vector<float> X_data = {1.f, 2.f, 3.f, 4.f};
  test.AddInput<float>("X", {2, 1, 2}, X_data);

  std::vector<float> Y_data = {1.f, 2.f, 3.f, 4.f};
  test.AddOutput<float>("Y", {2, 1, 2}, Y_data);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(GatherLastToken, MultiTokenFloat16) {
  OpTester test("GatherLastToken", 1, kMSDomain);

  std::vector<float> X_data = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f, 3.f, 4.f};
  test.AddInput<MLFloat16>("X", {2, 2, 2}, ToFloat16(X_data));

  std::vector<float> Y_data = {3.f, 4.f, 3.f, 4.f};
  test.AddOutput<MLFloat16>("Y", {2, 1, 2}, ToFloat16(Y_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(GatherLastToken, SingleTokenFloat16_NoInputBufferReuse) {
  OpTester test("GatherLastToken", 1, kMSDomain);

  // The input buffer will not be re-used for the output as
  // the input will be a graph input
  std::vector<float> X_data = {1.f, 2.f, 3.f, 4.f};
  test.AddInput<MLFloat16>("X", {2, 1, 2}, ToFloat16(X_data));

  std::vector<float> Y_data = {1.f, 2.f, 3.f, 4.f};
  test.AddOutput<MLFloat16>("Y", {2, 1, 2}, ToFloat16(Y_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Constructs a simple graph like this:
// FusedMatMul (kMsDomain) -> GatherLastToken (kMsDomain) -> Gelu (kMsDomain)
// The GatherLastToken is placed in the middle so that it doesn't consume
// graph inputs and doesn't produce graph outputs.
class FusedMatMulGatherLastTokenGeluOpTester : public OpTester {
 public:
  FusedMatMulGatherLastTokenGeluOpTester(const RunOptions& /*options*/, int opset_version)
      : OpTester("GatherLastToken", opset_version, kMSDomain) {
  }

 protected:
  void AddNodes(onnxruntime::Graph& graph,
                std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                std::vector<std::function<void(onnxruntime::Node& node)>>& /*add_attribute_funcs*/) override {
    ASSERT_EQ(graph_output_defs.size(), 1u);

    std::vector<NodeArg*> inputs;
    std::vector<NodeArg*> outputs;

    ONNX_NAMESPACE::TypeProto float_type;
    float_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    // add FusedMatMul node so that Slice input does not flow from graph input
    {
      auto& matmul_out = graph.GetOrCreateNodeArg("fused_matmul_out", &float_type);

      inputs = {graph_input_defs[0], graph_input_defs[1]};
      outputs = {&matmul_out};

      auto& matmul_node = graph.AddNode("fused_matmul_0", "FusedMatMul", "FusedMatMul",
                                        inputs, outputs, nullptr, kMSDomain);
      ORT_UNUSED_PARAMETER(matmul_node);  // Silence warning about unused var
    }

    // add GatherLastToken node
    {
      auto& gather_last_token_out = graph.GetOrCreateNodeArg("GatherLastToken_out", &float_type);

      inputs = {&graph.GetOrCreateNodeArg("fused_matmul_out", &float_type)};

      outputs = {&gather_last_token_out};

      auto& gather_last_token_node = graph.AddNode("GatherLastToken_0", "GatherLastToken",
                                                   "GatherLastToken", inputs, outputs, nullptr, kMSDomain);
      ORT_UNUSED_PARAMETER(gather_last_token_node);  // Silence warning about unused var
    }

    // add Gelu node so that Slice output does not flow into graph output
    {
      inputs = {&graph.GetOrCreateNodeArg("GatherLastToken_out", &float_type)};
      outputs = {graph_output_defs[0]};

      auto& gelu_node = graph.AddNode("gelu-0", "Gelu", "Gelu", inputs, outputs, nullptr, kMSDomain);
      ORT_UNUSED_PARAMETER(gelu_node);  // Silence warning about unused var
    }
  }
};

// This is to test the MayInPlace() code path of GatherLastToken.
TEST(GatherLastToken, OutputAndInputShapesAreSameAndReUseInputBuffer) {
  RunOptions options{};
  FusedMatMulGatherLastTokenGeluOpTester test{options, 1};

  test.AddInput<float>("matmul_data_0", {2, 1, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  test.AddInput<float>("matmul_data_1", {2, 2}, {1.0f, 1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("data_out", {2, 1, 2}, {1.954f, 1.954f, 1.954f, 1.954f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

}  // namespace test
}  // namespace onnxruntime