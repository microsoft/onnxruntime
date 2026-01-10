// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/unittest_util/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

namespace {

// Helper function to build GELU Pattern 1: root -> Mul -> Div -> Erf -> Add -> Mul
// Pattern 1:
//                   +-------Mul(0.5)---------------------+
//                   |                                    |
//                   |                                    v
//                [root] --> Div -----> Erf  --> Add --> Mul ==>
//                          (B=1.4142...)        (1)
GetTestModelFn BuildGeluPattern1TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    // Create input
    NodeArg* input = MakeTestInput<float>(builder, input_def);

    // Create Mul(0.5) branch: input * 0.5
    NodeArg* half_initializer = builder.MakeScalarInitializer<float>(half);
    NodeArg* mul_half_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {input, half_initializer}, {mul_half_output});

    // Create main branch: input / sqrt(2)
    NodeArg* sqrt2_initializer = builder.MakeScalarInitializer<float>(sqrt_2);
    NodeArg* div_output = builder.MakeIntermediate();
    builder.AddNode("Div", {input, sqrt2_initializer}, {div_output});

    // Erf
    NodeArg* erf_output = builder.MakeIntermediate();
    builder.AddNode("Erf", {div_output}, {erf_output});

    // Add 1.0
    NodeArg* one_initializer = builder.MakeScalarInitializer<float>(one);
    NodeArg* add_output = builder.MakeIntermediate();
    builder.AddNode("Add", {erf_output, one_initializer}, {add_output});

    // Final Mul: (add_output) * (mul_half_output)
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("Mul", {add_output, mul_half_output}, {output});
  };
}

// Helper function to build GELU Pattern 2: Mul(0.5) after the main sequence
// Pattern 2:
//                   +------------------------------------+
//                   |                                    |
//                   |                                    v
//                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
//                          (B=1.4142...)        (1)            (0.5)
GetTestModelFn BuildGeluPattern2TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    // Create input
    NodeArg* input = MakeTestInput<float>(builder, input_def);

    // Main branch: input / sqrt(2)
    NodeArg* sqrt2_initializer = builder.MakeScalarInitializer<float>(sqrt_2);
    NodeArg* div_output = builder.MakeIntermediate();
    builder.AddNode("Div", {input, sqrt2_initializer}, {div_output});

    // Erf
    NodeArg* erf_output = builder.MakeIntermediate();
    builder.AddNode("Erf", {div_output}, {erf_output});

    // Add 1.0
    NodeArg* one_initializer = builder.MakeScalarInitializer<float>(one);
    NodeArg* add_output = builder.MakeIntermediate();
    builder.AddNode("Add", {erf_output, one_initializer}, {add_output});

    // Mul with input: input * add_output
    NodeArg* mul_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {input, add_output}, {mul_output});

    // Final Mul with 0.5: mul_output * 0.5
    NodeArg* half_initializer = builder.MakeScalarInitializer<float>(half);
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("Mul", {mul_output, half_initializer}, {output});
  };
}

// Helper function to build QDQ GELU Pattern 1
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPattern1TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    // Create input
    NodeArg* input = MakeTestInput<float>(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);

    // Quantize input once
    NodeArg* input_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(input, input_qparams.scale, input_qparams.zero_point, input_q);

    // Create quantized constants with individual quantization parameters
    // For scalar constants, use range [0, value] to ensure proper quantization
    QuantParams<QuantType> sqrt2_qparams = GetTestInputQuantParams<QuantType>(TestInputDef<float>({}, true, 0.0f, sqrt_2));
    NodeArg* sqrt2_initializer = builder.MakeScalarInitializer<float>(sqrt_2);
    NodeArg* sqrt2_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(sqrt2_initializer, sqrt2_qparams.scale, sqrt2_qparams.zero_point, sqrt2_q);

    QuantParams<QuantType> one_qparams = GetTestInputQuantParams<QuantType>(TestInputDef<float>({}, true, 0.0f, one));
    NodeArg* one_initializer = builder.MakeScalarInitializer<float>(one);
    NodeArg* one_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(one_initializer, one_qparams.scale, one_qparams.zero_point, one_q);

    QuantParams<QuantType> half_qparams = GetTestInputQuantParams<QuantType>(TestInputDef<float>({}, true, 0.0f, half));
    NodeArg* half_initializer = builder.MakeScalarInitializer<float>(half);
    NodeArg* half_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(half_initializer, half_qparams.scale, half_qparams.zero_point, half_q);

    NodeArg* input_dq_1 = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_q, input_qparams.scale, input_qparams.zero_point, input_dq_1);
    NodeArg* sqrt2_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(sqrt2_q, sqrt2_qparams.scale, sqrt2_qparams.zero_point, sqrt2_dq);
    NodeArg* div_output = builder.MakeIntermediate();
    builder.AddNode("Div", {input_dq_1, sqrt2_dq}, {div_output});
    NodeArg* div_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(div_output, input_qparams.scale, input_qparams.zero_point, div_q);

    // DQ -> Erf -> Q
    NodeArg* div_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(div_q, input_qparams.scale, input_qparams.zero_point, div_dq);
    NodeArg* erf_output = builder.MakeIntermediate();
    builder.AddNode("Erf", {div_dq}, {erf_output});
    NodeArg* erf_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(erf_output, input_qparams.scale, input_qparams.zero_point, erf_q);

    // DQ -> Add -> Q
    NodeArg* erf_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(erf_q, input_qparams.scale, input_qparams.zero_point, erf_dq);
    NodeArg* one_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(one_q, one_qparams.scale, one_qparams.zero_point, one_dq);
    NodeArg* add_output = builder.MakeIntermediate();
    builder.AddNode("Add", {erf_dq, one_dq}, {add_output});
    NodeArg* add_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(add_output, input_qparams.scale, input_qparams.zero_point, add_q);

    // DQ -> Mul (with input) -> Q
    NodeArg* input_dq_2 = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_q, input_qparams.scale, input_qparams.zero_point, input_dq_2);
    NodeArg* add_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(add_q, input_qparams.scale, input_qparams.zero_point, add_dq);
    NodeArg* mul_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {input_dq_2, add_dq}, {mul_output});
    NodeArg* mul_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(mul_output, input_qparams.scale, input_qparams.zero_point, mul_q);

    // Final DQ -> Mul (with 0.5) -> Q
    NodeArg* mul_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(mul_q, input_qparams.scale, input_qparams.zero_point, mul_dq);
    NodeArg* half_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(half_q, half_qparams.scale, half_qparams.zero_point, half_dq);
    NodeArg* mul_final_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {mul_dq, half_dq}, {mul_final_output});

    // Add output QDQ
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, mul_final_output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point);
  };
}

// Helper function to build QDQ GELU Pattern 2
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQGeluPattern2TestCase(const TestInputDef<float>& input_def) {
  return [input_def](ModelTestBuilder& builder, std::vector<QuantParams<QuantType>>& output_qparams) -> void {
    constexpr float sqrt_2 = 1.4142135381698608f;
    constexpr float half = 0.5f;
    constexpr float one = 1.0f;

    // Create input
    NodeArg* input = MakeTestInput<float>(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);

    // Quantize input once
    NodeArg* input_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(input, input_qparams.scale, input_qparams.zero_point, input_q);

    // Create quantized constants with individual quantization parameters
    // For scalar constants, use range [0, value] to ensure proper quantization
    QuantParams<QuantType> sqrt2_qparams = GetTestInputQuantParams<QuantType>(TestInputDef<float>({}, true, 0.0f, sqrt_2));
    NodeArg* sqrt2_initializer = builder.MakeScalarInitializer<float>(sqrt_2);
    NodeArg* sqrt2_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(sqrt2_initializer, sqrt2_qparams.scale, sqrt2_qparams.zero_point, sqrt2_q);

    QuantParams<QuantType> one_qparams = GetTestInputQuantParams<QuantType>(TestInputDef<float>({}, true, 0.0f, one));
    NodeArg* one_initializer = builder.MakeScalarInitializer<float>(one);
    NodeArg* one_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(one_initializer, one_qparams.scale, one_qparams.zero_point, one_q);

    QuantParams<QuantType> half_qparams = GetTestInputQuantParams<QuantType>(TestInputDef<float>({}, true, 0.0f, half));
    NodeArg* half_initializer = builder.MakeScalarInitializer<float>(half);
    NodeArg* half_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(half_initializer, half_qparams.scale, half_qparams.zero_point, half_q);

    // Main branch: DQ -> Div -> Q
    NodeArg* input_dq_1 = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_q, input_qparams.scale, input_qparams.zero_point, input_dq_1);
    NodeArg* sqrt2_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(sqrt2_q, sqrt2_qparams.scale, sqrt2_qparams.zero_point, sqrt2_dq);
    NodeArg* div_output = builder.MakeIntermediate();
    builder.AddNode("Div", {input_dq_1, sqrt2_dq}, {div_output});
    NodeArg* div_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(div_output, input_qparams.scale, input_qparams.zero_point, div_q);

    // DQ -> Erf -> Q
    NodeArg* div_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(div_q, input_qparams.scale, input_qparams.zero_point, div_dq);
    NodeArg* erf_output = builder.MakeIntermediate();
    builder.AddNode("Erf", {div_dq}, {erf_output});
    NodeArg* erf_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(erf_output, input_qparams.scale, input_qparams.zero_point, erf_q);

    // DQ -> Add -> Q
    NodeArg* erf_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(erf_q, input_qparams.scale, input_qparams.zero_point, erf_dq);
    NodeArg* one_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(one_q, one_qparams.scale, one_qparams.zero_point, one_dq);
    NodeArg* add_output = builder.MakeIntermediate();
    builder.AddNode("Add", {erf_dq, one_dq}, {add_output});
    NodeArg* add_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(add_output, input_qparams.scale, input_qparams.zero_point, add_q);

    // DQ -> Mul (with input) -> Q
    NodeArg* input_dq_2 = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(input_q, input_qparams.scale, input_qparams.zero_point, input_dq_2);
    NodeArg* add_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(add_q, input_qparams.scale, input_qparams.zero_point, add_dq);
    NodeArg* mul_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {input_dq_2, add_dq}, {mul_output});
    NodeArg* mul_q = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<QuantType>(mul_output, input_qparams.scale, input_qparams.zero_point, mul_q);

    // Final DQ -> Mul (with 0.5)
    NodeArg* mul_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(mul_q, input_qparams.scale, input_qparams.zero_point, mul_dq);
    NodeArg* half_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(half_q, half_qparams.scale, half_qparams.zero_point, half_dq);
    NodeArg* mul_final_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {mul_dq, half_dq}, {mul_final_output});

    // Add output QDQ
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, mul_final_output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point);
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

// Test GELU Pattern 1 with float32 model (for baseline comparison)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_Float32) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);
}

// Test GELU Pattern 2 with float32 model (for baseline comparison)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_Float32) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);
}

// Test GELU Pattern 1 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_LargeInput) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 128, 768}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);
}

// Test GELU Pattern 2 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_LargeInput) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 128, 768}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);
}

// Test GELU Pattern 1 with 3D input
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_3D) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);
}

// Test GELU Pattern 2 with 3D input
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_3D) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-3f);
}

// Test GELU Pattern 1 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_2D) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);
}

// Test GELU Pattern 2 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_2D) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/2e-3f);
}

// Test GELU Pattern 1 with QDQ
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_QDQ_U8) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def),
                       BuildQDQGeluPattern1TestCase<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);
}

// Test GELU Pattern 2 with QDQ
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_QDQ_U8) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern2TestCase(input_def),
                       BuildQDQGeluPattern2TestCase<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
