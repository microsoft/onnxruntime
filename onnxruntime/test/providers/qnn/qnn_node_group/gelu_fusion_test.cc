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

// Helper function to build GELU Pattern 1: Mul(0.5) before the main sequence
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

    // Final Mul: (mul_half_output) * (add_output)
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("Mul", {mul_half_output, add_output}, {output});
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

    // Create input with QDQ
    NodeArg* input = MakeTestInput<float>(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    // Create Mul(0.5) branch: input * 0.5
    // Quantize half constant with DequantizeLinear node (quant_value=255, scale=half/255)
    NodeArg* half_initializer_quant = builder.MakeInitializer<QuantType>({}, {static_cast<QuantType>(255)});
    NodeArg* half_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(half_initializer_quant, half / 255.0f, 0, half_dq);
    NodeArg* mul_half_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {input_qdq, half_dq}, {mul_half_output});

    // Create main branch: input / sqrt(2)
    // Quantize sqrt(2) constant with DequantizeLinear node (quant_value=255, scale=sqrt_2/255)
    NodeArg* sqrt2_initializer_quant = builder.MakeInitializer<QuantType>({}, {static_cast<QuantType>(255)});
    NodeArg* sqrt2_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(sqrt2_initializer_quant, sqrt_2 / 255.0f, 0, sqrt2_dq);
    NodeArg* div_output = builder.MakeIntermediate();
    builder.AddNode("Div", {input_qdq, sqrt2_dq}, {div_output});

    // Erf
    NodeArg* erf_output = builder.MakeIntermediate();
    builder.AddNode("Erf", {div_output}, {erf_output});

    // Add 1.0
    // Quantize one constant with DequantizeLinear node (quant_value=255, scale=one/255)
    NodeArg* one_initializer_quant = builder.MakeInitializer<QuantType>({}, {static_cast<QuantType>(255)});
    NodeArg* one_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(one_initializer_quant, one / 255.0f, 0, one_dq);
    NodeArg* add_output = builder.MakeIntermediate();
    builder.AddNode("Add", {erf_output, one_dq}, {add_output});

    // Final Mul: (mul_half_output) * (add_output)
    NodeArg* mul_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {mul_half_output, add_output}, {mul_output});

    // Add output QDQ
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, mul_output, output_qparams[0].scale,
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

    // Create input with QDQ
    NodeArg* input = MakeTestInput<float>(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    // Main branch: input / sqrt(2)
    // Quantize sqrt(2) constant with DequantizeLinear node (quant_value=255, scale=sqrt_2/255)
    NodeArg* sqrt2_initializer_quant = builder.MakeInitializer<QuantType>({}, {static_cast<QuantType>(255)});
    NodeArg* sqrt2_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(sqrt2_initializer_quant, sqrt_2 / 255.0f, 0, sqrt2_dq);
    NodeArg* div_output = builder.MakeIntermediate();
    builder.AddNode("Div", {input_qdq, sqrt2_dq}, {div_output});

    // Erf
    NodeArg* erf_output = builder.MakeIntermediate();
    builder.AddNode("Erf", {div_output}, {erf_output});

    // Add 1.0
    // Quantize one constant with DequantizeLinear node (quant_value=255, scale=one/255)
    NodeArg* one_initializer_quant = builder.MakeInitializer<QuantType>({}, {static_cast<QuantType>(255)});
    NodeArg* one_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(one_initializer_quant, one / 255.0f, 0, one_dq);
    NodeArg* add_output = builder.MakeIntermediate();
    builder.AddNode("Add", {erf_output, one_dq}, {add_output});

    // Mul with input: input * add_output
    NodeArg* mul_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {input_qdq, add_output}, {mul_output});

    // Final Mul with 0.5
    // Quantize half constant with DequantizeLinear node (quant_value=255, scale=half/255)
    NodeArg* half_initializer_quant = builder.MakeInitializer<QuantType>({}, {static_cast<QuantType>(255)});
    NodeArg* half_dq = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<QuantType>(half_initializer_quant, half / 255.0f, 0, half_dq);
    NodeArg* mul_final_output = builder.MakeIntermediate();
    builder.AddNode("Mul", {mul_output, half_dq}, {mul_final_output});

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
                  /*fp32_abs_err=*/1e-4f);
}

// Test GELU Pattern 2 with float32 model (for baseline comparison)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_Float32) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-4f);
}

// Test GELU Pattern 1 with QDQ (uint8)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_QDQ_U8) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def),
                       BuildQDQGeluPattern1TestCase<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                       /*tolerance=*/QDQTolerance(0.005f));
}

// Test GELU Pattern 2 with QDQ (uint8)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_QDQ_U8) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern2TestCase(input_def),
                       BuildQDQGeluPattern2TestCase<uint8_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                       /*tolerance=*/QDQTolerance(0.005f));
}

// Test GELU Pattern 1 with QDQ (uint16)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_QDQ_U16) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern1TestCase(input_def),
                       BuildQDQGeluPattern1TestCase<uint16_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                       /*tolerance=*/QDQTolerance(0.002f));
}

// Test GELU Pattern 2 with QDQ (uint16)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_QDQ_U16) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 2, 3, 4}, false, -1.0f, 1.0f);

  TestQDQModelAccuracy(BuildGeluPattern2TestCase(input_def),
                       BuildQDQGeluPattern2TestCase<uint16_t>(input_def),
                       provider_options,
                       /*opset_version=*/13,
                       /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                       /*tolerance=*/QDQTolerance(0.002f));
}

// Test GELU Pattern 1 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_LargeInput) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 128, 768}, false, -2.0f, 2.0f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-4f);
}

// Test GELU Pattern 2 with larger input shape
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_LargeInput) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 128, 768}, false, -2.0f, 2.0f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-4f);
}

// Test GELU Pattern 1 with different input ranges
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_DifferentRange) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -3.0f, 3.0f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-4f);
}

// Test GELU Pattern 2 with different input ranges
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_DifferentRange) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({1, 16, 32}, false, -3.0f, 3.0f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-4f);
}

// Test GELU Pattern 1 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern1_2D) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern1TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-4f);
}

// Test GELU Pattern 2 with 2D input (typical for linear layers)
TEST_F(QnnHTPBackendTests, GeluFusionPattern2_2D) {
  ProviderOptions provider_options = GetProviderOptions();
  auto input_def = TestInputDef<float>({32, 512}, false, -1.5f, 1.5f);

  RunQnnModelTest(BuildGeluPattern2TestCase(input_def),
                  provider_options,
                  /*opset_version=*/13,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::All,
                  /*fp32_abs_err=*/1e-4f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
