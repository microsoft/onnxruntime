// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Inverse operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunInverseTest(const std::vector<TestInputDef<DataType>>& input_defs,
                           const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                           ExpectedEPNodeAssignment expected_ep_assignment,
                           float fp32_abs_err = 1e-5,
                           const std::string& backend_name = "cpu",
                           int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<DataType>("Inverse", input_defs, {}, attrs, kMSDomain),  // Inverse Op exist in kMSDomain
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

TEST_F(QnnCPUBackendTests, Inverse_2d_test) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -100.0f, 100.0f);

  RunInverseTest<float>({TestInputDef<float>({2, 2}, false, input_vector)},
                        {},
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Inverse_3d_test) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{5, 2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -100.0f, 100.0f);

  RunInverseTest<float>({TestInputDef<float>({5, 2, 2}, false, input_vector)},
                        {},
                        ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Inverse_4d_test) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{1, 5, 2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -100.0f, 100.0f);

  RunInverseTest<float>({TestInputDef<float>({1, 5, 2, 2}, false, input_vector)},
                        {},
                        ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Returns a function that builds a model with a QDQ Inverse node.
template <typename QuantType>
GetTestQDQModelFn<QuantType> BuildQDQInverseTestCase(const std::vector<TestInputDef<float>>& input_defs,
                                                     const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                     bool use_contrib_qdq = false) {
  return [input_defs, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                              std::vector<QuantParams<QuantType>>& output_qparams) {
    const size_t num_inputs = input_defs.size();
    std::vector<NodeArg*> op_inputs;
    op_inputs.reserve(num_inputs);

    // Process input 0
    NodeArg* input0 = MakeTestInput<float>(builder, input_defs[0]);
    QuantParams<QuantType> input0_qparams = GetTestInputQuantParams<QuantType>(input_defs[0]);
    NodeArg* input0_after_qdq = AddQDQNodePair<QuantType>(builder, input0, input0_qparams.scale,
                                                          input0_qparams.zero_point, use_contrib_qdq);
    op_inputs.push_back(input0_after_qdq);

    // Op -> op_output
    auto* Inverse_output = builder.MakeIntermediate();
    Node& Inverse_node = builder.AddNode("Inverse", op_inputs, {Inverse_output}, kMSDomain);

    for (const auto& attr : attrs) {
      Inverse_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, Inverse_output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point, use_contrib_qdq);
  };
}

template <typename QuantType>
static void RunQDQInverseOpTest(const TestInputDef<float>& input_defs,
                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                QDQTolerance tolerance = QDQTolerance(),  // Default 0.4%
                                int opset = 18) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildOpTestCase<float>("Inverse", {input_defs}, {}, attrs, kMSDomain),  // Inverse Op exist in kMSDomain
                       BuildQDQInverseTestCase<QuantType>({input_defs}, attrs, true),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

TEST_F(QnnHTPBackendTests, Inverse_2d) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -100.0f, 100.0f);

  RunInverseTest<float>({TestInputDef<float>({2, 2}, false, input_vector)},
                        {},
                        ExpectedEPNodeAssignment::All,
                        1e-3f,
                        "htp");
}

TEST_F(QnnHTPBackendTests, Inverse_3d) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{10, 2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -100.0f, 100.0f);

  RunInverseTest<float>({TestInputDef<float>({10, 2, 2}, false, input_vector)},
                        {},
                        ExpectedEPNodeAssignment::All,
                        1e-3f,
                        "htp");
}

TEST_F(QnnHTPBackendTests, Inverse_4d) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{1, 10, 2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -100.0f, 100.0f);

  RunInverseTest<float>({TestInputDef<float>({1, 10, 2, 2}, false, input_vector)},
                        {},
                        ExpectedEPNodeAssignment::All,
                        1e-3f,
                        "htp");
}

TEST_F(QnnHTPBackendTests, Inverse_qdq_2d) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -10.0f, 10.0f);

  RunQDQInverseOpTest<uint8_t>(TestInputDef<float>({2, 2}, false, input_vector),
                               {},
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, Inverse_qdq_3d) {
#ifdef _M_ARM64
  // output_range=0.31888091564178467, tolerance=0.40000000596046448%.
  // Expected val (f32@CPU_EP): 0.069747790694236755
  // qdq@QNN_EP val: 0.067527718842029572 (err: 0.0022200718522071838, err/output_range: 0.69620716571807861%)
  // qdq@CPU_EP val: 0.070028752088546753 (err: 0.00028096139430999756, err/output_range: 0.088108561933040619%)
  QDQTolerance tolerance = QDQTolerance(0.008f);  // 2x of default 0.4%
#else
  QDQTolerance tolerance = QDQTolerance();  // Default 0.4%
#endif
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{10, 2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -10.0f, 10.0f);

  RunQDQInverseOpTest<uint8_t>(TestInputDef<float>({10, 2, 2}, false, input_vector),
                               {},
                               ExpectedEPNodeAssignment::All,
                               tolerance);
}

TEST_F(QnnHTPBackendTests, Inverse_qdq_4d) {
#ifdef _M_ARM64
  // output_range=0.31888091564178467, tolerance=0.40000000596046448%.
  // Expected val (f32@CPU_EP): 0.069747790694236755
  // qdq@QNN_EP val: 0.067527718842029572 (err: 0.0022200718522071838, err/output_range: 0.69620716571807861%)
  // qdq@CPU_EP val: 0.070028752088546753 (err: 0.00028096139430999756, err/output_range: 0.088108561933040619%)
  QDQTolerance tolerance = QDQTolerance(0.008f);
#else
  QDQTolerance tolerance = QDQTolerance();  // Default 0.4%
#endif
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> input_shape{1, 10, 2, 2};
  auto input_vector = rand_gen_.Uniform<float>(input_shape, -10.0f, 10.0f);

  RunQDQInverseOpTest<uint8_t>(TestInputDef<float>({1, 10, 2, 2}, false, input_vector),
                               {},
                               ExpectedEPNodeAssignment::All,
                               tolerance);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
