// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <vector>

#include "gtest/gtest.h"

#include "core/graph/node_attr_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename DataType>
inline GetTestModelFn BuildNonZeroTestCase(const TestInputDef<DataType>& input_def, const bool fix_shape) {
  return [input_def, fix_shape](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput<DataType>(builder, input_def);

    NodeArg* output;
    if (fix_shape) {
      // Fix NonZero output shape to maximum possible size.
      const std::vector<int64_t>& shape = input_def.GetShape();
      std::vector<int64_t> output_shape;
      output_shape.push_back(static_cast<int64_t>(shape.size()));
      output_shape.push_back(std::accumulate(shape.begin(),
                                             shape.end(),
                                             static_cast<int64_t>(1),
                                             std::multiplies<int64_t>()));

      output = builder.MakeOutput<int64_t>(output_shape);
    } else {
      output = builder.MakeOutput();
    }

    builder.AddNode("NonZero", {input}, {output});
  };
}

template <typename DataType>
static void RunNonZeroTestOnCPU(const TestInputDef<DataType>& input_def,
                                const bool fix_shape,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  // Note that since QNN supported fixed-shape NonZero is in fact not align with ONNX opdef, it could not be executed
  // by CPU EP.
  RunQnnModelTest(BuildNonZeroTestCase<DataType>(input_def, fix_shape),
                  provider_options,
                  13,
                  expected_ep_assignment,
                  /*fp32_abs_err*/ 1e-5f,
                  /*log_severity*/ logging::Severity::kERROR,
                  /*verify_outputs*/ false);
}

// Test NonZero having static shape which is supported by QNN.
TEST_F(QnnCPUBackendTests, NonZero_StaticShape) {
  RunNonZeroTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                             true,
                             ExpectedEPNodeAssignment::All);
}

// Test NonZero having dynamic shape which is not supported by QNN.
TEST_F(QnnCPUBackendTests, NonZero_DynamicShape) {
  RunNonZeroTestOnCPU<float>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                             false,
                             ExpectedEPNodeAssignment::None);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

template <typename QuantType>
inline GetTestModelFn BuildQDQNonZeroTestCase(const TestInputDef<float>& input_def, const bool fix_shape) {
  return [input_def, fix_shape](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder,
                                                   input,
                                                   input_qparams.scale,
                                                   input_qparams.zero_point);

    NodeArg* output;
    if (fix_shape) {
      // Fix NonZero output shape to maximum possible size.
      const std::vector<int64_t>& shape = input_def.GetShape();
      std::vector<int64_t> output_shape;
      output_shape.push_back(static_cast<int64_t>(shape.size()));
      output_shape.push_back(std::accumulate(shape.begin(),
                                             shape.end(),
                                             static_cast<int64_t>(1),
                                             std::multiplies<int64_t>()));

      output = builder.MakeOutput<int64_t>(output_shape);
    } else {
      output = builder.MakeOutput();
    }

    builder.AddNode("NonZero", {input_qdq}, {output});
  };
}

template <typename QuantType>
static void RunQDQNonZeroTestOnHTP(const TestInputDef<float>& input_def,
                                   const bool fix_shape,
                                   ExpectedEPNodeAssignment expected_ep_assignment,
                                   int opset = 13) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Note that since QNN supported fixed-shape NonZero is in fact not align with ONNX opdef, it could not be executed
  // by CPU EP.
  RunQnnModelTestHTPNoVerify(BuildQDQNonZeroTestCase<QuantType>(input_def, fix_shape),
                             provider_options,
                             opset,
                             expected_ep_assignment);
}

// Test 8-bit NonZero having static shape which is supported by QNN.
TEST_F(QnnHTPBackendTests, NonZero_U8_StaticShape) {
  RunQDQNonZeroTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                  true,
                                  ExpectedEPNodeAssignment::All);
}

// Test 8-bit NonZero having dynamic shape which is not supported by QNN.
TEST_F(QnnHTPBackendTests, NonZero_U8_DynamicShape) {
  RunQDQNonZeroTestOnHTP<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                  false,
                                  ExpectedEPNodeAssignment::None);
}

// Test 16-bit NonZero having static shape which is supported by QNN.
TEST_F(QnnHTPBackendTests, NonZero_U16_StaticShape) {
  RunQDQNonZeroTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                   true,
                                   ExpectedEPNodeAssignment::All,
                                   21);
}

// Test 16-bit NonZero having dynamic shape which is not supported by QNN.
TEST_F(QnnHTPBackendTests, NonZero_U16_DynamicShape) {
  RunQDQNonZeroTestOnHTP<uint16_t>(TestInputDef<float>({1, 3, 4, 4}, false, GetFloatDataInRange(-10.0f, 10.0f, 48)),
                                   false,
                                   ExpectedEPNodeAssignment::None,
                                   21);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
