// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <vector>
#include <cmath>
#include <optional>
#include <utility>
#include <array>
#include <memory>
#include <unordered_map>

#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/util/include/qdq_test_utils.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64)

namespace {

GetQDQTestCaseFn BuildLPBQGemmTestCase() {
  return [](ModelTestBuilder& builder) -> void {
    // Define the test case for LPBQGemm fusion here
    const int64_t input_channels = 16;
    const int64_t output_channels = 16;
    const int64_t blocks_per_axis = 4;
    const std::vector<int64_t> input_shape{1, input_channels};
    auto input_def = TestInputDef<float>(input_shape, false, -0.5f, 0.5f);
    NodeArg* input = MakeTestInput<float>(builder, input_def);

    // QuantizeLinear for Activation
    NodeArg* act_ql_output = builder.MakeIntermediate();
    NodeArg* act_ql_scale = builder.MakeScalarInitializer<float>(0.00005509183756657876f);
    NodeArg* act_ql_zero_point = builder.MakeScalarInitializer<uint16_t>(23715);
    builder.AddNode("QuantizeLinear", {input, act_ql_scale, act_ql_zero_point}, {act_ql_output});

    // DequantizeLinear for Activation
    NodeArg* act_dql_output = builder.MakeIntermediate();
    NodeArg* act_dql_scale = builder.MakeScalarInitializer<float>(0.00005509183756657876f);
    NodeArg* act_dql_zero_point = builder.MakeScalarInitializer<uint16_t>(23715);
    builder.AddNode("DequantizeLinear", {act_ql_output, act_dql_scale, act_dql_zero_point}, {act_dql_output});

    // DequantizeLinear for Scale
    NodeArg* scale_dql_input = builder.MakeInitializer<uint8_t>({blocks_per_axis, output_channels}, 1, 15);
    NodeArg* scale_dql_scale = builder.MakeInitializer<float>({output_channels}, 0.01f, 0.02f);
    std::vector<uint8_t> dql_zero_points_data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    NodeArg* scale_dql_zero_point = builder.Make1DInitializer<uint8_t>(dql_zero_points_data);
    NodeArg* scale_dql_output = builder.MakeIntermediate();
    Node& scale_dql = builder.AddNode("DequantizeLinear", {scale_dql_input, scale_dql_scale, scale_dql_zero_point}, {scale_dql_output});
    scale_dql.AddAttribute("axis", static_cast<int64_t>(1));

    // QuantizeLinear for Weight
    NodeArg* w_ql_input = builder.MakeInitializer<float>({input_channels, output_channels}, -1.0f, 1.0f);
    std::vector<Int4x2> zero_points_data;
    size_t num_storage_elems = blocks_per_axis * output_channels;
    zero_points_data.resize(Int4x2::CalcNumInt4Pairs(num_storage_elems));
    for (size_t i = 0; i < num_storage_elems; ++i) {
      size_t r = i >> 1;
      size_t c = i & 0x1;
      zero_points_data[r].SetElem(c, 0);
    }
    NodeArg* w_ql_zero_point = builder.MakeInitializer<Int4x2>({blocks_per_axis, output_channels}, zero_points_data);
    NodeArg* w_ql_output = builder.MakeIntermediate();
    Node& w_ql = builder.AddNode("QuantizeLinear", {w_ql_input, scale_dql_output, w_ql_zero_point}, {w_ql_output});
    w_ql.AddAttribute("axis", static_cast<int64_t>(0));
    w_ql.AddAttribute("block_size", static_cast<int64_t>(4));

    // DequantizeLinear for Weight
    NodeArg* w_dql_zero_point = builder.MakeInitializer<Int4x2>({blocks_per_axis, output_channels}, zero_points_data);
    NodeArg* w_dql_output = builder.MakeIntermediate();
    Node& w_dql = builder.AddNode("DequantizeLinear", {w_ql_output, scale_dql_output, w_dql_zero_point}, {w_dql_output});
    w_dql.AddAttribute("axis", static_cast<int64_t>(0));
    w_dql.AddAttribute("block_size", static_cast<int64_t>(4));

    // Gemm
    NodeArg* gemm_bias = builder.MakeInitializer<float>({output_channels}, -1.0f, 1.0f);
    NodeArg* gemm_output = builder.MakeIntermediate();
    builder.AddNode("Gemm", {act_dql_output, w_dql_output, gemm_bias}, {gemm_output});

    // QuantizeLinear for Output
    NodeArg* output_ql_scale = builder.MakeScalarInitializer<float>(0.00019595865160226822f);
    NodeArg* output_ql_zero_point = builder.MakeScalarInitializer<uint16_t>(31693);
    NodeArg* output_ql_output = builder.MakeIntermediate();
    builder.AddNode("QuantizeLinear", {gemm_output, output_ql_scale, output_ql_zero_point}, {output_ql_output});

    // DequantizeLinear for Output
    NodeArg* output_dql_scale = builder.MakeScalarInitializer<float>(0.00019595865160226822f);
    NodeArg* output_dql_zero_point = builder.MakeScalarInitializer<uint16_t>(31693);
    NodeArg* output_dql_output = builder.MakeOutput();
    builder.AddNode("DequantizeLinear", {output_ql_output, output_dql_scale, output_dql_zero_point}, {output_dql_output});
  };
}

ProviderOptions GetProviderOptions() {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";
  return provider_options;
}

}  // namespace

TEST_F(QnnHTPBackendTests, LPBQGemmFusion) {
  ProviderOptions provider_options = GetProviderOptions();
  RunQnnModelTest(BuildLPBQGemmTestCase(),
                  provider_options,
                  /*opset_version=*/21,
                  /*expected_ep_assignment=*/ExpectedEPNodeAssignment::Some,
                  /*fp32_abs_err=*/1e-2f,
                  /*log_severity =*/logging::Severity::kERROR,
                  /*verify_outputs=*/false);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
