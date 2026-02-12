// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "gtest/gtest.h"

#include "core/mlas/inc/mlas_q4.h"
#include "core/session/ort_env.h"
#include "test/providers/qnn/qnn_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(_M_ARM64)

// Re-implement testcases from test/contrib_ops/matmul_4bits_test.cc.

constexpr int QBits = 4;

void QuantizeDequantize(std::vector<float>& raw_vals,
                        std::vector<uint8_t>& quant_vals,
                        std::vector<float>& scales,
                        std::vector<uint8_t>* zp,
                        int32_t N,
                        int32_t K,
                        int32_t block_size) {
  auto& ortenv = **GetOrtEnv();
  onnxruntime::concurrency::ThreadPool* tp = ortenv.GetEnvironment().GetIntraOpThreadPool();

  MlasQuantizeBlockwise<float, QBits>(quant_vals.data(),
                                      scales.data(),
                                      zp != nullptr ? zp->data() : nullptr,
                                      raw_vals.data(),
                                      block_size,
                                      true,
                                      K,
                                      N,
                                      N,
                                      tp);

  // Note that raw_vals is NxK after dequant
  MlasDequantizeBlockwise<float, QBits>(raw_vals.data(),                       // dequantized output
                                        quant_vals.data(),                     // quantized input
                                        scales.data(),                         // quantization scales
                                        zp != nullptr ? zp->data() : nullptr,  // quantization zero points
                                        block_size,                            // quantization block size
                                        true,                                  // columnwise quantization
                                        K,                                     // number of rows
                                        N,                                     // number of columns
                                        tp);
}

struct TestParams4Bits {
  int64_t batch_count{1};
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t block_size{32};
  int64_t accuracy_level{4};

  bool has_zero_point{false};
  bool zp_is_4bit{true};
  bool has_g_idx{false};
  bool has_bias{false};
};

static void RunMatMul4BitsTest(const TestParams4Bits params,
                               ExpectedEPNodeAssignment expected_ep_assignment = ExpectedEPNodeAssignment::All,
                               const std::string& backend_name = "gpu",
                               float fp32_abs_err = 0.05f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  auto model_builder = [&params](ModelTestBuilder& builder) {
    std::vector<NodeArg*> inputs;

    RandomValueGenerator random{1234};
    std::vector<float> input0_vals(random.Gaussian<float>(AsSpan({params.batch_count, params.M, params.K}),
                                                          0.0f,
                                                          0.25f));
    std::vector<float> input1_f_vals(random.Gaussian<float>(AsSpan({params.K, params.N}), 0.0f, 0.25f));

    auto input0_def = TestInputDef<float>({params.batch_count, params.M, params.K}, false, input0_vals);
    inputs.push_back(MakeTestInput<float>(builder, input0_def));

    int64_t k_blocks = (params.K + params.block_size - 1) / params.block_size;
    int64_t blob_size = (params.block_size * QBits + 7) / 8;
    size_t q_scale_size = static_cast<size_t>(params.N * k_blocks);
    size_t q_data_size_in_bytes = static_cast<size_t>(params.N * k_blocks * blob_size);  // packed as UInt4x2
    const int64_t zero_point_blob_size = (k_blocks * QBits + 7) / 8;
    size_t q_zp_size_in_bytes = static_cast<size_t>(params.N * zero_point_blob_size);  // packed as UInt4x2

    std::vector<uint8_t> input1_vals(q_data_size_in_bytes);
    std::vector<float> scales(q_scale_size);
    // TODO
    // Not sure why zp is not calculated from QuantizeDequantize. Since QNN GPU only support zp=8, hardcode it here
    // as workaround.
    std::vector<uint8_t> zp(q_zp_size_in_bytes, 0b10001000);

    QuantizeDequantize(input1_f_vals,
                       input1_vals,
                       scales,
                       nullptr,  // params.has_zero_point ? &zp : nullptr,
                       static_cast<int32_t>(params.N),
                       static_cast<int32_t>(params.K),
                       static_cast<int32_t>(params.block_size));

    auto input1_def = TestInputDef<uint8_t>({params.N, k_blocks, blob_size}, true, input1_vals);
    inputs.push_back(MakeTestInput<uint8_t>(builder, input1_def));

    auto scales_def = TestInputDef<float>({params.N, k_blocks}, true, scales);
    inputs.push_back(MakeTestInput<float>(builder, scales_def));

    if (params.has_zero_point) {
      auto zp_def = TestInputDef<uint8_t>({params.N, zero_point_blob_size}, true, zp);
      inputs.push_back(MakeTestInput<uint8_t>(builder, zp_def));
    }

    auto* output = builder.MakeOutput();

    Node& node = builder.AddNode("MatMulNBits", inputs, {output}, kMSDomain);
    node.AddAttribute("K", static_cast<int64_t>(params.K));
    node.AddAttribute("N", static_cast<int64_t>(params.N));
    node.AddAttribute("block_size", static_cast<int64_t>(params.block_size));
    node.AddAttribute("bits", static_cast<int64_t>(QBits));
    node.AddAttribute("accuracy_level", static_cast<int64_t>(params.accuracy_level));
  };

  RunQnnModelTest(model_builder,
                  provider_options,
                  13,  // opset version for contrib ops
                  expected_ep_assignment,
                  fp32_abs_err);
}

// QNN GPU only support FP16 activations and Q4_0 weights, with zero_points = 8
// Accumulation with larger channel accumulates more error. Set higher abs_error with respect to K.
TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M1_N128_K512_withZp) {
  TestParams4Bits params;
  params.M = 1;
  params.N = 128;
  params.K = 512;
  params.has_zero_point = true;
  RunMatMul4BitsTest(params);
}

TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M1_N128_K512) {
  TestParams4Bits params;
  params.M = 1;
  params.N = 128;
  params.K = 512;
  params.has_zero_point = false;
  RunMatMul4BitsTest(params);
}

TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M10_N128_K512_withZp) {
  TestParams4Bits params;
  params.M = 10;
  params.N = 128;
  params.K = 512;
  params.has_zero_point = true;
  RunMatMul4BitsTest(params);
}

TEST_F(QnnGPUBackendTests, MatMulNBits_Basic_M10_N128_K512) {
  TestParams4Bits params;
  params.M = 10;
  params.N = 128;
  params.K = 512;
  params.has_zero_point = false;
  RunMatMul4BitsTest(params);
}
#endif

}  // namespace test
}  // namespace onnxruntime

#endif
