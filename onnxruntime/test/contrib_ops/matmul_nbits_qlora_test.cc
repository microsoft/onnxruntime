// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)

TEST(MatMulNBitsQLoRA, EmptyAndActiveAdapters) {
  for (const char* mode : {"auto", "fuse_down", "fuse_up"}) {
    for (int64_t m : {1, 37}) {
      for (auto ranks : {std::pair{0, 0}, {0, 1}, {32, 0}, {32, 1}}) {
        auto ep = DefaultWebGpuExecutionProvider();
        ASSERT_NE(ep, nullptr);
        OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
        test.AddAttribute<int64_t>("K", 64);
        test.AddAttribute<int64_t>("N", 128);
        test.AddInput<MLFloat16>("X", {1, m, 64}, std::vector<MLFloat16>(m * 64, MLFloat16(0.5f)));
        test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(4096, 0x99), true);
        test.AddInput<MLFloat16>("S", {128, 2}, std::vector<MLFloat16>(256, MLFloat16(0.03125f)), true);
        test.AddInput<uint8_t>("AQ", {ranks.first, 2, 32}, std::vector<uint8_t>(ranks.first * 64, 129));
        test.AddInput<MLFloat16>("AS", {32, 2}, std::vector<MLFloat16>(64, MLFloat16(0.03125f)));
        test.AddInput<uint8_t>("BQ", {128, ranks.second, 32}, std::vector<uint8_t>(128 * ranks.second * 32, 129));
        test.AddInput<MLFloat16>("BS", {128, 1}, std::vector<MLFloat16>(128, MLFloat16(0.0625f)));
        const float expected = ranks.first && ranks.second ? 3.0f : 1.0f;
        test.AddOutput<MLFloat16>("Y", {1, m, 128}, std::vector<MLFloat16>(m * 128, MLFloat16(expected)));
        SessionOptions options;
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
        test.Config(options);
        std::vector<std::unique_ptr<IExecutionProvider>> eps;
        eps.push_back(std::move(ep));
        test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
      }
    }
  }
}

TEST(MatMulNBitsQLoRA, RejectsUnsupportedAdapterShapeAndBits) {
  for (bool wrong_bits : {false, true}) {
    auto ep = DefaultWebGpuExecutionProvider();
    ASSERT_NE(ep, nullptr);
    OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", 64);
    test.AddAttribute<int64_t>("N", 128);
    if (wrong_bits) test.AddAttribute<int64_t>("lora_bits", 4);
    test.AddInput<MLFloat16>("X", {1, 1, 64}, std::vector<MLFloat16>(64, MLFloat16(0.5f)));
    test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(4096, 0x99), true);
    test.AddInput<MLFloat16>("S", {128, 2}, std::vector<MLFloat16>(256, MLFloat16(0.03125f)), true);
    const int64_t rank = wrong_bits ? 32 : 16;
    test.AddInput<uint8_t>("AQ", {rank, 2, 32}, std::vector<uint8_t>(rank * 64, 129));
    test.AddInput<MLFloat16>("AS", {32, 2}, std::vector<MLFloat16>(64, MLFloat16(0.03125f)));
    test.AddInput<uint8_t>("BQ", {128, 1, 32}, std::vector<uint8_t>(4096, 129));
    test.AddInput<MLFloat16>("BS", {128, 1}, std::vector<MLFloat16>(128, MLFloat16(0.0625f)));
    test.AddOutput<MLFloat16>("Y", {1, 1, 128}, std::vector<MLFloat16>(128, MLFloat16(3.0f)));
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             wrong_bits ? "Unsupported MatMulNBitsQLoRA attributes" : "Invalid quantized adapter dimensions", {},
             nullptr, &eps);
  }
}

#endif
}  // namespace onnxruntime::test
