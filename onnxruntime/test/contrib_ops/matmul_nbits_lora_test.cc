// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)

TEST(MatMulNBitsLoRA, EmptyAdapterSkipsProjection) {
  for (const char* mode : {"", "auto", "fuse_down", "fuse_up"}) {
    SCOPED_TRACE(mode[0] ? mode : "default");
    for (bool transposed : {false, true}) {
      for (int64_t m : {1, 37}) {
        for (auto ranks : {std::pair{0, 0}, {0, 32}, {32, 0}, {32, 32}}) {
          auto ep = DefaultWebGpuExecutionProvider();
          ASSERT_NE(ep, nullptr);
          OpTester test("MatMulNBitsLoRA", 1, kMSDomain);
          test.AddAttribute<int64_t>("K", 64);
          test.AddAttribute<int64_t>("N", 128);
          test.AddAttribute<int64_t>("bits", 4);
          test.AddAttribute<int64_t>("block_size", 32);
          test.AddAttribute<int64_t>("adapter_transposed", transposed);
          test.AddInput<MLFloat16>("X", {1, m, 64}, std::vector<MLFloat16>(m * 64, MLFloat16(0.5f)));
          // Packed nibbles 9 => dequantized weight 1/32, so the base is exactly 1.
          test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(128 * 2 * 16, 0x99), true);
          test.AddInput<MLFloat16>("scales", {128, 2}, std::vector<MLFloat16>(256, MLFloat16(0.03125f)), true);
          test.AddInput<MLFloat16>("lora_A", transposed ? std::vector<int64_t>{ranks.first, 64} : std::vector<int64_t>{64, ranks.first},
                                   std::vector<MLFloat16>(64 * ranks.first, MLFloat16(0.0625f)));
          test.AddInput<MLFloat16>("lora_B", transposed ? std::vector<int64_t>{128, ranks.second} : std::vector<int64_t>{ranks.second, 128},
                                   std::vector<MLFloat16>(128 * ranks.second, MLFloat16(0.03125f)));
          const float expected = ranks.first && ranks.second ? 3.0f : 1.0f;
          test.AddOutput<MLFloat16>("Y", {1, m, 128}, std::vector<MLFloat16>(m * 128, MLFloat16(expected)));
          std::vector<std::unique_ptr<IExecutionProvider>> eps;
          eps.push_back(std::move(ep));
          SessionOptions options;
          if (mode[0]) {
            ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
          }
          test.Config(options);
          test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
        }
      }
    }
  }
}

TEST(MatMulNBitsLoRA, RejectInvalidAdapterShape) {
  auto ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(ep, nullptr);
  OpTester test("MatMulNBitsLoRA", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", 64);
  test.AddAttribute<int64_t>("N", 128);
  test.AddInput<MLFloat16>("X", {1, 64}, std::vector<MLFloat16>(64));
  test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(4096), true);
  test.AddInput<MLFloat16>("scales", {128, 2}, std::vector<MLFloat16>(256), true);
  test.AddInput<MLFloat16>("lora_A", {63, 32}, std::vector<MLFloat16>(63 * 32));
  test.AddInput<MLFloat16>("lora_B", {32, 128}, std::vector<MLFloat16>(32 * 128));
  test.AddOutput<MLFloat16>("Y", {1, 128}, std::vector<MLFloat16>(128));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(std::move(ep));
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid adapter dimensions", {}, nullptr, &eps);
}

#endif
}  // namespace onnxruntime::test
