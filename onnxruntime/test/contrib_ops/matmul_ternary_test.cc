// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime::test {
namespace {

void RunT8(int64_t m, int64_t n, int64_t k, bool initializer, bool identity = false, bool batched = false,
           const char* packing = "current", const char* row_tile = "1", bool disable_prepacking = false) {
  const int64_t units = (n + 4) / 5, sets = k / 32, groups = k / 128;
  std::vector<uint32_t> planes(static_cast<size_t>(units * 8 * sets), 0);
  std::vector<MLFloat16> scales(static_cast<size_t>(units * groups * 5), MLFloat16(0.0f));
  std::vector<float> weights(static_cast<size_t>(n * k));
  for (int64_t q = 0; q < units; ++q) {
    for (int64_t i = 0; i < k; ++i) {
      uint32_t byte = 0, power = 1;
      // Across the first 243 inputs, exercise every possible five-trit byte.
      uint32_t digits = static_cast<uint32_t>((i + 17 * q) % 243);
      for (int64_t j = 0; j < 5; ++j, power *= 3, digits /= 3) {
        const int64_t row = 5 * q + j;
        const uint32_t trit = row < n ? digits % 3 : 0;
        byte += power * trit;
        if (row < n) {
          const float scale = static_cast<float>(1 + j + i / 128) * 0.125f;
          scales[static_cast<size_t>((q * groups + i / 128) * 5 + j)] = MLFloat16(scale);
          weights[static_cast<size_t>(row * k + i)] =
              (trit == 1 ? 1.0f : trit == 2 ? -1.0f
                                            : 0.0f) *
              scale;
        }
      }
      for (int64_t b = 0; b < 8; ++b) {
        planes[static_cast<size_t>((q * 8 + b) * sets + i / 32)] |= ((byte >> b) & 1u) << (i % 32);
      }
    }
  }
  std::vector<float> a(static_cast<size_t>(m * k), 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t i = 0; i < k; ++i) {
      a[static_cast<size_t>(row * k + i)] = identity ? (row == i ? 1.0f : 0.0f)
                                                     : static_cast<float>((i + row * 3) % 17 - 8) * 0.125f;
    }
  }
  std::vector<float> expected(static_cast<size_t>(m * n), 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      for (int64_t i = 0; i < k; ++i) {
        expected[static_cast<size_t>(row * n + col)] +=
            a[static_cast<size_t>(row * k + i)] * weights[static_cast<size_t>(col * k + i)];
      }
    }
  }
  OpTester test("MatMulTernary", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", k);
  test.AddAttribute<int64_t>("N", n);
  test.AddInput<float>("A", batched ? std::vector<int64_t>{1, m, k} : std::vector<int64_t>{m, k}, a);
  test.AddInput<uint32_t>("B", {units, 8, sets}, planes, initializer);
  test.AddInput<MLFloat16>("scales", {units, groups, 5}, scales, initializer);
  test.AddOutput<float>("Y", batched ? std::vector<int64_t>{1, m, n} : std::vector<int64_t>{m, n}, expected);
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigTernaryPrepack, packing));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigTernaryRowTile, row_tile));
  if (disable_prepacking) ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "1"));
  size_t prepacked_count = 0;
  test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps, {}, &prepacked_count);
  EXPECT_EQ(prepacked_count, initializer && std::string(packing) != "current" && !disable_prepacking ? 1u : 0u);
}

TEST(MatMulTernaryTest, EveryByteAndLaneExactly) { RunT8(256, 5, 256, true, true); }
TEST(MatMulTernaryTest, DecodePaddedRowsAndMultipleGroups) { RunT8(1, 7, 512, true); }
TEST(MatMulTernaryTest, DynamicWeightsAndBatchedPrefill) { RunT8(9, 17, 256, false, false, true); }
TEST(MatMulTernaryTest, EmptyActivations) { RunT8(0, 7, 128, true); }

TEST(MatMulTernaryTest, PackedLayoutsAndMaskReuse) {
  for (const char* packing : {"tiled", "masks"}) {
    for (const char* rows : {"1", "2"}) {
      SCOPED_TRACE(std::string(packing) + "/" + rows);
      RunT8(256, 5, 256, true, true, false, packing, rows);
      RunT8(1, 7, 512, true, false, false, packing, rows);
      RunT8(9, 17, 256, true, false, true, packing, rows);
      RunT8(9, 17, 256, false, false, true, packing, rows);
      RunT8(3, 7, 128, true, false, false, packing, rows, true);
      RunT8(0, 7, 128, true, false, false, packing, rows);
    }
  }
}

TEST(MatMulTernaryTest, RejectsInvalidPackingOptions) {
  for (const auto& entry : std::vector<std::pair<std::string, std::string>>{
           {kOrtSessionOptionsConfigTernaryPrepack, "invalid"},
           {kOrtSessionOptionsConfigTernaryRowTile, "3"},
           {kOrtSessionOptionsConfigTernaryRowTile, "2"}}) {
    OpTester test("MatMulTernary", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", 128);
    test.AddAttribute<int64_t>("N", 5);
    test.AddInput<float>("A", {1, 128}, std::vector<float>(128));
    test.AddInput<uint32_t>("B", {1, 8, 4}, std::vector<uint32_t>(32), true);
    test.AddInput<MLFloat16>("scales", {1, 1, 5}, std::vector<MLFloat16>(5));
    test.AddOutput<float>("Y", {1, 5}, std::vector<float>(5));
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(entry.first.c_str(), entry.second.c_str()));
    test.Run(options, OpTester::ExpectResult::kExpectFailure,
             entry.second == "2" ? "current T8 kernel requires row_tile 1"
                                 : entry.first + " must be");
  }
}

TEST(MatMulTernaryTest, VectorInput) {
  OpTester test("MatMulTernary", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", 128);
  test.AddAttribute<int64_t>("N", 1);
  test.AddInput<float>("A", {128}, std::vector<float>(128, 1.0f));
  test.AddInput<uint32_t>("B", {1, 8, 4}, std::vector<uint32_t>(32, 0));
  test.AddInput<MLFloat16>("scales", {1, 1, 5}, std::vector<MLFloat16>(5, MLFloat16(1.0f)));
  test.AddOutput<float>("Y", {1}, {0.0f});
  test.Run();
}

TEST(MatMulTernaryTest, RejectsAllInvalidBytes) {
  for (uint32_t byte = 243; byte < 256; ++byte) {
    for (bool initializer : {false, true}) {
      OpTester test("MatMulTernary", 1, kMSDomain);
      test.AddAttribute<int64_t>("K", 128);
      test.AddAttribute<int64_t>("N", 5);
      std::vector<uint32_t> planes(32, 0);
      for (size_t bit = 0; bit < 8; ++bit) planes[bit * 4] = ((byte >> bit) & 1u) << 31;
      test.AddInput<float>("A", {1, 128}, std::vector<float>(128));
      test.AddInput<uint32_t>("B", {1, 8, 4}, planes, initializer);
      test.AddInput<MLFloat16>("scales", {1, 1, 5}, std::vector<MLFloat16>(5, MLFloat16(1.0f)));
      test.AddOutput<float>("Y", {1, 5}, std::vector<float>(5));
      test.Run(OpTester::ExpectResult::kExpectFailure, "invalid T8 byte");
    }
  }
}

TEST(MatMulTernaryTest, RejectsUnsupportedVersion) {
  OpTester test("MatMulTernary", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", 128);
  test.AddAttribute<int64_t>("N", 5);
  test.AddAttribute<int64_t>("format_version", 2);
  test.AddInput<float>("A", {1, 128}, std::vector<float>(128));
  test.AddInput<uint32_t>("B", {1, 8, 4}, std::vector<uint32_t>(32));
  test.AddInput<MLFloat16>("scales", {1, 1, 5}, std::vector<MLFloat16>(5));
  test.AddOutput<float>("Y", {1, 5}, std::vector<float>(5));
  test.Run(OpTester::ExpectResult::kExpectFailure, "Unsupported MatMulTernary");
}

TEST(MatMulTernaryTest, RejectsPackedShapeMismatch) {
  OpTester test("MatMulTernary", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", 128);
  test.AddAttribute<int64_t>("N", 5);
  test.AddInput<float>("A", {1, 128}, std::vector<float>(128));
  test.AddInput<uint32_t>("B", {1, 8, 3}, std::vector<uint32_t>(24));
  test.AddInput<MLFloat16>("scales", {1, 1, 5}, std::vector<MLFloat16>(5));
  test.AddOutput<float>("Y", {1, 5}, std::vector<float>(5));
  test.Run(OpTester::ExpectResult::kExpectFailure, "invalid packed tensor dimension");
}

TEST(MatMulTernaryTest, RejectsActivationAndScaleShapes) {
  for (int invalid = 0; invalid < 3; ++invalid) {
    OpTester test("MatMulTernary", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", 128);
    test.AddAttribute<int64_t>("N", 5);
    if (invalid == 0) {
      test.AddInput<float>("A", {}, {0.0f});
    } else {
      const int64_t width = invalid == 1 ? 129 : 128;
      test.AddInput<float>("A", {1, width}, std::vector<float>(static_cast<size_t>(width)));
    }
    test.AddInput<uint32_t>("B", {1, 8, 4}, std::vector<uint32_t>(32));
    if (invalid == 2) {
      test.AddInput<MLFloat16>("scales", {1, 5}, std::vector<MLFloat16>(5));
    } else {
      test.AddInput<MLFloat16>("scales", {1, 1, 5}, std::vector<MLFloat16>(5));
    }
    test.AddOutput<float>("Y", {1, 5}, std::vector<float>(5));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             invalid == 0 ? "A must have rank >= 1" : invalid == 1 ? "A last dimension must equal K"
                                                                   : "B and scales must have rank 3");
  }
}

TEST(MatMulTernaryTest, RejectsUnsupportedLayout) {
  OpTester test("MatMulTernary", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", 128);
  test.AddAttribute<int64_t>("N", 5);
  test.AddAttribute<std::string>("layout", "invalid");
  test.AddInput<float>("A", {1, 128}, std::vector<float>(128));
  test.AddInput<uint32_t>("B", {1, 8, 4}, std::vector<uint32_t>(32));
  test.AddInput<MLFloat16>("scales", {1, 1, 5}, std::vector<MLFloat16>(5));
  test.AddOutput<float>("Y", {1, 5}, std::vector<float>(5));
  test.Run(OpTester::ExpectResult::kExpectFailure, "Unsupported MatMulTernary");
}

}  // namespace
}  // namespace onnxruntime::test
