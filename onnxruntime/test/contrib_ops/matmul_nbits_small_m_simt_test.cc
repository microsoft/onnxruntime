// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS) && !defined(__wasm__)

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <tuple>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/platform/env.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime::test {
namespace {

// Dyadic inputs keep the FP16 dequantization exact. The independent reference
// decodes the packed bytes and accumulates in FP64 before the final FP16 rounding.
void RunSmallMSimtCase(int64_t m, int64_t k, int64_t n, int64_t block_size,
                       bool has_zero_points, bool has_bias, uint32_t padding,
                       const std::string& mode = "simt", const std::string& failure = "",
                       uint32_t bn = 16, uint32_t bk = 512, bool rank3 = false) {
  const auto original_severity = logging::LoggingManager::DefaultLogger().GetSeverity();
  logging::LoggingManager::SetDefaultLoggerSeverity(logging::Severity::kINFO);
  auto restore_severity = gsl::finally([original_severity] {
    logging::LoggingManager::SetDefaultLoggerSeverity(original_severity);
  });
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is unavailable";
  }
  const ScopedEnvironmentVariables settings{{{"ORT_WEBGPU_SMALL_M_DIAGNOSTIC", mode == "unset" ? optional<std::string>{} : mode},
                                             {"ORT_WEBGPU_SMALL_M_BN", std::to_string(bn)},
                                             {"ORT_WEBGPU_SMALL_M_BK", std::to_string(bk)},
                                             {"ORT_WEBGPU_SMALL_M_A_BLOCK_PADDING", std::to_string(padding)}}};
  const int64_t blocks = (k + block_size - 1) / block_size;
  const int64_t zero_bytes = (blocks + 1) / 2;
  InlinedVector<MLFloat16> a(m * k);
  InlinedVector<uint8_t> b(n * blocks * block_size / 2);
  InlinedVector<MLFloat16> scales(n * blocks);
  InlinedVector<uint8_t> zeros(n * zero_bytes, 0);
  InlinedVector<MLFloat16> bias(n);
  for (int64_t i = 0; i < m * k; ++i) {
    a[i] = MLFloat16(static_cast<float>(i % 17 - 8) / 8.0f);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<uint8_t>((i * 29 + 7) % 256);
  }
  for (int64_t col = 0; col < n; ++col) {
    bias[col] = MLFloat16(static_cast<float>(col % 7 - 3) / 16.0f);
    for (int64_t block = 0; block < blocks; ++block) {
      scales[col * blocks + block] = MLFloat16(static_cast<float>(1 + block % 3) / 32.0f);
      zeros[col * zero_bytes + block / 2] |= static_cast<uint8_t>((6 + (col + block) % 5) << (4 * (block % 2)));
    }
  }
  InlinedVector<MLFloat16> expected(m * n);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      double sum = has_bias ? bias[col].ToFloat() : 0.0;
      for (int64_t inner = 0; inner < k; ++inner) {
        const auto packed = b[(col * blocks + inner / block_size) * (block_size / 2) +
                              (inner % block_size) / 2];
        const int weight = (packed >> (4 * (inner % 2))) & 15;
        const int zero = has_zero_points
                             ? (zeros[col * zero_bytes + (inner / block_size) / 2] >>
                                (4 * ((inner / block_size) % 2))) &
                                   15
                             : 8;
        sum += static_cast<double>(a[row * k + inner].ToFloat()) * (weight - zero) *
               scales[col * blocks + inner / block_size].ToFloat();
      }
      expected[row * n + col] = MLFloat16(static_cast<float>(sum));
    }
  }

  const auto directory = std::filesystem::temp_directory_path() /
                         ("ort_small_m_simt_" + std::to_string(Env::Default().NowMicros()));
  TemporaryDirectory profiles{directory.native(), false};
  SessionOptions session_options;
  session_options.session_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  session_options.enable_profiling = true;
  session_options.profile_file_prefix = (directory / "profile").native();
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test{"MatMulNBits", 1, kMSDomain};
  test.AddAttribute<int64_t>("K", k);
  test.AddAttribute<int64_t>("N", n);
  test.AddAttribute<int64_t>("bits", 4);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  const std::vector<int64_t> input_shape = rank3 ? std::vector<int64_t>{2, m / 2, k} : std::vector<int64_t>{m, k};
  const std::vector<int64_t> output_shape = rank3 ? std::vector<int64_t>{2, m / 2, n} : std::vector<int64_t>{m, n};
  test.AddInput<MLFloat16>("A", input_shape, a.data(), a.size());
  test.AddInput<uint8_t>("B", {n, blocks, block_size / 2}, b.data(), b.size(), true);
  test.AddInput<MLFloat16>("scales", {n, blocks}, scales.data(), scales.size(), true);
  if (has_zero_points) {
    test.AddInput<uint8_t>("zero_points", {n, zero_bytes}, zeros.data(), zeros.size(), true);
  } else {
    test.AddOptionalInputEdge<uint8_t>();
  }
  test.AddOptionalInputEdge<int32_t>();
  if (has_bias) {
    test.AddInput<MLFloat16>("bias", {n}, bias.data(), bias.size(), true);
  }
  test.AddOutput<MLFloat16>("Y", output_shape, expected.data(), expected.size());
  test.SetOutputTolerance(mode == "simt" ? 0.0f : 0.03125f, 0.0f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  RunOptions run_options;
  run_options.run_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  testing::internal::CaptureStderr();
  test.Run(session_options, failure.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
           failure, {}, &run_options, &providers);
  const auto log = testing::internal::GetCapturedStderr();
  if (!failure.empty()) {
    return;
  }

  bool saw_kernel = false;
  size_t offset = 0;
  while ((offset = log.find("Starting program \"", offset)) != std::string::npos) {
    offset += std::string_view{"Starting program \""}.size();
    const auto name = log.substr(offset, log.find_first_of("[:\"", offset) - offset);
    EXPECT_EQ(name == "MatMulNBitsSmallMSimt", mode == "simt") << log;
    if (mode == "regular") {
      EXPECT_EQ(name, "MatMulNBits") << log;
    }
    saw_kernel = true;
  }
  bool saw_node = false;
  for (const auto& path : std::filesystem::directory_iterator(directory)) {
    std::ifstream stream{path.path()};
    ASSERT_TRUE(stream.is_open());
    const auto profile = nlohmann::json::parse(stream);
    for (const auto& event : profile) {
      if (event.value("cat", "") == "Node" && event.contains("args") &&
          event["args"].contains("provider")) {
        EXPECT_EQ(event["args"]["provider"], kWebGpuExecutionProvider);
        saw_node = true;
      }
    }
  }
  EXPECT_TRUE(saw_kernel) << "No actual program dispatch log; routing is unverified.\n"
                          << log;
  EXPECT_TRUE(saw_node) << "No WebGPU node profiling event.";
}

using SmallMSimtParameters = std::tuple<int64_t, uint32_t>;
class MatMulNBitsSmallMSimtTest : public testing::TestWithParam<SmallMSimtParameters> {};

TEST_P(MatMulNBitsSmallMSimtTest, PackedReferenceAndActualRouting) {
  const auto [m, padding] = GetParam();
  RunSmallMSimtCase(m, m % 2 == 0 ? 100 : 513, 17 + m, m <= 4 ? 32 : 64,
                    m % 3 != 0, m % 2 == 0, padding, "simt", "", m <= 4 ? 8 : 16,
                    m <= 4 ? 256 : 512, m == 4 || m == 8);
}

INSTANTIATE_TEST_SUITE_P(BothStridesAndAllRows, MatMulNBitsSmallMSimtTest,
                         testing::Combine(testing::Range<int64_t>(1, 9), testing::Values(0u, 1u)));

TEST(MatMulNBitsSmallMSimtRoutingTest, DefaultsAndForcedRegular) {
  for (const auto& mode : {"unset", "auto", "regular"}) {
    SCOPED_TRACE(mode);
    RunSmallMSimtCase(4, 512, 64, 32, false, false, 0, mode);
  }
}

TEST(MatMulNBitsSmallMSimtRoutingTest, ExplicitUnsupportedSettingsFailInsteadOfFallingBack) {
  RunSmallMSimtCase(9, 64, 17, 32, false, false, 0, "simt", "M <= 8");
  RunSmallMSimtCase(4, 64, 17, 32, false, false, 2, "simt", "Unsupported diagnostic tile setting");
  RunSmallMSimtCase(4, 64, 17, 32, false, false, 0, "invalid",
                    "ORT_WEBGPU_SMALL_M_DIAGNOSTIC must be");
}

TEST(MatMulNBitsSmallMSimtRoutingTest, LargerTileAndBlock) {
  RunSmallMSimtCase(3, 1056, 33, 128, true, true, 1, "simt", "", 32, 1024);
}

}  // namespace
}  // namespace onnxruntime::test

#endif
