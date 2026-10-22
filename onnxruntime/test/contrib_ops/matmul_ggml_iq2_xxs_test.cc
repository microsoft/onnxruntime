// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/graph/model.h"

#include "contrib_ops/cuda/quantization/iq2_xxs_grid.h"
#if defined(USE_CUDA)
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/graph_optimizer_registry.h"
#include "core/providers/cuda/cuda_provider_options.h"
#endif

namespace onnxruntime::test {
namespace {

// Scalar reference follows canonical dequantize_row_iq2_xxs, independently of CUDA lane mapping:
// ggml-org/llama.cpp@b76b13379d3b7d67ab0bc424516bb6fd2522bd8e,
// ggml/src/ggml-quants.c:2489-2514. See ContribOperators.md for the pinned URL.
std::vector<float> Decode(const std::vector<uint8_t>& bytes) {
  std::vector<float> values;
  values.reserve(bytes.size() / 66 * 256);
  for (size_t block = 0; block < bytes.size() / 66; ++block) {
    const auto* p = bytes.data() + block * 66;
    const float d = MLFloat16::FromBits(static_cast<uint16_t>(p[0] | (p[1] << 8))).ToFloat();
    for (int group = 0; group < 8; ++group) {
      const auto* record = p + 2 + group * 8;
      const uint32_t aux = record[4] | (uint32_t{record[5]} << 8) |
                           (uint32_t{record[6]} << 16) | (uint32_t{record[7]} << 24);
      const float scale = d * (0.5f + (aux >> 28)) * 0.25f;
      for (int vector = 0; vector < 4; ++vector) {
        unsigned sign_bits = (aux >> (7 * vector)) & 127;
        unsigned parity = 0;
        for (int bit = 0; bit < 7; ++bit) {
          parity ^= (sign_bits >> bit) & 1;
        }
        sign_bits |= parity << 7;
        const uint64_t grid = contrib::cuda::kIQ2XXSGrid[record[vector]];
        for (int j = 0; j < 8; ++j) {
          const float magnitude = static_cast<float>((grid >> (8 * j)) & 255);
          values.push_back(scale * magnitude * ((sign_bits & (1u << j)) ? -1.0f : 1.0f));
        }
      }
    }
  }
  return values;
}

std::vector<uint8_t> Pack(int64_t n, int64_t k_blocks) {
  constexpr std::array<uint16_t, 8> scale_bits{0x2c00, 0x3401, 0xb001, 0x0000,
                                               0x0001, 0x0400, 0x3080, 0x3555};
  std::vector<uint8_t> bytes(static_cast<size_t>(n * k_blocks * 66));
  for (int64_t block = 0; block < n * k_blocks; ++block) {
    auto* p = bytes.data() + block * 66;
    const auto d = scale_bits[static_cast<size_t>(block) % scale_bits.size()];
    p[0] = static_cast<uint8_t>(d);
    p[1] = static_cast<uint8_t>(d >> 8);
    for (int group = 0; group < 8; ++group) {
      auto* record = p + 2 + group * 8;
      uint32_t aux = static_cast<uint32_t>((block + group) % 16) << 28;
      for (int vector = 0; vector < 4; ++vector) {
        record[vector] = static_cast<uint8_t>(block * 32 + group * 4 + vector);
        aux |= static_cast<uint32_t>((block * 32 + group * 4 + vector) % 128) << (7 * vector);
      }
      for (int j = 0; j < 4; ++j) {
        record[4 + j] = static_cast<uint8_t>(aux >> (8 * j));
      }
    }
  }
  return bytes;
}

template <typename T>
void RunCase(const std::vector<int64_t>& shape, int64_t n, int64_t k_blocks, bool use_bias) {
  const int64_t m = std::accumulate(shape.begin(), shape.end() - 1, int64_t{1},
                                    std::multiplies<int64_t>{});
  const int64_t k = k_blocks * 256;
  const auto bytes = Pack(n, k_blocks);
  const auto weights = Decode(bytes);
  std::vector<T> a(static_cast<size_t>(m * k));
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = T(static_cast<float>(static_cast<int>(i % 31) - 15) / 32.0f);
  }
  std::vector<T> bias(static_cast<size_t>(n));
  for (int64_t col = 0; col < n; ++col) {
    bias[static_cast<size_t>(col)] = T(static_cast<float>(col % 7) / 16.0f);
  }
  std::vector<T> expected;
  expected.reserve(static_cast<size_t>(m * n));
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0;
      for (int64_t kk = 0; kk < k; ++kk) {
        sum = std::fma(static_cast<float>(a[static_cast<size_t>(row * k + kk)]),
                       static_cast<float>(T(weights[static_cast<size_t>(col * k + kk)])), sum);
      }
      const T rounded(sum);
      expected.emplace_back(use_bias ? static_cast<float>(rounded) + static_cast<float>(bias[static_cast<size_t>(col)])
                                     : static_cast<float>(rounded));
    }
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<T>("A", shape, a);
  test.AddInput<uint8_t>("B", {n, k_blocks, 66}, bytes);
  if (use_bias) {
    test.AddInput<T>("bias", {n}, bias);
  }
  auto output_shape = shape;
  output_shape.back() = n;
  test.AddOutput<T>("Y", output_shape, expected);
  // FP32 accumulation order differs between the scalar reference and warp tree reduction.
  test.SetOutputTolerance(std::is_same_v<T, float> ? 0.0005f : 0.03125f,
                          std::is_same_v<T, float> ? 0.00002f : 0.008f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace

#if defined(USE_CUDA)
TEST(MatMulGGMLIQ2XXS, DecodeFloatAllGridSignsScales) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  // 256 columns x two superblocks visit every grid and sign index and all local scales.
  RunCase<float>({1, 512}, 256, 2, false);
}

TEST(MatMulGGMLIQ2XXS, PrefillFloatOddLeadingDimensions) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<float>({3, 5, 768}, 7, 3, true);
}

TEST(MatMulGGMLIQ2XXS, DecodeFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<MLFloat16>({512}, 9, 2, true);
}

TEST(MatMulGGMLIQ2XXS, PrefillFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<MLFloat16>({17, 512}, 13, 2, false);
}

TEST(MatMulGGMLIQ2XXS, DecodeBFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<BFloat16>({1, 512}, 9, 2, true);
}

TEST(MatMulGGMLIQ2XXS, PrefillBFloat16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<BFloat16>({3, 7, 512}, 13, 2, true);
}

TEST(MatMulGGMLIQ2XXS, EmptyDimensionsAndEmptyContraction) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunCase<float>({0, 256}, 3, 1, false);
  RunCase<float>({2, 256}, 0, 1, false);
  RunCase<float>({3, 0}, 5, 0, true);
  RunCase<MLFloat16>({0, 256}, 3, 1, true);
  RunCase<MLFloat16>({2, 256}, 0, 1, false);
  RunCase<MLFloat16>({3, 0}, 5, 0, true);
  RunCase<BFloat16>({0, 256}, 3, 1, true);
  RunCase<BFloat16>({2, 256}, 0, 1, false);
  RunCase<BFloat16>({3, 0}, 5, 0, true);
}

TEST(MatMulGGMLIQ2XXS, EveryGridElementInCanonicalOrder) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  constexpr int64_t n = 256;
  std::vector<uint8_t> bytes(n * 66, 0);
  std::vector<float> a(8 * 256, 0);
  std::vector<float> expected(8 * n);
  for (int j = 0; j < 8; ++j) {
    a[j * 256 + j] = 1;
    for (int col = 0; col < n; ++col) {
      bytes[col * 66 + 1] = 0x48;  // d=8, local_scale=0 -> scale=1.
      bytes[col * 66 + 2] = static_cast<uint8_t>(col);
      expected[j * n + col] = static_cast<float>((contrib::cuda::kIQ2XXSGrid[col] >> (8 * j)) & 255);
    }
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<float>("A", {8, 256}, a);
  test.AddInput<uint8_t>("B", {n, 1, 66}, bytes);
  test.AddOutput<float>("Y", {8, n}, expected);
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(MatMulGGMLIQ2XXS, RoundsReconstructedWeightsBeforeMultiplication) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  std::vector<uint8_t> bytes(66, 0);
  bytes[0] = 0x55;
  bytes[1] = 0x35;  // d=0.333251953125
  bytes[2] = 14;    // First vector is [8,43,25,25,8,8,8,8].
  // The FP32 weights cancel exactly; native reconstructed weights must not.
  const auto run = [&]<typename T>() {
    std::vector<T> a(256, T(0.0f));
    a[1] = T(25.0f);
    a[2] = T(-43.0f);
    OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
    test.AddInput<T>("A", {1, 256}, a);
    test.AddInput<uint8_t>("B", {1, 1, 66}, bytes);
    const T expected(25.0f * static_cast<float>(T(43.0f * 0.333251953125f / 8.0f)) -
                     43.0f * static_cast<float>(T(25.0f * 0.333251953125f / 8.0f)));
    if constexpr (!std::is_same_v<T, float>) {
      ASSERT_NE(static_cast<float>(expected), 0.0f);
    }
    test.AddOutput<T>("Y", {1, 1}, {expected});
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<float>();
  run.template operator()<MLFloat16>();
  run.template operator()<BFloat16>();
}

TEST(MatMulGGMLIQ2XXS, RoundsMatMulOutputBeforeBias) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  // d=1 and grid0=8 produce exactly unit weights.
  std::vector<uint8_t> bytes(66, 0);
  bytes[1] = 0x3c;
  const auto run = [&]<typename T>(float half_ulp) {
    std::vector<T> a(256, T(0.0f));
    a[0] = T(1.0f);
    a[8] = T(half_ulp);
    OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
    test.AddInput<T>("A", {1, 256}, a);
    test.AddInput<uint8_t>("B", {1, 1, 66}, bytes);
    test.AddInput<T>("bias", {1}, {T(-1.0f)});
    test.AddOutput<T>("Y", {1, 1}, {T(0.0f)});
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<float>(0x1p-24f);
  run.template operator()<MLFloat16>(0x1p-11f);
  run.template operator()<BFloat16>(0x1p-8f);
}

TEST(MatMulGGMLIQ2XXS, RejectsInvalidShapes) {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{1, 1, 65}, {1, 66}}) {
    OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
    test.AddInput<float>("A", {1, 256}, std::vector<float>(256));
    const auto count = std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                       std::multiplies<int64_t>{});
    test.AddInput<uint8_t>("B", shape, std::vector<uint8_t>(static_cast<size_t>(count)));
    test.AddOutput<float>("Y", {1, 1}, {0});
    test.Run(OpTester::ExpectResult::kExpectFailure, "B must have shape");
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<float>("A", {1, 255}, std::vector<float>(255));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<float>("Y", {1, 1}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "A's last dimension");
}

TEST(MatMulGGMLIQ2XXS, RejectsInvalidBiasAndScalarActivation) {
  OpTester bias_test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  bias_test.AddInput<float>("A", {1, 256}, std::vector<float>(256));
  bias_test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  bias_test.AddInput<float>("bias", {2}, {0, 0});
  bias_test.AddOutput<float>("Y", {1, 1}, {0});
  bias_test.Run(OpTester::ExpectResult::kExpectFailure, "bias must have shape");
  OpTester scalar_test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  scalar_test.AddInput<float>("A", {}, {0});
  scalar_test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  scalar_test.AddOutput<float>("Y", {1}, {0});
  scalar_test.Run(OpTester::ExpectResult::kExpectFailure, "A must have rank");
}

TEST(MatMulGGMLIQ2XXS, RuntimeShapeValidation) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {1, 255}, std::vector<float>(255));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<float>("Y", {1, 1}, {0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "A's last dimension", {}, nullptr, &providers);
}

TEST(MatMulGGMLIQ2XXS, RejectsUnsupportedActivationType) {
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<double>("A", {1, 256}, std::vector<double>(256));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<double>("Y", {1, 1}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "invalid");
}

TEST(MatMulGGMLIQ2XXS, NoCpuFallback) {
  OpTester test("MatMulGGMLIQ2XXS", 1, kMSDomain);
  test.AddInput<float>("A", {1, 256}, std::vector<float>(256));
  test.AddInput<uint8_t>("B", {1, 1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<float>("Y", {1, 1}, {0});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "Could not find an implementation",
           {}, nullptr, &providers);
}
#endif

namespace {

void IQ2MoEAttributes(OpTester& test, int64_t k) {
  test.AddAttribute("quant_type", std::string("iq2_xxs"));
  test.AddAttribute("activation_type", std::string("relu2"));
  test.AddAttribute("expert_weight_bits", int64_t{2});
  test.AddAttribute("block_size", int64_t{256});
  test.AddAttribute("weights_prepacked", int64_t{0});
  test.AddAttribute("k", k);
}

template <typename T>
void IQ2MoEInputs(OpTester& test, const std::vector<int64_t>& shape, const std::vector<T>& input,
                  int64_t experts, int64_t inter, int64_t k, const std::vector<uint8_t>& up,
                  const std::vector<uint8_t>& down, const std::vector<int64_t>& ids,
                  const std::vector<float>& weights, bool initializers = true) {
  const int64_t hidden = shape.back();
  const int64_t tokens = std::accumulate(shape.begin(), shape.end() - 1, int64_t{1},
                                         std::multiplies<int64_t>{});
  test.AddInput<T>("input", shape, input);
  test.AddOptionalInputEdge<T>();
  test.AddInput<uint8_t>("up", {experts, inter, hidden / 256, 66}, up, initializers);
  test.AddOptionalInputEdge<T>();
  test.AddOptionalInputEdge<T>();
  test.AddInput<uint8_t>("down", {experts, hidden, inter / 256, 66}, down, initializers);
  for (int index = 6; index <= 20; ++index) {
    test.AddOptionalInputEdge<T>();
  }
  test.AddInput<int64_t>("selected_experts", {tokens, k}, ids);
  test.AddInput<float>("selected_expert_weights", {tokens, k}, weights);
}

template <typename T>
std::vector<T> IQ2MoEOracle(const std::vector<T>& input, int64_t experts, int64_t hidden,
                            int64_t inter, int64_t k, const std::vector<uint8_t>& up_bytes,
                            const std::vector<uint8_t>& down_bytes, const std::vector<int64_t>& ids,
                            const std::vector<float>& weights) {
  const auto up = Decode(up_bytes);
  const auto down = Decode(down_bytes);
  const size_t tokens = input.size() / static_cast<size_t>(hidden);
  std::vector<T> result;
  result.reserve(input.size());
  for (size_t token = 0; token < tokens; ++token) {
    std::vector<float> total(static_cast<size_t>(hidden), 0.0f);
    // Independently enumerate experts, not the CUDA route sorting or lane reduction.
    for (int64_t expert = 0; expert < experts; ++expert) {
      const auto begin = ids.begin() + token * k;
      const auto found = std::find(begin, begin + k, expert);
      if (found == begin + k) {
        continue;
      }
      const float weight = weights[static_cast<size_t>(found - ids.begin())];
      std::vector<T> activated(static_cast<size_t>(inter));
      for (int64_t n = 0; n < inter; ++n) {
        float sum = 0.0f;
        for (int64_t j = 0; j < hidden; ++j) {
          sum = std::fma(static_cast<float>(input[token * hidden + j]),
                         static_cast<float>(T(up[static_cast<size_t>((expert * inter + n) * hidden + j)])), sum);
        }
        const float native = static_cast<float>(T(sum));
        const float relu = native < 0.0f ? 0.0f : native;
        activated[static_cast<size_t>(n)] = T(relu * relu);
      }
      for (int64_t n = 0; n < hidden; ++n) {
        float sum = 0.0f;
        for (int64_t j = 0; j < inter; ++j) {
          sum = std::fma(static_cast<float>(activated[static_cast<size_t>(j)]),
                         static_cast<float>(T(down[static_cast<size_t>((expert * hidden + n) * inter + j)])), sum);
        }
        const volatile float product = static_cast<float>(T(sum)) * weight;
        total[static_cast<size_t>(n)] += product;
      }
    }
    for (const auto value : total) {
      result.emplace_back(value);
    }
  }
  return result;
}

template <typename T>
void RunIQ2MoE(const std::vector<int64_t>& shape, int64_t experts, int64_t inter, int64_t k,
               bool permute, bool initializers = true) {
  const int64_t hidden = shape.back();
  const int64_t tokens = std::accumulate(shape.begin(), shape.end() - 1, int64_t{1},
                                         std::multiplies<int64_t>{});
  auto up = Pack(experts * inter, hidden / 256);
  auto down = Pack(experts * hidden, inter / 256);
  for (auto* bank : {&up, &down}) {
    for (size_t offset = 0; offset < bank->size(); offset += 66) {
      const auto bits = static_cast<uint16_t>((*bank)[offset] | ((*bank)[offset + 1] << 8));
      const auto scaled = MLFloat16(MLFloat16::FromBits(bits).ToFloat() / 16.0f).val;
      (*bank)[offset] = static_cast<uint8_t>(scaled);
      (*bank)[offset + 1] = static_cast<uint8_t>(scaled >> 8);
    }
  }
  std::vector<T> input(static_cast<size_t>(tokens * hidden));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = T(static_cast<float>(static_cast<int>(i % 31) - 15) / 512.0f);
  }
  std::vector<int64_t> ids(static_cast<size_t>(tokens * k));
  std::vector<float> weights(ids.size());
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t r = 0; r < k; ++r) {
      const int64_t route = permute ? k - 1 - r : r;
      ids[static_cast<size_t>(token * k + r)] = (route + token) % experts;
      weights[static_cast<size_t>(token * k + r)] = static_cast<float>(route + 1) * 0.1234567f;
    }
  }
  const auto expected = IQ2MoEOracle(input, experts, hidden, inter, k, up, down, ids, weights);
  OpTester test("QMoE", 1, kMSDomain);
  IQ2MoEAttributes(test, k);
  test.AddShapeToTensorData(false);
  IQ2MoEInputs(test, shape, input, experts, inter, k, up, down, ids, weights, initializers);
  test.AddOutput<T>("output", shape, expected);
  test.SetOutputTolerance(std::is_same_v<T, float> ? 0.00001f : 0.0005f,
                          std::is_same_v<T, float> ? 0.0001f : 0.02f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace

#if defined(USE_CUDA)
TEST(QMoEIQ2XXS, FloatCapabilityPreservesLegacyCpuPlacement) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  class Lookup final : public IExecutionProvider::IKernelLookup {
   public:
    const KernelCreateInfo* LookUpKernel(const Node&) const override { return &info_; }

   private:
    KernelCreateInfo info_{KernelDefBuilder{}.SetName("QMoE").SetDomain(kMSDomain).SinceVersion(1).Provider(kCudaExecutionProvider).Build(),
                           nullptr};
  } lookup;
  auto provider = DefaultCudaExecutionProvider();
  ASSERT_NE(provider, nullptr);
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  provider->SetLogger(&logger);
  for (const bool native : {false, true}) {
    OpTester test("QMoE", 1, kMSDomain);
    if (native) {
      IQ2MoEAttributes(test, 1);
    }
    test.AddInput<float>("input", {1, 256}, std::vector<float>(256));
    if (native) {
      test.AddOptionalInputEdge<float>();
    } else {
      test.AddInput<float>("router", {1, 2}, {0.5f, 0.5f});
    }
    test.AddInput<uint8_t>("up", {2, 256, 1, 66}, std::vector<uint8_t>(2 * 256 * 66));
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
    test.AddInput<uint8_t>("down", {2, 256, 1, 66}, std::vector<uint8_t>(2 * 256 * 66));
    if (native) {
      for (int index = 6; index <= 20; ++index) {
        test.AddOptionalInputEdge<float>();
      }
      test.AddInput<int64_t>("selected_experts", {1, 1}, {0});
      test.AddInput<float>("selected_expert_weights", {1, 1}, {1.0f});
    }
    test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    auto& graph = test.BuildModel().MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());
    const auto capabilities = provider->GetCapability(
        GraphViewer(graph), lookup, GraphOptimizerRegistry(nullptr, nullptr, &logger), nullptr);
    EXPECT_EQ(capabilities.size(), native ? 1u : 0u);
  }
}

TEST(QMoEIQ2XXS, SelectedExpertOracleAndDynamicPrefill) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunIQ2MoE<float>({1, 256}, 3, 256, 2, true);
  RunIQ2MoE<MLFloat16>({2, 3, 512}, 4, 256, 2, false);
  RunIQ2MoE<BFloat16>({5, 256}, 5, 512, 3, true);
  RunIQ2MoE<BFloat16>({5, 256}, 5, 512, 3, false, false);
  RunIQ2MoE<float>({2, 512}, 2, 512, 1, false);
}

TEST(QMoEIQ2XXS, EmptyTokenAndBatchDimensions) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  RunIQ2MoE<float>({0, 256}, 2, 256, 2, true);
  RunIQ2MoE<MLFloat16>({2, 0, 256}, 3, 256, 2, true);
  RunIQ2MoE<BFloat16>({0, 3, 256}, 3, 256, 2, true);
}

TEST(QMoEIQ2XXS, RoundsNativeStoresAndPreservesFloatRoutingWeights) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  const auto run = [&]<typename T>() {
    constexpr int64_t hidden = 256;
    std::vector<uint8_t> up(hidden * 66, 0);
    std::vector<uint8_t> down(hidden * 66, 0);
    // Exactly one nonzero row in each projection; grid 0 and d=1 reconstruct unit weights.
    up[1] = down[1] = 0x3c;
    std::vector<T> input(hidden, T(0.0f));
    input[0] = T(1.0078125f);
    input[8] = T(0.0003f);
    const T projected(static_cast<float>(input[0]) + static_cast<float>(input[8]));
    const T squared(static_cast<float>(projected) * static_cast<float>(projected));
    const float mixing_weight = std::is_same_v<T, BFloat16> ? 0.5058f : 0.5017f;
    const T expected(static_cast<float>(squared) * mixing_weight);
    if constexpr (!std::is_same_v<T, float>) {
      ASSERT_NE(static_cast<float>(expected),
                static_cast<float>(T(static_cast<float>(squared) * static_cast<float>(T(mixing_weight)))));
    }
    std::vector<T> output(hidden, T(0.0f));
    output[0] = expected;
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 1);
    IQ2MoEInputs(test, {1, hidden}, input, 1, hidden, 1, up, down, {0}, {mixing_weight});
    test.AddOutput<T>("output", {1, hidden}, output);
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<float>();
  run.template operator()<MLFloat16>();
  run.template operator()<BFloat16>();
}

TEST(QMoEIQ2XXS, AscendingExpertReductionAndNoUnselectedWeightAccess) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  constexpr int64_t hidden = 256;
  std::vector<uint8_t> up(4 * hidden * 66, 0);
  std::vector<uint8_t> down(up.size(), 0);
  for (int expert = 0; expert < 4; ++expert) {
    up[expert * hidden * 66 + 1] = down[expert * hidden * 66 + 1] = 0x3c;
  }
  // An unused expert is deliberately non-finite. Dense-all-expert execution would contaminate the sum.
  up[3 * hidden * 66] = 1;
  up[3 * hidden * 66 + 1] = 0x7c;
  std::vector<float> input(hidden, 0);
  input[0] = 1;
  for (const auto& ids : {std::vector<int64_t>{2, 0, 1}, std::vector<int64_t>{1, 2, 0}}) {
    std::vector<float> weights;
    for (const auto id : ids) {
      weights.push_back(id == 0 ? 0x1p24f : id == 1 ? -0x1p24f
                                                    : 1.0f);
    }
    std::vector<float> output(hidden, 0);
    output[0] = 1;
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 3);
    IQ2MoEInputs(test, {1, hidden}, input, 4, hidden, 3, up, down, ids, weights);
    test.AddOutput<float>("output", {1, hidden}, output);
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

TEST(QMoEIQ2XXS, RoundsReconstructedWeightsBeforeRelu2) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  const auto run = [&]<typename T>() {
    std::vector<uint8_t> up(256 * 66, 0);
    std::vector<uint8_t> down(256 * 66, 0);
    up[0] = 0x55;
    up[1] = 0x35;
    up[2] = 14;
    down[1] = 0x3c;
    const float first = 25.0f * static_cast<float>(T(43.0f * 0.333251953125f / 8.0f));
    const float second = 43.0f * static_cast<float>(T(25.0f * 0.333251953125f / 8.0f));
    const float direction = first < second ? -1.0f : 1.0f;
    const T projected(std::abs(first - second));
    const T expected(static_cast<float>(projected) * static_cast<float>(projected));
    if constexpr (!std::is_same_v<T, float>) {
      ASSERT_NE(static_cast<float>(expected), 0.0f);
    }
    std::vector<T> input(256, T(0.0f));
    input[1] = T(direction * 25.0f);
    input[2] = T(direction * -43.0f);
    std::vector<T> output(256, T(0.0f));
    output[0] = expected;
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 1);
    IQ2MoEInputs(test, {1, 256}, input, 1, 256, 1, up, down, {0}, {1.0f});
    test.AddOutput<T>("output", {1, 256}, output);
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<float>();
  run.template operator()<MLFloat16>();
  run.template operator()<BFloat16>();
}

TEST(QMoEIQ2XXS, RoundsDownOutputBeforeFloatAggregation) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  const auto run = [&]<typename T>(float extra_up) {
    std::vector<uint8_t> up(2 * 256 * 66, 0);
    std::vector<uint8_t> down(up.size(), 0);
    up[1] = up[256 * 66 + 1] = down[1] = down[256 * 66 + 1] = 0x3c;
    const auto bits = MLFloat16(extra_up).val;
    up[8 * 66] = static_cast<uint8_t>(bits);
    up[8 * 66 + 1] = static_cast<uint8_t>(bits >> 8);
    std::vector<T> input(256, T(0.0f));
    input[0] = T(1.0f);
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 2);
    IQ2MoEInputs(test, {1, 256}, input, 2, 256, 2, up, down, {1, 0}, {-1.0f, 1.0f});
    test.AddOutput<T>("output", {1, 256}, std::vector<T>(256, T(0.0f)));
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<MLFloat16>(0x1p-6f);
  run.template operator()<BFloat16>(0x1p-5f);
}

TEST(QMoEIQ2XXS, CastsAggregateOnlyOnce) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  const auto run = [&]<typename T>(float small_weight) {
    std::vector<uint8_t> bank(3 * 256 * 66, 0);
    for (int expert = 0; expert < 3; ++expert) {
      bank[expert * 256 * 66 + 1] = 0x3c;
    }
    std::vector<T> input(256, T(0.0f));
    input[0] = T(1.0f);
    std::vector<T> output(256, T(0.0f));
    output[0] = T(small_weight);
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 3);
    IQ2MoEInputs(test, {1, 256}, input, 3, 256, 3, bank, bank, {2, 0, 1}, {-1.0f, 1.0f, small_weight});
    test.AddOutput<T>("output", {1, 256}, output);
    test.SetOutputTolerance(0, 0);
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  };
  run.template operator()<MLFloat16>(0x1p-12f);
  run.template operator()<BFloat16>(0x1p-9f);
}

TEST(QMoEIQ2XXS, RejectsMalformedDeviceRoutesBeforeBankAccess) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  const auto run = [&](std::vector<int64_t> ids, std::vector<float> weights, const char* error) {
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 2);
    IQ2MoEInputs(test, {1, 256}, std::vector<float>(256), 2, 256, 2,
                 std::vector<uint8_t>(2 * 256 * 66), std::vector<uint8_t>(2 * 256 * 66), ids, weights);
    test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure, error, {}, nullptr, &providers);
  };
  run({-1, 0}, {0.5f, 0.5f}, "outside [0,E)");
  run({0, 2}, {0.5f, 0.5f}, "outside [0,E)");
  run({0, std::numeric_limits<int64_t>::max()}, {0.5f, 0.5f}, "outside [0,E)");
  run({1, 1}, {0.5f, 0.5f}, "unique per token");
  run({0, 1}, {std::numeric_limits<float>::infinity(), 0.5f}, "must be finite");
  run({0, 1}, {0.5f, std::numeric_limits<float>::quiet_NaN()}, "must be finite");
}

TEST(QMoEIQ2XXS, PrefillCrossesScratchTileBoundary) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  constexpr int64_t tokens = 513;
  constexpr int64_t hidden = 256;
  constexpr int64_t k = 32;
  std::vector<int64_t> ids(tokens * k);
  for (int64_t route = 0; route < tokens * k; ++route) {
    ids[static_cast<size_t>(route)] = k - 1 - route % k;
  }
  OpTester test("QMoE", 1, kMSDomain);
  IQ2MoEAttributes(test, k);
  // At FP32, 32 * (256 * 4 + 4) bytes per token permits 510 tokens per tile.
  IQ2MoEInputs(test, {tokens, hidden}, std::vector<float>(tokens * hidden), k, hidden, k,
               std::vector<uint8_t>(k * hidden * 66), std::vector<uint8_t>(k * hidden * 66),
               ids, std::vector<float>(tokens * k, 1.0f / 32.0f));
  test.AddOutput<float>("output", {tokens, hidden}, std::vector<float>(tokens * hidden));
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(QMoEIQ2XXS, RejectsRuntimeGeometryAndCudaGraphOption) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA SM80 or newer is required.";
  }
  for (const bool capture : {false, true}) {
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 1);
    test.AddShapeToTensorData(false);
    const int64_t hidden = capture ? 256 : 255;
    IQ2MoEInputs(test, {1, hidden}, std::vector<float>(static_cast<size_t>(hidden)),
                 1, 256, 1, std::vector<uint8_t>(capture ? 256 * 66 : 0),
                 std::vector<uint8_t>(static_cast<size_t>(hidden * 66)), {0}, {1.0f});
    test.AddOutput<float>("output", {1, hidden}, std::vector<float>(static_cast<size_t>(hidden)));
    OrtCUDAProviderOptionsV2 options{};
    options.enable_cuda_graph = capture ? 1 : 0;
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(CudaExecutionProviderWithOptions(&options));
    ASSERT_NE(providers.back(), nullptr);
    if (capture && providers.back()->GetOrtEp() != nullptr) {
      continue;  // Plugin capture is rejected at stream-capture entry, not this framework option check.
    }
    test.Run(OpTester::ExpectResult::kExpectFailure,
             capture ? "CUDA graph capture is enabled" : "divisible by 256", {}, nullptr, &providers);
  }
}
#endif

TEST(QMoEIQ2XXS, SchemaPreservesSymbolicTokenAndOutputType) {
  for (const bool batched : {false, true}) {
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 2);
    test.AddShapeToTensorData(true, batched ? 1 : 0);
    const std::vector<int64_t> shape = batched ? std::vector<int64_t>{1, 3, 256}
                                               : std::vector<int64_t>{3, 256};
    IQ2MoEInputs(test, shape, std::vector<BFloat16>(3 * 256), 4, 512, 2,
                 std::vector<uint8_t>(4 * 512 * 66), std::vector<uint8_t>(4 * 256 * 2 * 66),
                 {2, 0, 1, 3, 0, 2}, {0.7f, 0.3f, 0.1f, 0.9f, 0.2f, 0.8f});
    test.AddOutput<BFloat16>("output", shape, std::vector<BFloat16>(3 * 256));
    auto& graph = test.BuildModel().MainGraph();
    ASSERT_STATUS_OK(graph.Resolve());
    const auto* type = graph.GetOutputs()[0]->TypeAsProto();
    ASSERT_NE(type, nullptr);
    EXPECT_EQ(type->tensor_type().elem_type(), ONNX_NAMESPACE::TensorProto::BFLOAT16);
    EXPECT_FALSE(type->tensor_type().shape().dim(batched ? 1 : 0).has_dim_value());
  }
}

TEST(QMoEIQ2XXS, RejectsInvalidAttributesAndMissingRoutesAtResolve) {
  for (const bool missing_routes : {false, true}) {
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 1);
    if (!missing_routes) {
      test.AddAttribute("normalize_routing_weights", int64_t{1});
    }
    test.AddInput<float>("input", {1, 256}, std::vector<float>(256));
    test.AddOptionalInputEdge<float>();
    test.AddInput<uint8_t>("up", {1, 256, 1, 66}, std::vector<uint8_t>(256 * 66));
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
    test.AddInput<uint8_t>("down", {1, 256, 1, 66}, std::vector<uint8_t>(256 * 66));
    test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             missing_routes ? "requires selected_experts" : "no fused gate, normalization or sparse mixer");
  }
}

TEST(QMoEIQ2XXS, RejectsMalformedBanksAtResolve) {
  for (const auto& up_shape : std::vector<std::vector<int64_t>>{{2, 256, 1, 65}, {2, 255, 1, 66}, {2, 256, 2, 66}, {2, 256, 66}}) {
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 1);
    test.AddInput<float>("input", {1, 256}, std::vector<float>(256));
    test.AddOptionalInputEdge<float>();
    const auto size = std::accumulate(up_shape.begin(), up_shape.end(), int64_t{1},
                                      std::multiplies<int64_t>{});
    test.AddInput<uint8_t>("up", up_shape, std::vector<uint8_t>(static_cast<size_t>(size)));
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
    test.AddInput<uint8_t>("down", {2, 256, 1, 66}, std::vector<uint8_t>(2 * 256 * 66));
    for (int index = 6; index <= 20; ++index) {
      test.AddOptionalInputEdge<float>();
    }

    test.AddInput<int64_t>("ids", {1, 1}, {0});
    test.AddInput<float>("weights", {1, 1}, {1});
    test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    test.Run(OpTester::ExpectResult::kExpectFailure, "QMoE IQ2_XXS");
  }
}

TEST(QMoEIQ2XXS, RejectsRoutingShapeAndUnsupportedActivationTypeAtResolve) {
  for (const bool unsupported_type : {false, true}) {
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 2);
    if (unsupported_type) {
      IQ2MoEInputs(test, {1, 256}, std::vector<double>(256), 2, 256, 2,
                   std::vector<uint8_t>(2 * 256 * 66), std::vector<uint8_t>(2 * 256 * 66),
                   {0, 1}, {0.3f, 0.7f});
      test.AddOutput<double>("output", {1, 256}, std::vector<double>(256));
    } else {
      // k=2 but the selected tensors contain only one route.
      IQ2MoEInputs(test, {1, 256}, std::vector<float>(256), 2, 256, 1,
                   std::vector<uint8_t>(2 * 256 * 66), std::vector<uint8_t>(2 * 256 * 66),
                   {0}, {1.0f});
      test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    }
    test.Run(OpTester::ExpectResult::kExpectFailure, unsupported_type ? "invalid" : "flattened_tokens,k");
  }
}

TEST(QMoEIQ2XXS, RejectsInvalidNativeModeAttributesAtResolve) {
  for (const auto& [name, value] : std::vector<std::pair<const char*, int64_t>>{
           {"k", 0}, {"k", 33}, {"expert_weight_bits", 4}, {"fc1_expert_weight_bits", 4}, {"fc2_expert_weight_bits", 8}, {"fc3_expert_weight_bits", 4}, {"block_size", 128}, {"weights_prepacked", 1}, {"swiglu_fusion", 1}, {"use_sparse_mixer", 1}}) {
    SCOPED_TRACE(name);
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 1);
    test.AddAttribute(name, value);
    IQ2MoEInputs(test, {1, 256}, std::vector<float>(256), 2, 256, 1,
                 std::vector<uint8_t>(2 * 256 * 66), std::vector<uint8_t>(2 * 256 * 66),
                 {0}, {1.0f});
    test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    const auto status = test.BuildModel().MainGraph().Resolve();
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find("QMoE IQ2_XXS"), std::string::npos);
  }
}

TEST(QMoEIQ2XXS, RejectsUnexpectedNativeModeInputsAtResolve) {
  for (int forbidden = 1; forbidden <= 20; ++forbidden) {
    if (forbidden == 2 || forbidden == 5) {
      continue;
    }
    SCOPED_TRACE(forbidden);
    OpTester test("QMoE", 1, kMSDomain);
    IQ2MoEAttributes(test, 1);
    test.AddInput<float>("input", {1, 256}, std::vector<float>(256));
    for (int index = 1; index <= 20; ++index) {
      if (index == 2 || index == 5) {
        test.AddInput<uint8_t>(index == 2 ? "up" : "down", {2, 256, 1, 66},
                               std::vector<uint8_t>(2 * 256 * 66));
      } else if (index == forbidden) {
        if (index == 8 || (index >= 11 && index <= 13)) {
          test.AddInput<uint8_t>("forbidden", {1}, {0});
        } else {
          test.AddInput<float>("forbidden", {1}, {0.0f});
        }
      } else {
        test.AddOptionalInputEdge<float>();
      }
    }
    test.AddInput<int64_t>("selected_experts", {1, 1}, {0});
    test.AddInput<float>("selected_expert_weights", {1, 1}, {1.0f});
    test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    const auto status = test.BuildModel().MainGraph().Resolve();
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find("requires input " + std::to_string(forbidden) + " to be omitted"),
              std::string::npos);
  }
}

TEST(QMoEIQ2XXS, LegacyModesStillRequireRouterAndRejectSelectedRoutes) {
  for (const bool selected_routes : {false, true}) {
    OpTester test("QMoE", 1, kMSDomain);
    test.AddInput<float>("input", {1, 256}, std::vector<float>(256));
    if (selected_routes) {
      test.AddInput<float>("router", {1, 2}, {0.5f, 0.5f});
    } else {
      test.AddOptionalInputEdge<float>();
    }
    test.AddInput<uint8_t>("up", {2, 256, 128}, std::vector<uint8_t>(2 * 256 * 128));
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
    test.AddInput<uint8_t>("down", {2, 256, 128}, std::vector<uint8_t>(2 * 256 * 128));
    if (selected_routes) {
      for (int index = 6; index <= 20; ++index) {
        test.AddOptionalInputEdge<float>();
      }
      test.AddInput<int64_t>("selected_experts", {1, 1}, {0});
      test.AddInput<float>("selected_expert_weights", {1, 1}, {1.0f});
    }
    test.AddOutput<float>("output", {1, 256}, std::vector<float>(256));
    const auto status = test.BuildModel().MainGraph().Resolve();
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find(selected_routes ? "require quant_type='iq2_xxs'" : "router_probs is required"),
              std::string::npos);
  }
}

}  // namespace onnxruntime::test
