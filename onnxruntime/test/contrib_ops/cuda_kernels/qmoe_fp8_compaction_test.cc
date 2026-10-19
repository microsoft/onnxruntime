// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

namespace qmoe = contrib::cuda;

struct CudaBuffer {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaBuffer);

  explicit CudaBuffer(size_t size) : bytes(size) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }

  ~CudaBuffer() noexcept {
    const cudaError_t status = cudaFree(data);
    if (status != cudaSuccess) {
      std::fprintf(stderr, "cudaFree failed during QMoE test cleanup: %s\n", cudaGetErrorString(status));
    }
  }

  template <typename T>
  T* As() { return static_cast<T*>(data); }

  void Upload(const void* source) {
    CUDA_CALL_THROW(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice));
  }

  void Download(void* destination) {
    CUDA_CALL_THROW(cudaMemcpy(destination, data, bytes, cudaMemcpyDeviceToHost));
  }

  void* data = nullptr;
  size_t bytes;
};

TEST(CUDA_EP_Unittest, QMoEFp8CompactExpertsPreservesRoutesAndClearsUnusedSlots) {
  constexpr int num_experts = 512;
  constexpr int capacity = 6;
  CudaBuffer indices(capacity * sizeof(int));
  CudaBuffer compact_indices(capacity * sizeof(int));
  CudaBuffer expert_map(num_experts * sizeof(int));
  CudaBuffer slot_map(capacity * sizeof(int));

  for (const auto& routes : {std::array<int, capacity>{511, 7, 511, 7, 2, 7},
                             std::array<int, capacity>{0, 0, 0, 0, 0, 0}}) {
    indices.Upload(routes.data());
    qmoe::LaunchQMoECompactExperts(indices.As<int>(), compact_indices.As<int>(),
                                   expert_map.As<int>(), slot_map.As<int>(),
                                   num_experts, capacity, capacity, nullptr);
    std::array<int, capacity> original{};
    std::array<int, capacity> compact{};
    std::array<int, capacity> slots{};
    std::array<int, num_experts> reverse{};
    indices.Download(original.data());
    compact_indices.Download(compact.data());
    slot_map.Download(slots.data());
    expert_map.Download(reverse.data());
    EXPECT_EQ(original, routes);
    int used = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      if (reverse[expert] >= 0) {
        EXPECT_EQ(reverse[expert], used);
        EXPECT_EQ(slots[used++], expert);
      }
    }
    EXPECT_EQ(used, routes[0] == 511 ? 3 : 1);
    for (int slot = used; slot < capacity; ++slot) {
      EXPECT_EQ(slots[slot], -1);
    }
    for (int route = 0; route < capacity; ++route) {
      ASSERT_GE(compact[route], 0);
      ASSERT_LT(compact[route], used);
      EXPECT_EQ(slots[compact[route]], routes[route]);
    }
  }
}

template <typename T>
void TestSelectedDequantization(int layout) {
  constexpr int source_experts = 5;
  constexpr int capacity = 3;
  constexpr int n = 6;
  constexpr int k = 5;
  constexpr int block_size = 4;
  const bool split = layout == 1;
  const int output_n = split ? 2 * n : n;
  const std::array<int, capacity> experts{4, 1, -1};
  std::vector<uint8_t> weights(source_experts * n * k, 0x7f);
  std::vector<float> scales(source_experts * 4);
  std::vector<T> bias(source_experts * n);
  for (int expert = 0; expert < source_experts; ++expert) {
    for (int i = 0; i < n * k; ++i) {
      if (expert == 4 || expert == 1) {
        weights[expert * n * k + i] = 0x38;  // E4M3 1.0; unselected experts contain NaNs.
      }
    }
    for (int i = 0; i < 4; ++i) {
      scales[expert * 4 + i] = expert + 1.0f + i * 0.25f;
    }
    for (int row = 0; row < n; ++row) {
      bias[expert * n + row] = T(static_cast<float>(expert * n + row));
    }
  }
  std::vector<T> output(capacity * output_n * k, T(-123.0f));
  std::vector<T> output_bias(capacity * output_n, T(-123.0f));
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_scales(scales.size() * sizeof(float));
  CudaBuffer device_bias(bias.size() * sizeof(T));
  CudaBuffer device_output(output.size() * sizeof(T));
  CudaBuffer device_output_bias(output_bias.size() * sizeof(T));
  CudaBuffer device_experts(experts.size() * sizeof(int));
  device_weights.Upload(weights.data());
  device_scales.Upload(scales.data());
  device_bias.Upload(bias.data());
  device_output.Upload(output.data());
  device_output_bias.Upload(output_bias.data());
  device_experts.Upload(experts.data());

  for (int projection = 0; projection < (split ? 2 : 1); ++projection) {
    qmoe::LaunchQMoEDequantizeFp8Weights(
        device_weights.As<uint8_t>(), device_scales.As<float>(), device_output.As<T>() + projection * k,
        capacity, n, k, nullptr, block_size, split, layout == 2, device_experts.As<int>(),
        device_bias.As<T>(), device_output_bias.As<T>() + projection);
  }
  device_output.Download(output.data());
  device_output_bias.Download(output_bias.data());
  for (int slot = 0; slot < capacity; ++slot) {
    for (int output_row = 0; output_row < output_n; ++output_row) {
      const int row = split ? output_row / 2
                            : (layout == 2 ? output_row / 2 + (output_row % 2) * (n / 2) : output_row);
      const int expert = experts[slot];
      const float expected_bias = expert < 0 ? -123.0f : static_cast<float>(bias[expert * n + row]);
      EXPECT_EQ(static_cast<float>(output_bias[slot * output_n + output_row]), expected_bias);
      for (int col = 0; col < k; ++col) {
        const float expected = expert < 0 ? -123.0f
                                          : scales[expert * 4 + (row / block_size) * 2 + col / block_size];
        EXPECT_EQ(static_cast<float>(output[(slot * output_n + output_row) * k + col]), expected);
      }
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8DequantizesOnlySelectedExpertsAndGathersBias) {
  for (int layout : {0, 1, 2}) {
    SCOPED_TRACE(layout);
    TestSelectedDequantization<half>(layout);
    TestSelectedDequantization<__nv_bfloat16>(layout);
  }
}

[[maybe_unused]] float FiniteFp8Reference(int code) {
  const int exponent = (code >> 3) & 15;
  const int mantissa = code & 7;
  const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                        : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
  return code & 128 ? -magnitude : magnitude;
}

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
bool SupportsFp8Gemv() {
  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CALL_THROW(cudaGetDevice(&device));
  CUDA_CALL_THROW(cudaGetDeviceProperties(&properties, device));
  return properties.major >= 8;
}
#endif

template <typename T, typename S>
void TestFp8GemvCodeDecoding(int scale_type, int block_size, int k, int fusion, bool graph) {
  constexpr int n = 256;
  const int source_n = fusion == 1 ? n / 2 : n;
  const int scales_per_expert =
      ((source_n + block_size - 1) / block_size) * ((k + block_size - 1) / block_size);
  std::vector<uint8_t> weights(2 * source_n * k);
  std::vector<uint8_t> up_weights(weights.size());
  for (int expert = 0; expert < 2; ++expert) {
    for (int row = 0; row < source_n; ++row) {
      const int code = fusion == 1 ? row * 2 : row;
      std::fill_n(weights.begin() + (expert * source_n + row) * k, k, static_cast<uint8_t>(code));
      std::fill_n(up_weights.begin() + (expert * source_n + row) * k, k, static_cast<uint8_t>(code + 1));
    }
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  std::vector<S> scales(2 * scales_per_expert);
  for (int expert = 0; expert < 2; ++expert) {
    for (int block = 0; block < scales_per_expert; ++block) {
      scales[expert * scales_per_expert + block] =
          S((expert + 1) * 0.25f + ((block / k_blocks) % 4) * 0.125f + (block % k_blocks) * 0.25f);
    }
  }
  std::vector<T> input(2 * k);
  for (int col = 0; col < k; ++col) {
    input[col] = T(col == k - 1 ? 2.0f : (1 + col % 4) * 0.03125f);
    input[k + col] = T(col == k - 1 ? -4.0f : -(1 + col % 3) * 0.0625f);
  }
  const std::array<int, 2> experts{1, 0};
  const std::array<int, 2> source_rows{1, 0};
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_up_weights(up_weights.size());
  CudaBuffer device_scales(scales.size() * sizeof(S));
  CudaBuffer device_input(input.size() * sizeof(T));
  CudaBuffer device_experts(sizeof(experts));
  CudaBuffer device_source_rows(sizeof(source_rows));
  CudaBuffer device_output(2 * n * sizeof(T));
  device_weights.Upload(weights.data());
  device_up_weights.Upload(up_weights.data());
  device_scales.Upload(scales.data());
  device_input.Upload(input.data());
  device_experts.Upload(experts.data());
  device_source_rows.Upload(source_rows.data());
  qmoe::QMoEFp8ProjectionParams params;
  params.weights = device_weights.As<uint8_t>();
  params.scales = device_scales.data;
  params.scale_type = scale_type;
  if (fusion == 1) {
    params.up_weights = device_up_weights.As<uint8_t>();
    params.up_scales = device_scales.data;
    params.up_scale_type = scale_type;
  }
  params.fusion = fusion;
  params.experts = device_experts.As<int>();
  params.row_to_unpermuted = device_source_rows.As<int>();
  params.num_experts = 2;
  params.num_rows = 2;
  params.expanded_rows = 2;
  params.n = n;
  params.k = k;
  params.block_size = block_size;
  cudaStream_t stream;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    CUDA_CALL_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    qmoe::LaunchQMoEFp8Projection(params, device_input.As<T>(), device_output.As<T>(), stream);
    CUDA_CALL_THROW(cudaStreamEndCapture(stream, &captured));
    size_t node_count = 0;
    CUDA_CALL_THROW(cudaGraphGetNodes(captured, nullptr, &node_count));
    std::vector<cudaGraphNode_t> nodes(node_count);
    CUDA_CALL_THROW(cudaGraphGetNodes(captured, nodes.data(), &node_count));
    EXPECT_EQ(node_count, 1u);
    if (node_count == 1) {
      cudaKernelNodeParams node{};
      CUDA_CALL_THROW(cudaGraphKernelNodeGetParams(nodes[0], &node));
      EXPECT_EQ(qmoe::QMoEFp8GemvKernelUsesNativeDecode(node.func),
                ParseEnvironmentVariableWithDefault<bool>("ORT_QMOE_FP8_GEMV_NATIVE_DECODE", false));
    }
    CUDA_CALL_THROW(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    CUDA_CALL_THROW(cudaGraphLaunch(executable, stream));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    CUDA_CALL_THROW(cudaGraphExecDestroy(executable));
    CUDA_CALL_THROW(cudaGraphDestroy(captured));
  } else {
    qmoe::LaunchQMoEFp8Projection(params, device_input.As<T>(), device_output.As<T>(), stream);
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  }
  CUDA_CALL_THROW(cudaStreamDestroy(stream));
  std::array<T, 2 * n> actual{};
  device_output.Download(actual.data());
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < n; ++col) {
      const int code = fusion == 2 ? col / 2 + (col % 2) * (n / 2) : col;
      SCOPED_TRACE(row);
      SCOPED_TRACE(code);
      const float value = static_cast<float>(actual[row * n + col]);
      if ((code & 127) == 127) {
        EXPECT_TRUE(std::isnan(value));
        continue;
      }
      const int source_col = fusion == 1 ? col / 2 : code;
      double sum = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        const float scale = static_cast<float>(scales[experts[row] * scales_per_expert +
                                                      (source_col / block_size) * k_blocks + inner / block_size]);
        const T scaled_weight(FiniteFp8Reference(code) * scale);
        sum += static_cast<float>(scaled_weight) * static_cast<float>(input[source_rows[row] * k + inner]);
      }
      const float expected = static_cast<float>(T(static_cast<float>(sum)));
      const float tolerance = std::is_same_v<T, half> ? 0.001f : 0.008f;
      EXPECT_NEAR(value, expected, std::max(1e-7f, std::abs(expected) * tolerance));
    }
  }
}

void TestFp8GemvCodeConfigurations() {
  for (int block_size : {16, 128}) {
    for (int fusion : {0, 1, 2}) {
      for (bool graph : {false, true}) {
        SCOPED_TRACE(block_size);
        SCOPED_TRACE(fusion);
        SCOPED_TRACE(graph);
        const int k = block_size * 2 + 1;
        TestFp8GemvCodeDecoding<half, float>(0, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<half, half>(1, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<half, __nv_bfloat16>(2, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<__nv_bfloat16, float>(0, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<__nv_bfloat16, half>(1, block_size, k, fusion, graph);
        TestFp8GemvCodeDecoding<__nv_bfloat16, __nv_bfloat16>(2, block_size, k, fusion, graph);
      }
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8GemvAllCodesWithMappedRowsPartialKBlocksAndGraphReplay) {
  TestFp8GemvCodeConfigurations();
}

TEST(CUDA_EP_Unittest, QMoEFp8GemvNativeDecode) {
  if (!ParseEnvironmentVariableWithDefault<bool>("ORT_QMOE_FP8_GEMV_NATIVE_DECODE", false)) {
    GTEST_SKIP() << "Native decoding coverage runs in its flag-enabled CTest process.";
  }
  if (!qmoe::QMoEFp8GemvSupportsNativeDecode()) {
    GTEST_SKIP() << "Native E4M3 decoding requires FP8-enabled kernels compiled for SM90+.";
  }
  TestFp8GemvCodeConfigurations();
}

#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
template <typename T>
void TestNvfp4GemvScaleCodes(int k) {
  constexpr int n = 256;
  std::vector<uint8_t> weights(n * k / 2, 0);
  std::vector<uint8_t> scales(n * k / 16);
  const int blocks = k / 16;
  for (int row = 0; row < n; ++row) {
    for (int block = 0; block < blocks; ++block) {
      weights[row * k / 2 + block * 8] = 0x22;
      scales[row * blocks + block] =
          block == 0 ? static_cast<uint8_t>(row) : static_cast<uint8_t>(0x30 + (row + block) % 16);
    }
  }
  std::vector<T> input(k, T(0.0f));
  for (int block = 0; block < blocks; ++block) {
    input[block * 16] = T(1.0f + (block % 3) * 0.5f);
    input[block * 16 + 1] = T(0.5f + (block % 2) * 0.25f);
  }
  const float global = 0.37f;
  const int expert = 0;
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_scales(scales.size());
  CudaBuffer device_input(input.size() * sizeof(T));
  CudaBuffer device_global(sizeof(global));
  CudaBuffer device_expert(sizeof(expert));
  CudaBuffer device_output(n * sizeof(T));
  device_weights.Upload(weights.data());
  device_scales.Upload(scales.data());
  device_input.Upload(input.data());
  device_global.Upload(&global);
  device_expert.Upload(&expert);
  using namespace onnxruntime::llm::kernels::moe_gemv;
  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  launch_moe_gemv_fp4_symmetric<T>(
      device_input.As<T>(), device_weights.As<uint8_t>(), nullptr, device_scales.As<uint8_t>(),
      device_global.As<float>(), nullptr, device_output.As<T>(), nullptr, device_expert.As<int>(),
      1, 1, n, k, 16, 90, MoeGemvConfig::kDefault, false, true, stream, true);
  CUDA_CALL_THROW(cudaStreamEndCapture(stream, &graph));
  size_t node_count = 0;
  CUDA_CALL_THROW(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CUDA_CALL_THROW(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  EXPECT_EQ(node_count, 1u);
  if (node_count == 1) {
    cudaKernelNodeParams node{};
    CUDA_CALL_THROW(cudaGraphKernelNodeGetParams(nodes[0], &node));
    EXPECT_EQ(Fp4MoeGemvKernelUsesNativeScale(node.func),
              ParseEnvironmentVariableWithDefault<bool>("ORT_NVFP4_GEMV_NATIVE_SCALE", false));
  }
  CUDA_CALL_THROW(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUDA_CALL_THROW(cudaGraphLaunch(executable, stream));
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaGraphExecDestroy(executable));
  CUDA_CALL_THROW(cudaGraphDestroy(graph));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));
  std::array<T, n> actual{};
  device_output.Download(actual.data());
  for (int code = 0; code < 256; ++code) {
    SCOPED_TRACE(code);
    const float value = static_cast<float>(actual[code]);
    if ((code & 127) == 127) {
      EXPECT_TRUE(std::isnan(value));
    } else {
      double sum = 0.0;
      for (int block = 0; block < blocks; ++block) {
        const float scale = static_cast<float>(T(FiniteFp8Reference(scales[code * blocks + block]) * global));
        sum += scale * static_cast<float>(input[block * 16]);
        sum += scale * static_cast<float>(input[block * 16 + 1]);
      }
      const float expected = static_cast<float>(T(static_cast<float>(sum)));
      const float tolerance = std::is_same_v<T, half> ? 0.001f : 0.008f;
      EXPECT_NEAR(value, expected, std::max(1e-7f, std::abs(expected) * tolerance));
    }
  }
}

void TestNvfp4GemvScaleCodeConfigurations() {
  for (int k : {16, 80, 1040}) {
    SCOPED_TRACE(k);
    TestNvfp4GemvScaleCodes<half>(k);
    TestNvfp4GemvScaleCodes<__nv_bfloat16>(k);
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4GemvAllScaleCodesPreserveScaledValuesAndNaNs) {
  if (!SupportsFp8Gemv()) {
    GTEST_SKIP() << "NVFP4 GEMV requires SM80+.";
  }
  TestNvfp4GemvScaleCodeConfigurations();
}

TEST(CUDA_EP_Unittest, QMoENvfp4GemvNativeScale) {
  if (!ParseEnvironmentVariableWithDefault<bool>("ORT_NVFP4_GEMV_NATIVE_SCALE", false)) {
    GTEST_SKIP() << "Native-scale coverage runs in its flag-enabled CTest process.";
  }
  if (!SupportsFp8Gemv() || !onnxruntime::llm::kernels::moe_gemv::Fp4MoeGemvSupportsNativeScale()) {
    GTEST_SKIP() << "Native E4M3 scale conversion requires FP8-enabled kernels compiled for SM90+.";
  }
  TestNvfp4GemvScaleCodeConfigurations();
}
#endif

template <typename T>
void TestAllFp8Codes() {
  std::array<uint8_t, 256> codes{};
  for (int i = 0; i < 256; ++i) {
    codes[i] = static_cast<uint8_t>(i);
  }
  const float scale = 1.0f;
  CudaBuffer weights(codes.size());
  CudaBuffer scales(sizeof(scale));
  CudaBuffer output(codes.size() * sizeof(T));
  weights.Upload(codes.data());
  scales.Upload(&scale);
  qmoe::LaunchQMoEDequantizeFp8Weights(weights.As<uint8_t>(), scales.As<float>(), output.As<T>(),
                                       1, 1, 256, nullptr);
  std::array<T, 256> actual{};
  output.Download(actual.data());
  for (int code = 0; code < 256; ++code) {
    SCOPED_TRACE(code);
    const float value = static_cast<float>(actual[code]);
    if ((code & 127) == 127) {
      EXPECT_TRUE(std::isnan(value));
      continue;
    }
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                          : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
    const float expected = code & 128 ? -magnitude : magnitude;
    EXPECT_EQ(value, expected);
    EXPECT_EQ(std::signbit(value), std::signbit(expected));
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8AllCodesPreserveFiniteValuesSignedZerosAndNaNs) {
  TestAllFp8Codes<half>();
  TestAllFp8Codes<__nv_bfloat16>();
}

template <typename T, typename ScaleT, typename OutputT = T>
void TestFp8ProjectionScaleBlocks(int fusion, int block_size, int k, bool graph, bool gemm) {
  constexpr int n = 6;
  constexpr int rows = 3;
  constexpr int num_experts = 2;
  const bool split = fusion == 1;
  const int source_n = split ? n / 2 : n;
  const int scale_n = (source_n + block_size - 1) / block_size;
  const int scale_k = (k + block_size - 1) / block_size;
  const std::array<int, rows> experts{0, 0, 1};
  const std::array<int, rows> row_map{5, 0, 4};
  const std::array<int64_t, num_experts + 1> expert_offsets{0, 2, 3};
  std::vector<uint8_t> weights(num_experts * source_n * k);
  std::vector<uint8_t> up_weights(weights.size());
  std::vector<ScaleT> scales(num_experts * scale_n * scale_k);
  std::vector<ScaleT> up_scales(scales.size());
  std::vector<T> input(rows * k);
  std::vector<OutputT> output(rows * n);
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<uint8_t>(i % 3 == 0 ? 0xb8 : (i % 3 == 1 ? 0x38 : 0x40));
    up_weights[i] = 0x30;
  }
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = ScaleT(static_cast<float>(i % 5 + 1) / 32.0f);
    up_scales[i] = ScaleT(static_cast<float>(i % 5 + 2) / 32.0f);
  }
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = T(static_cast<float>(static_cast<int>(i % 7) - 3) / 16.0f);
  }
  CudaBuffer d_weights(weights.size()), d_up_weights(up_weights.size());
  CudaBuffer d_scales(scales.size() * sizeof(ScaleT)), d_up_scales(up_scales.size() * sizeof(ScaleT));
  CudaBuffer d_input(input.size() * sizeof(T)), d_output(output.size() * sizeof(OutputT));
  CudaBuffer d_experts(sizeof(experts)), d_rows(sizeof(row_map)), d_offsets(sizeof(expert_offsets));
  CudaBuffer d_tiles((num_experts + 1) * sizeof(int));
  d_weights.Upload(weights.data());
  d_up_weights.Upload(up_weights.data());
  d_scales.Upload(scales.data());
  d_up_scales.Upload(up_scales.data());
  d_input.Upload(input.data());
  d_experts.Upload(experts.data());
  d_rows.Upload(row_map.data());
  d_offsets.Upload(expert_offsets.data());
  qmoe::QMoEFp8ProjectionParams params;
  params.weights = d_weights.As<uint8_t>();
  params.scales = d_scales.data;
  params.scale_type = std::is_same_v<ScaleT, float> ? 0 : (std::is_same_v<ScaleT, half> ? 1 : 2);
  params.up_scale_type = params.scale_type;
  params.up_weights = split ? d_up_weights.As<uint8_t>() : nullptr;
  params.up_scales = split ? d_up_scales.data : nullptr;
  params.experts = d_experts.As<int>();
  params.row_to_unpermuted = d_rows.As<int>();
  params.expert_offsets = d_offsets.As<int64_t>();
  params.num_experts = num_experts;
  params.num_rows = rows;
  params.expanded_rows = rows;
  params.n = n;
  params.k = k;
  params.block_size = block_size;
  params.fusion = fusion;
  if (gemm) {
    qmoe::LaunchQMoEFp8ExpertTiles(params.expert_offsets, d_tiles.As<int>(), num_experts, nullptr);
    params.tile_offsets = d_tiles.As<int>();
    CUDA_CALL_THROW(cudaDeviceSynchronize());
  }
  cudaStream_t stream;
  CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  if (graph) {
    CUDA_CALL_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  }
  qmoe::LaunchQMoEFp8Projection(params, d_input.As<T>(), d_output.As<OutputT>(), stream);
  if (graph) {
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    CUDA_CALL_THROW(cudaStreamEndCapture(stream, &captured));
    CUDA_CALL_THROW(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    for (int replay = 0; replay < 3; ++replay) {
      CUDA_CALL_THROW(cudaGraphLaunch(executable, stream));
    }
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    CUDA_CALL_THROW(cudaGraphExecDestroy(executable));
    CUDA_CALL_THROW(cudaGraphDestroy(captured));
  }
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));
  d_output.Download(output.data());
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < n; ++col) {
      const bool up = split && col % 2;
      const int source_col = split ? col / 2 : (fusion == 2 ? col / 2 + (col % 2) * (n / 2) : col);
      float expected = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        const int weight_index = (experts[row] * source_n + source_col) * k + inner;
        const int scale_index = (experts[row] * scale_n + source_col / block_size) * scale_k +
                                inner / block_size;
        const uint8_t code = up ? up_weights[weight_index] : weights[weight_index];
        const float value = code == 0xb8 ? -1.0f : (code == 0x38 ? 1.0f : (code == 0x40 ? 2.0f : 0.5f));
        const T weight(value * static_cast<float>(up ? up_scales[scale_index] : scales[scale_index]));
        expected += static_cast<float>(input[(row_map[row] % rows) * k + inner]) * static_cast<float>(weight);
      }
      ASSERT_EQ(static_cast<float>(output[row * n + col]), static_cast<float>(OutputT(expected)))
          << "row=" << row << " col=" << col;
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8ProjectionPreservesScaleBlocksLayoutsAndGraphReplay) {
  int device = 0;
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  for (int fusion : {0, 1, 2}) {
    for (int block_size : {64, 128}) {
      for (int k : {127, 128, 257}) {
        for (bool graph : {false, true}) {
          for (bool gemm : {false, true}) {
            if (gemm && properties.major < 8) {
              continue;
            }
            SCOPED_TRACE(::testing::Message() << "fusion=" << fusion << " block=" << block_size
                                              << " k=" << k << " graph=" << graph << " gemm=" << gemm);
            TestFp8ProjectionScaleBlocks<half, float>(fusion, block_size, k, graph, gemm);
            TestFp8ProjectionScaleBlocks<half, half>(fusion, block_size, k, graph, gemm);
            TestFp8ProjectionScaleBlocks<half, __nv_bfloat16>(fusion, block_size, k, graph, gemm);
            TestFp8ProjectionScaleBlocks<__nv_bfloat16, float>(fusion, block_size, k, graph, gemm);
            TestFp8ProjectionScaleBlocks<__nv_bfloat16, half>(fusion, block_size, k, graph, gemm);
            TestFp8ProjectionScaleBlocks<__nv_bfloat16, __nv_bfloat16>(fusion, block_size, k, graph, gemm);
            TestFp8ProjectionScaleBlocks<half, float, float>(fusion, block_size, k, graph, gemm);
            TestFp8ProjectionScaleBlocks<__nv_bfloat16, float, float>(fusion, block_size, k, graph, gemm);
          }
        }
      }
    }
  }
}

template <typename T>
void TestNvfp4RowMajorDequantization(int k, bool compact, bool with_bias) {
  constexpr int source_experts = 5;
  constexpr int n = 3;
  const std::array<int, 3> selected{4, 1, -1};
  const int capacity = compact ? static_cast<int>(selected.size()) : source_experts;
  constexpr std::array<float, 8> magnitudes{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  constexpr std::array<uint8_t, 8> scale_codes{0x00, 0x01, 0x20, 0x38, 0x41, 0x7e, 0x80, 0xb8};
  constexpr std::array<float, 8> scale_values{0.0f, 1.0f / 512.0f, 0.125f, 1.0f, 2.25f, 448.0f, -0.0f, -1.0f};
  std::vector<uint8_t> weights(source_experts * n * k / 2);
  std::vector<uint8_t> scales(source_experts * n * k / 16);
  std::vector<float> globals(source_experts);
  std::vector<T> bias(source_experts * n);
  for (int expert = 0; expert < source_experts; ++expert) {
    globals[expert] = 0.37f * (expert + 1);
    for (int row = 0; row < n; ++row) {
      const int source_row = expert * n + row;
      bias[source_row] = T(static_cast<float>(source_row));
      for (int col = 0; col < k; col += 2) {
        const int low = (col + row + expert) % 16;
        const int high = (low + 1) % 16;
        weights[source_row * (k / 2) + col / 2] = static_cast<uint8_t>(low | (high << 4));
      }
      for (int group = 0; group < k / 16; ++group) {
        scales[source_row * (k / 16) + group] = scale_codes[(group + source_row) % scale_codes.size()];
      }
    }
  }
  std::vector<T> output(capacity * n * k, T(-123.0f));
  std::vector<T> output_bias(capacity * n, T(-123.0f));
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_scales(scales.size());
  CudaBuffer device_globals(globals.size() * sizeof(float));
  CudaBuffer device_bias(bias.size() * sizeof(T));
  CudaBuffer device_output(output.size() * sizeof(T));
  CudaBuffer device_output_bias(output_bias.size() * sizeof(T));
  CudaBuffer device_selected(selected.size() * sizeof(int));
  device_weights.Upload(weights.data());
  device_scales.Upload(scales.data());
  device_globals.Upload(globals.data());
  device_bias.Upload(bias.data());
  device_output.Upload(output.data());
  device_output_bias.Upload(output_bias.data());
  device_selected.Upload(selected.data());
  qmoe::LaunchQMoEDequantizeNvfp4Weights(
      device_weights.As<uint8_t>(), device_scales.As<uint8_t>(), device_globals.As<float>(),
      device_output.As<T>(), capacity, n, k, nullptr,
      compact ? device_selected.As<int>() : nullptr,
      with_bias ? device_bias.As<T>() : nullptr,
      with_bias ? device_output_bias.As<T>() : nullptr, true);
  device_output.Download(output.data());
  device_output_bias.Download(output_bias.data());
  for (int slot = 0; slot < capacity; ++slot) {
    const int expert = compact ? selected[slot] : slot;
    for (int row = 0; row < n; ++row) {
      const T expected_bias = with_bias && expert >= 0 ? bias[expert * n + row] : T(-123.0f);
      EXPECT_EQ(std::memcmp(&output_bias[slot * n + row], &expected_bias, sizeof(T)), 0);
      for (int col = 0; col < k; ++col) {
        T expected(-123.0f);
        if (expert >= 0) {
          const int code = (col + row + expert) % 16;
          const float value = code < 8 ? magnitudes[code] : -magnitudes[code - 8];
          expected = T(value * scale_values[(col / 16 + expert * n + row) % scale_values.size()] * globals[expert]);
        }
        ASSERT_EQ(std::memcmp(&output[(slot * n + row) * k + col], &expected, sizeof(T)), 0)
            << "slot=" << slot << " row=" << row << " col=" << col;
      }
    }
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4RowMajorVectorDequantizationPreservesValuesAndBias) {
  for (int k : {16, 48, 1280, 2560}) {
    for (bool compact : {false, true}) {
      for (bool with_bias : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "k=" << k << " compact=" << compact << " bias=" << with_bias);
        TestNvfp4RowMajorDequantization<half>(k, compact, with_bias);
        TestNvfp4RowMajorDequantization<__nv_bfloat16>(k, compact, with_bias);
      }
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
