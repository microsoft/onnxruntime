// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_int2.h"
#if !defined(BUILD_CUDA_EP_AS_PLUGIN)
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"
#endif
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::test {
namespace {

using llm::kernels::cutlass_kernels::Int2GroupedGemmParams;
using llm::kernels::cutlass_kernels::IsInt2GroupedGemmSupported;
using llm::kernels::cutlass_kernels::RunInt2GroupedGemm;
#if defined(ENABLE_BF16)
using llm::kernels::cutlass_kernels::Bf16Int2GroupedGemmParams;
#endif

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) { CUDA_CALL_THROW(cudaMalloc(&data_, bytes)); }
  ~DeviceBuffer() { cudaFree(data_); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeviceBuffer);

  template <typename Element>
  Element* Data() { return static_cast<Element*>(data_); }

 private:
  void* data_ = nullptr;
};

class DeviceEvent {
 public:
  DeviceEvent() { CUDA_CALL_THROW(cudaEventCreate(&event_)); }
  ~DeviceEvent() { cudaEventDestroy(event_); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeviceEvent);
  cudaEvent_t Get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class Int2GroupedGemmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceProperties(&properties_, device), cudaSuccess);
    if (properties_.major != 8) {
      GTEST_SKIP() << "INT2 grouped GEMM tests require SM8x";
    }
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
  }

  void TearDown() override {
    if (stream_) {
      EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
    }
  }

  void RunCase(const std::vector<int64_t>& expert_rows, int num_columns, int reduction_size,
               int weight_bits = 2, bool sample_reference = false, bool benchmark = false,
               int dense_mode = 0, int tile_rows = 32, size_t packed_offset_bytes = 0) {
#if defined(BUILD_CUDA_EP_AS_PLUGIN)
    ASSERT_EQ(weight_bits, 2);
    ASSERT_EQ(dense_mode, 0);
#endif
    std::vector<int64_t> row_ends;
    int64_t num_rows = 0;
    for (const auto rows : expert_rows) {
      num_rows += rows;
      row_ends.push_back(num_rows);
    }
    const int num_experts = static_cast<int>(expert_rows.size());
    const int pack_factor = 8 / weight_bits;
    const int quant_levels = 1 << weight_bits;
    const size_t expert_bytes = static_cast<size_t>(num_columns) * reduction_size / pack_factor;
    const int num_blocks = reduction_size / 64;
    std::vector<uint8_t> raw_weights(expert_bytes * num_experts, 0);
    std::vector<half> scales(static_cast<size_t>(num_experts) * num_blocks * num_columns);
    std::vector<half> activations(static_cast<size_t>(num_rows) * reduction_size);
    std::vector<half> output(static_cast<size_t>(num_rows) * num_columns);
    const auto quantized_value = [quant_levels](int expert, int column, int depth) {
      return (expert * 3 + column * 5 + depth + depth / 7) % quant_levels;
    };
    for (int expert = 0; expert < num_experts; ++expert) {
      for (int column = 0; column < num_columns; ++column) {
        for (int depth = 0; depth < reduction_size; ++depth) {
          const size_t index = static_cast<size_t>(expert) * expert_bytes +
                               static_cast<size_t>(column) * reduction_size / pack_factor + depth / pack_factor;
          raw_weights[index] |= static_cast<uint8_t>(quantized_value(expert, column, depth) << (weight_bits * (depth % pack_factor)));
        }
        for (int block = 0; block < num_blocks; ++block) {
          scales[(static_cast<size_t>(expert) * num_blocks + block) * num_columns + column] =
              __float2half(0.03125f * (1 + (expert * 3 + block * 5 + column) % 7));
        }
      }
    }
    for (int64_t row = 0; row < num_rows; ++row) {
      for (int depth = 0; depth < reduction_size; ++depth) {
        activations[row * reduction_size + depth] =
            __float2half(static_cast<float>((row * 3 + depth * 5 + depth / 7) % 17 - 8) / 16.0f);
      }
    }
    DeviceBuffer device_raw(raw_weights.size());
    DeviceBuffer device_packed(raw_weights.size() + packed_offset_bytes);
    auto* packed_weights = device_packed.Data<uint8_t>() + packed_offset_bytes;
    DeviceBuffer device_transposed(expert_bytes);
    DeviceBuffer device_permutation(64 * sizeof(int32_t));
    DeviceBuffer device_scales(scales.size() * sizeof(half));
    DeviceBuffer device_activations(activations.size() * sizeof(half));
    DeviceBuffer device_output(output.size() * sizeof(half));
    DeviceBuffer device_row_ends(row_ends.size() * sizeof(int64_t));
    CUDA_CALL_THROW(cudaMemcpy(device_raw.Data<uint8_t>(), raw_weights.data(), raw_weights.size(), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemcpy(device_scales.Data<half>(), scales.data(), scales.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemcpy(device_activations.Data<half>(), activations.data(), activations.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemcpy(device_row_ends.Data<int64_t>(), row_ends.data(), row_ends.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemsetAsync(device_output.Data<half>(), 0xff, output.size() * sizeof(half), stream_));
    DeviceEvent timing_start;
    DeviceEvent timing_end;
    if (benchmark) {
      CUDA_CALL_THROW(cudaEventRecord(timing_start.Get(), stream_));
    }
    for (int expert = 0; expert < num_experts; ++expert) {
      if (weight_bits == 2) {
        llm::kernels::fpA_intB_gemv::unpack_uint2_transposed_to_int8_direct_cuda(
            stream_, device_transposed.Data<int8_t>(), device_raw.Data<uint8_t>() + expert * expert_bytes,
            num_columns, reduction_size);
      } else if (weight_bits == 4) {
        llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_direct_cuda(
            stream_, device_transposed.Data<int8_t>(), device_raw.Data<uint8_t>() + expert * expert_bytes,
            num_columns, reduction_size);
      } else {
        llm::kernels::fpA_intB_gemv::transpose_uint8_matrix_and_convert_to_int8(
            stream_, device_transposed.Data<int8_t>(), device_raw.Data<uint8_t>() + expert * expert_bytes,
            num_columns, reduction_size);
      }
      llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
          stream_, 80, reinterpret_cast<int8_t*>(packed_weights) + expert * expert_bytes, device_transposed.Data<int8_t>(),
          device_permutation.Data<int32_t>(), {static_cast<size_t>(reduction_size), static_cast<size_t>(num_columns)},
          weight_bits == 2 ? llm::kernels::weight_only::QuantType::W2_A16
                           : (weight_bits == 4 ? llm::kernels::weight_only::QuantType::W4_A16
                                               : llm::kernels::weight_only::QuantType::W8_A16),
          false);
    }
    float prepack_ms = 0.0f;
    if (benchmark) {
      CUDA_CALL_THROW(cudaEventRecord(timing_end.Get(), stream_));
      CUDA_CALL_THROW(cudaEventSynchronize(timing_end.Get()));
      CUDA_CALL_THROW(cudaEventElapsedTime(&prepack_ms, timing_start.Get(), timing_end.Get()));
    }
    Int2GroupedGemmParams params;
    params.activations = device_activations.Data<half>();
    params.packed_weights = packed_weights;
    params.block_scales = device_scales.Data<half>();
    params.expert_row_ends = device_row_ends.Data<int64_t>();
    params.output = device_output.Data<half>();
    params.num_rows = num_rows;
    params.num_columns = num_columns;
    params.reduction_size = reduction_size;
    params.num_experts = num_experts;
    params.sm = 80;
    params.tile_rows = tile_rows;
    params.multiprocessor_count = properties_.multiProcessorCount;
    params.stream = stream_;
    ASSERT_TRUE(IsInt2GroupedGemmSupported(params));
    const size_t dense_bytes = static_cast<size_t>(num_experts) * num_columns * reduction_size * sizeof(half);
#if !defined(BUILD_CUDA_EP_AS_PLUGIN)
    using namespace llm::kernels::cutlass_kernels;
    const auto tile_config = tile_rows == 32
                                 ? llm::cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64
                                 : (tile_rows == 64
                                        ? llm::cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64
                                        : llm::cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64);
    std::unique_ptr<DeviceBuffer> dense_weights;
    std::unique_ptr<DeviceBuffer> dense_scales;
    std::unique_ptr<MoeGemmRunner<half, half, half>> dense_runner;
    const auto dequantize = [&]() {
      ORT_THROW_IF_ERROR((contrib::cuda::DequantizeNBits<half, uint8_t>(
          weight_bits, dense_weights->Data<half>(), device_raw.Data<uint8_t>(),
          dense_scales->Data<half>(), nullptr, nullptr, reduction_size,
          num_experts * num_columns, 64, stream_)));
    };
    if (dense_mode != 0) {
      dense_weights = std::make_unique<DeviceBuffer>(dense_bytes);
      dense_scales = std::make_unique<DeviceBuffer>(scales.size() * sizeof(half));
      std::vector<half> raw_scales(scales.size());
      for (int expert = 0; expert < num_experts; ++expert) {
        for (int column = 0; column < num_columns; ++column) {
          for (int block = 0; block < num_blocks; ++block) {
            raw_scales[(static_cast<size_t>(expert) * num_columns + column) * num_blocks + block] =
                scales[(static_cast<size_t>(expert) * num_blocks + block) * num_columns + column];
          }
        }
      }
      CUDA_CALL_THROW(cudaMemcpy(dense_scales->Data<half>(), raw_scales.data(),
                                 raw_scales.size() * sizeof(half), cudaMemcpyHostToDevice));
      dense_runner = std::make_unique<MoeGemmRunner<half, half, half>>();
      dequantize();
    }
    std::unique_ptr<MoeGemmRunner<half, cutlass::uint4b_t, half>> int4_runner;
    std::unique_ptr<MoeGemmRunner<half, uint8_t, half>> int8_runner;
    if (weight_bits == 4) {
      int4_runner = std::make_unique<MoeGemmRunner<half, cutlass::uint4b_t, half>>();
    } else if (weight_bits == 8) {
      int8_runner = std::make_unique<MoeGemmRunner<half, uint8_t, half>>();
    }
    const auto launch_existing = [&](auto& runner, auto* weight_type) {
      using WeightType = std::remove_pointer_t<decltype(weight_type)>;
      GroupedGemmInput<half, WeightType, half, half> inputs{
          params.activations, params.expert_row_ends, reinterpret_cast<const WeightType*>(params.packed_weights), params.block_scales, nullptr, nullptr, params.output, nullptr, nullptr, ActivationType::Identity, params.num_rows, params.num_columns, params.reduction_size, params.num_experts, params.block_size, true, false, params.stream, {}, {}};
      inputs.gemm_config = llm::cutlass_extensions::CutlassGemmConfig(
          llm::cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
          llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K, 1, 4);
      runner.moeGemm(inputs, {});
    };
    const auto launch = [&]() {
      if (dense_mode != 0) {
        if (dense_mode == 2) {
          dequantize();
        }
        GroupedGemmInput<half, half, half, half> inputs{
            params.activations, params.expert_row_ends, dense_weights->Data<half>(), nullptr, nullptr, nullptr, params.output, nullptr, nullptr, ActivationType::Identity, params.num_rows, params.num_columns, params.reduction_size, params.num_experts, 0, false, false, params.stream, {}, {}};
        inputs.gemm_config = llm::cutlass_extensions::CutlassGemmConfig(
            tile_config,
            llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K, 1, 4);
        dense_runner->moeGemm(inputs, {});
      } else if (weight_bits == 2) {
        RunInt2GroupedGemm(params);
      } else if (weight_bits == 4) {
        launch_existing(*int4_runner, static_cast<cutlass::uint4b_t*>(nullptr));
      } else {
        launch_existing(*int8_runner, static_cast<uint8_t*>(nullptr));
      }
    };
#else
    const auto launch = [&]() { RunInt2GroupedGemm(params); };
#endif
    launch();
    if (benchmark) {
      for (int warmup = 0; warmup < 5; ++warmup) {
        launch();
      }
      std::vector<float> timings;
      constexpr int iterations = 30;
      for (int trial = 0; trial < 3; ++trial) {
        CUDA_CALL_THROW(cudaEventRecord(timing_start.Get(), stream_));
        for (int iteration = 0; iteration < iterations; ++iteration) {
          launch();
        }
        CUDA_CALL_THROW(cudaEventRecord(timing_end.Get(), stream_));
        CUDA_CALL_THROW(cudaEventSynchronize(timing_end.Get()));
        float elapsed_ms = 0.0f;
        CUDA_CALL_THROW(cudaEventElapsedTime(&elapsed_ms, timing_start.Get(), timing_end.Get()));
        timings.push_back(elapsed_ms / iterations);
      }
      std::sort(timings.begin(), timings.end());
      std::cout << "GROUPED_GEMM bits=" << weight_bits << " P=" << num_rows
                << " dense_mode=" << dense_mode << " dense_bytes=" << (dense_mode ? dense_bytes : 0)
                << " tile_rows=" << tile_rows
                << " N=" << num_columns << " K=" << reduction_size << " E=" << num_experts
                << " kernel_ms_min=" << timings.front() << " kernel_ms_median=" << timings[1]
                << " kernel_ms_max=" << timings.back() << " prepack_ms=" << prepack_ms
                << " packed_bytes=" << raw_weights.size() << " scale_bytes=" << scales.size() * sizeof(half)
                << " prepack_scratch_bytes=" << expert_bytes + 64 * sizeof(int32_t) << '\n';
    }
    CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaMemcpy(output.data(), device_output.Data<half>(), output.size() * sizeof(half), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(std::all_of(output.begin(), output.end(), [](half value) { return std::isfinite(__half2float(value)); }));
    int64_t row_start = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      for (int64_t row = row_start; row < row_ends[expert]; ++row) {
        if (sample_reference && row != row_start && row != row_ends[expert] - 1 &&
            row != (row_start + row_ends[expert]) / 2) {
          continue;
        }
        for (int column = 0; column < num_columns; ++column) {
          if (sample_reference && column % 64 != 0 && column % 64 != 63) {
            continue;
          }
          float expected = 0.0f;
          for (int depth = 0; depth < reduction_size; ++depth) {
            const float scale = __half2float(scales[(static_cast<size_t>(expert) * num_blocks + depth / 64) * num_columns + column]);
            const float weight = static_cast<float>(quantized_value(expert, column, depth) - quant_levels / 2) * scale;
            expected += __half2float(activations[row * reduction_size + depth]) * weight;
          }
          expected = __half2float(__float2half(expected));
          ASSERT_NEAR(__half2float(output[row * num_columns + column]), expected, 0.001f + std::abs(expected) * 0.002f)
              << "expert=" << expert << " row=" << row << " column=" << column;
        }
      }
      row_start = row_ends[expert];
    }
  }

#if defined(ENABLE_BF16)
  void RunBf16Case(const std::vector<int64_t>& expert_rows, int num_columns, int reduction_size,
                   int tile_rows = 32, bool sample_reference = false, size_t packed_offset_bytes = 0) {
    std::vector<int64_t> row_ends;
    int64_t num_rows = 0;
    for (const auto rows : expert_rows) {
      num_rows += rows;
      row_ends.push_back(num_rows);
    }
    const int num_experts = static_cast<int>(expert_rows.size());
    const size_t expert_bytes = static_cast<size_t>(num_columns) * reduction_size / 4;
    const int num_blocks = reduction_size / 64;
    std::vector<uint8_t> raw_weights(expert_bytes * num_experts, 0);
    std::vector<__nv_bfloat16> scales(static_cast<size_t>(num_experts) * num_blocks * num_columns);
    std::vector<__nv_bfloat16> activations(static_cast<size_t>(num_rows) * reduction_size);
    std::vector<__nv_bfloat16> output(static_cast<size_t>(num_rows) * num_columns);
    const auto quantized_value = [](int expert, int column, int depth) {
      return (expert * 3 + column * 5 + depth + depth / 7) % 4;
    };
    for (int expert = 0; expert < num_experts; ++expert) {
      for (int column = 0; column < num_columns; ++column) {
        for (int depth = 0; depth < reduction_size; ++depth) {
          const size_t index = static_cast<size_t>(expert) * expert_bytes +
                               static_cast<size_t>(column) * reduction_size / 4 + depth / 4;
          raw_weights[index] |= static_cast<uint8_t>(quantized_value(expert, column, depth) << (2 * (depth % 4)));
        }
        for (int block = 0; block < num_blocks; ++block) {
          scales[(static_cast<size_t>(expert) * num_blocks + block) * num_columns + column] =
              static_cast<__nv_bfloat16>(0.03125f * (1 + (expert * 3 + block * 5 + column) % 7));
        }
      }
    }
    for (int64_t row = 0; row < num_rows; ++row) {
      for (int depth = 0; depth < reduction_size; ++depth) {
        activations[row * reduction_size + depth] =
            static_cast<__nv_bfloat16>(static_cast<float>((row * 3 + depth * 5 + depth / 7) % 17 - 8) / 16.0f);
      }
    }

    DeviceBuffer device_raw(raw_weights.size());
    DeviceBuffer device_packed(raw_weights.size() + packed_offset_bytes);
    auto* packed_weights = device_packed.Data<uint8_t>() + packed_offset_bytes;
    DeviceBuffer device_transposed(expert_bytes);
    DeviceBuffer device_permutation(64 * sizeof(int32_t));
    DeviceBuffer device_scales(scales.size() * sizeof(__nv_bfloat16));
    DeviceBuffer device_activations(activations.size() * sizeof(__nv_bfloat16));
    DeviceBuffer device_output(output.size() * sizeof(__nv_bfloat16));
    DeviceBuffer device_row_ends(row_ends.size() * sizeof(int64_t));
    CUDA_CALL_THROW(cudaMemcpy(device_raw.Data<uint8_t>(), raw_weights.data(), raw_weights.size(), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemcpy(device_scales.Data<__nv_bfloat16>(), scales.data(),
                               scales.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemcpy(device_activations.Data<__nv_bfloat16>(), activations.data(),
                               activations.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemcpy(device_row_ends.Data<int64_t>(), row_ends.data(),
                               row_ends.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL_THROW(cudaMemsetAsync(device_output.Data<__nv_bfloat16>(), 0xff,
                                    output.size() * sizeof(__nv_bfloat16), stream_));
    for (int expert = 0; expert < num_experts; ++expert) {
      llm::kernels::fpA_intB_gemv::unpack_uint2_transposed_to_int8_direct_cuda(
          stream_, device_transposed.Data<int8_t>(), device_raw.Data<uint8_t>() + expert * expert_bytes,
          num_columns, reduction_size);
      llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
          stream_, 80, reinterpret_cast<int8_t*>(packed_weights) + expert * expert_bytes, device_transposed.Data<int8_t>(),
          device_permutation.Data<int32_t>(), {static_cast<size_t>(reduction_size), static_cast<size_t>(num_columns)},
          llm::kernels::weight_only::QuantType::W2_A16, false);
    }

    Bf16Int2GroupedGemmParams params;
    params.activations = device_activations.Data<__nv_bfloat16>();
    params.packed_weights = packed_weights;
    params.block_scales = device_scales.Data<__nv_bfloat16>();
    params.expert_row_ends = device_row_ends.Data<int64_t>();
    params.output = device_output.Data<__nv_bfloat16>();
    params.num_rows = num_rows;
    params.num_columns = num_columns;
    params.reduction_size = reduction_size;
    params.num_experts = num_experts;
    params.tile_rows = tile_rows;
    params.sm = 80;
    params.multiprocessor_count = properties_.multiProcessorCount;
    params.stream = stream_;
    ASSERT_TRUE(IsInt2GroupedGemmSupported(params));
    RunInt2GroupedGemm(params);
    CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    CUDA_CALL_THROW(cudaMemcpy(output.data(), device_output.Data<__nv_bfloat16>(),
                               output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    ASSERT_TRUE(std::all_of(output.begin(), output.end(),
                            [](__nv_bfloat16 value) { return std::isfinite(static_cast<float>(value)); }));

    int64_t row_start = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      for (int64_t row = row_start; row < row_ends[expert]; ++row) {
        if (sample_reference && row != row_start && row != row_ends[expert] - 1 &&
            row != (row_start + row_ends[expert]) / 2) {
          continue;
        }
        for (int column = 0; column < num_columns; ++column) {
          if (sample_reference && column % 64 != 0 && column % 64 != 63) {
            continue;
          }
          float expected = 0.0f;
          for (int depth = 0; depth < reduction_size; ++depth) {
            const float scale = static_cast<float>(
                scales[(static_cast<size_t>(expert) * num_blocks + depth / 64) * num_columns + column]);
            const float weight = static_cast<float>(quantized_value(expert, column, depth) - 2) * scale;
            expected += static_cast<float>(activations[row * reduction_size + depth]) * weight;
          }
          expected = static_cast<float>(static_cast<__nv_bfloat16>(expected));
          ASSERT_NEAR(static_cast<float>(output[row * num_columns + column]), expected,
                      0.02f + std::abs(expected) * 0.02f)
              << "expert=" << expert << " row=" << row << " column=" << column;
        }
      }
      row_start = row_ends[expert];
    }
  }
#endif

  cudaDeviceProp properties_{};
  cudaStream_t stream_ = nullptr;
};

TEST_F(Int2GroupedGemmTest, SingleExpert) {
  RunCase({128}, 128, 256);
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN)
TEST_F(Int2GroupedGemmTest, DenseBaselineParity) {
  RunCase({0, 1, 31, 96}, 192, 192, 2, false, false, 1);
  RunCase({0, 1, 31, 96}, 192, 192, 2, false, false, 2);
}
#endif

TEST_F(Int2GroupedGemmTest, CandidateTacticsParity) {
  for (int tile_rows : {64}) {
    SCOPED_TRACE(tile_rows);
    RunCase({0, 1, 31, 129, 0, 351}, 192, 192, 2, false, false, 0, tile_rows);
    RunCase({1, 0, 2}, 64, 64, 2, false, false, 0, tile_rows);
  }
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN)
TEST_F(Int2GroupedGemmTest, DISABLED_GptOssTacticBenchmark) {
  for (int tokens : {128, 512, 2048}) {
    for (bool skewed : {false, true}) {
      std::vector<int64_t> rows(32, tokens * 4 / 32);
      if (skewed) {
        rows.assign(32, 0);
        rows[1] = 1;
        rows[7] = 31;
        rows[15] = tokens - 32;
        rows[19] = tokens;
        rows[23] = tokens;
        rows[31] = tokens;
      }
      for (int tile_rows : {32, 64}) {
        std::cout << "DISTRIBUTION skewed=" << skewed << '\n';
        RunCase(rows, 5760, 2880, 2, true, true, 0, tile_rows);
        RunCase(rows, 5760, 2880, 2, true, true, 1, tile_rows);
        RunCase(rows, 5760, 2880, 2, true, true, 2, tile_rows);
      }
    }
  }
}

TEST_F(Int2GroupedGemmTest, DISABLED_GptOssDenseBenchmark) {
  for (int tokens : {128, 512, 2048}) {
    const std::vector<int64_t> rows(32, tokens * 4 / 32);
    RunCase(rows, 5760, 2880, 2, true, true, 1);
    RunCase(rows, 5760, 2880, 2, true, true, 2);
  }
}
#endif

TEST_F(Int2GroupedGemmTest, AlignedPackedWeightOffsets) {
  for (int tile_rows : {32, 64}) {
    for (size_t offset : {16, 32, 48}) {
      SCOPED_TRACE(testing::Message() << "tile_rows=" << tile_rows << " offset=" << offset);
      RunCase({0, 1, 33, 0}, 64, 128, 2, false, false, 0, tile_rows, offset);
    }
  }
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN)
TEST_F(Int2GroupedGemmTest, Int4Int8AlignedPackedWeightOffsets) {
  for (int weight_bits : {4, 8}) {
    for (size_t offset : {16, 32, 48}) {
      SCOPED_TRACE(testing::Message() << "weight_bits=" << weight_bits << " offset=" << offset);
      RunCase({0, 1, 33, 0}, 64, 128, weight_bits, false, false, 0, 32, offset);
    }
  }
}
#endif

TEST_F(Int2GroupedGemmTest, MinimumAlignedShape) {
  RunCase({1, 0, 2}, 64, 64);
}

TEST_F(Int2GroupedGemmTest, EmptyExpertsAndPartialTiles) {
  RunCase({0, 1, 0, 39, 88, 0}, 64, 128);
}

TEST_F(Int2GroupedGemmTest, Prefill512) {
  RunCase({3, 253, 0, 256}, 192, 192);
}

TEST_F(Int2GroupedGemmTest, Prefill2048) {
  RunCase({0, 1025, 1, 1022}, 128, 256);
}

#if defined(ENABLE_BF16)
TEST_F(Int2GroupedGemmTest, Bf16AlignedPackedWeightOffsets) {
  for (int tile_rows : {32, 64}) {
    for (size_t offset : {16, 32, 48}) {
      SCOPED_TRACE(testing::Message() << "tile_rows=" << tile_rows << " offset=" << offset);
      RunBf16Case({0, 1, 33, 0}, 64, 128, tile_rows, false, offset);
    }
  }
}

TEST_F(Int2GroupedGemmTest, Bf16CandidateTacticsParity) {
  for (int tile_rows : {32, 64}) {
    SCOPED_TRACE(tile_rows);
    RunBf16Case({1, 0, 2}, 64, 64, tile_rows);
  }
}

TEST_F(Int2GroupedGemmTest, Bf16EmptyExpertsAndPartialTiles) {
  RunBf16Case({0, 1, 0, 39, 88, 0}, 64, 128);
}

TEST_F(Int2GroupedGemmTest, Bf16GptOssFc1SampledParity) {
  RunBf16Case(std::vector<int64_t>(32, 16), 5760, 2880, 32, true);
}
#endif

#if !defined(BUILD_CUDA_EP_AS_PLUGIN)
TEST_F(Int2GroupedGemmTest, Int4Block64EmptyExpertsAndPartialTiles) {
  RunCase({0, 1, 0, 39, 88, 0}, 64, 128, 4);
}

TEST_F(Int2GroupedGemmTest, Int4Block64Prefill512) {
  RunCase({3, 253, 0, 256}, 192, 192, 4);
}

TEST_F(Int2GroupedGemmTest, Int8Block64EmptyExpertsAndPartialTiles) {
  RunCase({0, 1, 0, 39, 88, 0}, 64, 128, 8);
}

TEST_F(Int2GroupedGemmTest, Int8Block64Prefill512) {
  RunCase({3, 253, 0, 256}, 192, 192, 8);
}
#endif

TEST_F(Int2GroupedGemmTest, GptOssFc1BalancedSampledParity) {
  for (int tokens : {128, 512, 2048}) {
    SCOPED_TRACE(tokens);
    RunCase(std::vector<int64_t>(32, tokens * 4 / 32), 5760, 2880, 2, true);
  }
}

TEST_F(Int2GroupedGemmTest, GptOssFc1SkewedSampledParity) {
  for (int tokens : {128, 512, 2048}) {
    SCOPED_TRACE(tokens);
    std::vector<int64_t> rows(32, 0);
    rows[1] = 1;
    rows[7] = 31;
    rows[15] = tokens - 32;
    rows[19] = tokens;
    rows[23] = tokens;
    rows[31] = tokens;
    RunCase(rows, 5760, 2880, 2, true);
  }
}

#if !defined(BUILD_CUDA_EP_AS_PLUGIN)
TEST_F(Int2GroupedGemmTest, GptOssFc2Int4SampledParity) {
  for (int tokens : {128, 512, 2048}) {
    SCOPED_TRACE(tokens);
    RunCase(std::vector<int64_t>(32, tokens * 4 / 32), 2880, 2880, 4, true);
  }
}

TEST_F(Int2GroupedGemmTest, DISABLED_GptOssKernelBenchmark) {
  for (int tokens : {128, 512, 2048}) {
    const std::vector<int64_t> rows(32, tokens * 4 / 32);
    RunCase(rows, 5760, 2880, 2, true, true);
    RunCase(rows, 5760, 2880, 4, true, true);
    RunCase(rows, 2880, 2880, 4, true, true);
  }
}
#endif

template <typename ElementType>
void TestMisalignedBuffers() {
  alignas(16) ElementType activations[16]{};
  alignas(16) uint8_t packed_weights[32]{};
  alignas(16) ElementType scales[16]{};
  alignas(16) ElementType output[16]{};
  int64_t row_end = 1;
  llm::kernels::cutlass_kernels::Int2GroupedGemmParamsT<ElementType> aligned;
  aligned.activations = activations;
  aligned.packed_weights = packed_weights;
  aligned.block_scales = scales;
  aligned.output = output;
  aligned.expert_row_ends = &row_end;
  aligned.num_rows = 1;
  aligned.num_columns = 64;
  aligned.reduction_size = 64;
  aligned.num_experts = 1;
  aligned.sm = 80;
  aligned.multiprocessor_count = 108;
  for (int tile_rows : {32, 64}) {
    SCOPED_TRACE(tile_rows);
    aligned.tile_rows = tile_rows;
    for (int buffer_index = 0; buffer_index < 4; ++buffer_index) {
      SCOPED_TRACE(buffer_index);
      auto params = aligned;
      switch (buffer_index) {
        case 0:
          params.activations += 8 / sizeof(ElementType);
          break;
        case 1:
          params.packed_weights += 8;
          break;
        case 2:
          params.block_scales += 8 / sizeof(ElementType);
          break;
        case 3:
          params.output += 8 / sizeof(ElementType);
          break;
      }
      ASSERT_TRUE(IsInt2GroupedGemmSupported(params));
      try {
        RunInt2GroupedGemm(params);
        FAIL() << "Expected buffer alignment rejection before any CUDA access";
      } catch (const OnnxRuntimeException& error) {
        EXPECT_NE(std::string(error.what()).find("requires 16-byte aligned"), std::string::npos)
            << error.what();
      }
    }
  }
}

TEST(Int2GroupedGemmValidationTest, RejectsMisalignedFp16Buffers) {
  TestMisalignedBuffers<half>();
}

#if defined(ENABLE_BF16)
TEST(Int2GroupedGemmValidationTest, RejectsMisalignedBf16Buffers) {
  TestMisalignedBuffers<__nv_bfloat16>();
}
#endif

TEST(Int2GroupedGemmValidationTest, RejectsUnsupportedConfiguration) {
  Int2GroupedGemmParams params;
  params.num_rows = 128;
  params.num_columns = 128;
  params.reduction_size = 256;
  params.num_experts = 4;
  params.sm = 80;
  params.multiprocessor_count = 108;
  ASSERT_TRUE(IsInt2GroupedGemmSupported(params));
  EXPECT_THROW(RunInt2GroupedGemm(params), OnnxRuntimeException);
  for (int sm : {70, 75, 86, 89, 90, 100, 120}) {
    params.sm = sm;
    EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  }
  params.sm = 80;
  params.tile_rows = 16;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  params.tile_rows = 128;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  params.tile_rows = 32;
  for (int block_size : {0, 16, 32, 128, 256}) {
    params.block_size = block_size;
    EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  }
  params.block_size = 64;
  params.num_columns = 65;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  params.num_columns = 128;
  params.reduction_size = 65;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  params.reduction_size = 256;
  params.num_rows = static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  params.num_rows = 0;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  params.num_rows = 128;
  params.num_experts = 0;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
  params.num_experts = 4;
  params.multiprocessor_count = 0;
  EXPECT_FALSE(IsInt2GroupedGemmSupported(params));
}

}  // namespace
}  // namespace onnxruntime::test
