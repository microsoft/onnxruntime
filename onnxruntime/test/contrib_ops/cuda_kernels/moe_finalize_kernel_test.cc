// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_util_kernels.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::test {
namespace {

namespace ck = onnxruntime::llm::kernels::cutlass_kernels;

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) { CUDA_CALL_THROW(cudaMalloc(&data_, bytes)); }
  ~DeviceBuffer() { cudaFree(data_); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeviceBuffer);

  void* Data() { return data_; }

 private:
  void* data_ = nullptr;
};

// Runs the generic finalizer with the decode geometry (at most two rows, staged routing) and checks every
// output element. Values and scales are dyadic, so all sums are exact in half and bfloat16.
template <typename T>
void RunDecodeFinalizeWithWideColumns(int num_rows, int experts_per_token, int64_t cols) {
  SCOPED_TRACE(::testing::Message() << "rows=" << num_rows << " top_k=" << experts_per_token << " cols=" << cols);
  constexpr int kNumExperts = 4;
  const int expanded_rows = num_rows * experts_per_token;
  const auto source_value = [](int expanded_row, int64_t col) {
    return static_cast<float>((col * 11 + col / 4099 * 7 + expanded_row * 37) % 17 - 8) / 4.0f;
  };
  const auto route_scale = [](int route) { return route % 2 == 0 ? 1.0f : 0.5f; };
  // The finalizer gathers expanded row unpermuted_row_to_permuted_row[row + route * num_rows]; reverse the
  // order so a wrong gather cannot pass.
  const auto permuted_row = [&](int row, int route) { return expanded_rows - 1 - (row + route * num_rows); };

  std::vector<T> input(static_cast<size_t>(expanded_rows) * static_cast<size_t>(cols));
  for (int expanded_row = 0; expanded_row < expanded_rows; ++expanded_row) {
    T* destination = input.data() + static_cast<size_t>(expanded_row) * static_cast<size_t>(cols);
    for (int64_t col = 0; col < cols; ++col) {
      destination[col] = T(source_value(expanded_row, col));
    }
  }
  std::vector<int> unpermuted_row_to_permuted_row(expanded_rows), selected_experts(expanded_rows);
  std::vector<float> scales(expanded_rows);
  for (int row = 0; row < num_rows; ++row) {
    for (int route = 0; route < experts_per_token; ++route) {
      unpermuted_row_to_permuted_row[row + route * num_rows] = permuted_row(row, route);
      selected_experts[row * experts_per_token + route] = (row + route) % kNumExperts;
      scales[row * experts_per_token + route] = route_scale(route);
    }
  }

  const size_t output_elements = static_cast<size_t>(num_rows) * static_cast<size_t>(cols);
  DeviceBuffer d_input(input.size() * sizeof(T)), d_output(output_elements * sizeof(T));
  DeviceBuffer d_row_map(expanded_rows * sizeof(int)), d_experts(expanded_rows * sizeof(int));
  DeviceBuffer d_scales(expanded_rows * sizeof(float));
  CUDA_CALL_THROW(cudaMemcpy(d_input.Data(), input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(d_row_map.Data(), unpermuted_row_to_permuted_row.data(),
                             expanded_rows * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(d_experts.Data(), selected_experts.data(), expanded_rows * sizeof(int),
                             cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(d_scales.Data(), scales.data(), expanded_rows * sizeof(float),
                             cudaMemcpyHostToDevice));
  // 0xff bytes are NaN in half and bfloat16, so any column the launch skips fails the comparison.
  CUDA_CALL_THROW(cudaMemset(d_output.Data(), 0xff, output_elements * sizeof(T)));

  ck::finalizeMoeRoutingKernelLauncher<T, T, T>(
      static_cast<const T*>(d_input.Data()), static_cast<T*>(d_output.Data()), nullptr,
      static_cast<const float*>(d_scales.Data()), static_cast<const int*>(d_row_map.Data()), nullptr,
      static_cast<const int*>(d_experts.Data()), nullptr, num_rows, cols, experts_per_token, kNumExperts,
      ck::MOEParallelismConfig{}, false, nullptr);
  CUDA_CALL_THROW(cudaGetLastError());
  CUDA_CALL_THROW(cudaDeviceSynchronize());

  std::vector<T> actual(output_elements);
  CUDA_CALL_THROW(cudaMemcpy(actual.data(), d_output.Data(), output_elements * sizeof(T), cudaMemcpyDeviceToHost));
  int64_t mismatches = 0;
  for (int row = 0; row < num_rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      float expected = 0.0f;
      for (int route = 0; route < experts_per_token; ++route) {
        expected += route_scale(route) * source_value(permuted_row(row, route), col);
      }
      const float got = static_cast<float>(actual[static_cast<size_t>(row) * static_cast<size_t>(cols) +
                                                  static_cast<size_t>(col)]);
      if (!(got == expected) && mismatches++ < 5) {
        ADD_FAILURE() << "row=" << row << " col=" << col << " expected=" << expected << " got=" << got;
      }
    }
  }
  EXPECT_EQ(mismatches, 0);
}

}  // namespace

// The decode geometry gives each 32-thread block 32 column vectors (eight half/bfloat16 values each) and
// maps column tiles to grid-Y, which CUDA limits to 65535 blocks. Wider outputs must be covered by the
// kernel's column grid-stride loop instead of launching an invalid grid.
TEST(CUDA_EP_Unittest, MoeFinalizeDecodeColumnTilesRespectGridYLimit) {
  constexpr int64_t kMaxTileColumns = 65535LL * 32 * 8;
  RunDecodeFinalizeWithWideColumns<half>(2, 2, kMaxTileColumns);
  RunDecodeFinalizeWithWideColumns<half>(2, 2, kMaxTileColumns + 8);
  RunDecodeFinalizeWithWideColumns<__nv_bfloat16>(2, 2, 16777216);
  // A single row needs more than four experts per token to leave the dedicated one-row kernel.
  RunDecodeFinalizeWithWideColumns<__nv_bfloat16>(1, 5, 16777216);
}

}  // namespace onnxruntime::test
