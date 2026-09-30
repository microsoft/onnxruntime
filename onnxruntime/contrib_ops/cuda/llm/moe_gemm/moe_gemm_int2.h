#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace onnxruntime::llm::kernels::cutlass_kernels {

struct Int2GroupedGemmParams {
  const half* activations = nullptr;
  const uint8_t* packed_weights = nullptr;
  const half* block_scales = nullptr;
  const int64_t* expert_row_ends = nullptr;
  half* output = nullptr;
  int64_t num_rows = 0;
  int num_columns = 0;
  int reduction_size = 0;
  int num_experts = 0;
  int block_size = 64;
  int tile_rows = 32;
  int sm = 0;
  int multiprocessor_count = 0;
  cudaStream_t stream = nullptr;
};

bool IsInt2GroupedGemmSupported(const Int2GroupedGemmParams& params);
void RunInt2GroupedGemm(const Int2GroupedGemmParams& params);

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
