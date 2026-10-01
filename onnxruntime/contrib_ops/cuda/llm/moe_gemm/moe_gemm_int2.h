#pragma once

#if defined(ENABLE_BF16)
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace onnxruntime::llm::kernels::cutlass_kernels {

template <typename ElementType>
struct Int2GroupedGemmParamsT {
  const ElementType* activations = nullptr;
  const uint8_t* packed_weights = nullptr;
  const ElementType* block_scales = nullptr;
  const int64_t* expert_row_ends = nullptr;
  ElementType* output = nullptr;
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

using Int2GroupedGemmParams = Int2GroupedGemmParamsT<half>;
#if defined(ENABLE_BF16)
using Bf16Int2GroupedGemmParams = Int2GroupedGemmParamsT<__nv_bfloat16>;
#endif

bool IsInt2GroupedGemmSupported(const Int2GroupedGemmParams& params);
void RunInt2GroupedGemm(const Int2GroupedGemmParams& params);
#if defined(ENABLE_BF16)
bool IsInt2GroupedGemmSupported(const Bf16Int2GroupedGemmParams& params);
void RunInt2GroupedGemm(const Bf16Int2GroupedGemmParams& params);
#endif

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
