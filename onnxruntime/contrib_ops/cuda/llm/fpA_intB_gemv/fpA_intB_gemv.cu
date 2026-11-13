/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#if USE_FPA_INTB_GEMM
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/details.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/int8_dequant.cuh"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

void kernel_launcher(int kernel_arch, Params& params, cudaStream_t s) {
#define EXEC(KType, A, B, Layout, ConverterInterleave)                                                       \
  if (params.type == KType) {                                                                                \
    select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 64>>( \
        params, s);                                                                                          \
    return;                                                                                                  \
  }

// This is not used since there is no alpha for MatMulNBits currently.
#define EXEC_W4A8(KType, A, B, Layout, ConverterInterleave)                                                   \
  if (params.type == KType && params.apply_alpha_in_advance) {                                                \
    select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 128>>( \
        params, s);                                                                                           \
    return;                                                                                                   \
  }

  ORT_ENFORCE(kernel_arch >= 75, "Unsupported CUDA kernel architecture: ", kernel_arch);
#if USE_COMPACT_FPA_INTB_GEMM
  ORT_ENFORCE(kernel_arch < 90 || kernel_arch >= 100,
              "The compact fpA_intB GEMV does not support the SM90 weight layout");
  ORT_ENFORCE(params.type == KernelType::FP16Int8Groupwise || params.type == KernelType::FP16Int4Groupwise ||
                  params.type == KernelType::FP16Int2Groupwise ||
                  params.type == KernelType::BF16Int8Groupwise || params.type == KernelType::BF16Int4Groupwise ||
                  params.type == KernelType::BF16Int2Groupwise,
              "The compact fpA_intB GEMV supports only FP16/BF16 groupwise kernels");
  EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
  EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
  EXEC(KernelType::FP16Int2Groupwise, FP16DetailsA, Int2DetailsW, ColumnMajorInterleaved, true);
  EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
  EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
  EXEC(KernelType::BF16Int2Groupwise, BF16DetailsA, Int2DetailsW, ColumnMajorInterleaved, true);
#else
  if (kernel_arch < 80) {
    EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int2Groupwise, FP16DetailsA, Int2DetailsW, ColumnMajorInterleaved, true);
#ifndef EXCLUDE_SM_90
  } else if (kernel_arch >= 90 && kernel_arch < 100) {
    // Dispatchers for W4A8 groupwise
    // EXEC_W4A8(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
    // EXEC_W4A8(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);

    EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);

    EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
#endif
  } else {
    // if (arch >= 89)
    // {
    //     EXEC_W4A8(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    //     EXEC_W4A8(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    // }
    EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int2Groupwise, FP16DetailsA, Int2DetailsW, ColumnMajorInterleaved, true);

    EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::BF16Int2Groupwise, BF16DetailsA, Int2DetailsW, ColumnMajorInterleaved, true);
  }
#endif
#undef EXEC_W4A8
#undef EXEC
  ORT_THROW("No fpA_intB GEMV kernel for this (kernel_arch, KernelType) combination");
}

bool is_supported(int device_arch, int kernel_arch, KernelType kernel_type) {
  if (device_arch < 75 || kernel_arch < 75) {
    return false;
  }

  const bool is_fp16 = kernel_type == KernelType::FP16Int8Groupwise ||
                       kernel_type == KernelType::FP16Int4Groupwise ||
                       kernel_type == KernelType::FP16Int2Groupwise;
  const bool is_bf16 = kernel_type == KernelType::BF16Int8Groupwise ||
                       kernel_type == KernelType::BF16Int4Groupwise ||
                       kernel_type == KernelType::BF16Int2Groupwise;
#if USE_COMPACT_FPA_INTB_GEMM
  const bool is_sm90_layout = kernel_arch >= 90 && kernel_arch < 100;
  if (!is_fp16 && !is_bf16) {
    return false;
  }
  if ((device_arch < 80 || kernel_arch < 80) && is_bf16) {
    return false;
  }
  return !is_sm90_layout;
#else
  if (!is_fp16 && !is_bf16) {
    return false;
  }
  if ((device_arch < 80 || kernel_arch < 80) && is_bf16) {
    return false;
  }
  if (kernel_arch >= 90 && kernel_arch < 100) {
    // No 2-bit instantiation exists for the native Hopper (ColumnMajorInterleavedForHopper) layout.
    if (kernel_type == KernelType::FP16Int2Groupwise || kernel_type == KernelType::BF16Int2Groupwise) {
      return false;
    }
    if (device_arch < 90 || device_arch >= 100) {
      return false;
    }
#ifdef EXCLUDE_SM_90
    return false;
#endif
  }
  return true;
#endif
}

bool dequantize_int8_interleaved(KernelType type, const void* weight, const void* scales, void* out, int n, int k,
                                 int group_size, cudaStream_t stream) {
  const bool is_bf16 = type == KernelType::BF16Int8Groupwise;
  if (!is_bf16 && type != KernelType::FP16Int8Groupwise) {
    return false;
  }
  const bool group_ok = group_size >= dequant::kChunkK && (group_size & (group_size - 1)) == 0 && k % group_size == 0;
  const auto aligned = [](const void* p, uintptr_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; };
  if (n <= 0 || n % 16 != 0 || k <= 0 || k % 64 != 0 || !group_ok || !aligned(weight, 16) || !aligned(out, 16) ||
      !aligned(scales, 4) || static_cast<int64_t>(n / 2) * (k / dequant::kChunkK) >= (int64_t{1} << 31) - dequant::kThreads) {
    return false;
  }

  int chunk_group_shift = 0;
  while ((dequant::kChunkK << chunk_group_shift) < group_size) {
    ++chunk_group_shift;
  }
  const int total = (n / 2) * (k / dequant::kChunkK);
  const int blocks = (total + dequant::kThreads - 1) / dequant::kThreads;
  if (is_bf16) {
    dequant::DequantizeInt8InterleavedKernel<__nv_bfloat16><<<blocks, dequant::kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(weight), static_cast<const __nv_bfloat16*>(scales),
        static_cast<__nv_bfloat16*>(out), n, k, chunk_group_shift);
  } else {
    dequant::DequantizeInt8InterleavedKernel<half><<<blocks, dequant::kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(weight), static_cast<const half*>(scales), static_cast<half*>(out), n, k,
        chunk_group_shift);
  }
  return true;
}

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
#endif
