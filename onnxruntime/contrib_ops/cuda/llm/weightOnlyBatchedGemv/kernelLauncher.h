/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include "contrib_ops/cuda/llm/common/cudaUtils.h"
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/common.h"
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/details.h"

namespace onnxruntime::llm {
namespace kernels {
namespace weight_only {
template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s);

inline void kernel_launcher(int arch, Params& params, cudaStream_t s) {
#define EXEC(KType, A, B, Layout, ConverterInterleave)                                                       \
  if (params.type == KType) {                                                                                \
    select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 64>>( \
        params, s);                                                                                          \
    return;                                                                                                  \
  }
#define EXEC_W4A8(KType, A, B, Layout, ConverterInterleave)                                                   \
  if (params.type == KType && params.apply_alpha_in_advance) {                                                \
    select_gs<kernel_type_traits<KType>::isGroupwise, KernelDetails<A, B, Layout, ConverterInterleave, 128>>( \
        params, s);                                                                                           \
    return;                                                                                                   \
  }
  if (arch >= 75 && arch < 80) {
    EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int4PerChannel, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
  } else if ((arch >= 80 && arch < 90) || arch >= 100) {
    if (arch == 89) {
      EXEC_W4A8(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
      EXEC_W4A8(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    }
    EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::FP16Int4PerChannel, FP16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
    EXEC(KernelType::BF16Int4PerChannel, BF16DetailsA, Int4DetailsW, ColumnMajorInterleaved, true);
  } else if (arch >= 90) {
    // Dispatchers for W4A8 groupwise
    EXEC_W4A8(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC_W4A8(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);

    EXEC(KernelType::FP16Int8Groupwise, FP16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::BF16Int8Groupwise, BF16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::FP16Int4Groupwise, FP16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::BF16Int4Groupwise, BF16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::FP16Int8PerChannel, FP16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::BF16Int8PerChannel, BF16DetailsA, Int8DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::FP16Int4PerChannel, FP16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
    EXEC(KernelType::BF16Int4PerChannel, BF16DetailsA, Int4DetailsW, ColumnMajorInterleavedForHopper, true);
  }
#undef EXEC
}

inline bool is_supported(int arch, KernelType kernel_type) {
#define SUPPORT(Type)      \
  if (kernel_type == Type) \
    return true;
  if (arch >= 75 && arch < 80) {
    SUPPORT(KernelType::FP16Int4Groupwise);
    SUPPORT(KernelType::FP16Int8PerChannel);
    SUPPORT(KernelType::FP16Int4PerChannel);
  } else if (arch >= 80 && arch < 90) {
    SUPPORT(KernelType::FP16Int8Groupwise);
    SUPPORT(KernelType::BF16Int8Groupwise);
    SUPPORT(KernelType::FP16Int4Groupwise);
    SUPPORT(KernelType::BF16Int4Groupwise);
    SUPPORT(KernelType::FP16Int8PerChannel);
    SUPPORT(KernelType::BF16Int8PerChannel);
    SUPPORT(KernelType::FP16Int4PerChannel);
    SUPPORT(KernelType::BF16Int4PerChannel);
  } else if (arch >= 90 && arch != 120) {
    SUPPORT(KernelType::FP16Int8Groupwise);
    SUPPORT(KernelType::BF16Int8Groupwise);
    SUPPORT(KernelType::FP16Int4Groupwise);
    SUPPORT(KernelType::BF16Int4Groupwise);
    SUPPORT(KernelType::FP16Int8PerChannel);
    SUPPORT(KernelType::BF16Int8PerChannel);
    SUPPORT(KernelType::FP16Int4PerChannel);
    SUPPORT(KernelType::BF16Int4PerChannel);
  }
  return false;
#undef SUPPORT
}
}  // namespace weight_only
}  // namespace kernels
}  // namespace onnxruntime::llm
