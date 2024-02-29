// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

struct GatherScatterElementsArgs {
  enum class Operation {
    NONE,
    ADD,
    MUL,
    MAX,
    MIN
  };

  int64_t rank;
  int64_t axis;
  int64_t input_size;
  int64_t input_dim_along_axis;
  int64_t input_stride_along_axis;
  TArray<int64_t> masked_input_strides;
  TArray<fast_divmod> indices_fdms;
  TArray<int64_t> indices_strides;
  int64_t indices_size;
  // operation used to combine values associated the same
  // memory location in the output tensor.
  Operation operation;
};

template <typename T, typename TIndex>
void GatherElementsImpl(cudaStream_t stream, const T* input_data, const TIndex* indices_data, T* output_data,
                        const GatherScatterElementsArgs& args);

}  // namespace cuda
}  // namespace onnxruntime
