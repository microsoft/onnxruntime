/*All contributions by Facebook :
Copyright(c) 2016 Facebook Inc.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "fft_ops.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void PostProcess(cudaStream_t stream, const std::vector<int64_t>& signal_dims, int64_t N, T* output_data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
