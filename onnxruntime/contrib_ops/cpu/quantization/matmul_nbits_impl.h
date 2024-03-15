// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/common.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <typename inputT, typename zeroT>
void DequantizeBlockwise(
    inputT* output,              // dequantized output
    const uint8_t* quant_data,   // quantized input
    const inputT* scales_data,   // quantization scales
    const zeroT* zero_points,    // quantization zero points
    const int32_t* reorder_idx,  // quantization zero points
    int32_t block_size,          // quantization block size
    bool,                        // columnwise quantization or row-wise
    int32_t K,                   // number of rows in quantized input
    int32_t N,                   // number of columns in quantized input
    onnxruntime::concurrency::ThreadPool* thread_pool);

}  // namespace contrib
}  // namespace onnxruntime
