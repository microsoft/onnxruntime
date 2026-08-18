// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CUDA DeformConv kernel entry points (im2col, bias). Host pipeline and chunking: `deform_conv.cc`
// (see CPU `nn/deform_conv.cc` for the same high-level im2col → GEMM → bias flow).

#pragma once

#include <stdint.h>
#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {

// Adds bias to output: Y[n,m,oh,ow] += B[m]. Y is [N, M, out_h, out_w], B is [M].
template <typename T>
Status DeformConvAddBiasImpl(
    cudaStream_t stream,
    T* Y,
    const T* B,
    int64_t N,
    int64_t M,
    int64_t out_h,
    int64_t out_w,
    int64_t max_grid_y);

// Fills col_buffer with deformable im2col. Row-major [C*kH*kW, parallel_imgs*out_h*out_w]:
//   row = c * (kH*kW) + (i*kW + j),  col = n * (out_h*out_w) + (oh*out_w + ow),  same semantics as ONNX DeformConv im2col.
// Called once per batch chunk; caller GEMM + bias.
template <typename T>
Status DeformConvIm2ColImpl(
    cudaStream_t stream,
    const T* input,   // [parallel_imgs, C, H, W]
    const T* offset,  // [parallel_imgs, offset_group*2*kH*kW, out_h, out_w]
    const T* mask,    // [parallel_imgs, offset_group*kH*kW, out_h, out_w] or nullptr
    T* col_buffer,    // [C*kH*kW, parallel_imgs*out_h*out_w]
    int64_t parallel_imgs,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kH,
    int64_t kW,
    int64_t out_h,
    int64_t out_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t offset_group,
    bool use_mask);

}  // namespace cuda
}  // namespace onnxruntime
