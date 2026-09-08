// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_FLOAT8_TYPES) || !defined(DISABLE_FLOAT4_TYPES)

#include <vector>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

// GatherFpQuantized: gathers rows from a block-scaled low-precision floating point (FP8 or FP4) constant
// table and dequantizes them on the fly. Unlike GatherBlockQuantized (integer block quantization with an
// optional zero point), the quantized type here is always an FP8 or FP4 floating point type and there is
// no zero point: FP8/FP4 quantization is symmetric, so dequantization is simply `float(data) * scale`.
// On any axis other than quantize_axis, `scales` may have dimension 1 to broadcast a single scale along
// that axis (e.g. one scale shared by every row), including the degenerate case where `scales` holds a
// single global per-tensor scale (as used by, e.g., a FP8-quantized embedding table with one scalar scale).
template <typename T1, typename Tind>
class GatherFpQuantized : public OpKernel {
 public:
  explicit GatherFpQuantized(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("gather_axis", &gather_axis_).IsOK()) {
      gather_axis_ = 0;
    }

    if (!info.GetAttr<int64_t>("quantize_axis", &quantize_axis_).IsOK()) {
      quantize_axis_ = 1;
    }

    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 0;
    }

    ORT_ENFORCE(block_size_ == 0 || (block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0),
                "'block_size' must be 0, or a power of 2 and not less than 16.");
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  struct Prepare {
    const Tensor* data_tensor;
    const Tensor* indices_tensor;
    const Tensor* scales_tensor;
    Tensor* output_tensor;
    int64_t gather_axis;
    int64_t quantize_axis;
    // Row-major strides of `data`, used to decompose a flat data index into per-axis indices.
    std::vector<int64_t> data_strides;
    // Row-major strides of `scales`. For a broadcast axis (scales dim == 1, data dim > 1) the
    // corresponding per-axis index contribution is always 0, regardless of this stride.
    std::vector<int64_t> scale_strides;
    // Per-axis flag (indexed like data/scales axes), true when that axis is broadcast in `scales`
    // (i.e. scales dim == 1 while data dim != 1). Unused/ignored at quantize_axis, which is always
    // handled via block-index division instead.
    std::vector<bool> scale_broadcast_axis;
  };

  Status PrepareForCompute(OpKernelContext* context, Prepare& args) const;

  template <typename T2>
  Status CopyDataAndDequantize(const T1* data_ptr,
                               const Tind* indices_ptr,
                               const T2* scales_ptr,
                               T2* output_ptr,
                               int64_t gather_M,
                               int64_t gather_N,
                               int64_t gather_axis_dim,
                               int64_t gather_block,
                               int64_t quantize_axis,
                               int64_t effective_block_size,
                               const std::vector<int64_t>& data_strides,
                               const std::vector<int64_t>& scale_strides,
                               const std::vector<bool>& scale_broadcast_axis,
                               concurrency::ThreadPool* tp) const;

 private:
  int64_t gather_axis_;
  int64_t quantize_axis_;
  int64_t block_size_;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_FLOAT8_TYPES) || !defined(DISABLE_FLOAT4_TYPES)
