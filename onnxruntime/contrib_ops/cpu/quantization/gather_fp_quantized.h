// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_FLOAT8_TYPES) || !defined(DISABLE_FLOAT4_TYPES)

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

// GatherFpQuantized: gathers rows from a block-scaled low-precision floating point (FP8 or FP4) constant
// table and dequantizes them on the fly. Unlike GatherBlockQuantized (integer block quantization with an
// optional zero point), the quantized type here is always an FP8 or FP4 floating point type and there is
// no zero point: FP8/FP4 quantization is symmetric, so dequantization is simply `float(data) * scale`.
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
                               int64_t quantize_axis_dim,
                               int64_t quantize_N,
                               int64_t effective_block_size,
                               concurrency::ThreadPool* tp) const;

 private:
  int64_t gather_axis_;
  int64_t quantize_axis_;
  int64_t block_size_;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_FLOAT8_TYPES) || !defined(DISABLE_FLOAT4_TYPES)
