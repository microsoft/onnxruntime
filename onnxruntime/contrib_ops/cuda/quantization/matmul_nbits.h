// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulNBits operator, it is basically
// matmul float with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//
#pragma once
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class MatMulNBits final : public onnxruntime::cuda::CudaKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for MatMulNBits op,"
                " additional bits support is planned.");
    int64_t column_wise_quant_blk = 1;
    info.GetAttrOrDefault<int64_t>("column_wise_blocking", &column_wise_quant_blk, int64_t(1));
    column_wise_quant_blk_ = column_wise_quant_blk != 0;
    info.GetAttrOrDefault<int64_t>("prepacked", &prepack_, int64_t(0));
  }

#if !defined(USE_MIGRAPHX) && !defined(USE_ROCM)
  Status PrepackedGemm([[maybe_unused]] cudaStream_t stream,
                       [[maybe_unused]] int M,
                       [[maybe_unused]] const Tensor* a,
                       [[maybe_unused]] const Tensor* b,
                       [[maybe_unused]] const Tensor* scales,
                       [[maybe_unused]] const Tensor* zero_points,
                       [[maybe_unused]] Tensor* Y) const {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Prepacked gemm is not supported for MatMulNBits op.");
  }
#endif  // !defined(USE_MIGRAPHX) && !defined(USE_ROCM)

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool column_wise_quant_blk_{true};
  int64_t prepack_{0};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
