// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulFp32Q4 operator, it is basically
// matmul float32 with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

template <typename T>
class MatMulNBits final : public CudaKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<std::string>("packing", &packing_));
    if (packing_ == "default"){
      ORT_ENFORCE(nbits_ == 4,
                  "Only 4b quantization is supported for MatMulNBits op,"
                  " additional bits support is planned.");
    } else if (packing_ == "gptq"){
      ORT_ENFORCE(nbits_ > 1 && nbits_ < 8, "nbits_ should be in range of 2-8.");
    } else if (packing_ == "hqq"){
      ORT_ENFORCE(nbits_ == 4, "nbits_ should be in range of 4.");
    } else {
      ORT_THROW("Unsupported packing type: ", packing_);
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status ComputeInternalGPTQ(OpKernelContext* context) const;
  //Status ComputeInternalHQQ(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  std::string packing_;
  bool column_wise_quant_blk_{true};
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
