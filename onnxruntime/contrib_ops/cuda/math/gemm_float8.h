// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

// see https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
// D = alpha*(A*B) + beta*(C)

namespace onnxruntime {
namespace cuda {

template <typename AType, typename BType, typename CType, typename DType, typename BiasType>
class GemmFloat8 final : public CudaKernel {
  using Base = CudaKernel;

 public:
  GemmFloat8(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t temp;

    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    fast_accumulation_mode_ = info.GetAttrOrDefault<int64_t>("fastAccumulationMode", 1) != 0;

    std::string stemp = info.GetAttrOrDefault<std::string>("computeType", "CUBLAS_COMPUTE_16F");
    if (stemp == "CUBLAS_COMPUTE_16F")
      compute_type_ = CUBLAS_COMPUTE_16F;
    else if (stemp == "CUBLAS_COMPUTE_32F")
      compute_type_ = CUBLAS_COMPUTE_32F;
    else if (stemp == "CUBLAS_COMPUTE_32F_FAST_16F")
      compute_type_ = CUBLAS_COMPUTE_32F_FAST_16F;
    else if (stemp == "CUBLAS_COMPUTE_32F_FAST_16BF")
      compute_type_ = CUBLAS_COMPUTE_32F_FAST_16BF;
    else if (stemp == "CUBLAS_COMPUTE_32F_FAST_TF32")
      compute_type_ = CUBLAS_COMPUTE_32F_FAST_TF32;
    else {
      ORT_THROW("Unexpected value for compute_type: ", stemp, ".");
    }

    sm_count_ = info.GetAttrOrDefault<int64_t>("sm_count", 0);
    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
    CastTo(alpha_, alpha_cast_);
    CastTo(beta_, beta_cast_);
    ORT_ENFORCE(!(trans_A_ && trans_B_), "Case both trans_A and trans_B are true is not implemented.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  void CastTo(float value, DType& dest);

  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasLtMatmulDescAttributes_t#cublasltmatmuldescattributes-t
  bool fast_accumulation_mode_;
  // see https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasComputeType_t#cublascomputetype-t
  cublasComputeType_t compute_type_;
  int64_t sm_count_;
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;
  DType alpha_cast_;
  DType beta_cast_;
};
}  // namespace cuda
}  // namespace onnxruntime
