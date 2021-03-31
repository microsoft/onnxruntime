// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "cusparse_support.h"

namespace onnxruntime {

class MatMulComputeHelper;

namespace cuda {
namespace cusparse_helper {
/// <summary>
/// Captures Prepack() information along the data and its shape
/// </summary>
struct SparseInfo {
  OpKernel::PrepackParam param_;
  TensorShape shape_;
  std::vector<IAllocatorUniquePtr<uint8_t>> prepack_buffers_;  // Typed buffer
#ifdef USE_CUSPARSE
  onnxruntime::optional<cusparseLtHandle_t> handle_lt_;
#endif
  onnxruntime::optional<cusparseSpMatDescr_t> sparse_desc_;

  explicit SparseInfo(const TensorShape& shape)
      : param_(), shape_(shape), prepack_buffers_() {}

  SparseInfo(const SparseInfo&) = delete;
  SparseInfo& operator=(const SparseInfo&) = delete;

  ~SparseInfo() {
    if (sparse_desc_.has_value()) {
      cusparseDestroySpMat(*sparse_desc_);
    }
#ifdef USE_CUSPARSE
    if (handle_lt_.has_value()) {
      cusparseLtDestroy(&*handle_lt_);
    }
#endif
  }
};
}  // namespace cusparse_helper

template <typename T>
class MatMul final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMul(const OpKernelInfo& info)
      : CudaKernel(info),
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        trans_A_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        // trans_A_(true),
        trans_B_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0} {
    // trans_B_(true) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status PrePack(const Tensor& tensor, const PrepackParam& param, bool& is_packed) override;

  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  // Argument 1 is a sparse weight coming from constant initializer
  // if set
  std::unique_ptr<cusparse_helper::SparseInfo> sparse_info_;
};
}  // namespace cuda
}  // namespace onnxruntime
