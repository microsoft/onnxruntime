// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename SrcT>
class Cast final : public CudaKernel {
 public:
  Cast(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

    int64_t saturate = info.GetAttrOrDefault("saturate", int64_t{1});
    if (saturate == 0 &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FN &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E4M3FNUZ &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2 &&
        to != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT8E5M2FNUZ) {
      ORT_THROW("Attribute saturate is only used for cast to float 8 types.");
    }
    saturate_ = saturate == 1;
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
  bool saturate_;
};

template <typename SrcT>
class CastLike final : public CudaKernel {
 public:
  CastLike(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
