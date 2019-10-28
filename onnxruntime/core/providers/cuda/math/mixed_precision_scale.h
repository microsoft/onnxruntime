// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename SrcT, typename DstT>
void Impl_MixedPrecisionScale(
    const SrcT* input_data,
    const float* scale_data,
    DstT* output_data,
    size_t count);

template <typename SrcT>
class MixedPrecisionScale final : public CudaKernel {
 public:
  MixedPrecisionScale(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t to;
    Status status = info.GetAttr("to", &to);
    ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
    to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  ONNX_NAMESPACE::TensorProto_DataType to_;
};

}  // namespace cuda
}  // namespace onnxruntime
