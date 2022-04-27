// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

void CoalesceDimensions(TensorShapeVector& input_shape, TensorShapeVector& indices_shape, int64_t axis,
                        int64_t& new_axis, int64_t& new_rank, int64_t& input_stride_along_axis,
                        TArray<int64_t>& masked_input_strides, TArray<fast_divmod>& indices_fdms);
ONNX_NAMESPACE::TensorProto_DataType GetElementType(size_t element_size);

class GatherElements final : public CudaKernel {
 public:
  GatherElements(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(), "Missing/Invalid 'axis' attribute value");
  }
  ~GatherElements() = default;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  struct ComputeImpl;

  int64_t axis_;
};

}  // namespace cuda
}  // namespace onnxruntime
