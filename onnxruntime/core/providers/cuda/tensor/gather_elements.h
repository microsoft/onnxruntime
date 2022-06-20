// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

// Coalesce those contiguous axes that have same dim values for both input and indices (exclude the gather/scatter axis)
// so that we will less divmod to compute data offset during the kernels.
// For example:
// shape(input)=[2,2,2], shape(indices)=[2,2,3], axis=2 is same as shape(input)=[4,2], shape(indices)=[4,3], axis=1
// shape(input)=[2,1,2,2,3,2,2], shape(indices)=[2,1,2,2,2,2,2], axis=3) is same as
//     shape(input)=[4,2,3,4],shape(indices)=[4,2,2,4], axis=1
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
