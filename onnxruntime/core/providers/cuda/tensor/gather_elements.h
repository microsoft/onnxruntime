// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace cuda {

struct GatherScatterElementsArgs;

// Coalesce those contiguous axes that have same dim values for both input and indices (exclude the gather/scatter axis)
// so that we will have less divmod to compute during the kernels.
// For example:
// shape(input)=[2,2,2], shape(indices)=[2,2,3], axis=2 is same as shape(input)=[4,2], shape(indices)=[4,3], axis=1
// shape(input)=[2,1,2,2,3,2,2], shape(indices)=[2,1,2,2,2,2,2], axis=3) is same as
//     shape(input)=[4,2,3,4],shape(indices)=[4,2,2,4], axis=1
// If indices is strided, dim i (outer) and dim j is contiguous when strides[i] = shape[j] * strides[j].
// For example:
// shape(indices)=[2,3,4,5], strides(indices)=[0,20,5,1], then dim-2 and dim-3 is contiguous (5==5*1),
// dim-1 and dim-2 is contiguous (20==4*5), but dim-0 and dim-1 is not contiguous (0!=3*20).
void CoalesceDimensions(TensorShapeVector& input_shape, TensorShapeVector& indices_shape,
                        TensorShapeVector* p_indices_strides, int64_t axis, GatherScatterElementsArgs& args);
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
