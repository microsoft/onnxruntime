// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk.h"
#include "topk_impl.h"
//#include "core/common/common.h"
//#include "core/framework/op_kernel.h"
//#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
  TopK,
  kOnnxDomain,
  11,
  kCudaExecutionProvider,
  KernelDefBuilder()
    .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
  TopK);

TopK::TopK(const OpKernelInfo& info) : CudaKernel(info) {
  info.GetAttrOrDefault<int64_t>("axis",    &axis_,   -1);
  info.GetAttrOrDefault<int64_t>("largest", &largest_, 1);
  info.GetAttrOrDefault<int64_t>("sorted",  &sorted_,  1);
}

#define TOPKIMPL(T) TopKImpl<T> (tensor_X->Data<T>(),                               \
                                 static_cast<T*>(tensor_V->MutableDataRaw()),       \
                                 static_cast<int64_t*>(tensor_I->MutableDataRaw()), \
                                 elem_nums.data(),                                  \
                                 elem_nums.size(),                                  \
                                 axis, K, largest_, sorted_)

Status TopK::ComputeInternal(OpKernelContext* ctx) const {
    auto tensor_X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(nullptr != tensor_X);
    auto axis = axis_ < 0 ? tensor_X->Shape().NumDimensions() + axis_ : axis_;
    ORT_ENFORCE(axis > -1 && axis < tensor_X->Shape().NumDimensions());

    auto tensor_K = ctx->Input<Tensor>(1);
    ORT_ENFORCE(nullptr != tensor_K);
    auto K = *(tensor_K->Data<int64_t>());
    ORT_ENFORCE(K > 0 && K <= tensor_X->Shape().GetDims()[axis]);

    auto output_shape = tensor_X->Shape();
    output_shape[axis] = K;
    auto tensor_V = ctx->Output(0, output_shape);
    auto tensor_I = ctx->Output(1, output_shape);

    auto elem_nums = tensor_X->Shape().GetDims();
    for (size_t i = elem_nums.size()-2; i >= 0; --i) {
      elem_nums[i] *= elem_nums[i+1];
    }

    if (tensor_X->DataType() == DataTypeImpl::GetType<uint8_t>()) {
      return TOPKIMPL(uint8_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<uint16_t>()) {
      return TOPKIMPL(uint16_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<uint32_t>()) {
      return TOPKIMPL(uint32_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<uint64_t>()) {
      return TOPKIMPL(uint64_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int8_t>()) {
      return TOPKIMPL(int8_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int16_t>()) {
      return TOPKIMPL(int16_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int32_t>()) {
      return TOPKIMPL(int32_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int64_t>()) {
      return TOPKIMPL(int64_t);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<float>()) {
      return TOPKIMPL(float);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
      return TOPKIMPL(MLFloat16);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<double>()) {
      return TOPKIMPL(double);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for TopK operator");
    }
}

}  // namespace cuda
}  // namespace onnxruntime
