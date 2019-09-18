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

Status TopK::ComputeInternal(OpKernelContext* ctx) const {
    auto tensor_X = ctx->Input<Tensor>(0);
    ORT_ENFORCE(nullptr != tensor_X);
    auto axis = axis_ < 0 ? tensor_X->Shape().NumDimensions() + axis_ : axis_;
    ORT_ENFORCE(axis > -1 && axis < tensor_X->Shape().NumDimensions());
    // auto vars = tensor_X->Shape().Size() / tensor_X->Shape().GetDims()[axis];

    auto tensor_K = ctx->Input<Tensor>(1);
    ORT_ENFORCE(nullptr != tensor_K);
    auto K = *(tensor_K->Data<int64_t>());
    ORT_ENFORCE(K > 0 && K <= tensor_X->Shape().GetDims()[axis]);

    auto output_shape = tensor_X->Shape();
    output_shape[axis] = K;
    auto tensor_V = ctx->Output(0, output_shape);
    auto tensor_I = ctx->Output(1, output_shape);

    if (tensor_X->DataType() == DataTypeImpl::GetType<uint8_t>()) {
      return TopKImpl<uint8_t> (tensor_X->Data<uint8_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<uint16_t>()) {
      return TopKImpl<uint16_t> (tensor_X->Data<uint16_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<uint32_t>()) {
      return TopKImpl<uint32_t> (tensor_X->Data<uint32_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<uint64_t>()) {
      return TopKImpl<uint64_t> (tensor_X->Data<uint64_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int8_t>()) {
      return TopKImpl<int8_t> (tensor_X->Data<int8_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int16_t>()) {
      return TopKImpl<int16_t> (tensor_X->Data<int16_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int32_t>()) {
      return TopKImpl<int32_t> (tensor_X->Data<int32_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<int64_t>()) {
      return TopKImpl<int64_t> (tensor_X->Data<int64_t>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<float>()) {
      return TopKImpl<float> (tensor_X->Data<float>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<MLFloat16>()) {
      return TopKImpl<MLFloat16> (tensor_X->Data<MLFloat16>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else if (tensor_X->DataType() == DataTypeImpl::GetType<double>()) {
      return TopKImpl<double> (tensor_X->Data<double>(),
                                tensor_V->MutableDataRaw(),
                                tensor_I->MutableDataRaw(),
                                tensor_X->Shape().GetDims().data(),
                                tensor_X->Shape().Size(),
                                axis, K, largest_, sorted_);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for TopK operator");
    }
}

}  // namespace cuda
}  // namespace onnxruntime
