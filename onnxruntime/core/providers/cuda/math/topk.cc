// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk.h"
#include "topk_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    1,9,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    TopK<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    10,10,
    kCudaExecutionProvider,
    KernelDefBuilder().InputMemoryType<OrtMemTypeCPUInput>(1).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    TopK<true>);

ONNX_OPERATOR_KERNEL_EX(
    TopK,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    KernelDefBuilder().InputMemoryType<OrtMemTypeCPUInput>(1).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    TopK<true>);

template <bool inputk>
TopK<inputk>::TopK(const OpKernelInfo& info) : CudaKernel(info) {
  info.GetAttrOrDefault<int64_t>("axis", &axis_, -1);
  info.GetAttrOrDefault<int64_t>("largest", &largest_, 1);
  info.GetAttrOrDefault<int64_t>("sorted", &sorted_, 1);
  if (!inputk) {
    info.GetAttrOrDefault<int64_t>("k", &K_, 0);
  }
}

#define ISTYPE(T) tensor_X->DataType() == DataTypeImpl::GetType<T>()
#define TOPKIMPL(T) TopKImpl<T>(this, tensor_X->Data<T>(),                         \
                                static_cast<T*>(tensor_V->MutableDataRaw()),       \
                                static_cast<int64_t*>(tensor_I->MutableDataRaw()), \
                                elem_nums_cuda.GpuPtr(),                           \
                                elem_nums.size(),                                  \
                                axis, K_, largest_, sorted_, N, dimension)

template <bool inputk>
Status TopK<inputk>::ComputeInternal(OpKernelContext* ctx) const {
  auto tensor_X = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != tensor_X);
  auto rank = static_cast<int64_t>(tensor_X->Shape().NumDimensions());
  auto axis = axis_ < 0 ? rank + axis_ : axis_;
  ORT_ENFORCE(axis > -1 && axis < rank);

  if (inputk) {
    auto tensor_K = ctx->Input<Tensor>(1);
    ORT_ENFORCE(nullptr != tensor_K);
    K_ = *tensor_K->Data<int64_t>();
    ORT_ENFORCE(K_ >= 0 && K_ <= tensor_X->Shape().GetDims()[axis]);
  }

  auto output_shape = tensor_X->Shape();
  output_shape[axis] = K_;
  auto tensor_V = ctx->Output(0, output_shape);
  auto tensor_I = ctx->Output(1, output_shape);

  if (0 == K_) {
    return Status::OK();
  }

  auto elem_nums = tensor_X->Shape().GetDims();
  auto dimension = elem_nums[axis];
  for (auto i = static_cast<int32_t>(elem_nums.size()) - 2; i >= 0; --i) {
    elem_nums[i] *= elem_nums[i + 1];
  }

  auto N = elem_nums[0] / dimension;
  CudaAsyncBuffer<int64_t> elem_nums_cuda(this, elem_nums);
  ORT_RETURN_IF_ERROR(elem_nums_cuda.CopyToGpu());

  if (ISTYPE(uint8_t)) return TOPKIMPL(uint8_t);
  if (ISTYPE(uint16_t)) return TOPKIMPL(uint16_t);
  if (ISTYPE(uint32_t)) return TOPKIMPL(uint32_t);
  if (ISTYPE(uint64_t)) return TOPKIMPL(uint64_t);
  if (ISTYPE(int8_t)) return TOPKIMPL(int8_t);
  if (ISTYPE(int16_t)) return TOPKIMPL(int16_t);
  if (ISTYPE(int32_t)) return TOPKIMPL(int32_t);
  if (ISTYPE(int64_t)) return TOPKIMPL(int64_t);
  if (ISTYPE(float)) return TOPKIMPL(float);
  if (ISTYPE(double)) return TOPKIMPL(double);
  if (ISTYPE(uint8_t)) return TOPKIMPL(uint8_t);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for TopK operator");
}

}  // namespace cuda
}  // namespace onnxruntime
