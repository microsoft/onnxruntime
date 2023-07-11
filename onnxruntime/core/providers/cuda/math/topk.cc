// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk.h"
#include "topk_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    1, 9,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()}),
    TopK<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    TopK,
    kOnnxDomain,
    10, 10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    TopK<true>);

ONNX_OPERATOR_KERNEL_EX(
    TopK,
    kOnnxDomain,
    11,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>(),
                              DataTypeImpl::GetTensorType<int32_t>(),
                              DataTypeImpl::GetTensorType<int64_t>()})
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
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

#define IS_PRIM_TYPE(T) utils::IsPrimitiveDataType<T>(prim_type)
#define TOPKIMPL(T) TopKImpl<T>(this, use_deterministic_compute,                   \
                                ctx->GetComputeStream(), tensor_X->Data<T>(),      \
                                static_cast<T*>(tensor_V->MutableDataRaw()),       \
                                static_cast<int64_t*>(tensor_I->MutableDataRaw()), \
                                elem_nums_cuda,                                    \
                                elem_nums.size(),                                  \
                                axis, K_, largest_, sorted_, N, dimension)

template <bool inputk>
Status TopK<inputk>::ComputeInternal(OpKernelContext* ctx) const {
  auto tensor_X = ctx->Input<Tensor>(0);
  ORT_ENFORCE(nullptr != tensor_X);
  int32_t rank = static_cast<int32_t>(tensor_X->Shape().NumDimensions());
  int32_t axis = static_cast<int32_t>(axis_ < 0 ? rank + axis_ : axis_);
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

  auto elem_nums = tensor_X->Shape().AsShapeVector();
  auto dimension = elem_nums[axis];
  for (auto i = static_cast<int64_t>(elem_nums.size()) - 2; i >= 0; --i) {
    elem_nums[i] *= elem_nums[i + 1];
  }

  auto N = elem_nums[0] / dimension;
  TArray<int64_t> elem_nums_cuda(elem_nums);

  auto prim_type = tensor_X->DataType()->AsPrimitiveDataType();
  if (prim_type == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for TopK operator");
  }

  bool use_deterministic_compute = ctx->GetUseDeterministicCompute();

  if (IS_PRIM_TYPE(int32_t)) return TOPKIMPL(int32_t);
  if (IS_PRIM_TYPE(int64_t)) return TOPKIMPL(int64_t);
  if (IS_PRIM_TYPE(MLFloat16)) return TOPKIMPL(MLFloat16);
  if (IS_PRIM_TYPE(float)) return TOPKIMPL(float);
  if (IS_PRIM_TYPE(double)) return TOPKIMPL(double);

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for TopK operator");
}

}  // namespace cuda
}  // namespace onnxruntime
