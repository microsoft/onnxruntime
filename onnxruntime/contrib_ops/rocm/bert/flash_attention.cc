// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/flash_attention.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm.h"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FlashAttention,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                                            \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),              \
      FlashAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
FlashAttention<T>::FlashAttention(const OpKernelInfo& info) : RocmKernel(info) {
  int64_t num_heads;
  info.GetAttrOrDefault("num_heads", &num_heads, static_cast<int64_t>(0));
  this->num_heads_ = num_heads;
}

template <typename T>
Status FlashAttention<T>::CheckInputs(const TensorShape &query_shape,
                                 const TensorShape &key_shape,
                                 const TensorShape &value_shape,
                                 const Tensor* att_mask,
                                 const Tensor* att_bias) const {
  return Status::OK();
}

template <typename T>
Status FlashAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* att_mask = context->Input<Tensor>(3);  // optional
  const Tensor* att_bias = context->Input<Tensor>(4);  // optional

  ORT_RETURN_IF_ERROR(CheckInputs(query->Shape(),
                                  key->Shape(),
                                  value->Shape(),
                                  att_mask,
                                  att_bias));

  auto query_shape = query->Shape();
  auto value_shape = value->Shape();
  TensorShape output_shape(query_shape);
  output_shape[query_shape.NumDimensions() - 1] = value_shape[value_shape.NumDimensions() - 1];

  Tensor* output = context->Output(0, output_shape);
  ORT_UNUSED_PARAMETER(output);

  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
