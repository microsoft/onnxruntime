// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dsv4_inv_rope_group.h"

#include "contrib_ops/cuda/math/dsv4_inv_rope_group_impl.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      DSV4InvRopeGroup,                                               \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<float>()), \
      DSV4InvRopeGroup<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
DSV4InvRopeGroup<T>::DSV4InvRopeGroup(const OpKernelInfo& info) : CudaKernel(info) {
  num_heads_ = info.GetAttrOrDefault<int64_t>("num_heads", static_cast<int64_t>(0));
  head_dim_ = info.GetAttrOrDefault<int64_t>("head_dim", static_cast<int64_t>(0));
  rope_head_dim_ = info.GetAttrOrDefault<int64_t>("rope_head_dim", static_cast<int64_t>(0));
  num_groups_ = info.GetAttrOrDefault<int64_t>("num_groups", static_cast<int64_t>(1));

  ORT_ENFORCE(num_heads_ > 0, "num_heads must be positive, got ", num_heads_);
  ORT_ENFORCE(head_dim_ > 0, "head_dim must be positive, got ", head_dim_);
  ORT_ENFORCE(rope_head_dim_ >= 0 && rope_head_dim_ <= head_dim_,
              "rope_head_dim must be in [0, head_dim], got ", rope_head_dim_);
  ORT_ENFORCE(rope_head_dim_ % 2 == 0, "rope_head_dim must be even, got ", rope_head_dim_);
  ORT_ENFORCE(num_groups_ > 0 && (num_heads_ * head_dim_) % num_groups_ == 0,
              "num_heads * head_dim must divide evenly into num_groups, got ", num_groups_);
}

template <typename T>
Status DSV4InvRopeGroup<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* cos_table = context->Input<Tensor>(1);
  const Tensor* sin_table = context->Input<Tensor>(2);

  const auto& dims = input->Shape().GetDims();
  if (dims.size() != 2 || dims[1] != num_heads_ * head_dim_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input must be [tokens, ",
                           num_heads_ * head_dim_, "], got ", input->Shape());
  }
  const int64_t tokens = dims[0];
  const int64_t rope_elems = tokens * rope_head_dim_;
  if (rope_head_dim_ > 0 &&
      (cos_table->Shape().Size() != rope_elems || sin_table->Shape().Size() != rope_elems)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "the rotary slices must hold ",
                           rope_elems, " elements each.");
  }

  const int64_t group_dim = num_heads_ * head_dim_ / num_groups_;
  Tensor* output = context->Output(0, TensorShape({num_groups_, tokens, group_dim}));
  if (tokens == 0) return Status::OK();

  DSV4InvRopeGroupParams params;
  params.num_tokens = static_cast<int>(tokens);
  params.num_heads = static_cast<int>(num_heads_);
  params.head_dim = static_cast<int>(head_dim_);
  params.rope_head_dim = static_cast<int>(rope_head_dim_);
  params.nope_dim = static_cast<int>(head_dim_ - rope_head_dim_);
  params.num_groups = static_cast<int>(num_groups_);
  params.group_dim = static_cast<int>(group_dim);

  return LaunchDSV4InvRopeGroup<T>(Stream(context), params, input->Data<T>(),
                                   cos_table->Data<float>(), sin_table->Data<float>(),
                                   output->MutableData<T>());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
