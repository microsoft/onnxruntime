// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/paged_attention.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      PagedAttention,                                                   \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()), \
      PagedAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
PagedAttention<T>::PagedAttention(const OpKernelInfo& info)
    : OpKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);

  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
}

template <typename T>
Status PagedAttention<T>::Compute(OpKernelContext* context) const {
  ORT_UNUSED_PARAMETER(context);

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "PagedAttention execution is not implemented on CPU. "
                         "Use CUDA Execution Provider for runtime execution.");
}

template class PagedAttention<MLFloat16>;
template class PagedAttention<BFloat16>;

}  // namespace contrib
}  // namespace onnxruntime