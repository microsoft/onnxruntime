// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decoder_attention.h"
#include "attention_impl.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      DecoderAttention,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DecoderAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
DecoderAttention<T>::DecoderAttention(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  static_kv_ = info.GetAttrOrDefault<int64_t>("static_kv", 0) == 1;
  use_past_ = info.GetAttrOrDefault<int64_t>("use_past", 0) == 1;
}

template <typename T>
Status DecoderAttention<T>::ComputeInternal(OpKernelContext* context) const {
  

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
