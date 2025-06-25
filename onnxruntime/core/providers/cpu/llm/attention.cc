// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/attention.h"

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;
namespace onnxruntime {

#define REGISTER_ONNX_KERNEL_TYPED(T)                                \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                    \
      Attention,                                                     \
      23,                                                            \
      T,                                                             \
      KernelDefBuilder()                                             \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>())    \
          .TypeConstraint("U", DataTypeImpl::GetTensorType<bool>()), \
      Attention<T>);

REGISTER_ONNX_KERNEL_TYPED(float)

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : OpKernel(info) {
  is_causal = static_cast<int>(info.GetAttrOrDefault<int64_t>("is_causal", 0)) == 1;
  kv_num_heads = static_cast<int>(info.GetAttrOrDefault<int64_t>("kv_num_heads", 0));
  q_num_heads = static_cast<int>(info.GetAttrOrDefault<int64_t>("q_num_heads", 0));
  int mode = static_cast<int>(info.GetAttrOrDefault<int64_t>("qk_matmul_output_mode", 0));
  qk_matmul_output_mode = static_cast<QKMatMulOutputMode>(mode);
  ORT_ENFORCE(qk_matmul_output_mode == QKMatMulOutputMode::kNone ||
                  qk_matmul_output_mode == QKMatMulOutputMode::kQK ||
                  qk_matmul_output_mode == QKMatMulOutputMode::kQKV,
              "qk_matmul_output_mode must be 0 (None), 1 (QK), or 2 (QKV)");
  scale = info.GetAttrOrDefault<float>("scale", 1.0f);
  softcap = info.GetAttrOrDefault<float>("softcap", 0.0f);
  softmax_precision = static_cast<int>(info.GetAttrOrDefault<int64_t>("softmax_precision", 0));
  ORT_ENFORCE(scale > 0, "Scale must be greater than 0");
  ORT_ENFORCE(softmax_precision == 1, "only float32 is supported for now");
}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  const Tensor* Q = context->Input<Tensor>(0);
  const Tensor* K = context->Input<Tensor>(1);
  const Tensor* V = context->Input<Tensor>(2);
  ORT_ENFORCE(Q != nullptr && K != nullptr && V != nullptr,
              "Q, K, and V inputs must not be null");
  ORT_ENFORCE(Q->Shape().NumDimensions() == 3 || Q->Shape().NumDimensions() == 4, "Q must be a 3D or 4D tensor");
  ORT_ENFORCE(K->Shape().NumDimensions() == 3 || K->Shape().NumDimensions() == 4, "K must be a 3D or 4D tensor");
  ORT_ENFORCE(V->Shape().NumDimensions() == 3 || V->Shape().NumDimensions() == 4, "V must be a 3D or 4D tensor");
  // const T* p_q = Q->Data<T>();
  // const T* p_k = K->Data<T>();
  // const T* p_v = V->Data<T>();

  ORT_THROW("not implemented yet");
}

}  // namespace onnxruntime

// TODO: rotary embedding in place