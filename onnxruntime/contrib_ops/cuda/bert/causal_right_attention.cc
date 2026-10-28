// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if USE_MEMORY_EFFICIENT_ATTENTION

#include <cmath>

#include "contrib_ops/cuda/bert/causal_right_attention.h"
#include "contrib_ops/cuda/bert/causal_right_attention_impl.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime::contrib::cuda {

using namespace onnxruntime::cuda;

#define REGISTER_CAUSAL_RIGHT_ATTENTION(T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      CausalRightAttention, kMSDomain, 1, T, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),    \
      CausalRightAttention<T>);

REGISTER_CAUSAL_RIGHT_ATTENTION(MLFloat16)
REGISTER_CAUSAL_RIGHT_ATTENTION(BFloat16)

#undef REGISTER_CAUSAL_RIGHT_ATTENTION

template <typename T>
CausalRightAttention<T>::CausalRightAttention(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<float>("scale", &scale_).IsOK(), "scale is required");
  ORT_ENFORCE(std::isfinite(scale_) && scale_ > 0, "scale must be finite and positive");
}

template <typename T>
Status CausalRightAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const auto* query = context->Input<Tensor>(0);
  const auto* key = context->Input<Tensor>(1);
  const auto* value = context->Input<Tensor>(2);
  const auto& q = query->Shape();
  const auto& k = key->Shape();
  const auto& v = value->Shape();
  ORT_RETURN_IF_NOT(q.NumDimensions() == 4 && k.NumDimensions() == 4 && v == k,
                    "CausalRightAttention requires 4D Q and equal-shape K/V");
  ORT_RETURN_IF_NOT(q[0] >= 1 && q[0] <= 8 && q[1] >= 1 && q[1] <= 32 &&
                        q[2] >= 1 && q[2] <= 256 && q[3] == 256 &&
                        k[0] == q[0] && k[1] == q[1] && k[2] >= q[2] && k[2] <= 256 && k[3] == 256,
                    "CausalRightAttention requires B=1..8, equal H=1..32, 1<=S<=T<=256, D=256");
  ORT_RETURN_IF_NOT(GetDeviceProp().major == 9, "CausalRightAttention currently requires compute capability 9.x");
  auto stream = Stream(context);
  cudaStreamCaptureStatus capture;
  CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture));
  ORT_RETURN_IF_NOT(capture == cudaStreamCaptureStatusNone,
                    "CausalRightAttention does not support CUDA graph capture");

  using CudaT = typename OrtToCudaType<T>::type;
  std::call_once(configure_, [] { ConfigureCausalRightAttention<CudaT>(); });
  const int batch = static_cast<int>(q[0]);
  const int heads = static_cast<int>(q[1]);
  const int queries = static_cast<int>(q[2]);
  const int keys = static_cast<int>(k[2]);
  auto workspace = GetScratchBuffer<void>(CausalRightAttentionWorkspaceSize(batch, heads, queries, keys),
                                          GetComputeStream(context));
  auto* output = context->Output(0, q);
  return LaunchCausalRightAttention(
      stream, reinterpret_cast<const CudaT*>(query->Data<T>()), reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(value->Data<T>()), reinterpret_cast<CudaT*>(output->MutableData<T>()),
      workspace.get(), batch, heads, queries, keys, scale_);
}

template class CausalRightAttention<MLFloat16>;
template class CausalRightAttention<BFloat16>;

}  // namespace onnxruntime::contrib::cuda

#endif  // USE_MEMORY_EFFICIENT_ATTENTION
