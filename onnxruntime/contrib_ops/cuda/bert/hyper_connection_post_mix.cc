// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection_post_mix.h"

#include <limits>

#include "contrib_ops/cuda/bert/hyper_connection_post_mix_impl.h"
#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime::contrib::cuda {

#define REGISTER_HYPER_CONNECTION_POST_MIX_CUDA_KERNEL(T)                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      HyperConnectionPostMix, kMSDomain, 1, T, kCudaExecutionProvider,     \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      HyperConnectionPostMix<T>);

REGISTER_HYPER_CONNECTION_POST_MIX_CUDA_KERNEL(float)
REGISTER_HYPER_CONNECTION_POST_MIX_CUDA_KERNEL(MLFloat16)
REGISTER_HYPER_CONNECTION_POST_MIX_CUDA_KERNEL(BFloat16)

#undef REGISTER_HYPER_CONNECTION_POST_MIX_CUDA_KERNEL

template <typename T>
HyperConnectionPostMix<T>::HyperConnectionPostMix(const OpKernelInfo& info)
    : CudaKernel(info), num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status HyperConnectionPostMix<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  const auto* streams = context->Input<Tensor>(0);
  const auto* branch_output = context->Input<Tensor>(1);
  const auto* post_mix = context->Input<Tensor>(2);
  const auto* stream_mix = context->Input<Tensor>(3);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_NOT(params.branches <= std::numeric_limits<int>::max() &&
                        params.hidden <= std::numeric_limits<int>::max(),
                    "branch or hidden dimension is too large for CUDA");
  ORT_RETURN_IF_ERROR(hyper_connection::ValidateReduced(branch_output->Shape(), params));
  hyper_connection::GateLayout layout;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveGateShape(post_mix->Shape(), streams->Shape(), params, false, layout));
  if (stream_mix != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateStreamMix(stream_mix->Shape(), streams->Shape(), params));
  }
  auto* y = context->Output(0, streams->Shape());
  return LaunchHyperConnectionPostMix<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(streams->Data<T>()),
      reinterpret_cast<const CudaT*>(branch_output->Data<T>()), post_mix->DataRaw(),
      stream_mix == nullptr ? nullptr : stream_mix->DataRaw(), post_mix->GetElementType(),
      reinterpret_cast<CudaT*>(y->MutableData<T>()), streams->Shape().Size(),
      static_cast<int>(params.branches), static_cast<int>(params.hidden), static_cast<int>(layout));
}

template class HyperConnectionPostMix<float>;
template class HyperConnectionPostMix<MLFloat16>;
template class HyperConnectionPostMix<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
