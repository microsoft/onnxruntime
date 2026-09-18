// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection_pre_mix.h"

#include <limits>

#include "contrib_ops/cuda/bert/hyper_connection_pre_mix_impl.h"
#include "contrib_ops/hyper_connection_helper.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime::contrib::cuda {

#define REGISTER_HYPER_CONNECTION_PRE_MIX_CUDA_KERNEL(T)                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      HyperConnectionPreMix, kMSDomain, 1, T, kCudaExecutionProvider,      \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      HyperConnectionPreMix<T>);

REGISTER_HYPER_CONNECTION_PRE_MIX_CUDA_KERNEL(float)
REGISTER_HYPER_CONNECTION_PRE_MIX_CUDA_KERNEL(MLFloat16)
REGISTER_HYPER_CONNECTION_PRE_MIX_CUDA_KERNEL(BFloat16)

#undef REGISTER_HYPER_CONNECTION_PRE_MIX_CUDA_KERNEL

template <typename T>
HyperConnectionPreMix<T>::HyperConnectionPreMix(const OpKernelInfo& info)
    : CudaKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      reduction_scale_(info.GetAttrOrDefault<float>("reduction_scale", 1.0f)) {}

template <typename T>
Status HyperConnectionPreMix<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename onnxruntime::cuda::OrtToCudaType<T>::type;
  const auto* streams = context->Input<Tensor>(0);
  const auto* pre_mix = context->Input<Tensor>(1);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_NOT(params.branches <= std::numeric_limits<int>::max() &&
                        params.hidden <= std::numeric_limits<int>::max(),
                    "branch or hidden dimension is too large for CUDA");
  hyper_connection::GateLayout layout;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveGateShape(pre_mix->Shape(), streams->Shape(), params, false, layout));
  auto* y = context->Output(0, TensorShape(params.reduced_shape));
  return LaunchHyperConnectionPreMix<CudaT>(
      Stream(context), reinterpret_cast<const CudaT*>(streams->Data<T>()),
      pre_mix->DataRaw(), pre_mix->GetElementType(),
      reinterpret_cast<CudaT*>(y->MutableData<T>()), y->Shape().Size(),
      static_cast<int>(params.branches), static_cast<int>(params.hidden),
      static_cast<int>(layout), reduction_scale_);
}

template class HyperConnectionPreMix<float>;
template class HyperConnectionPreMix<MLFloat16>;
template class HyperConnectionPreMix<BFloat16>;

}  // namespace onnxruntime::contrib::cuda
