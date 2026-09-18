// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/hyper_connection_pre_mix.h"

#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "core/platform/threadpool.h"

namespace onnxruntime::contrib {

#define REGISTER_HYPER_CONNECTION_PRE_MIX_CPU_KERNEL(T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      HyperConnectionPreMix, kMSDomain, 1, T, kCpuExecutionProvider,       \
      KernelDefBuilder()                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      HyperConnectionPreMix<T>);

REGISTER_HYPER_CONNECTION_PRE_MIX_CPU_KERNEL(float)
REGISTER_HYPER_CONNECTION_PRE_MIX_CPU_KERNEL(MLFloat16)
REGISTER_HYPER_CONNECTION_PRE_MIX_CPU_KERNEL(BFloat16)

#undef REGISTER_HYPER_CONNECTION_PRE_MIX_CPU_KERNEL

namespace {

float LoadFloat(const Tensor& tensor, int64_t index) {
  switch (tensor.GetElementType()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return tensor.Data<float>()[index];
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return static_cast<float>(tensor.Data<MLFloat16>()[index]);
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      return static_cast<float>(tensor.Data<BFloat16>()[index]);
    default:
      ORT_THROW("Unsupported mixing tensor type.");
  }
}

}  // namespace

template <typename T>
HyperConnectionPreMix<T>::HyperConnectionPreMix(const OpKernelInfo& info)
    : OpKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      reduction_scale_(info.GetAttrOrDefault<float>("reduction_scale", 1.0f)) {}

template <typename T>
Status HyperConnectionPreMix<T>::Compute(OpKernelContext* context) const {
  const auto* streams = context->Input<Tensor>(0);
  const auto* pre_mix = context->Input<Tensor>(1);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  hyper_connection::GateLayout gate_layout;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveGateShape(pre_mix->Shape(), streams->Shape(), params, false, gate_layout, true));

  auto* y = context->Output(0, TensorShape(params.reduced_shape));
  const auto* x_data = streams->Data<T>();
  auto* y_data = y->MutableData<T>();
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(y->Shape().Size()),
      [&](ptrdiff_t i) {
        const int64_t row = i / params.hidden;
        const int64_t h = i % params.hidden;
        float sum = 0.0f;
        for (int64_t c = 0; c < params.branches; ++c) {
          const int64_t x_index = (row * params.branches + c) * params.hidden + h;
          const int64_t gate_index =
              hyper_connection::GateOffset(gate_layout, row, c, h, params.branches, params.hidden);
          sum += static_cast<float>(x_data[x_index]) * LoadFloat(*pre_mix, gate_index);
        }
        y_data[i] = static_cast<T>(sum * reduction_scale_);
      },
      0);
  return Status::OK();
}

template class HyperConnectionPreMix<float>;
template class HyperConnectionPreMix<MLFloat16>;
template class HyperConnectionPreMix<BFloat16>;

}  // namespace onnxruntime::contrib
