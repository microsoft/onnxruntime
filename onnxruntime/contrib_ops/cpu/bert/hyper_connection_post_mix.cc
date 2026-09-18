// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/hyper_connection_post_mix.h"

#include "contrib_ops/hyper_connection_helper.h"
#include "core/platform/threadpool.h"

namespace onnxruntime::contrib {

#define REGISTER_HYPER_CONNECTION_POST_MIX_CPU_KERNEL(T)                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      HyperConnectionPostMix, kMSDomain, 1, T, kCpuExecutionProvider,      \
      KernelDefBuilder()                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      HyperConnectionPostMix<T>);

REGISTER_HYPER_CONNECTION_POST_MIX_CPU_KERNEL(float)
REGISTER_HYPER_CONNECTION_POST_MIX_CPU_KERNEL(MLFloat16)
REGISTER_HYPER_CONNECTION_POST_MIX_CPU_KERNEL(BFloat16)

#undef REGISTER_HYPER_CONNECTION_POST_MIX_CPU_KERNEL

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
HyperConnectionPostMix<T>::HyperConnectionPostMix(const OpKernelInfo& info)
    : OpKernel(info), num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status HyperConnectionPostMix<T>::Compute(OpKernelContext* context) const {
  const auto* streams = context->Input<Tensor>(0);
  const auto* branch_output = context->Input<Tensor>(1);
  const auto* post_mix = context->Input<Tensor>(2);
  const auto* stream_mix = context->Input<Tensor>(3);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(hyper_connection::ValidateReduced(branch_output->Shape(), params));
  hyper_connection::GateLayout gate_layout;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveGateShape(post_mix->Shape(), streams->Shape(), params, false, gate_layout));
  if (stream_mix != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateStreamMix(stream_mix->Shape(), streams->Shape(), params));
  }

  auto* y = context->Output(0, streams->Shape());
  const auto* x_data = streams->Data<T>();
  const auto* branch_data = branch_output->Data<T>();
  auto* y_data = y->MutableData<T>();
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(streams->Shape().Size()),
      [&](ptrdiff_t i) {
        const int64_t h = i % params.hidden;
        const int64_t c = (i / params.hidden) % params.branches;
        const int64_t row = i / (params.hidden * params.branches);
        float value = stream_mix == nullptr ? static_cast<float>(x_data[i]) : 0.0f;
        if (stream_mix != nullptr) {
          for (int64_t source = 0; source < params.branches; ++source) {
            value += LoadFloat(*stream_mix, (row * params.branches + source) * params.branches + c) *
                     static_cast<float>(x_data[(row * params.branches + source) * params.hidden + h]);
          }
        }
        const int64_t gate_index =
            hyper_connection::GateOffset(gate_layout, row, c, h, params.branches, params.hidden);
        value += LoadFloat(*post_mix, gate_index) *
                 static_cast<float>(branch_data[row * params.hidden + h]);
        y_data[i] = static_cast<T>(value);
      },
      0);
  return Status::OK();
}

template class HyperConnectionPostMix<float>;
template class HyperConnectionPostMix<MLFloat16>;
template class HyperConnectionPostMix<BFloat16>;

}  // namespace onnxruntime::contrib
