// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/hyper_connection.h"

#include <cmath>

#include "contrib_ops/hyper_connection_helper.h"
#include "core/platform/threadpool.h"

namespace onnxruntime::contrib {

using hyper_connection::GateLayout;
using hyper_connection::StreamShape;

#define REGISTER_HC_CPU_KERNEL(Op, T)             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                  \
      Op, kMSDomain, 1, T, kCpuExecutionProvider, \
      KernelDefBuilder()                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(), \
                                 DataTypeImpl::GetTensorType<MLFloat16>(), \
                                 DataTypeImpl::GetTensorType<BFloat16>()}), \
      Op<T>);

#define REGISTER_HC_CPU_TYPES(Op)       \
  REGISTER_HC_CPU_KERNEL(Op, float)     \
  REGISTER_HC_CPU_KERNEL(Op, MLFloat16) \
  REGISTER_HC_CPU_KERNEL(Op, BFloat16)

REGISTER_HC_CPU_TYPES(BranchwiseRMSNorm)
REGISTER_HC_CPU_TYPES(ScaledSiLU)
REGISTER_HC_CPU_TYPES(HyperConnectionPreMix)
REGISTER_HC_CPU_TYPES(HyperConnectionPostMix)

#undef REGISTER_HC_CPU_TYPES
#undef REGISTER_HC_CPU_KERNEL

namespace {

inline float Sigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-value));
  }
  const float e = std::exp(value);
  return e / (1.0f + e);
}

inline float LoadFloat(const Tensor& tensor, int64_t index) {
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
BranchwiseRMSNorm<T>::BranchwiseRMSNorm(const OpKernelInfo& info)
    : OpKernel(info),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status BranchwiseRMSNorm<T>::Compute(OpKernelContext* context) const {
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  StreamShape params;
  ORT_RETURN_IF_ERROR(
      hyper_connection::ResolveStreamShape(x->Shape(), num_branches_, params));
  if (scale != nullptr) {
    ORT_RETURN_IF_ERROR(
        hyper_connection::ValidateScale(scale->Shape(), params));
  }

  auto* y = context->Output(0, x->Shape());
  const auto* x_data = x->Data<T>();
  auto* y_data = y->MutableData<T>();
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(),
      onnxruntime::narrow<ptrdiff_t>(x->Shape().Size() / params.hidden),
      [&](ptrdiff_t group) {
        const int64_t offset = group * params.hidden;
        float sum_sq = 0.0f;
        for (int64_t h = 0; h < params.hidden; ++h) {
          const float value = static_cast<float>(x_data[offset + h]);
          sum_sq += value * value;
        }
        const float inv_rms =
            1.0f / std::sqrt(sum_sq / static_cast<float>(params.hidden) + epsilon_);
        for (int64_t h = 0; h < params.hidden; ++h) {
          const int64_t scale_index =
              scale != nullptr && scale->Shape().Size() == params.hidden
                  ? h
                  : (group % params.branches) * params.hidden + h;
          const float weight = scale == nullptr ? 1.0f : LoadFloat(*scale, scale_index);
          y_data[offset + h] =
              static_cast<T>(static_cast<float>(x_data[offset + h]) * inv_rms *
                             weight);
        }
      },
      0);
  return Status::OK();
}

template <typename T>
ScaledSiLU<T>::ScaledSiLU(const OpKernelInfo& info)
    : OpKernel(info), alpha_(info.GetAttrOrDefault<float>("alpha", 1.0f)) {}

template <typename T>
Status ScaledSiLU<T>::Compute(OpKernelContext* context) const {
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  ORT_RETURN_IF_NOT(scale == nullptr || scale->Shape().NumDimensions() == 0,
                    "scale must be a scalar");
  const float scale_value = scale == nullptr ? alpha_ : LoadFloat(*scale, 0);
  auto* y = context->Output(0, x->Shape());
  const auto* x_data = x->Data<T>();
  auto* y_data = y->MutableData<T>();
  const int64_t count = x->Shape().Size();
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), onnxruntime::narrow<ptrdiff_t>(count),
      [&](ptrdiff_t i) {
        const T z = static_cast<T>(static_cast<float>(x_data[i]) * scale_value);
        const T sigmoid = static_cast<T>(Sigmoid(static_cast<float>(z)));
        y_data[i] =
            static_cast<T>(static_cast<float>(z) * static_cast<float>(sigmoid));
      },
      0);
  return Status::OK();
}

template <typename T>
HyperConnectionPreMix<T>::HyperConnectionPreMix(const OpKernelInfo& info)
    : OpKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)),
      reduction_scale_(
          info.GetAttrOrDefault<float>("reduction_scale", 1.0f)) {}

template <typename T>
Status HyperConnectionPreMix<T>::Compute(OpKernelContext* context) const {
  const auto* streams = context->Input<Tensor>(0);
  const auto* pre_mix = context->Input<Tensor>(1);
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      streams->Shape(), num_branches_, params));
  GateLayout gate_layout;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
      pre_mix->Shape(), streams->Shape(), params, false, gate_layout));

  auto* y = context->Output(0, TensorShape(params.reduced_shape));
  const auto* x_data = streams->Data<T>();
  auto* y_data = y->MutableData<T>();
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(),
      onnxruntime::narrow<ptrdiff_t>(y->Shape().Size()),
      [&](ptrdiff_t i) {
        const int64_t row = i / params.hidden;
        const int64_t h = i % params.hidden;
        float sum = 0.0f;
        for (int64_t c = 0; c < params.branches; ++c) {
          const int64_t x_index =
              (row * params.branches + c) * params.hidden + h;
          const int64_t gate_index = hyper_connection::GateOffset(
              gate_layout, row, c, h, params.branches, params.hidden);
          sum += static_cast<float>(x_data[x_index]) *
                 LoadFloat(*pre_mix, gate_index);
        }
        y_data[i] = static_cast<T>(sum * reduction_scale_);
      },
      0);
  return Status::OK();
}

template <typename T>
HyperConnectionPostMix<T>::HyperConnectionPostMix(const OpKernelInfo& info)
    : OpKernel(info),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status HyperConnectionPostMix<T>::Compute(OpKernelContext* context) const {
  const auto* streams = context->Input<Tensor>(0);
  const auto* branch_output = context->Input<Tensor>(1);
  const auto* post_mix = context->Input<Tensor>(2);
  const auto* stream_mix = context->Input<Tensor>(3);
  StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(
      streams->Shape(), num_branches_, params));
  ORT_RETURN_IF_ERROR(
      hyper_connection::ValidateReduced(branch_output->Shape(), params));
  GateLayout gate_layout;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveGateShape(
      post_mix->Shape(), streams->Shape(), params, false, gate_layout));
  if (stream_mix != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateStreamMix(
        stream_mix->Shape(), streams->Shape(), params));
  }

  auto* y = context->Output(0, streams->Shape());
  const auto* x_data = streams->Data<T>();
  const auto* branch_data = branch_output->Data<T>();
  auto* y_data = y->MutableData<T>();
  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(),
      onnxruntime::narrow<ptrdiff_t>(streams->Shape().Size()),
      [&](ptrdiff_t i) {
        const int64_t h = i % params.hidden;
        const int64_t c = (i / params.hidden) % params.branches;
        const int64_t row = i / (params.hidden * params.branches);
        float value = stream_mix == nullptr
                          ? static_cast<float>(x_data[i])
                          : 0.0f;
        if (stream_mix != nullptr) {
          for (int64_t source = 0; source < params.branches; ++source) {
            value += LoadFloat(
                         *stream_mix,
                         (row * params.branches + source) * params.branches + c) *
                     static_cast<float>(
                         x_data[(row * params.branches + source) *
                                    params.hidden +
                                h]);
          }
        }
        const int64_t gate_index = hyper_connection::GateOffset(
            gate_layout, row, c, h, params.branches, params.hidden);
        value += LoadFloat(*post_mix, gate_index) *
                 static_cast<float>(branch_data[row * params.hidden + h]);
        y_data[i] = static_cast<T>(value);
      },
      0);
  return Status::OK();
}

template class BranchwiseRMSNorm<float>;
template class BranchwiseRMSNorm<MLFloat16>;
template class BranchwiseRMSNorm<BFloat16>;
template class ScaledSiLU<float>;
template class ScaledSiLU<MLFloat16>;
template class ScaledSiLU<BFloat16>;
template class HyperConnectionPreMix<float>;
template class HyperConnectionPreMix<MLFloat16>;
template class HyperConnectionPreMix<BFloat16>;
template class HyperConnectionPostMix<float>;
template class HyperConnectionPostMix<MLFloat16>;
template class HyperConnectionPostMix<BFloat16>;

}  // namespace onnxruntime::contrib
