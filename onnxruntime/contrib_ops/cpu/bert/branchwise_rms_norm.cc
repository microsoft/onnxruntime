// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/branchwise_rms_norm.h"

#include <cmath>

#include "contrib_ops/cpu/hyper_connection_helper.h"
#include "core/platform/threadpool.h"

namespace onnxruntime::contrib {

#define REGISTER_BRANCHWISE_RMS_NORM_CPU_KERNEL(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      BranchwiseRMSNorm, kMSDomain, 1, T, kCpuExecutionProvider,           \
      KernelDefBuilder()                                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())           \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<float>(),      \
                                DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                DataTypeImpl::GetTensorType<BFloat16>()}), \
      BranchwiseRMSNorm<T>);

REGISTER_BRANCHWISE_RMS_NORM_CPU_KERNEL(float)
REGISTER_BRANCHWISE_RMS_NORM_CPU_KERNEL(MLFloat16)
REGISTER_BRANCHWISE_RMS_NORM_CPU_KERNEL(BFloat16)

#undef REGISTER_BRANCHWISE_RMS_NORM_CPU_KERNEL

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
      ORT_THROW("Unsupported scale tensor type.");
  }
}

}  // namespace

template <typename T>
BranchwiseRMSNorm<T>::BranchwiseRMSNorm(const OpKernelInfo& info)
    : LayerNormImpl(info, -1, info.GetAttrOrDefault<float>("epsilon", 1e-5f), true),
      epsilon_(info.GetAttrOrDefault<float>("epsilon", 1e-5f)),
      num_branches_(info.GetAttrOrDefault<int64_t>("num_branches", 0)) {}

template <typename T>
Status BranchwiseRMSNorm<T>::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                     bool& is_packed, PrePackedWeights* prepacked_weights) {
  ORT_UNUSED_PARAMETER(tensor);
  ORT_UNUSED_PARAMETER(input_idx);
  ORT_UNUSED_PARAMETER(alloc);
  ORT_UNUSED_PARAMETER(prepacked_weights);
  is_packed = false;
  return Status::OK();
}

template <typename T>
Status BranchwiseRMSNorm<T>::Compute(OpKernelContext* context) const {
  const auto* x = context->Input<Tensor>(0);
  const auto* scale = context->Input<Tensor>(1);
  hyper_connection::StreamShape params;
  ORT_RETURN_IF_ERROR(hyper_connection::ResolveStreamShape(x->Shape(), num_branches_, params));
  if (scale != nullptr) {
    ORT_RETURN_IF_ERROR(hyper_connection::ValidateScale(scale->Shape(), params));
  }

  auto* y = context->Output(0, x->Shape());
  if (scale != nullptr && scale->GetElementType() == x->GetElementType()) {
    TensorShapeVector grouped_dims(x->Shape().GetDims().begin(),
                                   x->Shape().GetDims().begin() + params.prefix_rank);
    grouped_dims.push_back(params.branches);
    grouped_dims.push_back(params.hidden);
    TensorShapeVector scale_dims{params.branches, params.hidden};
    const TensorShape normalized_scale_shape =
        scale->Shape().Size() == params.hidden ? scale->Shape() : TensorShape(scale_dims);
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    return ComputeWithoutContext<T, float>(
        x->Data<T>(), TensorShape(grouped_dims), scale->Data<T>(), normalized_scale_shape,
        nullptr, TensorShape(), y->MutableData<T>(), nullptr, nullptr,
        context->GetOperatorThreadPool(), -1, epsilon_, true, allocator);
  }

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
        const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(params.hidden) + epsilon_);
        for (int64_t h = 0; h < params.hidden; ++h) {
          const int64_t scale_index =
              scale != nullptr && scale->Shape().Size() == params.hidden
                  ? h
                  : (group % params.branches) * params.hidden + h;
          const float weight = scale == nullptr ? 1.0f : LoadFloat(*scale, scale_index);
          y_data[offset + h] =
              static_cast<T>(static_cast<float>(x_data[offset + h]) * inv_rms * weight);
        }
      },
      0);
  return Status::OK();
}

template class BranchwiseRMSNorm<float>;
template class BranchwiseRMSNorm<MLFloat16>;
template class BranchwiseRMSNorm<BFloat16>;

}  // namespace onnxruntime::contrib
