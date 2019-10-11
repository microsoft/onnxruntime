// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "nchwc_ops.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

#define ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMSNchwcDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    ReorderInput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderInput<float>);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    ReorderOutput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderOutput<float>);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    Conv,
    1,
    float,
    KernelDefBuilder()
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcConv);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    MaxPool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcMaxPool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    GlobalMaxPool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcMaxPool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    AveragePool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcAveragePool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    GlobalAveragePool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcAveragePool);

template <typename T>
Status ReorderInput<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);
  auto* Y = context->Output(0, X_shape);
  MlasReorderInput(X_shape.GetDims().data(), X->template Data<T>(), Y->template MutableData<T>());
  return Status::OK();
}

template <typename T>
Status ReorderOutput<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  std::vector<int64_t> Y_shape(X_shape.GetDims());
  ORT_ENFORCE(channels_ <= Y_shape[1]);
  Y_shape[1] = channels_;
  auto* Y = context->Output(0, Y_shape);
  MlasReorderOutput(Y_shape.data(), X->template Data<T>(), Y->template MutableData<T>());
  return Status::OK();
}

Status NchwcConv::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* B = context->Input<Tensor>(2);
  const auto* Sum = context->Input<Tensor>(3);

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  const auto& X_shape = X->Shape();
  const auto& W_shape = W->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();
  ORT_ENFORCE((static_cast<size_t>(X_shape[1]) < nchwc_block_size) || ((X_shape[1] % nchwc_block_size) == 0));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));
  if (kernel_shape.size() != 2) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported convolution size.");
  }

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {X_shape[0], W_shape[0]});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  auto* Y = context->Output(0, Y_dims);
  auto* y_data = Y->template MutableData<float>();

  // Check for the optional Conv/Sum fusion.
  if (Sum != nullptr) {
    const auto& sum_shape = Sum->Shape();
    ORT_RETURN_IF_NOT(Y->Shape() == sum_shape, "output and sum shape must match");
    // If the output was not allocated inplace with the sum tensor, then copy here.
    const auto* sum_data = Sum->template Data<float>();
    if (y_data != sum_data) {
      memcpy(y_data, sum_data, sum_shape.Size() * sizeof(float));
    }
  }

  MlasNchwcConv(kernel_shape.size(),
                X_shape.GetDims().data(),
                kernel_shape.data(),
                dilations.data(),
                pads.data(),
                strides.data(),
                Y_dims.data(),
                static_cast<size_t>(conv_attrs_.group),
                X->template Data<float>(),
                W->template Data<float>(),
                B != nullptr ? B->template Data<float>() : nullptr,
                y_data,
                &activation_,
                Sum == nullptr,
                const_cast<concurrency::ThreadPool*>(static_cast<OpKernelContextInternal*>(context)->GetOperatorThreadPool()));

  return Status::OK();
}

Status NchwcPoolBase::NchwcPool(OpKernelContext* context, MLAS_POOLING_KIND kind) const {
  const auto* X = context->Input<Tensor>(0);

  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);


  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(X_shape, X_shape[1], &pads);
  auto* Y = context->Output(0, output_dims);

  MlasNchwcPool(kind,
                2,
                X_shape.GetDims().data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.kernel_shape.data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.dilations.data(),
                pool_attrs_.global_pooling ? nullptr : pads.data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.strides.data(),
                output_dims.data(),
                X->template Data<float>(),
                Y->template MutableData<float>(),
                const_cast<concurrency::ThreadPool*>(static_cast<OpKernelContextInternal*>(context)->GetOperatorThreadPool()));

  return Status::OK();
}

Status NchwcMaxPool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, MlasMaximumPooling);
}

Status NchwcAveragePool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, pool_attrs_.count_include_pad ? MlasAveragePoolingIncludePad :
                                                                           MlasAveragePoolingExcludePad);
}

}  // namespace contrib
}  // namespace onnxruntime
