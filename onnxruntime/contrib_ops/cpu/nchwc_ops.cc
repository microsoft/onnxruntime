// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
    ReorderInput);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    ReorderOutput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderOutput);

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

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    Upsample,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcUpsample);

Status ReorderInput::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);

  auto* Y = context->Output(0, X_shape);
  MlasReorderInput(X_shape.GetDims().data(), X->template Data<float>(), Y->template MutableData<float>());

  return Status::OK();
}

Status ReorderOutput::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  const auto X_rank = X_shape.NumDimensions();
  ORT_ENFORCE(X_rank == 4);
  ORT_ENFORCE(channels_ <= X_shape[1]);

  // Build the output shape in NCHW or NHWC order.
  std::vector<int64_t> Y_shape(X_rank);
  Y_shape[0] = X_shape[0];
  Y_shape[channels_last_ ? X_rank - 1 : 1] = channels_;
  auto* Y_spatial_dims = Y_shape.data() + (channels_last_ ? 1 : 2);
  for (size_t i = 0; i < X_rank - 2; i++) {
    Y_spatial_dims[i] = X_shape[2 + i];
  }
  auto* Y = context->Output(0, Y_shape);

  const auto* x_data = X->template Data<float>();
  auto* y_data = Y->template MutableData<float>();
  if (channels_last_) {
    MlasReorderOutputNhwc(Y_shape.data(), x_data, y_data);
  } else {
    MlasReorderOutputNchw(Y_shape.data(), x_data, y_data);
  }

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
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
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

  MlasNchwcConv(X_shape.GetDims().data(),
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
                context->GetOperatorThreadPool());

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
                X_shape.GetDims().data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.kernel_shape.data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.dilations.data(),
                pool_attrs_.global_pooling ? nullptr : pads.data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.strides.data(),
                output_dims.data(),
                X->template Data<float>(),
                Y->template MutableData<float>(),
                context->GetOperatorThreadPool());

  return Status::OK();
}

Status NchwcMaxPool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, MlasMaximumPooling);
}

Status NchwcAveragePool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, pool_attrs_.count_include_pad ? MlasAveragePoolingIncludePad
                                                                         : MlasAveragePoolingExcludePad);
}

Status NchwcUpsample::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);

  TensorShape Y_shape{X_shape[0], X_shape[1], X_shape[2] * scales_[2], X_shape[3] * scales_[3]};
  auto* Y = context->Output(0, Y_shape);

  MlasNchwcUpsample(X_shape.GetDims().data(),
                    scales_.data() + 2,
                    X->template Data<float>(),
                    Y->template MutableData<float>());

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
