// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/conv_integer.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/providers/common.h"

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    ConvInteger,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    ConvInteger);

Status ConvInteger::Compute(OpKernelContext* context) const {

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  uint8_t input_offset = 0;
  uint8_t filter_offset = 0;
  if (num_inputs >= 3) {
    const auto* X_Zero_Point = context->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(X_Zero_Point), "Must be a scalar or 1D tensor or size 1.");
    input_offset = *(X_Zero_Point->Data<uint8_t>());
  }
  if (num_inputs >= 4) {
    const auto* W_Zero_Point = context->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(W_Zero_Point), "Non per-tensor quantization is not supported now.");
    filter_offset = *(W_Zero_Point->Data<uint8_t>());
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

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
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const auto* Xdata = X->template Data<uint8_t>();
  auto* Ydata = Y->template MutableData<int32_t>();

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  auto col_data = alloc->Alloc(sizeof(uint8_t) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  TensorShape image_shape = X->Shape().Slice(1);
  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), output_shape.GetDims().begin(),
                          output_shape.GetDims().end());

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      math::Im2colNd<uint8_t, CPUMathUtil, StorageOrder::NCHW>()(
          Xdata + group_id * X_offset,
          image_shape.GetDims().data(),
          col_buffer_shape.data(),
          C * input_image_size,
          col_buffer_size,
          kernel_shape.data(),
          strides.data(),
          dilations.data(),
          pads.data(),
          static_cast<int>(kernel_shape.size()),
          col_buffer_data,
          &CPUMathUtil::Instance(),
          false,
          input_offset);

      QGemmu8u8_s32(static_cast<int>(M / conv_attrs_.group),
                    static_cast<int>(output_image_size),
                    static_cast<int>(kernel_dim),
                    W->template Data<uint8_t>() + group_id * W_offset,
                    static_cast<int>(kernel_dim),
                    filter_offset,
                    col_buffer_data,
                    static_cast<int>(output_image_size),
                    input_offset,
                    Ydata + group_id * Y_offset,
                    static_cast<int>(output_image_size),
                    nullptr);
    }

    Xdata += X_offset * conv_attrs_.group;
    Ydata += Y_offset * conv_attrs_.group;
  }

  return Status::OK();
}
}  // namespace onnxruntime
