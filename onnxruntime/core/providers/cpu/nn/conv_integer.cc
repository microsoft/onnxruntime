// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "core/providers/cpu/nn/conv_integer.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/gemmlowp_common_wrapper.h"

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
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  int32_t input_offset = 0;
  int32_t filter_offset = 0;
  if (num_inputs >= 3) {
    const Tensor* X_Zero_Point = context->Input<Tensor>(2);
    if (X_Zero_Point->Shape().NumDimensions() == 0 ||
        (X_Zero_Point->Shape().NumDimensions() == 1 && X_Zero_Point->Shape().GetDims().size() == 1)) {
      input_offset = static_cast<int32_t>(*(X_Zero_Point->Data<uint8_t>()));
    } else {
      //TODO: Add support for per-channel quantization.
      return Status(common::ONNXRUNTIME, common::FAIL, "Non per-tensor quantization is not supported now.");
    }
  }
  if (num_inputs >= 4) {
    const Tensor* W_Zero_Point = context->Input<Tensor>(3);
    if (W_Zero_Point->Shape().NumDimensions() == 0 ||
        (W_Zero_Point->Shape().NumDimensions() == 1 && W_Zero_Point->Shape().GetDims().size() == 1)) {
      filter_offset = static_cast<int32_t>(*(W_Zero_Point->Data<uint8_t>()));
    } else {
      //TODO: Add support for per-channel quantization.
      return Status(common::ONNXRUNTIME, common::FAIL, "Non per-tensor quantization is not supported now.");
    }
  }

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(ValidateInputShape(X, W));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(ComputeKernelShape(W->Shape(), kernel_shape));

  std::vector<int64_t> pads(pads_);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(strides_);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const uint8_t* Xdata = X->template Data<uint8_t>();
  int32_t* Ydata = Y->template MutableData<int32_t>();

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / group_ * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / group_;
  const int64_t W_offset = W->Shape().Size() / group_;
  const int64_t kernel_dim = C / group_ * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  auto col_data = alloc->Alloc(sizeof(uint8_t) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  uint8_t* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  TensorShape image_shape = X->Shape().Slice(1);
  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), output_shape.GetDims().begin(),
                          output_shape.GetDims().end());

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
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

      const uint8_t* filter_data_as_uint8 = W->template Data<uint8_t>() + group_id * W_offset;
      static const gemmlowp::MapOrder ResultOrder = gemmlowp::MapOrder::RowMajor;
      static const gemmlowp::MapOrder LhsOrder = gemmlowp::MapOrder::RowMajor;
      static const gemmlowp::MapOrder RhsOrder = gemmlowp::MapOrder::RowMajor;
      gemmlowp::MatrixMap<const std::uint8_t, LhsOrder> lhs(
          filter_data_as_uint8, static_cast<int>(M / group_), static_cast<int>(kernel_dim));
      gemmlowp::MatrixMap<const std::uint8_t, RhsOrder> rhs(
          col_buffer_data, static_cast<int>(kernel_dim), static_cast<int>(output_image_size));
      gemmlowp::MatrixMap<std::int32_t, ResultOrder> result(
          Ydata + group_id * Y_offset, static_cast<int>(M / group_), static_cast<int>(output_image_size));
      const std::tuple<> empty_pipeline = {};

      gemmlowp::GemmContext gemm_context;
      // TODO: worker thread pool needs to be handled.
      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                       gemmlowp::DefaultL8R8BitDepthParams>(
          &gemm_context, lhs, rhs, &result, -filter_offset, -input_offset,
          empty_pipeline);
    }

    Xdata += X_offset * group_;
    Ydata += Y_offset * group_;
  }

  return Status::OK();
}
}  // namespace onnxruntime
