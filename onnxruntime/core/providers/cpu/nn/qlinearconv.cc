// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "core/providers/cpu/nn/qlinearconv.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
ONNX_OPERATOR_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>()),
    QLinearConv);

Status QLinearConv::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(3);

  // validate scale and zero points
  auto input_scale = context->Input<Tensor>(1);
  auto input_offset = context->Input<Tensor>(2);
  ScaleAndZeropointPairValidationHelper(input_scale, input_offset);
  auto filter_scale = context->Input<Tensor>(4);
  auto filter_offset = context->Input<Tensor>(5);
  ScaleAndZeropointPairValidationHelper(filter_scale, filter_offset);
  auto result_scale = context->Input<Tensor>(6);
  auto result_offset = context->Input<Tensor>(7);
  ScaleAndZeropointPairValidationHelper(result_scale, result_offset);

  auto input_scale_data = *(input_scale->template Data<float>());
  auto filter_scale_data = *(filter_scale->template Data<float>());
  auto result_scale_data = *(result_scale->template Data<float>());

  auto input_offset_data = *(input_offset->template Data<uint8_t>());
  auto filter_offset_data = *(filter_offset->template Data<uint8_t>());
  auto result_offset_data = *(result_offset->template Data<uint8_t>());

  const float real_multiplier = (input_scale_data * filter_scale_data) / result_scale_data;
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* bias = nullptr;
  if (num_inputs == 9) {
    bias = context->Input<Tensor>(8);
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

  const auto* Xdata = X->template Data<uint8_t>();
  auto* Ydata = Y->template MutableData<uint8_t>();

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / group_ * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / group_;
  const int64_t W_offset = W->Shape().Size() / group_;  
  const int64_t kernel_dim = C / group_ * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;
  const int bias_offset = static_cast<int>(M / group_);

  auto col_data = alloc->Alloc(sizeof(uint8_t) * col_buffer_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

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
          input_offset_data);

      const uint8_t* filter_data_as_uint8 = W->template Data<uint8_t>() + group_id * W_offset;
      static const gemmlowp::MapOrder MatOrder = gemmlowp::MapOrder::RowMajor;
      gemmlowp::MatrixMap<const std::uint8_t, MatOrder> lhs(
          filter_data_as_uint8, static_cast<int>(M / group_), static_cast<int>(kernel_dim));
      gemmlowp::MatrixMap<const std::uint8_t, MatOrder> rhs(
          col_buffer_data, static_cast<int>(kernel_dim), static_cast<int>(output_image_size));
      gemmlowp::MatrixMap<std::uint8_t, MatOrder> result(
          Ydata + group_id * Y_offset, static_cast<int>(M / group_), static_cast<int>(output_image_size));

      // TODO: worker thread pool needs to be handled.
      gemmlowp::GemmContext gemm_context;
      if (bias == nullptr) {
        auto output_pipeline = MakeOutputPipelineWithOutBias(result_offset_data, 
            integer_multiplier, right_shift);
        gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                         gemmlowp::DefaultL8R8BitDepthParams>(
            &gemm_context, lhs, rhs, &result, -filter_offset_data, -input_offset_data,
            output_pipeline);        
      } else {
        auto output_pipeline = MakeOutputPipelineWithBias(bias->template Data<int32_t>() + group_id * bias_offset, 
            static_cast<int>(M / group_), result_offset_data, integer_multiplier, right_shift);
        gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                         gemmlowp::DefaultL8R8BitDepthParams>(
            &gemm_context, lhs, rhs, &result, -filter_offset_data, -input_offset_data,
            output_pipeline);
      }      
    }

    Xdata += X_offset * group_;
    Ydata += Y_offset * group_;
  }

  return Status::OK();
}

void QLinearConv::QuantizeMultiplier(float fp_multiplier, std::int32_t* integer_multiplier, int* right_shift) const {
  auto* fp_as_bits = reinterpret_cast<uint32_t*>(&fp_multiplier);
  auto current_exponent = (*fp_as_bits >> 23);
  // bring multiplier in [.5,1) range and calculate the shift
  auto bumped_multiplier_as_bits =
      (*fp_as_bits & UINT32_C(0x007fffff)) | UINT32_C(0x3f000000);
  auto* bumped_multiplier = reinterpret_cast<float*>(&bumped_multiplier_as_bits);
  auto shift = 126 - current_exponent;
  // convert to fixed point number
  auto int_multiplier = static_cast<std::int64_t>(std::round(*bumped_multiplier * (1ll << 31)));

  *integer_multiplier = static_cast<int32_t>(int_multiplier);
  *right_shift = shift;
}

void QLinearConv::ScaleAndZeropointPairValidationHelper(const Tensor* scale, const Tensor* zeropoint) const {
  ORT_ENFORCE(scale->Shape().NumDimensions() == 0 ||
                  (scale->Shape().NumDimensions() == 1 && scale->Shape().GetDims().size() == 1),
              "scale must be a scalar");
  ORT_ENFORCE(zeropoint->Shape().NumDimensions() == 0 ||
                  (zeropoint->Shape().NumDimensions() == 1 && zeropoint->Shape().GetDims().size() == 1),
              "zeropoint must be a scalar");
}
}  // namespace onnxruntime
