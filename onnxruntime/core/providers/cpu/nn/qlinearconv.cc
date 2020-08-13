// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/util/gemmlowp_common.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {}

  Status Compute(OpKernelContext* context) const override;

  ConvAttributes conv_attrs_;
};

ONNX_OPERATOR_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);

Status QLinearConv::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(3);

  // validate offsets
  auto X_zero_point = context->Input<Tensor>(2);
  auto W_zero_point = context->Input<Tensor>(5);
  auto Y_zero_point = context->Input<Tensor>(7);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_zero_point),
              "QLinearConv : input zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_zero_point),
              "QLinearConv : filter zero point must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_zero_point),
              "QLinearConv : result zero point must be a scalar or 1D tensor of size 1");

  auto X_zero_point_value = *(X_zero_point->template Data<uint8_t>());
  auto W_zero_point_value = *(W_zero_point->template Data<uint8_t>());
  auto Y_zero_point_value = *(Y_zero_point->template Data<uint8_t>());

  // validate scale
  auto X_scale = context->Input<Tensor>(1);
  auto W_scale = context->Input<Tensor>(4);
  auto Y_scale = context->Input<Tensor>(6);
  ORT_ENFORCE(IsScalarOr1ElementVector(X_scale),
              "QLinearConv : input scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(W_scale),
              "QLinearConv : filter scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(IsScalarOr1ElementVector(Y_scale),
              "QLinearConv : result scale must be a scalar or 1D tensor of size 1");

  auto X_scale_value = *(X_scale->template Data<float>());
  auto W_scale_value = *(W_scale->template Data<float>());
  auto Y_scale_value = *(Y_scale->template Data<float>());

  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* B = nullptr;
  if (num_inputs == 9) {
    B = context->Input<Tensor>(8);
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

  std::vector<int64_t> Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t B_offset = M / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  const size_t kernel_rank = kernel_shape.size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr col_buffer;
  std::vector<int64_t> col_buffer_shape;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(uint8_t)) * col_buffer_size);
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(alloc));

    if (kernel_rank != 2) {
      const auto& output_dims = output_shape.GetDims();
      col_buffer_shape.reserve(1 + output_dims.size());
      col_buffer_shape.push_back(kernel_dim);
      col_buffer_shape.insert(col_buffer_shape.end(), output_dims.begin(), output_dims.end());
    }
  }

  auto* col_buffer_data = static_cast<uint8_t*>(col_buffer.get());

  const float real_multiplier = (X_scale_value * W_scale_value) / Y_scale_value;

#ifdef MLAS_SUPPORTS_GEMM_U8X8
  // Use an intermediate int32_t buffer for the GEMM computation before
  // requantizing to the output type.
  auto gemm_output_data = alloc->Alloc(SafeInt<size_t>(sizeof(int32_t)) * Y_offset);
  BufferUniquePtr gemm_output_buffer(gemm_output_data, BufferDeleter(alloc));
  auto* gemm_output = static_cast<int32_t*>(gemm_output_buffer.get());
#else
  // Compute the fixed point multiplier and shift for requantizing with GEMMLOWP.
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier(real_multiplier, &integer_multiplier, &right_shift);
#endif

  const auto* Xdata = X->template Data<uint8_t>();
  const auto* Wdata = W->template Data<uint8_t>();
  const auto* Bdata = B != nullptr ? B->template Data<int32_t>() : nullptr;
  auto* Ydata = Y->template MutableData<uint8_t>();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (col_buffer_data != nullptr) {
        if (kernel_rank == 2) {
          math::Im2col<uint8_t, StorageOrder::NCHW>()(
              Xdata,
              C / conv_attrs_.group,
              input_shape[0],
              input_shape[1],
              kernel_shape[0],
              kernel_shape[1],
              dilations[0],
              dilations[1],
              pads[0],
              pads[1],
              pads[2],
              pads[3],
              strides[0],
              strides[1],
              col_buffer_data,
              X_zero_point_value);
        } else {
          math::Im2colNd<uint8_t, StorageOrder::NCHW>()(
              Xdata,
              X->Shape().GetDims().data() + 1,
              col_buffer_shape.data(),
              C * input_image_size,
              col_buffer_size,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_rank),
              col_buffer_data,
              false,
              X_zero_point_value);
        }
      }

#ifdef MLAS_SUPPORTS_GEMM_U8X8
      QGemm(static_cast<int>(M / conv_attrs_.group),
            static_cast<int>(output_image_size),
            static_cast<int>(kernel_dim),
            Wdata + group_id * W_offset,
            static_cast<int>(kernel_dim),
            W_zero_point_value,
            col_buffer_data == nullptr ? Xdata : col_buffer_data,
            static_cast<int>(output_image_size),
            X_zero_point_value,
            false,
            gemm_output,
            static_cast<int>(output_image_size),
            context->GetOperatorThreadPool());

      MlasRequantizeOutput(gemm_output,
                           Ydata,
                           Bdata != nullptr ? Bdata + group_id * B_offset : nullptr,
                           static_cast<size_t>(M / conv_attrs_.group),
                           static_cast<size_t>(output_image_size),
                           real_multiplier,
                           Y_zero_point_value);
#else
      GemmlowpMultiplyu8u8_u8(Wdata + group_id * W_offset,
                              col_buffer_data == nullptr ? Xdata : col_buffer_data,
                              Ydata,
                              W_zero_point_value,
                              X_zero_point_value,
                              Y_zero_point_value,
                              static_cast<int>(M / conv_attrs_.group),
                              static_cast<int>(output_image_size),
                              static_cast<int>(kernel_dim),
                              integer_multiplier,
                              right_shift,
                              Bdata != nullptr ? Bdata + group_id * B_offset : nullptr);
#endif

      Xdata += X_offset;
      Ydata += Y_offset;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
