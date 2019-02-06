/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#ifdef _MSC_VER
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include "core/providers/cpu/nn/qlinearconv_transpose.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/gemmlowp_common_wrapper.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    QLinearConvTranspose,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConvTranspose<uint8_t, uint8_t, uint8_t, int32_t>);

template <bool TransposeA, bool TransposeB, bool TransposeC>
Status GemmlowpMultiply(const uint8_t* lhs_data, const uint8_t* rhs_data,
                        int32_t* result_data, const int lhs_offset, const int rhs_offset,
                        int m, int n, int k, const int lda, const int ldb, const int ldc) {
  const std::tuple<> empty_pipeline = {};

  const auto lhsOrder = TransposeA ? gemmlowp::MapOrder::ColMajor : gemmlowp::MapOrder::RowMajor;
  const auto rhsOrder = TransposeB ? gemmlowp::MapOrder::ColMajor : gemmlowp::MapOrder::RowMajor;
  const auto resultOrder = TransposeC ? gemmlowp::MapOrder::ColMajor : gemmlowp::MapOrder::RowMajor;
  gemmlowp::MatrixMap<const std::uint8_t, lhsOrder> lhs(lhs_data, m, k, lda);
  gemmlowp::MatrixMap<const std::uint8_t, rhsOrder> rhs(rhs_data, k, n, ldb);
  gemmlowp::MatrixMap<std::int32_t, resultOrder> result(result_data, m, n, ldc);

  gemmlowp::GemmContext gemm_context;
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs, &result, -lhs_offset, -rhs_offset, empty_pipeline);

  return Status::OK();
}

void QuantizeMultiplier1(float fp_multiplier, std::int32_t* integer_multiplier, int* right_shift) {
  uint32_t* fp_as_bits = reinterpret_cast<uint32_t*>(&fp_multiplier);
  auto current_exponent = (*fp_as_bits >> 23);
  // bring multiplier in [.5,1) range and calculate the shift
  auto bumped_multiplier_as_bits =
      (*fp_as_bits & UINT32_C(0x007fffff)) | UINT32_C(0x3f000000);
  float* bumped_multiplier =
      reinterpret_cast<float*>(&bumped_multiplier_as_bits);
  auto shift = 126 - current_exponent;
  // convert to fixed point number
  std::int64_t int_multiplier =
      static_cast<std::int64_t>(std::round(*bumped_multiplier * (1ll << 31)));

  *integer_multiplier = static_cast<int32_t>(int_multiplier);
  *right_shift = shift;
}

void ScaleAndZeropointPairValidationHelper1(const Tensor* scale, const Tensor* zeropoint) {
  ORT_ENFORCE(scale->Shape().NumDimensions() == 0 ||
                  (scale->Shape().NumDimensions() == 1 && scale->Shape().GetDims().size() == 1),
              "scale must be a scalar");
  ORT_ENFORCE(zeropoint->Shape().NumDimensions() == 0 ||
                  (zeropoint->Shape().NumDimensions() == 1 && zeropoint->Shape().GetDims().size() == 1),
              "zeropoint must be a scalar");
}

int32_t FixedPointMultiply(int32_t a, int32_t b) {
  return a * b;
}

template <>
Status QLinearConvTranspose<uint8_t, uint8_t, uint8_t, int32_t>::Compute(OpKernelContext* ctx) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, num_inputs == 9, p, 0, 3, 8));

  // get scale and zero_point
  // validate scale and zero points
  auto x_scale = ctx->Input<Tensor>(1);
  auto x_zero_point = ctx->Input<Tensor>(2);
  ScaleAndZeropointPairValidationHelper1(x_scale, x_zero_point);
  auto f_scale = ctx->Input<Tensor>(4);
  auto f_zero_point = ctx->Input<Tensor>(5);
  ScaleAndZeropointPairValidationHelper1(f_scale, f_zero_point);
  auto y_scale = ctx->Input<Tensor>(6);
  auto y_zero_point = ctx->Input<Tensor>(7);
  ScaleAndZeropointPairValidationHelper1(y_scale, y_zero_point);

  auto x_scale_data = *(x_scale->template Data<float>());
  auto f_scale_data = *(f_scale->template Data<float>());
  auto y_scale_data = *(y_scale->template Data<float>());

  const float real_multiplier = (x_scale_data * f_scale_data) / y_scale_data;
  int32_t integer_multiplier;
  int right_shift;
  QuantizeMultiplier1(real_multiplier, &integer_multiplier, &right_shift);

  const int64_t input_image_size = p.H * p.W;
  const int64_t X_offset = p.num_input_channels / group_ * input_image_size;
  const int64_t Y_offset = p.Y->Shape().Size() / p.Y->Shape()[0] / group_;
  const int64_t W_offset = p.F->Shape().Size() / group_;
  const int64_t kernel_dim = p.num_output_channels / group_ * p.kernel_shape[0] * p.kernel_shape[1];
  const int64_t output_image_size = p.Y->Shape()[2] * p.Y->Shape()[3];

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  auto col_data = alloc->Alloc(sizeof(int32_t) * kernel_dim * p.H * p.W);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(alloc));
  int32_t* col_buffer_data = static_cast<int32_t*>(col_buffer.get());

  // temp int32_t accumulator to hold col2im data per channel
  auto acc32_data = alloc->Alloc(sizeof(int32_t) * output_image_size);
  BufferUniquePtr acc32_buffer(acc32_data, BufferDeleter(alloc));
  int32_t* out_acc32 = static_cast<int32_t*>(acc32_buffer.get());

  const auto* Xdata = p.X->template Data<uint8_t>();
  const auto* filter_data = p.F->template Data<uint8_t>();
  auto* Ydata = p.Y->template MutableData<uint8_t>();

  const int lda = kernel_dim;
  const int ldb = input_image_size;
  const int ldc = input_image_size;

  const auto result_offset = static_cast<int32_t>(*(y_zero_point->template Data<uint8_t>()));

  // multiply the input
  auto quantizeDownToUInt8_func = [&integer_multiplier, &right_shift, &result_offset](const int32_t input_data) { 
      auto mulhigh_val = gemmlowp::SaturatingRoundingDoublingHighMul(input_data, integer_multiplier);
      auto output = gemmlowp::Add(gemmlowp::RoundingDivideByPOT(mulhigh_val, right_shift), 
          result_offset);
      return static_cast<uint8_t>(output);
  };

  for (auto image_id = 0; image_id < p.N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
      // Weight term
      GemmlowpMultiply<true, false, false>(
          filter_data + group_id * W_offset,
          Xdata + group_id * X_offset,
          col_buffer_data,
          *x_zero_point->template Data<uint8_t>(),
          *f_zero_point->template Data<uint8_t>(),
          kernel_dim, input_image_size,
          p.num_input_channels / group_,
          lda,
          ldb,
          ldc);

      // Col2im
      // int32_t Col2im + int32_t per channel bias => quantize and cast to uint8_t
      math::Col2im<int32_t, uint8_t, CPUMathUtil, StorageOrder::NCHW>(
          col_buffer_data,
          num_inputs == 9 ? p.B->template Data<int32_t>() : nullptr,
          p.num_output_channels / group_,
          p.Y->Shape()[2],
          p.Y->Shape()[3],
          p.kernel_shape[0],
          p.kernel_shape[1],
          1,
          1,
          p.pads[0],
          p.pads[1],
          p.pads[2],
          p.pads[3],
          p.strides[0],
          p.strides[1],
          Ydata + group_id * Y_offset,
          out_acc32,
          &CPUMathUtil::Instance(),
          quantizeDownToUInt8_func);
    }

    Xdata += X_offset * group_;
    Ydata += Y_offset * group_;
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
