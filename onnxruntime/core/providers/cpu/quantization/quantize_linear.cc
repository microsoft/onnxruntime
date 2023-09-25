// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/safeint.h>
#include "core/framework/element_type_lists.h"
#include "core/framework/float8.h"
#include "core/framework/float16.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"

namespace onnxruntime {

template <typename T>
class DequantizeLinear final : public OpKernel {
 public:
  explicit DequantizeLinear(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

template <typename T>
class QuantizeLinear final : public OpKernel {
 public:
  explicit QuantizeLinear(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("axis", &axis_).IsOK()) {
      axis_ = 1;
    }
    if (!info.GetAttr<int64_t>("saturate", &saturate_).IsOK()) {
      saturate_ = 1;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
};

static void PrepareForQDQ(const TensorShape& input_shape,
                          const Tensor& scale,
                          const Tensor* zero_point_ptr,
                          int64_t axis,
                          int64_t& block_count,
                          int64_t& broadcast_dim,
                          int64_t& block_size) {
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear
    block_count = 1;
    broadcast_dim = 1;
    block_size = static_cast<size_t>(input_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel QuantizeLinear/DequantizeLinear
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());
    block_count = input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis_no_neg));
    broadcast_dim = input_shape[onnxruntime::narrow<size_t>(axis_no_neg)];
    block_size = input_shape.SizeFromDimension(SafeInt<size_t>(axis_no_neg) + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                "scale must be 1D tensor with size ",
                broadcast_dim);
    ORT_ENFORCE(zero_point_ptr == nullptr || (zero_point_ptr->Shape().NumDimensions() == 1 && zero_point_ptr->Shape()[0] == broadcast_dim),
                "x_zero_point must be null or 1D tensor with size ",
                broadcast_dim);
  }
}

#define REGISTER_DEQUANTIZELINEAR(T)                                         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                            \
      DequantizeLinear,                                                      \
      19,                                                                    \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED(T)                    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DequantizeLinear,                                           \
      13,                                                         \
      18,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);                                       \
                                                                  \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DequantizeLinear,                                           \
      10,                                                         \
      12,                                                         \
      T,                                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DequantizeLinear<T>);

REGISTER_DEQUANTIZELINEAR(int8_t)
REGISTER_DEQUANTIZELINEAR(uint8_t)
REGISTER_DEQUANTIZELINEAR(int32_t)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FN)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_DEQUANTIZELINEAR(Float8E5M2)
REGISTER_DEQUANTIZELINEAR(Float8E5M2FNUZ)
#endif
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t)

#if !defined(DISABLE_CONTRIB_OPS)
namespace contrib {

// Register alternate MS domain versions of the DequantizeLinear kernel.
// The MS domain versions additionally support 16-bit integer quantization types.
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<int8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    uint16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint16_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<uint16_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    int16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int16_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<int16_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    int32_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<int32_t>);

}  // namespace contrib
#endif  // !defined(DISABLE_CONTRIB_OPS)

template <typename T, typename OutT>
struct DequantizeLinearApply {
  void op(int64_t N, int64_t broadcast_dim, int64_t block_size, const T* input, const OutT* scale, OutT* output, const T* zero_point) {
    for (size_t n = 0; n < static_cast<size_t>(N); n++) {
      for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
        auto zp = zero_point ? static_cast<int32_t>(zero_point[bd]) : 0;
        auto sc = static_cast<float>(scale[bd]);
        for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++) {
          *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
        }
      }
    }
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

#define DEQUANTIZE_LINEAR_APPLY_FLOAT8(T)                                                                                      \
  template <typename OutT>                                                                                                     \
  struct DequantizeLinearApply<T, OutT> {                                                                                      \
    void op(int64_t N, int64_t broadcast_dim, int64_t block_size, const T* input, const OutT* scale, OutT* output, const T*) { \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                                    \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                                                   \
          auto sc = scale[bd];                                                                                                 \
          for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++, input++) {                                           \
            *output++ = static_cast<OutT>(input->ToFloat() * sc);                                                              \
          }                                                                                                                    \
        }                                                                                                                      \
      }                                                                                                                        \
    }                                                                                                                          \
  };

DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E4M3FN)
DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E4M3FNUZ)
DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E5M2)
DEQUANTIZE_LINEAR_APPLY_FLOAT8(Float8E5M2FNUZ)

#endif

// formula is Y = (X - ZeroPoint) * Scale
template <typename T>
Status DequantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& x_scale = *ctx->Input<Tensor>(1);
  auto* x_zero_point = ctx->Input<Tensor>(2);

  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;

  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, N, broadcast_dim, block_size);

  const T* zero_point = x_zero_point ? x_zero_point->Data<T>() : nullptr;

#if !defined(DISABLE_FLOAT8_TYPES)
  if constexpr (boost::mp11::mp_contains<boost::mp11::mp_append<element_type_lists::AllFloat8,
                                                                TypeList<int32_t>>,
                                         T>::value) {
    ORT_ENFORCE(zero_point == nullptr ||
                    std::all_of(zero_point,
                                zero_point + x_zero_point->Shape().Size(),
                                [](T zp) { return zp == T{0}; }),
                "DequantizeLinear with type int32 or float8 should have no zero point or all zero points should be 0");
  }
#endif

  const auto to = x_scale.GetElementType();
  const T* input = x.Data<T>();

  if (to == ONNX_NAMESPACE::TensorProto::FLOAT) {
    const float* scale = x_scale.Data<float>();
    float* output = y.MutableData<float>();
    DequantizeLinearApply<T, float>().op(N, broadcast_dim, block_size, input, scale, output, zero_point);
  } else if (to == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    const MLFloat16* scale = x_scale.Data<MLFloat16>();
    MLFloat16* output = y.MutableData<MLFloat16>();
    DequantizeLinearApply<T, MLFloat16>().op(N, broadcast_dim, block_size, input, scale, output, zero_point);
  } else if (to == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
    ORT_THROW("DequantizeLinear into BFLOAT16 is not implemented yet.");
  } else {
    ORT_THROW("DequantizeLinear only outputs FLOAT16, FLOAT or BFLOAT16.");
  }

  return Status::OK();
}

#define REGISTER_QUANTIZELINEAR(T)                                          \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                           \
      QuantizeLinear,                                                       \
      19,                                                                   \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED(T)                          \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      QuantizeLinear,                                                 \
      13,                                                             \
      18,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);                                             \
                                                                      \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                           \
      QuantizeLinear,                                                 \
      10,                                                             \
      12,                                                             \
      T,                                                              \
      KernelDefBuilder()                                              \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),    \
      QuantizeLinear<T>);

REGISTER_QUANTIZELINEAR(int8_t)
REGISTER_QUANTIZELINEAR(uint8_t)

#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR(Float8E4M3FN)
REGISTER_QUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_QUANTIZELINEAR(Float8E5M2)
REGISTER_QUANTIZELINEAR(Float8E5M2FNUZ)
#endif

REGISTER_QUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t)

#if !defined(DISABLE_CONTRIB_OPS)
namespace contrib {

// Register alternate MS domain versions of the QuantizeLinear kernel.
// The MS domain versions additionally support 16-bit integer quantization types.
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    uint8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    QuantizeLinear<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    int8_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>()),
    QuantizeLinear<int8_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    uint16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint16_t>()),
    QuantizeLinear<uint16_t>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    int16_t,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int16_t>()),
    QuantizeLinear<int16_t>);
}  // namespace contrib
#endif  // !defined(DISABLE_CONTRIB_OPS)

template <typename InputType, typename OutputType>
void ParQuantizeLinear(const InputType* Input,
                       OutputType* Output,
                       size_t N,
                       InputType Scale,
                       size_t bd,
                       const OutputType* ZeroPoint,
                       bool saturate,
                       concurrency::ThreadPool* thread_pool) {
#if !defined(DISABLE_FLOAT8_TYPES)
  if constexpr (!boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputType>::value) {
#endif
    ORT_UNUSED_PARAMETER(saturate);
    ParQuantizeLinearStd(Input, Output, N, Scale, ZeroPoint != nullptr ? ZeroPoint[bd] : (OutputType)0, thread_pool);
#if !defined(DISABLE_FLOAT8_TYPES)
  } else {
    ParQuantizeLinearSat(Input, Output, N, Scale, ZeroPoint != nullptr ? ZeroPoint[bd] : OutputType(static_cast<InputType>(static_cast<float>(0)), true), saturate, thread_pool);
  }
#endif
}

template <typename T, typename InT>
void ComputeLoop(OpKernelContext* ctx, const InT* input, const InT* scale, const T* zero_point, T* output, int64_t N, int64_t broadcast_dim, int64_t block_size, bool saturate) {
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      ParQuantizeLinear(input, output, static_cast<size_t>(block_size), scale[bd], bd, zero_point, saturate, ctx->GetOperatorThreadPool());
      input += block_size;
      output += block_size;
    }
  }
}

// formula is Y = X / Scale + ZeroPoint
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, N, broadcast_dim, block_size);

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
  T* output = y.MutableData<T>();

  if (x.IsDataType<float>()) {
    ComputeLoop<T, float>(ctx, x.Data<float>(), y_scale.Data<float>(), zero_point, output, N, broadcast_dim, block_size, saturate_);
  } else if (x.IsDataType<MLFloat16>()) {
    ComputeLoop<T, MLFloat16>(ctx, x.Data<MLFloat16>(), y_scale.Data<MLFloat16>(), zero_point, output, N, broadcast_dim, block_size, saturate_);
  } else {
    ORT_THROW("Unsupported input type.");
  }

  return Status::OK();
}
}  // namespace onnxruntime
