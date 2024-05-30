// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/safeint.h>
#include "core/framework/element_type_lists.h"
#include "core/framework/float8.h"
#include "core/framework/float16.h"
#include "core/framework/int4.h"
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

    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 0;
    }

    // TODO(adrianlizarraga): Support the block_size attribute added in opset 21.
    if (block_size_ != 0) {
      ORT_THROW("DequantizeLinear does not yet support the 'block_size' attribute.");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t block_size_;
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

    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 0;
    }

    // TODO(adrianlizarraga): Support the block_size attribute added in opset 21.
    if (block_size_ != 0) {
      ORT_THROW("QuantizeLinear does not yet support the 'block_size' attribute.");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
  int64_t block_size_;
};

static void PrepareForQDQ(const TensorShape& input_shape,
                          const Tensor& scale,
                          const Tensor* zero_point_ptr,
                          int64_t axis,
                          int64_t& quant_block_count,  // A "quant block" is a block of elems with the same scale/zp
                          int64_t& axis_dim_val,
                          int64_t& quant_block_size) {
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear
    quant_block_count = 1;
    axis_dim_val = 1;
    quant_block_size = static_cast<size_t>(input_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
  } else {  // per-channel QuantizeLinear/DequantizeLinear
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());
    quant_block_count = input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis_no_neg));
    axis_dim_val = input_shape[onnxruntime::narrow<size_t>(axis_no_neg)];
    quant_block_size = input_shape.SizeFromDimension(SafeInt<size_t>(axis_no_neg) + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == axis_dim_val,
                "scale must be 1D tensor with size ",
                axis_dim_val);
    ORT_ENFORCE(zero_point_ptr == nullptr ||
                    (zero_point_ptr->Shape().NumDimensions() == 1 && zero_point_ptr->Shape()[0] == axis_dim_val),
                "x_zero_point must be null or 1D tensor with size ",
                axis_dim_val);
  }
}

#define REGISTER_DEQUANTIZELINEAR(T)                                         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                            \
      DequantizeLinear,                                                      \
      21,                                                                    \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED(T)                               \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                  \
      DequantizeLinear,                                                      \
      19,                                                                    \
      20,                                                                    \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(T)             \
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

// Opset 21 added 16-bit and 4-bit int to DQ.
// TODO(adrianlizarraga): Also support 4-bit int types and 'block' quantization.
REGISTER_DEQUANTIZELINEAR(int8_t)
REGISTER_DEQUANTIZELINEAR(uint8_t)
REGISTER_DEQUANTIZELINEAR(int16_t)
REGISTER_DEQUANTIZELINEAR(uint16_t)
REGISTER_DEQUANTIZELINEAR(int32_t)
REGISTER_DEQUANTIZELINEAR(Int4x2)
REGISTER_DEQUANTIZELINEAR(UInt4x2)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FN)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_DEQUANTIZELINEAR(Float8E5M2)
REGISTER_DEQUANTIZELINEAR(Float8E5M2FNUZ)
#endif

// Opset 19 added 8-bit float inputs and 16-bit float outputs to DQ.
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FN)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ)
#endif

// Before opset 19, DQ only supported int8, uint8 and int32.
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(int8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(uint8_t)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_19(int32_t)

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

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    Int4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<Int4x2>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<Int4x2>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    DequantizeLinear,
    1,
    UInt4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<UInt4x2>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    DequantizeLinear<UInt4x2>);

}  // namespace contrib
#endif  // !defined(DISABLE_CONTRIB_OPS)

template <typename T, typename OutT>
struct DequantizeLinearApply {
  void op(int64_t N, int64_t axis_dim_val, int64_t quant_block_size, const T* input, const OutT* scale, OutT* output,
          const T* zero_point) {
    for (size_t n = 0; n < static_cast<size_t>(N); n++) {
      for (size_t bd = 0; bd < static_cast<size_t>(axis_dim_val); bd++) {
        auto zp = zero_point ? static_cast<int32_t>(zero_point[bd]) : 0;
        auto sc = static_cast<float>(scale[bd]);
        for (size_t bs = 0; bs < static_cast<size_t>(quant_block_size); bs++) {
          *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
        }
      }
    }
  }
};

#define DEQUANTIZE_LINEAR_APPLY_INT4(T)                                                                   \
  template <typename OutT>                                                                                \
  struct DequantizeLinearApply<T, OutT> {                                                                 \
    void op(int64_t N, int64_t axis_dim_val, int64_t quant_block_size, const T* input, const OutT* scale, \
            OutT* output, const T* zero_point) {                                                          \
      size_t input_index = 0;                                                                             \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                               \
        for (size_t bd = 0; bd < static_cast<size_t>(axis_dim_val); bd++) {                               \
          size_t bd_i = bd >> 1;  /*bd / 2*/                                                              \
          size_t bd_j = bd & 0x1; /*bd % 2*/                                                              \
          auto zp = zero_point ? static_cast<int32_t>(zero_point[bd_i].GetElem(bd_j)) : 0;                \
          auto sc = static_cast<float>(scale[bd]);                                                        \
          for (size_t bs = 0; bs < static_cast<size_t>(quant_block_size); bs++) {                         \
            size_t input_i = input_index >> 1;                                                            \
            size_t input_j = input_index & 0x1;                                                           \
            int32_t val = static_cast<int32_t>(input[input_i].GetElem(input_j));                          \
            *output++ = static_cast<OutT>(static_cast<float>(val - zp) * sc);                             \
            input_index += 1;                                                                             \
          }                                                                                               \
        }                                                                                                 \
      }                                                                                                   \
      assert(input_index == static_cast<size_t>(N * axis_dim_val * quant_block_size));                    \
    }                                                                                                     \
  };

DEQUANTIZE_LINEAR_APPLY_INT4(Int4x2);
DEQUANTIZE_LINEAR_APPLY_INT4(UInt4x2);

#if !defined(DISABLE_FLOAT8_TYPES)

#define DEQUANTIZE_LINEAR_APPLY_FLOAT8(T)                                                                 \
  template <typename OutT>                                                                                \
  struct DequantizeLinearApply<T, OutT> {                                                                 \
    void op(int64_t N, int64_t axis_dim_val, int64_t quant_block_size, const T* input, const OutT* scale, \
            OutT* output, const T*) {                                                                     \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                               \
        for (size_t bd = 0; bd < static_cast<size_t>(axis_dim_val); bd++) {                               \
          auto sc = scale[bd];                                                                            \
          for (size_t bs = 0; bs < static_cast<size_t>(quant_block_size); bs++, input++) {                \
            *output++ = static_cast<OutT>(input->ToFloat() * sc);                                         \
          }                                                                                               \
        }                                                                                                 \
      }                                                                                                   \
    }                                                                                                     \
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
  int64_t axis_dim_val;
  int64_t quant_block_size;

  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, N, axis_dim_val, quant_block_size);

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
    DequantizeLinearApply<T, float>().op(N, axis_dim_val, quant_block_size, input, scale, output, zero_point);
  } else if (to == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    const MLFloat16* scale = x_scale.Data<MLFloat16>();
    MLFloat16* output = y.MutableData<MLFloat16>();
    DequantizeLinearApply<T, MLFloat16>().op(N, axis_dim_val, quant_block_size, input, scale, output, zero_point);
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
      21,                                                                   \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED(T)                                \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                 \
      QuantizeLinear,                                                       \
      19,                                                                   \
      20,                                                                   \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED_PRE_19(T)                   \
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

// Opset 21 added 16-bit and 4-bit int support to Q ops.
// TODO(adrianlizarraga): Support int4 and block quantization.
REGISTER_QUANTIZELINEAR(int8_t)
REGISTER_QUANTIZELINEAR(uint8_t)
REGISTER_QUANTIZELINEAR(int16_t)
REGISTER_QUANTIZELINEAR(uint16_t)
REGISTER_QUANTIZELINEAR(Int4x2)
REGISTER_QUANTIZELINEAR(UInt4x2)

#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR(Float8E4M3FN)
REGISTER_QUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_QUANTIZELINEAR(Float8E5M2)
REGISTER_QUANTIZELINEAR(Float8E5M2FNUZ)
#endif

// Opset 19 added 8-bit floats to Q ops.
REGISTER_QUANTIZELINEAR_VERSIONED(int8_t)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t)

#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FN)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ)
#endif

// Before opset 19, Q only supported int8 and uint8.
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_19(int8_t)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_19(uint8_t)

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

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    Int4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<Int4x2>()),
    QuantizeLinear<Int4x2>);

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    QuantizeLinear,
    1,
    UInt4x2,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<UInt4x2>()),
    QuantizeLinear<UInt4x2>);
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
    ParQuantizeLinearSat(Input, Output, N, Scale,
                         ZeroPoint != nullptr ? ZeroPoint[bd]
                                              : OutputType(static_cast<InputType>(static_cast<float>(0)), true),
                         saturate, thread_pool);
  }
#endif
}

template <typename T, typename InT>
void ComputeLoop(OpKernelContext* ctx, const InT* input, const InT* scale, const T* zero_point, T* output, int64_t N,
                 int64_t axis_dim_val, int64_t quant_block_size, bool saturate) {
  for (size_t n = 0; n < static_cast<size_t>(N); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(axis_dim_val); bd++) {
      ParQuantizeLinear(input, output, static_cast<size_t>(quant_block_size), scale[bd], bd, zero_point, saturate,
                        ctx->GetOperatorThreadPool());
      input += quant_block_size;
      output += quant_block_size;
    }
  }
}

// Quantizes float32 to INT4 (in-place) using MLAS kernel.
#define DEFINE_COMPUTE_LOOP_FP32_TO_INT4(INT4_TYPE, QUANT_FUNC)                                                   \
  template <>                                                                                                     \
  void ComputeLoop(OpKernelContext* ctx, const float* input, const float* scale, const INT4_TYPE* zero_point,     \
                   INT4_TYPE* output, int64_t N, int64_t axis_dim_val, int64_t quant_block_size, bool saturate) { \
    ORT_UNUSED_PARAMETER(saturate);                                                                               \
    size_t output_index = 0;                                                                                      \
    for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                         \
      for (size_t bd = 0; bd < static_cast<size_t>(axis_dim_val); bd++) {                                         \
        size_t bd_i = bd >> 1;  /*bd / 2*/                                                                        \
        size_t bd_j = bd & 0x1; /*bd % 2*/                                                                        \
        INT4_TYPE::UnpackedType zp = zero_point ? zero_point[bd_i].GetElem(bd_j) : 0;                             \
        QUANT_FUNC(input, output, output_index, output_index + static_cast<size_t>(quant_block_size),             \
                   scale[bd], INT4_TYPE(zp, 0), ctx->GetOperatorThreadPool());                                    \
        input += quant_block_size;                                                                                \
        output_index += static_cast<size_t>(quant_block_size);                                                    \
      }                                                                                                           \
    }                                                                                                             \
    assert(output_index == static_cast<size_t>(N * axis_dim_val * quant_block_size));                             \
  }

DEFINE_COMPUTE_LOOP_FP32_TO_INT4(Int4x2, ParQuantizeLinearStdS4)
DEFINE_COMPUTE_LOOP_FP32_TO_INT4(UInt4x2, ParQuantizeLinearStdU4)

// Defines functions to quantize MLFloat16 to INT4.
// This is not an efficient implementation: we allocate a buffer, quantize to INT8, and then copy/clamp/pack
// into output INT4 buffer.
#define DEFINE_COMPUTE_LOOP_FP16_TO_INT4(INT4_TYPE)                                                             \
  template <>                                                                                                   \
  void ComputeLoop<INT4_TYPE, MLFloat16>(OpKernelContext * ctx, const MLFloat16* input, const MLFloat16* scale, \
                                         const INT4_TYPE* zero_point, INT4_TYPE* output, int64_t N,             \
                                         int64_t axis_dim_val, int64_t quant_block_size, bool saturate) {       \
    ORT_UNUSED_PARAMETER(saturate);                                                                             \
                                                                                                                \
    size_t total_size = static_cast<size_t>(N * axis_dim_val * quant_block_size);                               \
    auto tmp_buf = std::make_unique<INT4_TYPE::UnpackedType[]>(total_size);                                     \
    size_t tmp_buf_index = 0;                                                                                   \
                                                                                                                \
    for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                                       \
      for (size_t bd = 0; bd < static_cast<size_t>(axis_dim_val); bd++) {                                       \
        size_t bd_i = bd >> 1;  /*bd / 2*/                                                                      \
        size_t bd_j = bd & 0x1; /*bd % 2*/                                                                      \
        INT4_TYPE::UnpackedType zp = zero_point ? zero_point[bd_i].GetElem(bd_j) : 0;                           \
        ParQuantizeLinearStd<INT4_TYPE::UnpackedType>(input, tmp_buf.get() + tmp_buf_index,                     \
                                                      static_cast<size_t>(quant_block_size), scale[bd],         \
                                                      zp, ctx->GetOperatorThreadPool());                        \
        input += quant_block_size;                                                                              \
        tmp_buf_index += static_cast<size_t>(quant_block_size);                                                 \
      }                                                                                                         \
    }                                                                                                           \
                                                                                                                \
    for (size_t i = 0; i < total_size; i++) {                                                                   \
      tmp_buf[i] = std::min<INT4_TYPE::UnpackedType>(INT4_TYPE::max_val,                                        \
                                                     std::max<INT4_TYPE::UnpackedType>(INT4_TYPE::min_val,      \
                                                                                       tmp_buf[i]));            \
    }                                                                                                           \
                                                                                                                \
    size_t num_int4_pairs = (total_size + 1) / 2;                                                               \
    auto dst = gsl::make_span<INT4_TYPE>(output, num_int4_pairs);                                               \
    auto src = gsl::make_span<const INT4_TYPE::UnpackedType>(tmp_buf.get(), total_size);                        \
    INT4_TYPE::Pack(dst, src);                                                                                  \
  }

DEFINE_COMPUTE_LOOP_FP16_TO_INT4(Int4x2)
DEFINE_COMPUTE_LOOP_FP16_TO_INT4(UInt4x2)

// formula is Y = X / Scale + ZeroPoint
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t N;
  int64_t axis_dim_val;
  int64_t quant_block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, N, axis_dim_val, quant_block_size);

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
  T* output = y.MutableData<T>();

  if (x.IsDataType<float>()) {
    ComputeLoop<T, float>(ctx, x.Data<float>(), y_scale.Data<float>(), zero_point, output, N, axis_dim_val,
                          quant_block_size, saturate_);
  } else if (x.IsDataType<MLFloat16>()) {
    ComputeLoop<T, MLFloat16>(ctx, x.Data<MLFloat16>(), y_scale.Data<MLFloat16>(), zero_point, output, N,
                              axis_dim_val, quant_block_size, saturate_);
  } else {
    ORT_THROW("Unsupported input type.");
  }

  return Status::OK();
}
}  // namespace onnxruntime
