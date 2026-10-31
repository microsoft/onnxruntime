// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <type_traits>
#include <utility>
#include <core/common/safeint.h>
#include "core/framework/element_type_lists.h"
#include "core/common/float8.h"
#include "core/common/float16.h"
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

    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
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

    ORT_ENFORCE(block_size_ >= 0, "'block_size' must be non-negative.");
    precision_ = info.node().SinceVersion() < 23
                     ? static_cast<int64_t>(ONNX_NAMESPACE::TensorProto::FLOAT)
                     : info.GetAttrOrDefault<int64_t>("precision", 0);
    ORT_ENFORCE(precision_ == 0 || precision_ == ONNX_NAMESPACE::TensorProto::FLOAT ||
                    precision_ == ONNX_NAMESPACE::TensorProto::FLOAT16,
                "CPU QuantizeLinear only supports precision 0 (use scale type), FLOAT, and FLOAT16.");
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  int64_t saturate_;
  int64_t block_size_;
  int64_t precision_;
};

static void PrepareForQDQ(const TensorShape& input_shape,
                          const Tensor& scale,
                          const Tensor* zero_point_ptr,
                          int64_t axis,
                          int64_t quant_block_size,
                          int64_t& process_block_count,
                          int64_t& broadcast_dim,
                          int64_t& process_block_size) {
  if (IsScalarOr1ElementVector(&scale)) {  // per-tensor QuantizeLinear/DequantizeLinear
    process_block_count = 1;
    broadcast_dim = 1;
    process_block_size = static_cast<size_t>(input_shape.Size());

    // enforce that zero point are scalars
    ORT_ENFORCE(zero_point_ptr == nullptr || IsScalarOr1ElementVector(zero_point_ptr),
                "x_zero_point must be null or a scalar or 1D tensor or size 1.");
    ORT_ENFORCE(quant_block_size == 0, "block_size must be 0 for per-tensor quantization.");
  } else {  // per-axis or blocked QuantizeLinear/DequantizeLinear
    const int64_t axis_no_neg = HandleNegativeAxis(axis, input_shape.NumDimensions());
    process_block_count = input_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis_no_neg));
    broadcast_dim = input_shape[onnxruntime::narrow<size_t>(axis_no_neg)];
    process_block_size = input_shape.SizeFromDimension(SafeInt<size_t>(axis_no_neg) + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    if (quant_block_size) {  // blocked quantization
      ORT_ENFORCE(scale.Shape().NumDimensions() == input_shape.NumDimensions(),
                  "x_scale and x must have the same rank for blocked quantization");
      ORT_ENFORCE(zero_point_ptr == nullptr || zero_point_ptr->Shape().NumDimensions() == input_shape.NumDimensions(),
                  "x_zero_point must be null or have the same rank as x for blocked quantization");

      for (size_t i = 0, ndim = input_shape.NumDimensions(); i < ndim; ++i) {
        if (i == SafeInt<size_t>(axis_no_neg)) {
          ORT_ENFORCE(scale.Shape()[i] == (input_shape[i] + quant_block_size - 1) / quant_block_size,
                      "x_scale must be ceil(Di/block_size) on the quantize axis i for blocked quantization");
        } else {
          ORT_ENFORCE(scale.Shape()[i] == input_shape[i],
                      "x_scale and x must have the same shape despite the quantize axis for blocked quantization");
        }

        if (zero_point_ptr) {
          ORT_ENFORCE(zero_point_ptr->Shape()[i] == scale.Shape()[i],
                      "x_zero_point and x_scale must have the same shape for blocked quantization");
        }
      }
    } else {  // per-axis quantization
      ORT_ENFORCE(scale.Shape().NumDimensions() == 1 && scale.Shape()[0] == broadcast_dim,
                  "For per axis quantization, scale must be 1D tensor with size ",
                  broadcast_dim);
      ORT_ENFORCE(zero_point_ptr == nullptr || (zero_point_ptr->Shape().NumDimensions() == 1 &&
                                                zero_point_ptr->Shape()[0] == broadcast_dim),
                  "For per axis quantization, x_zero_point must be null or 1D tensor with size ",
                  broadcast_dim);
    }
  }
}

#define REGISTER_DEQUANTIZELINEAR(T)                                         \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                            \
      DequantizeLinear,                                                      \
      25,                                                                    \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()})  \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED(T, start_version, end_version)   \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                  \
      DequantizeLinear,                                                      \
      start_version,                                                         \
      end_version,                                                           \
      T,                                                                     \
      KernelDefBuilder()                                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()})  \
          .TypeConstraint("T3", {DataTypeImpl::GetTensorType<float>(),       \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}), \
      DequantizeLinear<T>);

#define REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(T, start_version, end_version) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                       \
      DequantizeLinear,                                                           \
      start_version,                                                              \
      end_version,                                                                \
      T,                                                                          \
      KernelDefBuilder()                                                          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())                 \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),            \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}),      \
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

// Opset25
REGISTER_DEQUANTIZELINEAR(int8_t)
REGISTER_DEQUANTIZELINEAR(uint8_t)
REGISTER_DEQUANTIZELINEAR(int16_t)
REGISTER_DEQUANTIZELINEAR(uint16_t)
REGISTER_DEQUANTIZELINEAR(int32_t)
REGISTER_DEQUANTIZELINEAR(Int4x2)
REGISTER_DEQUANTIZELINEAR(UInt4x2)
REGISTER_DEQUANTIZELINEAR(Int2x4)
REGISTER_DEQUANTIZELINEAR(UInt2x4)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FN)
REGISTER_DEQUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_DEQUANTIZELINEAR(Float8E5M2)
REGISTER_DEQUANTIZELINEAR(Float8E5M2FNUZ)
#endif

// opset24
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int16_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint16_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Int4x2, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(UInt4x2, 24, 24)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FN, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2, 24, 24)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 24, 24)
#endif

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
REGISTER_DEQUANTIZELINEAR_VERSIONED(int8_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint8_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int16_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(uint16_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(int32_t, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Int4x2, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(UInt4x2, 23, 23)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FN, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2, 23, 23)
REGISTER_DEQUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 23, 23)
#endif

// Opset 21 added 16-bit and 4-bit int to DQ.
// TODO(adrianlizarraga): Also support 4-bit int types and 'block' quantization.
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int16_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(uint16_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int32_t, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Int4x2, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(UInt4x2, 21, 22)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 21, 22)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 21, 22)
#endif

// Opset 19 added 8-bit float inputs and 16-bit float outputs to DQ.
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(int32_t, 19, 20)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 19, 20)
REGISTER_DEQUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 19, 20)
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

template <typename T, typename ScaleT, typename OutT, bool is_sub_byte, int elements_per_byte = 0>
struct DequantizeLinearApply;

// The dimensions before quantize axis and after quantize axis can be flattened.
// After flattening, the tensor can be represented by a rank-3 tensor.
// If the quantization happens on the first or last axis, the flattened tensor is
// effectively rank-2.
// For per tensor quantization, the tensor is effectively rank-1.
template <typename T, typename ScaleT, typename OutT, int elements_per_byte>
struct DequantizeLinearApply<T, ScaleT, OutT, false, elements_per_byte> {
  /**
   * @brief Calculate per-tensor/layer or per-axis quantization of DequantizeLinear on the
   *        flattened tensors.
   * @param[in]    M                      size of dimensions before the quantize axis
   * @param[in]    K                      dimension on the quantize axis
   * @param[in]    N                      size of dimensions after the quantize axis
   * @param[in]    input                  1D array of flattened [D0, ..., Di, ..., Dn]
   * @param[in]    scale                  scalar for per-tensor/layer quantization and 1D array [Di]
   *                                      for per-axis quantization. i is the quantize axis.
   * @param[out]   output                 same shape as input
   * @param[in]    zero_point             same shape as scale
   */
  void op(size_t M, size_t K, size_t N, const T* input,
          const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    for (size_t m = 0; m < M; m++) {
      for (size_t k = 0; k < K; k++) {
#if defined(ORT_CLIENT_PACKAGE_BUILD)
        // TODO: Only using multithreaded/SIMD DQ when ORT is built for client/on-device workloads.
        // Make this the default behavior after more testing.
        if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) &&
                      std::is_same_v<ScaleT, OutT>) {
          ParDequantizeLinearStd<T>(input, output, N, scale[k], zero_point ? zero_point[k] : 0, thread_pool);
          input += N;
          output += N;
        } else {
          auto zp = zero_point ? static_cast<int32_t>(zero_point[k]) : 0;
          auto sc = static_cast<float>(scale[k]);
          for (size_t n = 0; n < N; n++) {
            *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
          }
        }
#else
        ORT_UNUSED_PARAMETER(thread_pool);
        auto zp = zero_point ? static_cast<int32_t>(zero_point[k]) : 0;
        auto sc = static_cast<float>(scale[k]);
        for (size_t n = 0; n < N; n++) {
          *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
        }
#endif  // defined(ORT_CLIENT_PACKAGE_BUILD)
      }
    }
  }

  /**
   * @brief Calculate blocked quantization of DequantizeLinear on the flattened tensors.
   *        TODO(fajin): add mlas kernel to utilize multithreading, refer MlasDequantizeBlockwise.
   * @param[in]    M                      size of dimensions before the quantize axis
   * @param[in]    K                      dimension of the quantize axis
   * @param[in]    N                      size of dimensions after the quantize axis
   * @param[in]    quant_block_size       quantize block size along the quantize axis
   * @param[in]    input                  1D array of flattened [D0, ..., Di, ..., Dn]
   * @param[in]    scale                  1D array of flattened [D0, ..., ceil(Di/quant_block_size), ..., Dn].
   *                                      i is the quantize axis.
   * @param[out]   output                 same shape as input
   * @param[in]    zero_point             same shape as scale
   */
  void op(size_t M, size_t K, size_t N, size_t quant_block_size,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    ORT_UNUSED_PARAMETER(thread_pool);
    if (zero_point) {
      for (size_t m = 0; m < M; m++) {
        for (size_t bd = 0; bd < K; bd += quant_block_size) {
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {
            // within the quantize block, the zero point and scale are the same.
            for (size_t bs = 0; bs < N; bs++) {
              auto zp = static_cast<int32_t>(zero_point[bs]);
              auto sc = static_cast<float>(scale[bs]);
              *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++) - zp) * sc);
            }
          }

          // move to the next quantize block
          zero_point += N;
          scale += N;
        }
      }
    } else {
      for (size_t m = 0; m < M; m++) {
        for (size_t bd = 0; bd < K; bd += quant_block_size) {
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {
            // within the quantize block, the zero point and scale are the same.
            for (size_t bs = 0; bs < N; bs++) {
              auto sc = static_cast<float>(scale[bs]);
              *output++ = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(*input++)) * sc);
            }
          }

          // move to the next quantize block
          scale += N;
        }
      }
    }
  }
};

namespace {

#if defined(__GNUC__) && !defined(__wasm__)
#define ORT_SUBBYTE_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define ORT_SUBBYTE_RESTRICT __restrict
#else
#define ORT_SUBBYTE_RESTRICT
#endif

// Number of sub-byte elements unpacked into a stack buffer at a time.
constexpr size_t kSubByteChunk = 512;

// Unpacks `len` sub-byte elements starting at element index `first` into int8 values.
template <typename T, int elements_per_byte>
void UnpackSubByteElements(const T* src, size_t first, size_t len, int8_t* ORT_SUBBYTE_RESTRICT dst) {
  constexpr int bits = 8 / elements_per_byte;
  constexpr uint8_t mask = (1 << bits) - 1;
  constexpr bool is_signed = std::is_signed_v<typename T::UnpackedType>;

  size_t i = 0;
  for (; i < len && (first + i) % elements_per_byte != 0; ++i) {
    dst[i] = static_cast<int8_t>(src[(first + i) / elements_per_byte].GetElem((first + i) % elements_per_byte));
  }

  const size_t num_bytes = (len - i) / elements_per_byte;
  const uint8_t* ORT_SUBBYTE_RESTRICT packed = reinterpret_cast<const uint8_t*>(src + (first + i) / elements_per_byte);
  int8_t* ORT_SUBBYTE_RESTRICT out = dst + i;
  for (size_t b = 0; b < num_bytes; ++b) {
    const uint8_t v = packed[b];
    int8_t* o = out + b * elements_per_byte;
    // Unrolled over the elements of the byte so that the compiler vectorizes with 8-bit lanes.
    [&]<size_t... J>(std::index_sequence<J...>) {
      if constexpr (is_signed) {
        // Move element J to the top bits, then arithmetic-shift it back down to sign-extend it.
        ((o[J] = static_cast<int8_t>(static_cast<int8_t>(v << (8 - bits * (J + 1))) >> (8 - bits))), ...);
      } else {
        ((o[J] = static_cast<int8_t>((v >> (bits * J)) & mask)), ...);
      }
    }(std::make_index_sequence<elements_per_byte>{});
  }
  i += num_bytes * elements_per_byte;

  for (; i < len; ++i) {
    dst[i] = static_cast<int8_t>(src[(first + i) / elements_per_byte].GetElem((first + i) % elements_per_byte));
  }
}

// output[i] = (values[i] - zp) * scale. Same expression as the scalar kernel, so the result is identical.
template <typename OutT>
void DequantizeValues(const int8_t* ORT_SUBBYTE_RESTRICT values, size_t n, int32_t zp, float scale,
                      OutT* ORT_SUBBYTE_RESTRICT output) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(values[i]) - zp) * scale);
  }
}

// output[i] = (values[i] - zps[i]) * scales[i].
template <typename ScaleT, typename OutT>
void DequantizeValues(const int8_t* ORT_SUBBYTE_RESTRICT values, const int8_t* ORT_SUBBYTE_RESTRICT zps,
                      const ScaleT* ORT_SUBBYTE_RESTRICT scales, size_t n, OutT* ORT_SUBBYTE_RESTRICT output) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(values[i]) - zps[i]) *
                                  static_cast<float>(scales[i]));
  }
}

template <typename T, int elements_per_byte>
int32_t GetZeroPoint(const T* zero_point, size_t index) {
  return zero_point ? static_cast<int32_t>(zero_point[index / elements_per_byte].GetElem(index % elements_per_byte))
                    : 0;
}

// output[i] = (input[first + i] - zp) * scale for i in [0, len).
// With per_element_params, zp and scale advance with i, otherwise they are fixed.
// The zero point is read from packed element index zp_first.
template <typename T, typename ScaleT, typename OutT, int elements_per_byte, bool per_element_params>
void DequantizeSubByteRun(const T* input, size_t first, size_t len, const ScaleT* scale, const T* zero_point,
                          size_t zp_first, OutT* output) {
  alignas(64) int8_t values[kSubByteChunk];
  alignas(64) int8_t zps[kSubByteChunk];
  if constexpr (per_element_params) {
    if (!zero_point) {
      std::fill_n(zps, std::min(len, kSubByteChunk), int8_t{0});
    }
  }

  for (size_t c = 0; c < len; c += kSubByteChunk) {
    const size_t n = std::min(kSubByteChunk, len - c);
    UnpackSubByteElements<T, elements_per_byte>(input, first + c, n, values);
    if constexpr (per_element_params) {
      if (zero_point) {
        UnpackSubByteElements<T, elements_per_byte>(zero_point, zp_first + c, n, zps);
      }
      DequantizeValues(values, zps, scale + c, n, output + c);
    } else {
      DequantizeValues(values, n, GetZeroPoint<T, elements_per_byte>(zero_point, zp_first),
                       static_cast<float>(scale[0]), output + c);
    }
  }
}

// Runs fn(unit) for unit in [0, num_units) on the thread pool. Each unit dequantizes about unit_size elements.
template <typename OutT, typename Fn>
void ParallelForSubByte(concurrency::ThreadPool* thread_pool, size_t num_units, size_t unit_size, const Fn& fn) {
  const double n = static_cast<double>(unit_size);
  const TensorOpCost unit_cost{n, n * sizeof(OutT), n * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, static_cast<std::ptrdiff_t>(num_units), unit_cost,
                                          [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
                                            for (std::ptrdiff_t u = begin; u < end; ++u) {
                                              fn(static_cast<size_t>(u));
                                            }
                                          });
}

// Rows of N < kSubByteChunk elements, row r = m * K + k. Each unit unpacks several whole rows at once.
// Per-axis (quant_block_size == 0): row r uses scale k for all N elements.
// Blocked: row r uses the N consecutive scales starting at (m * num_blocks + k / quant_block_size) * N.
template <typename T, typename ScaleT, typename OutT, int elements_per_byte, bool blocked>
void DequantizeSubByteShortRows(size_t M, size_t K, size_t N, size_t quant_block_size, const T* input,
                                const ScaleT* scale, OutT* output, const T* zero_point,
                                concurrency::ThreadPool* thread_pool) {
  const size_t num_rows = M * K;
  const size_t rows_per_unit = kSubByteChunk / N;
  const size_t num_blocks = quant_block_size ? (K + quant_block_size - 1) / quant_block_size : 0;
  ParallelForSubByte<OutT>(
      thread_pool, (num_rows + rows_per_unit - 1) / rows_per_unit, rows_per_unit * N, [&](size_t unit) {
        const size_t row_begin = unit * rows_per_unit;
        const size_t row_end = std::min(num_rows, row_begin + rows_per_unit);
        alignas(64) int8_t values[kSubByteChunk];
        alignas(64) int8_t zps[kSubByteChunk];
        UnpackSubByteElements<T, elements_per_byte>(input, row_begin * N, (row_end - row_begin) * N, values);
        if constexpr (blocked) {
          if (!zero_point) {
            std::fill_n(zps, N, int8_t{0});
          }
        }

        // Walk (m, k) and the block of k without a division per row.
        size_t m = row_begin / K;
        size_t k = row_begin % K;
        size_t block = quant_block_size ? k / quant_block_size : 0;
        size_t in_block = quant_block_size ? k % quant_block_size : 0;
        size_t zps_begin = std::numeric_limits<size_t>::max();
        for (size_t row = row_begin; row < row_end; ++row) {
          const int8_t* row_values = values + (row - row_begin) * N;
          OutT* row_output = output + row * N;
          if constexpr (blocked) {
            const size_t s = (m * num_blocks + block) * N;
            // Rows of the same block share their N zero points, so unpack them only when the block changes.
            if (zero_point && s != zps_begin) {
              UnpackSubByteElements<T, elements_per_byte>(zero_point, s, N, zps);
              zps_begin = s;
            }
            for (size_t i = 0; i < N; ++i) {
              row_output[i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(row_values[i]) - zps[i]) *
                                                static_cast<float>(scale[s + i]));
            }
          } else {
            const int32_t zp = GetZeroPoint<T, elements_per_byte>(zero_point, k);
            const float sc = static_cast<float>(scale[k]);
            for (size_t i = 0; i < N; ++i) {
              row_output[i] = static_cast<OutT>(static_cast<float>(static_cast<int32_t>(row_values[i]) - zp) * sc);
            }
          }
          if (++k == K) {
            k = 0;
            ++m;
            block = 0;
            in_block = 0;
          } else if constexpr (blocked) {
            if (++in_block == quant_block_size) {
              in_block = 0;
              ++block;
            }
          }
        }
      });
}

}  // namespace

template <typename T, typename ScaleT, typename OutT, int elements_per_byte>
struct DequantizeLinearApply<T, ScaleT, OutT, true, elements_per_byte> {
  // per-tensor/layer or per-axis quantization for sub-byte types
  void op(size_t M, size_t K, size_t N,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    if (M * K * N == 0) {
      return;
    }
    if (N == 1) {
      // Scale and zero point change with every element. Split each row of K elements into chunks.
      const size_t chunks_per_row = (K + kSubByteChunk - 1) / kSubByteChunk;
      ParallelForSubByte<OutT>(thread_pool, M * chunks_per_row, std::min(K, kSubByteChunk), [&](size_t unit) {
        const size_t m = unit / chunks_per_row;
        const size_t k = (unit % chunks_per_row) * kSubByteChunk;
        const size_t len = std::min(kSubByteChunk, K - k);
        DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, true>(input, m * K + k, len, scale + k, zero_point,
                                                                       k, output + m * K + k);
      });
      return;
    }

    // Each row of N elements shares one scale and zero point.
    if (N >= kSubByteChunk) {
      // Long rows: split each row into chunks so that a single row (e.g. per-tensor) is still parallel.
      const size_t chunks_per_row = (N + kSubByteChunk - 1) / kSubByteChunk;
      ParallelForSubByte<OutT>(thread_pool, M * K * chunks_per_row, kSubByteChunk, [&](size_t unit) {
        const size_t row = unit / chunks_per_row;
        const size_t n = (unit % chunks_per_row) * kSubByteChunk;
        const size_t len = std::min(kSubByteChunk, N - n);
        const size_t k = row % K;
        DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, false>(input, row * N + n, len, scale + k,
                                                                        zero_point, k, output + row * N + n);
      });
      return;
    }

    DequantizeSubByteShortRows<T, ScaleT, OutT, elements_per_byte, false>(M, K, N, 0, input, scale, output, zero_point,
                                                                          thread_pool);
  }

  // Blocked quantization
  void op(size_t M, size_t K, size_t N, size_t quant_block_size,
          const T* input, const ScaleT* scale, OutT* output, const T* zero_point, concurrency::ThreadPool* thread_pool) {
    if (M * K * N == 0) {
      return;
    }
    const size_t num_blocks = (K + quant_block_size - 1) / quant_block_size;

    if (N == 1) {
      // Each quantize block is a contiguous run of elements with one scale and zero point.
      // Short blocks are grouped so that every unit unpacks about kSubByteChunk elements at once.
      const size_t blocks_per_unit = std::max<size_t>(1, kSubByteChunk / quant_block_size);
      const size_t units_per_row = (num_blocks + blocks_per_unit - 1) / blocks_per_unit;
      ParallelForSubByte<OutT>(
          thread_pool, M * units_per_row, std::min(K, blocks_per_unit * quant_block_size), [&](size_t unit) {
            const size_t m = unit / units_per_row;
            const size_t first_block = (unit % units_per_row) * blocks_per_unit;
            const size_t k_begin = first_block * quant_block_size;
            const size_t k_end = std::min(K, k_begin + blocks_per_unit * quant_block_size);
            if (blocks_per_unit == 1) {
              const size_t s = m * num_blocks + first_block;
              DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, false>(
                  input, m * K + k_begin, k_end - k_begin, scale + s, zero_point, s, output + m * K + k_begin);
              return;
            }

            alignas(64) int8_t values[kSubByteChunk];
            UnpackSubByteElements<T, elements_per_byte>(input, m * K + k_begin, k_end - k_begin, values);
            for (size_t k = k_begin, block = m * num_blocks + first_block; k < k_end; k += quant_block_size, ++block) {
              DequantizeValues(values + (k - k_begin), std::min(quant_block_size, k_end - k),
                               GetZeroPoint<T, elements_per_byte>(zero_point, block), static_cast<float>(scale[block]),
                               output + m * K + k);
            }
          });
      return;
    }

    // Each row of N elements uses N consecutive scales and zero points.
    if (N >= kSubByteChunk) {
      const size_t chunks_per_row = (N + kSubByteChunk - 1) / kSubByteChunk;
      ParallelForSubByte<OutT>(thread_pool, M * K * chunks_per_row, kSubByteChunk, [&](size_t unit) {
        const size_t row = unit / chunks_per_row;
        const size_t n = (unit % chunks_per_row) * kSubByteChunk;
        const size_t len = std::min(kSubByteChunk, N - n);
        const size_t s = (row / K * num_blocks + row % K / quant_block_size) * N + n;
        DequantizeSubByteRun<T, ScaleT, OutT, elements_per_byte, true>(input, row * N + n, len, scale + s, zero_point,
                                                                       s, output + row * N + n);
      });
      return;
    }

    DequantizeSubByteShortRows<T, ScaleT, OutT, elements_per_byte, true>(M, K, N, quant_block_size, input, scale, output,
                                                                         zero_point, thread_pool);
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

#define DEQUANTIZE_LINEAR_APPLY_FLOAT8(T)                                                            \
  template <typename ScaleT, typename OutT, int elements_per_byte>                                   \
  struct DequantizeLinearApply<T, ScaleT, OutT, false, elements_per_byte> {                          \
    /* Per-tensor/layer or per-axis quantization */                                                  \
    void op(size_t M, size_t K, size_t N,                                                            \
            const T* input, const ScaleT* scale, OutT* output, const T*, concurrency::ThreadPool*) { \
      for (size_t m = 0; m < M; m++) {                                                               \
        for (size_t bd = 0; bd < K; bd++) {                                                          \
          auto sc = scale[bd];                                                                       \
          for (size_t bs = 0; bs < N; bs++, input++) {                                               \
            *output++ = static_cast<OutT>(input->ToFloat() * sc);                                    \
          }                                                                                          \
        }                                                                                            \
      }                                                                                              \
    }                                                                                                \
    /* Blocked quantization */                                                                       \
    void op(size_t M, size_t K, size_t N, size_t quant_block_size,                                   \
            const T* input, const ScaleT* scale, OutT* output, const T*, concurrency::ThreadPool*) { \
      for (size_t m = 0; m < M; m++) {                                                               \
        for (size_t bd = 0; bd < K; bd += quant_block_size) {                                        \
          for (size_t qb = 0, qb_end = std::min(quant_block_size, K - bd); qb < qb_end; ++qb) {      \
            for (size_t bs = 0; bs < N; bs++, input++) {                                             \
              auto sc = static_cast<float>(scale[bs]);                                               \
              *output++ = static_cast<OutT>(input->ToFloat() * sc);                                  \
            }                                                                                        \
          }                                                                                          \
          scale += N;                                                                                \
        }                                                                                            \
      }                                                                                              \
    }                                                                                                \
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

  int64_t process_block_count;
  int64_t broadcast_dim;
  int64_t process_block_size;

  PrepareForQDQ(x.Shape(), x_scale, x_zero_point, axis_, block_size_,
                process_block_count, broadcast_dim, process_block_size);

  const T* zero_point = x_zero_point ? x_zero_point->Data<T>() : nullptr;

#if !defined(DISABLE_FLOAT8_TYPES)
  if constexpr (boost::mp11::mp_contains<element_type_lists::AllFloat8, T>::value) {
    ORT_ENFORCE(zero_point == nullptr ||
                    std::all_of(zero_point,
                                zero_point + x_zero_point->Shape().Size(),
                                [](T zp) { return zp == T{0}; }),
                "DequantizeLinear with type float8 should have no zero point or all zero points should be 0");
  }
#endif

  const auto to = y.GetElementType();
  const T* input = x.Data<T>();
  constexpr bool is_sub_byte = boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2, Int2x4, UInt2x4>, T>::value;
  // Determine elements_per_byte: Int4x2/UInt4x2 = 2, Int2x4/UInt2x4 = 4
  constexpr int elements_per_byte =
      boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2>, T>::value ? 2 : boost::mp11::mp_contains<TypeList<Int2x4, UInt2x4>, T>::value ? 4
                                                                                                                                        : 0;
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  auto dequantize = [&]<typename ScaleT, typename OutT>(const ScaleT* scale, OutT* output) {
    DequantizeLinearApply<T, ScaleT, OutT, is_sub_byte, elements_per_byte> apply;
    if (block_size_) {
      apply.op(static_cast<size_t>(process_block_count),
               static_cast<size_t>(broadcast_dim),
               static_cast<size_t>(process_block_size),
               static_cast<size_t>(block_size_),
               input, scale, output, zero_point, thread_pool);
    } else {
      apply.op(static_cast<size_t>(process_block_count),
               static_cast<size_t>(broadcast_dim),
               static_cast<size_t>(process_block_size),
               input, scale, output, zero_point, thread_pool);
    }
  };

  if (to == ONNX_NAMESPACE::TensorProto::FLOAT) {
    float* output = y.MutableData<float>();
    if (x_scale.IsDataType<float>()) {
      dequantize(x_scale.Data<float>(), output);
    } else {
      dequantize(x_scale.Data<MLFloat16>(), output);
    }
  } else if (to == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    MLFloat16* output = y.MutableData<MLFloat16>();
    if (x_scale.IsDataType<float>()) {
      dequantize(x_scale.Data<float>(), output);
    } else {
      dequantize(x_scale.Data<MLFloat16>(), output);
    }
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
      25,                                                                   \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED(T, start_version, end_version)    \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                 \
      QuantizeLinear,                                                       \
      start_version,                                                        \
      end_version,                                                          \
      T,                                                                    \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),      \
                                 DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>()),          \
      QuantizeLinear<T>);

#define REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(T, start_version, end_version) \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                     \
      QuantizeLinear,                                                           \
      start_version,                                                            \
      end_version,                                                              \
      T,                                                                        \
      KernelDefBuilder()                                                        \
          .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),          \
                                 DataTypeImpl::GetTensorType<MLFloat16>()})     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),              \
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

// Opset 25
REGISTER_QUANTIZELINEAR(int8_t)
REGISTER_QUANTIZELINEAR(uint8_t)
REGISTER_QUANTIZELINEAR(int16_t)
REGISTER_QUANTIZELINEAR(uint16_t)
REGISTER_QUANTIZELINEAR(Int4x2)
REGISTER_QUANTIZELINEAR(UInt4x2)
REGISTER_QUANTIZELINEAR(Int2x4)
REGISTER_QUANTIZELINEAR(UInt2x4)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR(Float8E4M3FN)
REGISTER_QUANTIZELINEAR(Float8E4M3FNUZ)
REGISTER_QUANTIZELINEAR(Float8E5M2)
REGISTER_QUANTIZELINEAR(Float8E5M2FNUZ)
#endif

// Opset 24
REGISTER_QUANTIZELINEAR_VERSIONED(int8_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(int16_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(uint16_t, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Int4x2, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(UInt4x2, 24, 24)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FN, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2, 24, 24)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 24, 24)
#endif

// Opset 23 added support for float4e2m1.
REGISTER_QUANTIZELINEAR_VERSIONED(int8_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(uint8_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(int16_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(uint16_t, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Int4x2, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(UInt4x2, 23, 23)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FN, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E4M3FNUZ, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2, 23, 23)
REGISTER_QUANTIZELINEAR_VERSIONED(Float8E5M2FNUZ, 23, 23)
#endif

// Opset 21 added 16-bit and 4-bit int support to Q ops.
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(int16_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(uint16_t, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Int4x2, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(UInt4x2, 21, 22)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 21, 22)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 21, 22)
#endif

// Opset 19 added 8-bit floats to Q ops.
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(int8_t, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(uint8_t, 19, 20)
#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FN, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E4M3FNUZ, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2, 19, 20)
REGISTER_QUANTIZELINEAR_VERSIONED_PRE_23(Float8E5M2FNUZ, 19, 20)
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
                       float Scale,
                       size_t bd,
                       const OutputType* ZeroPoint,
                       bool saturate,
                       concurrency::ThreadPool* thread_pool,
                       bool half_precision) {
#if !defined(DISABLE_FLOAT8_TYPES)
  if constexpr (!boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputType>::value) {
#endif
    ORT_UNUSED_PARAMETER(saturate);
    ParQuantizeLinearStd(Input, Output, N, Scale, ZeroPoint != nullptr ? ZeroPoint[bd] : (OutputType)0,
                         thread_pool, half_precision);
#if !defined(DISABLE_FLOAT8_TYPES)
  } else {
    ParQuantizeLinearSat(Input, Output, N, Scale,
                         ZeroPoint != nullptr ? ZeroPoint[bd]
                                              : OutputType(static_cast<InputType>(static_cast<float>(0)), true),
                         saturate, thread_pool, half_precision);
  }
#endif
}

/**
 * @brief Compute per-tensor or per-axis quantization.
 */
template <typename T, typename InT>
void ComputeLoop(OpKernelContext* ctx, const InT* input, QuantizeLinearScale scale, const T* zero_point, T* output,
                 int64_t process_block_count, int64_t broadcast_dim, int64_t process_block_size,
                 bool saturate, bool half_precision) {
  for (size_t n = 0; n < static_cast<size_t>(process_block_count); n++) {
    for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {
      ParQuantizeLinear(input, output, static_cast<size_t>(process_block_size), scale[bd], bd, zero_point,
                        saturate, ctx->GetOperatorThreadPool(), half_precision);
      input += process_block_size;
      output += process_block_size;
    }
  }
}

// Helper macros to create zero point with correct number of constructor arguments
#define CREATE_SUB_BYTE_ZP_2(TYPE, zp) TYPE(zp, 0)
#define CREATE_SUB_BYTE_ZP_4(TYPE, zp) TYPE(zp, 0, 0, 0)
#define CREATE_SUB_BYTE_ZP(TYPE, zp, ELEMENTS_PER_BYTE) CREATE_SUB_BYTE_ZP_##ELEMENTS_PER_BYTE(TYPE, zp)

// Quantizes float32 to sub-byte types using MLAS kernel (4-bit) or generic quantization (2-bit).
#define DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(SUB_BYTE_TYPE, QUANT_FUNC, ELEMENTS_PER_BYTE)                               \
  template <>                                                                                                            \
  void ComputeLoop(OpKernelContext* ctx, const float* input, QuantizeLinearScale scale, const SUB_BYTE_TYPE* zero_point, \
                   SUB_BYTE_TYPE* output, int64_t M, int64_t K, int64_t N, bool saturate, bool half_precision) {         \
    ORT_UNUSED_PARAMETER(saturate);                                                                                      \
    size_t output_index = 0;                                                                                             \
    constexpr size_t shift_bits = (ELEMENTS_PER_BYTE == 2) ? 1 : 2; /* log2(ELEMENTS_PER_BYTE) */                        \
    constexpr size_t mask = ELEMENTS_PER_BYTE - 1;                  /* For modulo operation */                           \
    for (size_t m = 0; m < static_cast<size_t>(M); m++) {                                                                \
      for (size_t bd = 0; bd < static_cast<size_t>(K); bd++) {                                                           \
        size_t bd_i = bd >> shift_bits; /* bd / ELEMENTS_PER_BYTE */                                                     \
        size_t bd_j = bd & mask;        /* bd % ELEMENTS_PER_BYTE */                                                     \
        SUB_BYTE_TYPE::UnpackedType zp = zero_point ? zero_point[bd_i].GetElem(bd_j) : 0;                                \
        QUANT_FUNC(input, output, output_index, output_index + static_cast<size_t>(N),                                   \
                   scale[bd], CREATE_SUB_BYTE_ZP(SUB_BYTE_TYPE, zp, ELEMENTS_PER_BYTE),                                  \
                   ctx->GetOperatorThreadPool(), half_precision);                                                        \
        input += N;                                                                                                      \
        output_index += static_cast<size_t>(N);                                                                          \
      }                                                                                                                  \
    }                                                                                                                    \
    assert(output_index == static_cast<size_t>(SafeInt<size_t>(M) * K * N));                                             \
  }

DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(Int4x2, ParQuantizeLinearStdS4, 2)
DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(UInt4x2, ParQuantizeLinearStdU4, 2)
DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(Int2x4, ParQuantizeLinearStdS2, 4)
DEFINE_COMPUTE_LOOP_FP32_TO_SUB_BYTE(UInt2x4, ParQuantizeLinearStdU2, 4)

// Defines functions to quantize MLFloat16 to sub-byte types.
// This is not an efficient implementation: we allocate a buffer, quantize to the unpacked type, and then clamp/pack
// into output sub-byte buffer.
#define DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(SUB_BYTE_TYPE, ELEMENTS_PER_BYTE)                                         \
  template <>                                                                                                          \
  void ComputeLoop<SUB_BYTE_TYPE, MLFloat16>(OpKernelContext * ctx, const MLFloat16* input, QuantizeLinearScale scale, \
                                             const SUB_BYTE_TYPE* zero_point, SUB_BYTE_TYPE* output, int64_t M,        \
                                             int64_t K, int64_t N, bool saturate, bool half_precision) {               \
    ORT_UNUSED_PARAMETER(saturate);                                                                                    \
                                                                                                                       \
    size_t total_size = static_cast<size_t>(SafeInt<size_t>(M) * K * N);                                               \
    auto tmp_buf = std::make_unique<SUB_BYTE_TYPE::UnpackedType[]>(total_size);                                        \
    size_t tmp_buf_index = 0;                                                                                          \
    constexpr size_t shift_bits = (ELEMENTS_PER_BYTE == 2) ? 1 : 2; /* log2(ELEMENTS_PER_BYTE) */                      \
    constexpr size_t mask = ELEMENTS_PER_BYTE - 1;                  /* For modulo operation */                         \
                                                                                                                       \
    for (size_t m = 0; m < static_cast<size_t>(M); m++) {                                                              \
      for (size_t bd = 0; bd < static_cast<size_t>(K); bd++) {                                                         \
        size_t bd_i = bd >> shift_bits; /* bd / ELEMENTS_PER_BYTE */                                                   \
        size_t bd_j = bd & mask;        /* bd % ELEMENTS_PER_BYTE */                                                   \
        SUB_BYTE_TYPE::UnpackedType zp = zero_point ? zero_point[bd_i].GetElem(bd_j) : 0;                              \
        ParQuantizeLinearStd<SUB_BYTE_TYPE::UnpackedType>(input, tmp_buf.get() + tmp_buf_index,                        \
                                                          static_cast<size_t>(N), scale[bd],                           \
                                                          zp, ctx->GetOperatorThreadPool(), half_precision);           \
        input += N;                                                                                                    \
        tmp_buf_index += static_cast<size_t>(N);                                                                       \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    for (size_t i = 0; i < total_size; i++) {                                                                          \
      tmp_buf[i] = std::min<SUB_BYTE_TYPE::UnpackedType>(SUB_BYTE_TYPE::max_val,                                       \
                                                         std::max<SUB_BYTE_TYPE::UnpackedType>(SUB_BYTE_TYPE::min_val, \
                                                                                               tmp_buf[i]));           \
    }                                                                                                                  \
                                                                                                                       \
    size_t num_packed = (total_size + ELEMENTS_PER_BYTE - 1) / ELEMENTS_PER_BYTE;                                      \
    auto dst = gsl::make_span<SUB_BYTE_TYPE>(output, num_packed);                                                      \
    auto src = gsl::make_span<const SUB_BYTE_TYPE::UnpackedType>(tmp_buf.get(), total_size);                           \
    SUB_BYTE_TYPE::Pack(dst, src);                                                                                     \
  }

DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(Int4x2, 2)
DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(UInt4x2, 2)
DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(Int2x4, 4)
DEFINE_COMPUTE_LOOP_FP16_TO_SUB_BYTE(UInt2x4, 4)

// formula is Y = X / Scale + ZeroPoint
template <typename T>
Status QuantizeLinear<T>::Compute(OpKernelContext* ctx) const {
  auto& x = *ctx->Input<Tensor>(0);
  auto& y_scale = *ctx->Input<Tensor>(1);
  auto* y_zero_point = ctx->Input<Tensor>(2);
  const auto& x_shape = x.Shape();
  auto& y = *ctx->Output(0, x_shape);

  int64_t process_block_count;
  int64_t broadcast_dim;
  int64_t process_block_size;
  PrepareForQDQ(x.Shape(), y_scale, y_zero_point, axis_, block_size_,
                process_block_count, broadcast_dim, process_block_size);

  const T* zero_point = y_zero_point != nullptr ? y_zero_point->Data<T>() : nullptr;
  T* output = y.MutableData<T>();

  const auto precision = precision_ == 0 ? y_scale.GetElementType() : precision_;
  const bool half_precision = precision == ONNX_NAMESPACE::TensorProto::FLOAT16;
  const QuantizeLinearScale scale = y_scale.IsDataType<float>()
                                        ? QuantizeLinearScale(y_scale.Data<float>())
                                        : QuantizeLinearScale(y_scale.Data<MLFloat16>());
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  if constexpr (boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2, Int2x4, UInt2x4>, T>::value) {
    // Every element is written below, so only the unused lanes of a partial last byte need zeroing.
    constexpr size_t elems_per_byte = boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2>, T>::value ? 2 : 4;
    if (static_cast<size_t>(x_shape.Size()) % elems_per_byte != 0) {
      output[y.SizeInBytes() / sizeof(T) - 1] = T{};
    }
  }

  constexpr int output_type_group_ =
      boost::mp11::mp_contains<TypeList<Int4x2, UInt4x2>, T>::value   ? 2
      : boost::mp11::mp_contains<TypeList<Int2x4, UInt2x4>, T>::value ? 3
#if !defined(DISABLE_FLOAT8_TYPES)
      : boost::mp11::mp_contains<element_type_lists::AllFloat8, T>::value ? 1
#endif
                                                                          : 0;

  if (x.IsDataType<float>()) {
    if (block_size_) {
      if (process_block_size > 1) {
        BlockedQuantizeLinear<float, T, output_type_group_>::opNotLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<float>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(process_block_size),
            static_cast<std::ptrdiff_t>(block_size_),
            128,
            saturate_,
            half_precision);
      } else {
        BlockedQuantizeLinear<float, T, output_type_group_>::opLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<float>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(block_size_),
            saturate_,
            half_precision);
      }
    } else {
      ComputeLoop<T, float>(ctx, x.Data<float>(), scale, zero_point, output,
                            process_block_count, broadcast_dim, process_block_size, saturate_, half_precision);
    }
  } else if (x.IsDataType<MLFloat16>()) {
    if (block_size_) {
      if (process_block_size > 1) {
        BlockedQuantizeLinear<MLFloat16, T, output_type_group_>::opNotLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<MLFloat16>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(process_block_size),
            static_cast<std::ptrdiff_t>(block_size_),
            128,
            saturate_,
            half_precision);
      } else {
        BlockedQuantizeLinear<MLFloat16, T, output_type_group_>::opLastAxis(
            ctx->GetOperatorThreadPool(),
            x.Data<MLFloat16>(),
            scale,
            zero_point,
            output,
            static_cast<std::ptrdiff_t>(process_block_count),
            static_cast<std::ptrdiff_t>(broadcast_dim),
            static_cast<std::ptrdiff_t>(block_size_),
            saturate_,
            half_precision);
      }
    } else {
      ComputeLoop<T, MLFloat16>(ctx, x.Data<MLFloat16>(), scale, zero_point, output,
                                process_block_count, broadcast_dim, process_block_size, saturate_, half_precision);
    }
  } else {
    ORT_THROW("Unsupported input type.");
  }

  return Status::OK();
}
}  // namespace onnxruntime
