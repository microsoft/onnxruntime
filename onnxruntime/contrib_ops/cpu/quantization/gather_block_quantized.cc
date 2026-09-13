// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <algorithm>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/float16.h"
#include "core/framework/int4.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"

#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"
#endif
#if !defined(DISABLE_FLOAT4_TYPES)
#include "core/framework/float4.h"
#endif

namespace onnxruntime {
namespace contrib {

namespace {
template <typename T1>
int32_t Get4BitElement(const T1* data_ptr, int64_t data_idx) {
  return static_cast<int32_t>(data_ptr[data_idx >> 1].GetElem(narrow<size_t>(data_idx & 1)));
}

template <>
int32_t Get4BitElement<uint8_t>(const uint8_t* data_ptr, int64_t data_idx) {
  const uint8_t data_val_u8 = data_ptr[data_idx >> 1];
  // Weights are stored as (nibble2)(nibble1) in uint8_t.
  auto data_val = static_cast<int32_t>((data_idx & 1) ? ((data_val_u8 >> 4) & 0x0F) : (data_val_u8 & 0x0F));
  return data_val;
}

// Extracts a 2-bit element from uint8_t storage. Four elements are packed per byte,
// with element index 0 in the lowest 2 bits and index 3 in the highest 2 bits.
int32_t Get2BitElementUint8(const uint8_t* data_ptr, int64_t data_idx) {
  const uint8_t data_val_u8 = data_ptr[data_idx >> 2];
  const int shift = static_cast<int>((data_idx & 3) * 2);
  return static_cast<int32_t>((data_val_u8 >> shift) & 0x03);
}

// Max number of elements processed per SIMD batch call in the uint8_t data fast path below. Bounds
// the size of the on-stack unpack buffer; larger runs are simply split into several batches, which is
// harmless because scale/zero-point are already known to be constant across the whole run.
constexpr int64_t kUint8DequantBatch = 256;

// Unpacks `count` (<= kUint8DequantBatch) consecutive bits_-wide elements, starting at element index
// `data_idx`, from packed uint8_t storage into `out`, one code (0..255) per output byte. For bits_==8
// this is a no-op copy (the codes are already unpacked bytes); for bits_==2/4 it expands the packed
// nibbles/crumbs so the resulting buffer can be fed to a single SIMD dequantization call.
void UnpackUint8Elements(const uint8_t* data_ptr, int64_t data_idx, int64_t count, int64_t bits,
                         uint8_t* out) {
  if (bits == 8) {
    memcpy(out, data_ptr + data_idx, narrow<size_t>(count));
  } else if (bits == 4) {
    for (int64_t i = 0; i < count; ++i) {
      out[i] = static_cast<uint8_t>(Get4BitElement(data_ptr, data_idx + i));
    }
  } else {  // bits == 2
    for (int64_t i = 0; i < count; ++i) {
      out[i] = static_cast<uint8_t>(Get2BitElementUint8(data_ptr, data_idx + i));
    }
  }
}

// Trait identifying the FP8/FP4 data types supported by GatherBlockQuantized. Unlike the integer
// block-quantized types (uint8_t/UInt4x2/Int4x2), these have no zero point (symmetric quantization)
// and their "block_size" attribute may be 0, meaning a single block spans the whole quantize_axis.
template <typename T1>
struct IsFpQuantized : std::false_type {};

#if !defined(DISABLE_FLOAT8_TYPES)
template <>
struct IsFpQuantized<Float8E4M3FN> : std::true_type {};
template <>
struct IsFpQuantized<Float8E4M3FNUZ> : std::true_type {};
template <>
struct IsFpQuantized<Float8E5M2> : std::true_type {};
template <>
struct IsFpQuantized<Float8E5M2FNUZ> : std::true_type {};
#endif  // !defined(DISABLE_FLOAT8_TYPES)
#if !defined(DISABLE_FLOAT4_TYPES)
template <>
struct IsFpQuantized<Float4E2M1x2> : std::true_type {};
#endif  // !defined(DISABLE_FLOAT4_TYPES)

template <typename T1>
constexpr bool IsFpQuantizedV = IsFpQuantized<T1>::value;

// Reads the logical element at `idx` from an FP8/FP4 quantized data buffer and returns it as a float.
// FP8 types store one element per byte, so this is the default. FP4 (Float4E2M1x2) packs two logical
// elements per byte; the tensor's shape is still the logical shape (as with the existing Int4x2/UInt4x2
// sub-byte types), so the physical byte and the sub-element within it must be derived from the logical
// index.
template <typename T1>
inline float DequantizedFpElem(const T1* data_ptr, int64_t idx) {
  return data_ptr[idx].ToFloat();
}

#if !defined(DISABLE_FLOAT4_TYPES)
template <>
inline float DequantizedFpElem<Float4E2M1x2>(const Float4E2M1x2* data_ptr, int64_t idx) {
  return data_ptr[idx >> 1].GetElem(narrow<size_t>(idx & 1));
}
#endif  // !defined(DISABLE_FLOAT4_TYPES)

}  // namespace

template <typename T1, typename Tind>
class GatherBlockQuantized : public OpKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info) : OpKernel(info) {
    if (!info.GetAttr<int64_t>("gather_axis", &gather_axis_).IsOK()) {
      gather_axis_ = 0;
    }

    if (!info.GetAttr<int64_t>("quantize_axis", &quantize_axis_).IsOK()) {
      quantize_axis_ = 1;
    }

    if (!info.GetAttr<int64_t>("block_size", &block_size_).IsOK()) {
      block_size_ = 128;
    }

    if constexpr (IsFpQuantizedV<T1>) {
      ORT_ENFORCE(block_size_ == 0 || (block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0),
                  "'block_size' must be 0, or a power of 2 and not less than 16.");
    } else {
      ORT_ENFORCE(block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0,
                  "'block_size' must be a power of 2 and not less than 16.");
    }

    constexpr int64_t default_bits = 4;
    info.GetAttrOrDefault("bits", &bits_, default_bits);
    if constexpr (IsFpQuantizedV<T1>) {
      // 'bits' is not applicable to FP8/FP4 data: each element occupies a fixed number of bits
      // determined by the type itself (8 for FP8, 4 for FP4), and there is no integer zero point.
    } else if constexpr (std::is_same_v<T1, uint8_t>) {
      ORT_ENFORCE(bits_ == 2 || bits_ == 4 || bits_ == 8,
                  "GatherBlockQuantized with uint8 data only supports bits==2, 4, or 8");
    } else {
      ORT_ENFORCE(bits_ == 4, "GatherBlockQuantized with int4/uint4 data only supports bits==4");
    }
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  struct Prepare {
    const Tensor* data_tensor;
    const Tensor* indices_tensor;
    const Tensor* scales_tensor;
    const Tensor* zero_points_tensor;
    Tensor* output_tensor;
    int64_t gather_axis;
    int64_t quantize_axis;
    // The following are only populated/used for FP8/FP4 data (IsFpQuantizedV<T1>), which supports an
    // arbitrary quantize_axis position and broadcastable (dim==1) scales on non-quantize_axis axes.
    int64_t effective_block_size;
    // Row-major strides of `data`, used to decompose a flat data index into per-axis indices.
    TensorShapeVector data_strides;
    // Row-major strides of `scales`. For a broadcast axis (scales dim == 1, data dim > 1) the
    // corresponding per-axis index contribution is always 0, regardless of this stride.
    TensorShapeVector scale_strides;
    // Per-axis flag (indexed like data/scales axes), true when that axis is broadcast in `scales`
    // (i.e. scales dim == 1 while data dim != 1). Unused/ignored at quantize_axis, which is always
    // handled via block-index division instead.
    InlinedVector<bool> scale_broadcast_axis;
  };

  Status PrepareForCompute(OpKernelContext* context, Prepare& args) const;

  template <typename T2>
  Status CopyDataAndDequantize(const T1* data_ptr,
                               const Tind* indices_ptr,
                               const T2* scales_ptr,
                               const T1* zero_points_ptr,
                               T2* output_ptr,
                               const int64_t gather_M,
                               const int64_t gather_N,
                               const int64_t gather_axis_dim,
                               const int64_t gather_block,
                               const int64_t quantize_axis_dim,
                               const int64_t quantize_N,
                               const Prepare& p,
                               concurrency::ThreadPool* tp) const;

 private:
  int64_t gather_axis_;
  int64_t quantize_axis_;
  int64_t block_size_;
  int64_t bits_;
};

template <typename T1, typename Tind>
Status GatherBlockQuantized<T1, Tind>::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  p.data_tensor = context->Input<Tensor>(0);
  p.indices_tensor = context->Input<Tensor>(1);
  p.scales_tensor = context->Input<Tensor>(2);
  p.zero_points_tensor = context->Input<Tensor>(3);

  const auto& data_shape = p.data_tensor->Shape();
  const auto data_rank = data_shape.NumDimensions();
  p.gather_axis = HandleNegativeAxis(gather_axis_, narrow<int64_t>(data_rank));

  p.quantize_axis = HandleNegativeAxis(quantize_axis_, narrow<int64_t>(data_rank));
  if constexpr (IsFpQuantizedV<T1>) {
    ORT_RETURN_IF_NOT(p.zero_points_tensor == nullptr,
                      "zero_points must not be provided when data is an FP8 or FP4 type.");
  } else if constexpr (std::is_same_v<T1, uint8_t>) {
    ORT_RETURN_IF_NOT(p.gather_axis == 0, "For uint8_t data, gather_axis must be 0.");
    ORT_RETURN_IF_NOT(p.quantize_axis == static_cast<int64_t>(data_rank) - 1, "For uint8_t data, quantize_axis must be the last dimension.");
    ORT_RETURN_IF_NOT(p.gather_axis != p.quantize_axis, "gather_axis and quantize_axis must not be the same.");
  }

  const auto& indices_shape = p.indices_tensor->Shape();
  const auto indices_rank = indices_shape.NumDimensions();

  TensorShapeVector shape;
  shape.reserve(data_rank - 1 + indices_rank);

  // get output tensor
  // replace the dimension for p.gather_axis with the shape from the indices
  for (int64_t i = 0; i < p.gather_axis; ++i)
    shape.push_back(data_shape[narrow<size_t>(i)]);

  for (const auto dim : indices_shape.GetDims())
    shape.push_back(dim);

  for (int64_t i = p.gather_axis + 1; i < static_cast<int64_t>(data_rank); ++i)
    shape.push_back(data_shape[narrow<size_t>(i)]);

  // When bits==4 and data is stored as uint8_t, each element has two int4 values.
  // The shape in the onnx model reflects that by having the last dimension be half the number of values.
  // Example: For a true data size of 2000x3072, the packed uint8 tensor has shape 2000x1536.
  // However the outputs still need to be of size 2000x3072. Therefore we x2 the last dimension here.
  uint32_t components = 1;
  if constexpr (std::is_same_v<T1, uint8_t>) {
    components = 8 / static_cast<int>(bits_);
    if (components > 1) {
      // To handle quantize_axis that is not the last dimension:
      //  shape[(p.quantize_axis < p.gather_axis) ? p.quantize_axis : p.quantize_axis + indices_rank - 1] *= components;
      // Since we constraint the last dimension to be the quantize_axis, we can simplify it to:
      shape.back() *= components;
    }
  }

  p.output_tensor = context->Output(0, TensorShape(std::move(shape)));

  // validate quantization parameters
  const auto& scales_shape = p.scales_tensor->Shape();
  ORT_RETURN_IF_NOT(data_shape.NumDimensions() == scales_shape.NumDimensions(),
                    "data and scales must have the same rank.");

  const int64_t quantize_axis_dim_raw = data_shape[narrow<size_t>(p.quantize_axis)];
  p.effective_block_size = block_size_;
  if constexpr (IsFpQuantizedV<T1>) {
    if (block_size_ == 0) {
      p.effective_block_size = std::max<int64_t>(quantize_axis_dim_raw, 1);
    }
  }

  const size_t rank = data_shape.NumDimensions();
  if constexpr (IsFpQuantizedV<T1>) {
    p.data_strides.assign(rank, 1);
    p.scale_strides.assign(rank, 1);
    p.scale_broadcast_axis.assign(rank, false);
  }
  for (size_t i = 0; i < rank; ++i) {
    bool dims_match;
    if (i == static_cast<size_t>(p.quantize_axis)) {
      const int64_t num_blocks =
          (data_shape[i] * components + p.effective_block_size - 1) / p.effective_block_size;
      dims_match = num_blocks == scales_shape[i];
    } else {
      dims_match = data_shape[i] == scales_shape[i];
    }
    // On axes other than quantize_axis, a scales dimension of 1 broadcasts along that axis (e.g. a
    // single scale shared by every row, including a single global per-tensor scale). Only applicable
    // to FP8/FP4 data.
    bool broadcastable = IsFpQuantizedV<T1> && i != static_cast<size_t>(p.quantize_axis) && scales_shape[i] == 1;
    ORT_RETURN_IF_NOT(dims_match || broadcastable, "data and scales do not match shapes.");
    if constexpr (IsFpQuantizedV<T1>) {
      p.scale_broadcast_axis[i] = broadcastable && !dims_match;
    }
  }

  if constexpr (IsFpQuantizedV<T1>) {
    // Compute row-major strides from the trailing axis inward.
    for (size_t i = rank; i-- > 0;) {
      if (i + 1 < rank) {
        p.data_strides[i] = p.data_strides[i + 1] * data_shape[i + 1];
        p.scale_strides[i] = p.scale_strides[i + 1] * scales_shape[i + 1];
      }
    }
  }

  if (p.zero_points_tensor) {
    const auto& zero_points_shape = p.zero_points_tensor->Shape();
    ORT_RETURN_IF_NOT(scales_shape.NumDimensions() == zero_points_shape.NumDimensions(),
                      "scales and zero_points must have the same rank.");
    for (size_t i = 0; i < scales_shape.NumDimensions(); ++i) {
      if (components > 1 && i == static_cast<size_t>(p.quantize_axis)) {
        // For uint8_t with bits=4, zero points is stored as 2 components per byte.
        ORT_RETURN_IF_NOT((scales_shape[i] + components - 1) / components == zero_points_shape[i],
                          "scales and zero_points shape does not match.");
      } else {
        ORT_RETURN_IF_NOT(scales_shape[i] == zero_points_shape[i],
                          "scales and zero_points must have the same shape.");
      }
    }
  }

  return Status::OK();
}

template <typename T1, typename Tind>
template <typename T2>
Status GatherBlockQuantized<T1, Tind>::CopyDataAndDequantize(const T1* data_ptr,
                                                             const Tind* indices_ptr,
                                                             const T2* scales_ptr,
                                                             const T1* zero_points_ptr,
                                                             T2* output_ptr,
                                                             const int64_t gather_M,
                                                             const int64_t gather_N,
                                                             const int64_t gather_axis_dim,
                                                             const int64_t gather_block,
                                                             const int64_t quantize_axis_dim,
                                                             const int64_t quantize_N,
                                                             const Prepare& p,
                                                             concurrency::ThreadPool* tp) const {
  auto data_full_block = gather_axis_dim * gather_block;

  if constexpr (IsFpQuantizedV<T1>) {
    // FP8/FP4: symmetric dequantization (no zero point), with scales possibly broadcast along axes
    // other than quantize_axis. The scale index is derived by decomposing the flat data index into
    // per-axis indices via `p.data_strides`, then mapping each axis to its contribution in `scales`:
    // block-index division at quantize_axis, 0 for a broadcast axis, otherwise the axis index as-is.
    const int64_t rank = static_cast<int64_t>(p.data_strides.size());
    const int64_t effective_block_size = p.effective_block_size;
    const int64_t quantize_axis = p.quantize_axis;

    auto lambda = [&](int64_t gather_MN_idx) {
      int64_t gather_M_idx = gather_MN_idx / gather_N;
      int64_t gather_N_idx = gather_MN_idx % gather_N;

      int64_t indices_val = static_cast<int64_t>(indices_ptr[gather_N_idx]);
      ORT_ENFORCE(indices_val >= -gather_axis_dim && indices_val < gather_axis_dim,
                  "indices element out of data bounds, idx=", indices_val,
                  " must be within the inclusive range [", -gather_axis_dim, ",", gather_axis_dim - 1, "]");

      indices_val = indices_val < 0 ? indices_val + gather_axis_dim : indices_val;
      int64_t output_idx_base = gather_MN_idx * gather_block;
      int64_t data_idx_base = gather_M_idx * data_full_block + indices_val * gather_block;

      int64_t output_idx = output_idx_base;
      int64_t data_idx = data_idx_base;
      for (int64_t i = 0; i < gather_block; ++i, ++output_idx, ++data_idx) {
        const float data_val = DequantizedFpElem(data_ptr, data_idx);

        int64_t remaining = data_idx;
        int64_t scale_idx = 0;
        for (int64_t axis = 0; axis < rank; ++axis) {
          const size_t axis_u = narrow<size_t>(axis);
          int64_t axis_idx = remaining / p.data_strides[axis_u];
          remaining -= axis_idx * p.data_strides[axis_u];
          int64_t contribution = axis == quantize_axis
                                     ? axis_idx / effective_block_size
                                     : (p.scale_broadcast_axis[axis_u] ? 0 : axis_idx);
          scale_idx += contribution * p.scale_strides[axis_u];
        }
        const float scale_val = static_cast<float>(scales_ptr[scale_idx]);

        output_ptr[output_idx] = static_cast<T2>(data_val * scale_val);
      }
    };

    concurrency::ThreadPool::TryParallelFor(
        tp,
        SafeInt<ptrdiff_t>(gather_M) * gather_N,
        static_cast<double>(gather_block * 2),
        [&lambda](ptrdiff_t first, ptrdiff_t last) {
          for (auto index = static_cast<int64_t>(first), end = static_cast<int64_t>(last);
               index < end;
               ++index) {
            lambda(index);
          }
        });

    return Status::OK();
  } else {
    auto quantize_full_block = quantize_axis_dim * quantize_N;
    auto scale_full_block = (quantize_axis_dim + block_size_ - 1) / block_size_ * quantize_N;

    auto lambda = [&](int64_t gather_MN_idx, std::unordered_map<int64_t, int64_t>& cache) {
      int64_t gather_M_idx = gather_MN_idx / gather_N;
      int64_t gather_N_idx = gather_MN_idx % gather_N;

      int64_t indices_val = static_cast<int64_t>(indices_ptr[gather_N_idx]);
      ORT_ENFORCE(indices_val >= -gather_axis_dim && indices_val < gather_axis_dim,
                  "indices element out of data bounds, idx=", indices_val,
                  " must be within the inclusive range [", -gather_axis_dim, ",", gather_axis_dim - 1, "]");

      indices_val = indices_val < 0 ? indices_val + gather_axis_dim : indices_val;
      int64_t output_idx_base = gather_MN_idx * gather_block;
      int64_t data_idx_base = gather_M_idx * data_full_block + indices_val * gather_block;

      if (auto it = cache.find(data_idx_base); it != cache.end()) {
        int64_t output_src_idx = it->second;
        memcpy(output_ptr + output_idx_base, output_ptr + output_src_idx, narrow<size_t>(gather_block * sizeof(T2)));
        return;
      }

      if constexpr (std::is_same_v<T1, uint8_t>) {
        // Fast path: uint8_t-packed data (bits_ == 2, 4, or 8). Since quantize_axis is enforced to be
        // the last dimension for uint8_t data, quantize_N == 1, so scale/zero-point only change every
        // `block_size_` elements (or at a quantize-axis-dim boundary, whichever comes first) along the
        // contiguous `gather_block` run being produced here. Rather than recomputing scale_idx/zp_val
        // and doing a scalar multiply-subtract per element, batch each constant-scale run through
        // MlasDequantizeLinear, which is SIMD-optimized (AVX2/AVX512/NEON) for uint8_t input.
        uint8_t unpacked[kUint8DequantBatch];
        float dequantized[kUint8DequantBatch];

        int64_t output_idx = output_idx_base;
        int64_t data_idx = data_idx_base;
        int64_t i = 0;
        while (i < gather_block) {
          const int64_t y = data_idx % quantize_full_block;  // quantize_N == 1, so z == 0 always.
          const int64_t scale_idx = data_idx / quantize_full_block * scale_full_block + y / block_size_;
          // Bound the run so it neither crosses into the next scale block nor past the end of the
          // current quantize-axis span (a partial last block when quantize_axis_dim isn't a multiple
          // of block_size_), then cap it to the SIMD batch buffer size.
          int64_t run_len = std::min(block_size_ - y % block_size_, quantize_full_block - y);
          run_len = std::min({run_len, gather_block - i, kUint8DequantBatch});

          const auto scale_val = static_cast<float>(scales_ptr[scale_idx]);
          int32_t zp_val;
          if (zero_points_ptr) {
            // For uint8 we enforce quantize_axis == last dim, which makes quantize_N == 1
            // and scale_full_block == scale_qaxis_dim. Zero points are packed only along
            // the quantize axis, so the packed byte must be addressed using the scale row
            // index and the within-row quantize-axis index, not the flat scale_idx; the
            // latter crosses row boundaries when scale_qaxis_dim is not a multiple of the
            // packing factor.
            const int64_t scale_qaxis_dim = scale_full_block;
            const int64_t scale_row = scale_idx / scale_qaxis_dim;
            const int64_t q_in_row = scale_idx % scale_qaxis_dim;
            if (bits_ == 2) {
              const int64_t packed_zp_qaxis_dim = (scale_qaxis_dim + 3) / 4;
              const int64_t byte_idx = scale_row * packed_zp_qaxis_dim + (q_in_row >> 2);
              const int shift = static_cast<int>((q_in_row & 3) * 2);
              zp_val = static_cast<int32_t>((zero_points_ptr[byte_idx] >> shift) & 0x03);
            } else if (bits_ == 4) {
              const int64_t packed_zp_qaxis_dim = (scale_qaxis_dim + 1) / 2;
              const int64_t byte_idx = scale_row * packed_zp_qaxis_dim + (q_in_row >> 1);
              uint8_t packed = zero_points_ptr[byte_idx];
              zp_val = static_cast<int32_t>((q_in_row & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F));
            } else {  // bits_ == 8
              zp_val = static_cast<int32_t>(zero_points_ptr[scale_idx]);
            }
          } else {
            // Default zero point is 2^(bits-1): 2 for 2-bit, 8 for 4-bit, 128 for 8-bit.
            zp_val = 1 << (static_cast<int>(bits_) - 1);
          }

          UnpackUint8Elements(data_ptr, data_idx, run_len, bits_, unpacked);
          if constexpr (std::is_same_v<T2, float>) {
            MlasDequantizeLinear(unpacked, output_ptr + output_idx, narrow<size_t>(run_len), scale_val,
                                 static_cast<uint8_t>(zp_val));
          } else {
            MlasDequantizeLinear(unpacked, dequantized, narrow<size_t>(run_len), scale_val,
                                 static_cast<uint8_t>(zp_val));
            MlasConvertFloatToHalfBuffer(dequantized, output_ptr + output_idx, narrow<size_t>(run_len));
          }

          i += run_len;
          output_idx += run_len;
          data_idx += run_len;
        }
      } else {
        int64_t output_idx = output_idx_base;
        int64_t data_idx = data_idx_base;
        for (int64_t i = 0; i < gather_block; ++i, ++output_idx, ++data_idx) {
          int32_t data_val = Get4BitElement(data_ptr, data_idx);

          int64_t x = data_idx / quantize_full_block;
          int64_t y = data_idx % quantize_full_block / quantize_N;
          int64_t z = data_idx % quantize_N;
          int64_t scale_idx = x * scale_full_block + y / block_size_ * quantize_N + z;
          auto scale_val = static_cast<float>(scales_ptr[scale_idx]);
          int32_t zp_val = zero_points_ptr
                               ? static_cast<int32_t>(zero_points_ptr[scale_idx >> 1].GetElem(narrow<size_t>(scale_idx & 1)))
                               : 0;

          output_ptr[output_idx] = static_cast<T2>(static_cast<float>(data_val - zp_val) * scale_val);
        }
      }

      cache[data_idx_base] = output_idx_base;
    };

    concurrency::ThreadPool::TryParallelFor(
        tp,
        SafeInt<ptrdiff_t>(gather_M) * gather_N,
        static_cast<double>(gather_block * 3),
        [&lambda](ptrdiff_t first, ptrdiff_t last) {
          // cache dequantized gather_block. Key is data_idx_base. Value is the output_idx_base.
          // cache is per thread to avoid contention.
          std::unordered_map<int64_t, int64_t> cache;

          for (auto index = static_cast<int64_t>(first), end = static_cast<int64_t>(last);
               index < end;
               ++index) {
            lambda(index, cache);
          }
        });

    return Status::OK();
  }
}

template <typename T1, typename Tind>
Status GatherBlockQuantized<T1, Tind>::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));
  int64_t components = std::is_same_v<T1, uint8_t> ? (8 / static_cast<int>(bits_)) : 1;
  const auto& data_shape = p.data_tensor->Shape();
  // re-shape the data tensor to [gather_M, gather_axis_dim, gather_block]
  // re-shape the indices tensor to [gather_N]
  // re-shape the output tensor to [gather_M, gather_N, gather_block]
  // For an index i in the output tensor:
  //  1> the output block index is blk_i = i / gather_block, block element index is blk_ele_i = i % gather_block,
  //  2> block is picked from data based on value from indices: axis_i = indices[blk_i % gather_N],
  //  3> get the corresponding block in data tensor: data_blk = data[blk_i / gather_N, axis_i, :],
  //  4> pick the element from the block: value_i = data_blk[blk_ele_i]
  const int64_t gather_block = data_shape.SizeFromDimension(SafeInt<size_t>(p.gather_axis) + 1) * components;
  const int64_t gather_axis_dim = data_shape[narrow<size_t>(p.gather_axis)];
  const int64_t gather_M = data_shape.SizeToDimension(narrow<size_t>(p.gather_axis));
  const int64_t gather_N = p.indices_tensor->Shape().Size();
  // re-shape the data tensor to [quantize_M, quantize_axis_dim, quantize_N]
  // For an index i in the output tensor:
  //  1> based on previous comment, corresponding data index is (blk_i / gather_N, axis_i, blk_ele_i)
  //  2> flatten the data index:
  //     data_i = blk_i / gather_N * gather_axis_dim * gather_block + axis_i * gather_block + blk_ele_i
  //  3> map data_i to quantize shape: (x, y, z) =
  //     (data_i / (quantize_axis_dim * quantize_N),
  //      data_i % (quantize_axis_dim * quantize_N) / quantize_N,
  //      data_i % quantize_N)
  //  4> get scale index: (x, y / block_size_, z)
  const int64_t quantize_axis_dim = data_shape[narrow<size_t>(p.quantize_axis)] * components;
  const int64_t quantize_N = data_shape.SizeFromDimension(SafeInt<size_t>(p.quantize_axis) + 1);

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto* data_ptr = p.data_tensor->template Data<T1>();
  const auto* indices_ptr = p.indices_tensor->template Data<Tind>();
  const auto* zero_points_ptr = p.zero_points_tensor ? p.zero_points_tensor->template Data<T1>() : nullptr;
  const auto dequantized_type = p.scales_tensor->GetElementType();

  if (dequantized_type == ONNX_NAMESPACE::TensorProto::FLOAT) {
    const auto* scales_ptr = p.scales_tensor->template Data<float>();
    auto* output_ptr = p.output_tensor->template MutableData<float>();

    return CopyDataAndDequantize<float>(data_ptr, indices_ptr, scales_ptr, zero_points_ptr,
                                        output_ptr, gather_M, gather_N, gather_axis_dim, gather_block,
                                        quantize_axis_dim, quantize_N, p,
                                        tp);
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    const auto* scales_ptr = p.scales_tensor->template Data<MLFloat16>();
    auto* output_ptr = p.output_tensor->template MutableData<MLFloat16>();

    return CopyDataAndDequantize<MLFloat16>(data_ptr, indices_ptr, scales_ptr, zero_points_ptr,
                                            output_ptr, gather_M, gather_N, gather_axis_dim, gather_block,
                                            quantize_axis_dim, quantize_N, p,
                                            tp);
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
    ORT_THROW("DequantizeLinear into BFLOAT16 is not implemented yet.");
  } else {
    ORT_THROW("Unsupported dequantized type: ", dequantized_type);
  }
}

#define REGISTER_GATHERBLOCKQUANTIZED(T1, Tind)                                                                   \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                                                              \
      GatherBlockQuantized,                                                                                       \
      kMSDomain, 1,                                                                                               \
      T1, Tind,                                                                                                   \
      kCpuExecutionProvider,                                                                                      \
      KernelDefBuilder()                                                                                          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                                                \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()}) \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>()),                                           \
      GatherBlockQuantized<T1, Tind>);

REGISTER_GATHERBLOCKQUANTIZED(uint8_t, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(uint8_t, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, int64_t);

#if !defined(DISABLE_FLOAT8_TYPES)
REGISTER_GATHERBLOCKQUANTIZED(Float8E4M3FN, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float8E4M3FN, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Float8E4M3FNUZ, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float8E4M3FNUZ, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Float8E5M2, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float8E5M2, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Float8E5M2FNUZ, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float8E5M2FNUZ, int64_t);
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES)
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, int64_t);
#endif  // !defined(DISABLE_FLOAT4_TYPES)

}  // namespace contrib
}  // namespace onnxruntime
