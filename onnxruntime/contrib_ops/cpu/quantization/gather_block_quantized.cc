// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/framework/float16.h"
#include "core/framework/int4.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/providers/common.h"

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

    ORT_ENFORCE(block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0,
                "'block_size' must be a power of 2 and not less than 16.");

    constexpr int64_t default_bits = 4;
    info.GetAttrOrDefault("bits", &bits_, default_bits);
    ORT_ENFORCE(bits_ == 4 || bits_ == 8, "GatherBlockQuantized only support bits==4 or 8");

    ORT_ENFORCE(block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0,
                "'block_size' must be 2's power and not less than 16.");
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
  if constexpr (std::is_same_v<T1, uint8_t>) {
    ORT_RETURN_IF_NOT(p.gather_axis == 0, "For uint8_t data, gather_axis must be 0.");
    ORT_RETURN_IF_NOT(p.quantize_axis == static_cast<int64_t>(data_rank) - 1, "For uint8_t data, quantize_axis must be the last dimension.");
    ORT_RETURN_IF_NOT(p.gather_axis != p.quantize_axis, "gather_axis and quantize_axis must not be the same.");
  }

  const auto& indices_shape = p.indices_tensor->Shape();
  const auto indices_rank = indices_shape.NumDimensions();

  std::vector<int64_t> shape;
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
  for (size_t i = 0; i < data_shape.NumDimensions(); ++i) {
    ORT_RETURN_IF_NOT(i == static_cast<size_t>(p.quantize_axis)
                          ? (data_shape[i] * components + block_size_ - 1) / block_size_ == scales_shape[i]
                          : data_shape[i] == scales_shape[i],
                      "data and scales do not match shapes.");
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
                                                             concurrency::ThreadPool* tp) const {
  auto data_full_block = gather_axis_dim * gather_block;
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

    // TODO(fajin): use SIMD
    int64_t output_idx = output_idx_base;
    int64_t data_idx = data_idx_base;
    for (int64_t i = 0; i < gather_block; ++i, ++output_idx, ++data_idx) {
      int32_t data_val;
      if constexpr (!std::is_same_v<T1, uint8_t>) {
        data_val = Get4BitElement(data_ptr, data_idx);
      } else {  // unit8_t
        if (bits_ == 4) {
          data_val = Get4BitElement(data_ptr, data_idx);
        } else {  // buts_ == 8
          data_val = static_cast<int32_t>(data_ptr[data_idx]);
        }
      }

      int64_t x = data_idx / quantize_full_block;
      int64_t y = data_idx % quantize_full_block / quantize_N;
      int64_t z = data_idx % quantize_N;
      int64_t scale_idx = x * scale_full_block + y / block_size_ * quantize_N + z;
      auto scale_val = static_cast<float>(scales_ptr[scale_idx]);
      int32_t zp_val;

      if constexpr (std::is_same_v<T1, uint8_t>) {
        if (zero_points_ptr) {
          if (bits_ == 4) {
            uint8_t packed = zero_points_ptr[scale_idx >> 1];
            if (scale_idx & 1) {
              zp_val = static_cast<int32_t>((packed >> 4) & 0x0F);
            } else {
              zp_val = static_cast<int32_t>(packed & 0x0F);
            }
          } else {  // bits_ == 8
            zp_val = static_cast<int32_t>(zero_points_ptr[scale_idx]);
          }
        } else {
          const int32_t default_zero_point = bits_ == 4 ? 8 : 128;
          zp_val = default_zero_point;
        }
      } else {
        zp_val = zero_points_ptr
                     ? static_cast<int32_t>(zero_points_ptr[scale_idx >> 1].GetElem(narrow<size_t>(scale_idx & 1)))
                     : 0;
      }

      output_ptr[output_idx] = static_cast<T2>(static_cast<float>(data_val - zp_val) * scale_val);
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
                                        quantize_axis_dim, quantize_N,
                                        tp);
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    const auto* scales_ptr = p.scales_tensor->template Data<MLFloat16>();
    auto* output_ptr = p.output_tensor->template MutableData<MLFloat16>();

    return CopyDataAndDequantize<MLFloat16>(data_ptr, indices_ptr, scales_ptr, zero_points_ptr,
                                            output_ptr, gather_M, gather_N, gather_axis_dim, gather_block,
                                            quantize_axis_dim, quantize_N,
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

}  // namespace contrib
}  // namespace onnxruntime
