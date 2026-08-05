// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/quantization/gather_block_quantized.h"
#include "contrib_ops/rocm/quantization/gather_block_quantized.cuh"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

// ---------------------------------------------------------------------------
// Kernel registration
// ---------------------------------------------------------------------------
#define REGISTER_GBQ(T1, T2, Tind)                                                    \
  ONNX_OPERATOR_THREE_TYPED_KERNEL_EX(                                                \
      GatherBlockQuantized,                                                           \
      kMSDomain, 1,                                                                   \
      T1, T2, Tind,                                                                   \
      kRocmExecutionProvider,                                                         \
      (*KernelDefBuilder::Create())                                                   \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())                    \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>()),               \
      GatherBlockQuantized<T1, T2, Tind>);

// uint8_t packed weight (bits < 8 packed into bytes)
REGISTER_GBQ(uint8_t, float,     int32_t)
REGISTER_GBQ(uint8_t, float,     int64_t)
REGISTER_GBQ(uint8_t, MLFloat16, int32_t)
REGISTER_GBQ(uint8_t, MLFloat16, int64_t)

// UInt4x2 (unsigned 4-bit pairs, native ORT type)
REGISTER_GBQ(UInt4x2, float,     int32_t)
REGISTER_GBQ(UInt4x2, float,     int64_t)
REGISTER_GBQ(UInt4x2, MLFloat16, int32_t)
REGISTER_GBQ(UInt4x2, MLFloat16, int64_t)

// Int4x2 (signed 4-bit pairs, native ORT type)
REGISTER_GBQ(Int4x2, float,     int32_t)
REGISTER_GBQ(Int4x2, float,     int64_t)
REGISTER_GBQ(Int4x2, MLFloat16, int32_t)
REGISTER_GBQ(Int4x2, MLFloat16, int64_t)

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
template <typename T1, typename T2, typename Tind>
GatherBlockQuantized<T1, T2, Tind>::GatherBlockQuantized(const OpKernelInfo& info)
    : RocmKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("bits", &bits_).IsOK());
  block_size_    = info.GetAttrOrDefault<int64_t>("block_size",    0);
  gather_axis_   = info.GetAttrOrDefault<int64_t>("gather_axis",   0);
  quantize_axis_ = info.GetAttrOrDefault<int64_t>("quantize_axis", 0);

  ORT_ENFORCE(block_size_ == 0 ||
              (block_size_ >= 16 && (block_size_ & (block_size_ - 1)) == 0),
              "block_size must be 0 or a power-of-2 >= 16");
}

// ---------------------------------------------------------------------------
// ComputeInternal
// ---------------------------------------------------------------------------
template <typename T1, typename T2, typename Tind>
Status GatherBlockQuantized<T1, T2, Tind>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* data     = ctx->Input<Tensor>(0);
  const Tensor* indices  = ctx->Input<Tensor>(1);
  const Tensor* scales   = ctx->Input<Tensor>(2);
  const Tensor* zp_tensor = ctx->Input<Tensor>(3);  // optional

  const auto data_shape    = data->Shape().GetDims();
  const int64_t data_rank  = static_cast<int64_t>(data->Shape().NumDimensions());
  const auto idx_shape     = indices->Shape().GetDims();
  const int64_t idx_rank   = static_cast<int64_t>(indices->Shape().NumDimensions());

  // quantize_axis must be the last dimension (the standard exporter layout).
  ORT_RETURN_IF_NOT(quantize_axis_ == data_rank - 1,
                    "GatherBlockQuantized ROCm: quantize_axis must be the last dimension.");

  // Build output shape:  data[0..gather_axis) ++ indices.shape ++ data(gather_axis+1..]
  TensorShapeVector output_shape;
  output_shape.reserve(static_cast<size_t>(data_rank - 1 + idx_rank));

  int64_t after_gather_dim = 1;
  int64_t ind_dim = 1;

  for (int64_t i = 0; i < gather_axis_; ++i) {
    output_shape.push_back(data_shape[i]);
  }
  for (auto d : idx_shape) {
    output_shape.push_back(d);
    ind_dim *= d;
  }
  for (int64_t i = gather_axis_ + 1; i < data_rank; ++i) {
    output_shape.push_back(data_shape[i]);
    after_gather_dim *= data_shape[i];
  }

  // For packed uint8_t (bits < 8), the last dim expands by (8/bits) in output.
  if constexpr (std::is_same_v<T1, uint8_t>) {
    const int64_t components = 8 / bits_;
    if (components > 1) {
      output_shape.back() *= components;
    }
  }

  Tensor* output = ctx->Output(0, TensorShape(output_shape));

  int64_t N = 1;
  for (auto d : output_shape) N *= d;
  if (N == 0) return Status::OK();

  // after_gather_dim for packed uint8_t must reflect the unpacked element count.
  int64_t after_gather_dim_unpacked = after_gather_dim;
  if constexpr (std::is_same_v<T1, uint8_t>) {
    const int64_t components = 8 / bits_;
    if (components > 1) after_gather_dim_unpacked *= components;
  }

  GatherBlockQuantizedParam param;
  param.stream           = Stream(ctx);
  param.after_gather_dim = after_gather_dim_unpacked;
  param.gather_axis_dim  = data_shape[gather_axis_];
  param.ind_dim          = ind_dim;
  param.bits             = bits_;
  param.block_size       = block_size_;
  param.gather_axis      = gather_axis_;
  param.N                = N;

  const T1* data_ptr       = data->Data<T1>();
  const Tind* indices_ptr  = indices->Data<Tind>();
  const T1* zp_ptr         = (zp_tensor != nullptr) ? zp_tensor->Data<T1>() : nullptr;

  using HipT2 = typename ToHipType<T2>::MappedType;
  const HipT2* scales_ptr = reinterpret_cast<const HipT2*>(scales->DataRaw());
  HipT2* output_ptr       = reinterpret_cast<HipT2*>(output->MutableDataRaw());

  LaunchGatherBlockQuantizedKernel(
      data_ptr, indices_ptr, scales_ptr, zp_ptr, output_ptr, param);

  HIP_RETURN_IF_ERROR(hipGetLastError());
  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
