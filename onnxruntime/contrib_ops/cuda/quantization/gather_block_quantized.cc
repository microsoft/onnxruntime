// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/gather_block_quantized.h"
#include "contrib_ops/cuda/quantization/gather_block_quantized.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

#define REGISTER_GATHERBLOCKQUANTIZED(T1, T2, Tind)                     \
  ONNX_OPERATOR_THREE_TYPED_KERNEL_EX(                                  \
      GatherBlockQuantized,                                             \
      kMSDomain, 1,                                                     \
      T1, T2, Tind,                                                     \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())      \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())      \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>()), \
      GatherBlockQuantized<T1, T2, Tind>);

REGISTER_GATHERBLOCKQUANTIZED(uint8_t, float, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(uint8_t, float, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, float, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, float, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, float, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, float, int64_t);

REGISTER_GATHERBLOCKQUANTIZED(uint8_t, MLFloat16, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(uint8_t, MLFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, MLFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, MLFloat16, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, MLFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, MLFloat16, int64_t);

REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, BFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(UInt4x2, BFloat16, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(uint8_t, BFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(uint8_t, BFloat16, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, BFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Int4x2, BFloat16, int64_t);

template <typename T1, typename T2, typename Tind>
GatherBlockQuantized<T1, T2, Tind>::GatherBlockQuantized(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr("bits", &bits_).IsOK());

  block_size_ = info.GetAttrOrDefault<int64_t>("block_size", 0);
  gather_axis_ = info.GetAttrOrDefault<int64_t>("gather_axis", 0);
  quantize_axis_ = info.GetAttrOrDefault<int64_t>("quantize_axis", 0);

  // If block size is set, it has to be no smaller than 16 and must be power of 2
  // block_size_ & (block_size_ - 1) == 0 checks if block_size_ only has 1 bit set
  ORT_ENFORCE(block_size_ == 0 || (block_size_ >= 16 && ((block_size_ & (block_size_ - 1)) == 0)));
}

template <typename T1, typename T2, typename Tind>
Status GatherBlockQuantized<T1, T2, Tind>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* data = ctx->Input<Tensor>(0);
  const Tensor* indices = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);

  auto data_shape = data->Shape().GetDims();
  auto data_rank = data->Shape().NumDimensions();

  auto indices_shape = indices->Shape().GetDims();
  auto indices_rank = indices->Shape().NumDimensions();

  ORT_ENFORCE(quantize_axis_ == static_cast<int64_t>(data_rank) - 1);

  TensorShapeVector output_shape;
  output_shape.reserve(data_rank - 1 + indices_rank);

  // Dimension after gather axis
  int64_t after_gather_dim = 1;

  // Dimension of indices
  int64_t ind_dim = 1;

  // 1) dims before gather_axis
  for (int64_t i = 0; i < gather_axis_; ++i) {
    output_shape.push_back(data_shape[i]);
  }

  // 2) all of indices.shape
  for (auto dim : indices_shape) {
    output_shape.push_back(dim);
    ind_dim *= dim;
  }

  // 3) dims after gather_axis
  for (int64_t i = gather_axis_ + 1; i < static_cast<int64_t>(data_rank); ++i) {
    output_shape.push_back(data_shape[i]);
    after_gather_dim *= data_shape[i];
  }

  // Special int4‐in‐uint8 packing tweak: expand the last dim by components
  if constexpr (std::is_same_v<T1, uint8_t>) {
    uint32_t components = 8 / static_cast<int>(bits_);
    if (components > 1) {
      output_shape.back() *= components;
    }
  }

  Tensor* output = ctx->Output(0, TensorShape(output_shape));

  int64_t N = 1;
  for (auto dim : output_shape) {
    N *= dim;
  }

  const auto* data_ptr = data->Data<T1>();
  const auto* indices_ptr = indices->Data<Tind>();
  const T1* zero_points_ptr = nullptr;
  if (zero_points != nullptr) {
    zero_points_ptr = zero_points->Data<T1>();
  }

  GatherBlockQuantizedParam param;
  param.stream = Stream(ctx);
  param.after_gather_dim = after_gather_dim;
  param.gather_axis_dim = data_shape[gather_axis_];
  param.ind_dim = ind_dim;
  param.bits = bits_;
  param.block_size = block_size_;
  param.gather_axis = gather_axis_;
  param.N = N;

  const auto dequantized_type = scales->GetElementType();
  if (dequantized_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const auto* scales_ptr = static_cast<const float*>(scales->DataRaw());
    auto* output_ptr = static_cast<float*>(output->MutableDataRaw());
    LaunchGatherBlockQuantizedKernel(data_ptr, indices_ptr, scales_ptr, zero_points_ptr, output_ptr, param);
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    const auto* scales_ptr = static_cast<const half*>(scales->DataRaw());
    auto* output_ptr = static_cast<half*>(output->MutableDataRaw());
    LaunchGatherBlockQuantizedKernel(data_ptr, indices_ptr, scales_ptr, zero_points_ptr, output_ptr, param);
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    const auto* scales_ptr = static_cast<const BFloat16*>(scales->DataRaw());
    auto* output_ptr = static_cast<BFloat16*>(output->MutableDataRaw());
    LaunchGatherBlockQuantizedKernel(data_ptr, indices_ptr, scales_ptr, zero_points_ptr, output_ptr, param);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
