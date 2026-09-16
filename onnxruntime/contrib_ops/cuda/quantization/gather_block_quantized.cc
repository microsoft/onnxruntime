// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/common/logging/logging.h"
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

#if !defined(DISABLE_FLOAT8_TYPES)
#define REGISTER_GATHERBLOCKQUANTIZED_FP8(T1)            \
  REGISTER_GATHERBLOCKQUANTIZED(T1, float, int32_t);     \
  REGISTER_GATHERBLOCKQUANTIZED(T1, float, int64_t);     \
  REGISTER_GATHERBLOCKQUANTIZED(T1, MLFloat16, int32_t); \
  REGISTER_GATHERBLOCKQUANTIZED(T1, MLFloat16, int64_t); \
  REGISTER_GATHERBLOCKQUANTIZED(T1, BFloat16, int32_t);  \
  REGISTER_GATHERBLOCKQUANTIZED(T1, BFloat16, int64_t);

REGISTER_GATHERBLOCKQUANTIZED_FP8(Float8E4M3FN);
REGISTER_GATHERBLOCKQUANTIZED_FP8(Float8E4M3FNUZ);
REGISTER_GATHERBLOCKQUANTIZED_FP8(Float8E5M2);
REGISTER_GATHERBLOCKQUANTIZED_FP8(Float8E5M2FNUZ);
#undef REGISTER_GATHERBLOCKQUANTIZED_FP8
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES)
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, float, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, float, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, MLFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, MLFloat16, int64_t);
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, BFloat16, int32_t);
REGISTER_GATHERBLOCKQUANTIZED(Float4E2M1x2, BFloat16, int64_t);
#endif  // !defined(DISABLE_FLOAT4_TYPES)

template <typename T1, typename T2, typename Tind>
GatherBlockQuantized<T1, T2, Tind>::GatherBlockQuantized(const OpKernelInfo& info)
    : CudaKernel(info), direct_host_data_(false), data_is_constant_(false) {
  if constexpr (IsFpQuantizedV<T1>) {
    bits_ = 0;  // Not applicable for FP8/FP4 data.
  } else {
    ORT_ENFORCE(info.GetAttr("bits", &bits_).IsOK());
  }

  block_size_ = info.GetAttrOrDefault<int64_t>("block_size", 128);
  gather_axis_ = info.GetAttrOrDefault<int64_t>("gather_axis", 0);
  quantize_axis_ = info.GetAttrOrDefault<int64_t>("quantize_axis", 1);

  const Tensor* constant_data = nullptr;
  data_is_constant_ = info.TryGetConstantInput(0, &constant_data);

  int pageable_memory_access = 0;
  int uses_host_page_tables = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10020
  const bool attributes_available =
      cudaDeviceGetAttribute(&pageable_memory_access, cudaDevAttrPageableMemoryAccess, GetDeviceId()) == cudaSuccess &&
      cudaDeviceGetAttribute(&uses_host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                             GetDeviceId()) == cudaSuccess;
  if (!attributes_available) {
    pageable_memory_access = 0;
    uses_host_page_tables = 0;
    cudaGetLastError();
  }
#endif

  const bool option_enabled = EnableHostPageableGather();
  direct_host_data_ =
      SelectGatherBlockQuantizedDataPolicy(option_enabled, pageable_memory_access != 0,
                                           uses_host_page_tables != 0, IsFp8QuantizedV<T1>,
                                           data_is_constant_) ==
      GatherBlockQuantizedDataPolicy::DirectHost;
  if (option_enabled && IsFp8QuantizedV<T1> && !direct_host_data_) {
    LOGS_DEFAULT(WARNING)
        << "enable_host_pageable_gather was requested, but direct host-pageable GatherBlockQuantized "
           "access is unavailable because input 0 is not a constant initializer or the CUDA device lacks pageable "
           "memory access through host page tables. Using the standard CUDA input path.";
  }

  // If block size is set, it has to be no smaller than 16 and must be power of 2.
  // block_size_ & (block_size_ - 1) == 0 checks if block_size_ only has 1 bit set.
  // block_size_ == 0 is only valid for FP8/FP4 data, meaning the whole quantize_axis dimension
  // is a single block (one scale per row).
  if (block_size_ == 0) {
    ORT_ENFORCE(IsFpQuantizedV<T1>, "block_size must be a power of 2 and not smaller than 16.");
  } else {
    ORT_ENFORCE(block_size_ >= 16 && ((block_size_ & (block_size_ - 1)) == 0));
  }
}

template <typename T1, typename T2, typename Tind>
Status GatherBlockQuantized<T1, T2, Tind>::CreateDeviceCopy(const Tensor& tensor, AllocatorPtr alloc) const {
  ORT_RETURN_IF_NOT(tensor.Location().device.Type() == OrtDevice::CPU,
                    "GatherBlockQuantized input 0 must reside in CPU memory.");

  const size_t bytes = tensor.SizeInBytes();
  if (bytes == 0) {
    data_shape_.assign(tensor.Shape().GetDims().begin(), tensor.Shape().GetDims().end());
    device_data_.reset();
    return Status::OK();
  }

  auto device_data = IAllocator::MakeUniquePtr<void>(alloc, bytes);
  ORT_RETURN_IF_NOT(device_data != nullptr, "Failed to allocate persistent CUDA storage for GatherBlockQuantized.");
  CUDA_RETURN_IF_ERROR(cudaMemcpy(device_data.get(), tensor.DataRaw(), bytes, cudaMemcpyHostToDevice));
  data_shape_.assign(tensor.Shape().GetDims().begin(), tensor.Shape().GetDims().end());
  device_data_ = std::move(device_data);
  return Status::OK();
}

template <typename T1, typename T2, typename Tind>
Status GatherBlockQuantized<T1, T2, Tind>::PrePack(
    const Tensor& tensor, int input_idx, AllocatorPtr alloc,
    bool& is_packed, PrePackedWeights* prepacked_weights) {
  is_packed = false;
  if (input_idx != 0) {
    return Status::OK();
  }

  std::lock_guard<std::mutex> lock(device_data_mutex_);
  if (direct_host_data_) {
    ORT_RETURN_IF_NOT(tensor.Location().device.Type() == OrtDevice::CPU,
                      "Direct host-pageable GatherBlockQuantized requires a CPU-resident initializer.");
    direct_host_data_ptr_ = tensor.Data<T1>();
    data_shape_.assign(tensor.Shape().GetDims().begin(), tensor.Shape().GetDims().end());
  } else {
    ORT_RETURN_IF_ERROR(CreateDeviceCopy(tensor, std::move(alloc)));
  }
  is_packed = true;
  if (prepacked_weights != nullptr) {
    prepacked_weights->has_kernel_owned_packed_weights_ = true;
  }
  return Status::OK();
}

template <typename T1, typename T2, typename Tind>
Status GatherBlockQuantized<T1, T2, Tind>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* data = ctx->Input<Tensor>(0);
  const Tensor* indices = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);

  const gsl::span<const int64_t> data_shape =
      data != nullptr ? data->Shape().GetDims() : gsl::span<const int64_t>{data_shape_};
  int64_t data_rank = static_cast<int64_t>(data_shape.size());
  const int64_t gather_axis = HandleNegativeAxis(gather_axis_, data_rank);
  const int64_t quantize_axis = HandleNegativeAxis(quantize_axis_, data_rank);

  auto indices_shape = indices->Shape().GetDims();
  int64_t indices_rank = static_cast<int64_t>(indices->Shape().NumDimensions());

  ORT_ENFORCE(quantize_axis == data_rank - 1,
              "GatherBlockQuantized CUDA requires quantize_axis to be the last axis.");

  TensorShapeVector output_shape;
  output_shape.reserve(data_rank - 1 + indices_rank);

  // Dimension after gather axis
  int64_t after_gather_dim = 1;

  // Dimension of indices
  int64_t ind_dim = 1;

  // 1) dims before gather_axis
  for (int64_t i = 0; i < gather_axis; ++i) {
    output_shape.push_back(data_shape[i]);
  }

  // 2) all of indices.shape
  for (auto dim : indices_shape) {
    output_shape.push_back(dim);
    ind_dim *= dim;
  }

  // 3) dims after gather_axis
  for (int64_t i = gather_axis + 1; i < static_cast<int64_t>(data_rank); ++i) {
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
  if (N == 0) {
    return Status::OK();
  }

  const T1* data_ptr = nullptr;
  if (direct_host_data_) {
    data_ptr = direct_host_data_ptr_ != nullptr ? direct_host_data_ptr_
               : data == nullptr                ? nullptr
                                                : data->Data<T1>();
  } else if (data_is_constant_) {
    {
      std::lock_guard<std::mutex> lock(device_data_mutex_);
      if (device_data_ == nullptr && data != nullptr &&
          data->Location().device.Type() == OrtDevice::CPU && data->SizeInBytes() != 0) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(Stream(ctx), &capture_status));
        ORT_RETURN_IF_NOT(
            capture_status == cudaStreamCaptureStatusNone,
            "GatherBlockQuantized cannot initialize its persistent CUDA fallback copy during CUDA Graph capture. "
            "Enable prepacking or run an uncaptured warmup iteration before capture.");
        ORT_RETURN_IF_ERROR(CreateDeviceCopy(*data, Info().GetAllocator(OrtMemTypeDefault)));
      }
      data_ptr = device_data_ != nullptr ? static_cast<const T1*>(device_data_.get())
                 : data == nullptr       ? nullptr
                                         : data->Data<T1>();
    }
  } else {
    data_ptr = data->Data<T1>();
  }
  ORT_RETURN_IF_NOT(N == 0 || data_ptr != nullptr,
                    "GatherBlockQuantized fallback has no device-resident input 0.");
  const auto* indices_ptr = indices->Data<Tind>();
  const T1* zero_points_ptr = nullptr;
  if (zero_points != nullptr) {
    ORT_ENFORCE(!IsFpQuantizedV<T1>, "zero_points must not be provided when data is an FP8 or FP4 type.");
    zero_points_ptr = zero_points->Data<T1>();
  }

  // For packed uint8_t with bits < 8,
  // after_gather_dim has to be adjusted to match
  // the unpacked output dims for correct kernel indexing
  int64_t after_gather_dim_unpacked = after_gather_dim;
  if constexpr (std::is_same_v<T1, uint8_t>) {
    uint32_t components = 8 / static_cast<int>(bits_);
    if (components > 1) {
      after_gather_dim_unpacked *= components;
    }
  }

  // block_size_ == 0 (FP8/FP4 only) means the whole quantize_axis dimension is a single block.
  // Clamp to at least 1 so an empty (0-sized) quantize axis doesn't divide by zero below.
  int64_t effective_block_size =
      block_size_ == 0 ? std::max<int64_t>(data_shape[quantize_axis], 1) : block_size_;

  GatherBlockQuantizedParam param;
  param.stream = Stream(ctx);
  param.after_gather_dim = after_gather_dim_unpacked;
  param.gather_axis_dim = data_shape[gather_axis];
  param.ind_dim = ind_dim;
  param.bits = bits_;
  param.block_size = effective_block_size;
  param.gather_axis = gather_axis;
  param.N = N;
  param.max_blocks_per_grid = GetDeviceProp().maxGridSize[0];

  if constexpr (IsFpQuantizedV<T1>) {
    // Build a generic per-axis description of `scales` so the kernel can (a) reset the block
    // index at every quantize-axis row boundary, even when data_shape[quantize_axis_] isn't a
    // multiple of effective_block_size, and (b) support broadcasting on any individual axis
    // (scales dim == 1 while the corresponding data/block dim isn't), matching the CPU kernel.
    const auto scales_shape = scales->Shape().GetDims();
    ORT_ENFORCE(static_cast<int64_t>(scales_shape.size()) == data_rank,
                "'scales' must have the same rank as 'data'.");
    ORT_RETURN_IF_NOT(data_rank <= 8,
                      "GatherBlockQuantized CUDA supports FP8/FP4 data with rank at most 8.");

    TArray<int64_t> data_dims(static_cast<int32_t>(data_rank));
    TArray<int64_t> scale_strides(static_cast<int32_t>(data_rank));
    TArray<int64_t> scale_broadcast_axis(static_cast<int32_t>(data_rank));

    int64_t stride = 1;
    for (int64_t i = data_rank - 1; i >= 0; --i) {
      data_dims[static_cast<int32_t>(i)] = data_shape[i];

      const int64_t expected_dim = (i == quantize_axis)
                                       ? (data_shape[i] + effective_block_size - 1) / effective_block_size
                                       : data_shape[i];
      const int64_t actual_dim = scales_shape[i];
      const bool is_broadcast = i != quantize_axis && actual_dim == 1 && actual_dim != expected_dim;
      ORT_ENFORCE(is_broadcast || actual_dim == expected_dim,
                  "'scales' shape does not match 'data' shape (and is not broadcastable) at axis ", i, ".");

      scale_broadcast_axis[static_cast<int32_t>(i)] = is_broadcast ? 1 : 0;
      scale_strides[static_cast<int32_t>(i)] = stride;
      stride *= actual_dim;
    }

    param.rank = static_cast<int32_t>(data_rank);
    param.quantize_axis = quantize_axis;
    param.data_dims = data_dims;
    param.scale_strides = scale_strides;
    param.scale_broadcast_axis = scale_broadcast_axis;
  }

  const auto dequantized_type = scales->GetElementType();
  if (dequantized_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const auto* scales_ptr = static_cast<const float*>(scales->DataRaw());
    auto* output_ptr = static_cast<float*>(output->MutableDataRaw());
    ORT_RETURN_IF_ERROR(
        LaunchGatherBlockQuantizedKernel(data_ptr, indices_ptr, scales_ptr, zero_points_ptr, output_ptr, param));
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    const auto* scales_ptr = static_cast<const half*>(scales->DataRaw());
    auto* output_ptr = static_cast<half*>(output->MutableDataRaw());
    ORT_RETURN_IF_ERROR(
        LaunchGatherBlockQuantizedKernel(data_ptr, indices_ptr, scales_ptr, zero_points_ptr, output_ptr, param));
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    const auto* scales_ptr = static_cast<const BFloat16*>(scales->DataRaw());
    auto* output_ptr = static_cast<BFloat16*>(output->MutableDataRaw());
    ORT_RETURN_IF_ERROR(
        LaunchGatherBlockQuantizedKernel(data_ptr, indices_ptr, scales_ptr, zero_points_ptr, output_ptr, param));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
