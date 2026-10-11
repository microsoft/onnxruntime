// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/dynamic_quantize_matmul_nbits.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include "core/common/safeint.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;

ONNX_OPERATOR_KERNEL_EX(
    DynamicQuantizeMatMulNBits, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, BFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float, MLFloat16, BFloat16>()),
    DynamicQuantizeMatMulNBits);

namespace {
Status CheckInput(bool valid, const char* message) {
  return valid ? Status::OK() : ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, message);
}

template <typename T>
Status ValidateConstantScales(const Tensor& scales) {
  // Initialization only. Chunking bounds host scratch even for very large initializers.
  std::array<T, 4096> host;
  const T* data = scales.Data<T>();
  const int64_t count = scales.Shape().Size();
  for (int64_t offset = 0; offset < count;) {
    const auto size = static_cast<size_t>(std::min<int64_t>(host.size(), count - offset));
    const T* values = data + offset;
    if (scales.Location().device.Type() != OrtDevice::CPU) {
      CUDA_RETURN_IF_ERROR(cudaMemcpy(host.data(), values, size * sizeof(T), cudaMemcpyDeviceToHost));
      values = host.data();
    }
    for (size_t i = 0; i < size; ++i) {
      ORT_RETURN_IF_ERROR(CheckInput(std::isfinite(static_cast<float>(values[i])), "scales must contain only finite values."));
    }
    offset += size;
  }
  return Status::OK();
}
}  // namespace

DynamicQuantizeMatMulNBits::DynamicQuantizeMatMulNBits(const OpKernelInfo& info)
    : CudaKernel(info),
      bits_(info.GetAttrOrDefault<int64_t>("bits", 8)),
      block_size_(info.GetAttrOrDefault<int64_t>("block_size", 128)),
      activation_block_size_(0) {
  enable_sm90_ = !ParseEnvironmentVariableWithDefault<bool>("ORT_DISABLE_DYNAMIC_QUANTIZED_SM90_GEMM", false);
  ORT_ENFORCE(bits_ == 2 || bits_ == 4 || bits_ == 8, "bits must be 2, 4, or 8.");
  ORT_ENFORCE(block_size_ >= 16 && (block_size_ & (block_size_ - 1)) == 0,
              "block_size must be a power of two and at least 16.");
  ORT_ENFORCE(bits_ != 2 || block_size_ <= 256, "INT2 block_size must not exceed 256.");
  const auto mode = info.GetAttrOrDefault<std::string>("activation_quantization", "dynamic_per_token");
  const bool has_block_size = info.GetAttr<int64_t>("activation_block_size", &activation_block_size_).IsOK();
  ORT_ENFORCE(mode == "dynamic_per_tensor" || mode == "dynamic_per_token" || mode == "dynamic_per_block",
              "Unrecognized activation_quantization.");
  mode_ = mode == "dynamic_per_tensor"  ? DynamicQuantizationMode::PerTensor
          : mode == "dynamic_per_token" ? DynamicQuantizationMode::PerToken
                                        : DynamicQuantizationMode::PerBlock;
  ORT_ENFORCE(mode_ == DynamicQuantizationMode::PerBlock ? has_block_size && activation_block_size_ > 0
                                                         : !has_block_size,
              "activation_block_size must be positive and present exactly for dynamic_per_block.");

  const Tensor* scales = nullptr;
  if (info.TryGetConstantInput(2, &scales)) {
    if (scales->IsDataType<float>()) {
      ORT_THROW_IF_ERROR(ValidateConstantScales<float>(*scales));
    } else if (scales->IsDataType<MLFloat16>()) {
      ORT_THROW_IF_ERROR(ValidateConstantScales<MLFloat16>(*scales));
    } else {
      ORT_THROW_IF_ERROR(ValidateConstantScales<BFloat16>(*scales));
    }
    constant_scales_validated_ = true;
  }
}

Status DynamicQuantizeMatMulNBits::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                           bool& is_packed, PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  const auto& shape = tensor.Shape();
  if (!enable_sm90_ || input_idx != 1 || GetDeviceProp().major != 9 || GetDeviceProp().minor != 0 ||
      shape.NumDimensions() != 3 || shape[0] % 8 != 0 || shape[2] != block_size_ * bits_ / 8) {
    return Status::OK();
  }
  // Single-interval W8 (one large block) uses the CUTLASS path; block-128 weights with several groups use the
  // grouped-scale path, which needs lossless signed INT8 codes for W2/W4 as well.
  const bool single_interval = bits_ == 8 && shape[0] >= 512 && shape[1] == 1 && block_size_ >= 128;
  const bool grouped = block_size_ == 128 && shape[1] >= 2 && shape[0] >= 256;
  if (!single_interval && !grouped) return Status::OK();
  bool available;
  ORT_RETURN_IF_ERROR(DynamicQuantizedSm90Available(available));
  if (!available) return Status::OK();
  const int64_t expanded_size = single_interval ? shape.Size() : static_cast<int64_t>(SafeInt<int64_t>(shape[0]) * shape[1] * 128);
  auto signed_weights = IAllocator::MakeUniquePtr<int8_t>(alloc, static_cast<size_t>(expanded_size), true);
  IAllocatorUniquePtr<uint8_t> staging;
  const uint8_t* source = tensor.Data<uint8_t>();
  const auto stream = cudaStreamPerThread;
  if (tensor.Location().device.Type() != OrtDevice::GPU) {
    staging = IAllocator::MakeUniquePtr<uint8_t>(alloc, tensor.SizeInBytes(), true);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(staging.get(), source, tensor.SizeInBytes(), cudaMemcpyDefault, stream));
    source = staging.get();
  }
  const auto status = single_interval
                          ? LaunchDynamicSignedWeights(source, signed_weights.get(), shape.Size(), stream)
                          : LaunchDynamicExpandedWeights(source, signed_weights.get(), static_cast<int>(bits_),
                                                         expanded_size, stream);
  // Initialization must finish before source/staging release or another Run stream reads the buffer.
  const auto sync = cudaStreamSynchronize(stream);
  ORT_RETURN_IF_ERROR(status);
  CUDA_RETURN_IF_ERROR(sync);
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&sm90_device_id_));
  sm90_sm_count_ = GetDeviceProp().multiProcessorCount;
  signed_n_ = shape[0];
  signed_stride_ = block_size_;
  signed_grouped_ = !single_interval;
  signed_weights_ = std::move(signed_weights);
  return Status::OK();
}

Status DynamicQuantizeMatMulNBits::ComputeInternal(OpKernelContext* context) const {
  const auto* a = context->Input<Tensor>(0);
  const auto* b = context->Input<Tensor>(1);
  const auto* scales = context->Input<Tensor>(2);
  const auto* bias = context->Input<Tensor>(3);
  const auto& as = a->Shape();
  const auto& bs = b->Shape();
  const auto& ss = scales->Shape();
  ORT_RETURN_IF_ERROR(CheckInput(as.NumDimensions() >= 1 && bs.NumDimensions() == 3,
                                 "A must have rank at least 1 and B must have rank 3."));
  const int64_t k = as[as.NumDimensions() - 1];
  const int64_t n = bs[0];
  const int64_t groups = k / block_size_ + (k % block_size_ != 0);
  const int64_t blob_size = block_size_ / (8 / bits_);
  const int64_t scale_count = SafeInt<int64_t>(n) * groups;
  ORT_RETURN_IF_ERROR(CheckInput(bs[1] == groups && bs[2] == blob_size, "B must have shape [N, ceil(K/block_size), blob_size]."));
  ORT_RETURN_IF_ERROR(CheckInput((ss.NumDimensions() == 1 && ss[0] == scale_count) ||
                                     (ss.NumDimensions() == 2 && ss[0] == n && ss[1] == groups),
                                 "scales must have shape [N, G] or [N * G]."));
  if (bias) {
    ORT_RETURN_IF_ERROR(CheckInput(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == n,
                                   "bias must have shape [N]."));
  }
  const int64_t activation_block = mode_ == DynamicQuantizationMode::PerBlock ? activation_block_size_ : k;
  const int64_t interval_bound = std::min({k, block_size_, activation_block});
  ORT_RETURN_IF_ERROR(CheckInput(interval_bound <= std::numeric_limits<int32_t>::max() / (127 * (int64_t{1} << (bits_ - 1))),
                                 "Common scale interval exceeds the INT32 accumulator bound."));

  TensorShapeVector output_shape(as.GetDims().begin(), as.GetDims().end());
  output_shape.back() = n;
  auto* y = context->Output(0, output_shape);
  const int64_t output_count = y->Shape().Size();
  if (output_count == 0) {
    return Status::OK();
  }
  const int64_t m = output_count / n;
  const bool bf16 = a->IsDataType<BFloat16>();
  const auto stream = Stream(context);
  const void* bias_data = bias ? bias->DataRaw() : nullptr;
  if (k == 0) {
    return LaunchDynamicQuantizedEmpty(y->MutableDataRaw(), bias_data, bf16, output_count, n, stream);
  }
  const auto scale_type = scales->IsDataType<float>()       ? DynamicWeightScaleType::Float
                          : scales->IsDataType<MLFloat16>() ? DynamicWeightScaleType::Half
                                                            : DynamicWeightScaleType::BFloat16;
  if (!constant_scales_validated_) {
    cudaStreamCaptureStatus capture;
    CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(stream, &capture));
    ORT_RETURN_IF_ERROR(CheckInput(capture == cudaStreamCaptureStatusNone,
                                   "CUDA graph capture requires non-overridable constant scales."));
    auto invalid = GetScratchBuffer<int>(1, GetComputeStream(context));
    ORT_RETURN_IF_ERROR(LaunchValidateDynamicWeightScales(scales->DataRaw(), scale_type, scale_count,
                                                          invalid.get(), stream));
    int host_invalid = 0;
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(&host_invalid, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    ORT_RETURN_IF_ERROR(CheckInput(host_invalid == 0, "scales must contain only finite values."));
  }

  const int64_t activation_groups = k / activation_block + (k % activation_block != 0);
  const bool per_tensor = mode_ == DynamicQuantizationMode::PerTensor;
  // Bound scratch by a row tile, retaining at least one complete activation row.
  constexpr int64_t scratch_budget = 16 * 1024 * 1024;
  const size_t tensor_partial_count = per_tensor ? DynamicTensorScaleScratchSize(a->Shape().Size()) : 0;
  const int64_t reduction_bytes = (SafeInt<int64_t>(tensor_partial_count) + (per_tensor ? 1 : 0)) * sizeof(float);
  const int64_t row_bytes = SafeInt<int64_t>(k) + SafeInt<int64_t>(per_tensor ? 0 : activation_groups) * sizeof(float);
  const int64_t tile_m = std::min(m, std::max<int64_t>(1, (scratch_budget - reduction_bytes) / row_bytes));
  auto quantized = GetScratchBuffer<int8_t>(SafeInt<size_t>(tile_m) * k, GetComputeStream(context));
  const size_t activation_scale_count = SafeInt<size_t>(tile_m) * (per_tensor ? 0 : activation_groups);
  auto activation_scales = GetScratchBuffer<float>(activation_scale_count, GetComputeStream(context));
  auto tensor_scale = GetScratchBuffer<float>(per_tensor ? 1 : 0, GetComputeStream(context));
  auto tensor_partials = GetScratchBuffer<float>(tensor_partial_count, GetComputeStream(context));
  if (per_tensor) {
    ORT_RETURN_IF_ERROR(LaunchDynamicTensorScale(a->DataRaw(), bf16, a->Shape().Size(), tensor_scale.get(),
                                                 tensor_partials.get(), stream));
  }
  const bool sm90_dims_ok = signed_weights_ && n == signed_n_ && activation_block >= k &&
                            m <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max() &&
                            k <= std::numeric_limits<int>::max();
  const bool sm90_shape_ok = sm90_dims_ok && !signed_grouped_ && !bf16 && groups == 1 && k >= 128 && k % 16 == 0;
  // The grouped kernel covers exactly one 128-wide scale block per K tile.
  const bool sm90_grouped_ok = sm90_dims_ok && signed_grouped_ && k % 128 == 0 && groups == k / 128;
  for (int64_t row = 0; row < m;) {
    const int64_t rows = std::min(tile_m, m - row);
    const size_t a_offset = SafeInt<size_t>(row) * k;
    const size_t y_offset = SafeInt<size_t>(row) * n;
    const auto* a_tile = static_cast<const uint16_t*>(a->DataRaw()) + a_offset;
    auto* y_tile = static_cast<uint16_t*>(y->MutableDataRaw()) + y_offset;
    ORT_RETURN_IF_ERROR(LaunchDynamicQuantizeRows(a_tile, bf16, rows, k, activation_block,
                                                  per_tensor ? tensor_scale.get() : nullptr,
                                                  quantized.get(), activation_scales.get(), stream));
    const float* a_scales = per_tensor ? tensor_scale.get() : activation_scales.get();
    const bool sm90_geometry = (k >= 1024 && n >= 1024) || (rows >= 128 && n >= 512);
    const bool aligned16 = reinterpret_cast<uintptr_t>(y_tile) % 16 == 0 &&
                           reinterpret_cast<uintptr_t>(scales->DataRaw()) % 16 == 0 &&
                           (!bias_data || reinterpret_cast<uintptr_t>(bias_data) % 16 == 0);
    if (sm90_shape_ok && sm90_geometry && rows >= 16 && aligned16) {
      ORT_RETURN_IF_ERROR(LaunchDynamicQuantizedSm90(
          quantized.get(), a_scales, signed_weights_.get(), scales->DataRaw(), scale_type,
          bias_data, y_tile, bf16, per_tensor, static_cast<int>(rows), static_cast<int>(n), static_cast<int>(k),
          signed_stride_, sm90_device_id_, sm90_sm_count_, stream));
    } else if (sm90_grouped_ok && aligned16) {
      ORT_RETURN_IF_ERROR(LaunchDynamicQuantizedSm90Grouped(
          quantized.get(), a_scales, signed_weights_.get(), scales->DataRaw(), scale_type, bias_data, y_tile,
          bf16, per_tensor, static_cast<int>(rows), static_cast<int>(n), static_cast<int>(k), sm90_sm_count_, stream));
    } else {
      ORT_RETURN_IF_ERROR(LaunchDynamicQuantizedMatMul(quantized.get(), a_scales, b->Data<uint8_t>(),
                                                       scales->DataRaw(), scale_type, bias_data, y_tile, bf16,
                                                       per_tensor, rows, n, k, static_cast<int>(bits_), block_size_, activation_block,
                                                       GetDeviceProp().major * 10 + GetDeviceProp().minor, stream));
    }
    row += rows;
  }
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
