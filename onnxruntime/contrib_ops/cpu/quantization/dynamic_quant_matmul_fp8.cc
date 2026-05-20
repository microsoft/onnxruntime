// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "dynamic_quant_matmul_fp8.h"

#include "core/common/common.h"
#include "core/common/fp8_common.h"
#include "core/framework/op_kernel.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/common/float16.h"
#include "core/common/float8.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/matmul_helper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace onnxruntime {
namespace contrib {

#if !defined(DISABLE_FLOAT8_TYPES)

namespace {

constexpr int64_t kDefaultBlockSize = 128;
constexpr int64_t kPackedBMetadataVersion = 1;
constexpr size_t kPackedBMetadataElementCount = 6;
constexpr size_t kPackedBMetadataSize = kPackedBMetadataElementCount * sizeof(int64_t);

enum PackedBMetadataIndex : size_t {
  kPackedBMetadataVersionIndex = 0,
  kPackedBMetadataRowsIndex,
  kPackedBMetadataColsIndex,
  kPackedBMetadataSizeIndex,
  kPackedBMetadataFp8ModeIndex,
  kPackedBMetadataHasFp8ModeIndex,
};

size_t CeilDiv(size_t value, size_t divisor) {
  ORT_ENFORCE(divisor != 0, "CeilDiv divisor must be non-zero.");
  return value == 0 ? 0 : ((value - 1) / divisor) + 1;
}

bool IsFp8DataType(ONNX_NAMESPACE::TensorProto_DataType elem_type) {
  return elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN ||
         elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ ||
         elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2 ||
         elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ;
}

bool IsValidFp8Mode(int64_t mode) {
  return mode >= static_cast<int64_t>(MLAS_FP8_MODE_E4M3_INF) &&
         mode < static_cast<int64_t>(MLAS_FP8_MODE_END);
}

Status RestorePackedBMetadata(const void* metadata_buffer,
                              size_t metadata_size,
                              size_t quantized_b_buffer_size,
                              TensorShape& b_shape,
                              size_t& quantized_b_size,
                              mlas_fp8_mode& b_type,
                              bool& has_b_type) {
  ORT_RETURN_IF(metadata_buffer == nullptr,
                "DynamicQuantMatMulFp8 requires shared prepacked B metadata.");
  ORT_RETURN_IF(metadata_size != kPackedBMetadataSize,
                "DynamicQuantMatMulFp8 shared prepacked B metadata has an unexpected size.");

  const auto* metadata = static_cast<const int64_t*>(metadata_buffer);
  ORT_RETURN_IF(metadata[kPackedBMetadataVersionIndex] != kPackedBMetadataVersion,
                "DynamicQuantMatMulFp8 shared prepacked B metadata has an unsupported version.");
  ORT_RETURN_IF(metadata[kPackedBMetadataRowsIndex] <= 0 || metadata[kPackedBMetadataColsIndex] <= 0,
                "DynamicQuantMatMulFp8 shared prepacked B metadata has an invalid B shape.");
  ORT_RETURN_IF(metadata[kPackedBMetadataSizeIndex] <= 0,
                "DynamicQuantMatMulFp8 shared prepacked B metadata has an invalid B buffer size.");
  ORT_RETURN_IF(metadata[kPackedBMetadataHasFp8ModeIndex] != 1 ||
                    !IsValidFp8Mode(metadata[kPackedBMetadataFp8ModeIndex]),
                "DynamicQuantMatMulFp8 shared prepacked B metadata has an invalid FP8 type.");

  const size_t rows = static_cast<size_t>(metadata[kPackedBMetadataRowsIndex]);
  const size_t cols = static_cast<size_t>(metadata[kPackedBMetadataColsIndex]);
  const size_t expected_quantized_b_size = SafeMul<size_t>(rows, cols);
  const size_t restored_quantized_b_size = static_cast<size_t>(metadata[kPackedBMetadataSizeIndex]);
  ORT_RETURN_IF(restored_quantized_b_size != expected_quantized_b_size ||
                    restored_quantized_b_size != quantized_b_buffer_size,
                "DynamicQuantMatMulFp8 shared prepacked B metadata does not match the B buffer size.");

  b_shape = TensorShape({metadata[kPackedBMetadataRowsIndex], metadata[kPackedBMetadataColsIndex]});
  quantized_b_size = restored_quantized_b_size;
  b_type = static_cast<mlas_fp8_mode>(metadata[kPackedBMetadataFp8ModeIndex]);
  has_b_type = true;
  return Status::OK();
}

// Reject invalid scales before quantization divides by them or MLAS dequantizes with them.
Status ValidatePositiveFiniteScales(const float* scales, size_t count, const char* scale_name) {
  for (size_t i = 0; i < count; ++i) {
    ORT_RETURN_IF(!std::isfinite(scales[i]) || scales[i] <= 0.0f,
                  "DynamicQuantMatMulFp8 requires ", scale_name, " values to be finite and positive.");
  }
  return Status::OK();
}

Status ValidateZeroPointValuesAreZero(const Tensor& zero_point, size_t expected_count,
                                      const char* zero_point_name) {
  const size_t actual_count = static_cast<size_t>(zero_point.Shape().Size());
  ORT_RETURN_IF(actual_count != expected_count,
                "DynamicQuantMatMulFp8 requires ", zero_point_name, " to have the expected number of elements.");

  const auto reject_non_zero = [zero_point_name](float value) {
    ORT_RETURN_IF(value != 0.0f,
                  "DynamicQuantMatMulFp8 supports symmetric quantization only; ",
                  zero_point_name, " values must be zero.");
    return Status::OK();
  };

  if (zero_point.IsDataType<Float8E4M3FN>()) {
    const auto* zp = static_cast<const uint8_t*>(zero_point.DataRaw());
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], MLAS_FP8_MODE_E4M3_INF)));
    }
  } else if (zero_point.IsDataType<Float8E4M3FNUZ>()) {
    const auto* zp = static_cast<const uint8_t*>(zero_point.DataRaw());
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], MLAS_FP8_MODE_E4M3_SAT)));
    }
  } else if (zero_point.IsDataType<Float8E5M2>()) {
    const auto* zp = static_cast<const uint8_t*>(zero_point.DataRaw());
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], MLAS_FP8_MODE_E5M2_INF)));
    }
  } else if (zero_point.IsDataType<Float8E5M2FNUZ>()) {
    const auto* zp = static_cast<const uint8_t*>(zero_point.DataRaw());
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], MLAS_FP8_MODE_E5M2_SAT)));
    }
  } else if (zero_point.IsDataType<float>()) {
    const auto* zp = zero_point.Data<float>();
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(zp[i]));
    }
  } else if (zero_point.IsDataType<MLFloat16>()) {
    const auto* zp = zero_point.Data<MLFloat16>();
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(static_cast<float>(zp[i])));
    }
  } else if (zero_point.IsDataType<BFloat16>()) {
    const auto* zp = zero_point.Data<BFloat16>();
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(static_cast<float>(zp[i])));
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unsupported zero point type for DynamicQuantMatMulFp8.");
  }
  return Status::OK();
}

template <typename SrcT>
void QuantizeBlockwiseFp8ABlock(const SrcT* src,
                                size_t M,
                                size_t K,
                                size_t block_size_m,
                                size_t block_size_k,
                                size_t blocks_k,
                                size_t block_m,
                                size_t block_k,
                                const float* scales,
                                mlas_fp8_mode mode,
                                uint8_t* dst) {
  const size_t m_begin = block_m * block_size_m;
  const size_t m_end = std::min(M, m_begin + block_size_m);
  const size_t k_begin = block_k * block_size_k;
  const size_t k_end = std::min(K, k_begin + block_size_k);
  const size_t idx = block_m * blocks_k + block_k;
  const float scale = scales[idx];
  for (size_t m = m_begin; m < m_end; ++m) {
    const size_t row_offset = m * K;
    for (size_t k = k_begin; k < k_end; ++k) {
      const float value = static_cast<float>(src[row_offset + k]);
      const float quantized = value / scale;
      dst[row_offset + k] = FloatToFp8Byte(quantized, mode);
    }
  }
}

template <typename SrcT, typename Fp8T>
void QuantizeBlockwiseFp8(const SrcT* src,
                          size_t K,
                          size_t N,
                          size_t block_size_k,
                          size_t block_size_n,
                          const float* scales,
                          uint8_t* dst) {
  // Block sizes come from op attributes; scale shapes only provide the number of blocks.
  const size_t blocks_n = N / block_size_n;
  for (size_t k = 0; k < K; ++k) {
    const size_t block_k = k / block_size_k;
    const size_t row_offset = k * N;
    for (size_t n = 0; n < N; ++n) {
      const size_t block_n = n / block_size_n;
      const size_t scale_idx = block_k * blocks_n + block_n;
      const float scale = scales[scale_idx];
      const float value = static_cast<float>(src[row_offset + n]);
      const float quantized = value / scale;
      const Fp8T fp8_value(quantized, true);
      dst[row_offset + n] = fp8_value.val;
    }
  }
}

template <typename SrcT>
Status QuantizeToFp8ByMode(mlas_fp8_mode fp8_mode,
                           const SrcT* src,
                           size_t K,
                           size_t N,
                           size_t block_size_k,
                           size_t block_size_n,
                           const float* scales,
                           uint8_t* dst) {
  // Dispatch quantization using the requested FP8 mode and runtime block sizes.
  switch (fp8_mode) {
    case MLAS_FP8_MODE_E4M3_INF:
      QuantizeBlockwiseFp8<SrcT, Float8E4M3FN>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    case MLAS_FP8_MODE_E4M3_SAT:
      QuantizeBlockwiseFp8<SrcT, Float8E4M3FNUZ>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    case MLAS_FP8_MODE_E5M2_INF:
      QuantizeBlockwiseFp8<SrcT, Float8E5M2>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    case MLAS_FP8_MODE_E5M2_SAT:
      QuantizeBlockwiseFp8<SrcT, Float8E5M2FNUZ>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported fp8 mode for DynamicQuantMatMulFp8.");
  }
}

}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    DynamicQuantMatMulFp8,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("TA", std::vector<MLDataType>{
                                  DataTypeImpl::GetTensorType<MLFloat16>(),
                                  DataTypeImpl::GetTensorType<BFloat16>(),
                                  DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("TB", std::vector<MLDataType>{DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<BFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<Float8E4M3FN>(), DataTypeImpl::GetTensorType<Float8E4M3FNUZ>(), DataTypeImpl::GetTensorType<Float8E5M2>(), DataTypeImpl::GetTensorType<Float8E5M2FNUZ>()})
        .TypeConstraint("TZ", std::vector<MLDataType>{DataTypeImpl::GetTensorType<Float8E4M3FN>(), DataTypeImpl::GetTensorType<Float8E4M3FNUZ>(), DataTypeImpl::GetTensorType<Float8E5M2>(), DataTypeImpl::GetTensorType<Float8E5M2FNUZ>()})
        .TypeConstraint("TS", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<BFloat16>()})
        .TypeConstraint("TY", std::vector<MLDataType>{DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<BFloat16>(), DataTypeImpl::GetTensorType<float>()}),
    DynamicQuantMatMulFp8);

DynamicQuantMatMulFp8::DynamicQuantMatMulFp8(const OpKernelInfo& info) : OpKernel(info) {
  const int64_t block_size_m = info.GetAttrOrDefault<int64_t>("block_size_m", kDefaultBlockSize);
  const int64_t block_size_k = info.GetAttrOrDefault<int64_t>("block_size_k", kDefaultBlockSize);
  const int64_t block_size_n = info.GetAttrOrDefault<int64_t>("block_size_n", kDefaultBlockSize);
  ORT_ENFORCE(block_size_m > 0,
              "DynamicQuantMatMulFp8 requires block_size_m to be greater than zero.");
  ORT_ENFORCE(block_size_k > 0,
              "DynamicQuantMatMulFp8 requires block_size_k to be greater than zero.");
  ORT_ENFORCE(block_size_n > 0,
              "DynamicQuantMatMulFp8 requires block_size_n to be greater than zero.");
  block_size_m_ = static_cast<size_t>(block_size_m);
  block_size_k_ = static_cast<size_t>(block_size_k);
  block_size_n_ = static_cast<size_t>(block_size_n);
}

Status DynamicQuantMatMulFp8::GetFp8Type(const Tensor& tensor, mlas_fp8_mode& out_type) {
  return GetFp8Type(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor.GetElementType()), out_type);
}

Status DynamicQuantMatMulFp8::GetFp8Type(ONNX_NAMESPACE::TensorProto_DataType elem_type,
                                         mlas_fp8_mode& out_type) {
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      out_type = MLAS_FP8_MODE_E4M3_INF;
      return Status::OK();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:
      out_type = MLAS_FP8_MODE_E4M3_SAT;
      return Status::OK();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      out_type = MLAS_FP8_MODE_E5M2_INF;
      return Status::OK();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ:
      out_type = MLAS_FP8_MODE_E5M2_SAT;
      return Status::OK();
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported fp8 type for DynamicQuantMatMulFp8.");
  }
}

Status DynamicQuantMatMulFp8::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                      /*out*/ bool& is_packed,
                                      /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;
  if (input_idx != GetBIdx()) {
    return Status::OK();
  }
  // Only prepack if B scale and zero point are constant initializers.
  const OrtValue* b_scale_ort = nullptr;
  const OrtValue* b_zp_ort = nullptr;
  const bool has_b_scale = Info().TryGetConstantInput(IN_B_SCALE, &b_scale_ort);
  const bool has_b_zp = Info().TryGetConstantInput(IN_B_ZERO_POINT, &b_zp_ort);
  if (!has_b_scale || !has_b_zp) {
    const auto b_elem_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor.GetElementType());
    ORT_RETURN_IF(!IsFp8DataType(b_elem_type),
                  "DynamicQuantMatMulFp8 requires B scale and B zero point to be constant initializers when B "
                  "is not FP8.");
    return Status::OK();
  }

  b_shape_ = tensor.Shape();
  if (b_shape_.NumDimensions() != 2) {
    return Status::OK();
  }

  const size_t K = static_cast<size_t>(b_shape_[0]);
  const size_t N = static_cast<size_t>(b_shape_[1]);
  if (K == 0) {
    return Status::OK();
  }

  const auto& b_scale = b_scale_ort->Get<Tensor>();
  const auto& b_zp = b_zp_ort->Get<Tensor>();

  const auto b_elem_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor.GetElementType());
  const auto b_zp_elem_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(b_zp.GetElementType());
  const bool b_is_fp8 = IsFp8DataType(b_elem_type);
  const bool zp_is_fp8 = IsFp8DataType(b_zp_elem_type);
  mlas_fp8_mode b_type{};
  if (b_is_fp8) {
    ORT_RETURN_IF_ERROR(GetFp8Type(tensor, b_type));
    if (zp_is_fp8) {
      mlas_fp8_mode b_zp_type{};
      ORT_RETURN_IF_ERROR(GetFp8Type(b_zp, b_zp_type));
      ORT_RETURN_IF(b_type != b_zp_type,
                    "DynamicQuantMatMulFp8 requires B and B zero point FP8 types to match.");
    }
  } else if (zp_is_fp8) {
    ORT_RETURN_IF_ERROR(GetFp8Type(b_zp, b_type));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "DynamicQuantMatMulFp8 requires fp8 zero points when B is not fp8.");
  }
  b_type_ = b_type;
  has_b_type_ = true;

  if (N == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(b_scale.IsDataType<float>() || b_scale.IsDataType<MLFloat16>() || b_scale.IsDataType<BFloat16>(),
                    "DynamicQuantMatMulFp8 requires B scale input to be float, float16, or bfloat16.");
  ORT_RETURN_IF_NOT(b_scale.Shape().NumDimensions() == 2,
                    "DynamicQuantMatMulFp8 requires B scale to be a 2D tensor.");
  ORT_RETURN_IF_NOT(b_zp.Shape().NumDimensions() == 2,
                    "DynamicQuantMatMulFp8 requires B zero point to be a 2D tensor.");
  ORT_RETURN_IF_NOT(b_zp.Shape()[0] == b_scale.Shape()[0] &&
                        b_zp.Shape()[1] == b_scale.Shape()[1],
                    "DynamicQuantMatMulFp8 requires B scale and zero point to have the same shape.");
  const size_t blocks_k = static_cast<size_t>(b_scale.Shape()[0]);
  const size_t blocks_n = static_cast<size_t>(b_scale.Shape()[1]);
  ORT_RETURN_IF_NOT(blocks_k != 0 && blocks_n != 0,
                    "DynamicQuantMatMulFp8 requires non-zero B scale dimensions.");
  ORT_RETURN_IF_NOT(K % block_size_k_ == 0,
                    "DynamicQuantMatMulFp8 requires K to be divisible by block_size_k.");
  ORT_RETURN_IF_NOT(N % block_size_n_ == 0,
                    "DynamicQuantMatMulFp8 requires N to be divisible by block_size_n.");
  const size_t expected_blocks_k = K / block_size_k_;
  const size_t expected_blocks_n = N / block_size_n_;
  ORT_RETURN_IF_NOT(blocks_k == expected_blocks_k,
                    "DynamicQuantMatMulFp8 requires B scale first dimension to be K / block_size_k.");
  ORT_RETURN_IF_NOT(blocks_n == expected_blocks_n,
                    "DynamicQuantMatMulFp8 requires B scale last dimension to be N / block_size_n.");

  const size_t b_scale_elems = static_cast<size_t>(b_scale.Shape().Size());
  ORT_RETURN_IF_ERROR(ValidateZeroPointValuesAreZero(b_zp, b_scale_elems, "B zero point"));

  const float* b_scales = nullptr;
  IAllocatorUniquePtr<float> b_scale_float;
  if (b_scale.IsDataType<float>()) {
    b_scales = b_scale.Data<float>();
  } else if (b_scale.IsDataType<MLFloat16>()) {
    b_scale_float = IAllocator::MakeUniquePtr<float>(alloc, b_scale_elems, true);
    for (size_t i = 0; i < b_scale_elems; ++i) {
      b_scale_float.get()[i] = static_cast<float>(b_scale.Data<MLFloat16>()[i]);
    }
    b_scales = b_scale_float.get();
  } else {
    b_scale_float = IAllocator::MakeUniquePtr<float>(alloc, b_scale_elems, true);
    for (size_t i = 0; i < b_scale_elems; ++i) {
      b_scale_float.get()[i] = static_cast<float>(b_scale.Data<BFloat16>()[i]);
    }
    b_scales = b_scale_float.get();
  }
  ORT_RETURN_IF_ERROR(ValidatePositiveFiniteScales(b_scales, b_scale_elems, "B scale"));
  // If B is not already FP8, quantize it once during prepack and reuse the cached FP8 buffer.
  if (!b_is_fp8) {
    const size_t quantized_b_size = SafeMul<size_t>(K, N);
    quantized_b_ = IAllocator::MakeUniquePtr<void>(alloc, quantized_b_size, true);
    quantized_b_size_ = quantized_b_size;
    auto* quantized_b_bytes = static_cast<uint8_t*>(quantized_b_.get());
    if (tensor.IsDataType<float>()) {
      ORT_RETURN_IF_ERROR(QuantizeToFp8ByMode(b_type, tensor.Data<float>(), K, N, block_size_k_, block_size_n_,
                                              b_scales, quantized_b_bytes));
    } else if (tensor.IsDataType<MLFloat16>()) {
      ORT_RETURN_IF_ERROR(QuantizeToFp8ByMode(b_type, tensor.Data<MLFloat16>(), K, N, block_size_k_, block_size_n_,
                                              b_scales, quantized_b_bytes));
    } else if (tensor.IsDataType<BFloat16>()) {
      ORT_RETURN_IF_ERROR(QuantizeToFp8ByMode(b_type, tensor.Data<BFloat16>(), K, N, block_size_k_, block_size_n_,
                                              b_scales, quantized_b_bytes));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported B type for DynamicQuantMatMulFp8 prepack.");
    }
    if (prepacked_weights != nullptr) {
      const std::array<int64_t, kPackedBMetadataElementCount> metadata_values = {
          kPackedBMetadataVersion,
          b_shape_[0],
          b_shape_[1],
          static_cast<int64_t>(quantized_b_size_),
          static_cast<int64_t>(b_type_),
          1,
      };
      auto metadata = IAllocator::MakeUniquePtr<void>(alloc, kPackedBMetadataSize, true);
      std::memcpy(metadata.get(), metadata_values.data(), kPackedBMetadataSize);
      prepacked_weights->buffers_.push_back(std::move(quantized_b_));
      prepacked_weights->buffer_sizes_.push_back(quantized_b_size_);
      prepacked_weights->buffers_.push_back(std::move(metadata));
      prepacked_weights->buffer_sizes_.push_back(kPackedBMetadataSize);
    }
    is_packed = true;
    return Status::OK();
  }

  return Status::OK();
}

Status DynamicQuantMatMulFp8::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                        gsl::span<const size_t> prepacked_buffer_sizes,
                                                        int input_idx,
                                                        /*out*/ bool& used_shared_buffers) {
  used_shared_buffers = false;
  if (input_idx != GetBIdx()) {
    return Status::OK();
  }

  ORT_RETURN_IF(prepacked_buffers.size() != 2 || prepacked_buffer_sizes.size() != 2,
                "DynamicQuantMatMulFp8 requires shared prepacked B data and metadata buffers.");
  ORT_RETURN_IF(prepacked_buffers[0].get() == nullptr,
                "DynamicQuantMatMulFp8 requires shared prepacked B data.");

  // Buffer 0 owns quantized B bytes; buffer 1 is metadata used only to restore kernel state.
  ORT_RETURN_IF_ERROR(RestorePackedBMetadata(prepacked_buffers[1].get(),
                                             prepacked_buffer_sizes[1],
                                             prepacked_buffer_sizes[0],
                                             b_shape_,
                                             quantized_b_size_,
                                             b_type_,
                                             has_b_type_));
  quantized_b_ = std::move(prepacked_buffers[0]);
  used_shared_buffers = true;
  return Status::OK();
}

Status DynamicQuantMatMulFp8::Compute(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(IN_A);
  const Tensor* b = quantized_b_ ? nullptr : context->Input<Tensor>(IN_B);
  const Tensor* a_scale = context->Input<Tensor>(IN_A_SCALE);
  const Tensor* a_zero_point = context->Input<Tensor>(IN_A_ZERO_POINT);
  const Tensor* b_scale = context->Input<Tensor>(IN_B_SCALE);
  const Tensor* b_zero_point = context->Input<Tensor>(IN_B_ZERO_POINT);
  const Tensor* y_scale = context->Input<Tensor>(IN_Y_SCALE);
  const Tensor* y_zero_point = context->Input<Tensor>(IN_Y_ZERO_POINT);

  // Runtime B uses one 2D B scale/zero-point layout, so reject batched B before MatMul broadcasts it.
  ORT_RETURN_IF(!quantized_b_ && b->Shape().NumDimensions() != 2,
                "DynamicQuantMatMulFp8 requires runtime B to be a 2D tensor.");

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(),
                                     quantized_b_ ? b_shape_ : b->Shape(),
                                     nullptr,
                                     nullptr));

  const size_t M = static_cast<size_t>(helper.M());
  const size_t N = static_cast<size_t>(helper.N());
  const size_t K = static_cast<size_t>(helper.K());
  Tensor* y = context->Output(OUT_Y, helper.OutputShape());
  const size_t y_size = static_cast<size_t>(y->Shape().Size());
  ORT_RETURN_IF(!y->IsDataType<float>() && !y->IsDataType<MLFloat16>() && !y->IsDataType<BFloat16>(),
                "DynamicQuantMatMulFp8 requires Y to be float, float16, or bfloat16.");

  if (y_zero_point != nullptr) {
    // Runtime tensors must match the schema scalar contract before reading element 0.
    ORT_RETURN_IF(y_zero_point->Shape().NumDimensions() != 0 || y_zero_point->Shape().Size() != 1,
                  "DynamicQuantMatMulFp8 requires Y zero point input to be a scalar.");
    ORT_RETURN_IF_ERROR(ValidateZeroPointValuesAreZero(*y_zero_point, 1, "Y zero point"));
  }

  float y_scale_storage = 0.0f;
  const float* y_scale_data = nullptr;
  if (y_scale != nullptr) {
    // Runtime tensors must match the schema scalar contract before reading element 0.
    ORT_RETURN_IF(y_scale->Shape().NumDimensions() != 0 || y_scale->Shape().Size() != 1,
                  "DynamicQuantMatMulFp8 requires Y scale input to be a scalar.");
    if (y_scale->IsDataType<float>()) {
      y_scale_data = y_scale->Data<float>();
    } else if (y_scale->IsDataType<MLFloat16>()) {
      y_scale_storage = static_cast<float>(y_scale->Data<MLFloat16>()[0]);
      y_scale_data = &y_scale_storage;
    } else if (y_scale->IsDataType<BFloat16>()) {
      y_scale_storage = static_cast<float>(y_scale->Data<BFloat16>()[0]);
      y_scale_data = &y_scale_storage;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "DynamicQuantMatMulFp8 requires Y scale input to be float, float16, or bfloat16.");
    }
    ORT_RETURN_IF_ERROR(ValidatePositiveFiniteScales(y_scale_data, 1, "Y scale"));
  }

  // Empty reduction does not need B data, so fill zeros before enforcing runtime FP8 B.
  if (K == 0) {
    if (y_size == 0) {
      return Status::OK();
    }
    if (y->IsDataType<float>()) {
      std::fill_n(y->MutableData<float>(), y_size, 0.0f);
    } else if (y->IsDataType<MLFloat16>()) {
      std::fill_n(y->MutableData<MLFloat16>(), y_size, MLFloat16::FromBits(0));
    } else if (y->IsDataType<BFloat16>()) {
      std::fill_n(y->MutableData<BFloat16>(), y_size, BFloat16::FromBits(0));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "DynamicQuantMatMulFp8 requires Y to be float, float16, or bfloat16.");
    }
    return Status::OK();
  }

  const bool a_is_supported =
      a->IsDataType<float>() || a->IsDataType<MLFloat16>() || a->IsDataType<BFloat16>();
  ORT_RETURN_IF(!a_is_supported, "DynamicQuantMatMulFp8 requires A to be float, float16, or bfloat16.");

  const auto b_elem_type = b ? static_cast<ONNX_NAMESPACE::TensorProto_DataType>(b->GetElementType())
                             : static_cast<ONNX_NAMESPACE::TensorProto_DataType>(0);
  const auto b_zp_elem_type =
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(b_zero_point->GetElementType());
  const bool b_is_fp8 = IsFp8DataType(b_elem_type);

  mlas_fp8_mode a_type{};
  ORT_RETURN_IF(a_zero_point == nullptr,
                "DynamicQuantMatMulFp8 requires FP8 zero point for A.");
  ORT_RETURN_IF_ERROR(GetFp8Type(*a_zero_point, a_type));
  mlas_fp8_mode b_type{};
  if (has_b_type_) {
    b_type = b_type_;
  } else if (b_is_fp8) {
    ORT_RETURN_IF_ERROR(GetFp8Type(b_elem_type, b_type));
    if (IsFp8DataType(b_zp_elem_type)) {
      mlas_fp8_mode b_zp_type{};
      ORT_RETURN_IF_ERROR(GetFp8Type(b_zp_elem_type, b_zp_type));
      ORT_RETURN_IF(b_type != b_zp_type,
                    "DynamicQuantMatMulFp8 requires B and B zero point FP8 types to match.");
    }
  } else {
    ORT_RETURN_IF_ERROR(GetFp8Type(b_zp_elem_type, b_type));
  }
  ORT_RETURN_IF(a_type != b_type,
                "DynamicQuantMatMulFp8 requires A/B FP8 types to match.");

  if (y_size == 0) {
    return Status::OK();
  }

  // Select the FP8 B buffer: prefer pre-quantized B from PrePack, otherwise accept FP8-typed B input.
  const uint8_t* b_fp8 = nullptr;
  if (quantized_b_) {
    b_fp8 = static_cast<const uint8_t*>(quantized_b_.get());
  } else if (b_is_fp8) {
    b_fp8 = static_cast<const uint8_t*>(b->DataRaw());
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "DynamicQuantMatMulFp8 requires runtime B input to be FP8. Non-FP8 B is only supported "
                           "when B is a constant initializer that can be quantized during prepack.");
  }

  const size_t num_gemms = helper.OutputOffsets().size();
  const size_t a_scale_rank = static_cast<size_t>(a_scale->Shape().NumDimensions());
  const size_t a_zp_rank = static_cast<size_t>(a_zero_point->Shape().NumDimensions());
  // The scale tensor layout carries the number of tiles because it physically stores one
  // scale per tile.
  // A scale/zero-point may be [blocks_m, blocks_k] or [prefix..., blocks_m, blocks_k].
  // The scale tensor shape provides the number of quantization blocks. It does not define
  // the block size. Block sizes come from the block_size_m/block_size_k/block_size_n attributes
  // so future models can choose different tile sizes without changing how scale tensors are
  // interpreted. The validation below binds both pieces of information together by requiring:
  //   blocks_m == ceil(M / block_size_m)
  //   blocks_k == K / block_size_k
  //   blocks_n == N / block_size_n
  // This prevents silently treating a malformed scale shape as a different runtime block size.
  ORT_RETURN_IF(a_scale_rank < 2,
                "DynamicQuantMatMulFp8 requires A scale to have rank >= 2.");
  ORT_RETURN_IF(a_zp_rank < 2,
                "DynamicQuantMatMulFp8 requires A zero point to have rank >= 2.");
  ORT_RETURN_IF(a_scale_rank != a_zp_rank,
                "DynamicQuantMatMulFp8 requires A scale and zero point to have the same rank.");
  const size_t blocks_m = static_cast<size_t>(a_scale->Shape()[a_scale_rank - 2]);
  const size_t blocks_k = static_cast<size_t>(a_scale->Shape()[a_scale_rank - 1]);
  for (size_t dim = 0; dim < a_scale_rank; ++dim) {
    ORT_RETURN_IF(a_scale->Shape()[dim] != a_zero_point->Shape()[dim],
                  "DynamicQuantMatMulFp8 requires A scale and zero point to have the same shape.");
  }
  if (a_scale_rank != 2) {
    const size_t a_rank = a->Shape().NumDimensions();
    ORT_RETURN_IF(a_scale_rank != a_rank,
                  "DynamicQuantMatMulFp8 requires A scale rank to be 2 or match A rank.");
    for (size_t dim = 0; dim < a_rank - 2; ++dim) {
      ORT_RETURN_IF(a_scale->Shape()[dim] != a->Shape()[dim],
                    "DynamicQuantMatMulFp8 requires A scale batch dimensions to match A.");
    }
  }
  // Scale tensor block counts must match the explicit block-size attributes before reading scale data.
  ORT_RETURN_IF(blocks_m == 0, "DynamicQuantMatMulFp8 requires non-zero A scale M dimension.");
  ORT_RETURN_IF(blocks_k == 0, "DynamicQuantMatMulFp8 requires non-zero A scale K dimension.");
  ORT_RETURN_IF(K % block_size_k_ != 0,
                "DynamicQuantMatMulFp8 requires K to be divisible by block_size_k.");
  const size_t expected_blocks_m = CeilDiv(M, block_size_m_);
  const size_t expected_blocks_k = K / block_size_k_;
  // If the scale tensor says it has a different number of M blocks than ceil(M / block_size_m),
  // return an error instead of running with wrong scale indexing.
  ORT_RETURN_IF(blocks_m != expected_blocks_m,
                "DynamicQuantMatMulFp8 requires A scale M dimension to be ceil(M / block_size_m).");
  ORT_RETURN_IF(blocks_k != expected_blocks_k,
                "DynamicQuantMatMulFp8 requires A scale K dimension to be K / block_size_k.");

  ORT_RETURN_IF(b_scale->Shape().NumDimensions() != 2,
                "DynamicQuantMatMulFp8 requires B scale to be a 2D tensor.");
  ORT_RETURN_IF(b_zero_point->Shape().NumDimensions() != 2,
                "DynamicQuantMatMulFp8 requires B zero point to be a 2D tensor.");
  const size_t blocks_n = static_cast<size_t>(b_scale->Shape()[1]);
  ORT_RETURN_IF(b_zero_point->Shape()[0] != b_scale->Shape()[0] ||
                    b_zero_point->Shape()[1] != b_scale->Shape()[1],
                "DynamicQuantMatMulFp8 requires B scale and zero point to have the same shape.");
  ORT_RETURN_IF(static_cast<size_t>(b_scale->Shape()[0]) != blocks_k,
                "DynamicQuantMatMulFp8 requires B scale K dimension to match A scale K dimension.");
  ORT_RETURN_IF(blocks_n == 0, "DynamicQuantMatMulFp8 requires non-zero B scale N dimension.");
  ORT_RETURN_IF(N % block_size_n_ != 0,
                "DynamicQuantMatMulFp8 requires N to be divisible by block_size_n.");
  const size_t expected_blocks_n = N / block_size_n_;
  ORT_RETURN_IF(blocks_n != expected_blocks_n,
                "DynamicQuantMatMulFp8 requires B scale N dimension to be N / block_size_n.");

  size_t a_scale_prefix = 1;
  if (a_scale_rank > 2) {
    for (size_t dim = 0; dim < a_scale_rank - 2; ++dim) {
      a_scale_prefix = SafeMul<size_t>(
          a_scale_prefix, static_cast<size_t>(a_scale->Shape()[dim]));
    }
  }
  const size_t a_scale_batch_stride = SafeMul<size_t>(blocks_m, blocks_k);
  const size_t a_zp_count = SafeMul<size_t>(a_scale_prefix, a_scale_batch_stride);
  const size_t b_zp_count = SafeMul<size_t>(blocks_k, blocks_n);

  ORT_RETURN_IF_ERROR(ValidateZeroPointValuesAreZero(*a_zero_point, a_zp_count, "A zero point"));
  ORT_RETURN_IF_ERROR(ValidateZeroPointValuesAreZero(*b_zero_point, b_zp_count, "B zero point"));

  AllocatorPtr temp_allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&temp_allocator));

  const float* a_scales = nullptr;
  IAllocatorUniquePtr<float> a_scale_float;
  const size_t a_scale_elems = static_cast<size_t>(a_scale->Shape().Size());
  if (a_scale->IsDataType<float>()) {
    a_scales = a_scale->Data<float>();
  } else {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    a_scale_float = IAllocator::MakeUniquePtr<float>(allocator, a_scale_elems, true);
    if (a_scale->IsDataType<MLFloat16>()) {
      for (size_t i = 0; i < a_scale_elems; ++i) {
        a_scale_float.get()[i] = static_cast<float>(a_scale->Data<MLFloat16>()[i]);
      }
    } else if (a_scale->IsDataType<BFloat16>()) {
      for (size_t i = 0; i < a_scale_elems; ++i) {
        a_scale_float.get()[i] = static_cast<float>(a_scale->Data<BFloat16>()[i]);
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "DynamicQuantMatMulFp8 requires A scale input to be float, float16, or bfloat16.");
    }
    a_scales = a_scale_float.get();
  }

  const float* b_scales = nullptr;
  IAllocatorUniquePtr<float> b_scale_float;
  const size_t b_scale_elems = static_cast<size_t>(b_scale->Shape().Size());
  if (b_scale->IsDataType<float>()) {
    b_scales = b_scale->Data<float>();
  } else {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    b_scale_float = IAllocator::MakeUniquePtr<float>(allocator, b_scale_elems, true);
    if (b_scale->IsDataType<MLFloat16>()) {
      for (size_t i = 0; i < b_scale_elems; ++i) {
        b_scale_float.get()[i] = static_cast<float>(b_scale->Data<MLFloat16>()[i]);
      }
    } else if (b_scale->IsDataType<BFloat16>()) {
      for (size_t i = 0; i < b_scale_elems; ++i) {
        b_scale_float.get()[i] = static_cast<float>(b_scale->Data<BFloat16>()[i]);
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "DynamicQuantMatMulFp8 requires B scale input to be float, float16, or bfloat16.");
    }
    b_scales = b_scale_float.get();
  }

  // MLAS FP8 GEMM accumulates and stores float output. Use scratch for lower-precision Y,
  // then convert once after all batched GEMMs complete.
  IAllocatorUniquePtr<float> y_float_buffer;
  float* y_float_data = nullptr;
  if (y->IsDataType<float>()) {
    y_float_data = y->MutableData<float>();
  } else if (y->IsDataType<MLFloat16>() || y->IsDataType<BFloat16>()) {
    y_float_buffer = IAllocator::MakeUniquePtr<float>(temp_allocator, y_size, true);
    y_float_data = y_float_buffer.get();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "DynamicQuantMatMulFp8 requires Y to be float, float16, or bfloat16.");
  }

  MLAS_FP8_GEMM_SHAPE_PARAMS gemm_shape;
  gemm_shape.M = M;
  gemm_shape.N = N;
  gemm_shape.K = K;

  ORT_RETURN_IF_ERROR(ValidatePositiveFiniteScales(a_scales, a_scale_elems, "A scale"));
  ORT_RETURN_IF_ERROR(ValidatePositiveFiniteScales(b_scales, b_scale_elems, "B scale"));

  const size_t a_fp8_size = SafeMul<size_t>(M, K);
  const size_t a_num_elements = static_cast<size_t>(a->Shape().Size());
  ORT_RETURN_IF(a_num_elements % a_fp8_size != 0,
                "DynamicQuantMatMulFp8 requires A to contain complete MxK matrices.");
  const size_t a_batch_count = a_num_elements / a_fp8_size;
  ORT_RETURN_IF(a_scale_prefix != 1 && a_scale_prefix != a_batch_count,
                "DynamicQuantMatMulFp8 requires A scale batch dimensions to match A.");

  // Quantize the physical A tensor once. Broadcasted output GEMMs then reuse the same FP8 A slice.
  auto a_fp8_buffer = IAllocator::MakeUniquePtr<uint8_t>(temp_allocator, a_num_elements, true);
  const size_t a_quant_work_items = SafeMul<size_t>(a_batch_count, a_scale_batch_stride);
  ORT_RETURN_IF(a_quant_work_items > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                "DynamicQuantMatMulFp8 A quantization work item count exceeds ptrdiff_t range.");
  const auto a_quant_work_items_i = static_cast<std::ptrdiff_t>(a_quant_work_items);
  const size_t a_quant_block_elems = SafeMul<size_t>(block_size_m_, block_size_k_);
  const TensorOpCost a_quant_unit_cost{
      static_cast<double>(SafeMul<size_t>(a_quant_block_elems, sizeof(float))),
      static_cast<double>(SafeMul<size_t>(a_quant_block_elems, sizeof(uint8_t))),
      static_cast<double>(a_quant_block_elems) * 2.0};
  const auto quantize_a_batches = [&](const auto* a_data) {
    concurrency::ThreadPool::TryParallelFor(context->GetOperatorThreadPool(), a_quant_work_items_i,
                                            a_quant_unit_cost,
                                            [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
                                              for (std::ptrdiff_t tid = begin; tid < end; ++tid) {
                                                const size_t work_idx = static_cast<size_t>(tid);
                                                const size_t a_batch_idx = work_idx / a_scale_batch_stride;
                                                const size_t scale_block_idx = work_idx % a_scale_batch_stride;
                                                const size_t block_m = scale_block_idx / blocks_k;
                                                const size_t block_k = scale_block_idx % blocks_k;
                                                const size_t a_batch_offset = a_batch_idx * a_fp8_size;
                                                const size_t scale_batch_index = (a_scale_prefix == 1) ? 0 : a_batch_idx;
                                                const size_t a_scale_batch_offset = scale_batch_index * a_scale_batch_stride;
                                                QuantizeBlockwiseFp8ABlock(a_data + a_batch_offset,
                                                                           M, K, block_size_m_, block_size_k_, blocks_k,
                                                                           block_m, block_k,
                                                                           a_scales + a_scale_batch_offset, a_type,
                                                                           a_fp8_buffer.get() + a_batch_offset);
                                              }
                                            });
  };
  if (a->IsDataType<float>()) {
    quantize_a_batches(a->Data<float>());
  } else if (a->IsDataType<MLFloat16>()) {
    quantize_a_batches(a->Data<MLFloat16>());
  } else if (a->IsDataType<BFloat16>()) {
    quantize_a_batches(a->Data<BFloat16>());
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "DynamicQuantMatMulFp8 requires A to be float, float16, or bfloat16.");
  }

  std::vector<MLAS_FP8_GEMM_DATA_PARAMS> gemm_data_vec(num_gemms);
  for (size_t gemm_idx = 0; gemm_idx < num_gemms; ++gemm_idx) {
    const size_t a_offset = helper.LeftOffsets()[gemm_idx];
    ORT_RETURN_IF(a_offset >= a_num_elements || (a_offset % a_fp8_size) != 0,
                  "DynamicQuantMatMulFp8 requires A offsets to reference complete MxK matrices.");
    const size_t scale_batch_index = (a_scale_prefix == 1) ? 0 : a_offset / a_fp8_size;
    ORT_RETURN_IF(scale_batch_index >= a_scale_prefix,
                  "DynamicQuantMatMulFp8 requires A scale batch dimensions to match A.");
    const size_t a_scale_batch_offset = SafeMul<size_t>(scale_batch_index, a_scale_batch_stride);
    const float* a_scales_batch = a_scales + a_scale_batch_offset;
    auto& gemm_data = gemm_data_vec[gemm_idx];
    gemm_data.A = a_fp8_buffer.get() + a_offset;
    gemm_data.lda = K;
    gemm_data.B = b_fp8 + helper.RightOffsets()[gemm_idx];
    gemm_data.ldb = N;
    gemm_data.C = y_float_data + helper.OutputOffsets()[gemm_idx];
    gemm_data.ldc = N;
    gemm_data.ScaleA = a_scales_batch;
    gemm_data.ScaleB = b_scales;
    gemm_data.ScaleY = y_scale_data;
    gemm_data.Fp8Type = a_type;
    gemm_data.BlockSizeM = block_size_m_;
    gemm_data.BlockSizeK = block_size_k_;
    gemm_data.BlockSizeN = block_size_n_;
    gemm_data.BlocksM = blocks_m;
    gemm_data.BlocksK = blocks_k;
    gemm_data.BlocksN = blocks_n;
    gemm_data.ScaleAStrideK = 1;
    gemm_data.ScaleAStrideM = blocks_k;
    gemm_data.ScaleBStrideN = 1;
    gemm_data.ScaleBStrideK = blocks_n;
  }

  MlasFp8GemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms, context->GetOperatorThreadPool());

  if (y_float_buffer != nullptr) {
    if (y->IsDataType<MLFloat16>()) {
      auto* y_data = y->MutableData<MLFloat16>();
      for (size_t i = 0; i < y_size; ++i) {
        y_data[i] = static_cast<MLFloat16>(y_float_data[i]);
      }
    } else {
      auto* y_data = y->MutableData<BFloat16>();
      for (size_t i = 0; i < y_size; ++i) {
        y_data[i] = BFloat16(y_float_data[i]);
      }
    }
  }

  return Status::OK();
}

#endif  // !defined(DISABLE_FLOAT8_TYPES)

}  // namespace contrib
}  // namespace onnxruntime
