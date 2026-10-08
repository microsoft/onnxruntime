// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "contrib_ops/cpu/quantization/dynamic_quant_matmul_fp8.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/common/float8.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/matmul_helper.h"

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
  kPackedBMetadataScaleCountIndex,
  kPackedBMetadataFp8ModeIndex,
};

bool IsFp8DataType(ONNX_NAMESPACE::TensorProto_DataType elem_type) {
  return elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN ||
         elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ ||
         elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2 ||
         elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ;
}

Status RestorePackedBMetadata(const void* metadata_buffer,
                              size_t metadata_size,
                              size_t quantized_b_buffer_size,
                              size_t b_scale_buffer_size,
                              TensorShape& b_shape,
                              size_t& quantized_b_size,
                              size_t& b_scale_count,
                              Fp8Mode& b_type,
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
  ORT_RETURN_IF(metadata[kPackedBMetadataScaleCountIndex] <= 0,
                "DynamicQuantMatMulFp8 shared prepacked B metadata has an invalid B scale count.");
  const int64_t restored_fp8_mode = metadata[kPackedBMetadataFp8ModeIndex];
  ORT_RETURN_IF(restored_fp8_mode < static_cast<int64_t>(Fp8Mode::E4M3Inf) ||
                    restored_fp8_mode >= static_cast<int64_t>(Fp8Mode::End),
                "DynamicQuantMatMulFp8 shared prepacked B metadata has an invalid FP8 type.");

  const size_t rows = static_cast<size_t>(metadata[kPackedBMetadataRowsIndex]);
  const size_t cols = static_cast<size_t>(metadata[kPackedBMetadataColsIndex]);
  const size_t expected_quantized_b_size = SafeMul<size_t>(rows, cols);
  const size_t restored_quantized_b_size = static_cast<size_t>(metadata[kPackedBMetadataSizeIndex]);
  ORT_RETURN_IF(restored_quantized_b_size != expected_quantized_b_size ||
                    restored_quantized_b_size != quantized_b_buffer_size,
                "DynamicQuantMatMulFp8 shared prepacked B metadata does not match the B buffer size.");
  const size_t restored_b_scale_count = static_cast<size_t>(metadata[kPackedBMetadataScaleCountIndex]);
  ORT_RETURN_IF(restored_b_scale_count > std::numeric_limits<size_t>::max() / sizeof(float) ||
                    restored_b_scale_count * sizeof(float) != b_scale_buffer_size,
                "DynamicQuantMatMulFp8 shared prepacked B metadata does not match the B scale buffer size.");

  b_shape = TensorShape({metadata[kPackedBMetadataRowsIndex], metadata[kPackedBMetadataColsIndex]});
  quantized_b_size = restored_quantized_b_size;
  b_scale_count = restored_b_scale_count;
  b_type = static_cast<Fp8Mode>(metadata[kPackedBMetadataFp8ModeIndex]);
  has_b_type = true;
  return Status::OK();
}

// Reject invalid scales before quantization divides by them or the GEMM dequantizes with them.
template <typename T>
Status ValidatePositiveFiniteScaleTensorImpl(const Tensor& scale, const char* scale_name) {
  const auto* data = scale.Data<T>();
  const size_t count = static_cast<size_t>(scale.Shape().Size());

  for (size_t i = 0; i < count; ++i) {
    const float value = static_cast<float>(data[i]);
    ORT_RETURN_IF(!std::isfinite(value) || value <= 0.0f,
                  "DynamicQuantMatMulFp8 requires ", scale_name, " values to be finite and positive.");
  }

  return Status::OK();
}

Status ValidatePositiveFiniteScaleTensor(const Tensor& scale, const char* scale_name) {
  if (scale.IsDataType<float>()) {
    return ValidatePositiveFiniteScaleTensorImpl<float>(scale, scale_name);
  }

  if (scale.IsDataType<MLFloat16>()) {
    return ValidatePositiveFiniteScaleTensorImpl<MLFloat16>(scale, scale_name);
  }

  if (scale.IsDataType<BFloat16>()) {
    return ValidatePositiveFiniteScaleTensorImpl<BFloat16>(scale, scale_name);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "DynamicQuantMatMulFp8 requires ", scale_name,
                         " input to be float, float16, or bfloat16.");
}

uint8_t FloatToFp8Byte(float value, Fp8Mode mode) {
  switch (mode) {
    case Fp8Mode::E4M3Inf:
      return Float8E4M3FN(value, true).val;
    case Fp8Mode::E4M3Sat:
      return Float8E4M3FNUZ(value, true).val;
    case Fp8Mode::E5M2Inf:
      return Float8E5M2(value, true).val;
    case Fp8Mode::E5M2Sat:
      return Float8E5M2FNUZ(value, true).val;
    default:
      ORT_THROW("Unsupported FP8 mode.");
  }
}

float Fp8ByteToFloat(uint8_t value, Fp8Mode mode) {
  switch (mode) {
    case Fp8Mode::E4M3Inf:
      return static_cast<float>(Float8E4M3FN(value, Float8E4M3FN::FromBits()));
    case Fp8Mode::E4M3Sat:
      return static_cast<float>(Float8E4M3FNUZ(value, Float8E4M3FNUZ::FromBits()));
    case Fp8Mode::E5M2Inf:
      return static_cast<float>(Float8E5M2(value, Float8E5M2::FromBits()));
    case Fp8Mode::E5M2Sat:
      return static_cast<float>(Float8E5M2FNUZ(value, Float8E5M2FNUZ::FromBits()));
    default:
      ORT_THROW("Unsupported FP8 mode.");
  }
}

Status GetFp8MaxAbs(Fp8Mode mode, float& max_abs) {
  switch (mode) {
    case Fp8Mode::E4M3Inf:
    case Fp8Mode::E4M3Sat:
      max_abs = 448.0f;
      return Status::OK();
    case Fp8Mode::E5M2Inf:
    case Fp8Mode::E5M2Sat:
      max_abs = 57344.0f;
      return Status::OK();
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported fp8 mode for DynamicQuantMatMulFp8.");
  }
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
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], Fp8Mode::E4M3Inf)));
    }
  } else if (zero_point.IsDataType<Float8E4M3FNUZ>()) {
    const auto* zp = static_cast<const uint8_t*>(zero_point.DataRaw());
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], Fp8Mode::E4M3Sat)));
    }
  } else if (zero_point.IsDataType<Float8E5M2>()) {
    const auto* zp = static_cast<const uint8_t*>(zero_point.DataRaw());
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], Fp8Mode::E5M2Inf)));
    }
  } else if (zero_point.IsDataType<Float8E5M2FNUZ>()) {
    const auto* zp = static_cast<const uint8_t*>(zero_point.DataRaw());
    for (size_t i = 0; i < actual_count; ++i) {
      ORT_RETURN_IF_ERROR(reject_non_zero(Fp8ByteToFloat(zp[i], Fp8Mode::E5M2Sat)));
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
void QuantizeBlockwiseFp8ABlockDynamic(const SrcT* src,
                                       size_t K,
                                       size_t block_size_k,
                                       size_t blocks_k,
                                       size_t row,
                                       size_t block_k,
                                       float fp8_max_abs,
                                       Fp8Mode mode,
                                       uint8_t* dst,
                                       float* scales) {
  const size_t k_begin = block_k * block_size_k;
  const size_t k_end = std::min(K, k_begin + block_size_k);
  const size_t idx = row * blocks_k + block_k;

  // Use one dynamic quantization scale per A row and K block. The scale depends on
  // the block max_abs, so this reference path scans first and quantizes in a second pass.
  float max_abs = 0.0f;
  const size_t row_offset = row * K;
  for (size_t k = k_begin; k < k_end; ++k) {
    max_abs = std::max(max_abs, std::fabs(static_cast<float>(src[row_offset + k])));
  }

  const float scale = max_abs == 0.0f ? 1.0f : max_abs / fp8_max_abs;
  scales[idx] = scale;

  for (size_t k = k_begin; k < k_end; ++k) {
    const float value = static_cast<float>(src[row_offset + k]);
    const float quantized = value / scale;
    dst[row_offset + k] = FloatToFp8Byte(quantized, mode);
  }
}

template <typename SrcT, typename Fp8T>
void QuantizeBlockwiseFp8WithScales(const SrcT* src,
                                    size_t K,
                                    size_t N,
                                    size_t block_size_k,
                                    size_t block_size_n,
                                    const float* scales,
                                    uint8_t* dst) {
  // Block sizes come from op attributes; scale shapes only provide the number of blocks.
  const size_t blocks_k = K / block_size_k;
  for (size_t k = 0; k < K; ++k) {
    const size_t block_k = k / block_size_k;
    const size_t row_offset = k * N;
    for (size_t n = 0; n < N; ++n) {
      const size_t block_n = n / block_size_n;
      const size_t scale_idx = block_n * blocks_k + block_k;
      const float scale = scales[scale_idx];
      const float value = static_cast<float>(src[row_offset + n]);
      const float quantized = value / scale;
      const Fp8T fp8_value(quantized, true);
      dst[row_offset + n] = fp8_value.val;
    }
  }
}

template <typename SrcT>
void ComputeBlockwiseScalesFromInput(const SrcT* src,
                                     size_t K,
                                     size_t N,
                                     size_t block_size_k,
                                     size_t block_size_n,
                                     float fp8_max_abs,
                                     float* scales) {
  // Reference-style dynamic quantization: derive one positive scale from each source block.
  const size_t blocks_k = K / block_size_k;
  const size_t blocks_n = N / block_size_n;
  for (size_t block_k = 0; block_k < blocks_k; ++block_k) {
    const size_t k_begin = block_k * block_size_k;
    const size_t k_end = k_begin + block_size_k;
    for (size_t block_n = 0; block_n < blocks_n; ++block_n) {
      const size_t n_begin = block_n * block_size_n;
      const size_t n_end = n_begin + block_size_n;
      float max_abs = 0.0f;
      for (size_t k = k_begin; k < k_end; ++k) {
        const size_t row_offset = k * N;
        for (size_t n = n_begin; n < n_end; ++n) {
          max_abs = std::max(max_abs, std::fabs(static_cast<float>(src[row_offset + n])));
        }
      }
      // Use one scale per N block and K block. Quantization runs after all block
      // scales are known, so this reference path intentionally reads B in two phases.
      scales[block_n * blocks_k + block_k] = max_abs == 0.0f ? 1.0f : max_abs / fp8_max_abs;
    }
  }
}

template <typename SrcT>
Status QuantizeToFp8ByModeWithScales(Fp8Mode fp8_mode,
                                     const SrcT* src,
                                     size_t K,
                                     size_t N,
                                     size_t block_size_k,
                                     size_t block_size_n,
                                     const float* scales,
                                     uint8_t* dst) {
  // Dispatch quantization using the requested FP8 mode and runtime block sizes.
  switch (fp8_mode) {
    case Fp8Mode::E4M3Inf:
      QuantizeBlockwiseFp8WithScales<SrcT, Float8E4M3FN>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    case Fp8Mode::E4M3Sat:
      QuantizeBlockwiseFp8WithScales<SrcT, Float8E4M3FNUZ>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    case Fp8Mode::E5M2Inf:
      QuantizeBlockwiseFp8WithScales<SrcT, Float8E5M2>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    case Fp8Mode::E5M2Sat:
      QuantizeBlockwiseFp8WithScales<SrcT, Float8E5M2FNUZ>(src, K, N, block_size_k, block_size_n, scales, dst);
      return Status::OK();
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unsupported fp8 mode for DynamicQuantMatMulFp8.");
  }
}

struct Fp8GemmShapeParams {
  size_t M;
  size_t N;
  size_t K;
  Fp8Mode Fp8Type;
};

struct Fp8GemmDataParams {
  const void* A = nullptr;
  size_t lda = 0;
  const void* B = nullptr;
  size_t ldb = 0;
  void* C = nullptr;
  size_t ldc = 0;
  const float* ScaleA = nullptr;
  const float* ScaleB = nullptr;
  const float* ScaleY = nullptr;
  size_t BlockSizeM = 0;
  size_t BlockSizeK = 0;
  size_t BlockSizeN = 0;
  size_t ScaleAStrideK = 0;
  size_t ScaleAStrideM = 0;
  size_t ScaleBStrideK = 0;
  size_t ScaleBStrideN = 0;
};

Status ReferenceFp8GemmBatch(const Fp8GemmShapeParams& shape,
                             const Fp8GemmDataParams* data_params,
                             size_t batch_count,
                             concurrency::ThreadPool* thread_pool) {
  const size_t M = shape.M;
  const size_t N = shape.N;
  const size_t K = shape.K;

  if (batch_count == 0 || M == 0 || N == 0) {
    return Status::OK();
  }

  size_t work_items_size = 0;
  ORT_RETURN_IF(batch_count > std::numeric_limits<size_t>::max() / M,
                "DynamicQuantMatMulFp8 scalar GEMM work item count overflow.");
  work_items_size = batch_count * M;
  ORT_RETURN_IF(work_items_size > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()),
                "DynamicQuantMatMulFp8 scalar GEMM work item count exceeds ptrdiff_t range.");
  const auto work_items = static_cast<std::ptrdiff_t>(work_items_size);

  const TensorOpCost unit_cost{
      static_cast<double>(SafeMul<size_t>(K, sizeof(uint8_t)) * 2),
      static_cast<double>(N * sizeof(float)),
      static_cast<double>(SafeMul<size_t>(K, N) * 2)};
  concurrency::ThreadPool::TryParallelFor(thread_pool, work_items, unit_cost,
                                          [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
                                            for (std::ptrdiff_t tid = begin; tid < end; ++tid) {
                                              const size_t batch = static_cast<size_t>(tid) / M;
                                              const size_t m = static_cast<size_t>(tid) % M;
                                              const auto& params = data_params[batch];

                                              const auto* a_fp8 = static_cast<const uint8_t*>(params.A);
                                              const auto* b_fp8 = static_cast<const uint8_t*>(params.B);
                                              auto* c = static_cast<float*>(params.C);
                                              const auto* scale_a = params.ScaleA;
                                              const auto* scale_b = params.ScaleB;

                                              const size_t block_m = m / params.BlockSizeM;
                                              for (size_t n = 0; n < N; ++n) {
                                                const size_t block_n = n / params.BlockSizeN;
                                                float acc = 0.0f;
                                                for (size_t k = 0; k < K; ++k) {
                                                  const size_t block_k = k / params.BlockSizeK;
                                                  const size_t a_scale_idx =
                                                      block_m * params.ScaleAStrideM + block_k * params.ScaleAStrideK;
                                                  const size_t b_scale_idx =
                                                      block_k * params.ScaleBStrideK + block_n * params.ScaleBStrideN;
                                                  const float scale_a_val = scale_a ? scale_a[a_scale_idx] : 1.0f;
                                                  const float scale_b_val = scale_b ? scale_b[b_scale_idx] : 1.0f;
                                                  const float a_val =
                                                      Fp8ByteToFloat(a_fp8[m * params.lda + k], shape.Fp8Type);
                                                  const float b_val =
                                                      Fp8ByteToFloat(b_fp8[k * params.ldb + n], shape.Fp8Type);
                                                  acc += (a_val * scale_a_val) * (b_val * scale_b_val);
                                                }

                                                if (params.ScaleY != nullptr) {
                                                  acc *= params.ScaleY[0];
                                                }
                                                c[m * params.ldc + n] = acc;
                                              }
                                            }
                                          });
  return Status::OK();
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
  const int64_t block_size_k = info.GetAttrOrDefault<int64_t>("block_size_k", kDefaultBlockSize);
  const int64_t block_size_n = info.GetAttrOrDefault<int64_t>("block_size_n", kDefaultBlockSize);
  const int64_t fp8_type =
      info.GetAttrOrDefault<int64_t>("fp8_type", ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN);
  ORT_ENFORCE(block_size_k > 0,
              "DynamicQuantMatMulFp8 requires block_size_k to be greater than zero.");
  ORT_ENFORCE(block_size_n > 0,
              "DynamicQuantMatMulFp8 requires block_size_n to be greater than zero.");
  block_size_k_ = static_cast<size_t>(block_size_k);
  block_size_n_ = static_cast<size_t>(block_size_n);
  ORT_THROW_IF_ERROR(GetFp8Type(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(fp8_type), fp8_type_));
}

Status DynamicQuantMatMulFp8::GetFp8Type(const Tensor& tensor, Fp8Mode& out_type) {
  return GetFp8Type(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor.GetElementType()), out_type);
}

Status DynamicQuantMatMulFp8::GetFp8Type(ONNX_NAMESPACE::TensorProto_DataType elem_type,
                                         Fp8Mode& out_type) {
  switch (elem_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN:
      out_type = Fp8Mode::E4M3Inf;
      return Status::OK();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ:
      out_type = Fp8Mode::E4M3Sat;
      return Status::OK();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2:
      out_type = Fp8Mode::E5M2Inf;
      return Status::OK();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ:
      out_type = Fp8Mode::E5M2Sat;
      return Status::OK();
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported fp8 type for DynamicQuantMatMulFp8.");
  }
}

Status DynamicQuantMatMulFp8::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                                      /*out*/ bool& is_packed,
                                      /*out*/ PrePackedWeights* prepacked_weights) {
  is_packed = false;

  const Tensor* constant_b = nullptr;
  const bool model_b_scale_is_ignored =
      OpKernel::Info().TryGetConstantInput(GetBIdx(), &constant_b) &&
      !IsFp8DataType(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(constant_b->GetElementType()));

  if (input_idx == IN_B_SCALE) {
    // Non-FP8 constant B computes its own prepacked scales, so model B_scale values are not consumed.
    if (!model_b_scale_is_ignored) {
      ORT_RETURN_IF_ERROR(ValidatePositiveFiniteScaleTensor(tensor, "B scale"));
      constant_b_scale_values_validated_ = true;
    }
    return Status::OK();
  }

  if (input_idx == IN_B_ZERO_POINT) {
    ORT_RETURN_IF_ERROR(ValidateZeroPointValuesAreZero(
        tensor, static_cast<size_t>(tensor.Shape().Size()), "B zero point"));
    constant_b_zero_point_values_validated_ = true;
    return Status::OK();
  }

  if (input_idx != GetBIdx()) {
    return Status::OK();
  }

  b_shape_ = tensor.Shape();
  if (b_shape_.NumDimensions() != 2) {
    return Status::OK();
  }

  const size_t K = static_cast<size_t>(b_shape_[0]);
  const size_t N = static_cast<size_t>(b_shape_[1]);
  const auto b_elem_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(tensor.GetElementType());
  const bool b_is_fp8 = IsFp8DataType(b_elem_type);
  if (b_is_fp8) {
    ORT_RETURN_IF_ERROR(GetFp8Type(tensor, b_type_));
    has_b_type_ = true;
    return Status::OK();
  }

  b_type_ = fp8_type_;
  has_b_type_ = true;
  if (K == 0 || N == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(K % block_size_k_ == 0,
                    "DynamicQuantMatMulFp8 requires K to be divisible by block_size_k.");
  ORT_RETURN_IF_NOT(N % block_size_n_ == 0,
                    "DynamicQuantMatMulFp8 requires N to be divisible by block_size_n.");
  const size_t blocks_k = K / block_size_k_;
  const size_t blocks_n = N / block_size_n_;
  b_scale_count_ = SafeMul<size_t>(blocks_k, blocks_n);
  b_scales_ = IAllocator::MakeUniquePtr<void>(alloc, b_scale_count_ * sizeof(float), true);
  auto* prepacked_b_scales = static_cast<float*>(b_scales_.get());
  float fp8_max_abs = 0.0f;
  ORT_RETURN_IF_ERROR(GetFp8MaxAbs(b_type_, fp8_max_abs));

  const size_t quantized_b_size = SafeMul<size_t>(K, N);
  quantized_b_ = IAllocator::MakeUniquePtr<void>(alloc, quantized_b_size, true);
  quantized_b_size_ = quantized_b_size;
  auto* quantized_b_bytes = static_cast<uint8_t*>(quantized_b_.get());
  if (tensor.IsDataType<float>()) {
    ComputeBlockwiseScalesFromInput(tensor.Data<float>(), K, N, block_size_k_, block_size_n_,
                                    fp8_max_abs, prepacked_b_scales);
    ORT_RETURN_IF_ERROR(QuantizeToFp8ByModeWithScales(b_type_, tensor.Data<float>(), K, N, block_size_k_, block_size_n_,
                                                      prepacked_b_scales, quantized_b_bytes));
  } else if (tensor.IsDataType<MLFloat16>()) {
    ComputeBlockwiseScalesFromInput(tensor.Data<MLFloat16>(), K, N, block_size_k_, block_size_n_,
                                    fp8_max_abs, prepacked_b_scales);
    ORT_RETURN_IF_ERROR(QuantizeToFp8ByModeWithScales(b_type_, tensor.Data<MLFloat16>(), K, N, block_size_k_, block_size_n_,
                                                      prepacked_b_scales, quantized_b_bytes));
  } else if (tensor.IsDataType<BFloat16>()) {
    ComputeBlockwiseScalesFromInput(tensor.Data<BFloat16>(), K, N, block_size_k_, block_size_n_,
                                    fp8_max_abs, prepacked_b_scales);
    ORT_RETURN_IF_ERROR(QuantizeToFp8ByModeWithScales(b_type_, tensor.Data<BFloat16>(), K, N, block_size_k_, block_size_n_,
                                                      prepacked_b_scales, quantized_b_bytes));
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
        static_cast<int64_t>(b_scale_count_),
        static_cast<int64_t>(b_type_),
    };
    auto metadata = IAllocator::MakeUniquePtr<void>(alloc, kPackedBMetadataSize, true);
    std::memcpy(metadata.get(), metadata_values.data(), kPackedBMetadataSize);
    prepacked_weights->buffers_.push_back(std::move(quantized_b_));
    prepacked_weights->buffer_sizes_.push_back(quantized_b_size_);
    prepacked_weights->buffers_.push_back(std::move(b_scales_));
    prepacked_weights->buffer_sizes_.push_back(b_scale_count_ * sizeof(float));
    prepacked_weights->buffers_.push_back(std::move(metadata));
    prepacked_weights->buffer_sizes_.push_back(kPackedBMetadataSize);
  }
  is_packed = true;
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

  ORT_RETURN_IF(prepacked_buffers.size() != 3 || prepacked_buffer_sizes.size() != 3,
                "DynamicQuantMatMulFp8 requires shared prepacked B data, scale, and metadata buffers.");
  ORT_RETURN_IF(prepacked_buffers[0].get() == nullptr,
                "DynamicQuantMatMulFp8 requires shared prepacked B data.");
  ORT_RETURN_IF(prepacked_buffers[1].get() == nullptr,
                "DynamicQuantMatMulFp8 requires shared prepacked B scales.");

  // Buffer 0 owns quantized B bytes; buffer 1 owns computed B scales; buffer 2 restores kernel state.
  ORT_RETURN_IF_ERROR(RestorePackedBMetadata(prepacked_buffers[2].get(),
                                             prepacked_buffer_sizes[2],
                                             prepacked_buffer_sizes[0],
                                             prepacked_buffer_sizes[1],
                                             b_shape_,
                                             quantized_b_size_,
                                             b_scale_count_,
                                             b_type_,
                                             has_b_type_));
  quantized_b_ = std::move(prepacked_buffers[0]);
  b_scales_ = std::move(prepacked_buffers[1]);
  used_shared_buffers = true;
  return Status::OK();
}

Status DynamicQuantMatMulFp8::Compute(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(IN_A);
  const Tensor* b = quantized_b_ ? nullptr : context->Input<Tensor>(IN_B);
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
    ORT_RETURN_IF(!std::isfinite(y_scale_data[0]) || y_scale_data[0] <= 0.0f,
                  "DynamicQuantMatMulFp8 requires Y scale values to be finite and positive.");
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
  const bool b_is_fp8 = IsFp8DataType(b_elem_type);

  Fp8Mode b_type{};
  if (has_b_type_) {
    b_type = b_type_;
  } else if (b_is_fp8) {
    ORT_RETURN_IF_ERROR(GetFp8Type(b_elem_type, b_type));
  } else {
    b_type = fp8_type_;
  }

  if (b_zero_point != nullptr) {
    const auto b_zp_elem_type =
        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(b_zero_point->GetElementType());
    Fp8Mode b_zp_type{};
    ORT_RETURN_IF_ERROR(GetFp8Type(b_zp_elem_type, b_zp_type));
    ORT_RETURN_IF(b_type != b_zp_type,
                  "DynamicQuantMatMulFp8 requires B and B zero point FP8 types to match.");
  }

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
  ORT_RETURN_IF(K % block_size_k_ != 0,
                "DynamicQuantMatMulFp8 requires K to be divisible by block_size_k.");
  const size_t expected_blocks_k = K / block_size_k_;
  const size_t blocks_m = M;
  const size_t blocks_k = expected_blocks_k;

  const bool uses_model_b_scale = b_scales_ == nullptr;
  ORT_RETURN_IF(uses_model_b_scale && b_scale == nullptr,
                "DynamicQuantMatMulFp8 requires B scale when B is already FP8.");
  ORT_RETURN_IF(uses_model_b_scale && b_scale->Shape().NumDimensions() != 2,
                "DynamicQuantMatMulFp8 requires B scale to be a 2D tensor.");
  ORT_RETURN_IF(N % block_size_n_ != 0,
                "DynamicQuantMatMulFp8 requires N to be divisible by block_size_n.");
  const size_t blocks_n = N / block_size_n_;
  ORT_RETURN_IF(blocks_n == 0, "DynamicQuantMatMulFp8 requires non-zero B scale N dimension.");
  ORT_RETURN_IF(uses_model_b_scale && static_cast<size_t>(b_scale->Shape()[0]) != blocks_n,
                "DynamicQuantMatMulFp8 requires B scale N dimension to be N / block_size_n.");
  ORT_RETURN_IF(uses_model_b_scale && static_cast<size_t>(b_scale->Shape()[1]) != blocks_k,
                "DynamicQuantMatMulFp8 requires B scale K dimension to be K / block_size_k.");

  const size_t a_scale_batch_stride = SafeMul<size_t>(blocks_m, blocks_k);
  const size_t b_zp_count = SafeMul<size_t>(blocks_k, blocks_n);

  if (uses_model_b_scale && b_zero_point != nullptr) {
    ORT_RETURN_IF(b_zero_point->Shape().NumDimensions() != 2,
                  "DynamicQuantMatMulFp8 requires B zero point to be a 2D tensor.");
    ORT_RETURN_IF(b_zero_point->Shape()[0] != static_cast<int64_t>(blocks_n) ||
                      b_zero_point->Shape()[1] != static_cast<int64_t>(blocks_k),
                  "DynamicQuantMatMulFp8 requires B zero point to have shape [N / block_size_n, K / block_size_k].");
    if (!constant_b_zero_point_values_validated_) {
      ORT_RETURN_IF_ERROR(ValidateZeroPointValuesAreZero(*b_zero_point, b_zp_count, "B zero point"));
    }
  }

  AllocatorPtr temp_allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&temp_allocator));

  const float* b_scales = nullptr;
  IAllocatorUniquePtr<float> b_scale_float;
  size_t b_scale_elems = 0;
  if (b_scales_) {
    b_scales = static_cast<const float*>(b_scales_.get());
    b_scale_elems = b_scale_count_;
  } else if (b_scale->IsDataType<float>()) {
    b_scales = b_scale->Data<float>();
    b_scale_elems = static_cast<size_t>(b_scale->Shape().Size());
  } else {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    b_scale_elems = static_cast<size_t>(b_scale->Shape().Size());
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

  // The internal FP8 GEMM helper accumulates and stores float output. Use scratch for lower-precision Y,
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

  Fp8GemmShapeParams gemm_shape;
  gemm_shape.M = M;
  gemm_shape.N = N;
  gemm_shape.K = K;
  gemm_shape.Fp8Type = b_type;

  if (uses_model_b_scale && !constant_b_scale_values_validated_) {
    for (size_t i = 0; i < b_scale_elems; ++i) {
      ORT_RETURN_IF(!std::isfinite(b_scales[i]) || b_scales[i] <= 0.0f,
                    "DynamicQuantMatMulFp8 requires B scale values to be finite and positive.");
    }
  }

  const size_t a_fp8_size = SafeMul<size_t>(M, K);
  const size_t a_num_elements = static_cast<size_t>(a->Shape().Size());
  ORT_RETURN_IF(a_num_elements % a_fp8_size != 0,
                "DynamicQuantMatMulFp8 requires A to contain complete MxK matrices.");
  const size_t a_batch_count = a_num_elements / a_fp8_size;

  // Quantize the physical A tensor once. Broadcasted output GEMMs then reuse the same FP8 A slice.
  auto a_fp8_buffer = IAllocator::MakeUniquePtr<uint8_t>(temp_allocator, a_num_elements, true);
  const size_t a_scale_count = SafeMul<size_t>(a_batch_count, a_scale_batch_stride);
  auto a_scale_buffer = IAllocator::MakeUniquePtr<float>(temp_allocator, a_scale_count, true);
  const size_t a_quant_work_items = SafeMul<size_t>(a_batch_count, a_scale_batch_stride);
  ORT_RETURN_IF(a_quant_work_items > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                "DynamicQuantMatMulFp8 A quantization work item count exceeds ptrdiff_t range.");
  const auto a_quant_work_items_i = static_cast<std::ptrdiff_t>(a_quant_work_items);
  const size_t a_quant_block_elems = block_size_k_;
  const TensorOpCost a_quant_unit_cost{
      static_cast<double>(SafeMul<size_t>(a_quant_block_elems, sizeof(float))),
      static_cast<double>(SafeMul<size_t>(a_quant_block_elems, sizeof(uint8_t))),
      static_cast<double>(a_quant_block_elems) * 2.0};
  float fp8_max_abs = 0.0f;
  ORT_RETURN_IF_ERROR(GetFp8MaxAbs(b_type, fp8_max_abs));
  const auto quantize_a_batches = [&](const auto* a_data) {
    concurrency::ThreadPool::TryParallelFor(context->GetOperatorThreadPool(), a_quant_work_items_i,
                                            a_quant_unit_cost,
                                            [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
                                              for (std::ptrdiff_t tid = begin; tid < end; ++tid) {
                                                const size_t work_idx = static_cast<size_t>(tid);
                                                const size_t a_batch_idx = work_idx / a_scale_batch_stride;
                                                const size_t scale_block_idx = work_idx % a_scale_batch_stride;
                                                const size_t row = scale_block_idx / blocks_k;
                                                const size_t block_k = scale_block_idx % blocks_k;
                                                const size_t a_batch_offset = a_batch_idx * a_fp8_size;
                                                const size_t a_scale_batch_offset = a_batch_idx * a_scale_batch_stride;
                                                QuantizeBlockwiseFp8ABlockDynamic(
                                                    a_data + a_batch_offset,
                                                    K, block_size_k_, blocks_k,
                                                    row, block_k, fp8_max_abs, b_type,
                                                    a_fp8_buffer.get() + a_batch_offset,
                                                    a_scale_buffer.get() + a_scale_batch_offset);
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

  std::vector<Fp8GemmDataParams> gemm_data_vec(num_gemms);
  for (size_t gemm_idx = 0; gemm_idx < num_gemms; ++gemm_idx) {
    const size_t a_offset = helper.LeftOffsets()[gemm_idx];
    ORT_RETURN_IF(a_offset >= a_num_elements || (a_offset % a_fp8_size) != 0,
                  "DynamicQuantMatMulFp8 requires A offsets to reference complete MxK matrices.");
    const size_t scale_batch_index = a_offset / a_fp8_size;
    ORT_RETURN_IF(scale_batch_index >= a_batch_count,
                  "DynamicQuantMatMulFp8 requires A offsets to reference complete MxK matrices.");
    const size_t a_scale_batch_offset = SafeMul<size_t>(scale_batch_index, a_scale_batch_stride);
    const float* a_scales_batch = a_scale_buffer.get() + a_scale_batch_offset;
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
    gemm_data.BlockSizeM = 1;
    gemm_data.BlockSizeK = block_size_k_;
    gemm_data.BlockSizeN = block_size_n_;
    gemm_data.ScaleAStrideK = 1;
    gemm_data.ScaleAStrideM = blocks_k;
    gemm_data.ScaleBStrideN = blocks_k;
    gemm_data.ScaleBStrideK = 1;
  }

  ORT_RETURN_IF_ERROR(ReferenceFp8GemmBatch(gemm_shape, gemm_data_vec.data(), num_gemms,
                                            context->GetOperatorThreadPool()));

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
