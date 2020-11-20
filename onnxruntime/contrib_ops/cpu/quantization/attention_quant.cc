// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/attention_cpu_base.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/qmath.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

#ifdef USE_FBGEMM
#define FBGEMM_STATIC
#include <vector>
#include "fbgemm/Fbgemm.h"
#include "fbgemm/QuantUtils.h"
using namespace fbgemm;
#endif // USE_FBGEMM

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
class QAttention : public OpKernel, public AttentionCPUBase {
 public:
  QAttention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

#if defined(MLAS_SUPPORTS_PACKED_GEMM_U8X8) || defined(USE_FBGEMM)
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;
#endif

 private:
  BufferUniquePtr packed_weights_;
  size_t packed_weights_size_;
  TensorShape weight_shape_;
  bool weights_is_signed_;
#ifdef USE_FBGEMM
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> packed_weight_class_;
  BufferUniquePtr weight_col_offsets_;
  mutable bool zero_offset_applied_ = false;
#endif // USE_FBGEMM
};

#ifdef USE_FBGEMM
// This function computes the offset values for each column which are used for compensating the remainders of quantized values
// More detailed math is avilable in the FBGEMM's blog - https://engineering.fb.com/ml-applications/fbgemm/
inline void colOffsetsWithoutZeroPtS8acc32(
    bool transpose,
    int K,
    int N,
    const int8_t* Bint8,
    int32_t* col_offsets) {
  for (int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
    }
    col_offsets[n] = sum;
  }
}
#endif // USE_FBGEMM

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QAttention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QAttention<float>);

template <typename T>
QAttention<T>::QAttention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info) {}

#if defined(MLAS_SUPPORTS_PACKED_GEMM_U8X8) || defined(USE_FBGEMM)
template <typename T>
Status QAttention<T>::PrePack(const Tensor& weights, int input_idx, bool& is_packed) {
  is_packed = false;

  if (1 != input_idx) {
    return Status::OK();
  }

  weight_shape_ = weights.Shape();
  const auto& weights_dims = weight_shape_.GetDims();
  if (weights_dims.size() != 2) {
    return Status::OK();
  }

  const size_t hidden_size = static_cast<size_t>(weights_dims[0]);
  const size_t hidden_size_x3 = static_cast<size_t>(weights_dims[1]);
  const size_t head_size = hidden_size / num_heads_;

  // Bail out if the weights shape has an expected shape.
  if ((hidden_size == 0) || ((hidden_size % num_heads_) != 0) || (hidden_size_x3 != 3 * hidden_size)) {
    return Status::OK();
  }

  const auto* weights_data = static_cast<const uint8_t*>(weights.DataRaw());
  weights_is_signed_ = weights.IsDataType<int8_t>();

#ifndef USE_FBGEMM
  packed_weights_size_ = MlasGemmPackBSize(head_size_, hidden_size, weights_is_signed_);
  if (packed_weights_size_ == 0) {
    return Status::OK();
  }

  const size_t loop_len = 3 * num_heads_;
  auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
  auto* packed_weights_data = static_cast<uint8_t*>(alloc->Alloc(packed_weights_size_ * loop_len));
  packed_weights_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));

  for (size_t i = 0; i < loop_len; i++) {
    MlasGemmPackB(head_size_, hidden_size, weights_data, hidden_size_x3, weights_is_signed_, packed_weights_data);
    packed_weights_data += packed_weights_size_;
    weights_data += head_size_;
  }
#else // USE_FBGEMM
  packed_weights_size_ = fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t>, int8_t>::packedBufferSize(hidden_size, hidden_size_x3);
  if (packed_weights_size_ == 0) {
    return Status::OK();
  }

  // Allocate memory for packed matrix
  auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
  auto* packed_weights_data = static_cast<int8_t*>(alloc->Alloc(packed_weights_size_));
  packed_weights_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));

  // fbgemm packed B class
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> packedB(new fbgemm::PackBMatrix<int8_t>(
      fbgemm::matrix_op_t::NoTranspose, hidden_size, hidden_size_x3, (int8_t*)weights_data, hidden_size_x3, packed_weights_data, 1));
  packed_weight_class_ = std::move(packedB);

  // Column offsets
  auto* col_offset_data = static_cast<int32_t*>(alloc->Alloc(hidden_size_x3*sizeof(int32_t)));
  weight_col_offsets_ = BufferUniquePtr(col_offset_data, BufferDeleter(alloc));

  colOffsetsWithoutZeroPtS8acc32(
      false,
      hidden_size,
      hidden_size_x3,
      (int8_t*)weights_data,
      (int32_t*)col_offset_data);
#endif // USE_FBGEMM

  is_packed = true;
  return Status::OK();
}
#endif

template <typename T>
Status QAttention<T>::Compute(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input  0 - input             : (batch_size, sequence_length, hidden_size)
  //   Input  1 - weights           : (hidden_size, 3 * hidden_size)
  //   Input  2 - bias              : (3 * hidden_size)
  //   Input  3 - input_scale       : scalar
  //   Input  4 - weight_scale      : scalar for MLAS, (3 * hidden_size) for FBGEMMM
  //   Input  5 - mask_index        : nullptr, (batch_size), (2 * batch_size), (batch_size, 1), (1, 1) or (batch_size, past_sequence_length + sequence_length)
  //   Input  6 - input_zero_point  : scalar
  //   Input  7 - weight_zero_point : scalar for MLAS, (3 * hidden_size) for FBGEMMM
  //   Input  8 - past              : (2, batch_size, num_heads, past_sequence_length, head_size_)
  //   Output 0                     : (batch_size, sequence_length, hidden_size)
  //   Output 1 - present           : (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size_)
  //   ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = packed_weights_ ? nullptr : context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* input_scale_tensor = context->Input<Tensor>(3);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(5);
  const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);
  const Tensor* past_tensor = context->Input<Tensor>(8);

  ORT_RETURN_IF_ERROR(AttentionBase::CheckInputs(input->Shape(),
                                                 packed_weights_ ? weight_shape_ : weights->Shape(),
                                                 bias->Shape(),
                                                 mask_index,
                                                 past_tensor));

  const auto& shape = input->Shape();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int hidden_size = static_cast<int>(shape[2]);

  // For the head-pruned transformers, hidden_size != head_size_ * num_heads_
  int64_t output_shape_arr[] = {batch_size, sequence_length, head_size_ * num_heads_};
  TensorShape output_shape(output_shape_arr, 3);
  Tensor* output = context->Output(0, output_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  constexpr size_t element_size = sizeof(T);

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = Scale(input(BS, NH) x weights(NH, 3NH)) + bias(3NH)
  auto gemm_data = reinterpret_cast<T*>(allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * head_size_ * num_heads_ * element_size));
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + batch_size * sequence_length * hidden_size;
  auto V = K + batch_size * sequence_length * hidden_size;
  T* QKV[3] = {Q, K, V};

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  T input_scale = *(input_scale_tensor->template Data<T>());

#ifndef USE_FBGEMM
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(weight_scale_tensor),
                    "weight must be a scalar or 1D tensor of size 1");
  T weight_scale = *(weight_scale_tensor->template Data<T>());

  T dequant_scale = input_scale * weight_scale;

  uint8_t input_zero_point = 0;
  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    input_zero_point = *i_zp_tensor->template Data<uint8_t>();
  }

  uint8_t weight_zero_point = 0;
  if (w_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(w_zp_tensor),
                      "weight zero point must be a scalar or 1D tensor of size 1.");
    weight_zero_point = *static_cast<const uint8_t*>(w_zp_tensor->DataRaw());
  }

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<uint8_t>();
    const auto* bias_data = bias->template Data<T>();

    const auto* weights_data = packed_weights_ ? nullptr : static_cast<const uint8_t*>(weights->DataRaw());
    const bool weights_is_signed = packed_weights_ ? weights_is_signed_ : weights->IsDataType<int8_t>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * hidden_size;
        int weights_offset = qkv_index * hidden_size + head_index * head_size;
        float* qkv_dest = QKV[qkv_index];
        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

        //                   original           transposed            iteration
        // A: input          (BxSxNxH)          (B.)S x NH            S x NH
        // B: weights        (NxHx3xNxH)        NH  x (3.N.)H         NH x H
        // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H
#ifdef MLAS_SUPPORTS_PACKED_GEMM_U8X8
        if (packed_weights_) {
          const auto* packed_weight =
              static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);

          MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR scale_bias_processor(qkv_dest + qkv_offset,
                                                                      head_size,
                                                                      &dequant_scale,
                                                                      bias_data + weights_offset);
          MlasGemm(
              sequence_length,                                    // M      = S
              head_size,                                          // N      = H
              hidden_size,                                        // K      = NH
              input_data + input_offset,                          // A
              hidden_size,                                        // lda    = NH
              input_zero_point,                                   // input zero point
              packed_weight,                                      // B
              weight_zero_point,                                  // weight zero point
              weights_is_signed,                                  // weight data type
              reinterpret_cast<int32_t*>(qkv_dest + qkv_offset),  // C
              head_size,                                          // ldc
              nullptr,                                            // use single-thread
              &scale_bias_processor);                             // output processor

          continue;
        }
#endif
        QGemm(sequence_length,                // M      = S
              head_size,                      // N      = H
              hidden_size,                    // K      = NH
              input_data + input_offset,      // A
              hidden_size,                    // lda    = NH
              input_zero_point,               // input zero point
              weights_data + weights_offset,  // B
              3 * hidden_size,                // ldb    = 3NH
              weight_zero_point,              // weight zero point
              weights_is_signed,              // weight data type
              qkv_dest + qkv_offset,          // C
              head_size,                      // ldc
              &dequant_scale,                 // output scale
              bias_data + weights_offset,     // bias
              nullptr                         // use single-thread
        );
      }
    });
  }
#else
  const T* weight_scale = (weight_scale_tensor->template Data<T>());
  uint8_t input_zero_point = 0;
  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    input_zero_point = *i_zp_tensor->template Data<uint8_t>();
  }

  // TODO vector instruction
  int32_t* weight_zero_point = new int32_t[hidden_size*3];
  int8_t* weight_zero_point_int8 = nullptr;
  if (w_zp_tensor != nullptr) {
    weight_zero_point_int8 = (int8_t*)w_zp_tensor->template Data<int8_t>();
    for (int i = 0; i < hidden_size*3; i++)
      weight_zero_point[i] = (int32_t)weight_zero_point_int8[i];
  }

  const auto* input_data = input->template Data<uint8_t>();
  const auto* bias_data = bias->template Data<T>();

  const auto* weights_data = packed_weights_ ? nullptr : static_cast<const uint8_t*>(weights->DataRaw());

  auto gemm_intermediate = reinterpret_cast<T*>(allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * head_size_ * num_heads_ * element_size));
  BufferUniquePtr gemm_buffer1(gemm_intermediate, BufferDeleter(allocator));

  // fbgemm computation
  int32_t* col_offsets = nullptr;
  col_offsets = static_cast<int32_t*>(weight_col_offsets_.get());
  if (!zero_offset_applied_) {
    for (int i = 0; i < hidden_size * 3; i++) {
      col_offsets[i] -= weight_zero_point[i] * hidden_size;
    }
    zero_offset_applied_ = true;
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<int32_t> rowOffsetBuf(
        PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());

    PackAWithRowOffset<uint8_t> packAN(
        matrix_op_t::NoTranspose,
        batch_size * sequence_length,
        hidden_size,
        input_data,
        hidden_size,
        nullptr,
        1,
        rowOffsetBuf.data());

    DoNothing<float, float> doNothingObj{};
    ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL> outputProcObj(
        doNothingObj,
        input_scale,
        weight_scale,
        input_zero_point,
        weight_zero_point,
        packAN.getRowOffsetBuffer(),
        col_offsets,
        bias_data,
        3 * head_size_ * num_heads_);

    auto* packedB = packed_weight_class_.get();

#ifdef _OPENMP
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
#else
    int num_threads = 1;
    int tid = 0;
#endif
    // TODO check if inplace buffer is okay or not
    fbgemmPacked(
        packAN,
        *packedB,
        gemm_intermediate,
        (int32_t*) gemm_intermediate,
        3 * head_size_ * num_heads_,
        outputProcObj,
        tid,
        num_threads);
  }

  delete[] weight_zero_point;

  // Transpose gemm output
  //size_t m = batch_size * sequence_length;
  // (BxSx3xNxH) -> (3xBxNxSxH)
#ifdef _OPENMP
#ifndef _MSC_VER    // MS openMP doesn't support parallel for collapse
#pragma omp parallel for collapse(3)
#endif
#endif
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < sequence_length; j++) {
      for (size_t k = 0; k < 3; k++) {
        auto input_batch_offset = i * sequence_length * 3 * head_size_ * num_heads_;
        auto output_batch_offset = i * sequence_length * head_size_ * num_heads_;
        auto input_seq_offset = j * 3 * head_size_ * num_heads_;
        auto output_seq_offset = j * head_size_;
        auto input_qkv_offset = k * head_size_ * num_heads_;
        auto output_qkv_offset = k * batch_size * sequence_length * head_size_ * num_heads_;
        for (size_t l = 0; l < num_heads_; l++) {
          memcpy(gemm_data + output_batch_offset + output_seq_offset + output_qkv_offset + l * sequence_length * head_size_,
            gemm_intermediate + l * head_size_ + input_qkv_offset + input_seq_offset + input_batch_offset,
            head_size_ * sizeof(float));
        }
      }
    }
  }
#endif // USE_FBGEMM, FBGEMM uses column-wise quantization

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q, K, V, mask_index, past_tensor, output,
                        batch_size, sequence_length,
                        head_size_, hidden_size, context);
}

}  // namespace contrib
}  // namespace onnxruntime
