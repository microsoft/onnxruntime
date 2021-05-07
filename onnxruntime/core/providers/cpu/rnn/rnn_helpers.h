// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"
#include "core/mlas/inc/mlas.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

#include "gsl/gsl"

namespace onnxruntime {
//class Tensor;
//class OpKernelContext;

namespace rnn {
namespace detail {

enum Direction {
  kForward = 0,
  kReverse = 1,
  kBidirectional = 2
};

inline Direction MakeDirection(const std::string& direction) {
  if (direction == "forward") {
    return kForward;
  }
  if (direction == "reverse") {
    return kReverse;
  }
  if (direction == "bidirectional") {
    return kBidirectional;
  }
  ORT_THROW("Invalid 'direction' argument of '", direction,
            "'. Must be one of 'forward', 'reverse', or 'bidirectional'.");
}

/** Allocate a unique_ptr using allocator_, and return a span to the allocated memory so usage is safe
@param allocator IAllocator to use for the allocation.
@param size Allocation size. Number of elements of type TAlloc, or total size if TAlloc is 'void'.
@param unique_ptr unique_ptr that will control the lifetime of the allocated memory.
@param fill If true, fill the allocated memory with fill_value.
@param fill_value Value to use if 'fill' is true.
@returns A span to provide bounds checked access to the allocated memory.
*/
template <typename TAlloc>
gsl::span<TAlloc> Allocate(std::shared_ptr<IAllocator> allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr,
                           bool fill = false, TAlloc fill_value = TAlloc{}) {
  unique_ptr = IAllocator::MakeUniquePtr<TAlloc>(allocator, size);
  auto span = gsl::make_span(unique_ptr.get(), size);

  if (fill) {
    // Do't use span.begin() it will cause performance issue and stop compiler to optimize the code
    std::fill_n(unique_ptr.get(), size, fill_value);
  }

  return span;
}

// validate the common inputs to RNN, LSTM and GRU operators
Status ValidateCommonRnnInputs(const Tensor& X,
                               const TensorShape& W_shape,
                               const TensorShape& R_shape,
                               const Tensor* B,
                               int WRB_dim_1_multipler,  // multiplier used with hidden_size for W, R and B inputs
                               const Tensor* sequence_lens,
                               const Tensor* initial_h,
                               int64_t num_directions,
                               int64_t hidden_size);

/// Copy an input array repeatedly to an output array
/// @param input_begin Beginning of input
/// @param input_end End of input
/// @param output Output iterator
/// @param repetitions Number of times to repeat copy. Assumes output is sufficiently sized.
/// @returns Position of output iterator after copy is completed
template <typename TInIter, typename TOutIter>
TOutIter RepeatVectorToConstructArray(TInIter input_begin,
                                      TInIter input_end,
                                      TOutIter output,
                                      int64_t repetitions) {
  for (int64_t i = 0; i < repetitions; i++) {
    output = std::copy(input_begin, input_end, output);
  }

  return output;
}

// reverse an LSTM or GRU sequence which has shape [seq_length, batch_size, hidden_size]
// and output to shape [seq_length, num_directions, batch_size, hidden_size]
template <typename T>
void ReverseSequence(gsl::span<const T> inputs,
                     gsl::span<T> inputs_reverse,
                     gsl::span<const int> sequence_lengths,
                     const int max_sequence_length,
                     const int batch_size,
                     const int input_size,
                     const int num_directions,
                     concurrency::ThreadPool*) {
  for (int i = 0; i < batch_size; i++) {
    int seq_len = sequence_lengths[i];

    for (int j = 0; j < seq_len; j++) {
      gsl::span<const T> src = inputs.subspan(j * batch_size * input_size + i * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(num_directions * (seq_len - j - 1) * batch_size * input_size + i * input_size, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }

    for (int j = seq_len; j < max_sequence_length; j++) {
      gsl::span<const T> src = inputs.subspan(j * batch_size * input_size + i * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(num_directions * j * batch_size * input_size + i * input_size, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }
  }
}

// A has size M x K, B has size N x K (transposed), and C has size M x N
// We check that A, B and C are large enough before calling the lower level GEMM implementation
template <typename TSpanAIter, typename TSpanBIter, typename TSpanCIter>
void ComputeGemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 TSpanAIter A,
                 TSpanAIter A_end,
                 const int lda,
                 TSpanBIter B,
                 TSpanBIter B_end,
                 const int ldb,
                 const float beta,
                 TSpanCIter C,
                 TSpanCIter C_end,
                 const int ldc,
                 concurrency::ThreadPool* thread_pool) {
  // validate all the inputs
  // need to use the lda/ldb/ldc strides which should be >= the columns for the span
  ORT_ENFORCE(lda >= K && ldb >= K && ldc >= N);
  ORT_ENFORCE(A + (M * lda - (lda - K)) <= A_end);
  ORT_ENFORCE(B + (N * ldb - (ldb - K)) <= B_end);
  ORT_ENFORCE(C + (M * ldc - (ldc - N)) <= C_end);

  ::onnxruntime::math::GemmEx<float>(
      CblasNoTrans, CblasTrans,
      M, N, K, alpha,
      &*A, lda,
      &*B, ldb, beta,
      &*C, ldc, thread_pool);
}

struct PackedWeights {
  BufferUniquePtr buffer_;
  size_t weights_size_;
  TensorShape shape_;
};

struct QuantizationParameter {
  QuantizationParameter(const float* scale,
                        const uint8_t* zero_point,
                        bool is_signed,
                        size_t scale_size) : scale(scale),
                                             zero_point(zero_point),
                                             is_signed(is_signed),
                                             scale_size(scale_size) {}

  const float* scale;
  const uint8_t* zero_point;
  bool is_signed;
  size_t scale_size;
};

template <typename T>
struct GemmWeights {
  GemmWeights() = default;

  GemmWeights(int idx,
              const T* weights_data,
              size_t weights_size,
              const PackedWeights& packed_weights,
              QuantizationParameter* quant_para = nullptr) {
    Init(idx, weights_data, weights_size, packed_weights, quant_para);
  }

  void Init(int idx,
            const T* weights_data,
            size_t weights_size,
            const PackedWeights& packed_weights,
            QuantizationParameter* quant_para) {
    quant_para_ = quant_para;

    if (packed_weights.buffer_) {
      is_prepacked_ = true;
      buffer_ = static_cast<uint8_t*>(packed_weights.buffer_.get()) + packed_weights.weights_size_ * idx;
    } else {
      is_prepacked_ = false;
      buffer_ = weights_data + weights_size * idx;
    }
  }

  bool is_prepacked_{false};
  const void* buffer_{nullptr};
  QuantizationParameter* quant_para_{nullptr};
};

void ComputeGemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* A_end,
                 const GemmWeights<float>& weights,
                 const float beta,
                 float* C,
                 float* C_end,
                 const int ldc,
                 AllocatorPtr /*allocator*/,
                 concurrency::ThreadPool* thread_pool);

void ComputeGemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 const float* A,
                 const float* A_end,
                 const GemmWeights<uint8_t>& weights,
                 const float beta,
                 float* C,
                 float* C_end,
                 const int ldc,
                 AllocatorPtr allocator,
                 concurrency::ThreadPool* thread_pool);

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
const T* SafeRawConstPointer(typename gsl::span<T>::const_iterator cur,
                             typename gsl::span<T>::const_iterator end,
                             size_t size) {
  ORT_ENFORCE(cur + size <= end);
  return &*cur;
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
const T* SafeRawConstPointer(gsl::span<T> span, size_t offset, size_t size) {
  ORT_ENFORCE(offset + size <= size_t(span.size()));
  return span.data();
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
T* SafeRawPointer(typename gsl::span<T>::iterator cur,
                  typename gsl::span<T>::iterator end,
                  size_t size) {
  ORT_ENFORCE(cur + size <= end);
  return &*cur;
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
T* SafeRawPointer(typename gsl::span<T> span, size_t offset, size_t size) {
  ORT_ENFORCE(offset + size <= size_t(span.size()));
  return span.data() + offset;
}

void DumpMatrixImpl(const std::string& name, const float* src, int row, int col,
                    int offset = 0, int col_width = -1);

// Helper class to wrap the processing of the activation funcs and any alpha/beta values.
// The alpha/beta values are consumed in the order of the activation funcs. once they run out
// defaults will be used as needed.
// The Entries property contains the normalized function names and the alpha/beta value to use.
class ActivationFuncs {
 public:
  struct Entry {
    const std::string name;
    const float alpha;
    const float beta;
  };

  ActivationFuncs() = default;

  ActivationFuncs(const std::vector<std::string>& funcs,
                  const std::vector<float>& alphas,
                  const std::vector<float>& betas);

  const std::vector<Entry>& Entries() const {
    return entries_;
  }

 private:
  std::vector<Entry> entries_;
};

namespace deepcpu {

using AddBiasIntoFuncPtr = void (*)(const float*, float*, const int);
using ClipWithBiasFuncPtr = void (*)(float, const float*, float*, const int);
using ActivationFuncPtr = void (*)(float*, int, float, float);
using ActivationFuncBPtr = void (*)(const float*, float*, int, float, float);
using LstmMergeGatesFuncPtr = void (*)(const float*, float*, const float*, float*, int, float, float);
using GruResetGateFuncPtr = void (*)(const float*, float*, float*, int, float, float);
using GruOutputGateFuncPtr = void (*)(float*, const float*, const float*, float*, int, float, float);

ActivationFuncPtr ActivationFuncByName(const std::string& func);
LstmMergeGatesFuncPtr LstmMergeGatesFuncByName(const std::string& func);
GruResetGateFuncPtr GruResetGateFuncByName(const std::string& func);
GruOutputGateFuncPtr GruOutputGateFuncByName(const std::string& func);

void add_bias_into_ignore(const float* ignored, const float* pd, int c);
void add_bias_into(const float* ps, float* pd, int c);
void clip(float b, float* pd, int c);
void clip_add_bias(float b, const float* pb, float* pd, int c);
void clip_ignore_bias(float b, const float* pb, float* pd, int c);
void sigmoid_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void tanh_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void relu_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void sigmoid_exact_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void tanh_exact_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void sigmoid(float* pd, int c, float alpha, float beta);
void tanh(float* pd, int c, float alpha, float beta);
void relu(float* pd, int c, float alpha, float beta);
void sigmoid_exact(float* pd, int c, float alpha, float beta);
void tanh_exact(float* pd, int c, float alpha, float beta);
void merge_lstm_gates_to_memory(const float* pprev, const float* pi, const float* pf, const float* pg, float* pcurr,
                                int c);
void gru_reset_gate_tanh(const float* ps1, float* ps2, float* pd, int c, float alpha, float beta);
void gru_reset_gate_sigmoid(const float* ps1, float* ps2, float* pd, int c, float alpha, float beta);
void gru_reset_gate_relu(const float* ps1, const float* ps2, float* pd, int c, float alpha, float beta);
void gru_output_gate_tanh(float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta);
void gru_output_gate_sigmoid(float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta);
void gru_output_gate_relu(const float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta);

inline void elementwise_product(const float* op1, const float* op2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] += op1[i] * op2[i];
}

inline void elementwise_sum1(const float* src, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] += src[i];
}

inline void elementwise_sum2(const float* src1, const float* src2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] += src1[i] + src2[i];
}

}  // namespace deepcpu
}  // namespace detail
}  // namespace rnn
}  // namespace onnxruntime
