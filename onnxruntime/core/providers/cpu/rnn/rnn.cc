// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/rnn/rnn.h"

#include "core/common/safeint.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/cpu/rnn/rnn_activation_functors.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
// TODO: fix the warnings
#if defined(_MSC_VER) && !defined(__clang__)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif

namespace rnn_internal {
template <typename T>
T Clip(const T& x, T clip) {
  if (clip < 0)
    return x;

  return std::max(std::min(x, clip), -clip);
}

}  // namespace rnn_internal

namespace onnxruntime {
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    RNN,
    7,
    13,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int>()),
    RNN<float>);

ONNX_CPU_OPERATOR_KERNEL(
    RNN,
    14,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int>()),
    RNN<float>);

// #define DUMP_MATRIXES to provide diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif


template <typename T>
void ApplyActivationToBatches(const Tensor* sequence_lens, const T* h_prev, T* Y_buffer_data_current_frame,
                              int64_t time_step, int64_t batch_size, int64_t hidden_size,
                              T alpha, T beta, T clip, std::function<T(T, T, T)> activation_func) {
  const int* seq_len_data = sequence_lens ? sequence_lens->Data<int>() : nullptr;

  for (int batch = 0; batch < batch_size; batch++) {
    bool valid = true;
    if (nullptr != seq_len_data) {
      // sequence_lens is already validated to have batch_size entries
      valid = time_step < seq_len_data[batch];
    }

    for (int feature = 0; feature < hidden_size; ++feature) {
      int64_t y_index = batch * hidden_size + feature;
      if (!valid) {
        // copy from previous time_step if available
        Y_buffer_data_current_frame[y_index] = h_prev ? h_prev[batch * hidden_size + feature] : 0.f;
      } else {
        Y_buffer_data_current_frame[y_index] = activation_func(
            rnn_internal::Clip(Y_buffer_data_current_frame[y_index], clip), alpha, beta);
      }
    }
  }
}

template <typename T>
void Assign_Y_h(const T* Y_buffer_data, Tensor* Y_h, const Tensor* sequence_lens,
                int64_t num_directions, int direction, bool isReverse, int64_t batch_size, int64_t seq_length, int64_t hidden_size) {
  for (int batch = 0; batch < batch_size; batch++) {
    int64_t last_time_step = isReverse ? 0 : seq_length - 1;
    if (nullptr != sequence_lens && !isReverse)
      last_time_step = sequence_lens->Data<int>()[batch] - 1;
    int64_t y_offset = last_time_step * num_directions * batch_size * hidden_size +
                       direction * batch_size * hidden_size +
                       batch * hidden_size;
    int64_t Y_h_offset = direction * batch_size * hidden_size + batch * hidden_size;
    math::CopyVector<T, CPUMathUtil>(static_cast<int>(hidden_size), Y_buffer_data + y_offset,
                                     Y_h->MutableData<T>() + Y_h_offset,
                                     &CPUMathUtil::Instance());
  }
}

template <typename T>
void ClearMissingFrames(T* Y_buffer_data, const Tensor* sequence_lens,
                        int64_t num_directions, int64_t batch_size, int64_t seq_length, int64_t hidden_size) {
  for (int direction = 0; direction < num_directions; direction++) {
    for (int batch = 0; batch < batch_size; batch++) {
      if (sequence_lens->Data<int>()[batch] < seq_length) {
        for (int seq = sequence_lens->Data<int>()[batch]; seq < seq_length; seq++) {
          int64_t offset =
              seq * num_directions * batch_size * hidden_size +
              direction * batch_size * hidden_size +
              batch * hidden_size;
          math::Set<T, CPUMathUtil>(onnxruntime::narrow<size_t>(hidden_size), 0, Y_buffer_data + offset, &CPUMathUtil::Instance());
        }
      }
    }
  }
}

template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <>
Status RNN<float>::Compute(OpKernelContext* ctx) const {
  using namespace rnn::detail;
  concurrency::ThreadPool* tp = ctx->GetOperatorThreadPool();

  // inputs
  const Tensor& X = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->Input<Tensor>(1);
  const Tensor& R = *ctx->Input<Tensor>(2);

  // optional inputs
  const auto* B = ctx->Input<Tensor>(3);
  const auto* sequence_lens = ctx->Input<Tensor>(4);
  const auto* initial_h = ctx->Input<Tensor>(5);

  int64_t num_directions = direction_ == "bidirectional" ? 2 : 1;
  int64_t seq_length = X.Shape()[0];
  int64_t batch_size = X.Shape()[1];
  int64_t input_size = X.Shape()[2];

  auto status = rnn::detail::ValidateCommonRnnInputs(X, W.Shape(), R.Shape(), B, 1, sequence_lens, initial_h,
                                                     num_directions, hidden_size_);
  ORT_RETURN_IF_ERROR(status);

  // RNN outputs are optional
  std::vector<int64_t> Y_dims({seq_length, num_directions, batch_size, hidden_size_});
  Tensor* Y = ctx->Output(0, Y_dims);

  std::vector<int64_t> Y_h_dims({num_directions, batch_size, hidden_size_});
  Tensor* Y_h = ctx->Output(1, Y_h_dims);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&alloc));

  // X * W^t, each direction has shape of [seq_length, batch_size, hidden_size]
  auto x_matmul_data = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * seq_length * batch_size * hidden_size_);
  BufferUniquePtr x_matmul_buffer(x_matmul_data, BufferDeleter(alloc));
  auto* x_matmul_w_buffer_data = static_cast<float*>(x_matmul_buffer.get());

  float* Y_buffer_data;
  void* Y_data;
  BufferUniquePtr Y_matmul_buffer;
  if (Y != nullptr)
    Y_buffer_data = Y->MutableData<float>();
  else {
    Y_data = alloc->Alloc(SafeInt<size_t>(sizeof(float)) * seq_length * num_directions * batch_size * hidden_size_);
    Y_matmul_buffer = BufferUniquePtr(Y_data, BufferDeleter(alloc));
    Y_buffer_data = static_cast<float*>(Y_matmul_buffer.get());
  }

  int64_t Y_frame_size = batch_size * hidden_size_;

  for (int direction = 0; direction < num_directions; direction++) {
    auto activation_func = GetFuncByName<float>(activations_[direction], "Tanh");
    bool isReverse = direction_ == "reverse" || direction == 1;

    if (B != nullptr) {
      EigenMatrixMapRowMajor<float>(x_matmul_w_buffer_data, seq_length * SafeInt<size_t>(batch_size), onnxruntime::narrow<size_t>(hidden_size_)).rowwise() =
          ConstEigenVectorMap<float>(B->Data<float>() + direction * 2 * hidden_size_, onnxruntime::narrow<size_t>(hidden_size_)).transpose() +
          ConstEigenVectorMap<float>(B->Data<float>() + direction * 2 * hidden_size_ + hidden_size_, onnxruntime::narrow<size_t>(hidden_size_)).transpose();
    } else {
      math::Set<float, CPUMathUtil>(seq_length * batch_size * SafeInt<size_t>(hidden_size_), 0, x_matmul_w_buffer_data, &CPUMathUtil::Instance());
    }

    // X * W[direction]^t + B
    math::Gemm<float>(
        CblasNoTrans,
        CblasTrans,
        static_cast<int>(seq_length * batch_size),
        static_cast<int>(hidden_size_),
        static_cast<int>(input_size),
        1,
        X.Data<float>(),
        W.Data<float>() + direction * hidden_size_ * input_size,
        1,
        x_matmul_w_buffer_data,
        tp);

    for (int64_t t = 0; t < seq_length; t++) {
      int64_t time_step = isReverse ? (seq_length - t - 1) : t;
      int64_t Y_frame_offset = (time_step * num_directions + direction) * Y_frame_size;
      float* Y_buffer_data_current_frame = Y_buffer_data + Y_frame_offset;
      auto y_frame_mat = EigenMatrixMapRowMajor<float>(Y_buffer_data_current_frame, onnxruntime::narrow<size_t>(batch_size), onnxruntime::narrow<size_t>(hidden_size_));

      const float* h_prev = nullptr;
      if (t == 0) {
        if (initial_h != nullptr) {
          // the shape of initial_h is [num_directions, batch_size, hidden_size]
          // so pick the offset (multiple of Y_frame_size == batch_size * hidden_size_)
          // based on the direction
          h_prev = initial_h->Data<float>() + (direction * Y_frame_size);
        }
      } else {
        if (isReverse)
          h_prev = Y_buffer_data_current_frame + num_directions * Y_frame_size;
        else
          h_prev = Y_buffer_data_current_frame - num_directions * Y_frame_size;
      }

      if (h_prev != nullptr) {
        // H_t_1 * R[direction]^t
        math::Gemm<float>(
            CblasNoTrans,
            CblasTrans,
            static_cast<int>(batch_size),
            static_cast<int>(hidden_size_),
            static_cast<int>(hidden_size_),
            1,
            h_prev,
            R.Data<float>() + direction * hidden_size_ * hidden_size_,
            0,
            Y_buffer_data_current_frame,
            tp);
      } else {
        math::Set<float, CPUMathUtil>(batch_size * SafeInt<size_t>(hidden_size_), 0, Y_buffer_data_current_frame, &CPUMathUtil::Instance());
      }

      // X[time_step] * W^t + H_t_1 * R^t
      y_frame_mat += EigenMatrixMapRowMajor<float>(&x_matmul_w_buffer_data[time_step * Y_frame_size], onnxruntime::narrow<size_t>(batch_size), onnxruntime::narrow<size_t>(hidden_size_));

      // apply activation
      ApplyActivationToBatches<float>(sequence_lens, h_prev, Y_buffer_data_current_frame,
                                      time_step, batch_size, hidden_size_,
                                      activation_alpha_[direction], activation_beta_[direction], clip_, activation_func);
    }  // close sequence loop

    if (Y_h)
      Assign_Y_h<float>(Y_buffer_data, Y_h, sequence_lens,
                        num_directions, direction, isReverse, batch_size, seq_length, hidden_size_);
  }

  // Now the full sequence is completed. Set missing frames to zero.
  if (nullptr != sequence_lens) {
    ClearMissingFrames(Y_buffer_data, sequence_lens,
                       num_directions, batch_size, seq_length, hidden_size_);
  }

  if (Y != nullptr)
    DumpMatrix("Y", Y_buffer_data, (int)(seq_length * num_directions * batch_size), (int)hidden_size_);

  if (Y_h != nullptr)
    DumpMatrix("Y_h", Y_h->Data<float>(), (int)(num_directions * batch_size), (int)hidden_size_);

  return Status::OK();
}
}  // namespace onnxruntime
