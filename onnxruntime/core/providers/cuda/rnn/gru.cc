// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gru.h"
#include "rnn_impl.h"
#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cpu/math/gemm_helper.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      GRU,                                                               \
      kOnnxDomain,                                                       \
      7,                                                                 \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()), \
      GRU<T>);

REGISTER_KERNEL_TYPED(float);
REGISTER_KERNEL_TYPED(double);
REGISTER_KERNEL_TYPED(MLFloat16);

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

template <typename T>
void GRU<T>::ComputeOneDirection(const T* input,
                                 const T* w,
                                 const T* r,
                                 const T* b,
                                 const T* initial_h,
                                 T* y,
                                 int batch_size,
                                 int input_size,
                                 int hidden_size,
                                 int max_sequence_length) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const int total_rows = max_sequence_length * batch_size;
  CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
  CudaT beta = ToCudaType<T>::FromFloat(0.0f);
  IAllocatorUniquePtr<CudaT> output_zrh = GetScratchBuffer<CudaT>(3 * hidden_size * total_rows);
  IAllocatorUniquePtr<CudaT> bias_merged;
  if (b != nullptr) {
    bias_merged = GetScratchBuffer<CudaT>(3 * hidden_size);
    Add<CudaT>(b, b + 3 * hidden_size, bias_merged, 3 * hidden_size);
  }

  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      CublasHandle(),
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      3 * hidden_size, total_rows, input_size,
      &alpha,
      reinterpret_cast<const CudaT*>(w),
      3 * hidden_size,
      reinterpret_cast<const CudaT*>(input),
      input_size,
      &beta,
      output_zrh.get(),
      3 * hidden_size));

  CudaT beta_1 = ToCudaType<T>::FromFloat(1.0f);
  IAllocatorUniquePtr<CudaT> rt_ht_1 = GetScratchBuffer<CudaT>(hidden_size * batch_size);

  // for each item in sequence run all calculations
  CudaT* y_t_1 = initial_h;
  CudaT* y_t = y;
  for (int step = 0; step < max_sequence_length; step++) {
    CudaT* output_zr_t = ouput_zrh.get() + step * batch_size * 3 * hidden_size;
    CudaT* output_h_t = output_zr_t + 2 * hidden_size;
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        CublasHandle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        2 * hidden_size, batch_size, hidden_size,
        &alpha,
        reinterpret_cast<const CudaT*>(r),
        3 * hidden_size,
        reinterpret_cast<const CudaT*>(y_t),
        hidden_size,
        &beta_1,
        output_zr_t,
        3 * hidden_size));
    // activation zt rt
    for (int batch_inx = 0; batch_inx < batch_size; batch_inx++) {
      GRUZRActivation(output_zr_t + 3 * hidden_size * batch_inx, bias_merged.get(), 2 * hidden_size);
    }
    // rt(.)Ht-1
    CudaT* output_r_t = output_zr_t + hidden_size;
    for (int batch_inx = 0; batch_inx < batch_size; batch_inx++) {
      HadamardProduct(output_r_t + 3 * hidden_size * batch_inx, y_t_1 + batch_inx * hidden_size, rt_ht_1.get() + batch_inx * hidden_size, hidden_size);
    }
    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        CublasHandle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size,
        &alpha,
        reinterpret_cast<const CudaT*>(r) + 2 * hidden_size,
        3 * hidden_size,
        reinterpret_cast<const CudaT*>(y_h),
        hidden_size,
        &beta_1,
        output_h_t,
        3 * hidden_size));
    // activation ht
    for (int batch_inx = 0; batch_inx < batch_size; batch_inx++) {
      GRUHActivation(output_h_t + 3 * batch_inx * hidden_size, bias_merged.get() + 2 * hidden_size, hidden_size);
      GRUOutput(output_zr_t + 3 * hidden_size * batch_inx, output_h_t + 3 * batch_inx * hidden_size, y_t_1 + batch_inx * hidden_size, y_t + batch_inx * hidden_size, hidden_size);
    }
    y_t_1 = y_t;
    y_t = y_t_1 + batch_size * hidden_size;
  }
}

template <typename T>
Status GRU<T>::ComputeInternal(OpKernelContext* ctx) const {
  if (linear_before_reset_) {
    return CudnnRnnBase<T>::ComputeInternal(ctx);
  }

  const Tensor& X = *(ctx->Input<Tensor>(0));  // inputs. [seq_length, batch_size, input_size]
  const Tensor& W = *(ctx->Input<Tensor>(1));  // weights. [num_directions, 3*hidden_size, input_size]
  const Tensor& R = *(ctx->Input<Tensor>(2));  // recurrence weights. [num_directions, 3*hidden_size, hidden_size]

  // optional
  const Tensor* B = ctx->Input<Tensor>(3);              // bias. [num_directions, 6*hidden_size]
  const Tensor* sequence_lens = ctx->Input<Tensor>(4);  // [batch_size]
  const Tensor* initial_h = ctx->Input<Tensor>(5);      // initial hidden. [num_directions, batch_size, hidden_size]

  auto& X_shape = X.Shape();

  int64_t seq_length = X_shape[0];
  int64_t batch_size = X_shape[1];
  int64_t input_size = X_shape[2];

  //auto status = ValidateCommonRnnInputs(X, W, R, B, 3, sequence_lens, initial_h, num_directions_, hidden_size_);
  //ORT_RETURN_IF_ERROR(status);

  // GRU outputs are optional but must be in the same order
  TensorShape Y_dims{seq_length, num_directions_, batch_size, hidden_size_};
  Tensor* Y = ctx->Output(/*index*/ 0, Y_dims);

  TensorShape Y_h_dims{num_directions_, batch_size, hidden_size_};
  Tensor* Y_h = ctx->Output(/*index*/ 1, Y_h_dims);

  // Reset output and return if max sequence length is 0
  if (sequence_lens != nullptr) {
    int32_t max_sequence_length = *std::max_element(sequence_lens->Data<int32_t>(), sequence_lens->Data<int32_t>() + sequence_lens->Shape().Size());
    if (max_sequence_length == 0) {
      if (Y != nullptr) std::fill_n(Y->MutableData<T>(), Y_dims.Size(), T{});
      if (Y_h != nullptr) std::fill_n(Y_h->MutableData<T>(), Y_h_dims.Size(), T{});
      return Status::OK();
    }
  }

  AllocatorPtr alloc;
  auto status = ctx->GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);
  gsl::span<const T> input_weights = W.DataAsSpan<T>();
  gsl::span<const T> recurrent_weights = R.DataAsSpan<T>();
  gsl::span<const T> bias = B != nullptr ? B->DataAsSpan<T>() : gsl::span<const T>();

  // spans for first direction
  const size_t input_weights_size_per_direction = 3 * hidden_size_ * input_size;
  const size_t recurrent_weights_size_per_direction = 3 * hidden_size_ * hidden_size_;
  const size_t bias_size_per_direction = 6 * hidden_size_;

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // so it's not a case of all the output for one direction being first.
  // due to that we can only easily check that the end of the output for each direction is valid.
  const size_t output_size = Y != nullptr ? Y->Shape().Size() : 0;
  const size_t per_direction_offset = batch_size * hidden_size_;
  gsl::span<T> output = Y != nullptr ? Y->MutableDataAsSpan<T>() : gsl::span<T>();
  gsl::span<T> output_1 = output.empty()
                              ? output
                              : output.subspan(0, output_size - (num_directions_ - 1) * per_direction_offset);

  // UniDirectionalGru needs somewhere to write output, so even if we aren't returning Y_h
  // we provide an appropriately sized buffer for that purpose.
  const size_t hidden_output_size_per_direction = batch_size * hidden_size_;
  IAllocatorUniquePtr<T> local_hidden_output;
  gsl::span<T> hidden_output =
      Y_h ? Y_h->MutableDataAsSpan<T>()
          : Allocate<T>(alloc, hidden_output_size_per_direction * num_directions_, local_hidden_output);

  gsl::span<T> hidden_output_1 = hidden_output.subspan(0, hidden_output_size_per_direction);

  if (direction_ == Direction::kBidirectional) {
    // spans for second direction
    gsl::span<const T> input_weights_2 = input_weights.subspan(input_weights_size_per_direction,
                                                               input_weights_size_per_direction);
    gsl::span<const T> recurrent_weights_2 = recurrent_weights.subspan(recurrent_weights_size_per_direction,
                                                                       recurrent_weights_size_per_direction);
    gsl::span<const T> bias_2 = bias.empty() ? bias : bias.subspan(bias_size_per_direction, bias_size_per_direction);

    gsl::span<const T> initial_hidden_2 = initial_hidden.empty()
                                              ? initial_hidden
                                              : initial_hidden.subspan(initial_hidden_size_per_direction,
                                                                       initial_hidden_size_per_direction);
    gsl::span<T> output_2 = output.empty()
                                ? output
                                : output.subspan(per_direction_offset, output_size - per_direction_offset);

    gsl::span<T> hidden_output_2 = hidden_output.subspan(hidden_output_size_per_direction,
                                                         hidden_output_size_per_direction);

    std::unique_ptr<detail::UniDirectionalGru<T>> fw = std::make_unique<detail::UniDirectionalGru<T>>(
        alloc,
        seq_length, batch_size, input_size, hidden_size_, linear_before_reset_, Direction::kForward,
        bias_1, initial_hidden_1,
        activation_funcs_.Entries()[0],
        activation_funcs_.Entries()[1],
        clip_);
    fw->Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1, hidden_output_1);

    std::unique_ptr<detail::UniDirectionalGru<T>> bw = std::make_unique<detail::UniDirectionalGru<T>>(
        alloc,
        seq_length, batch_size, input_size, hidden_size_, linear_before_reset_, Direction::kReverse,
        bias_2, initial_hidden_2,
        activation_funcs_.Entries()[2],
        activation_funcs_.Entries()[3],
        clip_);
    bw->Compute(input, sequence_lens_span, num_directions_, input_weights_2, recurrent_weights_2, output_2, hidden_output_2);
  } else {
    std::unique_ptr<detail::UniDirectionalGru<T>> gru_p = std::make_unique<detail::UniDirectionalGru<T>>(
        alloc,
        seq_length, batch_size, input_size, hidden_size_, linear_before_reset_, direction_,
        bias_1, initial_hidden_1,
        activation_funcs_.Entries()[0],
        activation_funcs_.Entries()[1],
        clip_);

    gru_p->Compute(input, sequence_lens_span, num_directions_, input_weights_1, recurrent_weights_1, output_1, hidden_output_1);
  }

  if (!output.empty())
    DumpMatrix("Y", output.data(), seq_length * num_directions_ * batch_size, hidden_size_);

  DumpMatrix("Y_h", hidden_output.data(), num_directions_ * batch_size, hidden_size_);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
