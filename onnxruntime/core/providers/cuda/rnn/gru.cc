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
class UniDirectionalGru {
 public:
  UniDirectionalGru(AllocatorPtr allocator,
                    const int seq_length,
                    const int batch_size,
                    const int input_size,
                    const int hidden_size,
                    const bool linear_before_reset,
                    Direction direction,
                    const gsl::span<const T>& bias,
                    const gsl::span<const T>& initial_hidden_state,
                    const ActivationFuncs::Entry& activation_func_f,
                    const ActivationFuncs::Entry& activation_func_g,
                    const float clip);

  void Compute(const gsl::span<const T>& inputs,
               const gsl::span<const int>& sequence_lengths,
               const int num_directions,
               const gsl::span<const T>& input_weights,
               const gsl::span<const T>& recurrent_weights,
               gsl::span<T>& outputs,
               gsl::span<T>& final_hidden_state);

  ~UniDirectionalGru() = default;

 private:
  AllocatorPtr allocator_;

  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;
  bool linear_before_reset_;

  const float clip_;

  Direction direction_;
  bool use_bias_;

  IAllocatorUniquePtr<T> outputZRH_ptr_;
  gsl::span<T> outputZRH_;

  IAllocatorUniquePtr<T> cur_h_ptr_;
  IAllocatorUniquePtr<T> batched_hidden0_ptr_;
  IAllocatorUniquePtr<int> sequence_lengths_ptr_;
  gsl::span<T> cur_h_;
  gsl::span<T> batched_hidden0_;
  gsl::span<int> sequence_lengths_;

  // Wb[zr] and Rb[zr] can always be added together upfront, and repeated to match the batch size for
  // faster GEMM calculations, so these two members are all the
  // Wb[z] + Rb[z] values added together, repeated batch_size_ times
  IAllocatorUniquePtr<T> batched_bias_WRz_ptr_, batched_bias_WRr_ptr_;
  gsl::span<T> batched_bias_WRz_, batched_bias_WRr_;

  // Wbh and Rbh can only be combined upfront if linear_before_reset_ is false
  IAllocatorUniquePtr<T> batched_bias_WRh_ptr_;
  gsl::span<T> batched_bias_WRh_;

  // if linear_before_reset_ is true, we need to setup Wbh and Rbh separately
  IAllocatorUniquePtr<T> batched_bias_Wh_ptr_, batched_bias_Rh_ptr_;
  gsl::span<T> batched_bias_Wh_, batched_bias_Rh_;

  IAllocatorUniquePtr<T> linear_output_ptr_;
  gsl::span<T> linear_output_;

  IAllocatorUniquePtr<T> inputs_reverse_ptr_;
  IAllocatorUniquePtr<T> outputs_reverse_ptr_;
  gsl::span<T> inputs_reverse_;
  gsl::span<T> outputs_reverse_;

  deepcpu::ClipWithBiasFuncPtr clip_with_bias_ptr_{};

  float zr_alpha_{};
  float zr_beta_{};
  float h_alpha_{};
  float h_beta_{};

  deepcpu::GruResetGateFuncPtr reset_gate_{};
  deepcpu::ActivationFuncPtr update_gate_{};
  deepcpu::GruOutputGateFuncPtr output_gate_{};

  void AllocateBuffers();
};

template <typename T>
UniDirectionalGru<T>::UniDirectionalGru(AllocatorPtr allocator,
                                        const int seq_length,
                                        const int batch_size,
                                        const int input_size,
                                        const int hidden_size,
                                        const bool linear_before_reset,
                                        Direction direction,
                                        const gsl::span<const T>& bias,
                                        const gsl::span<const T>& initial_hidden_state,
                                        const ActivationFuncs::Entry& activation_func_f,
                                        const ActivationFuncs::Entry& activation_func_g,
                                        const float clip)
    : allocator_(allocator),
      seq_length_(seq_length),
      batch_size_(batch_size),
      input_size_(input_size),
      hidden_size_(hidden_size),
      linear_before_reset_(linear_before_reset),
      clip_(clip),
      direction_(direction),
      use_bias_(!bias.empty()) {
  clip_with_bias_ptr_ = use_bias_ ? deepcpu::clip_add_bias : deepcpu::clip_ignore_bias;

  // setup activation function pointers and alpha/beta values to use with them
  reset_gate_ = deepcpu::GruResetGateFuncByName(activation_func_f.name);
  update_gate_ = deepcpu::ActivationFuncByName(activation_func_f.name);
  output_gate_ = deepcpu::GruOutputGateFuncByName(activation_func_g.name);

  zr_alpha_ = activation_func_f.alpha;
  zr_beta_ = activation_func_f.beta;
  h_alpha_ = activation_func_g.alpha;
  h_beta_ = activation_func_g.beta;

  AllocateBuffers();

  if (use_bias_) {
    auto bias_Wz = bias.subspan(0 * hidden_size_, hidden_size_);
    auto bias_Wr = bias.subspan(1 * hidden_size_, hidden_size_);
    auto bias_Wo = bias.subspan(2 * hidden_size_, hidden_size_);
    auto bias_Rz = bias.subspan(3 * hidden_size_, hidden_size_);
    auto bias_Rr = bias.subspan(4 * hidden_size_, hidden_size_);
    auto bias_Ro = bias.subspan(5 * hidden_size_, hidden_size_);

    // add Wb[zr] and Rb[zr] and replicate so we have batch_size_ copies of the result
    auto combine_and_replicate = [&](gsl::span<const T>& bias_w,
                                     gsl::span<const T>& bias_r,
                                     gsl::span<T>& output) {
      // add once
      for (int i = 0; i < hidden_size_; ++i) {
        output[i] = bias_w[i] + bias_r[i];
      }

      // replicate what we just wrote to the start of the output span so we have batch_size_ copies
      auto values = output.cbegin();
      ORT_IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(values, values + hidden_size_,
                                                           output.begin() + hidden_size_,  // skip the first batch
                                                           batch_size_ - 1));              // and replicate batch size - 1 times
    };

    // we can always combine the z and r weights
    combine_and_replicate(bias_Wz, bias_Rz, batched_bias_WRz_);
    combine_and_replicate(bias_Wr, bias_Rr, batched_bias_WRr_);

    // how we treat the h weight depends on whether linear_before_reset_ is set
    if (linear_before_reset_) {
      // need to replicate Wb[o] and Rb[o] separately
      ORT_IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(bias_Wo.cbegin(), bias_Wo.cend(), batched_bias_Wh_.begin(), batch_size_));
      ORT_IGNORE_RETURN_VALUE(RepeatVectorToConstructArray(bias_Ro.cbegin(), bias_Ro.cend(), batched_bias_Rh_.begin(), batch_size_));
    } else {
      combine_and_replicate(bias_Wo, bias_Ro, batched_bias_WRh_);
    }
  }

  if (!initial_hidden_state.empty()) {
    gsl::copy(initial_hidden_state, batched_hidden0_);
  }
}

template <typename T>
Status GRU<T>::ComputeOneDirection(const T* input,
                                   const T* w,
                                   const T* r,
                                   const T* b,
                                   T* y,
                                   T* y_h,
                                   int batch_size,
                                   int input_size,
                                   int hidden_size,
                                   int max_sequence_length) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const int total_rows = max_sequence_length * batch_size;
  CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
  CudaT beta = ToCudaType<T>::FromFloat(0.0f);
  auto output_zrh = GetScratchBuffer<CudaT>(3 * hidden_size * total_rows);
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      CublasHandle(),
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      3 * hidden_size, total_rows, input_size,
      &alpha,
      reinterpret_cast<const CudaT*>(w),
      input_size,
      reinterpret_cast<const CudaT*>(input),
      input_size,
      &beta,
      output_zrh.get(),
      3 * hidden_size));

  // for each item in sequence run all calculations
  for (int step = 0; step < max_sequence_length; step++) {
  }
}

template <typename T>
void UniDirectionalGru<T>::Compute(const gsl::span<const T>& inputs_arg,
                                   const gsl::span<const int>& sequence_lengths_arg,
                                   const int num_directions,
                                   const gsl::span<const T>& input_weights,
                                   const gsl::span<const T>& recurrent_weights,
                                   gsl::span<T>& outputs,
                                   gsl::span<T>& final_hidden_state) {
  using span_T_const_iter = typename gsl::span<T>::const_iterator;
  using span_T_iter = typename gsl::span<T>::iterator;

  // copy inputs_arg as we may change it to point to inputs_reverse_
  gsl::span<const T> inputs = inputs_arg;
  gsl::span<const int> sequence_lengths = sequence_lengths_arg;

  // if sequence lengths weren't provided, use internal array and init all to seq_length
  if (sequence_lengths.empty()) {
    sequence_lengths_ = Allocate(allocator_, batch_size_, sequence_lengths_ptr_, true, seq_length_);
    sequence_lengths = sequence_lengths_;
  }

  DumpMatrix("Inputs", inputs.data(), seq_length_ * batch_size_, input_size_);
  DumpMatrix("input_weights", input_weights.data(), 3 * hidden_size_, input_size_);
  DumpMatrix("recurrent_weights", recurrent_weights.data(), 3 * hidden_size_, hidden_size_);

  gsl::span<const T> recurrent_weightsZR = recurrent_weights.subspan(0, 2 * hidden_size_ * hidden_size_);
  gsl::span<const T> recurrent_weightsH = recurrent_weights.subspan(2 * hidden_size_ * hidden_size_, hidden_size_ * hidden_size_);

  gsl::span<T> original_outputs = outputs;
  const bool output_sequence = !outputs.empty();

  if (direction_ == kReverse) {
    ReverseSequence(inputs, inputs_reverse_, sequence_lengths, seq_length_, batch_size_, input_size_, 1);
    // DumpMatrix("Reversed inputs", inputs_reverse_.data(), seq_length_ * batch_size_, input_size_);

    inputs = inputs_reverse_;

    if (output_sequence) {
      outputs = outputs_reverse_;
    }
  }

  // Calculate the max and min length
  int32_t max_sequence_length = *std::max_element(sequence_lengths.cbegin(), sequence_lengths.cend());
  int32_t min_sequence_length = std::min(seq_length_, *std::min_element(sequence_lengths.cbegin(),
                                                                        sequence_lengths.cend()));

  const int hidden_size_x2 = 2 * hidden_size_;
  const int hidden_size_x3 = 3 * hidden_size_;
  const int total_rows = max_sequence_length * batch_size_;

  float alpha = 1.0f;
  float beta = 0.0f;  // zero out outputZRH_ when calling ComputeGemm.

  // apply weights to all the inputs
  ComputeGemm(total_rows, hidden_size_x3, input_size_, alpha,
              inputs.cbegin(), inputs.cend(),
              input_size_,
              input_weights.cbegin(), input_weights.cend(),
              input_size_, beta,
              outputZRH_.begin(), outputZRH_.end(),
              hidden_size_x3);

  DumpMatrix("inputs with weights applied", outputZRH_.data(), seq_length_ * batch_size_ * 3, hidden_size_);

  // set to 1 so the weighted inputs in outputZRH_ are added to the result in the next call to ComputeGemm
  beta = 1.0f;

  // output shape is [seq_length, num_directions, batch_size, hidden_size]
  // if we are doing 2 directions and this is the forward pass we're writing to the real output so
  // need to include num_directions in the step length.
  // we do not need to do that if there are two directions and we're doing the backwards pass as we
  // are writing to a temporary buffer (as outputs == outputs_reverse_) which is later copied
  // to the real output by ReverseSequence. this later copy includes num_directions in the step length.
  int output_step_length = batch_size_ * hidden_size_;
  if (direction_ == kForward && num_directions == 2)
    output_step_length = 2 * batch_size_ * hidden_size_;

  // convenience end iterators we use in the loops below to detect any bounds issues
  span_T_const_iter batched_bias_WRz_local_end = batched_bias_WRz_.cend();
  span_T_const_iter batched_bias_WRr_local_end = batched_bias_WRr_.cend();
  span_T_const_iter batched_bias_Wh_local_end = batched_bias_Wh_.cend();
  span_T_const_iter batched_bias_Rh_local_end = batched_bias_Rh_.cend();
  span_T_const_iter batched_bias_WRh_local_end = batched_bias_WRh_.cend();

  size_t out_added_offset;

  span_T_const_iter prev_Ht = batched_hidden0_.cbegin();  // Ht-1
  span_T_const_iter prev_Ht_end = batched_hidden0_.cend();
  span_T_iter cur_h_local = cur_h_.begin();
  span_T_iter cur_h_local_end = cur_h_.end();

  span_T_const_iter batched_bias_WRz_local;
  span_T_const_iter batched_bias_WRr_local;
  span_T_const_iter batched_bias_WRh_local;
  span_T_const_iter batched_bias_Wh_local;
  span_T_const_iter batched_bias_Rh_local;

  if (use_bias_) {
    batched_bias_WRz_local = batched_bias_WRz_.cbegin();
    batched_bias_WRr_local = batched_bias_WRr_.cbegin();

    if (linear_before_reset_) {
      batched_bias_Wh_local = batched_bias_Wh_.cbegin();
      batched_bias_Rh_local = batched_bias_Rh_.cbegin();
    } else {
      batched_bias_WRh_local = batched_bias_WRh_.cbegin();
    }
  }

  // for each item in sequence run all calculations
  for (int step = 0; step < max_sequence_length; step++) {
    const std::string seqno_str = " [seqno=" + std::to_string(step) + "]";

    DumpMatrix("Ht-1" + seqno_str, &*prev_Ht, batch_size_, hidden_size_);

    out_added_offset = (step * batch_size_) * hidden_size_x3;

    // calculate Ht-1*R[zr], and add to the weighted inputs that are in outputZRH_
    // Ht-1 * R[zr] + Xt*(W[zr]^T)
    ComputeGemm(batch_size_, hidden_size_x2, hidden_size_, alpha,
                prev_Ht, prev_Ht_end,
                hidden_size_,
                recurrent_weightsZR.cbegin(), recurrent_weightsZR.cend(),
                hidden_size_, beta,
                outputZRH_.begin() + out_added_offset, outputZRH_.end(),
                hidden_size_x3);

    DumpMatrix("Ht-1 * R[zr] + Xt*(W[zr]^T)" + seqno_str,
               outputZRH_.data() + out_added_offset, batch_size_, hidden_size_x2, 0, hidden_size_x3);

    // 1st Set Of Activations
    for (int r = 0; r < batch_size_; r++) {
      const T* p_bias_r = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRr_local + r * hidden_size_,
                                                             batched_bias_WRr_local_end, hidden_size_)
                                    : nullptr;

      // initialize p_rt with input to calculate rt. outputZRH_ has Xt*(Wr^T) + Ht-1*(Rr^T).
      T* p_rt = SafeRawPointer(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_, hidden_size_);

      // add the bias and clip. post: p_rt == Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr
      clip_with_bias_ptr_(clip_, p_bias_r, p_rt, hidden_size_);

      if (linear_before_reset_) {
        // p_linear_output = Ht-1 * (Rh^T) + Rbh
        T* p_linear_output = SafeRawPointer<T>(linear_output_, r * hidden_size_, hidden_size_);
        T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

        // calculate rt in-place [p_rt = f(p_rt)]
        // calculate rt (.) (Ht-1 * (Rh^T) + Rbh) using p_linear_output. write to p_cur_h
        reset_gate_(p_linear_output, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);

      } else {
        const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
        T* p_cur_h = SafeRawPointer<T>(cur_h_local + r * hidden_size_, cur_h_local_end, hidden_size_);

        // calculate rt in-place [p_rt = f(p_rt)]
        // calculate rt (.) Ht-1 using p_prev_Ht, and write to p_cur_h
        reset_gate_(p_prev_Ht, p_rt, p_cur_h, hidden_size_, zr_alpha_, zr_beta_);
      }
    }

    std::string label = linear_before_reset_ ? "rt (.) (Ht-1 * (Rh^T) + Rbh)" : "rt (.) Ht-1";
    DumpMatrix(label + seqno_str, &*cur_h_local, batch_size_, hidden_size_);

    label += " * Rh^T";
    // out_H currently contains Xt*(Wh^T).
    auto out_H = outputZRH_.begin() + out_added_offset + hidden_size_x2;
    // Calculate Xt*(Wh^T) + rt (.) Ht-1 * Rh
    ComputeGemm(batch_size_, hidden_size_, hidden_size_, alpha,
                cur_h_local, cur_h_local_end,  // rt (.) Ht-1
                hidden_size_,
                recurrent_weightsH.cbegin(), recurrent_weightsH.cend(),  // Rh^T
                hidden_size_, beta,
                out_H, outputZRH_.end(),
                hidden_size_x3);
    DumpMatrix("Xt*(Wh^T) + (" + label + ")" + seqno_str, outputZRH_.data() + out_added_offset,
               batch_size_, hidden_size_, hidden_size_x2, hidden_size_x3);

    //2nd Set of Activations
    span_T_iter output;
    span_T_iter output_end;
    if (output_sequence) {
      output = outputs.begin() + step * output_step_length;
      output_end = outputs.end();

    } else {
      output = final_hidden_state.begin();
      output_end = final_hidden_state.end();
    }

    for (int r = 0; r < batch_size_; r++) {
      if (step >= min_sequence_length && step >= sequence_lengths[r]) {
        if (output_sequence) {
          auto fill_output = output + r * hidden_size_;
          std::fill_n(fill_output, hidden_size_, T{});
        }

        continue;
      }

      const T* p_bias_z = use_bias_ ? SafeRawConstPointer<T>(batched_bias_WRz_local,
                                                             batched_bias_WRz_local_end, hidden_size_)
                                    : nullptr;

      // initialize p_zt with Xt*(Wz^T) + Ht-1*(Rz^T), which is most of the input to calculate zt:
      T* p_zt = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3, hidden_size_);

      // using p_zt, add bias and clip in-place
      clip_with_bias_ptr_(clip_, p_bias_z, p_zt, hidden_size_);

      // calculate zt in-place. p_zt = f(p_zt)
      update_gate_(p_zt, hidden_size_, zr_alpha_, zr_beta_);

      DumpMatrix("zt[" + std::to_string(r) + "]" + seqno_str, p_zt, 1, hidden_size_);

      const T* p_bias_h = nullptr;
      if (use_bias_) {
        if (linear_before_reset_) {
          // Wbh
          p_bias_h = SafeRawConstPointer<T>(batched_bias_Wh_local + r * hidden_size_,
                                            batched_bias_Wh_local_end, hidden_size_);

        } else {
          // Wbh + Wrh
          p_bias_h = SafeRawConstPointer<T>(batched_bias_WRh_local + r * hidden_size_,
                                            batched_bias_WRh_local_end, hidden_size_);
        }
      }

      // setup p_ht with input to calculate ht
      // p_ht = Xt*(Wh^T) + (rt (.) Ht-1 * Rh^T)          #  linear_before_reset_ == false
      //      = Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh))  #  linear_before_reset_ == true
      T* p_ht = SafeRawPointer<T>(outputZRH_, out_added_offset + r * hidden_size_x3 + hidden_size_x2, hidden_size_);

      // add Wbh [and Wrh] and clip
      clip_with_bias_ptr_(clip_, p_bias_h, p_ht, hidden_size_);  // post: p_ht == input to g() for calculating ht

      DumpMatrix("ht input [" + std::to_string(r) + "]" + seqno_str, p_ht, 1, hidden_size_);

      const T* p_prev_Ht = SafeRawConstPointer<T>(prev_Ht + r * hidden_size_, prev_Ht_end, hidden_size_);
      T* p_Ht = SafeRawPointer<T>(output + r * hidden_size_, output_end, hidden_size_);

      // calculate ht = g(p_ht) and write in-place to p_ht
      // calculate Ht = (1 - zt) (.) ht + zt (.) Ht-1 and write to p_Ht
      output_gate_(p_ht, p_zt, p_prev_Ht, p_Ht, hidden_size_, h_alpha_, h_beta_);  // calculate ht and Ht
    }

    DumpMatrix("output" + seqno_str, &*output, batch_size_, hidden_size_);

    prev_Ht = output;
    prev_Ht_end = output_end;
  }

  // copy last output to final_hidden_state
  for (int i = 0; i < batch_size_; i++) {
    const int seq_len = sequence_lengths[i];
    if (seq_len == 0) {
      auto final_hidden_state_dst = final_hidden_state.begin() + i * hidden_size_;
      std::fill_n(final_hidden_state_dst, hidden_size_, T{});
      continue;
    } else if (output_sequence) {
      auto src = outputs.subspan((seq_len - 1) * output_step_length + i * hidden_size_, hidden_size_);
      auto dest = final_hidden_state.subspan(i * hidden_size_, hidden_size_);
      gsl::copy(src, dest);
    }
  }

  if (output_sequence && direction_ == kReverse) {
    ReverseSequence<T>(outputs, original_outputs,
                       sequence_lengths, seq_length_,
                       batch_size_, hidden_size_, num_directions);
  }
}

template <typename T>
void UniDirectionalGru<T>::AllocateBuffers() {
  cur_h_ = Allocate(allocator_, hidden_size_ * batch_size_, cur_h_ptr_);
  batched_hidden0_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_hidden0_ptr_, true);

  if (use_bias_) {
    batched_bias_WRz_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRz_ptr_);
    batched_bias_WRr_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRr_ptr_);

    if (linear_before_reset_) {
      batched_bias_Wh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_Wh_ptr_);
      batched_bias_Rh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_Rh_ptr_);
      linear_output_ = Allocate(allocator_, batch_size_ * hidden_size_, linear_output_ptr_);
    } else {
      batched_bias_WRh_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_bias_WRh_ptr_);
    }
  }

  auto batch_times_seq_length = batch_size_ * seq_length_;

  outputZRH_ = Allocate(allocator_, hidden_size_ * 3 * batch_times_seq_length, outputZRH_ptr_, true);

  if (direction_ == kReverse) {
    inputs_reverse_ = Allocate(allocator_, batch_times_seq_length * input_size_, inputs_reverse_ptr_);
    outputs_reverse_ = Allocate(allocator_, batch_times_seq_length * hidden_size_, outputs_reverse_ptr_);
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

  int seq_length = gsl::narrow<int>(X_shape[0]);
  int batch_size = gsl::narrow<int>(X_shape[1]);
  int input_size = gsl::narrow<int>(X_shape[2]);

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

  gsl::span<const T> input_weights_1 = input_weights.subspan(0, input_weights_size_per_direction);
  gsl::span<const T> recurrent_weights_1 = recurrent_weights.subspan(0, recurrent_weights_size_per_direction);
  gsl::span<const T> bias_1 = bias.empty() ? bias : bias.subspan(0, bias_size_per_direction);

  gsl::span<const T> input = X.DataAsSpan<T>();
  gsl::span<const int> sequence_lens_span = sequence_lens != nullptr ? sequence_lens->DataAsSpan<int>()
                                                                     : gsl::span<const int>();

  const size_t initial_hidden_size_per_direction = batch_size * hidden_size_;
  gsl::span<const T> initial_hidden = initial_h != nullptr ? initial_h->DataAsSpan<T>() : gsl::span<const T>();
  gsl::span<const T> initial_hidden_1 = initial_hidden.empty()
                                            ? initial_hidden
                                            : initial_hidden.subspan(0, initial_hidden_size_per_direction);

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
