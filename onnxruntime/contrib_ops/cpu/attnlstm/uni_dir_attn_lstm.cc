// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// copied from gsl_algorithm, gsl disable 4996 for gsl::copy()
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include "uni_dir_attn_lstm.h"

#include <thread>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace onnxruntime::rnn::detail;

namespace onnxruntime {
namespace contrib {
namespace rnn {
namespace detail {

// #define DUMP_MATRIXES to provide lots of diagnostic output
#if defined(DUMP_MATRIXES)
#define DumpMatrix(...) ::onnxruntime::rnn::detail::DumpMatrixImpl(__VA_ARGS__)
#else
#define DumpMatrix(...) ((void)0)
#endif

template <typename T>
UniDirectionalAttnLstm<T>::UniDirectionalAttnLstm(AllocatorPtr allocator,
                                                  const logging::Logger& logger,
                                                  const int seq_length,
                                                  const int batch_size,
                                                  const int input_size,
                                                  const int hidden_size,
                                                  Direction direction,
                                                  const bool input_forget,
                                                  AttentionWrapper<T>& attention_wrapper,
                                                  const gsl::span<const T>& bias,
                                                  const gsl::span<const T>& peephole_weights,
                                                  const gsl::span<const T>& initial_hidden_state,
                                                  const gsl::span<const T>& initial_cell_state,
                                                  const ActivationFuncs::Entry& activation_func_f,
                                                  const ActivationFuncs::Entry& activation_func_g,
                                                  const ActivationFuncs::Entry& activation_func_h,
                                                  const float clip,
                                                  onnxruntime::concurrency::ThreadPool* ttp)
    : allocator_(allocator),
      logger_(logger),
      seq_length_(seq_length),
      batch_size_(batch_size),
      input_size_(input_size),
      hidden_size_(hidden_size),
      direction_(direction),
      input_forget_(input_forget),
      clip_(clip),
      use_bias_(!bias.empty()),
      use_peepholes_(!peephole_weights.empty()),
      attention_wrapper_(attention_wrapper),
      ttp_(ttp) {
  activation_f_ = {deepcpu::ActivationFuncByName(activation_func_f.name),
                   activation_func_f.alpha,
                   activation_func_f.beta};

  activation_g_ = {deepcpu::ActivationFuncByName(activation_func_g.name),
                   activation_func_g.alpha,
                   activation_func_g.beta};

  activation_h_ = {deepcpu::LstmMergeGatesFuncByName(activation_func_h.name),
                   activation_func_h.alpha,
                   activation_func_h.beta};

  clip_with_bias_ptr_ = use_bias_ ? deepcpu::clip_add_bias : deepcpu::clip_ignore_bias;

  attention_size_ = attention_wrapper_.GetAttentionSize();
  attention_context_size_ = attention_wrapper_.GetAttentionContextSize();

  SetNumThreads();
  AllocateBuffers();
  InitializeBuffers(initial_hidden_state, initial_cell_state);

  if (!peephole_weights.empty())
    LoadPeepholeWeights(peephole_weights);
  if (!bias.empty())
    LoadBias(bias);
}

template <typename T>
void UniDirectionalAttnLstm<T>::AllocateBuffers() {
  // allocate and fill with 0's.
  const bool fill = true;
  hidden0_ = Allocate(allocator_, hidden_size_, hidden0_ptr_, fill);
  internal_memory_prev_ = Allocate(allocator_, hidden_size_, internal_memory_prev_ptr_, fill);
  internal_memory_cur_ = Allocate(allocator_, hidden_size_, internal_memory_cur_ptr_, fill);
  batched_hidden0_ = Allocate(allocator_, batch_size_ * hidden_size_, batched_hidden0_ptr_, fill);

  batched_internal_memory_prev_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                           batched_internal_memory_prev_ptr_, fill);
  batched_internal_memory_cur_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                          batched_internal_memory_cur_ptr_, fill);
  batched_internal_memory_clipped_ = Allocate(allocator_, batch_size_ * hidden_size_,
                                              batched_internal_memory_clipped_ptr_, fill);

  output_iofc_ = Allocate(allocator_, hidden_size_ * 4 * batch_size_ * seq_length_, output_iofc_ptr_, fill);

  if (use_bias_) {
    bias_WRi_ = Allocate(allocator_, hidden_size_, bias_WRi_ptr_);
    bias_WRf_ = Allocate(allocator_, hidden_size_, bias_WRf_ptr_);
    bias_WRo_ = Allocate(allocator_, hidden_size_, bias_WRo_ptr_);
    bias_WRc_ = Allocate(allocator_, hidden_size_, bias_WRc_ptr_);
  }

  if (direction_ == kReverse) {
    inputs_reverse_ = Allocate(allocator_, seq_length_ * batch_size_ * input_size_, inputs_reverse_ptr_);
    outputs_reverse_ = Allocate(allocator_, seq_length_ * batch_size_ * hidden_size_, outputs_reverse_ptr_);
  }

#if !defined(LSTM_NO_PEEPHOLE_COPY)
  if (use_peepholes_) {
    peephole_i_ = Allocate(allocator_, hidden_size_, peephole_i_ptr_);
    peephole_f_ = Allocate(allocator_, hidden_size_, peephole_f_ptr_);
    peephole_o_ = Allocate(allocator_, hidden_size_, peephole_o_ptr_);
  }
#endif
}

template <typename T>
void UniDirectionalAttnLstm<T>::InitializeBuffers(const gsl::span<const T>& initial_hidden_state,
                                                  const gsl::span<const T>& initial_cell_state) {
  if (!initial_hidden_state.empty()) {
    gsl::copy(initial_hidden_state, batched_hidden0_);
  } else {
    std::fill_n(batched_hidden0_.data(), batched_hidden0_.size(), T{});
  }

  if (!initial_cell_state.empty()) {
    gsl::copy(initial_cell_state, batched_internal_memory_prev_);
  } else {
    std::fill_n(batched_internal_memory_prev_.data(), batched_internal_memory_prev_.size(), T{});
  }
}

template <typename T>
void UniDirectionalAttnLstm<T>::LoadPeepholeWeights(const gsl::span<const T>& peephole_weights) {
  int i = 0;
#if defined(LSTM_NO_PEEPHOLE_COPY)
  // just use spans. we don't change these values so there's no point copying to them
  peephole_i_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);
  peephole_o_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);
  peephole_f_ = peephole_weights.subspan((i++ * hidden_size_), hidden_size_);

#else
  DumpMatrix("P[i]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("P[o]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);
  DumpMatrix("P[f]", peephole_weights.data() + (i++ * hidden_size_), 1, hidden_size_);

  auto copy_weight = [this, &peephole_weights](int offset, gsl::span<T>& out) {
    typename gsl::span<const T>::const_iterator in_iter = peephole_weights.cbegin() + offset;
    std::copy(in_iter, in_iter + hidden_size_, out.begin());
  };

  i = 0;
  copy_weight((i++ * hidden_size_), peephole_i_);
  copy_weight((i++ * hidden_size_), peephole_o_);
  copy_weight((i++ * hidden_size_), peephole_f_);
#endif

  /*
  DumpMatrix("peephole_i_", peephole_i_.data(), 1, hidden_size_);
  DumpMatrix("peephole_o_", peephole_o_.data(), 1, hidden_size_);
  DumpMatrix("peephole_f_", peephole_f_.data(), 1, hidden_size_);
  */
}

template <typename T>
void UniDirectionalAttnLstm<T>::LoadBias(const gsl::span<const T>& WbRb_values) {
  // add Wb and Rb
  auto copy_fused_bias = [this, &WbRb_values](int offset, gsl::span<T>& out) {
    // gap between Wb and Wb value for an entry
    const int Wb_to_Rb_offset = 4 * hidden_size_;
    for (int j = 0; j < hidden_size_; ++j) {
      out[j] = WbRb_values[j + offset] + WbRb_values[j + offset + Wb_to_Rb_offset];
    }
  };

  int i = 0;
  copy_fused_bias((i++) * hidden_size_, bias_WRi_);
  copy_fused_bias((i++) * hidden_size_, bias_WRo_);
  copy_fused_bias((i++) * hidden_size_, bias_WRf_);
  copy_fused_bias((i++) * hidden_size_, bias_WRc_);
}

template <typename T>
void UniDirectionalAttnLstm<T>::Compute(const gsl::span<const T>& inputs_arg,
                                        const gsl::span<const int>& sequence_lengths_arg,
                                        const int num_directions,
                                        const gsl::span<const T>& input_weights,
                                        const gsl::span<const T>& recurrent_weights,
                                        gsl::span<T>& outputs,
                                        gsl::span<T>& final_hidden_state,
                                        gsl::span<T>& final_cell_state) {
  // copy spans (just T* and size, not data in span) as we may change them
  gsl::span<const T> inputs = inputs_arg;
  gsl::span<const int> sequence_lengths = sequence_lengths_arg;

  // if sequence lengths weren't provided, use internal array and init all to seq_length
  if (sequence_lengths.empty()) {
    sequence_lengths_ = Allocate(allocator_, batch_size_, sequence_lengths_ptr_, true, seq_length_);
    sequence_lengths = sequence_lengths_;
  }

  // LSTM Layer
  gsl::span<T> batched_hidden_state_one_step = batched_hidden0_;
  gsl::span<T> batched_internal_state_prev_one_step = batched_internal_memory_prev_;
  gsl::span<T> batched_internal_state_clipped_one_step = batched_internal_memory_clipped_;

  int output_step_length = batch_size_ * hidden_size_;

  // The bidirectional LSTM wrapper wraps this LSTM class and produces bi-directional output
  // the output has layout [seq,num_direction,batch,neurons].
  // When num_direction is 2, then this class will compute forward or backward LSTM.
  // The outputs corresponds to either [seq,0,batch,neurons] or [seq,1,batch,neurons]
  // Setting output_step_length this way allows writing the output directly without requiring
  // additional memcpy. Note that if direction is kReverse, we write to output_reverse buffer
  // which is then copied to output buffer, and ReverseSequence method handles the step length.
  if (direction_ == Direction::kForward && num_directions == 2)
    output_step_length = 2 * batch_size_ * hidden_size_;

  gsl::span<T> original_outputs = outputs;
  const bool output_sequence = !outputs.empty();

  if (direction_ == Direction::kReverse) {
    ReverseSequence(inputs, inputs_reverse_, sequence_lengths, seq_length_, batch_size_, input_size_, 1, ttp_);
    inputs = inputs_reverse_;

    if (output_sequence)
      outputs = outputs_reverse_;
  }

  // Calculate the max and min length
  int32_t max_sequence_length = *std::max_element(sequence_lengths.cbegin(), sequence_lengths.cend());
  int32_t min_sequence_length = std::min(seq_length_, *std::min_element(sequence_lengths.cbegin(),
                                                                        sequence_lengths.cend()));

  ///**************************LSTM Calculations****************************/
  const int hidden_size_x4 = 4 * hidden_size_;
  const int total_rows = max_sequence_length * batch_size_;

  // apply the weights to all the inputs and save to output_IOFC
  ComputeGemm(total_rows, hidden_size_x4, input_size_, T{1.0},
              inputs.cbegin(), inputs.cend(),
              input_size_,
              input_weights.cbegin(), input_weights.cend(),  // W[iofc]^T
              input_size_ + attention_size_, T{0.0},
              output_iofc_.begin(), output_iofc_.end(),
              hidden_size_x4, ttp_);

  DumpMatrix("Xt*(W[iofc]^T)", output_iofc_.data(), total_rows, hidden_size_x4);

  int fused_hidden_rows = batch_size_ / hidden_num_threads_;
  if (batch_size_ % hidden_num_threads_ != 0)
    fused_hidden_rows++;

  // NOTE: we could refine the bounds checking in the calls below that use these values to instead
  // explicitly check just the range for each iteration, however if it's going to run over
  // it should also run over on the last iteration, so this should be good enough to catch any
  // logic errors causing bounds violations.
  span_T_iter C_prev_end = batched_internal_state_prev_one_step.end();
  span_T_iter C_prev_clipped_end = batched_internal_state_clipped_one_step.end();
  span_T_const_iter previous_state_end = batched_hidden_state_one_step.end();

  {
    span_T_iter c_prev = batched_internal_state_prev_one_step.begin();
    span_T_iter c_prev_clipped = batched_internal_state_clipped_one_step.begin();

    // hidden state can be provided as input for first step, so need to special case that.
    // after the first step this will switch to the output from the previous step
    span_T_const_iter previous_state = batched_hidden_state_one_step.cbegin();

    //run through steps sequentially
    for (int step = 0; step < max_sequence_length; step++) {
      const std::string seqno_str = " [seqno=" + std::to_string(step) + "]";

      DumpMatrix("previous_state" + seqno_str, &*previous_state, batch_size_, hidden_size_);

      span_T_iter step_out_IOFC = output_iofc_.begin() + (step * batch_size_) * hidden_size_x4;

      // shape is [ attention_size_ ]
      const gsl::span<const T> attention = attention_wrapper_.GetAttnStates();

      // Xt*(W[iofc]^T) = INPUTt * W[iofc]^T + At-1 * WA[iofc]
      ComputeGemm(batch_size_, hidden_size_x4, attention_size_, T{1.0},
                  attention.cbegin(), attention.cend(),  // At-1
                  attention_size_,
                  input_weights.cbegin() + input_size_, input_weights.cend(),  // WA[iofc]
                  input_size_ + attention_size_, T{1.0},
                  step_out_IOFC, output_iofc_.end(),  // input contains Xt*(W[iofc]^T)
                  hidden_size_x4, ttp_);

      // calculate Xt*(W[iofc]^T) + Ht-1*R[iofc]
      ComputeGemm(batch_size_, hidden_size_x4, hidden_size_, T{1.0},
                  previous_state, previous_state_end,  // Ht-1
                  hidden_size_,
                  recurrent_weights.cbegin(), recurrent_weights.cend(),  // R[iofc]
                  hidden_size_, T{1.0},
                  step_out_IOFC, output_iofc_.end(),  // input contains Xt*(W[iofc]^T)
                  hidden_size_x4, ttp_);

      span_T_iter batched_output, batched_output_end;
      if (output_sequence) {
        batched_output = outputs.begin() + step * output_step_length;
        batched_output_end = outputs.end();
      } else {
        batched_output = final_hidden_state.begin();
        batched_output_end = final_hidden_state.end();
      }

      span_T_iter step_out_IOFC_end = step_out_IOFC + batch_size_ * hidden_size_x4;
      GateComputations(step_out_IOFC, step_out_IOFC_end,
                       c_prev, C_prev_end,
                       c_prev_clipped, C_prev_clipped_end,
                       batched_output, batched_output_end,
                       sequence_lengths, min_sequence_length, step, 0, batch_size_, output_sequence);

      // copy last row to final_cell_state
      for (int lrow = 0; lrow < batch_size_; lrow++) {
        if ((step + 1) == sequence_lengths[lrow]) {
          auto src = batched_internal_memory_prev_.subspan(lrow * hidden_size_, hidden_size_);
          auto dst = final_cell_state.subspan(lrow * hidden_size_, hidden_size_);
          gsl::copy(src, dst);
        }
      }

      if (output_sequence) {
        //set to 0 if step >= sequence_length
        for (int lrow = 0; lrow < batch_size_; lrow++) {
          if (step >= min_sequence_length && step >= sequence_lengths[lrow]) {
            auto dst = outputs.data() + step * output_step_length + lrow * hidden_size_;
            std::fill_n(dst, hidden_size_, T{});
          }
        }
      }

      previous_state = batched_output;
      previous_state_end = batched_output_end;

      attention_wrapper_.ProcessOutput(outputs.subspan(step * output_step_length, batch_size_ * hidden_size_));
    }
  }

  if (output_sequence) {
    // copy last output to final_hidden_state
    for (int i = 0; i < batch_size_; i++) {
      const int seq_len = sequence_lengths[i];
      auto src = outputs.subspan((seq_len - 1) * output_step_length + i * hidden_size_, hidden_size_);
      auto dest = final_hidden_state.subspan(i * hidden_size_, hidden_size_);
      gsl::copy(src, dest);
    }

    if (direction_ == Direction::kReverse)
      ReverseSequence<T>(outputs, original_outputs, sequence_lengths, max_sequence_length,
                         batch_size_, hidden_size_, num_directions, ttp_);
  }
}

template <typename T>
void UniDirectionalAttnLstm<T>::GateComputations(span_T_iter& out, span_T_iter& out_end,
                                                 span_T_iter& C_prev, span_T_iter& C_prev_end,  // Ct-1 value not 'ct'. using 'C' for clarity
                                                 span_T_iter& C_prev_clipped, span_T_iter& C_prev_clipped_end,
                                                 span_T_iter& batched_output, span_T_iter& batched_output_end,
                                                 const gsl::span<const int>& seq_lengths,
                                                 const int min_sequence_length,
                                                 const int step,
                                                 const int row,
                                                 const int local_fused_hidden_rows,
                                                 bool output_sequence) {
  int hidden_size_x4 = 4 * hidden_size_;

  // Activation gates.
  for (int b = 0; b < local_fused_hidden_rows; b++) {
    if (step >= min_sequence_length && step >= seq_lengths[row + b]) {
      if (output_sequence) {
        auto fill_output = batched_output + (row + b) * hidden_size_;
        std::fill(fill_output, fill_output + hidden_size_, T{});
      }

      continue;
    }

    std::string row_str = " row[" + std::to_string(row + b) + "]";

    // check that we have hidden_size_x4 left starting at cur_out + b * hidden_size_x4, and get a raw pointer to that
    float* pi = SafeRawPointer<T>(out + b * hidden_size_x4, out_end, hidden_size_x4);
    float* po = pi + hidden_size_;
    float* pf = po + hidden_size_;
    float* pc = pf + hidden_size_;

    float* pCprev_hidden_size = SafeRawPointer<T>(C_prev + b * hidden_size_, C_prev_end, hidden_size_);

    // Input Gate
    if (use_peepholes_) {
      deepcpu::elementwise_product(pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_i_, 0, hidden_size_),
                                   pi, hidden_size_);
    }

    const float* pBi = use_bias_ ? SafeRawConstPointer<T>(bias_WRi_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBi, pi, hidden_size_);  // post: pi has input to f() to calculate i
    activation_f_.func(pi, hidden_size_, activation_f_.alpha, activation_f_.beta);
    //DumpMatrix("i" + row_str, pi, 1, hidden_size_);

    // Forget Gate
    if (input_forget_) {
      for (int i = 0; i < hidden_size_; i++) {
        pf[i] = 1.0f - pi[i];
      }
    } else {
      if (use_peepholes_) {
        deepcpu::elementwise_product(
            pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_f_, 0, hidden_size_), pf, hidden_size_);
      }

      const float* pBf = use_bias_ ? SafeRawConstPointer<T>(bias_WRf_, 0, hidden_size_) : nullptr;
      clip_with_bias_ptr_(clip_, pBf, pf, hidden_size_);
      activation_f_.func(pf, hidden_size_, activation_f_.alpha, activation_f_.beta);
    }

    // Block Gate
    const float* pBc = use_bias_ ? SafeRawConstPointer<T>(bias_WRc_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBc, pc, hidden_size_);
    activation_g_.func(pc, hidden_size_, activation_g_.alpha, activation_g_.beta);

    // C_current. use previous C value as input, and update in-place
    float* pC_cur = pCprev_hidden_size;
    deepcpu::merge_lstm_gates_to_memory(pCprev_hidden_size, pi, pf, pc, pC_cur, hidden_size_);

    // Output Gate
    if (use_peepholes_) {
      deepcpu::elementwise_product(
          pCprev_hidden_size, SafeRawConstPointer<const T>(peephole_o_, 0, hidden_size_), po, hidden_size_);
    }

    // calculate 'ot'
    const float* pBo = use_bias_ ? SafeRawConstPointer<T>(bias_WRo_, 0, hidden_size_) : nullptr;
    clip_with_bias_ptr_(clip_, pBo, po, hidden_size_);
    activation_f_.func(po, hidden_size_, activation_f_.alpha, activation_f_.beta);
    // DumpMatrix("o" + row_str, po, 1, hidden_size_);

    // calculate 'Ht'
    float* pH = SafeRawPointer<T>(batched_output + row * hidden_size_ + b * hidden_size_,
                                  batched_output_end, hidden_size_);

    // the C_prev_clipped location is not actually used as input - it's temporary storage for writing
    // the clipped Ct value to, before calling h(). As such a) it could just be a local variable
    // of std::vector<float> with size of hidden_size_, b) the previous version wasn't 'broken' by never
    // incrementing what C_prev_clipped pointed to.
    float* pC_prev_clipped = SafeRawPointer<T>(C_prev_clipped + b * hidden_size_, C_prev_clipped_end, hidden_size_);

    activation_h_.func(pC_cur, pC_prev_clipped, po, pH, hidden_size_, activation_h_.alpha, activation_h_.beta);
  }

  auto num_rows = local_fused_hidden_rows - row;
  std::string rows_str = " rows[" + std::to_string(row) + ".." + std::to_string(num_rows) + "]";

  DumpMatrix("i" + rows_str, &*out, num_rows, hidden_size_, 0, hidden_size_x4);
  DumpMatrix("o" + rows_str, &*out, num_rows, hidden_size_, 1 * hidden_size_, hidden_size_x4);
  DumpMatrix("f" + rows_str, &*out, num_rows, hidden_size_, 2 * hidden_size_, hidden_size_x4);
  DumpMatrix("c" + rows_str, &*out, num_rows, hidden_size_, 3 * hidden_size_, hidden_size_x4);
  DumpMatrix("C" + rows_str, &*C_prev, num_rows, hidden_size_);  // Ct overwrites the input C_prev value
  DumpMatrix("H" + rows_str, &*batched_output, num_rows, hidden_size_);
}

template <typename T>
void UniDirectionalAttnLstm<T>::SetNumThreads() {
  int threads = std::thread::hardware_concurrency() - 1;

  if (threads < 1)
    threads = 1;

  int hmt = threads;
  batch_parallel_ = false;

  // For readability of the below logic

  // TODO: Temperately removed path: parallelize by partitioning the batch rows,
  //       and its logic in the Compute() method. Will evaluate and finialize it
  //       in later performance tuning stage.
  const auto num_columns = hidden_size_;
  {
    if (hmt > 2 && num_columns <= 128)
      hmt = 2;
    if (hmt > 5 && num_columns <= 256)
      hmt = 5;
    if (hmt > 7 && num_columns <= 512)
      hmt = 7;
    if (hmt > 11 && num_columns <= 1024)
      hmt = 11;

    hidden_num_threads_ = hmt;
  }

  VLOGS(logger_, 1) << "Hidden Threads : " << hidden_num_threads_;
}

template class UniDirectionalAttnLstm<float>;

}  // namespace detail
}  // namespace rnn
}  // namespace contrib
}  // namespace onnxruntime
