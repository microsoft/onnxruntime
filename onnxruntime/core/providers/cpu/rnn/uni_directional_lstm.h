// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {
namespace lstm {

using namespace rnn::detail;

// Helper struct for an activation function call information
template <typename TFunc>
struct ActivationInfo {
  TFunc func;
  float alpha;
  float beta;
};

// copying the peephole values into UniDirectionalLstm seems unnecessary. don't do that until proven necessary
#define LSTM_NO_PEEPHOLE_COPY

template <typename T>
class UniDirectionalLstm {
 public:
  UniDirectionalLstm(AllocatorPtr allocator, const logging::Logger& logger, int seq_length, int batch_size,
                     int input_size, int hidden_size, Direction direction, bool input_forget,
                     const gsl::span<const T>& bias, const gsl::span<const T>& peephole_weights,
                     const gsl::span<const T>& initial_hidden_state, const gsl::span<const T>& initial_cell_state,
                     const ActivationFuncs::Entry& activation_func_f, const ActivationFuncs::Entry& activation_func_g,
                     const ActivationFuncs::Entry& activation_func_h, float clip, concurrency::ThreadPool* thread_pool);

  template <typename WeightT>
  void Compute(const gsl::span<const T>& inputs, const gsl::span<const int>& sequence_lengths, int num_directions,
               const GemmWeights<WeightT>& input_weights, const GemmWeights<WeightT>& recurrent_weights, gsl::span<T>& outputs,
               gsl::span<T>& final_hidden_state, gsl::span<T>& final_cell_state);

  ~UniDirectionalLstm() = default;

 private:
  using span_T_const_iter = typename gsl::span<T>::const_iterator;
  using span_T_iter = typename gsl::span<T>::iterator;

  void SetNumThreads();

  void GateComputations(span_T_iter& out, span_T_iter& out_end, span_T_iter& C_prev,
                        const span_T_iter& C_prev_end,  // Ct-1 value not 'ct'. using 'C' for clarity
                        span_T_iter& C_prev_clipped, const span_T_iter& C_prev_clipped_end, span_T_iter& batched_output,
                        span_T_iter& batched_output_end, const gsl::span<const int>& seq_lengths,
                        int min_sequence_length, int step, int row, int local_fused_hidden_rows, bool output_sequence);

  void AllocateBuffers();

  void InitializeBuffers(const gsl::span<const T>& initial_hidden_state, const gsl::span<const T>& initial_cell_state);

  void LoadPeepholeWeights(const gsl::span<const T>& peephole_weights);
  void LoadBias(const gsl::span<const T>& WbRb_values);

  AllocatorPtr allocator_;
  const logging::Logger& logger_;

  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;

  Direction direction_;
  bool input_forget_;
  float clip_;

  bool batch_parallel_;

  bool use_bias_;
  bool use_peepholes_;

  int num_threads_ = -1;

  IAllocatorUniquePtr<T> output_iofc_ptr_;
  IAllocatorUniquePtr<T> hidden0_ptr_, batched_hidden0_ptr_;
  gsl::span<T> output_iofc_;
  gsl::span<T> hidden0_, batched_hidden0_;

  IAllocatorUniquePtr<T> internal_memory_prev_ptr_, batched_internal_memory_prev_ptr_;
  IAllocatorUniquePtr<T> batched_internal_memory_clipped_ptr_;
  gsl::span<T> internal_memory_prev_, batched_internal_memory_prev_;
  gsl::span<T> batched_internal_memory_clipped_;

  IAllocatorUniquePtr<T> bias_WRi_ptr_, bias_WRf_ptr_, bias_WRo_ptr_, bias_WRc_ptr_;
  IAllocatorUniquePtr<T> peephole_i_ptr_, peephole_f_ptr_, peephole_o_ptr_;
  IAllocatorUniquePtr<T> inputs_reverse_ptr_, outputs_reverse_ptr_;
  gsl::span<T> bias_WRi_, bias_WRf_, bias_WRo_, bias_WRc_;
  gsl::span<T> inputs_reverse_, outputs_reverse_;

#if defined(LSTM_NO_PEEPHOLE_COPY)
  gsl::span<const T> peephole_i_, peephole_f_, peephole_o_;
#else
  gsl::span<T> peephole_i_, peephole_f_, peephole_o_;
#endif

  IAllocatorUniquePtr<int> sequence_lengths_ptr_;
  gsl::span<int> sequence_lengths_;

  deepcpu::ClipWithBiasFuncPtr clip_with_bias_ptr_;

  ActivationInfo<deepcpu::ActivationFuncPtr> activation_f_;
  ActivationInfo<deepcpu::ActivationFuncPtr> activation_g_;
  ActivationInfo<deepcpu::LstmMergeGatesFuncPtr> activation_h_;

  concurrency::ThreadPool* thread_pool_;

  // Quantized operation related allocation members
  template <typename WeightT>
  void AllocateQuantizeBuffers(int max_sequence_length);

  // Buffer shared for quantized input whole, and quantized a each sequence step
  IAllocatorUniquePtr<uint8_t> quantized_input_or_a_ptr_;
  gsl::span<uint8_t> quantized_input_or_a_;

  IAllocatorUniquePtr<int32_t> quantize_agg_C_ptr_;
  gsl::span<int32_t> quantize_agg_C_;
};

}  // namespace lstm
}  // namespace onnxruntime
