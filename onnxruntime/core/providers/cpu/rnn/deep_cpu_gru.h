// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {

/// The class represents GRU operator using DeepCPU implementation for
/// fast inference computation on CPU machines.
class DeepCpuGruOp final : public OpKernel {
 public:
  DeepCpuGruOp(const OpKernelInfo& info) : OpKernel(info) {
    // required attributes
    std::string direction;
    ORT_ENFORCE(info.GetAttr("direction", &direction).IsOK());

    int64_t int64_value;
    ORT_ENFORCE(info.GetAttr("linear_before_reset", &int64_value).IsOK());
    linear_before_reset_ = narrow<int>(int64_value);

    ORT_ENFORCE(info.GetAttr("hidden_size", &int64_value).IsOK() && int64_value > 0);
    hidden_size_ = narrow<int>(int64_value);

    // optional attributes
    std::vector<std::string> activation_func_names = info.GetAttrsOrDefault<std::string>("activations");
    std::vector<float> activation_func_alphas = info.GetAttrsOrDefault<float>("activation_alpha");
    std::vector<float> activation_func_betas = info.GetAttrsOrDefault<float>("activation_beta");

    clip_ = info.GetAttrOrDefault<float>("clip", std::numeric_limits<float>::max());
    ORT_ENFORCE(clip_ > 0.f);

    direction_ = rnn::detail::MakeDirection(direction);
    num_directions_ = direction_ == rnn::detail::Direction::kBidirectional ? 2 : 1;

    if (activation_func_names.empty()) {
      for (int i = 0; i < num_directions_; ++i) {
        activation_func_names.emplace_back("sigmoid");
        activation_func_names.emplace_back("tanh");
      }
    }

    ORT_ENFORCE(activation_func_names.size() == static_cast<size_t>(num_directions_) * 2);

    activation_funcs_ = rnn::detail::ActivationFuncs(activation_func_names,
                                                     activation_func_alphas,
                                                     activation_func_betas);

    layout_ = info.GetAttrOrDefault("layout", static_cast<int64_t>(0));
    ORT_ENFORCE(layout_ == 0,
                "Batchwise recurrent operations (layout == 1) are not supported. If you need support create a github issue with justification.");
  }

  Status Compute(OpKernelContext* context) const override;

  ~DeepCpuGruOp() override = default;

 private:
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

  bool TryPackInputWeights(const Tensor& weight, AllocatorPtr& alloc);

  bool TryPackRecurrentWeights(const Tensor& weights, AllocatorPtr& alloc);

  rnn::detail::Direction direction_;
  int num_directions_;

  int hidden_size_{};
  float clip_;
  int linear_before_reset_{};
  int64_t layout_;

  rnn::detail::ActivationFuncs activation_funcs_;

  // This kernel supports either forward or bidirectional
  // This is split in half for bidirectional, but we prepack it in the same buffer
  rnn::detail::PackedWeights pre_packed_input_weights_;
  // recurrent_weights_ZR_ fwd, followed by bwd
  rnn::detail::PackedWeights pre_packed_recurrent_ZR_;
  // recurrent_weights_H_ fwd, followed by bwd
  rnn::detail::PackedWeights pre_packed_recurrent_H_;

  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;
};

namespace detail {

template <typename T>
class UniDirectionalGru {
 public:
  UniDirectionalGru(AllocatorPtr allocator, int seq_length, int batch_size, int input_size, int hidden_size,
                    bool linear_before_reset, rnn::detail::Direction direction, gsl::span<const T> bias,
                    gsl::span<const T> initial_hidden_state, const rnn::detail::ActivationFuncs::Entry& activation_func_f,
                    const rnn::detail::ActivationFuncs::Entry& activation_func_g, float clip,
                    onnxruntime::concurrency::ThreadPool* ttp,
                    const bool training_mode = false);

  void Compute(gsl::span<const T> inputs, gsl::span<const int> sequence_lengths, int num_directions,
               const rnn::detail::GemmWeights<T>& input_weights,
               const rnn::detail::GemmWeights<T>& recurrent_weights_ZR,
               const rnn::detail::GemmWeights<T>& recurrent_weights_H,
               gsl::span<T>& outputs, gsl::span<T>& final_hidden_state);

  // This function overloads the above one by adding two additional reference inputs that are computed in this kernel:
  //   - zrh: intermediate gate computations
  // This extra output is needed for training for gradient computation.
  void Compute(gsl::span<const T> inputs, gsl::span<const int> sequence_lengths, int num_directions,
               const rnn::detail::GemmWeights<T>& input_weights,
               const rnn::detail::GemmWeights<T>& recurrent_weights_ZR,
               const rnn::detail::GemmWeights<T>& recurrent_weights_H,
               gsl::span<T>& outputs, gsl::span<T>& final_hidden_state,
               gsl::span<T>& zrh);

  ~UniDirectionalGru() = default;

 private:
  void ComputeImpl(gsl::span<const T> inputs, gsl::span<const int> sequence_lengths, int num_directions,
                   const rnn::detail::GemmWeights<T>& input_weights,
                   const rnn::detail::GemmWeights<T>& recurrent_weights_ZR,
                   const rnn::detail::GemmWeights<T>& recurrent_weights_H,
                   gsl::span<T>& outputs, gsl::span<T>& final_hidden_state,
                   gsl::span<T>& zrh);

  AllocatorPtr allocator_;

  int seq_length_;
  int batch_size_;
  int input_size_;
  int hidden_size_;
  bool linear_before_reset_;

  const float clip_;

  rnn::detail::Direction direction_;
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

  rnn::detail::deepcpu::ClipWithBiasFuncPtr clip_with_bias_ptr_{};

  float zr_alpha_{};
  float zr_beta_{};
  float h_alpha_{};
  float h_beta_{};

  rnn::detail::deepcpu::GruResetGateFuncPtr reset_gate_{};
  rnn::detail::deepcpu::ActivationFuncPtr update_gate_{};
  rnn::detail::deepcpu::GruOutputGateFuncPtr output_gate_{};

  void AllocateBuffers();

  onnxruntime::concurrency::ThreadPool* ttp_;

  const bool training_mode_ = false;
};
}  // namespace detail

}  // namespace onnxruntime
