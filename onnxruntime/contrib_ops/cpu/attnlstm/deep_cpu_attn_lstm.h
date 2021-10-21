// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "attention_wrapper.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {
namespace contrib {

using onnxruntime::rnn::detail::ActivationFuncs;
using onnxruntime::rnn::detail::Direction;
using onnxruntime::rnn::detail::MakeDirection;

// The class represents DeepCPU implementation of a long short term memory (LSTM) plus a Bahdanau Attention wraper.
// The equivilent python usage could be checked int the corresponding op test directory, attention_lstm_data_gen.py.
// Also please note that detail implementation re-used lot of code from current ONNXRuntime LSTM operator, refactor
// is needed in future if this is become part of ONNX.
class DeepCpuAttnLstmOp final : public OpKernel {
 public:
  DeepCpuAttnLstmOp(const OpKernelInfo& info)
      : OpKernel(info), clip_(info.GetAttrOrDefault<float>("clip", std::numeric_limits<float>::max())) {
    std::string direction;
    ORT_ENFORCE(info.GetAttr("direction", &direction).IsOK());

    int64_t int64_value;
    ORT_ENFORCE(info.GetAttr("hidden_size", &int64_value).IsOK() && int64_value > 0);
    hidden_size_ = gsl::narrow<int>(int64_value);

    // optional attributes
    std::vector<std::string> activation_func_names = info.GetAttrsOrDefault<std::string>("activations");
    std::vector<float> activation_func_alphas = info.GetAttrsOrDefault<float>("activation_alpha");
    std::vector<float> activation_func_betas = info.GetAttrsOrDefault<float>("activation_beta");
    ORT_ENFORCE(clip_ > 0.f);

    if (info.GetAttr("input_forget", &int64_value).IsOK())
      input_forget_ = int64_value != 0;

    direction_ = MakeDirection(direction);
    num_directions_ = direction_ == Direction::kBidirectional ? 2 : 1;

    if (activation_func_names.empty()) {
      for (int i = 0; i < num_directions_; ++i) {
        activation_func_names.emplace_back("sigmoid");
        activation_func_names.emplace_back("tanh");
        activation_func_names.emplace_back("tanh");
      }
    }

    ORT_ENFORCE(static_cast<int>(activation_func_names.size()) == num_directions_ * 3);

    activation_funcs_ = ActivationFuncs(activation_func_names,
                                        activation_func_alphas,
                                        activation_func_betas);
  }

  Status Compute(OpKernelContext* context) const override;

  ~DeepCpuAttnLstmOp() override = default;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;

  Status ValidateInputs(const Tensor& X,
                        const Tensor& W,
                        const Tensor& R,
                        const Tensor* B,
                        const Tensor* sequence_lens,
                        const Tensor* initial_h,
                        const Tensor* initial_c,
                        const Tensor* P,
                        int batch_size,
                        const Tensor& am_query_layer_weights,
                        const Tensor& am_memory_layer_weights,
                        const Tensor& am_v_weights,
                        const Tensor& attn_memory,
                        const Tensor* attn_memory_seq_lens,
                        const Tensor* attn_layer_weights) const;

  Direction direction_;
  int num_directions_;

  int hidden_size_ = 0;
  float clip_;
  bool input_forget_ = false;

  ActivationFuncs activation_funcs_;
};

}  // namespace contrib
}  // namespace onnxruntime
