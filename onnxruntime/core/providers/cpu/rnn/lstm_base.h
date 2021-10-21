// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {

/// The class represents DeepCPU implementation of a long short term memory (LSTM) operator.
/// For details, refer to http://aka.ms/dl-optimization/.
class LSTMBase {
 protected:
  LSTMBase(const OpKernelInfo& info)
      : clip_(info.GetAttrOrDefault<float>("clip", std::numeric_limits<float>::max())),
        layout_(info.GetAttrOrDefault("layout", static_cast<int64_t>(0))) {
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

    direction_ = rnn::detail::MakeDirection(direction);
    num_directions_ = direction_ == rnn::detail::Direction::kBidirectional ? 2 : 1;

    if (activation_func_names.empty()) {
      for (int i = 0; i < num_directions_; ++i) {
        activation_func_names.emplace_back("sigmoid");
        activation_func_names.emplace_back("tanh");
        activation_func_names.emplace_back("tanh");
      }
    }

    ORT_ENFORCE(activation_func_names.size() == static_cast<size_t>(num_directions_) * 3);

    activation_funcs_ = rnn::detail::ActivationFuncs(activation_func_names,
                                                     activation_func_alphas,
                                                     activation_func_betas);

    ORT_ENFORCE(layout_ == 0,
                "Batchwise recurrent operations (layout == 1) are not supported. If you need support create a github issue with justification.");
  }

  ~LSTMBase() = default;

  template <typename InputT, typename WeightT>
  Status ComputeImpl(OpKernelContext& context,
                     const rnn::detail::GemmWeights<WeightT>& W_1,
                     const rnn::detail::GemmWeights<WeightT>& W_2,
                     const rnn::detail::GemmWeights<WeightT>& R_1,
                     const rnn::detail::GemmWeights<WeightT>& R_2) const;

  Status ValidateInputs(const Tensor& X,
                        const Tensor* B,
                        const Tensor* sequence_lens,
                        const Tensor* initial_h,
                        const Tensor* initial_c,
                        const Tensor* P) const;

  rnn::detail::Direction direction_;
  int num_directions_;

  int hidden_size_ = 0;
  float clip_;
  bool input_forget_ = false;
  int64_t layout_;

  rnn::detail::ActivationFuncs activation_funcs_;
};

}  // namespace onnxruntime
