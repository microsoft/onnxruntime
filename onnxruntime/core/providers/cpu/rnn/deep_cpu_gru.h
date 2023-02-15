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

}  // namespace onnxruntime
