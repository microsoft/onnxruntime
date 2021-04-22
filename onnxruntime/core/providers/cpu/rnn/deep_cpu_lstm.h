// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "lstm_base.h"

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"

namespace onnxruntime {

/// The class represents DeepCPU implementation of a long short term memory (LSTM) operator.
/// For details, refer to http://aka.ms/dl-optimization/.
class DeepCpuLstmOp final : public OpKernel, public LSTMBase {
 public:
  DeepCpuLstmOp(const OpKernelInfo& info) : OpKernel(info), LSTMBase(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed,
                 /*in_out*/ PackedWeight& cached_prepacked_tensor,
                 /*out*/ bool& read_from_cache,
                 AllocatorPtr alloc_for_caching) override;
  Status Compute(OpKernelContext* context) const override;

  ~DeepCpuLstmOp() override = default;

 private:
  Status TryPackWeights(const Tensor& weights, rnn::detail::PackedWeights& packed_weights,
                        bool& is_packed, AllocatorPtr alloc);

  template <typename T>
  Status ComputeImpl(OpKernelContext& context) const;

  rnn::detail::PackedWeights packed_W_;
  rnn::detail::PackedWeights packed_R_;
};

}  // namespace onnxruntime
