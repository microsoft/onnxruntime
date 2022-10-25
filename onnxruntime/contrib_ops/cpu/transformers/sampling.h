// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "contrib_ops/cpu/transformers/greedy_search.h"
#include "contrib_ops/cpu/transformers/sampling_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

class Sampling : public GreedySearch {
 public:
  explicit Sampling(const OpKernelInfo& info) : GreedySearch(info) {}

  Status Compute(OpKernelContext* ctx) const override;

 private:
  SamplingParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
