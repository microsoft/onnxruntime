// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <string>
#include "core/common/common.h"
#include "contrib_ops/cpu/transformers/greedy_search.h"
#include "contrib_ops/cpu/transformers/sampling_parameters.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"


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
