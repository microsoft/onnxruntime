// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/transformers/greedy_search_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

struct SamplingParameters : public GreedySearchParameters {
  void ParseFromAttributes(const OpKernelInfo& info) override;

  void ParseFromInputs(OpKernelContext* context);
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
