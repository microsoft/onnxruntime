// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/transformers/greedy_search_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

struct SamplingParameters : public GreadySearchParameters {
  void ParseFromAttributes(const OpKernelInfo& info);

  float presence_penalty;
  float temperature;
  float top_p;
  gsl::span<const int32_t> presence_mask;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
