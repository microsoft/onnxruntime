// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/greedy_search.h"

namespace onnxruntime {
class SessionState;

namespace contrib {
namespace cuda {

class GreedySearch final : public onnxruntime::contrib::transformers::GreedySearch {
 public:
  GreedySearch(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeInternal(OpKernelContext* context) const;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
