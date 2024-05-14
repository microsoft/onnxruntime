// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/sampling.h"

namespace onnxruntime {
class SessionState;

namespace contrib {
namespace cuda {

class Sampling final : public onnxruntime::contrib::transformers::Sampling {
 public:
  Sampling(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  Status ComputeInternal(OpKernelContext* context) const;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
