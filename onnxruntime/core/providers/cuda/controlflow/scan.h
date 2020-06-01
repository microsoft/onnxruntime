// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"

#include "core/common/common.h"
#include "core/providers/cpu/controlflow/scan.h"
#include "core/providers/cuda/cuda_execution_provider.h"

namespace onnxruntime {
class SessionState;

namespace cuda {

// Use the CPU implementation for the logic
template <int OpSet>
class Scan final : public onnxruntime::Scan<OpSet> {
 public:
  Scan(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

 private:
};
}  // namespace cuda
}  // namespace onnxruntime
