// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"

#include "core/providers/cpu/controlflow/scan.h"

namespace onnxruntime {
class SessionState;

namespace cuda {

// Use the CPU implementation for the logic
template <int OpSet>
class Scan final : public OpKernel {
 public:
  Scan(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

 private:
  std::unique_ptr<onnxruntime::Scan<OpSet>> scan_cpu_;
};
}  // namespace cuda
}  // namespace onnxruntime
