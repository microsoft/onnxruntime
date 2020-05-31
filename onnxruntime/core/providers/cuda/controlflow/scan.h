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
  // We need to access to the CUDA EP instance to get the cublas handle which is
  // needed for the CUDA Transpose operation
  CUDAExecutionProvider* cuda_ep_;
};
}  // namespace cuda
}  // namespace onnxruntime
