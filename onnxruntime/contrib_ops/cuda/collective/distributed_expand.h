// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sharding_spec.h"
#include "sharding.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <algorithm>
#include <tuple>
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T>
class DistributedExpand final : public DistributedKernel {
 public:
  explicit DistributedExpand(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
