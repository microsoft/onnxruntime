// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "sharding_spec.h"
#include "nccl_kernels.h"

#include <algorithm>
#include <tuple>
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T>
class DistributedSliice final : public DistributedKernel {
 public:
  explicit DistributedSliice(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
