// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <tuple>
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>

#include "sharding.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(ORT_USE_NCCL)

template <typename T, typename Tind>
class DistributedSqueeze final : public DistributedKernel {
 public:
  explicit DistributedSqueeze(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
