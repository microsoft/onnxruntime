// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "sharding_spec.h"
#include "core/providers/cuda/cuda_kernel.h"

#if defined(ORT_USE_NCCL)
#include <algorithm>
#include <tuple>
#include <optional>
#include <string>
#include <nccl.h>
#include <sstream>
#endif

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class DistributedMatMul final : public NcclKernel {
 public:
  explicit DistributedMatMul(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::vector<TensorPartitionSpec> input_shard_specs_;
  std::vector<TensorPartitionSpec> output_shard_specs_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
