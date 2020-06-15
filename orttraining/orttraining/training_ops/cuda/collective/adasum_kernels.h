// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "nccl_common.h"

namespace onnxruntime {
namespace cuda {

class AdasumAllReduce final : public NcclKernel {
 public:
  explicit AdasumAllReduce(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
