// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/tensor/gather.h"
#include "contrib_ops/cpu/tensor/shrunken_gather.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

class ShrunkenGather final : public onnxruntime::cuda::Gather, public ShrunkenGatherCommon {
 public:
  ShrunkenGather(const OpKernelInfo& info) : onnxruntime::cuda::Gather(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif
