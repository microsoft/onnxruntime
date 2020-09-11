// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "orttraining/training_ops/cuda/collective/nccl_common.h"

#include "orttraining/core/framework/adasum/adasum_interface.h"

namespace onnxruntime {
namespace cuda {

class AdasumAllReduce final : public NcclKernel {
 public:
  explicit AdasumAllReduce(const OpKernelInfo& info) : NcclKernel(info) {
    unique_name_ = "AdasumAllReduceNode_" + info.node().Name();
    int64_t adasum_type = training::AdasumReductionType::None;
    info.GetAttrOrDefault("reduce_type", &adasum_type, static_cast<int64_t>(training::AdasumReductionType::None));
    adasum_reduction_type_ = static_cast<training::AdasumReductionType>(adasum_type);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string unique_name_;
  training::AdasumReductionType adasum_reduction_type_ = training::AdasumReductionType::None;
};

}  // namespace cuda
}  // namespace onnxruntime
