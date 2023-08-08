// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef USE_MPI
#pragma once

#include "core/framework/op_kernel.h"

#include "orttraining/core/framework/adasum/adasum_mpi.h"

namespace onnxruntime {
namespace contrib {
class AdasumAllReduce final : public OpKernel {
 public:
  AdasumAllReduce(const OpKernelInfo& info) : OpKernel(info) {
    int64_t adasum_reduce_algo;
    info.GetAttrOrDefault("reduce_algo", &adasum_reduce_algo, static_cast<int64_t>(0));
    adasum_reduce_algo_ = static_cast<training::AdasumReductionType>(adasum_reduce_algo);
    adasum_reducer_ = std::make_unique<training::AdasumMPI>();
    if (!adasum_reducer_->IsAdasumInitialized()) {
      adasum_reducer_->InitializeVHDDReductionComms();
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  training::AdasumReductionType adasum_reduce_algo_ = training::AdasumReductionType::CpuReduction;
  std::unique_ptr<training::AdasumMPI> adasum_reducer_;
};
}  // namespace contrib
}  // namespace onnxruntime
#endif  // USE_MPI
