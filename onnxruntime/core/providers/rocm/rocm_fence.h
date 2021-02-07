// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include "core/graph/basic_types.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
class GPUDataTransfer;

class ROCMFence : public IFence {
 public:
  ROCMFence(const GPUDataTransfer* data_transfer);
  virtual ~ROCMFence();
  void BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int queue_id) override;
  void BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) override;
  void AfterUsedAsInput(int queue_id) override;
  void AfterUsedAsOutput(int queue_id) override;
  bool CanRelease() override;

 private:
  hipEvent_t read_event_;
  hipEvent_t write_event_;
  const GPUDataTransfer* data_transfer_;
};

}  // namespace onnxruntime
