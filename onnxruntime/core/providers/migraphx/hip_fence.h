// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/tensor.h"
#include "core/graph/basic_types.h"

namespace onnxruntime {
class GPUDataTransfer;

class HIPFence : public IFence {
 public:
  HIPFence(const GPUDataTransfer* data_transfer);
  virtual ~HIPFence();
  virtual void BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int queue_id) override;
  virtual void BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) override;
  virtual void AfterUsedAsInput(int queue_id) override;
  virtual void AfterUsedAsOutput(int queue_id) override;
  virtual bool CanRelease() override;

 private:
  hipEvent_t read_event_;
  hipEvent_t write_event_;
  const GPUDataTransfer* data_transfer_;
};

}  // namespace onnxruntime
