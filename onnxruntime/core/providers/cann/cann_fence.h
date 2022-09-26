// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/fence.h"
#include "core/providers/cann/cann_inc.h"

namespace onnxruntime {
class NPUDataTransfer;

class CANNFence : public IFence {
 public:
  explicit CANNFence(const NPUDataTransfer* data_transfer);
  virtual ~CANNFence();
  void BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int queue_id) override;
  void BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) override;
  void AfterUsedAsInput(int queue_id) override;
  void AfterUsedAsOutput(int queue_id) override;
  bool CanRelease() override;

 private:
  aclrtEvent read_event_;
  aclrtEvent write_event_;
  const NPUDataTransfer* data_transfer_;
};

}  // namespace onnxruntime
