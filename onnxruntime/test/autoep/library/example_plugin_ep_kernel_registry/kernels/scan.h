// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

class Scan : public OrtKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Scan>& kernel) noexcept;
  Scan(const OrtKernelInfo* info, void* state, PrivateTag);
  ~Scan();

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetControlFlowKernelImpl(OrtKernelImpl* this_ptr, OrtKernelImpl** out) noexcept;

 private:
  const OrtKernelInfo* info_;
  OrtDataTransferImpl* data_transfer_impl_;  // Custom state passed from OrtEp
  OrtKernelImpl* control_flow_kernel_{};
};
