// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

class Squeeze : public OrtKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel) noexcept;
  Squeeze(const OrtKernelInfo* info, void* state, PrivateTag);

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  const OrtKernelInfo* info_;
  OrtDataTransferImpl* data_transfer_impl_;  // Custom state passed from OrtEp
};
