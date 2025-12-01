// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

struct BaseKernelImpl : public OrtKernelImpl {
  BaseKernelImpl(const OrtKernelInfo* info, void* state);
  virtual ~BaseKernelImpl() = default;

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

  // derived classes implement DoCompute.
  virtual OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) noexcept = 0;

 protected:
  const OrtKernelInfo* info_;
  void* state_;  // Custom state passed from OrtEp
};
