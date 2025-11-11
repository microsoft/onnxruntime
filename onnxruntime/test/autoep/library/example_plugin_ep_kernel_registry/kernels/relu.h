// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "utils.h"
#include "../../plugin_ep_utils.h"

struct Relu : public OrtKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Relu>& kernel);

  Relu(const OrtKernelInfo* info, void* state, PrivateTag);

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

  OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) noexcept;

 private:
  const OrtKernelInfo* info_;
  void* state_{nullptr};  // Custom state passed from OrtEp
};
