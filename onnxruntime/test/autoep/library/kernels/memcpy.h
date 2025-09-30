// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../example_plugin_ep_utils.h"

struct Memcpy : public OrtKernelImpl {
  static OrtStatus* Create(const OrtKernelInfo* info, /*out*/ std::unique_ptr<Memcpy>& kernel);

  Memcpy(ApiPtrs api, const OrtKernelInfo* info);

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

  OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) noexcept;

  ApiPtrs api;
  const OrtKernelInfo* info;
};
