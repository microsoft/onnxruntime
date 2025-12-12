// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "base.h"
#include "../../plugin_ep_utils.h"

class Mul : public BaseKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static OrtStatus* Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Mul>& kernel);
  Mul(const OrtKernelInfo* info, void* state, PrivateTag);

 private:
  OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) override;
};
