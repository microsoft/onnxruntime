// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

struct If {
  static OrtStatus* ORT_API_CALL CreateKernel(void* kernel_create_func_state,
                                              const OrtKernelInfo* info,
                                              OrtKernelImpl** kernel_out) noexcept;
};
