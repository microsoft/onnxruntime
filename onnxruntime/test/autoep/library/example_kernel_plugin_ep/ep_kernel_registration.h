// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "../plugin_ep_utils.h"

size_t GetNumKernels();

OrtStatus* CreateKernelRegistry(const char* ep_name, void* create_kernel_state, OrtKernelRegistry** kernel_registry);
