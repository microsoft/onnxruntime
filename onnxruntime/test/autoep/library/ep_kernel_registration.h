// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "example_plugin_ep_utils.h"

size_t GetNumKernels();

OrtStatus* CreateKernelCreateInfos(const char* ep_name, std::vector<OrtKernelCreateInfo*>& kernel_create_infos);
