// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "example_plugin_ep_utils.h"

size_t GetNumKernels();

OrtStatus* CreateKernelRegistry(const char* ep_name, OrtKernelRegistry** kernel_registry);
