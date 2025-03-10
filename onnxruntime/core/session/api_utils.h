// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include <string_view>

onnxruntime::common::Status CopyStringToOutputArg(std::string_view str, const char* err_msg, char* out, size_t* size);
