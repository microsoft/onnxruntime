// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/string_utils.h"

// macro "SS" - declare an ostream variable and its string buffer
#define SS(ss, reserve_size) ::onnxruntime::webgpu::OStringStream ss(reserve_size)

// macro "SS_GET" - get the string from the ostream
#define SS_GET(ss) (std::move(ss).str())

// macro "SS_APPEND" - use function call style to append to the ostream
#define SS_APPEND(ss, ...) ::onnxruntime::webgpu::detail::OStringStreamAppend(ss, __VA_ARGS__)
