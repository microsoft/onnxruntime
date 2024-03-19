// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#if defined(__GNUC__)
#include "onnxruntime_config.h"
#pragma GCC diagnostic push

#ifdef HAS_SHORTEN_64_TO_32
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif
#endif

#include "flatbuffers/flatbuffers.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif