// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// File to include the required TRT headers with workarounds for warnings we can't fix.

// Ignore warning C4100: unreferenced formal parameter
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
