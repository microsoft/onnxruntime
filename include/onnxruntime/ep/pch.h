// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// This header is only used when building WebGPU/CUDA EP as a shared library.
//
// This header file is used as a precompiled header so it is always included first.

#pragma push_macro("ORT_EP_API_HEADER_INCLUDED")
#define ORT_EP_API_HEADER_INCLUDED

#include "api.h"
//#include "ep.h"
#include "common.h"
#include "logging.h"
#include "kernel_registry.h"

#pragma pop_macro("ORT_EP_API_HEADER_INCLUDED")

namespace onnxruntime {
namespace webgpu {
EP_SPECIFIC_USING_DECLARATIONS
}  // namespace webgpu

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
namespace webgpu {
EP_SPECIFIC_USING_DECLARATIONS
}  // namespace webgpu
}  // namespace contrib
#endif

}  // namespace onnxruntime
