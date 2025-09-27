// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include "core/framework/kernel_def_builder.h"

struct OrtKernelDefBuilder : onnxruntime::KernelDefBuilder {
};

struct OrtKernelDef : onnxruntime::KernelDef {
};

struct OrtKernelCreateInfo {
  OrtKernelDef kernel_def;
  OrtKernelCreateFunc kernel_create_func;
  void* kernel_create_func_state;
};
