// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include "core/framework/data_types.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/op_kernel.h"

struct OrtMLDataType : onnxruntime::DataTypeImpl {};

struct OrtKernelDefBuilder : onnxruntime::KernelDefBuilder {};

struct OrtKernelDef : onnxruntime::KernelDef {};

struct OrtKernelCreateInfo {
  OrtKernelDef kernel_def;
  OrtKernelCreateFunc kernel_create_func;
  void* kernel_create_func_state;
};

namespace onnxruntime {

Status InitKernelRegistry(OrtEp& ort_ep, /*out*/ std::shared_ptr<KernelRegistry>& kernel_registry);

}  // namespace onnxruntime
