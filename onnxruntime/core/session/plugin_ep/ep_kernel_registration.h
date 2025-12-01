// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/data_types.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

/// <summary>
/// Make a KernelCreateInfo for a plugin EP's kernel. A KernelCreateInfo contains the function and state
/// necessary to create a kernel.
/// </summary>
/// <param name="kernel_def"></param>
/// <param name="kernel_create_func"></param>
/// <param name="kernel_create_func_state"></param>
/// <returns></returns>
KernelCreateInfo MakePluginEpKernelCreateInfo(const KernelDef* kernel_def,
                                              OrtKernelCreateFunc kernel_create_func,
                                              void* kernel_create_func_state);

/// <summary>
/// Gets the kernel registry for a plugin EP.
/// </summary>
/// <param name="ort_ep">The OrtEp instance.</param>
/// <param name="kernel_registry">Output parameter set to the EP's registry.</param>
/// <returns>A status indicating success or an error</returns>
Status GetPluginEpKernelRegistry(OrtEp& ort_ep, /*out*/ std::shared_ptr<KernelRegistry>& kernel_registry);

}  // namespace onnxruntime
