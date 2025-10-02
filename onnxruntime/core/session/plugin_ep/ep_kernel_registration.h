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

struct OrtMLDataType : onnxruntime::DataTypeImpl {};

struct OrtKernelDefBuilder : onnxruntime::KernelDefBuilder {};

struct OrtKernelDef : onnxruntime::KernelDef {};

struct OrtKernelRegistry {
  std::shared_ptr<onnxruntime::KernelRegistry> registry;
};

namespace onnxruntime {

/// <summary>
/// A functor that creates a PluginEpOpKernel instance using the creation function (+ state) provided by a plugin EP.
/// </summary>
class PluginEpKernelCreateFunctor {
 public:
  PluginEpKernelCreateFunctor();
  PluginEpKernelCreateFunctor(OrtKernelCreateFunc create_func, void* state);

  Status operator()(FuncManager& fn_manager, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out);

 private:
  OrtKernelCreateFunc kernel_create_func_;
  void* kernel_create_func_state_;
};

Status InitKernelRegistry(OrtEp& ort_ep, /*out*/ std::shared_ptr<KernelRegistry>& kernel_registry);

}  // namespace onnxruntime
