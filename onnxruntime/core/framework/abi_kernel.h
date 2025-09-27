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

class PluginEpOpKernel final : public OpKernel {
 public:
  PluginEpOpKernel(const OpKernelInfo& info, OrtKernelImpl* kernel_impl)
      : OpKernel{info}, kernel_impl_{kernel_impl} {}

  ~PluginEpOpKernel() {
    kernel_impl_->Release(kernel_impl_);
  }

  Status Compute(OpKernelContext* ctx) const override {
    return ToStatusAndRelease(kernel_impl_->Compute(kernel_impl_, reinterpret_cast<OrtKernelContext*>(ctx)));
  }

 private:
  OrtKernelImpl* kernel_impl_;
};

class PluginEpKernelCreateFunctor {
 public:
  PluginEpKernelCreateFunctor(OrtKernelCreateFunc create_func, void* state)
      : kernel_create_func_{create_func}, kernel_create_func_state_{state} {}

  Status operator()(FuncManager& fn_manager, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
    OrtKernelCreateContext* create_ctx = reinterpret_cast<OrtKernelCreateContext*>(&fn_manager);
    const OrtKernelInfo* kernel_info = reinterpret_cast<const OrtKernelInfo*>(&info);
    OrtKernelImpl* kernel_impl = nullptr;

    ToStatusAndRelease(kernel_create_func_(create_ctx, kernel_create_func_state_, kernel_info, &kernel_impl));

    out = std::make_unique<PluginEpOpKernel>(info, kernel_impl);
    return Status::OK();
  }

 private:
  OrtKernelCreateFunc kernel_create_func_;
  void* kernel_create_func_state_;
};

}  // namespace onnxruntime
