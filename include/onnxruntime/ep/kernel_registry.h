// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/ep.h instead."
#endif

#include <memory>

#include "kernel_def_builder.h"
#include "op_kernel_info.h"
#include "op_kernel.h"

#include "core/graph/basic_types.h"

namespace onnxruntime {
namespace ep {
namespace detail {

struct FuncManager {};
using KernelCreatePtrFn = std::add_pointer<Status(FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out)>::type;

struct KernelCreateInfo {
  Ort::KernelDef kernel_def;
  KernelCreatePtrFn kernel_create_func;
  Status status;

  KernelCreateInfo(Ort::KernelDef definition,
                   KernelCreatePtrFn create_func)
      : kernel_def(std::move(definition)),
        kernel_create_func(create_func) {
    assert(kernel_def != nullptr);
  }

  KernelCreateInfo(KernelCreateInfo&& other) noexcept
      : kernel_def(std::move(other.kernel_def)),
        kernel_create_func(std::move(other.kernel_create_func)) {}

  KernelCreateInfo() = default;
};

struct KernelRegistry {
  KernelRegistry() = default;

  Status Register(KernelCreateInfo&& create_info) {
    registry_.AddKernel(create_info.kernel_def, [](void* kernel_create_func_state, const OrtKernelInfo* info, OrtKernelImpl** out) -> OrtStatus* {
                              FuncManager func_mgr; // not used
                              std::unique_ptr<OpKernel> kernel;
                              KernelCreatePtrFn* create_func = reinterpret_cast<KernelCreatePtrFn*>(kernel_create_func_state);
                              Status status = (*create_func)(func_mgr, OpKernelInfo(info), kernel);
                              if (!status.IsOK()) {
                                return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, status.ErrorMessage().c_str());
                              }
                              *out = new KernelImpl(std::move(kernel));
                              return nullptr; }, static_cast<void*>(create_info.kernel_create_func));
    return Status::OK();
  }

  // Implicit conversion to OrtKernelRegistry* for compatibility with C API
  operator OrtKernelRegistry*() const noexcept {
    return registry_.operator OrtKernelRegistry*();
  }

  // Release ownership of the underlying OrtKernelRegistry*
  OrtKernelRegistry* release() {
    return registry_.release();
  }

 private:
  Ort::KernelRegistry registry_;
};

}  // namespace detail
}  // namespace ep
}  // namespace onnxruntime

//
// EP specific using declarations
//

#define EP_SPECIFIC_USING_DECLARATIONS                                             \
  using FuncManager = onnxruntime::ep::detail::FuncManager;                        \
  using KernelCreatePtrFn = onnxruntime::ep::detail::KernelCreatePtrFn;            \
  using KernelDefBuilder = onnxruntime::ep::detail::KernelDefBuilder;              \
  using KernelRegistry = onnxruntime::ep::detail::KernelRegistry;                  \
  using KernelCreateInfo = onnxruntime::ep::detail::KernelCreateInfo;              \
  using BuildKernelCreateInfoFn = onnxruntime::ep::detail::KernelCreateInfo (*)(); \
  using OpKernelInfo = onnxruntime::ep::detail::OpKernelInfo;                      \
  using OpKernelContext = onnxruntime::ep::detail::OpKernelContext;                \
  using OpKernel = onnxruntime::ep::detail::OpKernel;

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
