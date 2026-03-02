// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>

#include "kernel_def_builder.h"
#include "op_kernel_info.h"
#include "op_kernel.h"

#include "core/graph/basic_types.h"
#include "core/framework/error_code_helper.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

struct FuncManager {};
using KernelCreatePtrFn = std::add_pointer<Status(FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out)>::type;

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::KernelCreateInfo`.
/// </summary>
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

/// <summary>
/// An adapter class partially implementing the interface of `onnxruntime::KernelRegistry`.
/// </summary>
struct KernelRegistry {
  KernelRegistry() = default;

  static OrtStatus* CreateKernel(void* kernel_create_func_state, const OrtKernelInfo* info, OrtKernelImpl** out) noexcept {
    try {
      FuncManager func_mgr;  // not used
      std::unique_ptr<OpKernel> kernel;
      KernelCreatePtrFn create_func = reinterpret_cast<KernelCreatePtrFn>(kernel_create_func_state);
      Status status = create_func(func_mgr, OpKernelInfo(info), kernel);
      if (!status.IsOK()) {
        return ToOrtStatus(status);
      }
      *out = nullptr;

      // Try to create a control flow kernel implementation if applicable.
      // For kernel based plugin EPs, the implementation should create the control flow kernel directly using one of the
      // following APIs:
      // - `OrtEpApi::CreateIfKernel`
      // - `OrtEpApi::CreateLoopKernel`
      // - `OrtEpApi::CreateScanKernel`
      //
      // If the kernel being created is one of the control flow kernels, `CreateControlFlowKernelImpl` should be overriden
      // to write the value of `out` to the created `OrtKernelImpl`, and the returned status should be OK.
      status = kernel->CreateControlFlowKernelImpl(info, out);
      if (!status.IsOK()) {
        return ToOrtStatus(status);
      }
      if (*out == nullptr) {
        // If the kernel is not a control flow kernel, create a regular kernel implementation.
        *out = new KernelImpl(std::move(kernel));
      }
      return nullptr;
    } catch (const Ort::Exception& ex) {
      Ort::Status ort_status(ex);
      return ort_status.release();
    } catch (const std::exception& ex) {
      Ort::Status ort_status(ex.what(), ORT_EP_FAIL);
      return ort_status.release();
    } catch (...) {
      Ort::Status ort_status("Unknown exception in CreateKernel", ORT_EP_FAIL);
      return ort_status.release();
    }
  }

  Status Register(KernelCreateInfo&& create_info) {
    registry_.AddKernel(create_info.kernel_def,
                        KernelRegistry::CreateKernel,
                        reinterpret_cast<void*>(create_info.kernel_create_func));
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

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
