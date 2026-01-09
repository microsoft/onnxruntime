// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/scan.h"

namespace onnxruntime {

/// <summary>
/// Base class for ORT-defined OrtKernelImpl classes for control flow operators.
/// Provides polymorphic access to the controlflow::IControlFlowKernel interface, which allows setting up subgraph
/// session state.
/// </summary>
struct PluginEpControlFlowKernelImpl : public OrtKernelImpl {
  PluginEpControlFlowKernelImpl();
  virtual controlflow::IControlFlowKernel& GetIControlFlowKernel() = 0;
};

/// <summary>
/// OrtKernelImpl class for an If kernel. The OrtKernelImpl function calls are forwarded to an internal
/// onnxruntime::If operator kernel instance.
///
/// An EP can create an instance of this class by calling OrtEpApi::CreateIfKernel().
/// </summary>
class PluginEpIfKernelImpl final : public PluginEpControlFlowKernelImpl {
 public:
  PluginEpIfKernelImpl(const OpKernelInfo& info);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  If kernel_;
};

/// <summary>
/// OrtKernelImpl class for a Loop kernel. The OrtKernelImpl function calls are forwarded to an internal
/// onnxruntime::Loop operator kernel instance.
///
/// An EP can create an instance of this class by calling OrtEpApi::CreateLoopKernel().
/// </summary>
class PluginEpLoopKernelImpl final : public PluginEpControlFlowKernelImpl {
 public:
  PluginEpLoopKernelImpl(const OpKernelInfo& info, OrtLoopConcatOutputFunc ort_concat_func,
                         void* ort_concat_func_state);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Loop kernel_;
};

/// <summary>
/// OrtKernelImpl class for a Scan kernel (opset >= 9). The OrtKernelImpl function calls are forwarded to an internal
/// onnxruntime::Scan operator kernel instance.
///
/// An EP can create an instance of this class by calling OrtEpApi::CreateLoopKernel().
/// </summary>
class PluginEpScanKernelImpl final : public PluginEpControlFlowKernelImpl {
 public:
  PluginEpScanKernelImpl(const OpKernelInfo& info, OrtScanTransposeFunc ort_transpose_func,
                         void* ort_transpose_func_state);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Scan<9> kernel_;
};

}  // namespace onnxruntime
