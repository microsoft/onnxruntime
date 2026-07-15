// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/scan.h"

namespace onnxruntime {
/// <summary>
/// Flags that ORT can set on OrtKernelImpl instances.
/// Note: This enum can be moved to a more central location if/when we add other flags.
///
/// IMPORTANT: When adding a new flag, update kOrtKernelImplFlags_MAX_VALUE.
/// </summary>
enum OrtKernelImplFlags : uint32_t {
  // Denotes a control flow kernel created by ORT (i.e., a PluginEpControlFlowKernelImpl)
  kIsControlFlowKernelImpl = 1 << 0,

  // The largest flag value. Used to validate that flags are within the expected range.
  // Must be updated when a new flag is added.
  kOrtKernelImplFlags_MAX_VALUE = kIsControlFlowKernelImpl
};

/// <summary>
/// Base class for ORT-defined OrtKernelImpl classes for control flow operators.
/// Provides polymorphic access to the controlflow::IControlFlowKernel interface, which allows setting up subgraph
/// session state.
/// </summary>
struct PluginEpControlFlowKernelImpl : public OrtKernelImpl {
  PluginEpControlFlowKernelImpl();
  virtual ~PluginEpControlFlowKernelImpl() {}
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
  PluginEpLoopKernelImpl(const OpKernelInfo& info, gsl::not_null<OrtLoopKernelHelper*> helper);
  ~PluginEpLoopKernelImpl();

  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Loop kernel_;
  gsl::not_null<OrtLoopKernelHelper*> helper_;
};

/// <summary>
/// OrtKernelImpl class for a Scan kernel (opset >= 9). The OrtKernelImpl function calls are forwarded to an internal
/// onnxruntime::Scan operator kernel instance.
///
/// An EP can create an instance of this class by calling OrtEpApi::CreateScanKernel().
/// </summary>
class PluginEpScanKernelImpl final : public PluginEpControlFlowKernelImpl {
 public:
  PluginEpScanKernelImpl(const OpKernelInfo& info, gsl::not_null<OrtScanKernelHelper*> helper);
  ~PluginEpScanKernelImpl();

  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Scan<9> kernel_;
  gsl::not_null<OrtScanKernelHelper*> helper_;
};

}  // namespace onnxruntime
