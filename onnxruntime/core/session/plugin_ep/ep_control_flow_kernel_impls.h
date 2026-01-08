// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/scan.h"

struct OrtScanKernelConfig {
  OrtScanTransposeFunc transpose_func = nullptr;
  void* transpose_func_state = nullptr;

  OrtScanZeroDataFunc zero_data_func = nullptr;
  void* zero_data_func_state = nullptr;
};

struct OrtLoopKernelConfig {
  OrtLoopConcatOutputFunc concat_output_func = nullptr;
  void* concat_output_func_state = nullptr;
};

namespace onnxruntime {

struct PluginEpControlFlowKernelImpl : public OrtKernelImpl {
  PluginEpControlFlowKernelImpl();
  virtual controlflow::IControlFlowKernel& GetIControlFlowKernel() = 0;
};

class PluginEpIfKernelImpl final : public PluginEpControlFlowKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static Status Create(const OpKernelInfo& info, /*out*/ std::unique_ptr<PluginEpIfKernelImpl>& out);

  // Note: Must use ::Create() to create an instance.
  PluginEpIfKernelImpl(const OpKernelInfo& info, PrivateTag);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  If kernel_;
};

class PluginEpLoopKernelImpl final : public PluginEpControlFlowKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static Status Create(const OpKernelInfo& info, const OrtLoopKernelConfig& config,
                       /*out*/ std::unique_ptr<PluginEpLoopKernelImpl>& out);

  // Note: Must use ::Create() to create an instance.
  PluginEpLoopKernelImpl(const OpKernelInfo& info, Loop::ConcatOutput concat_func, PrivateTag);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Loop kernel_;
};

template <int OpSet>
class PluginEpScanKernelImpl final : public PluginEpControlFlowKernelImpl {
 private:
  struct PrivateTag {};

 public:
  static Status Create(const OpKernelInfo& info, const OrtScanKernelConfig& config,
                       /*out*/ std::unique_ptr<PluginEpScanKernelImpl<OpSet>>& out);

  // Note: Must use ::Create() to create an instance.
  PluginEpScanKernelImpl(const OpKernelInfo& info, const scan::detail::DeviceHelpers& device_helpers, PrivateTag);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Scan<OpSet> kernel_;
};

}  // namespace onnxruntime
