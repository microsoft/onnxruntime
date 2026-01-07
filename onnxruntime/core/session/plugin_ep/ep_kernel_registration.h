// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/common/inlined_containers_fwd.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/framework/prepacked_weights.h"

#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/loop.h"
#include "core/providers/cpu/controlflow/scan.h"

/// <summary>
/// Implementation of the public C API opaque type OrtSharedPrePackedWeightCache used by plugin EP kernels.
/// This wraps and fills out an instance of onnxruntime::PrePackedWeights via the
/// C API SharedPrePackedWeightCache_StoreWeightData.
/// </summary>
struct OrtSharedPrePackedWeightCache {
  /// <summary>
  /// Constructs an OrtSharedPrePackedWeightCache that will fill out the provided PrePackedWeights object.
  /// </summary>
  /// <param name="container">The PrePackedWeights container to fill out.</param>
  /// <param name="allocator">The allocator that will be used to free buffers set by the call to SetBuffers().</param>
  OrtSharedPrePackedWeightCache(onnxruntime::PrePackedWeights& container, onnxruntime::AllocatorPtr allocator);

  /// <summary>
  /// Sets data buffers for the shared weight. Ownership of the buffers is transferred to this class's contained
  /// PrePackedWeights instance, which will delete the buffers with `this->allocator_`.
  /// The buffer data is required to have been allocated with `this->allocator_`.
  /// Refer to OrtKernelImpl::PrePackWeight and OrtEpApi::SharedPrePackedWeightCache_StoreWeightData.
  /// </summary>
  /// <param name="data_ptrs"></param>
  /// <param name="data_sizes"></param>
  /// <param name="num_buffers"></param>
  void SetBuffers(void** data_ptrs, size_t* data_sizes, size_t num_buffers);

  /// <summary>
  /// Returns true if this instance has any weight buffer data.
  /// </summary>
  /// <returns></returns>
  bool HasData() const noexcept;

  /// <summary>
  /// Releases all buffer data.
  /// Used within OrtEpApi::SharedPrePackedWeightCache_StoreWeightData() if an error occurs and ORT wants to
  /// release all data to allow caller to retain ownership of data.
  /// </summary>
  void ReleaseAllData() noexcept;

 private:
  onnxruntime::PrePackedWeights& container_;
  onnxruntime::AllocatorPtr allocator_;
};

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

struct PluginEpControlFlowKernel : public OrtKernelImpl {
  PluginEpControlFlowKernel() : OrtKernelImpl{} {}
  virtual controlflow::IControlFlowKernel& GetIControlFlowKernel() = 0;
};

class PluginEpIfKernel : public PluginEpControlFlowKernel {
 private:
  struct PrivateTag {};

 public:
  static Status Create(const OpKernelInfo& info, /*out*/ std::unique_ptr<PluginEpIfKernel>& out);

  // Note: Must use ::Create() to create an instance.
  PluginEpIfKernel(const OpKernelInfo& info, PrivateTag);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  If kernel_;
};

class PluginEpLoopKernel : public PluginEpControlFlowKernel {
 private:
  struct PrivateTag {};

 public:
  static Status Create(const OpKernelInfo& info, const OrtLoopKernelConfig& config,
                       /*out*/ std::unique_ptr<PluginEpLoopKernel>& out);

  // Note: Must use ::Create() to create an instance.
  PluginEpLoopKernel(const OpKernelInfo& info, Loop::ConcatOutput concat_func, PrivateTag);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Loop kernel_;
};

template <int OpSet>
class PluginEpScanKernel : public PluginEpControlFlowKernel {
 private:
  struct PrivateTag {};

 public:
  static Status Create(const OpKernelInfo& info, const OrtScanKernelConfig& config,
                       /*out*/ std::unique_ptr<PluginEpScanKernel<OpSet>>& out);

  // Note: Must use ::Create() to create an instance.
  PluginEpScanKernel(const OpKernelInfo& info, const scan::detail::DeviceHelpers& device_helpers, PrivateTag);
  controlflow::IControlFlowKernel& GetIControlFlowKernel() override { return kernel_; }

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  Scan<OpSet> kernel_;
};

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
