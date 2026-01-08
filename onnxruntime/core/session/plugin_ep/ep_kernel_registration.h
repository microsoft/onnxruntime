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
