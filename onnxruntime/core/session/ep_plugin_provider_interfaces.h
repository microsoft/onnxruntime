// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/framework/plugin_data_transfer.h"
#include "core/providers/providers.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
struct EpNode;
struct EpValueInfo;

/// <summary>
/// IExecutionProviderFactory that wraps a OrtEpFactory. Required for SessionOptionsAppendExecutionProvider_V2.
/// </summary>
struct PluginExecutionProviderFactory : public IExecutionProviderFactory {
 public:
  PluginExecutionProviderFactory(OrtEpFactory& ep_factory, gsl::span<const OrtEpDevice* const> ep_devices);

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ORT_NOT_IMPLEMENTED("CreateProvider without parameters is not supported.");
  }

 private:
  OrtEpFactory& ep_factory_;
  std::vector<const OrtEpDevice*> devices_;
  std::vector<const OrtHardwareDevice*> hardware_devices_;
  std::vector<const OrtKeyValuePairs*> ep_metadata_;
  std::vector<const OrtMemoryInfo*> ep_allocator_mem_infos_;
};

/// <summary>
/// Functor that deletes an instance of OrtEp. Used to create an std::unique_ptr<OrtEp, OrtEpDeleter>.
/// </summary>
struct OrtEpDeleter {
  explicit OrtEpDeleter(OrtEpFactory& ort_ep_factory) : ort_ep_factory_(ort_ep_factory) {}
  void operator()(OrtEp* ort_ep) {
    ort_ep_factory_.ReleaseEp(&ort_ep_factory_, ort_ep);
  }
  OrtEpFactory& ort_ep_factory_;
};

/// <summary>
/// Type that represents a std::unique_ptr for an instance of OrtEp.
/// </summary>
using UniqueOrtEp = std::unique_ptr<OrtEp, OrtEpDeleter>;

/// <summary>
/// IExecutionProvider that wraps an instance of OrtEp.
/// </summary>
class PluginExecutionProvider : public IExecutionProvider {
 public:
  explicit PluginExecutionProvider(UniqueOrtEp ep, std::vector<const OrtMemoryInfo*>& allocator_mem_infos);
  ~PluginExecutionProvider();

  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;

  // create per-session allocators.
  std::vector<AllocatorPtr> CreatePreferredAllocators() override;

 private:
  UniqueOrtEp plugin_ep_;

  // store the individual OrtMemoryInfo instances so we can create the specific allocators required
  std::vector<const OrtMemoryInfo*> allocator_mem_infos_;
};
}  // namespace onnxruntime
