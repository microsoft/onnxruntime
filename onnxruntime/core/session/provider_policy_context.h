// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"  // For OrtExecutionProviderDevicePolicy

namespace onnxruntime {

struct SelectionInfo {
  OrtEpFactory* ep_factory;
  std::vector<const OrtHardwareDevice*> devices;
  std::vector<const OrtKeyValuePairs*> ep_metadata;
};

class IEpPolicySelector {
 public:
  /// <summary>
  /// Select the OrtEpDevice instances to use.
  /// Selection is in priority order. Highest priority first.
  /// </summary>
  /// <param name="devices">Ordered devices.
  /// Type order is NPU -> GPU -> CPU
  /// Within a type: Discrete -> Integrated if GPU, EP vendor matches device vendor, vendor does not match
  /// ORT CPU EP is always last if available.
  /// </param>
  /// <param name="selected_devices"></param>
  virtual void SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                         std::vector<const OrtEpDevice*>& selected_devices) = 0;

  virtual ~IEpPolicySelector() = default;
};

class ProviderPolicyContext {
 public:
  ProviderPolicyContext() = default;

  Status SelectEpsForSession(const Environment& env, const OrtSessionOptions& options, InferenceSession& sess);
  Status AddEpDefaultOptionsToSession(InferenceSession& sess, std::vector<const OrtEpDevice*> devices);
  void RemoveOrtCpuDevice(std::vector<const OrtEpDevice*>& devices);
  Status CreateExecutionProvider(const Environment& env, OrtSessionOptions& options, const OrtLogger& logger,
                                 SelectionInfo& info, std::unique_ptr<IExecutionProvider>& ep);
  void FoldSelectedDevices(std::vector<const OrtEpDevice*> devices_selected,  // copy
                           std::vector<SelectionInfo>& eps_selected);

 private:
};

class DefaultEpPolicy : public IEpPolicySelector {
 public:
  void SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                 std::vector<const OrtEpDevice*>& selected_devices) override;
};

class PreferCpuEpPolicy : public IEpPolicySelector {
 public:
  void SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                 std::vector<const OrtEpDevice*>& selected_devices) override;
};

class PreferNpuEpPolicy : public IEpPolicySelector {
 public:
  void SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                 std::vector<const OrtEpDevice*>& selected_devices) override;
};

class PreferGpuEpPolicy : public IEpPolicySelector {
 public:
  void SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                 std::vector<const OrtEpDevice*>& selected_devices) override;
};

}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
