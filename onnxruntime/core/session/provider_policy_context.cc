// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/provider_policy_context.h"

#include <algorithm>
#include <memory>

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/ep_factory_internal.h"
#include "core/session/ep_plugin_provider_interfaces.h"
#include "core/session/inference_session.h"
#include "core/session/inference_session_utils.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
namespace {
bool MatchesEpVendor(const OrtEpDevice* d) {
  // TODO: Would be better to match on Id. Should the EP add that in EP metadata?
  return d->device->vendor == d->ep_vendor;
}

bool IsDiscreteDevice(const OrtEpDevice* d) {
  if (d->device->type != OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
    return false;
  }

  const auto& entries = d->device->metadata.Entries();
  if (auto it = entries.find("Discrete"); it != entries.end()) {
    return it->second == "1";
  }

  return false;
}

bool IsDefaultCpuEp(const OrtEpDevice* d) {
  return d->device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU &&
         d->ep_vendor == "Microsoft";
}

// Sort devices. NPU -> GPU -> CPU
// Within in type, vendor owned, not.
// Default CPU EP is last
std::vector<const OrtEpDevice*> OrderDevices(const std::vector<const OrtEpDevice*>& devices) {
  std::vector<const OrtEpDevice*> sorted_devices(devices.begin(), devices.end());
  std::sort(sorted_devices.begin(), sorted_devices.end(), [](const OrtEpDevice* a, const OrtEpDevice* b) {
    auto aDeviceType = a->device->type;
    auto bDeviceType = b->device->type;
    if (aDeviceType != bDeviceType) {
      // NPU -> GPU -> CPU
      // std::sort is ascending order, so NPU < GPU < CPU

      // one is an NPU
      if (aDeviceType == OrtHardwareDeviceType::OrtHardwareDeviceType_NPU) {
        return true;
      } else if (bDeviceType == OrtHardwareDeviceType::OrtHardwareDeviceType_NPU) {
        return false;
      }

      if (aDeviceType == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
        return true;
      } else if (bDeviceType == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
        return false;
      }

      // this shouldn't be reachable as it would imply both are CPU
      ORT_THROW("Unexpected combination of devices");
    }

    // both devices are the same

    if (aDeviceType == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      bool aDiscrete = IsDiscreteDevice(a);
      bool bDiscrete = IsDiscreteDevice(b);
      if (aDiscrete != bDiscrete) {
        return aDiscrete == true;  // prefer discrete
      }

      // both discrete or both integrated
    }

    // prefer device matching platform vendor
    bool aVendor = MatchesEpVendor(a);
    bool bVendor = MatchesEpVendor(b);
    if (aVendor != bVendor) {
      return aVendor == true;  // prefer the device that matches the EP vendor
    }

    // default CPU EP last
    bool aIsDefaultCpuEp = IsDefaultCpuEp(a);
    bool bIsDefaultCpuEp = IsDefaultCpuEp(b);
    if (!aIsDefaultCpuEp && !bIsDefaultCpuEp) {
      // neither are default CPU EP. both do/don't match vendor.
      // TODO: implement tie-breaker for this scenario. arbitrarily sort by ep name
      return a->ep_name < b->ep_name;
    }

    // one is the default CPU EP
    return aIsDefaultCpuEp == false;  // prefer the one that is not the default CPU EP
  });

  return sorted_devices;
}

OrtKeyValuePairs GetModelMetadata(const InferenceSession& session) {
  OrtKeyValuePairs metadata;
  auto status_and_metadata = session.GetModelMetadata();

  if (!status_and_metadata.first.IsOK()) {
    return metadata;
  }

  // use field names from onnx.proto
  const auto& model_metadata = *status_and_metadata.second;
  metadata.Add("producer_name", model_metadata.producer_name);
  metadata.Add("producer_version", model_metadata.producer_version);
  metadata.Add("domain", model_metadata.domain);
  metadata.Add("model_version", std::to_string(model_metadata.version));
  metadata.Add("doc_string", model_metadata.description);
  metadata.Add("graph_name", model_metadata.graph_name);                // name from main GraphProto
  metadata.Add("graph_description", model_metadata.graph_description);  // descriptions from main GraphProto
  for (const auto& entry : model_metadata.custom_metadata_map) {
    metadata.Add(entry.first, entry.second);
  }

  return metadata;
}
}  // namespace

// Select execution providers based on the device policy and available devices and add to session
Status ProviderPolicyContext::SelectEpsForSession(const Environment& env, const OrtSessionOptions& options,
                                                  InferenceSession& sess) {
  // Get the list of devices from the environment and order them.
  // Ordered by preference within each type. NPU -> GPU -> NPU
  // TODO: Should environment.cc do the ordering?
  std::vector<const OrtEpDevice*> execution_devices = OrderDevices(env.GetOrtEpDevices());

  // The list of devices selected by policies
  std::vector<const OrtEpDevice*> devices_selected;

  // Run the delegate if it was passed in lieu of any other policy
  if (options.value.ep_selection_policy.delegate) {
    auto model_metadata = GetModelMetadata(sess);
    OrtKeyValuePairs runtime_metadata;  // TODO: where should this come from?

    std::vector<const OrtEpDevice*> delegate_devices(execution_devices.begin(), execution_devices.end());
    std::array<const OrtEpDevice*, 8> selected_devices{nullptr};
    size_t num_selected = 0;

    EpSelectionDelegate delegate = options.value.ep_selection_policy.delegate;
    auto* status = delegate(delegate_devices.data(), delegate_devices.size(),
                            &model_metadata, &runtime_metadata,
                            selected_devices.data(), selected_devices.size(), &num_selected,
                            options.value.ep_selection_policy.state);

    // return or fall-through for both these cases
    // going with explicit failure for now so it's obvious to user what is happening
    if (status != nullptr) {
      std::string delegate_error_msg = OrtApis::GetErrorMessage(status);  // copy
      OrtApis::ReleaseStatus(status);

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "EP selection delegate failed: ", delegate_error_msg);
    }

    if (num_selected == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "EP selection delegate did not select anything.");
    }

    if (num_selected > selected_devices.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "EP selection delegate selected too many EP devices (", num_selected, "). ",
                             "The limit is ", selected_devices.size(), " EP devices.");
    }

    // Copy the selected devices to the output vector
    devices_selected.reserve(num_selected);
    for (size_t i = 0; i < num_selected; ++i) {
      devices_selected.push_back(selected_devices[i]);
    }
  } else {
    // Create the selector for the chosen policy
    std::unique_ptr<IEpPolicySelector> selector;
    switch (options.value.ep_selection_policy.policy) {
      case OrtExecutionProviderDevicePolicy_DEFAULT:
        selector = std::make_unique<DefaultEpPolicy>();
        break;
      case OrtExecutionProviderDevicePolicy_PREFER_CPU:
        selector = std::make_unique<PreferCpuEpPolicy>();
        break;
      case OrtExecutionProviderDevicePolicy_PREFER_NPU:
      case OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY:
      case OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER:
        selector = std::make_unique<PreferNpuEpPolicy>();
        break;
      case OrtExecutionProviderDevicePolicy_PREFER_GPU:
      case OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE:
        selector = std::make_unique<PreferGpuEpPolicy>();
        break;
    }

    // Execute policy

    selector->SelectProvidersForDevices(execution_devices, devices_selected);
  }

  // Fail if we did not find any device matches
  if (devices_selected.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "No execution providers selected. Please check the device policy and available devices.");
  }

  // Log telemetry for auto EP selection
  {
    std::vector<std::string> requested_ep_ids;
    requested_ep_ids.reserve(devices_selected.size());

    for (const auto* device : devices_selected) {
      if (device != nullptr) {
        requested_ep_ids.push_back(device->ep_name);
      }
    }

    // Extract available execution provider IDs
    std::vector<std::string> available_ep_ids;
    available_ep_ids.reserve(execution_devices.size());
    for (const auto* device : execution_devices) {
      available_ep_ids.push_back(device->ep_name);
    }

    std::string policy_type;
    if (options.value.ep_selection_policy.delegate) {
      policy_type = "custom_delegate";
    } else {
      switch (options.value.ep_selection_policy.policy) {
        case OrtExecutionProviderDevicePolicy_DEFAULT:
          policy_type = "DEFAULT";
          break;
        case OrtExecutionProviderDevicePolicy_PREFER_CPU:
          policy_type = "PREFER_CPU";
          break;
        case OrtExecutionProviderDevicePolicy_PREFER_NPU:
          policy_type = "PREFER_NPU";
          break;
        case OrtExecutionProviderDevicePolicy_PREFER_GPU:
          policy_type = "PREFER_GPU";
          break;
        case OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE:
          policy_type = "MAX_PERFORMANCE";
          break;
        case OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY:
          policy_type = "MAX_EFFICIENCY";
          break;
        case OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER:
          policy_type = "MIN_OVERALL_POWER";
          break;
        default:
          policy_type = "UNKNOWN";
          break;
      }
    }

    const Env& os_env = Env::Default();
    os_env.GetTelemetryProvider().LogAutoEpSelection(
        sess.GetCurrentSessionId(),
        policy_type,
        requested_ep_ids,
        available_ep_ids);
  }

  // Configure the session options for the devices. This updates the SessionOptions in the InferenceSession with any
  // EP options that have not been overridden by the user.
  ORT_RETURN_IF_ERROR(AddEpDefaultOptionsToSession(sess, devices_selected));

  // Create OrtSessionOptions for the CreateEp call.
  // Once the InferenceSession is created, its SessionOptions is the source of truth and contains all the values from
  // the user provided OrtSessionOptions. We do a copy for simplicity. The OrtSessionOptions instance goes away
  // once we exit this function so an EP implementation should not use OrtSessionOptions after it returns from
  // CreateEp.
  auto& session_options = sess.GetMutableSessionOptions();
  OrtSessionOptions ort_so;
  ort_so.value = session_options;
  const auto& session_logger = sess.GetLogger();
  const OrtLogger& api_session_logger = *session_logger->ToExternal();

  // Remove the ORT CPU EP if configured to do so
  bool disable_ort_cpu_ep = ort_so.value.config_options.GetConfigEntry(kOrtSessionOptionsDisableCPUEPFallback) == "1";
  if (disable_ort_cpu_ep) {
    RemoveOrtCpuDevice(devices_selected);
  }

  // Fold the EPs into a single structure per factory
  std::vector<SelectionInfo> eps_selected;
  FoldSelectedDevices(devices_selected, eps_selected);

  // Iterate through the selected EPs and create them
  for (size_t idx = 0; idx < eps_selected.size(); ++idx) {
    std::unique_ptr<IExecutionProvider> ep = nullptr;
    ORT_RETURN_IF_ERROR(CreateExecutionProvider(env, ort_so, api_session_logger, eps_selected[idx], ep));
    if (ep != nullptr) {
      ORT_RETURN_IF_ERROR(sess.RegisterExecutionProvider(std::move(ep)));
    }
  }

  return Status::OK();
}

void ProviderPolicyContext::FoldSelectedDevices(std::vector<const OrtEpDevice*> devices_selected,
                                                std::vector<SelectionInfo>& eps_selected) {
  while (devices_selected.size() > 0) {
    const auto ep_name = std::string(devices_selected[0]->ep_name);
    SelectionInfo info;
    info.ep_factory = devices_selected[0]->ep_factory;

    do {
      auto iter = std::find_if(devices_selected.begin(), devices_selected.end(), [&ep_name](const OrtEpDevice* d) {
        return d->ep_name == ep_name;
      });

      if (iter != devices_selected.end()) {
        info.devices.push_back(*iter);
        // hardware device and metadata come from the OrtEpDevice but we need a collection of just the pointers
        // to pass through to the CreateEp call. other info in the OrtEpDevice is used on the ORT side like the
        // allocator and data transfer setup.
        info.hardware_devices.push_back((*iter)->device);
        info.ep_metadata.push_back(&(*iter)->ep_metadata);
        devices_selected.erase(iter);
      } else {
        break;
      }
    } while (true);

    eps_selected.push_back(info);
  }
}

Status ProviderPolicyContext::CreateExecutionProvider(const Environment& env, OrtSessionOptions& options,
                                                      const OrtLogger& logger,
                                                      SelectionInfo& info, std::unique_ptr<IExecutionProvider>& ep) {
  EpFactoryInternal* internal_factory = env.GetEpFactoryInternal(info.ep_factory);

  if (internal_factory) {
    // this is a factory we created and registered internally for internal and provider bridge EPs
    ORT_RETURN_IF_ERROR(ToStatusAndRelease(
        internal_factory->CreateIExecutionProvider(info.hardware_devices.data(), info.ep_metadata.data(),
                                                   info.hardware_devices.size(), &options, &logger,
                                                   &ep)));
  } else {
    OrtEp* api_ep = nullptr;
    ORT_RETURN_IF_ERROR(ToStatusAndRelease(
        info.ep_factory->CreateEp(info.ep_factory, info.hardware_devices.data(), info.ep_metadata.data(),
                                  info.hardware_devices.size(), &options, &logger, &api_ep)));
    ep = std::make_unique<PluginExecutionProvider>(UniqueOrtEp(api_ep, OrtEpDeleter(*info.ep_factory)), options,
                                                   *info.ep_factory, info.devices, *logger.ToInternal());
  }

  return Status::OK();
}

Status ProviderPolicyContext::AddEpDefaultOptionsToSession(InferenceSession& sess,
                                                           std::vector<const OrtEpDevice*> devices) {
  auto& config_options = sess.GetMutableSessionOptions().config_options;
  for (auto device : devices) {
    const std::string ep_options_prefix = OrtSessionOptions::GetProviderOptionPrefix(device->ep_name.c_str());
    for (const auto& [key, value] : device->ep_options.Entries()) {
      const std::string option_key = ep_options_prefix + key;
      // preserve user-provided options as they override any defaults the EP factory specified earlier
      if (config_options.configurations.find(option_key) == config_options.configurations.end()) {
        // use AddConfigEntry for the error checking it does
        ORT_RETURN_IF_ERROR(config_options.AddConfigEntry(option_key.c_str(), value.c_str()));
      }
    }
  }

  return Status::OK();
}

void ProviderPolicyContext::RemoveOrtCpuDevice(std::vector<const OrtEpDevice*>& devices) {
  // Remove the Microsoft CPU EP. always last if available.
  if (IsDefaultCpuEp(devices.back())) {
    devices.pop_back();
  }
}

void DefaultEpPolicy::SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                                std::vector<const OrtEpDevice*>& selected) {
  // Default policy is prefer CPU
  PreferCpuEpPolicy().SelectProvidersForDevices(sorted_devices, selected);
}

void PreferCpuEpPolicy::SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                                  std::vector<const OrtEpDevice*>& selected) {
  // Select the first CPU device from sorted devices
  auto first_cpu = std::find_if(sorted_devices.begin(), sorted_devices.end(),
                                [](const OrtEpDevice* device) {
                                  return device->device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU;
                                });

  ORT_ENFORCE(first_cpu != sorted_devices.end(), "No CPU based execution providers were found.");
  selected.push_back(*first_cpu);

  // add ORT CPU EP as the final option to ensure maximum coverage of opsets and operators
  if (!IsDefaultCpuEp(*first_cpu) && IsDefaultCpuEp(sorted_devices.back())) {
    selected.push_back(sorted_devices.back());
  }
}

void PreferNpuEpPolicy::SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                                  std::vector<const OrtEpDevice*>& selected) {
  // Select the first NPU if there is one.
  if (sorted_devices.front()->device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_NPU) {
    selected.push_back(sorted_devices.front());
  }

  // CPU fallback
  PreferCpuEpPolicy().SelectProvidersForDevices(sorted_devices, selected);
}

void PreferGpuEpPolicy::SelectProvidersForDevices(const std::vector<const OrtEpDevice*>& sorted_devices,
                                                  std::vector<const OrtEpDevice*>& selected) {
  // Select the first GPU device
  auto first_gpu = std::find_if(sorted_devices.begin(), sorted_devices.end(),
                                [](const OrtEpDevice* device) {
                                  return device->device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU;
                                });

  if (first_gpu != sorted_devices.end()) {
    selected.push_back(*first_gpu);
  }

  // Add a CPU fallback
  PreferCpuEpPolicy().SelectProvidersForDevices(sorted_devices, selected);
}
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
