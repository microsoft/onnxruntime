// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/session/model_package/model_package_context.h"
#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/session/model_package/model_package_descriptor_parser.h"

namespace onnxruntime {
namespace {

std::string ToLower(std::string_view s) {
  std::string result(s);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

std::string DeviceTypeToString(const OrtHardwareDevice* hd) {
  if (hd == nullptr) {
    return {};
  }

  switch (hd->type) {
    case OrtHardwareDeviceType::OrtHardwareDeviceType_CPU:
      return "cpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_GPU:
      return "gpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_NPU:
      return "npu";
    default:
      return {};
  }
}

bool MatchesDevice(const OrtHardwareDevice* hd, std::string_view value) {
  if (value.empty() || hd == nullptr) {
    return value.empty();
  }

  return DeviceTypeToString(hd) == ToLower(value);
}

const OrtHardwareDevice* FindMatchingHardwareDevice(std::string_view device_constraint,
                                                    gsl::span<const OrtHardwareDevice* const> hardware_devices) {
  if (device_constraint.empty()) {
    return nullptr;
  }

  for (const auto* hd : hardware_devices) {
    if (MatchesDevice(hd, device_constraint)) {
      return hd;
    }
  }

  return nullptr;
}

const OrtEpDevice* FindEpDeviceForHardwareDevice(const VariantSelectionEpInfo& ep_info,
                                                 const OrtHardwareDevice* hardware_device) {
  if (hardware_device == nullptr) {
    return ep_info.ep_devices.size() == 1 ? ep_info.ep_devices.front() : nullptr;
  }

  for (const auto* ep_device : ep_info.ep_devices) {
    if (ep_device != nullptr && ep_device->device == hardware_device) {
      return ep_device;
    }
  }

  return ep_info.ep_devices.size() == 1 ? ep_info.ep_devices.front() : nullptr;
}

Status ValidateCompiledModelCompatibilityInfo(const VariantSelectionEpInfo& ep_info,
                                              const std::string& compatibility_info,
                                              std::vector<const OrtHardwareDevice*>& constraint_devices,
                                              OrtCompiledModelCompatibility* compiled_model_compatibility) {
  if (compatibility_info.empty()) {
    LOGS_DEFAULT(INFO) << "No compatibility info constraint for this variant. Skip compatibility validation.";
    return Status::OK();
  }

  auto* ep_factory = ep_info.ep_factory;

  if (ep_factory &&
      ep_factory->ort_version_supported >= 23 &&
      ep_factory->ValidateCompiledModelCompatibilityInfo != nullptr) {
    auto status = ep_factory->ValidateCompiledModelCompatibilityInfo(ep_factory,
                                                                     constraint_devices.data(),
                                                                     constraint_devices.size(),
                                                                     compatibility_info.c_str(),
                                                                     compiled_model_compatibility);
    ORT_RETURN_IF_ERROR(ToStatusAndRelease(status));
  }

  return Status::OK();
}

bool MatchesVariant(ModelVariantInfo& variant,
                    const VariantSelectionEpInfo& ep_info,
                    const OrtHardwareDevice** matched_hardware_device,
                    const OrtEpDevice** matched_ep_device) {
  if (matched_hardware_device != nullptr) {
    *matched_hardware_device = nullptr;
  }
  if (matched_ep_device != nullptr) {
    *matched_ep_device = nullptr;
  }

  LOGS_DEFAULT(INFO) << "Checking component '" << variant.component_model_name
                     << "' package variant '" << variant.package_variant_id
                     << "' with EP constraint '" << variant.ep
                     << "', device constraint '" << variant.device
                     << "' and compatibility info constraint '" << variant.compatibility_info
                     << "' against EP '" << ep_info.ep_name << "' with supported devices: "
                     << [&ep_info]() {
                          std::string devices_str;
                          for (const auto* hd : ep_info.hardware_devices) {
                            if (!devices_str.empty()) {
                              devices_str += ", ";
                            }

                            devices_str += hd->vendor + " " + std::to_string(hd->device_id);
                          }

                          return devices_str.empty() ? "none" : devices_str;
                        }();

  if (!variant.ep.empty() && variant.ep != ep_info.ep_name) {
    LOGS_DEFAULT(INFO) << "Variant EP constraint '" << variant.ep << "' does not match EP name '"
                       << ep_info.ep_name << "'. Skip this variant.";
    return false;
  }

  bool device_ok = variant.device.empty();
  std::vector<const OrtHardwareDevice*> constraint_devices = ep_info.hardware_devices;

  if (ep_info.hardware_devices.empty()) {
    device_ok = true;
  }

  if (!device_ok) {
    if (const auto* matched = FindMatchingHardwareDevice(variant.device, ep_info.hardware_devices)) {
      device_ok = true;
      constraint_devices = {matched};

      if (matched_hardware_device != nullptr) {
        *matched_hardware_device = matched;
      }
      if (matched_ep_device != nullptr) {
        *matched_ep_device = FindEpDeviceForHardwareDevice(ep_info, matched);
      }
    }
  } else {
    if (!ep_info.hardware_devices.empty() && matched_hardware_device != nullptr) {
      *matched_hardware_device = ep_info.hardware_devices.front();
    }
    if (matched_ep_device != nullptr) {
      *matched_ep_device = FindEpDeviceForHardwareDevice(ep_info, matched_hardware_device != nullptr ? *matched_hardware_device : nullptr);
    }
  }

  if (!device_ok) {
    LOGS_DEFAULT(INFO) << "Variant device constraint '" << variant.device
                       << "' does not match any device supported by EP '"
                       << ep_info.ep_name << "'. Skip this variant.";
    return false;
  }

  auto status = ValidateCompiledModelCompatibilityInfo(ep_info, variant.compatibility_info,
                                                       constraint_devices, &variant.compiled_model_compatibility);
  if (!status.IsOK()) {
    LOGS_DEFAULT(WARNING) << "Failed to validate compatibility info for variant with EP constraint '"
                          << variant.ep << "': " << status.ToString()
                          << ". Simply skip this compatibility validation.";

    variant.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  }

  if (variant.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
    LOGS_DEFAULT(INFO) << "Variant compatibility info indicates unsupported model for EP '" << ep_info.ep_name
                       << "'. Compatibility info: '" << variant.compatibility_info
                       << "'. Skip this variant.";
    return false;
  }

  LOGS_DEFAULT(INFO) << "This model variant is selected and could be used for EP '" << ep_info.ep_name << "'.";
  return true;
}

ProviderOptions MergeProviderOptions(const ModelVariantInfo& variant, const VariantSelectionEpInfo& ep_info) {
  ProviderOptions merged = variant.provider_options;
  for (const auto& [key, value] : ep_info.ep_options) {
    merged[key] = value;
  }

  return merged;
}

}  // namespace

int ModelPackageResolver::CalculateVariantScore(const ModelVariantInfo& variant) const {
  int score = 0;

  if (variant.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) {
    score += 100;
  } else if (variant.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION) {
    score += 50;
  }

  if (!variant.ep.empty()) {
    score += 10;
  }

  if (!variant.device.empty()) {
    score += 5;
  }

  return score;
}

Status ModelPackageResolver::Resolve(const ModelPackageContext& context,
                                     gsl::span<VariantSelectionEpInfo> ep_infos,
                                     std::optional<ResolvedModelPackageInfo>& resolved_package) const {
  resolved_package.reset();

  const auto& variants = context.GetModelVariantInfos();
  if (variants.empty() || ep_infos.empty()) {
    return Status::OK();
  }

  std::map<std::string, std::vector<const ModelVariantInfo*>> variants_by_package_id;
  for (const auto& variant : variants) {
    const std::string package_variant_id = variant.package_variant_id.empty() ? variant.variant_name : variant.package_variant_id;
    variants_by_package_id[package_variant_id].push_back(&variant);
  }

  ResolvedModelPackageInfo best_package{};
  bool best_package_found = false;

  for (const auto& [package_variant_id, grouped_variants] : variants_by_package_id) {
    ResolvedModelPackageInfo current_package{};
    current_package.package_variant_id = package_variant_id;
    current_package.score = 0;

    bool package_variant_valid = true;

    for (const auto& component_model_name : context.GetComponentModelNames()) {
      bool component_found = false;
      int best_component_score = std::numeric_limits<int>::min();
      ResolvedModelComponentInfo best_component{};

      for (const auto* grouped_variant : grouped_variants) {
        if (grouped_variant == nullptr || grouped_variant->component_model_name != component_model_name) {
          continue;
        }

        for (const auto& ep_info : ep_infos) {
          ModelVariantInfo candidate = *grouped_variant;
          const OrtHardwareDevice* matched_hardware_device = nullptr;
          const OrtEpDevice* matched_ep_device = nullptr;

          if (!MatchesVariant(candidate, ep_info, &matched_hardware_device, &matched_ep_device)) {
            continue;
          }

          const int candidate_score = CalculateVariantScore(candidate);
          if (!component_found || candidate_score > best_component_score) {
            component_found = true;
            best_component_score = candidate_score;

            best_component.component_model_name = candidate.component_model_name;
            best_component.variant_name = candidate.variant_name;
            best_component.package_variant_id = candidate.package_variant_id;
            best_component.ep_name = candidate.ep.empty() ? ep_info.ep_name : candidate.ep;
            best_component.device = candidate.device.empty() ? DeviceTypeToString(matched_hardware_device) : candidate.device;
            best_component.architecture = candidate.architecture;
            best_component.compatibility_info = candidate.compatibility_info;
            best_component.provider_options = MergeProviderOptions(candidate, ep_info);
            best_component.session_config_entries = candidate.session_config_entries;
            best_component.compiled_model_compatibility = candidate.compiled_model_compatibility;
            best_component.model_path = candidate.model_path;
            best_component.hardware_device = matched_hardware_device;
            best_component.ep_device = matched_ep_device;
          }
        }
      }

      if (!component_found) {
        package_variant_valid = false;
        break;
      }

      current_package.score += best_component_score;
      current_package.resolved_components.push_back(std::move(best_component));
    }

    if (!package_variant_valid) {
      continue;
    }

    if (!best_package_found || current_package.score > best_package.score) {
      best_package = std::move(current_package);
      best_package_found = true;
    }
  }

  if (best_package_found) {
    resolved_package = std::move(best_package);
  }

  return Status::OK();
}

Status ModelVariantSelector::SelectVariant(const ModelPackageContext& context,
                                           gsl::span<VariantSelectionEpInfo> ep_infos,
                                           std::optional<std::filesystem::path>& selected_variant_path) const {
  selected_variant_path.reset();

  std::optional<ResolvedModelPackageInfo> resolved_package;
  ModelPackageResolver resolver;
  ORT_RETURN_IF_ERROR(resolver.Resolve(context, ep_infos, resolved_package));

  if (!resolved_package.has_value() || resolved_package->resolved_components.empty()) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(resolved_package->resolved_components.size() == 1,
                    "Model package resolved to ", resolved_package->resolved_components.size(),
                    " component models. Direct session creation only supports model packages that resolve to a single component model.");

  selected_variant_path = resolved_package->resolved_components.front().model_path;
  return Status::OK();
}

ModelPackageContext::ModelPackageContext(const std::filesystem::path& package_root) {
  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  ORT_THROW_IF_ERROR(parser.ParseVariantsFromRoot(package_root, model_variant_infos_, &component_model_names_));
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
