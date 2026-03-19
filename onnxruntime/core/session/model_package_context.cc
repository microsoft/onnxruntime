// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <fstream>
#include <algorithm>
#include <string>
#include <cctype>

#include "core/session/model_package_context.h"

namespace onnxruntime {
namespace {
std::string ToLower(std::string_view s) {
  std::string result(s);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

bool MatchesDevice(const OrtHardwareDevice* hd,
                           std::string_view value) {
  if (value.empty() || hd == nullptr) {
    return value.empty();
  }

  std::string device_type = ToLower(value);
  switch (hd->type) {
    case OrtHardwareDeviceType::OrtHardwareDeviceType_CPU:
      return device_type == "cpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_GPU:
      return device_type == "gpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_NPU:
      return device_type == "npu";
    default:
      return false;
  }

  return false;
}

bool MatchesArchitecture(const OrtHardwareDevice* hd,
                         std::string_view value) {
  if (value.empty() || hd == nullptr) {
    return value.empty();
  }

  // For architecture, we check against a specific key in hardware device metadata 
  // since there's no standard key for it in OrtHardwareDevice. Skipping this check for now.

  return true;
}

bool MatchesProviderOptionsSpecificKeyForDeviceType(std::string_view provider_option_key, 
                                                    std::string_view provider_option_value,
                                                    std::string_view value) {
  // Currently, the provider option key is not standardized.
  // So we hardcode well-known keys for device matching for different EPs.
  // TODO: In the future, we may want to standardize some common keys like "device_type" and only check those.

  if (ToLower(provider_option_key) == "device_type") {
    return ToLower(provider_option_value) == ToLower(value);
  } else if (ToLower(provider_option_key) == "backend_type") {
    return ToLower(provider_option_value) == ToLower(value);
  }

  return false;
}

bool MatchesComponent(const EpContextVariantInfo& component, const SelectionEpInfo& ep_info) {
  // EP constraint
  if (!component.ep.empty() && component.ep != ep_info.ep_name) {
    return false;
  }

  // Device constraint
  // - Check "device" field against EP's hardware devices' OrtHardwareDeviceType
  // - Check "device" field against EP provider options. (currently the provider options key is not standardized).
  bool device_ok = component.device.empty();
  if (!device_ok) {
    for (const auto* hd : ep_info.hardware_devices) {
      if (MatchesDevice(hd, component.device)) {
        device_ok = true;
        break;
      }
    }
  }

  if (!device_ok) {
    return false;
  }

  device_ok = component.device.empty();
  for (const auto& [key, value] : ep_info.ep_options) {
    if (MatchesProviderOptionsSpecificKeyForDeviceType(key, value, component.device)) {
      device_ok = true;
      break;
    }
  }

  if (!device_ok) {
    return false;
  }

  // Architecture constraint
  bool arch_ok = component.architecture.empty();
  if (!arch_ok) {
    for (const auto* hd : ep_info.hardware_devices) {
      if (MatchesArchitecture(hd, component.architecture)) {
        arch_ok = true;
        break;
      }
    }
  }

  return arch_ok;
}

}  // namespace

Status ModelPackageManifestParser::ParseManifest(const std::filesystem::path& package_root,
                                                   /*out*/ std::vector<EpContextVariantInfo>& components) {
  components.clear();
  const auto manifest_path = package_root / kModelPackageManifestFileName;
  if (!std::filesystem::exists(manifest_path)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "No manifest.json found at ", manifest_path.string());
  }

  std::ifstream f(manifest_path, std::ios::binary);
  if (!f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to open manifest.json at ", manifest_path.string());
  }

  ORT_TRY {
    json doc = json::parse(f);
    if (!doc.is_object() || !doc.contains(kComponentsKey) || !doc[kComponentsKey].is_array()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "The \"components\" field in the manifest.json is missing or not an array");
    }

    for (const auto& comp : doc[kComponentsKey]) {
      if (!comp.is_object() || !comp.contains(kVariantNameKey) || !comp[kVariantNameKey].is_string()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "The \"variant_name\" field in a component is missing or not a string");
      }

      if (!comp.is_object() || !comp.contains(kFileKey) || !comp[kFileKey].is_string()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "The \"file\" field in a component is missing or not a string");
      }

      EpContextVariantInfo c;

      // variant name
      std::string variant_name = comp[kVariantNameKey].get<std::string>();
      c.metadata.Add(kVariantNameKey, variant_name);

      // Build model path: package_root / "models" / variant_name / file
      std::filesystem::path model_dir = package_root / "models" / variant_name;
      c.model_path = model_dir / comp[kFileKey].get<std::string>();

      if (comp.contains(kConstraintsKey) && comp[kConstraintsKey].is_object()) {
        const auto& cons = comp[kConstraintsKey];
        if (cons.contains(kEpKey) && cons[kEpKey].is_string()) c.ep = cons[kEpKey].get<std::string>();
        if (cons.contains(kDeviceKey) && cons[kDeviceKey].is_string()) c.device = cons[kDeviceKey].get<std::string>();
        if (cons.contains(kArchitectureKey) && cons[kArchitectureKey].is_string()) {
          c.architecture = cons[kArchitectureKey].get<std::string>();
        }
      }

      components.push_back(std::move(c));
    }
  }
  ORT_CATCH(const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json is not valid JSON: ", ex.what());
  }

  for (const auto& c : components) {
    LOGS(logger_, INFO) << "manifest component: file='" << c.model_path.string()
                        << "' ep='" << c.ep << "' device='" << c.device
                        << "' arch='" << c.architecture << "'";
  }

  return Status::OK();
}

Status ModelPackageContext::SelectComponent(const Environment& env,
                                            gsl::span<EpContextVariantInfo> components,
                                            gsl::span<SelectionEpInfo> ep_infos,
                                            std::optional<std::filesystem::path>& selected_component_path) {
  ORT_UNUSED_PARAMETER(env);

  selected_component_path = std::nullopt;

  if (components.empty()) {
    return Status::OK();
  }

  // 1) Check a component with no constraints.
  auto it_no_constraints = std::find_if(
      components.begin(), components.end(),
      [](const EpContextVariantInfo& c) {
        return c.ep.empty() && c.device.empty() && c.architecture.empty();
      });

  if (it_no_constraints != components.end()) {
    selected_component_path = it_no_constraints->model_path;
    return Status::OK();
  }

  // 2) Try to find the first component that matches the available EP/device selection.
  for (const auto& c : components) {
    for (const auto& ep_info : ep_infos) {
      if (MatchesComponent(c, ep_info)) {
        selected_component_path = c.model_path;
        return Status::OK();
      }
    }
  }

  // 3) Fallback to the first component.
  selected_component_path = components.front().model_path;
  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
