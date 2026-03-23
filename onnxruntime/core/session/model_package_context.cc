// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/session/model_package_context.h"

namespace onnxruntime {
namespace {
std::string ToLower(std::string_view s) {
  std::string result(s);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

bool MatchesDevice(const OrtHardwareDevice* hd, std::string_view value) {
  if (value.empty() || hd == nullptr) {
    return value.empty();
  }

  const std::string device_type = ToLower(value);
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
}

bool MatchesArchitecture(const OrtHardwareDevice* hd, std::string_view value) {
  if (value.empty() || hd == nullptr) {
    return value.empty();
  }

  // No standardized architecture key today. Assume match if provided.
  return true;
}

bool MatchesProviderOptionsSpecificKeyForDeviceType(std::string_view provider_option_key,
                                                    std::string_view provider_option_value,
                                                    std::string_view value) {
  const auto key_lower = ToLower(provider_option_key);

  // If provider option key is related to device type, then its value must match the device constraint value.
  if (key_lower == "device_type" || key_lower == "backend_type") {
    return ToLower(provider_option_value) == ToLower(value);
  }

  return true;
}

Status ValidateCompiledModelCompatibilityInfo(const SelectionEpInfo& ep_info,
                                              const std::string& compatibility_info,
                                              OrtCompiledModelCompatibility* compiled_model_compatibility) {
  if (compatibility_info.empty()) {
    return Status::OK();
  }

  auto* ep_factory = ep_info.ep_factory;

  if (ep_factory &&
      ep_factory->ort_version_supported >= 23 &&
      ep_factory->ValidateCompiledModelCompatibilityInfo != nullptr) {
    auto status = ep_factory->ValidateCompiledModelCompatibilityInfo(ep_factory,
                                                                     ep_info.hardware_devices.data(),
                                                                     ep_info.hardware_devices.size(),
                                                                     compatibility_info.c_str(),
                                                                     compiled_model_compatibility);
    ORT_RETURN_IF_ERROR(ToStatusAndRelease(status));
  }

  return Status::OK();
}

bool MatchesComponent(EpContextVariantInfo& component, const SelectionEpInfo& ep_info) {
  // Check EP constraint
  if (!component.ep.empty() && component.ep != ep_info.ep_name) {
    return false;
  }

  // Check device constraints
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

  // If provider options contain keys related to device type, then the value must match the device constraint value.
  for (const auto& [key, value] : ep_info.ep_options) {
    if (!MatchesProviderOptionsSpecificKeyForDeviceType(key, value, component.device)) {
      device_ok = false;
      break;
    }
  }

  if (!device_ok) {
    return false;
  }

  // Check compatibility info constraint and save the compatibility result for later use if needed.
  auto status = ValidateCompiledModelCompatibilityInfo(ep_info, component.compatibility_info,
                                                       &component.compiled_model_compatibility);
  if (!status.IsOK()) {
    LOGS_DEFAULT(WARNING) << "Failed to validate compatibility info for component with EP constraint '"
                          << component.ep << "': " << status.ToString()
                          << ". Simply skip this compatibility validation.";

    component.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  }

  if (component.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
    return false;
  }

  // Check hardware architecture constraint
  bool arch_ok = component.architecture.empty();
  if (!arch_ok) {
    for (const auto* hd : ep_info.hardware_devices) {
      if (MatchesArchitecture(hd, component.architecture)) {
        arch_ok = true;
        break;
      }
    }
  }

  if (!arch_ok) {
    return false;
  }

  return true;
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
    if (!doc.contains(kModelNameKey) || !doc[kModelNameKey].is_string()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "The \"name\" field in the manifest.json is missing or not a string");
    }
    const std::string model_name = doc[kModelNameKey].get<std::string>();

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

      // Build model path: package_root / "models" / model_name / variant_name / file
      std::filesystem::path model_dir = package_root / "models" / model_name / variant_name;
      c.model_path = model_dir / comp[kFileKey].get<std::string>();

      if (comp.contains(kConstraintsKey) && comp[kConstraintsKey].is_object()) {
        const auto& cons = comp[kConstraintsKey];
        if (cons.contains(kEpKey) && cons[kEpKey].is_string()) c.ep = cons[kEpKey].get<std::string>();
        if (cons.contains(kDeviceKey) && cons[kDeviceKey].is_string()) c.device = cons[kDeviceKey].get<std::string>();
        if (cons.contains(kArchitectureKey) && cons[kArchitectureKey].is_string()) {
          c.architecture = cons[kArchitectureKey].get<std::string>();
        }
        if (cons.contains(kEpCompatibilityInfoKey) && cons[kEpCompatibilityInfoKey].is_string()) {
          c.compatibility_info = cons[kEpCompatibilityInfoKey].get<std::string>();
        }
        if (cons.contains(kSdkVersionKey) && cons[kSdkVersionKey].is_string()) {
          c.metadata[kSdkVersionKey] = cons[kSdkVersionKey].get<std::string>();
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

// Calculate a score for the component based on its constraints and metadata.
//
// It's only used to choose the best component among multiple candidates that match constraints.
// Higher score means more preferred.
//
// For example:
// If one component/EPContext is compatible with the EP and has compatiliby value indicating optimal compatibility
// (e.g. compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) while another component/EPContext
// is also compatible with the EP but has compatibility value indicating prefer recompilation
// (e.g. compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION),
// the former will have a higher score and thus be selected.
//
int ModelPackageContext::CalculateComponentScore(const EpContextVariantInfo& component) const {
  int score = 0;

  if (component.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) {
    score += 100;
  } else if (component.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION) {
    score += 50;
  }

  return score;
}

Status ModelPackageContext::SelectComponent(gsl::span<EpContextVariantInfo> components,
                                            gsl::span<SelectionEpInfo> ep_infos,
                                            std::optional<std::filesystem::path>& selected_component_path) const {
  selected_component_path.reset();

  if (components.empty()) {
    return Status::OK();
  }

  // For simplicity, there is a constraint in this initial implementation:
  // - Only one SelectionEpInfo in `ep_infos` is supported

  std::unordered_set<size_t> candidate_indices_set;

  // 1) Check unconstrained components (ep/device/arch all empty).
  for (size_t i = 0, end = components.size(); i < end; ++i) {
    const auto& c = components[i];
    if (c.ep.empty() && c.device.empty() && c.architecture.empty()) {
      candidate_indices_set.insert(i);
    }
  }

  // 2) Check all components that match any EP/device selection.
  for (size_t i = 0, end = components.size(); i < end; ++i) {
    if (candidate_indices_set.count(i) > 0) {
      continue;
    }
    auto& c = components[i];
    for (const auto& ep_info : ep_infos) {
      if (MatchesComponent(c, ep_info)) {
        candidate_indices_set.insert(i);
        break;
      }
    }
  }

  // Log all matched candidates.
  {
    std::ostringstream oss;
    oss << candidate_indices_set.size() << " Component(s) matched manifest constraints: ";
    size_t i = 0;
    for (size_t idx : candidate_indices_set) {
      const auto& path = components[idx].model_path;
      oss << path.string();
      if (i + 1 < candidate_indices_set.size()) {
        oss << "; ";
      }
      ++i;
    }
    LOGS_DEFAULT(INFO) << oss.str();
  }

  if (candidate_indices_set.empty()) {
    return Status::OK();
  }

  // 3) If only one candidate, select it.
  if (candidate_indices_set.size() == 1) {
    selected_component_path = components[*candidate_indices_set.begin()].model_path;
    return Status::OK();
  }

  // 4) If there are multiple candidates, choose the highest-score component among them.
  int best_score = std::numeric_limits<int>::min();
  size_t best_index = *candidate_indices_set.begin();

  for (size_t idx : candidate_indices_set) {
    const auto& c = components[idx];
    int component_best_score = std::numeric_limits<int>::min();
    component_best_score = std::max(component_best_score, CalculateComponentScore(c));

    if (component_best_score > best_score) {
      best_score = component_best_score;
      best_index = idx;
    }
  }

  selected_component_path = components[best_index].model_path;

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
