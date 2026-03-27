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
#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package/model_package_descriptor_parser.h"

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

bool MatchesDeviceTypeProviderOption(std::string_view provider_option_key,
                                     std::string_view provider_option_value,
                                     std::string_view device_constraint,
                                     gsl::span<const OrtHardwareDevice* const> hardware_devices,
                                     std::vector<const OrtHardwareDevice*>& constraint_devices) {
  if (device_constraint.empty()) {
    return true;
  }

  const auto key_lower = ToLower(provider_option_key);

  // If provider option key is related to device type, then its value must match the device constraint value.
  //
  // Those keys are not standardized and are defined by EPs, e.g. "device_type" for OpenVINO EP, "backend_type" for QNN EP.
  //   - "backend_type" has valid values: "cpu", "gpu", "htp" and "saver".
  //   - "device_type" has valid values: "CPU", "GPU", "GPU.0", "GPU.1" and "NPU".
  //
  // TODO: In the future, we can consider standardizing the key for device type in provider options and make it more generic for all EPs to use.
  if (key_lower == "device_type" || key_lower == "backend_type") {
    bool match = ToLower(provider_option_value).find(ToLower(device_constraint)) != std::string::npos;

    if (!match) {
      return false;  // provider option value does not match device constraint
    }

    // Get the matched device according to the device constraint.
    const auto* matched_device = FindMatchingHardwareDevice(device_constraint, hardware_devices);
    constraint_devices = {matched_device};
  }

  return true;
}

Status ValidateCompiledModelCompatibilityInfo(const SelectionEpInfo& ep_info,
                                              const std::string& compatibility_info,
                                              std::vector<const OrtHardwareDevice*>& constraint_devices,
                                              OrtCompiledModelCompatibility* compiled_model_compatibility) {
  if (compatibility_info.empty()) {
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

bool MatchesVariant(ModelVariantInfo& variant, const SelectionEpInfo& ep_info) {
  // 1) Check EP constraint
  if (variant.ep.empty() ||
      (!variant.ep.empty() && variant.ep != ep_info.ep_name)) {
    return false;
  }

  // 2) Check device constraint
  bool device_ok = variant.device.empty();

  // For some provider-bridge EPs, they may not implement OrtEpFactory::GetSupportedDevices, therefore ORT
  // won't have the supported device information for those EPs and ep_info.hardware_devices will be empty.
  // In that case, we will skip the device constraint validation for those EPs.
  if (ep_info.hardware_devices.empty()) {
    device_ok = true;
  }

  // The constraint_devices is the target device(s) and will be passed to ValidateCompiledModelCompatibilityInfo
  // for compatibility validation.
  std::vector<const OrtHardwareDevice*> constraint_devices = ep_info.hardware_devices;

  if (!device_ok) {
    if (const auto* matched = FindMatchingHardwareDevice(variant.device, ep_info.hardware_devices)) {
      device_ok = true;
      constraint_devices = {matched};
    }
  }

  if (!device_ok) {
    return false;
  }

  // If provider option contains key related to device type, then the value must match the device constraint if any.
  // Gets the target device if matched.
  for (const auto& [key, value] : ep_info.ep_options) {
    if (!MatchesDeviceTypeProviderOption(key, value, variant.device, ep_info.hardware_devices, constraint_devices)) {
      return false;
    }
  }

  // 3) Check ep_compatibility_info constraint
  //
  // ORT does not directly evaluate the architecture constraint. Instead, it relies on
  // the ep_compatibility_info constraint, which may encode architecture information
  // if needed.
  //
  // The ep_compatibility_info value is expected to match the EP compatibility string
  // stored in the EPContext model metadata.
  // (See OrtEp::GetCompiledModelCompatibilityInfo() for how this string is generated.)
  //
  // The EP implementation of EpFactory::ValidateCompiledModelCompatibilityInfo()
  // is responsible for validating the compatibility string against the target device
  // (i.e. OrtHardwareDevice), and returning the compatibility result.
  auto status = ValidateCompiledModelCompatibilityInfo(ep_info, variant.compatibility_info,
                                                       constraint_devices, &variant.compiled_model_compatibility);
  if (!status.IsOK()) {
    LOGS_DEFAULT(WARNING) << "Failed to validate compatibility info for variant with EP constraint '"
                          << variant.ep << "': " << status.ToString()
                          << ". Simply skip this compatibility validation.";

    variant.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  }

  if (variant.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
    return false;
  }

  return true;
}
}  // namespace

// Calculate a score for the model variant based on its constraints and metadata.
//
// It's only used to choose the best model variant among multiple candidates that match constraints.
// Higher score means more preferred.
//
// For example:
// If one model variant/EPContext is compatible with the EP and has compatiliby value indicating optimal compatibility
// (i.e. compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) while another model variant/EPContext
// is also compatible with the EP but has compatibility value indicating prefer recompilation
// (i.e. compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION),
// the former will have a higher score and thus be selected.
//
int ModelVariantSelector::CalculateVariantScore(const ModelVariantInfo& variant) const {
  int score = 0;

  if (variant.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) {
    score += 100;
  } else if (variant.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION) {
    score += 50;
  }

  return score;
}

Status ModelVariantSelector::SelectVariant(gsl::span<ModelVariantInfo> variants,
                                           gsl::span<SelectionEpInfo> ep_infos,
                                           std::optional<std::filesystem::path>& selected_variant_path) const {
  selected_variant_path.reset();

  if (variants.empty()) {
    return Status::OK();
  }

  // There is a constraint for this initial implementation:
  // - Only one SelectionEpInfo in `ep_infos` is supported

  std::unordered_set<size_t> candidate_indices_set;

  // 1) Check unconstrained variants (ep/device/arch all empty).
  for (size_t i = 0, end = variants.size(); i < end; ++i) {
    const auto& c = variants[i];
    if (c.ep.empty() && c.device.empty() && c.architecture.empty()) {
      candidate_indices_set.insert(i);
    }
  }

  // 2) Check all variants that match any EP/device selection.
  for (size_t i = 0, end = variants.size(); i < end; ++i) {
    if (candidate_indices_set.count(i) > 0) {
      continue;
    }
    auto& c = variants[i];
    for (const auto& ep_info : ep_infos) {
      if (MatchesVariant(c, ep_info)) {
        candidate_indices_set.insert(i);
        break;
      }
    }
  }

  // Log all matched candidates.
  {
    std::ostringstream oss;
    oss << candidate_indices_set.size() << " Model variant(s) matched constraints: ";
    size_t i = 0;
    for (size_t idx : candidate_indices_set) {
      const auto& path = variants[idx].model_path;
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
    selected_variant_path = variants[*candidate_indices_set.begin()].model_path;
    return Status::OK();
  }

  // 4) If there are multiple candidates, choose the highest-score model variant among them.
  int best_score = std::numeric_limits<int>::min();
  size_t best_index = *candidate_indices_set.begin();

  for (size_t idx : candidate_indices_set) {
    const auto& v = variants[idx];
    int variant_best_score = std::numeric_limits<int>::min();
    variant_best_score = std::max(variant_best_score, CalculateVariantScore(v));

    if (variant_best_score > best_score) {
      best_score = variant_best_score;
      best_index = idx;
    }
  }

  selected_variant_path = variants[best_index].model_path;

  return Status::OK();
}

ModelPackageContext::ModelPackageContext(const std::filesystem::path& package_root) {
  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  ORT_THROW_IF_ERROR(parser.ParseManifest(package_root, model_variant_infos_));
}

Status ModelPackageContext::SelectModelVariant(gsl::span<SelectionEpInfo> ep_infos) {
  ModelVariantSelector selector;
  return selector.SelectVariant(model_variant_infos_, ep_infos, selected_model_variant_path_);
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
