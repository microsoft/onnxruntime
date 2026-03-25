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
  // Check EP constraint
  if (variant.ep.empty() ||
      (!variant.ep.empty() && variant.ep != ep_info.ep_name)) {
    return false;
  }

  // Check device constraints
  bool device_ok = variant.device.empty();

  // This is the target device list that will be passed to ValidateCompiledModelCompatibilityInfo
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

  // ORT doesn't directly check architecture constraint, but relies on ep_compatibility_info constraint containing the
  // architecture information if needed.
  //
  // The ep_compatibility_info is expected be the same as ep compatibility string in EPContext model's metadata.
  // (see OrtEp::GetCompiledModelCompatibilityInfo() for how to create compatibility string in model metadata)
  //
  // EP implements EpFactory::ValidateCompiledModelCompatibilityInfo() should validate the compatibility string with given
  // the target device and OrtHardwareDevice
  // return the compatibility result.
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

// This function parses information from manifest.json and metadata.json for all component models as well as
// their associated model variants, producing a unified list of EpContextVariantInfo.
//
// If a model variant appears in both, it chooses component model's metadata.json as the source of truth, but
// falls back to manifest.json if metadata.json is missing required fields.
//
// Note: In this initial implementation, we expect only one component model existing in the package, in the future
//       we will have the "pipeline" ability to execute multiple component models in sequence to better provide the
//       ease of use for the cases where multiple models are needed (ex: pre/post processing, multi-stage models, etc).
//
// A manifest.json may look like this:
//
// {
//     "name" : <logical_model_name>,
//     "component_models" : {
//         <model_name_1> : {
//         }
//     }
// }
//
// or
//
// {
//     "name" : <logical_model_name>,
//     "component_models" : {
//         <model_name_1> : {
//             "model_variants" : {
//                 <variant_name_1> : {
//                     "file" : <ep_context_model_1 onnx file>,
//                     "constraints" : {
//                         "ep" : <ep_name>,
//                         "device" : <device_type>,
//                         "ep_compatibility_info" : <ep_compatibility_info_1>
//                     }
//                 }
//             }
//         }
//     }
// }
//
// A metadata.json for the component model may look like this:
//
// {
//    "model_name" : <model_name>,
//    "model_variants" : {
//        <variant_name_1> : {
//            "file" : <ep_context_model_1 onnx file>,
//            "constraints" : {
//                "ep" : <ep_name>,
//                "device" : <device_type>,
//                "ep_compatibility_info" : <ep_compatibility_info_1>
//            }
//        },
//        <variant_name_2> : {
//             "file" : <ep_context_model_2 onnx file>,
//             "constraints" : {
//                 "ep" : <ep_name>,
//                 "device" : <device_type>,
//                 "ep_compatibility_info" : <ep_compatibility_info_1>
//             }
//         }
//     }
// }
Status ModelPackageDescriptorParser::ParseManifest(const std::filesystem::path& package_root,
                                                   /*out*/ std::vector<ModelVariantInfo>& variants) const {
  variants.clear();
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

  json doc;
  ORT_TRY {
    doc = json::parse(f);
  }
  ORT_CATCH(const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json is not valid JSON: ", ex.what());
  }

  if (!doc.contains(kModelNameKey) || !doc[kModelNameKey].is_string()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "The \"name\" field in the manifest.json is missing or not a string");
  }

  if (!doc.contains(kComponentModelsKey) || !doc[kComponentModelsKey].is_object()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json must contain a \"component_models\" object");
  }

  const auto& components = doc[kComponentModelsKey];

  if (components.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json should contain exactly one model component field.");
  }

  for (const auto& item : components.items()) {
    const std::string& component_model_name = item.key();
    const auto& component_obj = item.value();

    // Load metadata.json (if present) for this component model
    json metadata_doc;
    const json* metadata_variants_obj = nullptr;
    const auto metadata_path = package_root / "models" / component_model_name / kComponentModelMetadataFileName;
    if (std::filesystem::exists(metadata_path)) {
      std::ifstream mf(metadata_path, std::ios::binary);
      if (mf) {
        ORT_TRY {
          metadata_doc = json::parse(mf);
          if (metadata_doc.contains(kModelVariantsKey) && metadata_doc[kModelVariantsKey].is_object()) {
            metadata_variants_obj = &metadata_doc[kModelVariantsKey];
          }
        }
        ORT_CATCH(const std::exception&) {
          // ignore metadata parse errors; fall back to manifest-only flow
        }
      }
    }

    const json* manifest_variants_obj =
        (component_obj.contains(kModelVariantsKey) && component_obj[kModelVariantsKey].is_object())
            ? &component_obj[kModelVariantsKey]
            : nullptr;

    // Build a combined, deterministic list of variant names:
    //   1) all manifest variants in manifest order
    //   2) any metadata-only variants appended after
    std::vector<std::string> variant_names;
    std::unordered_set<std::string> variant_name_set;
    if (manifest_variants_obj != nullptr) {
      for (const auto& variant_item : manifest_variants_obj->items()) {
        variant_names.push_back(variant_item.key());
        variant_name_set.insert(variant_item.key());
      }
    }
    if (metadata_variants_obj != nullptr) {
      for (const auto& variant_item : metadata_variants_obj->items()) {
        const std::string& variant_name = variant_item.key();
        if (variant_name_set.insert(variant_name).second) {
          variant_names.push_back(variant_name);
        }
      }
    }

    for (const auto& variant_name : variant_names) {
      const json* manifest_variant = nullptr;
      if (manifest_variants_obj != nullptr) {
        auto it = manifest_variants_obj->find(variant_name);
        if (it != manifest_variants_obj->end()) {
          manifest_variant = &it.value();
        }
      }

      const json* metadata_variant = nullptr;
      if (metadata_variants_obj != nullptr) {
        auto it = metadata_variants_obj->find(variant_name);
        if (it != metadata_variants_obj->end()) {
          metadata_variant = &it.value();
        }
      }

      // Pick the variant object (prefer metadata, fall back to manifest).
      const json* chosen_variant = metadata_variant != nullptr ? metadata_variant : manifest_variant;
      if (chosen_variant == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Model variant '", variant_name,
                               "' missing in both manifest and metadata for component model: ",
                               component_model_name);
      }

      // Local helper to parse a single variant for this component.
      auto parse_variant = [&](const std::string& variant_name,
                               const json& variant_json) -> Status {
        ModelVariantInfo variant;
        const std::filesystem::path model_dir =
            package_root / "models" / component_model_name / variant_name;

        if (!variant_json.contains(kFileKey) || !variant_json[kFileKey].is_string()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Variant '", variant_name, "' missing required \"file\" string");
        }

        variant.model_path = model_dir / variant_json[kFileKey].get<std::string>();

        if (!std::filesystem::exists(variant.model_path)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No model file found at ", variant.model_path.string());
        }

        if (variant_json.contains(kConstraintsKey) && variant_json[kConstraintsKey].is_object()) {
          ORT_RETURN_IF_ERROR(ParseModelVariantConstraints(variant_json[kConstraintsKey], variant));
        }

        variants.push_back(std::move(variant));
        return Status::OK();
      };

      ORT_RETURN_IF_ERROR(parse_variant(variant_name, *chosen_variant));
    }
  }

  for (const auto& v : variants) {
    LOGS(logger_, INFO) << "manifest variant: file='" << v.model_path.string()
                        << "' ep='" << v.ep << "' device='" << v.device
                        << "' arch='" << v.architecture << "'";
  }

  return Status::OK();
}

Status ModelPackageDescriptorParser::ParseModelVariantConstraints(const json& constraints, ModelVariantInfo& variant) const {
  if (!constraints.is_object()) {
    return Status::OK();
  }

  if (constraints.contains(kEpKey) && constraints[kEpKey].is_string()) {
    variant.ep = constraints[kEpKey].get<std::string>();
  }
  if (constraints.contains(kDeviceKey) && constraints[kDeviceKey].is_string()) {
    variant.device = constraints[kDeviceKey].get<std::string>();
  }
  if (constraints.contains(kArchitectureKey) && constraints[kArchitectureKey].is_string()) {
    variant.architecture = constraints[kArchitectureKey].get<std::string>();
  }
  if (constraints.contains(kEpCompatibilityInfoKey) && constraints[kEpCompatibilityInfoKey].is_string()) {
    variant.compatibility_info = constraints[kEpCompatibilityInfoKey].get<std::string>();
  }
  if (constraints.contains(kSdkVersionKey) && constraints[kSdkVersionKey].is_string()) {
    variant.metadata[kSdkVersionKey] = constraints[kSdkVersionKey].get<std::string>();
  }

  return Status::OK();
}

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
