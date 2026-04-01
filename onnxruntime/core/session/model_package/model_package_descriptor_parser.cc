// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/session/model_package/model_package_descriptor_parser.h"

namespace onnxruntime {

// The package_root could be either a component model root (contains metadata.json) or a model package root (contains
// manifest.json). The parsing logic will first check for metadata.json to see if it's a component model root, and if
// not found, it will look for manifest.json to treat it as a model package root. This allows flexibility in how the
// model package is structured.
//
// This function parses information from manifest.json and/or metadata.json for the model variants from the same
// component model, producing a unified list of ModelVariantInfo.
//
// If a model variant appears in both, it chooses component model's metadata.json as the source of truth, but
// falls back to manifest.json if metadata.json is missing required fields.
//
// Note: If the package_root is a model package root, currently it only supports one component model.
// #TODO: In the future we may want ORT to support running multiple component models in a single "virtual" session.
//
// A manifest.json may look like this:
//
// {
//     "model_name" : <logical_model_name>,
//     "component_models" : { // optional, if missing, ORT will discover component models by looking for folders with
//                            // metadata.json under model_package_root/models
//         <model_name_1> : {
//            ...             // Could be empty.
//         }
//     }
// }
//
// or
//
// {
//     "model_name" : <logical_model_name>,
//     "component_models" : {
//         <model_name_1> : {
//             "model_variants" : {
//                 <variant_name_1> : {
//                     "model_type": "onnx",
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
//    "component_model_name" : <component_model_name>,
//    "model_variants" : {
//        <variant_name_1> : {
//            "file" : <ep_context_model_1 onnx file>,
//            "constraints" : {
//                "model_type": "onnx",
//                "ep" : <ep_name>,
//                "device" : <device_type>,
//                "ep_compatibility_info" : <ep_compatibility_info_1>
//            }
//        },
//        <variant_name_2> : {
//             "file" : <ep_context_model_2 onnx file>,
//             "constraints" : {
//                 "model_type": "onnx",
//                 "ep" : <ep_name>,
//                 "device" : <device_type>,
//                 "ep_compatibility_info" : <ep_compatibility_info_1>
//             }
//         }
//     }
// }
Status ModelPackageDescriptorParser::ParseVariantsFromRoot(const std::filesystem::path& package_root,
                                                           /*out*/ std::vector<ModelVariantInfo>& variants) const {
  variants.clear();

  // package_root could be either a component model root (contains metadata.json) or a model package root (contains manifest.json).

  // Mode 1: package_root is a component model root (contains metadata.json).
  // In this mode metadata.json is the source of truth and manifest.json is ignored.
  const auto component_metadata_path = package_root / kComponentModelMetadataFileName;
  if (std::filesystem::exists(component_metadata_path) &&
      std::filesystem::is_regular_file(component_metadata_path)) {
    std::ifstream mf(component_metadata_path, std::ios::binary);
    if (!mf) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to open metadata.json at ", component_metadata_path.string());
    }

    json metadata_doc;
    ORT_TRY {
      metadata_doc = json::parse(mf);
    }
    ORT_CATCH(const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "metadata.json at ", component_metadata_path.string(),
                             " is not valid JSON: ", ex.what());
    }

    if (!metadata_doc.contains(kModelVariantsKey) || !metadata_doc[kModelVariantsKey].is_object()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "metadata.json at ", component_metadata_path.string(),
                             " must contain a \"model_variants\" object");
    }

    const std::string component_model_name =
        (metadata_doc.contains("component_model_name") && metadata_doc["component_model_name"].is_string())
            ? metadata_doc["component_model_name"].get<std::string>()
            : package_root.filename().string();

    ORT_RETURN_IF_ERROR(ParseVariantsFromComponent(component_model_name,
                                                   package_root,
                                                   nullptr,
                                                   &metadata_doc[kModelVariantsKey],
                                                   variants));

    for (const auto& v : variants) {
      LOGS(logger_, INFO) << "manifest variant: file='" << v.model_path.string()
                          << "' ep='" << v.ep << "' device='" << v.device
                          << "' arch='" << v.architecture << "'";
    }

    return Status::OK();
  }

  // Mode 2: package_root is a model package root, resolve via manifest.json.
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

  // Locate component models.
  const bool has_component_models = doc.contains(kComponentModelsKey) && doc[kComponentModelsKey].is_object();
  json components;
  std::unordered_map<std::string, json> discovered_metadata_docs;

  if (has_component_models) {
    components = doc[kComponentModelsKey];
  } else {
    // Fallback: discover component models under package_root/models where a metadata.json exists.
    const auto models_dir = package_root / "models";
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "manifest.json missing \"component_models\" and no discoverable models directory at ",
                             models_dir.string());
    }

    json discovered_components = json::object();
    for (const auto& entry : std::filesystem::directory_iterator(models_dir)) {
      if (!entry.is_directory()) {
        continue;
      }

      const auto component_model_name = entry.path().filename().string();
      const auto metadata_path = entry.path() / kComponentModelMetadataFileName;
      if (!std::filesystem::exists(metadata_path)) {
        continue;  // only folders with metadata.json count as component models
      }

      std::ifstream mf(metadata_path, std::ios::binary);
      if (!mf) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Failed to open metadata.json at ", metadata_path.string());
      }

      json metadata_doc;
      ORT_TRY {
        metadata_doc = json::parse(mf);
      }
      ORT_CATCH(const std::exception& ex) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "metadata.json at ", metadata_path.string(), " is not valid JSON: ", ex.what());
      }

      if (!metadata_doc.contains(kModelVariantsKey) || !metadata_doc[kModelVariantsKey].is_object()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "metadata.json at ", metadata_path.string(),
                               " must contain a \"model_variants\" object");
      }

      // Remember the metadata for later reuse and record this as a discovered component.
      discovered_metadata_docs.emplace(component_model_name, std::move(metadata_doc));
      discovered_components[component_model_name] = json::object();
    }

    if (discovered_components.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "manifest.json missing \"component_models\" and no component model folders with metadata.json were found under ",
                             models_dir.string());
    }

    components = std::move(discovered_components);
  }

  const auto& components_ref = components;

  if (components_ref.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json should contain exactly one model component field or "
                           "the models directory should contain exactly one component model.");
  }

  for (const auto& item : components_ref.items()) {
    const std::string& component_model_name = item.key();
    const auto& component_obj = item.value();

    // Load metadata.json (if present) for this component model.
    json metadata_doc;
    const json* metadata_variants_obj = nullptr;
    const auto metadata_path = package_root / "models" / component_model_name / kComponentModelMetadataFileName;

    if (!has_component_models) {
      auto it_meta = discovered_metadata_docs.find(component_model_name);
      if (it_meta != discovered_metadata_docs.end()) {
        metadata_doc = it_meta->second;
        metadata_variants_obj = &metadata_doc[kModelVariantsKey];
      }
    } else if (std::filesystem::exists(metadata_path)) {
      std::ifstream mf(metadata_path, std::ios::binary);
      if (mf) {
        ORT_TRY {
          metadata_doc = json::parse(mf);
          if (metadata_doc.contains(kModelVariantsKey) && metadata_doc[kModelVariantsKey].is_object()) {
            metadata_variants_obj = &metadata_doc[kModelVariantsKey];
          }
        }
        ORT_CATCH(const std::exception&) {
          // Ignore metadata parse errors; fall back to manifest-only flow.
        }
      }
    }

    const json* manifest_variants_obj =
        (component_obj.contains(kModelVariantsKey) && component_obj[kModelVariantsKey].is_object())
            ? &component_obj[kModelVariantsKey]
            : nullptr;

    const auto component_model_root = package_root / "models" / component_model_name;
    ORT_RETURN_IF_ERROR(ParseVariantsFromComponent(component_model_name,
                                                   component_model_root,
                                                   manifest_variants_obj,
                                                   metadata_variants_obj,
                                                   variants));
  }

  for (const auto& v : variants) {
    LOGS(logger_, INFO) << "model variant: file='" << v.model_path.string()
                        << "' ep='" << v.ep << "' device='" << v.device
                        << "' arch='" << v.architecture << "'";
  }

  return Status::OK();
}

Status ModelPackageDescriptorParser::ParseVariantsFromComponent(
    const std::string& component_model_name,
    const std::filesystem::path& component_model_root,
    const json* manifest_variants_obj,
    const json* metadata_variants_obj,
    /*in,out*/ std::vector<ModelVariantInfo>& variants) const {
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

    ModelVariantInfo variant;
    const std::filesystem::path model_dir = component_model_root / variant_name;

    auto find_single_onnx = [&](const std::filesystem::path& search_dir,
                                std::filesystem::path& resolved_path) -> Status {
      std::vector<std::filesystem::path> onnx_files;
      for (const auto& entry : std::filesystem::directory_iterator(search_dir)) {
        if (!entry.is_regular_file()) {
          continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".onnx") {
          onnx_files.push_back(entry.path());
        }
      }

      if (onnx_files.empty()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "No ONNX model file found under ", search_dir.string());
      }

      if (onnx_files.size() > 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Multiple ONNX model files found under ", search_dir.string(),
                               ". Multiple ONNX files per variant are not supported yet.");
      }

      resolved_path = onnx_files.front();
      return Status::OK();
    };

    std::filesystem::path resolved_model_path;

    const bool has_file = chosen_variant->contains(kFileKey);
    if (has_file && (*chosen_variant)[kFileKey].is_string()) {
      const auto file_value = (*chosen_variant)[kFileKey].get<std::string>();
      const std::filesystem::path candidate_path = model_dir / file_value;

      if (!std::filesystem::exists(candidate_path)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Variant '", variant_name, "' file path does not exist: ",
                               candidate_path.string());
      }

      if (std::filesystem::is_regular_file(candidate_path)) {
        resolved_model_path = candidate_path;
      } else if (std::filesystem::is_directory(candidate_path)) {
        ORT_RETURN_IF_ERROR(find_single_onnx(candidate_path, resolved_model_path));
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Variant '", variant_name,
                               "' file path is neither a file nor a directory: ",
                               candidate_path.string());
      }
    } else if (has_file && !(*chosen_variant)[kFileKey].is_string()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Variant '", variant_name, "' has a non-string \"file\" field");
    } else {
      // No "file" provided: search the variant directory for a single ONNX file.
      ORT_RETURN_IF_ERROR(find_single_onnx(model_dir, resolved_model_path));
    }

    variant.model_path = resolved_model_path;

    // TODO: Might need to check agasint "model_type" field in the future if we support multiple model formats.
    if (chosen_variant->contains(kConstraintsKey) && (*chosen_variant)[kConstraintsKey].is_object()) {
      ORT_RETURN_IF_ERROR(ParseModelVariantConstraints((*chosen_variant)[kConstraintsKey], variant));
    }

    variants.push_back(std::move(variant));
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

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)