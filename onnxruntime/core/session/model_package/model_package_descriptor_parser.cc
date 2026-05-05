// # Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <optional>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/session/model_package/model_package_descriptor_parser.h"

namespace onnxruntime {
namespace {

struct VariantConstraintsSchema {
  std::optional<std::string> ep;
  std::optional<std::string> device;
  std::optional<std::string> architecture;
  std::optional<std::string> ep_compatibility_info;
};

struct VariantSchema {
  std::optional<std::string> model_type;
  std::optional<std::string> model_file;
  std::optional<std::string> model_id;
  std::optional<VariantConstraintsSchema> constraints;
};

struct ManifestSchema {
  std::string model_name;
  std::optional<std::string> model_version;
  std::optional<std::vector<std::string>> component_models;
};

struct MetadataSchema {
  std::optional<std::string> component_model_name;
  std::unordered_map<std::string, VariantSchema> model_variants;
};

void from_json(const json& j, VariantConstraintsSchema& c) {
  if (j.contains(kEpKey) && j[kEpKey].is_string()) c.ep = j[kEpKey].get<std::string>();
  if (j.contains(kDeviceKey) && j[kDeviceKey].is_string()) c.device = j[kDeviceKey].get<std::string>();
  if (j.contains(kArchitectureKey) && j[kArchitectureKey].is_string()) c.architecture = j[kArchitectureKey].get<std::string>();
  if (j.contains(kEpCompatibilityInfoKey) && j[kEpCompatibilityInfoKey].is_string()) {
    c.ep_compatibility_info = j[kEpCompatibilityInfoKey].get<std::string>();
  }
}

void from_json(const json& j, VariantSchema& v) {
  if (j.contains(kModelTypeKey) && j[kModelTypeKey].is_string()) v.model_type = j[kModelTypeKey].get<std::string>();
  if (j.contains(kModelFileKey) && j[kModelFileKey].is_string()) v.model_file = j[kModelFileKey].get<std::string>();
  if (j.contains(kModelIdKey) && j[kModelIdKey].is_string()) v.model_id = j[kModelIdKey].get<std::string>();
  if (j.contains(kConstraintsKey) && j[kConstraintsKey].is_object()) {
    v.constraints = j[kConstraintsKey].get<VariantConstraintsSchema>();
  }
}

void from_json(const json& j, ManifestSchema& m) {
  m.model_name = j.at(kModelNameKey).get<std::string>();  // required

  if (j.contains(kModelVersionKey) && j[kModelVersionKey].is_string()) {
    m.model_version = j[kModelVersionKey].get<std::string>();
  }

  if (j.contains(kComponentModelsKey)) {
    if (!j[kComponentModelsKey].is_array()) {
      throw std::invalid_argument("\"component_models\" must be an array of strings");
    }
    m.component_models = j[kComponentModelsKey].get<std::vector<std::string>>();
  }
}

void from_json(const json& j, MetadataSchema& m) {
  if (j.contains("component_model_name") && j["component_model_name"].is_string()) {
    m.component_model_name = j["component_model_name"].get<std::string>();
  }
  m.model_variants = j.at(kModelVariantsKey).get<std::unordered_map<std::string, VariantSchema>>();  // required
}

Status FindSingleOnnxFile(const std::filesystem::path& search_dir,
                          std::filesystem::path& resolved_path) {
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

void ApplyVariantConstraints(const VariantConstraintsSchema& c, ModelVariantInfo& variant) {
  if (c.ep.has_value()) variant.ep = *c.ep;
  if (c.device.has_value()) variant.device = *c.device;
  if (c.architecture.has_value()) variant.architecture = *c.architecture;
  if (c.ep_compatibility_info.has_value()) variant.compatibility_info = *c.ep_compatibility_info;
}

std::string SanitizeModelIdForPath(std::string model_id) {
  for (char& ch : model_id) {
    switch (ch) {
      case '<':
      case '>':
      case ':':
      case '"':
      case '/':
      case '\\':
      case '|':
      case '?':
      case '*':
        ch = '_';
        break;
      default:
        break;
    }
  }
  return model_id;
}

}  // namespace

// The package_root could be either a component model root (contains metadata.json) or a model package root (contains
// manifest.json). The parsing logic will first check for metadata.json to see if it's a component model root, and if
// not found, it will look for manifest.json to treat it as a model package root.
//
// This function parses information from manifest.json and/or metadata.json for the model variants from the same
// component model, producing a unified list of ModelVariantInfo.
//
// Note: If the package_root is a model package root, currently it only supports one component model.
// #TODO: In the future we may want ORT to support running multiple component models in a single "virtual" session.
//
// A manifest.json may look like this:
//
// {
//     "model_name" : <logical_model_name>,
//     "model_version": "1.0",
//     "component_models" : [<component_model_name_1>,
//                           <component_model_name_2>]
// }
//
// A metadata.json for the component model may look like this:
//
// {
//    "component_model_name" : <component_model_name>,
//    "model_variants" : {
//        <variant_name_1> : {
//            "model_type": "onnx",
//            "model_file" : <ep_context_model_1 onnx file>,
//            "constraints" : {
//                "ep" : <ep_name>,
//                "device" : <device_type>,
//                "ep_compatibility_info" : <ep_compatibility_info_1>
//            }
//        },
//        <variant_name_2> : {
//             "model_type": "onnx",
//             "model_file" : <ep_context_model_2 onnx file>,
//             "constraints" : {
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

    MetadataSchema metadata_schema;
    ORT_TRY {
      metadata_schema = metadata_doc.get<MetadataSchema>();
    }
    ORT_CATCH(const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "metadata.json at ", component_metadata_path.string(),
                             " has invalid schema: ", ex.what());
    }

    const std::string component_model_name =
        metadata_schema.component_model_name.has_value()
            ? *metadata_schema.component_model_name
            : package_root.filename().string();

    // component-root mode
    ORT_RETURN_IF_ERROR(ParseVariantsFromComponent(component_model_name,
                                                   package_root,
                                                   &metadata_doc[kModelVariantsKey],
                                                   variants));

    for (const auto& v : variants) {
      LOGS(logger_, INFO) << "model variant: file='" << v.model_path.string()
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

  ManifestSchema manifest_schema;
  ORT_TRY {
    manifest_schema = doc.get<ManifestSchema>();
  }
  ORT_CATCH(const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json has invalid schema: ", ex.what());
  }

  if (manifest_schema.model_name.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "The \"model_name\" field in the manifest.json is missing or empty");
  }

  // Locate component models.
  const bool has_component_models = manifest_schema.component_models.has_value();
  std::vector<std::string> component_model_names;
  std::unordered_map<std::string, json> discovered_metadata_docs;

  if (has_component_models) {
    component_model_names = *manifest_schema.component_models;
  } else {
    // Fallback: discover component models under package_root/models where a metadata.json exists.
    const auto models_dir = package_root / "models";
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "manifest.json missing \"component_models\" and no discoverable models directory at ",
                             models_dir.string());
    }

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
                               "metadata.json at ", metadata_path.string(),
                               " is not valid JSON: ", ex.what());
      }

      ORT_TRY {
        (void)metadata_doc.get<MetadataSchema>();
      }
      ORT_CATCH(const std::exception& ex) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "metadata.json at ", metadata_path.string(),
                               " has invalid schema: ", ex.what());
      }

      discovered_metadata_docs.emplace(component_model_name, std::move(metadata_doc));
      component_model_names.push_back(component_model_name);
    }

    if (component_model_names.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "manifest.json missing \"component_models\" and no component model folders with metadata.json were found under ",
                             models_dir.string());
    }
  }

  if (component_model_names.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json should contain exactly one component model name in \"component_models\" array "
                           "or the models directory should contain exactly one component model.");
  }

  for (const auto& component_model_name : component_model_names) {
    const auto component_model_root = package_root / "models" / component_model_name;

    // If component model is listed in manifest.json but corresponding directory is missing, warn and skip.
    if (has_component_models &&
        (!std::filesystem::exists(component_model_root) || !std::filesystem::is_directory(component_model_root))) {
      LOGS(logger_, WARNING) << "Component model '" << component_model_name
                             << "' is listed in manifest.json but directory does not exist: "
                             << component_model_root.string()
                             << ". Skipping this component model.";
      continue;
    }

    // Load metadata.json (if present) for this component model.
    json metadata_doc;
    const json* metadata_variants_obj = nullptr;
    const auto metadata_path = component_model_root / kComponentModelMetadataFileName;

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
          (void)metadata_doc.get<MetadataSchema>();  // typed schema validation
          metadata_variants_obj = &metadata_doc[kModelVariantsKey];
        }
        ORT_CATCH(const std::exception&) {
          // Ignore metadata parse/schema errors; fall back to metadata-absent handling below.
        }
      }
    }

    ORT_RETURN_IF_ERROR(ParseVariantsFromComponent(component_model_name,
                                                   component_model_root,
                                                   metadata_variants_obj,
                                                   variants));
  }

  if (variants.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "No valid component models were found under ", (package_root / "models").string());
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
    const json* metadata_variants_obj,
    /*in,out*/ std::vector<ModelVariantInfo>& variants) const {
  if (metadata_variants_obj == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Missing metadata model variants for component model: ",
                           component_model_name);
  }

  std::unordered_map<std::string, VariantSchema> metadata_variants;
  ORT_TRY {
    metadata_variants = metadata_variants_obj->get<std::unordered_map<std::string, VariantSchema>>();
  }
  ORT_CATCH(const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Invalid metadata variant schema for component model '",
                           component_model_name, "': ", ex.what());
  }

  for (const auto& kv : metadata_variants) {
    const std::string& variant_name = kv.first;
    const VariantSchema& variant_schema = kv.second;

    ModelVariantInfo variant{};
    std::filesystem::path model_dir = component_model_root / variant_name;

    if (variant_schema.model_id.has_value() && !variant_schema.model_id->empty()) {
      model_dir = component_model_root / SanitizeModelIdForPath(*variant_schema.model_id);
    }

    std::filesystem::path resolved_model_path;

    if (variant_schema.model_file.has_value()) {
      const std::filesystem::path candidate_path = model_dir / *variant_schema.model_file;

      if (!std::filesystem::exists(candidate_path)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Variant '", variant_name, "' file path does not exist: ",
                               candidate_path.string());
      }

      if (std::filesystem::is_regular_file(candidate_path)) {
        resolved_model_path = candidate_path;
      } else if (std::filesystem::is_directory(candidate_path)) {
        ORT_RETURN_IF_ERROR(FindSingleOnnxFile(candidate_path, resolved_model_path));
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Variant '", variant_name,
                               "' file path is neither a file nor a directory: ",
                               candidate_path.string());
      }
    } else {
      ORT_RETURN_IF_ERROR(FindSingleOnnxFile(model_dir, resolved_model_path));
    }

    variant.model_path = std::move(resolved_model_path);

    if (variant_schema.constraints.has_value()) {
      ApplyVariantConstraints(*variant_schema.constraints, variant);
    }

    variants.push_back(std::move(variant));
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)