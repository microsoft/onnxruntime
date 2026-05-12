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

struct VariantFileSchema {
  std::string filename;  // required
  std::optional<std::unordered_map<std::string, std::string>> session_options;
  std::optional<std::unordered_map<std::string, std::string>> provider_options;
  std::optional<std::unordered_map<std::string, std::string>> shared_files;
};

// Variant metadata schema (mapping from variant.json)
struct VariantMetadataSchema {
  std::vector<VariantFileSchema> files;  // required, non-empty
  std::optional<json> consumer_metadata;
};

struct EpCompatibilitySchema {
  std::optional<std::string> ep;  // nullable in schema
  std::optional<std::string> device;
  std::optional<std::vector<std::string>> compatibility_strings;
};

struct VariantSchema {
  std::vector<EpCompatibilitySchema> ep_compatibility;  // required, non-empty
};

// Component schema (mapping from component metadata.json)
struct ComponentSchema {
  std::optional<std::string> component_model_name;
  std::unordered_map<std::string, VariantSchema> variants;  // required, keys are variant names.
};

// Top-level manifest schema (mapping from manifest.json)
struct ManifestSchema {
  int64_t schema_version;                              // required
  std::optional<std::vector<std::string>> components;  // optional
};

std::string JsonScalarToString(const json& v, const char* key_name, const std::string& parent_key) {
  if (v.is_string()) return v.get<std::string>();
  if (v.is_number_integer()) return std::to_string(v.get<int64_t>());
  if (v.is_number_unsigned()) return std::to_string(v.get<uint64_t>());
  if (v.is_number_float()) return v.dump();
  if (v.is_boolean()) return v.get<bool>() ? "true" : "false";

  throw std::invalid_argument(MakeString("\"", key_name, "\" under '", parent_key,
                                         "' must contain scalar (string/number/bool) values."));
}

std::optional<std::unordered_map<std::string, std::string>> ParseFlatOptionsObject(const json& j,
                                                                                   const char* key_name) {
  if (!j.contains(key_name) || j[key_name].is_null()) {
    return std::nullopt;
  }

  const auto& obj = j[key_name];
  if (!obj.is_object()) {
    throw std::invalid_argument(MakeString("\"", key_name, "\" must be an object."));
  }

  std::unordered_map<std::string, std::string> result;
  result.reserve(obj.size());

  for (auto it = obj.begin(); it != obj.end(); ++it) {
    result.emplace(it.key(), JsonScalarToString(it.value(), key_name, it.key()));
  }

  return result;
}

std::optional<std::vector<std::string>> ParseCompatibilityStrings(const json& j, const char* key_name) {
  if (!j.contains(key_name) || j[key_name].is_null()) {
    return std::nullopt;
  }

  const auto& value = j[key_name];
  if (value.is_string()) {
    return std::vector<std::string>{value.get<std::string>()};  // backward-compatible
  }

  if (!value.is_array()) {
    throw std::invalid_argument(MakeString("\"", key_name, "\" must be a string or an array of strings."));
  }

  std::vector<std::string> result;
  result.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    if (!value[i].is_string()) {
      throw std::invalid_argument(MakeString("\"", key_name, "\" must contain only strings."));
    }
    result.push_back(value[i].get<std::string>());
  }

  return result;
}

void from_json(const json& j, EpCompatibilitySchema& c) {
  if (j.contains(kEpKey) && !j[kEpKey].is_null()) c.ep = j[kEpKey].get<std::string>();
  if (j.contains(kDeviceKey) && j[kDeviceKey].is_string()) c.device = j[kDeviceKey].get<std::string>();
  c.compatibility_strings = ParseCompatibilityStrings(j, kCompatibilityStringKey);
}

void from_json(const json& j, VariantSchema& v) {
  v.ep_compatibility = j.at(kEpCompatibilityKey).get<std::vector<EpCompatibilitySchema>>();
  if (v.ep_compatibility.empty()) {
    throw std::invalid_argument(MakeString("\"", kEpCompatibilityKey, "\" must contain at least one entry"));
  }
}

void from_json(const json& j, VariantFileSchema& f) {
  f.filename = j.at(kFilenameKey).get<std::string>();
  f.session_options = ParseFlatOptionsObject(j, kSessionOptionsKey);
  f.provider_options = ParseFlatOptionsObject(j, kProviderOptionsKey);
  f.shared_files = ParseFlatOptionsObject(j, kSharedFilesKey);
}

void from_json(const json& j, VariantMetadataSchema& v) {
  v.files = j.at(kFilesKey).get<std::vector<VariantFileSchema>>();
  if (v.files.empty()) {
    throw std::invalid_argument(MakeString("\"", kFilesKey, "\" must contain at least one entry"));
  }

  if (j.contains(kConsumerMetadataKey) && j[kConsumerMetadataKey].is_object()) {
    v.consumer_metadata = j[kConsumerMetadataKey];
  }
}

void from_json(const json& j, ManifestSchema& m) {
  m.schema_version = j.at(kSchemaVersionKey).get<int64_t>();  // required

  if (j.contains(kComponentsKey)) {
    if (!j[kComponentsKey].is_array()) {
      throw std::invalid_argument(MakeString("\"", kComponentsKey, "\" must be an array of strings"));
    }
    m.components = j[kComponentsKey].get<std::vector<std::string>>();
  }
}

void from_json(const json& j, ComponentSchema& m) {
  if (j.contains(kComponentModelNameInMetadataKey) && j[kComponentModelNameInMetadataKey].is_string()) {
    m.component_model_name = j[kComponentModelNameInMetadataKey].get<std::string>();
  }

  m.variants = j.at(kVariantsKey).get<std::unordered_map<std::string, VariantSchema>>();
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
}

std::string CompatibilityStringsToLogString(const std::optional<std::vector<std::string>>& v) {
  if (!v.has_value() || v->empty()) {
    return "";
  }

  std::ostringstream oss;
  for (size_t i = 0; i < v->size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << (*v)[i];
  }

  return oss.str();
}

std::string BuildModelInfoLogString(const ModelVariantInfo& variant) {
  std::ostringstream oss;
  oss << "component='" << variant.component_model_name
      << "' variant='" << variant.variant_name
      << "' ep_compat_count=" << variant.ep_compatibility.size()
      << "' file_count=" << variant.files.size();

  if (!variant.ep_compatibility.empty()) {
    oss << " ep_compat=[";
    for (size_t i = 0; i < variant.ep_compatibility.size(); ++i) {
      const auto& ec = variant.ep_compatibility[i];
      oss << "{ep='" << ec.ep.value_or("")
          << "', device='" << ec.device.value_or("")
          << "', compatibility_strings='" << CompatibilityStringsToLogString(ec.compatibility_strings)
          << "'}";
      if (i + 1 < variant.ep_compatibility.size()) {
        oss << ", ";
      }
    }
    oss << "]";
  }

  if (!variant.files.empty()) {
    oss << " files=[";
    for (size_t i = 0; i < variant.files.size(); ++i) {
      const auto& f = variant.files[i];
      oss << "{id='" << f.identifier
          << "', file='" << f.model_file_path.string()
          << "', has_session_options=" << (f.session_options.has_value() ? "true" : "false")
          << ", has_provider_options=" << (f.provider_options.has_value() ? "true" : "false")
          << ", shared_files_count=" << (f.shared_files.has_value() ? f.shared_files->size() : 0)
          << "}";
      if (i + 1 < variant.files.size()) {
        oss << ", ";
      }
    }
    oss << "]";
  }

  return oss.str();
}

void LogParsedVariants(const logging::Logger& logger,
                       const std::vector<ModelVariantInfo>& variants) {
  for (const auto& v : variants) {
    LOGS(logger, INFO) << "model variant: " << BuildModelInfoLogString(v);
  }
}

}  // namespace

Status ModelPackageDescriptorParser::ParseVariantsFromRoot(const std::filesystem::path& package_root,
                                                           std::vector<ModelVariantInfo>& variants) const {
  variants.clear();

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

    ComponentSchema metadata_schema;
    ORT_TRY {
      metadata_schema = metadata_doc.get<ComponentSchema>();
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
    const json* metadata_variants_obj = &metadata_doc.at(kVariantsKey);

    ORT_RETURN_IF_ERROR(ParseVariantsFromComponent(component_model_name,
                                                   package_root,
                                                   metadata_variants_obj,
                                                   variants));

    LogParsedVariants(logger_, variants);
    return Status::OK();
  }

  return ParseVariantsFromPackageRoot(package_root, variants);
}

Status ModelPackageDescriptorParser::ParseVariantsFromComponent(
    const std::string& component_model_name,
    const std::filesystem::path& component_model_root,
    const json* variants_obj,
    std::vector<ModelVariantInfo>& variant_infos) const {
  if (variants_obj == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Missing metadata variants for component model: ",
                           component_model_name);
  }

  std::unordered_map<std::string, VariantSchema> variants;
  ORT_TRY {
    variants = variants_obj->get<std::unordered_map<std::string, VariantSchema>>();
  }
  ORT_CATCH(const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Invalid metadata variant schema for component model '",
                           component_model_name, "': ", ex.what());
  }

  for (const auto& [variant_name, variant] : variants) {
    const std::filesystem::path variant_root = component_model_root / variant_name;
    const std::filesystem::path variant_descriptor_path = variant_root / kVariantDescriptorFileName;

    if (!std::filesystem::exists(variant_descriptor_path)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Missing variant.json for variant '", variant_name,
                             "' under component '", component_model_name,
                             "': ", variant_descriptor_path.string());
    }

    std::ifstream vf(variant_descriptor_path, std::ios::binary);
    if (!vf) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Failed to open variant.json at ", variant_descriptor_path.string());
    }

    json variant_doc;
    ORT_TRY {
      variant_doc = json::parse(vf);
    }
    ORT_CATCH(const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "variant.json at ", variant_descriptor_path.string(),
                             " is not valid JSON: ", ex.what());
    }

    VariantMetadataSchema variant_metadata_schema;
    ORT_TRY {
      variant_metadata_schema = variant_doc.get<VariantMetadataSchema>();
    }
    ORT_CATCH(const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "variant.json at ", variant_descriptor_path.string(),
                             " has invalid schema: ", ex.what());
    }

    ModelVariantInfo variant_info{};
    variant_info.component_model_name = component_model_name;
    variant_info.variant_name = variant_name;
    variant_info.consumer_metadata = variant_metadata_schema.consumer_metadata;

    std::unordered_set<std::string> identifiers_seen;

    for (const auto& file_schema : variant_metadata_schema.files) {
      const std::string identifier = file_schema.filename;  // deterministic identifier in v2 schema
      if (!identifiers_seen.insert(identifier).second) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Duplicate file identifier '", identifier,
                               "' in variant '", variant_name, "'.");
      }

      const std::filesystem::path candidate_path = variant_root / file_schema.filename;
      if (!std::filesystem::exists(candidate_path)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Variant '", variant_name, "', file '", file_schema.filename,
                               "' path does not exist: ", candidate_path.string());
      }

      std::filesystem::path resolved_model_path;
      if (std::filesystem::is_regular_file(candidate_path)) {
        resolved_model_path = candidate_path;
      } else if (std::filesystem::is_directory(candidate_path)) {
        ORT_RETURN_IF_ERROR(FindSingleOnnxFile(candidate_path, resolved_model_path));
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Variant '", variant_name, "', file '", file_schema.filename,
                               "' path is neither a file nor directory: ", candidate_path.string());
      }

      VariantModelInfo file_info{};
      file_info.identifier = identifier;
      file_info.model_file_path = std::move(resolved_model_path);
      file_info.session_options = file_schema.session_options;
      file_info.provider_options = file_schema.provider_options;
      file_info.shared_files = file_schema.shared_files;

      variant_info.files.push_back(std::move(file_info));
    }

    // Variant-level EP compatibility comes from metadata.json (not per-file).
    variant_info.ep_compatibility.clear();
    variant_info.ep_compatibility.reserve(variant.ep_compatibility.size());

    for (const auto& ec_schema : variant.ep_compatibility) {
      VariantEpCompatibilityInfo ec{};
      ec.ep = ec_schema.ep;
      ec.device = ec_schema.device;
      ec.compatibility_strings = ec_schema.compatibility_strings;

      const size_t compatibility_count = ec.compatibility_strings.has_value() ? ec.compatibility_strings->size() : 0;
      ec.compiled_model_compatibilities.assign(
          compatibility_count,
          OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
      variant_info.ep_compatibility.push_back(std::move(ec));
    }

    variant_infos.push_back(std::move(variant_info));
  }

  return Status::OK();
}

Status ModelPackageDescriptorParser::ParseVariantsFromPackageRoot(
    const std::filesystem::path& package_root,
    std::vector<ModelVariantInfo>& variants) const {
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

  ManifestSchema manifest_schema;
  ORT_TRY {
    manifest_schema = doc.get<ManifestSchema>();
  }
  ORT_CATCH(const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "manifest.json has invalid schema: ", ex.what());
  }

  if (manifest_schema.schema_version != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Unsupported schema_version in manifest.json: ", manifest_schema.schema_version,
                           ". Expected 1.");
  }

  const bool has_components = manifest_schema.components.has_value();
  std::vector<std::string> component_model_names;
  std::unordered_map<std::string, json> discovered_metadata_docs;

  if (has_components) {
    component_model_names = *manifest_schema.components;
  } else {
    const auto models_dir = package_root / "models";
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "manifest.json missing \"components\" and no discoverable models directory at ",
                             models_dir.string());
    }

    for (const auto& entry : std::filesystem::directory_iterator(models_dir)) {
      if (!entry.is_directory()) {
        continue;
      }

      const auto component_model_name = entry.path().filename().string();
      const auto metadata_path = entry.path() / kComponentModelMetadataFileName;
      if (!std::filesystem::exists(metadata_path)) {
        continue;
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
        (void)metadata_doc.get<ComponentSchema>();
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
                             "manifest.json missing \"components\" and no component model folders with metadata.json were found under ",
                             models_dir.string());
    }
  }

  for (const auto& component_model_name : component_model_names) {
    const auto component_model_root = package_root / "models" / component_model_name;

    if (has_components &&
        (!std::filesystem::exists(component_model_root) || !std::filesystem::is_directory(component_model_root))) {
      LOGS(logger_, WARNING) << "Component model '" << component_model_name
                             << "' is listed in manifest.json but directory does not exist: "
                             << component_model_root.string()
                             << ". Skipping this component model.";
      continue;
    }

    json metadata_doc;
    const json* variants_obj = nullptr;
    const auto metadata_path = component_model_root / kComponentModelMetadataFileName;

    if (!has_components) {
      auto it_meta = discovered_metadata_docs.find(component_model_name);
      if (it_meta != discovered_metadata_docs.end()) {
        metadata_doc = it_meta->second;
        variants_obj = &metadata_doc.at(kVariantsKey);
      }
    } else if (std::filesystem::exists(metadata_path)) {
      std::ifstream mf(metadata_path, std::ios::binary);
      if (mf) {
        ORT_TRY {
          metadata_doc = json::parse(mf);
          (void)metadata_doc.get<ComponentSchema>();
          variants_obj = &metadata_doc.at(kVariantsKey);
        }
        ORT_CATCH(const std::exception&) {
          // Ignore metadata parse/schema errors; fall back below.
        }
      }
    }

    if (!metadata_doc.is_null() &&
        metadata_doc.contains(kComponentModelNameInMetadataKey) &&
        metadata_doc[kComponentModelNameInMetadataKey].is_string()) {
      const auto metadata_component_name = metadata_doc[kComponentModelNameInMetadataKey].get<std::string>();
      if (metadata_component_name != component_model_name) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "metadata.json component_model_name '", metadata_component_name,
                               "' does not match directory/manifest component name '", component_model_name, "'.");
      }
    }

    ORT_RETURN_IF_ERROR(ParseVariantsFromComponent(component_model_name,
                                                   component_model_root,
                                                   variants_obj,
                                                   variants));
  }

  if (variants.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "No valid component models were found under ", (package_root / "models").string());
  }

  LogParsedVariants(logger_, variants);
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)
}  // namespace onnxruntime