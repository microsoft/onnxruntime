// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "parser.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace model_package {
namespace {

// ─────────────────────────────────────────────────────────────────────────────
// JSON key constants
// ─────────────────────────────────────────────────────────────────────────────

constexpr const char* kManifestFileName = "manifest.json";
constexpr const char* kMetadataFileName = "metadata.json";
constexpr const char* kVariantDescriptorFileName = "variant.json";

constexpr const char* kSchemaVersionKey = "schema_version";
constexpr const char* kComponentsKey = "components";
constexpr const char* kComponentNameKey = "component_name";
constexpr const char* kVariantsKey = "variants";

constexpr const char* kEpKey = "ep";
constexpr const char* kDeviceKey = "device";
constexpr const char* kCompatibilityStringKey = "compatibility_string";

constexpr const char* kFilenameKey = "filename";
constexpr const char* kSessionOptionsKey = "session_options";
constexpr const char* kProviderOptionsKey = "provider_options";
constexpr const char* kSharedFilesKey = "shared_files";
constexpr const char* kConsumerMetadataKey = "consumer_metadata";

// ─────────────────────────────────────────────────────────────────────────────
// Internal schema types for deserialization
// ─────────────────────────────────────────────────────────────────────────────

struct VariantMetadataSchema {
  std::string filename;
  std::optional<std::unordered_map<std::string, std::string>> session_options;
  std::optional<std::unordered_map<std::string, std::string>> provider_options;
  std::optional<std::unordered_map<std::string, std::string>> shared_files;
};

struct EpCompatibilitySchema {
  std::optional<std::string> ep;
  std::optional<std::string> device;
  std::optional<std::string> compatibility_string;
};

struct VariantSchema {
  EpCompatibilitySchema ep_info;
};

struct ComponentSchema {
  std::optional<std::string> component_name;
  std::unordered_map<std::string, VariantSchema> variants;
};

struct ManifestSchema {
  int64_t schema_version;
  std::optional<std::vector<std::string>> components;
};

// ─────────────────────────────────────────────────────────────────────────────
// JSON helpers
// ─────────────────────────────────────────────────────────────────────────────

std::string JsonScalarToString(const json& v, const char* key_name, const std::string& parent_key) {
  if (v.is_string()) return v.get<std::string>();
  if (v.is_number_integer()) return std::to_string(v.get<int64_t>());
  if (v.is_number_unsigned()) return std::to_string(v.get<uint64_t>());
  if (v.is_number_float()) return v.dump();
  if (v.is_boolean()) return v.get<bool>() ? "true" : "false";

  throw std::invalid_argument(
      std::string("\"") + key_name + "\" under '" + parent_key +
      "' must contain scalar (string/number/bool) values.");
}

std::optional<std::unordered_map<std::string, std::string>> ParseFlatOptionsObject(
    const json& j, const char* key_name) {
  if (!j.contains(key_name) || j[key_name].is_null()) {
    return std::nullopt;
  }

  const auto& obj = j[key_name];
  if (!obj.is_object()) {
    throw std::invalid_argument(std::string("\"") + key_name + "\" must be an object.");
  }

  std::unordered_map<std::string, std::string> result;
  result.reserve(obj.size());

  for (auto it = obj.begin(); it != obj.end(); ++it) {
    result.emplace(it.key(), JsonScalarToString(it.value(), key_name, it.key()));
  }

  return result;
}

std::optional<std::string> ParseOptionalString(const json& j, const char* key_name) {
  if (!j.contains(key_name) || j[key_name].is_null()) {
    return std::nullopt;
  }

  const auto& value = j[key_name];
  if (!value.is_string()) {
    throw std::invalid_argument(std::string("\"") + key_name + "\" must be a string.");
  }
  return value.get<std::string>();
}

// ─────────────────────────────────────────────────────────────────────────────
// nlohmann from_json overloads
// ─────────────────────────────────────────────────────────────────────────────

void from_json(const json& j, EpCompatibilitySchema& c) {
  if (!j.contains(kEpKey) || j[kEpKey].is_null()) {
    throw std::invalid_argument(std::string("\"") + kEpKey + "\" is required in each ep_compatibility entry.");
  }
  if (!j[kEpKey].is_string()) {
    throw std::invalid_argument(std::string("\"") + kEpKey + "\" must be a string.");
  }
  c.ep = j[kEpKey].get<std::string>();
  if (c.ep->empty()) {
    throw std::invalid_argument(std::string("\"") + kEpKey + "\" must be a non-empty string.");
  }

  if (j.contains(kDeviceKey) && !j[kDeviceKey].is_null()) {
    if (!j[kDeviceKey].is_string()) {
      throw std::invalid_argument(std::string("\"") + kDeviceKey + "\" must be a string when present.");
    }
    c.device = j[kDeviceKey].get<std::string>();
  }
  c.compatibility_string = ParseOptionalString(j, kCompatibilityStringKey);
}

void from_json(const json& j, VariantSchema& v) {
  // EP fields (ep, device, compatibility_string) are now directly on the variant object.
  // "ep" is required.
  v.ep_info = j.get<EpCompatibilitySchema>();
}

void from_json(const json& j, VariantMetadataSchema& v) {
  v.filename = j.at(kFilenameKey).get<std::string>();
  v.session_options = ParseFlatOptionsObject(j, kSessionOptionsKey);
  v.provider_options = ParseFlatOptionsObject(j, kProviderOptionsKey);
  v.shared_files = ParseFlatOptionsObject(j, kSharedFilesKey);
}

void from_json(const json& j, ManifestSchema& m) {
  m.schema_version = j.at(kSchemaVersionKey).get<int64_t>();

  if (j.contains(kComponentsKey)) {
    if (!j[kComponentsKey].is_array()) {
      throw std::invalid_argument(std::string("\"") + kComponentsKey + "\" must be an array of strings");
    }
    m.components = j[kComponentsKey].get<std::vector<std::string>>();
  }
}

void from_json(const json& j, ComponentSchema& m) {
  if (j.contains(kComponentNameKey) && j[kComponentNameKey].is_string()) {
    m.component_name = j[kComponentNameKey].get<std::string>();
  }

  m.variants = j.at(kVariantsKey).get<std::unordered_map<std::string, VariantSchema>>();
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsing variants in declaration order (from the JSON object)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::pair<std::string, VariantSchema>> ParseVariantsInOrder(const json& variants_obj) {
  std::vector<std::pair<std::string, VariantSchema>> result;
  result.reserve(variants_obj.size());
  for (auto it = variants_obj.begin(); it != variants_obj.end(); ++it) {
    result.emplace_back(it.key(), it.value().get<VariantSchema>());
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Path validation
// ─────────────────────────────────────────────────────────────────────────────

bool ValidatePathSegment(const std::string& segment, const char* segment_type, std::string& error) {
  if (segment.empty()) {
    error = std::string(segment_type) + " must not be empty.";
    return false;
  }

  if (std::filesystem::path(segment).is_absolute()) {
    error = std::string(segment_type) + " must not be an absolute path: '" + segment + "'.";
    return false;
  }

  for (const auto& part : std::filesystem::path(segment)) {
    if (part == "..") {
      error = std::string(segment_type) + " must not contain '..' path components: '" + segment + "'.";
      return false;
    }
  }

  return true;
}

bool ValidatePathConfinement(const std::filesystem::path& resolved_path,
                             const std::filesystem::path& root,
                             const char* description,
                             std::string& error) {
  auto normal_root = root.lexically_normal();
  auto normal_path = resolved_path.lexically_normal();

  auto root_str = normal_root.string();
  auto path_str = normal_path.string();

  if (path_str.size() < root_str.size() ||
      path_str.compare(0, root_str.size(), root_str) != 0 ||
      (path_str.size() > root_str.size() && path_str[root_str.size()] != std::filesystem::path::preferred_separator
#ifndef _WIN32
       && path_str[root_str.size()] != '/'
#endif
       )) {
    error = std::string(description) + " resolves outside the package root. Path: '" +
            resolved_path.string() + "', Root: '" + root.string() + "'.";
    return false;
  }

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Find single ONNX file in directory
// ─────────────────────────────────────────────────────────────────────────────

bool FindSingleOnnxFile(const std::filesystem::path& search_dir,
                        std::filesystem::path& resolved_path,
                        std::string& error) {
  std::vector<std::filesystem::path> onnx_files;
  for (const auto& entry : std::filesystem::directory_iterator(search_dir)) {
    if (!entry.is_regular_file()) continue;

    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".onnx") {
      onnx_files.push_back(entry.path());
    }
  }

  if (onnx_files.empty()) {
    error = "No ONNX model file found under " + search_dir.string();
    return false;
  }

  if (onnx_files.size() > 1) {
    error = "Multiple ONNX model files found under " + search_dir.string() +
            ". Multiple ONNX files per variant are not supported yet.";
    return false;
  }

  resolved_path = onnx_files.front();
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse variants from a single component
// ─────────────────────────────────────────────────────────────────────────────

bool ParseVariantsFromComponent(const std::string& component_name,
                                const std::filesystem::path& component_root,
                                const json* variants_obj,
                                std::vector<Variant>& out_variants,
                                std::string& error) {
  if (variants_obj == nullptr) {
    error = "Missing metadata variants for component: " + component_name;
    return false;
  }

  std::vector<std::pair<std::string, VariantSchema>> variants;
  try {
    variants = ParseVariantsInOrder(*variants_obj);
  } catch (const std::exception& ex) {
    error = "Invalid metadata variant schema for component '" + component_name + "': " + ex.what();
    return false;
  }

  for (const auto& [variant_name, variant_schema] : variants) {
    if (!ValidatePathSegment(variant_name, "Variant name", error)) return false;

    const std::filesystem::path variant_root = component_root / variant_name;
    if (!ValidatePathConfinement(variant_root, component_root, "Variant directory", error)) return false;

    const std::filesystem::path variant_descriptor_path = variant_root / kVariantDescriptorFileName;

    Variant variant_info{};
    variant_info.name = variant_name;
    variant_info.folder_path = variant_root;

    // variant.json is optional. If present, it declares the file list,
    // per-file session/provider options, and consumer metadata.
    if (std::filesystem::exists(variant_descriptor_path)) {
      std::ifstream vf(variant_descriptor_path, std::ios::binary);
      if (!vf) {
        error = "Failed to open variant.json at " + variant_descriptor_path.string();
        return false;
      }

      json variant_doc;
      try {
        variant_doc = json::parse(vf);
      } catch (const std::exception& ex) {
        error = "variant.json at " + variant_descriptor_path.string() + " is not valid JSON: " + ex.what();
        return false;
      }

      VariantMetadataSchema variant_metadata;
      try {
        variant_metadata = variant_doc.get<VariantMetadataSchema>();
      } catch (const std::exception& ex) {
        error = "variant.json at " + variant_descriptor_path.string() + " has invalid schema: " + ex.what();
        return false;
      }

      // consumer_metadata is a top-level optional field parsed separately from the schema struct.
      if (variant_doc.contains(kConsumerMetadataKey) && variant_doc[kConsumerMetadataKey].is_object()) {
        variant_info.consumer_metadata_json = variant_doc[kConsumerMetadataKey].dump();
      }

      if (!ValidatePathSegment(variant_metadata.filename, "File name", error)) return false;

      const std::filesystem::path candidate_path = variant_root / variant_metadata.filename;
      if (!ValidatePathConfinement(candidate_path, variant_root, "Variant file path", error)) return false;

      if (!std::filesystem::exists(candidate_path)) {
        error = "Variant '" + variant_name + "', file '" + variant_metadata.filename +
                "' path does not exist: " + candidate_path.string();
        return false;
      }

      std::filesystem::path resolved_model_path;
      if (std::filesystem::is_regular_file(candidate_path)) {
        resolved_model_path = candidate_path;
      } else if (std::filesystem::is_directory(candidate_path)) {
        if (!FindSingleOnnxFile(candidate_path, resolved_model_path, error)) return false;
      } else {
        error = "Variant '" + variant_name + "', file '" + variant_metadata.filename +
                "' path is neither a file nor directory: " + candidate_path.string();
        return false;
      }

      VariantFile file_info{};
      file_info.filename = variant_metadata.filename;
      file_info.resolved_path = std::move(resolved_model_path);
      file_info.session_options = variant_metadata.session_options;
      file_info.provider_options = variant_metadata.provider_options;
      file_info.shared_files = variant_metadata.shared_files;

      variant_info.file = std::move(file_info);
    }

    // EP compatibility from metadata.json (single entry per variant)
    variant_info.ep_compatibility.ep = variant_schema.ep_info.ep;
    variant_info.ep_compatibility.device = variant_schema.ep_info.device;
    variant_info.ep_compatibility.compatibility_string = variant_schema.ep_info.compatibility_string;

    out_variants.push_back(std::move(variant_info));
  }

  return true;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public parser entry point
// ─────────────────────────────────────────────────────────────────────────────

bool ParsePackage(const std::filesystem::path& package_root,
                  PackageInfo& out_package,
                  std::string& out_error) {
  out_package = {};
  out_package.root_path = package_root;

  // Check for single-component mode: metadata.json at root
  const auto root_metadata_path = package_root / kMetadataFileName;
  if (std::filesystem::exists(root_metadata_path) &&
      std::filesystem::is_regular_file(root_metadata_path)) {
    std::ifstream mf(root_metadata_path, std::ios::binary);
    if (!mf) {
      out_error = "Failed to open metadata.json at " + root_metadata_path.string();
      return false;
    }

    json metadata_doc;
    try {
      metadata_doc = json::parse(mf);
    } catch (const std::exception& ex) {
      out_error = "metadata.json at " + root_metadata_path.string() + " is not valid JSON: " + ex.what();
      return false;
    }

    ComponentSchema metadata_schema;
    try {
      metadata_schema = metadata_doc.get<ComponentSchema>();
    } catch (const std::exception& ex) {
      out_error = "metadata.json at " + root_metadata_path.string() + " has invalid schema: " + ex.what();
      return false;
    }

    const std::string component_name =
        metadata_schema.component_name.has_value()
            ? *metadata_schema.component_name
            : package_root.filename().string();

    const json* variants_obj = &metadata_doc.at(kVariantsKey);

    Component component{};
    component.name = component_name;

    if (!ParseVariantsFromComponent(component_name, package_root, variants_obj,
                                    component.variants, out_error)) {
      return false;
    }

    out_package.schema_version = 0;  // Single-component mode doesn't have a manifest
    out_package.components.push_back(std::move(component));
    return true;
  }

  // Multi-component mode: manifest.json at root
  const auto manifest_path = package_root / kManifestFileName;
  if (!std::filesystem::exists(manifest_path)) {
    out_error = "No manifest.json found at " + manifest_path.string();
    return false;
  }

  std::ifstream f(manifest_path, std::ios::binary);
  if (!f) {
    out_error = "Failed to open manifest.json at " + manifest_path.string();
    return false;
  }

  json doc;
  try {
    doc = json::parse(f);
  } catch (const std::exception& ex) {
    out_error = std::string("manifest.json is not valid JSON: ") + ex.what();
    return false;
  }

  ManifestSchema manifest_schema;
  try {
    manifest_schema = doc.get<ManifestSchema>();
  } catch (const std::exception& ex) {
    out_error = std::string("manifest.json has invalid schema: ") + ex.what();
    return false;
  }

  if (manifest_schema.schema_version != 1) {
    out_error = "Unsupported schema_version in manifest.json: " +
                std::to_string(manifest_schema.schema_version) + ". Expected 1.";
    return false;
  }

  out_package.schema_version = manifest_schema.schema_version;

  const bool has_components = manifest_schema.components.has_value();
  std::vector<std::string> component_names;
  std::unordered_map<std::string, json> discovered_metadata_docs;

  if (has_components) {
    component_names = *manifest_schema.components;
  } else {
    const auto models_dir = package_root / "models";
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
      out_error = "manifest.json missing \"components\" and no discoverable models directory at " +
                  models_dir.string();
      return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(models_dir)) {
      if (!entry.is_directory()) continue;

      const auto name = entry.path().filename().string();
      const auto metadata_path = entry.path() / kMetadataFileName;
      if (!std::filesystem::exists(metadata_path)) continue;

      std::ifstream mf(metadata_path, std::ios::binary);
      if (!mf) {
        out_error = "Failed to open metadata.json at " + metadata_path.string();
        return false;
      }

      json metadata_doc;
      try {
        metadata_doc = json::parse(mf);
        (void)metadata_doc.get<ComponentSchema>();
      } catch (const std::exception& ex) {
        out_error = "metadata.json at " + metadata_path.string() +
                    " has invalid schema: " + std::string(ex.what());
        return false;
      }

      discovered_metadata_docs.emplace(name, std::move(metadata_doc));
      component_names.push_back(name);
    }

    if (component_names.empty()) {
      out_error =
          "manifest.json missing \"components\" and no component model folders with "
          "metadata.json were found under " +
          models_dir.string();
      return false;
    }
  }

  for (const auto& component_name : component_names) {
    if (!ValidatePathSegment(component_name, "Component name", out_error)) return false;

    const auto component_root = package_root / "models" / component_name;
    if (!ValidatePathConfinement(component_root, package_root, "Component directory", out_error)) return false;

    if (has_components &&
        (!std::filesystem::exists(component_root) || !std::filesystem::is_directory(component_root))) {
      // Skip missing component directories (just warn — standalone library doesn't have logging,
      // so we skip silently for now).
      continue;
    }

    json metadata_doc;
    const json* variants_obj = nullptr;
    const auto metadata_path = component_root / kMetadataFileName;

    if (!has_components) {
      auto it_meta = discovered_metadata_docs.find(component_name);
      if (it_meta != discovered_metadata_docs.end()) {
        metadata_doc = it_meta->second;
        variants_obj = &metadata_doc.at(kVariantsKey);
      }
    } else if (std::filesystem::exists(metadata_path)) {
      std::ifstream mf(metadata_path, std::ios::binary);
      if (mf) {
        try {
          metadata_doc = json::parse(mf);
          (void)metadata_doc.get<ComponentSchema>();
          variants_obj = &metadata_doc.at(kVariantsKey);
        } catch (const std::exception&) {
          // Ignore parse errors, fall through.
        }
      }
    }

    if (!metadata_doc.is_null() &&
        metadata_doc.contains(kComponentNameKey) &&
        metadata_doc[kComponentNameKey].is_string()) {
      const auto metadata_component_name = metadata_doc[kComponentNameKey].get<std::string>();
      if (metadata_component_name != component_name) {
        out_error = "metadata.json component_name '" + metadata_component_name +
                    "' does not match directory/manifest component name '" + component_name + "'.";
        return false;
      }
    }

    Component component{};
    component.name = component_name;

    if (!ParseVariantsFromComponent(component_name, component_root, variants_obj,
                                    component.variants, out_error)) {
      return false;
    }

    out_package.components.push_back(std::move(component));
  }

  if (out_package.components.empty()) {
    out_error = "No valid component models were found under " + (package_root / "models").string();
    return false;
  }

  return true;
}

}  // namespace model_package
