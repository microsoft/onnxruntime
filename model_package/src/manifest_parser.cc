// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "manifest_parser.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <system_error>

#include "path_resolver.h"
#include "status_impl.h"

namespace fs = std::filesystem;

namespace model_package {

namespace {

// The on-disk schema_version is a "<major>.<minor>" string (e.g. "1.0"). The major gates
// compatibility; the minor is informational and tells consumers which optional fields may
// be present. This build understands schema majors in [kMinSupportedSchemaMajor,
// kMaxSupportedSchemaMajor] and any minor: schema evolution within a major is additive and
// backward-compatible (newer minors only add optional fields), so a single parser reads
// every minor. A package whose major is below the minimum predates a breaking change and
// must be re-authored; one above the maximum was produced by a newer toolchain this build
// does not understand. kMaxKnownSchemaMinor is the highest minor this build authored/knows;
// a package with a higher minor is still accepted but may carry unknown optional fields,
// which are tolerated rather than rejected.
constexpr int64_t kMinSupportedSchemaMajor = 1;
constexpr int64_t kMaxSupportedSchemaMajor = 1;
constexpr int64_t kMaxKnownSchemaMinor = 0;
constexpr const char* kManifestFileName = "manifest.json";
constexpr const char* kComponentFileName = "component.json";

constexpr const char* kSchemaVersionKey = "schema_version";
constexpr const char* kPackageNameKey = "package_name";
constexpr const char* kPackageVersionKey = "package_version";
constexpr const char* kDescriptionKey = "description";
constexpr const char* kLayoutKey = "layout";
constexpr const char* kComponentsKey = "components";
constexpr const char* kSharedAssetsKey = "shared_assets";
constexpr const char* kAdditionalMetadataKey = "additional_metadata";

constexpr const char* kComponentNameKey = "component_name";
constexpr const char* kVariantsKey = "variants";

constexpr const char* kVariantDirectoryKey = "variant_directory";
constexpr const char* kEpKey = "ep";
constexpr const char* kDeviceKey = "device";
constexpr const char* kCompatibilityStringKey = "compatibility_string";
constexpr const char* kExecutorInfoKey = "executor_info";

static const std::set<std::string> kManifestKnownKeys = {
    kSchemaVersionKey,
    kPackageNameKey,
    kPackageVersionKey,
    kDescriptionKey,
    kLayoutKey,
    kComponentsKey,
    kSharedAssetsKey,
    kAdditionalMetadataKey,
};

static const std::set<std::string> kComponentKnownKeys = {
    kComponentNameKey,
    kVariantsKey,
    kAdditionalMetadataKey,
};

static const std::set<std::string> kVariantKnownKeys = {
    kVariantDirectoryKey,
    kEpKey,
    kDeviceKey,
    kCompatibilityStringKey,
    kExecutorInfoKey,
    kAdditionalMetadataKey,
};

ModelPackageStatus* ReadFileToString(const fs::path& path, std::string* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      "Cannot open file: '" + path.string() + "': " + std::strerror(errno));
  }
  std::ostringstream buf;
  buf << f.rdbuf();
  *out = buf.str();
  return nullptr;
}

ModelPackageStatus* ParseJsonFile(const fs::path& path, ordered_json* out) {
  std::string contents;
  if (auto* s = ReadFileToString(path, &contents)) return s;
  try {
    *out = ordered_json::parse(contents);
  } catch (const ordered_json::parse_error& e) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "Failed to parse JSON at '" + path.string() + "': " + e.what());
  }
  return nullptr;
}

ModelPackageStatus* ExpectObject(const ordered_json& j, const std::string& where) {
  if (!j.is_object()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA, where + ": expected a JSON object.");
  }
  return nullptr;
}

ModelPackageStatus* CheckUnknownFields(const ordered_json& obj,
                                       const std::set<std::string>& known,
                                       const std::string& where,
                                       bool strict) {
  if (!strict) return nullptr;
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    if (known.find(it.key()) == known.end()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        where + ": unknown field '" + it.key() + "'.");
    }
  }
  return nullptr;
}

ModelPackageStatus* ResolveVariantDirectory(const fs::path& component_dir,
                                            const fs::path& package_root,
                                            const ordered_json& variant_body,
                                            const std::string& variant_name,
                                            const PathResolverOptions& opts,
                                            bool require_exists,
                                            std::optional<fs::path>* out) {
  auto it = variant_body.find(kVariantDirectoryKey);
  bool explicitly_declared = (it != variant_body.end());
  std::string dir_input;
  if (explicitly_declared) {
    if (!it->is_string()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "variant '" + variant_name + "': variant_directory must be a string.");
    }
    dir_input = it->get<std::string>();
  } else {
    // Inferred default: missing-on-disk is fine; we just leave out unset.
    dir_input = variant_name;
  }

  fs::path resolved;
  // Explicit value must exist; inferred default may not.
  bool must_exist = require_exists || explicitly_declared;
  auto* status = ResolvePath(component_dir, package_root, dir_input, opts,
                             must_exist, &resolved);
  if (status) {
    if (!must_exist && ModelPackageStatus_Code(status) == MODEL_PACKAGE_ERR_NOT_FOUND) {
      ModelPackageStatus_Release(status);
      *out = std::nullopt;
      return nullptr;
    }
    return status;
  }
  std::error_code ec;
  if (fs::exists(resolved, ec)) {
    *out = resolved;
  } else {
    *out = std::nullopt;
  }
  return nullptr;
}

ModelPackageStatus* ParseVariant(const fs::path& component_dir,
                                 const fs::path& package_root,
                                 const PathResolverOptions& opts,
                                 bool strict,
                                 const std::string& variant_name,
                                 const ordered_json& variant_body,
                                 VariantRecord* out);
ModelPackageStatus* ParseComponent(const fs::path& package_root,
                                   const PathResolverOptions& opts,
                                   bool strict,
                                   const std::string& component_name,
                                   const ordered_json& body,
                                   const fs::path& component_dir,
                                   ComponentRecord* out);
ModelPackageStatus* LoadSharedAssets(ModelPackage* pkg, const PathResolverOptions& opts);
ModelPackageStatus* PopulatePackageMetadata(ModelPackage* pkg);

ModelPackageStatus* ParseVariant(const fs::path& component_dir,
                                 const fs::path& package_root,
                                 const PathResolverOptions& opts,
                                 bool strict,
                                 const std::string& variant_name,
                                 const ordered_json& variant_body,
                                 VariantRecord* out) {
  if (auto* s = ExpectObject(variant_body, "variant '" + variant_name + "'")) return s;
  if (auto* s = CheckUnknownFields(variant_body, kVariantKnownKeys,
                                   "variant '" + variant_name + "'", strict))
    return s;

  out->name = variant_name;
  out->body = variant_body;
  out->name_cache = variant_name;

  auto stringopt = [&](const char* key, std::optional<std::string>* dst) -> ModelPackageStatus* {
    auto it = variant_body.find(key);
    if (it == variant_body.end()) return nullptr;
    if (!it->is_string()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        std::string("variant '") + variant_name + "': '" + key +
                            "' must be a string.");
    }
    *dst = it->get<std::string>();
    return nullptr;
  };
  if (auto* s = stringopt(kEpKey, &out->ep_cache)) return s;
  if (auto* s = stringopt(kDeviceKey, &out->device_cache)) return s;
  if (auto* s = stringopt(kCompatibilityStringKey, &out->compatibility_string_cache)) return s;

  auto ei_it = variant_body.find(kExecutorInfoKey);
  if (ei_it != variant_body.end()) {
    if (!ei_it->is_object()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "variant '" + variant_name + "': executor_info must be an object.");
    }
    for (auto e = ei_it->begin(); e != ei_it->end(); ++e) {
      if (!e->is_string() && !e->is_object()) {
        return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                          "variant '" + variant_name + "': executor_info['" + e.key() +
                              "'] must be a string (path) or object (inline).");
      }
    }
  }

  // Resolve variant_directory if declared (records the resolved path when it
  // exists on disk). We do NOT require the directory to exist here: executor
  // semantics are not the library's concern, and executors must resolve their
  // own file references against variant_directory at load time anyway.
  std::optional<fs::path> resolved_dir;
  auto* status = ResolveVariantDirectory(component_dir, package_root, variant_body,
                                         variant_name, opts,
                                         /*require_exists=*/false, &resolved_dir);
  if (status) return status;
  out->resolved_directory = resolved_dir;
  out->resolved_directory_attempted = true;
  if (resolved_dir.has_value()) {
    out->resolved_directory_cache = resolved_dir->string();
  }

  return nullptr;
}

ModelPackageStatus* ParseComponent(const fs::path& package_root,
                                   const PathResolverOptions& opts,
                                   bool strict,
                                   const std::string& component_name,
                                   const ordered_json& body,
                                   const fs::path& component_dir,
                                   ComponentRecord* out) {
  if (auto* s = ExpectObject(body, "component '" + component_name + "'")) return s;
  if (auto* s = CheckUnknownFields(body, kComponentKnownKeys,
                                   "component '" + component_name + "'", strict))
    return s;
  out->name = component_name;
  out->name_cache = component_name;
  out->component_dir = component_dir;
  out->body = body;

  // Optional component_name override — for now we just sanity-check it.
  auto cn_it = body.find(kComponentNameKey);
  if (cn_it != body.end() && !cn_it->is_string()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "component '" + component_name + "': component_name must be a string.");
  }

  auto variants_it = body.find(kVariantsKey);
  if (variants_it == body.end()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "component '" + component_name + "': missing required 'variants' object.");
  }
  if (!variants_it->is_object()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "component '" + component_name + "': 'variants' must be an object.");
  }
  for (auto v = variants_it->begin(); v != variants_it->end(); ++v) {
    auto vr = std::make_unique<VariantRecord>();
    if (auto* s = ParseVariant(component_dir, package_root, opts, strict,
                               v.key(), v.value(), vr.get())) {
      return s;
    }
    out->variants.push_back(std::move(vr));
  }
  return nullptr;
}

ModelPackageStatus* LoadComponentForEntry(const fs::path& manifest_dir,
                                          const fs::path& package_root,
                                          const PathResolverOptions& opts,
                                          bool strict,
                                          const std::string& name,
                                          const ordered_json& value,
                                          std::unique_ptr<ComponentRecord>* out) {
  auto rec = std::make_unique<ComponentRecord>();
  if (value.is_string()) {
    rec->storage = ComponentStorage::kExternal;
    fs::path resolved;
    if (auto* s = ResolvePath(manifest_dir, package_root, value.get<std::string>(),
                              opts, /*must_exist=*/true, &resolved)) {
      return s;
    }
    std::error_code ec;
    if (fs::is_directory(resolved, ec)) {
      resolved /= kComponentFileName;
      if (!fs::exists(resolved)) {
        return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                          "component '" + name + "': directory has no '" +
                              kComponentFileName + "'.");
      }
    }
    rec->external_path = resolved;
    ordered_json body;
    if (auto* s = ParseJsonFile(resolved, &body)) return s;
    fs::path component_dir = resolved.parent_path();
    if (auto* s = ParseComponent(package_root, opts, strict, name, body, component_dir, rec.get())) {
      return s;
    }
  } else if (value.is_object()) {
    rec->storage = ComponentStorage::kInline;
    if (auto* s = ParseComponent(package_root, opts, strict, name, value, manifest_dir, rec.get())) {
      return s;
    }
  } else {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "component '" + name + "': value must be a string (path) or object (inline).");
  }
  *out = std::move(rec);
  return nullptr;
}

ModelPackageStatus* LoadSharedAssets(ModelPackage* pkg, const PathResolverOptions& opts) {
  // Source-of-truth ordering for the assembled shared_assets vector:
  //   1. Manifest overrides (in declaration order). These specify a custom
  //      on-disk path for an asset URI (e.g. a system-wide cache or another
  //      location outside <package_root>/shared_assets/).
  //   2. Discovered sha256-<hex> subdirectories under <package_root>/shared_assets/.
  //      These resolve to the default-convention path. Missing shared_assets/ is
  //      not an error: portable packages may not ship one yet, installed
  //      packages may route everything through overrides.
  //   3. Pending copy_in assets from the authoring API that haven't been
  //      committed yet. These surface immediately so callers see the asset
  //      they just added; the staged source dir is reported as resolved_path
  //      until commit materializes it under shared_assets/.
  // Within each tier, an URI that's already known is skipped.
  std::vector<std::string> ordered_uris;
  std::unordered_map<std::string, std::string> override_paths;

  auto sa_it = pkg->manifest.find(kSharedAssetsKey);
  if (sa_it != pkg->manifest.end()) {
    if (!sa_it->is_object()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "manifest: shared_assets must be an object.");
    }
    for (auto e = sa_it->begin(); e != sa_it->end(); ++e) {
      const std::string uri = e.key();
      if (!IsSha256AssetUri(uri)) {
        return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                          "manifest: shared_assets key '" + uri + "' is not a valid sha256:<hex> URI.");
      }
      if (!e->is_string()) {
        return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                          "manifest: shared_assets['" + uri + "'] must be a string path.");
      }
      ordered_uris.push_back(uri);
      override_paths.emplace(uri, e->get<std::string>());
    }
  }
  std::set<std::string> seen(ordered_uris.begin(), ordered_uris.end());

  // Tier 2: discover sha256-<hex> dirs under <package_root>/shared_assets/.
  fs::path assets_root = pkg->package_root / "shared_assets";
  std::error_code ec;
  if (!pkg->package_root.empty() && fs::is_directory(assets_root, ec)) {
    std::vector<std::string> discovered;
    for (const auto& entry : fs::directory_iterator(assets_root, ec)) {
      if (ec) break;
      if (!entry.is_directory(ec)) continue;
      std::string name = entry.path().filename().string();
      std::string uri = SharedAssetUriFromDirName(name);
      if (uri.empty()) continue;  // not a sha256-<hex> dir; ignore (.tmp staging, etc.)
      if (!seen.insert(uri).second) continue;
      discovered.push_back(std::move(uri));
    }
    std::sort(discovered.begin(), discovered.end());
    for (auto& uri : discovered) ordered_uris.push_back(std::move(uri));
  }

  // Tier 3: pending copy_in entries.
  for (const auto& [uri, src] : pkg->pending_shared_asset_copies) {
    if (!seen.insert(uri).second) continue;
    ordered_uris.push_back(uri);
  }

  for (const auto& uri : ordered_uris) {
    auto rec = std::make_unique<SharedAssetRecord>();
    rec->uri = uri;
    rec->uri_cache = uri;
    auto override_it = override_paths.find(uri);
    fs::path resolved;
    if (override_it != override_paths.end()) {
      if (auto* s = ResolvePath(pkg->package_root, pkg->package_root, override_it->second,
                                opts, /*must_exist=*/false, &resolved)) {
        return s;
      }
    } else if (auto pending_it = pkg->pending_shared_asset_copies.find(uri);
               pending_it != pkg->pending_shared_asset_copies.end() &&
               override_paths.find(uri) == override_paths.end()) {
      // Pending copy_in with no override: surface the staged source until commit.
      resolved = pending_it->second;
    } else {
      // Default convention: <package_root>/shared_assets/sha256-<hex>/
      resolved = assets_root / DefaultSharedAssetDirName(uri);
    }
    rec->resolved_path = resolved;
    rec->resolved_path_cache = resolved.string();
    pkg->shared_asset_index_by_uri.emplace(uri, pkg->shared_assets.size());
    pkg->shared_assets.push_back(std::move(rec));
  }
  return nullptr;
}

ModelPackageStatus* ParseSchemaVersion(ModelPackage* pkg) {
  auto sv_it = pkg->manifest.find(kSchemaVersionKey);
  if (sv_it == pkg->manifest.end()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "manifest: missing required 'schema_version'.");
  }

  // schema_version is a "<major>.<minor>" string (e.g. "1.0"). A bare integer is accepted
  // as shorthand for "<major>.0".
  int64_t major = 0;
  int64_t minor = 0;
  if (sv_it->is_string()) {
    const std::string sv = sv_it->get<std::string>();
    const size_t dot = sv.find('.');
    const std::string major_str = (dot == std::string::npos) ? sv : sv.substr(0, dot);
    const std::string minor_str = (dot == std::string::npos) ? std::string("0") : sv.substr(dot + 1);
    auto parse_part = [](const std::string& s, int64_t* out) -> bool {
      if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
      try {
        *out = std::stoll(s);
      } catch (const std::exception&) {
        return false;
      }
      return true;
    };
    if (dot != std::string::npos && minor_str.find('.') != std::string::npos) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "manifest: 'schema_version' must be a \"<major>.<minor>\" string.");
    }
    if (!parse_part(major_str, &major) || !parse_part(minor_str, &minor)) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "manifest: 'schema_version' must be a \"<major>.<minor>\" string.");
    }
  } else if (sv_it->is_number_integer() || sv_it->is_number_unsigned()) {
    major = sv_it->get<int64_t>();
    minor = 0;
  } else {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "manifest: 'schema_version' must be a \"<major>.<minor>\" string.");
  }

  if (major < kMinSupportedSchemaMajor || major > kMaxSupportedSchemaMajor) {
    std::string supported = (kMinSupportedSchemaMajor == kMaxSupportedSchemaMajor)
                                ? std::to_string(kMinSupportedSchemaMajor)
                                : std::to_string(kMinSupportedSchemaMajor) + "-" +
                                      std::to_string(kMaxSupportedSchemaMajor);
    return MakeStatus(MODEL_PACKAGE_ERR_VERSION,
                      "manifest: schema_version major " + std::to_string(major) +
                          " is not supported (this build supports major " + supported + ").");
  }
  pkg->schema_version_major = major;
  pkg->schema_version_minor = minor;

  // A package authored at a newer minor than this build knows may carry optional fields this
  // build does not recognize. Those are additive and must be tolerated rather than rejected,
  // so relax unknown-field strictness for a newer minor.
  if (minor > kMaxKnownSchemaMinor) {
    pkg->strict_unknown_fields = false;
  }
  return nullptr;
}

ModelPackageStatus* PopulatePackageMetadata(ModelPackage* pkg) {
  auto stropt = [&](const char* key, std::optional<std::string>* dst) -> ModelPackageStatus* {
    auto it = pkg->manifest.find(key);
    if (it == pkg->manifest.end()) {
      dst->reset();
      return nullptr;
    }
    if (!it->is_string()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        std::string("manifest: '") + key + "' must be a string.");
    }
    *dst = it->get<std::string>();
    return nullptr;
  };
  if (auto* s = stropt(kPackageNameKey, &pkg->package_name_cache)) return s;
  if (auto* s = stropt(kPackageVersionKey, &pkg->package_version_cache)) return s;
  if (auto* s = stropt(kDescriptionKey, &pkg->description_cache)) return s;

  // layout: default "portable"
  auto layout_it = pkg->manifest.find(kLayoutKey);
  if (layout_it != pkg->manifest.end()) {
    if (!layout_it->is_string()) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA, "manifest: 'layout' must be a string.");
    }
    pkg->layout = layout_it->get<std::string>();
    if (pkg->layout != "portable" && pkg->layout != "installed") {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        "manifest: 'layout' must be 'portable' or 'installed'.");
    }
  } else {
    pkg->layout = "portable";
  }
  pkg->layout_cache = pkg->layout;

  // additional_metadata: serialize as JSON string if present.
  auto am_it = pkg->manifest.find(kAdditionalMetadataKey);
  if (am_it != pkg->manifest.end()) {
    pkg->additional_metadata_cache = am_it->dump();
  } else {
    pkg->additional_metadata_cache.reset();
  }
  return nullptr;
}

}  // namespace

PathResolverOptions PathOptionsFor(const ModelPackage* pkg) {
  PathResolverOptions o;
  o.follow_symlinks = pkg->follow_symlinks;
  o.allow_external_paths = pkg->allow_external_paths || (pkg->layout == "installed");
  return o;
}

ModelPackageStatus* ParseVariantBody(const fs::path& component_dir,
                                     const fs::path& package_root,
                                     const PathResolverOptions& opts,
                                     bool strict,
                                     const std::string& variant_name,
                                     const ordered_json& variant_body,
                                     VariantRecord* out) {
  return ParseVariant(component_dir, package_root, opts, strict, variant_name, variant_body, out);
}

ModelPackageStatus* ParseComponentBody(const fs::path& package_root,
                                       const PathResolverOptions& opts,
                                       bool strict,
                                       const std::string& component_name,
                                       const ordered_json& body,
                                       const fs::path& component_dir,
                                       ComponentRecord* out) {
  return ParseComponent(package_root, opts, strict, component_name, body, component_dir, out);
}

ModelPackageStatus* RefreshPackageMetadata(ModelPackage* pkg) {
  pkg->package_name_cache.reset();
  pkg->package_version_cache.reset();
  pkg->description_cache.reset();
  pkg->additional_metadata_cache.reset();
  return PopulatePackageMetadata(pkg);
}

ModelPackageStatus* RefreshSharedAssets(ModelPackage* pkg, const PathResolverOptions& opts) {
  pkg->shared_assets.clear();
  pkg->shared_asset_index_by_uri.clear();
  return LoadSharedAssets(pkg, opts);
}

namespace {

ModelPackageStatus* ResolveExecutorInfoEntry(const ModelPackage* pkg,
                                             const VariantRecord& var,
                                             const std::string& ns,
                                             const ordered_json& entry,
                                             bool strict_missing_external,
                                             std::string* dst_json) {
  if (entry.is_object()) {
    *dst_json = entry.dump();
    return nullptr;
  }
  if (entry.is_string()) {
    if (!var.resolved_directory.has_value()) {
      if (!strict_missing_external) {
        dst_json->clear();
        return nullptr;
      }
      return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                        "variant '" + var.name + "': executor_info['" + ns +
                            "'] points at an external file but the variant has no "
                            "resolved variant_directory to anchor it.");
    }
    PathResolverOptions opts = PathOptionsFor(pkg);
    fs::path resolved;
    if (auto* s = ResolvePath(*var.resolved_directory, pkg->package_root,
                              entry.get<std::string>(), opts,
                              /*must_exist=*/strict_missing_external, &resolved)) {
      if (!strict_missing_external) {
        ModelPackageStatus_Release(s);
        dst_json->clear();
        return nullptr;
      }
      return s;
    }
    std::ifstream f(resolved, std::ios::binary);
    if (!f) {
      if (!strict_missing_external) {
        dst_json->clear();
        return nullptr;
      }
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "Cannot open executor_info file: '" + resolved.string() + "'.");
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    std::string contents = buf.str();
    try {
      auto _ = ordered_json::parse(contents);
      (void)_;
    } catch (const std::exception& e) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        std::string("Failed to parse executor_info JSON at '") +
                            resolved.string() + "': " + e.what());
    }
    *dst_json = std::move(contents);
    return nullptr;
  }
  return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                    "variant '" + var.name + "': executor_info['" + ns +
                        "'] must be a string or object.");
}

}  // namespace

ModelPackageStatus* RefreshExecutorInfoCache(ModelPackage* pkg, bool strict_missing_external) {
  for (auto& comp : pkg->components) {
    for (auto& vp : comp->variants) {
      VariantRecord& var = *vp;
      var.executor_info_resolved.clear();
      auto ei_it = var.body.find("executor_info");
      if (ei_it == var.body.end() || !ei_it->is_object()) continue;
      var.executor_info_resolved.reserve(ei_it->size());
      for (auto e = ei_it->begin(); e != ei_it->end(); ++e) {
        std::string body_json;
        if (auto* s = ResolveExecutorInfoEntry(pkg, var, e.key(), e.value(),
                                               strict_missing_external, &body_json)) {
          return s;
        }
        var.executor_info_resolved.emplace_back(e.key(), std::move(body_json));
      }
    }
  }
  return nullptr;
}

ModelPackageStatus* ParsePackage(const fs::path& package_root,
                                 const ModelPackageOpenOptions& opts,
                                 ModelPackage* pkg) {
  std::error_code ec;
  if (!fs::exists(package_root, ec) || !fs::is_directory(package_root, ec)) {
    return MakeStatus(MODEL_PACKAGE_ERR_IO,
                      "package_root '" + package_root.string() + "' is not a directory.");
  }
  pkg->package_root = fs::canonical(package_root, ec);
  if (ec) pkg->package_root = package_root;
  pkg->allow_external_paths = opts.allow_external_paths;
  pkg->follow_symlinks = opts.follow_symlinks;
  pkg->strict_unknown_fields = opts.strict_unknown_fields;

  fs::path manifest_path = pkg->package_root / kManifestFileName;
  if (auto* s = ParseJsonFile(manifest_path, &pkg->manifest)) return s;
  if (auto* s = ExpectObject(pkg->manifest, "manifest")) return s;

  // Validate the schema version first so an unsupported package fails fast, before any
  // component/asset parsing. May relax pkg->strict_unknown_fields for a newer minor.
  if (auto* s = ParseSchemaVersion(pkg)) return s;

  // Layout pre-read for path-resolver options. Done before strict-unknown
  // check because we need the layout value to decide path-confinement.
  PathResolverOptions presolve_opts;
  presolve_opts.follow_symlinks = opts.follow_symlinks;
  presolve_opts.allow_external_paths = opts.allow_external_paths;
  {
    auto layout_it = pkg->manifest.find(kLayoutKey);
    if (layout_it != pkg->manifest.end() && layout_it->is_string() &&
        layout_it->get<std::string>() == "installed") {
      presolve_opts.allow_external_paths = true;
    }
  }

  if (auto* s = CheckUnknownFields(pkg->manifest, kManifestKnownKeys, "manifest",
                                   pkg->strict_unknown_fields))
    return s;

  // Components.
  auto comps_it = pkg->manifest.find(kComponentsKey);
  if (comps_it == pkg->manifest.end()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "manifest: missing required 'components' object.");
  }
  if (!comps_it->is_object()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA, "manifest: 'components' must be an object.");
  }
  for (auto e = comps_it->begin(); e != comps_it->end(); ++e) {
    std::unique_ptr<ComponentRecord> rec;
    if (auto* s = LoadComponentForEntry(pkg->package_root, pkg->package_root,
                                        presolve_opts, pkg->strict_unknown_fields,
                                        e.key(), e.value(), &rec)) {
      return s;
    }
    pkg->component_index_by_name.emplace(rec->name, pkg->components.size());
    pkg->components.push_back(std::move(rec));
  }

  if (auto* s = LoadSharedAssets(pkg, presolve_opts)) return s;
  if (auto* s = PopulatePackageMetadata(pkg)) return s;
  if (auto* s = RefreshExecutorInfoCache(pkg, /*strict_missing_external=*/true)) return s;

  return nullptr;
}

}  // namespace model_package
