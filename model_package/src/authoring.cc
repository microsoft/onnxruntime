// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file authoring.cc
/// \brief Mutation (authoring) API implementation.

#include "model_package.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "asset_hasher.h"
#include "manifest_parser.h"
#include "model_package_impl.h"
#include "path_resolver.h"
#include "status_impl.h"

namespace fs = std::filesystem;
namespace mp = model_package;
using model_package::MakeStatus;
using nlohmann::ordered_json;

namespace {

// Schema version stamped into newly authored packages, written as a "<major>.<minor>"
// string. Keep in sync with the parser's supported major + highest known minor
// (manifest_parser.cc: kMaxSupportedSchemaMajor / kMaxKnownSchemaMinor).
constexpr const char* kAuthoredSchemaVersion = "1.0";

ModelPackageStatus* NullArg(const char* name) {
  return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                    std::string("model_package: '") + name + "' must not be null.");
}

ModelPackageStatus* ParseJsonString(const char* json, const char* where, ordered_json* out) {
  try {
    *out = ordered_json::parse(json);
  } catch (const ordered_json::parse_error& e) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      std::string(where) + ": JSON parse error: " + e.what());
  }
  return nullptr;
}

ModelPackageStatus* ExpectObject(const ordered_json& j, const char* where) {
  if (!j.is_object()) {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      std::string(where) + ": expected a JSON object.");
  }
  return nullptr;
}

void RebuildComponentIndex(ModelPackage* pkg) {
  pkg->component_index_by_name.clear();
  for (size_t i = 0; i < pkg->components.size(); ++i) {
    pkg->component_index_by_name[pkg->components[i]->name] = i;
  }
}

mp::ComponentRecord* FindComponentRecord(ModelPackage* pkg, const std::string& name) {
  auto it = pkg->component_index_by_name.find(name);
  if (it == pkg->component_index_by_name.end()) return nullptr;
  return pkg->components[it->second].get();
}

mp::VariantRecord* FindVariantRecord(mp::ComponentRecord* comp, const std::string& name) {
  for (auto& v : comp->variants) {
    if (v->name == name) return v.get();
  }
  return nullptr;
}

ModelPackageStatus* RefreshSharedAssetsHelper(ModelPackage* pkg) {
  return mp::RefreshSharedAssets(pkg, mp::PathOptionsFor(pkg));
}

ModelPackageStatus* PostMutate(ModelPackage* pkg, bool refresh_assets = true) {
  mp::DropViewCache(pkg);
  if (refresh_assets) {
    if (auto* s = RefreshSharedAssetsHelper(pkg)) return s;
  }
  if (auto* s = mp::RefreshPackageMetadata(pkg)) return s;
  return mp::RefreshExecutorInfoCache(pkg, /*strict_missing_external=*/false);
}

ordered_json& EnsureManifestComponentsObject(ModelPackage* pkg) {
  if (!pkg->manifest.contains("components")) {
    pkg->manifest["components"] = ordered_json::object();
  }
  return pkg->manifest["components"];
}

}  // namespace

extern "C" {

// ─────────────────────────────────────────────────────────────────────────────
// ModelPackage_New
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* ModelPackage_New(ModelPackage** out) {
  if (!out) return NullArg("out");
  auto pkg = std::make_unique<ModelPackage>();
  pkg->manifest = ordered_json::object();
  // Authored at this build's schema version, written as a "<major>.<minor>" string.
  pkg->manifest["schema_version"] = kAuthoredSchemaVersion;
  pkg->manifest["layout"] = "portable";
  pkg->manifest["components"] = ordered_json::object();
  pkg->layout = "portable";
  pkg->strict_unknown_fields = true;
  pkg->follow_symlinks = true;
  pkg->allow_external_paths = false;
  pkg->package_root = fs::path();
  if (auto* s = mp::RefreshPackageMetadata(pkg.get())) return s;
  *out = pkg.release();
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Components
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* ModelPackage_SetComponentInline(ModelPackage* pkg,
                                                    const char* name,
                                                    const char* component_json) {
  if (!pkg) return NullArg("pkg");
  if (!name) return NullArg("name");
  if (!component_json) return NullArg("component_json");

  ordered_json body;
  if (auto* s = ParseJsonString(component_json,
                                ("component '" + std::string(name) + "'").c_str(), &body)) return s;
  if (auto* s = ExpectObject(body, ("component '" + std::string(name) + "'").c_str())) return s;

  auto opts = mp::PathOptionsFor(pkg);
  auto rec = std::make_unique<mp::ComponentRecord>();
  rec->storage = mp::ComponentStorage::kInline;
  rec->component_dir = pkg->package_root;
  if (auto* s = mp::ParseComponentBody(pkg->package_root, opts, pkg->strict_unknown_fields,
                                       name, body, pkg->package_root, rec.get())) return s;

  EnsureManifestComponentsObject(pkg)[name] = body;

  if (auto* existing = FindComponentRecord(pkg, name)) {
    size_t idx = pkg->component_index_by_name[name];
    mp::RecordOrphanComponent(pkg, *pkg->components[idx]);
    pkg->components[idx] = std::move(rec);
  } else {
    pkg->components.push_back(std::move(rec));
  }
  RebuildComponentIndex(pkg);
  return PostMutate(pkg);
}

ModelPackageStatus* ModelPackage_SetComponentExternal(ModelPackage* pkg,
                                                      const char* name,
                                                      const char* path) {
  if (!pkg) return NullArg("pkg");
  if (!name) return NullArg("name");
  if (!path) return NullArg("path");
  if (pkg->package_root.empty()) {
    return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                      "SetComponentExternal requires a package_root (use _Open or _Commit "
                      "with a dest_root first; or rely on _Commit(dest_root) to materialize).");
  }

  auto opts = mp::PathOptionsFor(pkg);
  fs::path resolved;
  // Allow the file/dir to not exist yet (we'll initialize empty).
  if (auto* s = mp::ResolvePath(pkg->package_root, pkg->package_root, path, opts,
                                /*must_exist=*/false, &resolved)) return s;
  std::error_code ec;
  fs::path component_dir;
  fs::path file_path;
  if (fs::exists(resolved, ec) && fs::is_directory(resolved, ec)) {
    file_path = resolved / "component.json";
    component_dir = resolved;
  } else {
    file_path = resolved;
    component_dir = resolved.parent_path();
  }
  ordered_json body;
  if (fs::exists(file_path, ec)) {
    std::ifstream f(file_path, std::ios::binary);
    std::ostringstream buf;
    buf << f.rdbuf();
    if (auto* s = ParseJsonString(buf.str().c_str(),
                                  ("component '" + std::string(name) + "'").c_str(), &body)) return s;
  } else {
    body = ordered_json::object();
    body["variants"] = ordered_json::object();
  }
  if (auto* s = ExpectObject(body, ("component '" + std::string(name) + "'").c_str())) return s;

  auto rec = std::make_unique<mp::ComponentRecord>();
  rec->storage = mp::ComponentStorage::kExternal;
  rec->external_path = file_path;
  rec->component_dir = component_dir;
  if (auto* s = mp::ParseComponentBody(pkg->package_root, opts, pkg->strict_unknown_fields,
                                       name, body, component_dir, rec.get())) return s;

  EnsureManifestComponentsObject(pkg)[name] = std::string(path);

  if (FindComponentRecord(pkg, name)) {
    size_t idx = pkg->component_index_by_name[name];
    mp::RecordOrphanComponent(pkg, *pkg->components[idx]);
    pkg->components[idx] = std::move(rec);
  } else {
    pkg->components.push_back(std::move(rec));
  }
  RebuildComponentIndex(pkg);
  return PostMutate(pkg);
}

ModelPackageStatus* ModelPackage_RemoveComponent(ModelPackage* pkg, const char* name) {
  if (!pkg) return NullArg("pkg");
  if (!name) return NullArg("name");
  auto it = pkg->component_index_by_name.find(name);
  if (it == pkg->component_index_by_name.end()) return nullptr;
  size_t idx = it->second;
  mp::RecordOrphanComponent(pkg, *pkg->components[idx]);
  pkg->components.erase(pkg->components.begin() + idx);
  auto comps_it = pkg->manifest.find("components");
  if (comps_it != pkg->manifest.end() && comps_it->is_object()) {
    comps_it->erase(name);
  }
  RebuildComponentIndex(pkg);
  return PostMutate(pkg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Variants
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* ModelPackage_SetVariant(ModelPackage* pkg,
                                            const char* component_name,
                                            const char* variant_name,
                                            const char* variant_json) {
  if (!pkg) return NullArg("pkg");
  if (!component_name) return NullArg("component_name");
  if (!variant_name) return NullArg("variant_name");
  if (!variant_json) return NullArg("variant_json");
  auto* comp = FindComponentRecord(pkg, component_name);
  if (!comp) {
    return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                      std::string("SetVariant: component '") + component_name + "' not found.");
  }
  ordered_json body;
  if (auto* s = ParseJsonString(variant_json,
                                ("variant '" + std::string(variant_name) + "'").c_str(), &body)) return s;

  auto vr = std::make_unique<mp::VariantRecord>();
  auto opts = mp::PathOptionsFor(pkg);
  if (auto* s = mp::ParseVariantBody(comp->component_dir, pkg->package_root, opts,
                                     pkg->strict_unknown_fields,
                                     variant_name, body, vr.get())) return s;

  // Update component.body["variants"][variant_name]
  if (!comp->body.contains("variants") || !comp->body["variants"].is_object()) {
    comp->body["variants"] = ordered_json::object();
  }
  comp->body["variants"][variant_name] = body;
  // If component is inline, mirror into manifest.
  if (comp->storage == mp::ComponentStorage::kInline) {
    pkg->manifest["components"][comp->name] = comp->body;
  }
  // Replace or append.
  bool replaced = false;
  for (auto& v : comp->variants) {
    if (v->name == variant_name) {
      mp::RecordOrphanVariantDir(pkg, *v);
      v = std::move(vr);
      replaced = true;
      break;
    }
  }
  if (!replaced) comp->variants.push_back(std::move(vr));

  // Invalidate cached component JSON.
  comp->component_json_cache.reset();
  return PostMutate(pkg);
}

ModelPackageStatus* ModelPackage_RemoveVariant(ModelPackage* pkg,
                                               const char* component_name,
                                               const char* variant_name) {
  if (!pkg) return NullArg("pkg");
  if (!component_name) return NullArg("component_name");
  if (!variant_name) return NullArg("variant_name");
  auto* comp = FindComponentRecord(pkg, component_name);
  if (!comp) return nullptr;
  auto pred = [&](const std::unique_ptr<mp::VariantRecord>& v) {
    if (v->name == variant_name) {
      mp::RecordOrphanVariantDir(pkg, *v);
      return true;
    }
    return false;
  };
  comp->variants.erase(std::remove_if(comp->variants.begin(), comp->variants.end(), pred),
                       comp->variants.end());
  if (comp->body.contains("variants") && comp->body["variants"].is_object()) {
    comp->body["variants"].erase(variant_name);
  }
  if (comp->storage == mp::ComponentStorage::kInline) {
    pkg->manifest["components"][comp->name] = comp->body;
  }
  comp->component_json_cache.reset();
  return PostMutate(pkg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant executor_info
// ─────────────────────────────────────────────────────────────────────────────

namespace {

ModelPackageStatus* ReparseVariantInPlace(ModelPackage* pkg,
                                          mp::ComponentRecord* comp,
                                          mp::VariantRecord* var) {
  auto opts = mp::PathOptionsFor(pkg);
  auto rebuilt = std::make_unique<mp::VariantRecord>();
  if (auto* s = mp::ParseVariantBody(comp->component_dir, pkg->package_root, opts,
                                     pkg->strict_unknown_fields,
                                     var->name, var->body, rebuilt.get())) return s;
  *var = std::move(*rebuilt);
  return nullptr;
}

ModelPackageStatus* MutateExecutorInfo(ModelPackage* pkg,
                                       const char* component,
                                       const char* variant,
                                       const char* namespace_,
                                       const ordered_json* new_value /* null = remove */) {
  if (!pkg) return NullArg("pkg");
  if (!component) return NullArg("component");
  if (!variant) return NullArg("variant");
  if (!namespace_) return NullArg("namespace");
  auto* comp = FindComponentRecord(pkg, component);
  if (!comp) {
    return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                      std::string("component '") + component + "' not found.");
  }
  auto* var = FindVariantRecord(comp, variant);
  if (!var) {
    return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                      std::string("variant '") + variant + "' not found in component '" +
                          component + "'.");
  }
  if (!var->body.contains("executor_info") || !var->body["executor_info"].is_object()) {
    if (!new_value) return nullptr;  // remove on absent -> nothing to do
    var->body["executor_info"] = ordered_json::object();
  }
  if (new_value) {
    var->body["executor_info"][namespace_] = *new_value;
  } else {
    var->body["executor_info"].erase(namespace_);
    if (var->body["executor_info"].empty()) {
      var->body.erase("executor_info");
    }
  }
  comp->body["variants"][var->name] = var->body;
  if (comp->storage == mp::ComponentStorage::kInline) {
    pkg->manifest["components"][comp->name] = comp->body;
  }
  if (auto* s = ReparseVariantInPlace(pkg, comp, var)) return s;
  comp->component_json_cache.reset();
  return PostMutate(pkg, /*refresh_assets=*/false);
}

}  // namespace

ModelPackageStatus* ModelPackage_SetVariantExecutorInfoInline(ModelPackage* pkg,
                                                              const char* component,
                                                              const char* variant,
                                                              const char* namespace_,
                                                              const char* info_json) {
  if (!info_json) return NullArg("info_json");
  ordered_json body;
  if (auto* s = ParseJsonString(info_json, "executor_info", &body)) return s;
  if (auto* s = ExpectObject(body, "executor_info inline value")) return s;
  return MutateExecutorInfo(pkg, component, variant, namespace_, &body);
}

ModelPackageStatus* ModelPackage_SetVariantExecutorInfoExternal(ModelPackage* pkg,
                                                                const char* component,
                                                                const char* variant,
                                                                const char* namespace_,
                                                                const char* path) {
  if (!path) return NullArg("path");
  ordered_json body = std::string(path);
  return MutateExecutorInfo(pkg, component, variant, namespace_, &body);
}

ModelPackageStatus* ModelPackage_RemoveVariantExecutorInfo(ModelPackage* pkg,
                                                           const char* component,
                                                           const char* variant,
                                                           const char* namespace_) {
  return MutateExecutorInfo(pkg, component, variant, namespace_, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared assets
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* ModelPackage_AddSharedAsset(ModelPackage* pkg,
                                                const char* source_dir,
                                                const char* expected_uri_or_null,
                                                bool copy_in,
                                                const char** out_uri) {
  if (!pkg) return NullArg("pkg");
  if (!source_dir) return NullArg("source_dir");
  if (!out_uri) return NullArg("out_uri");
  *out_uri = nullptr;

  if (!copy_in && pkg->layout == "portable") {
    return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                      "AddSharedAsset: copy_in=false rejected in portable layout (the "
                      "path would point outside <package_root>).");
  }

  std::string computed_uri;
  if (auto* s = mp::ComputeDirectoryAssetUri(fs::path(source_dir), &computed_uri)) return s;
  if (expected_uri_or_null) {
    if (computed_uri != expected_uri_or_null) {
      return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                        std::string("AddSharedAsset: hash mismatch — computed ") +
                            computed_uri + ", expected " + expected_uri_or_null + ".");
    }
  }

  if (!pkg->manifest.contains("shared_assets") || !pkg->manifest["shared_assets"].is_object()) {
    pkg->manifest["shared_assets"] = ordered_json::object();
  }
  if (copy_in) {
    // No manifest entry needed — the asset will be materialized at the default
    // convention path on commit. LoadSharedAssets surfaces the staged source
    // immediately so the URI shows up in ModelPackage_Info() before commit.
    pkg->pending_shared_asset_copies[computed_uri] = fs::path(source_dir);
  } else {
    pkg->manifest["shared_assets"][computed_uri] = std::string(source_dir);
  }

  if (auto* s = PostMutate(pkg)) return s;

  // Look up the record and return its URI. After PostMutate, the URI is
  // always present in shared_assets_index_by_uri (either via the override
  // path or via the pending-copy tier of LoadSharedAssets).
  auto sit = pkg->shared_asset_index_by_uri.find(computed_uri);
  if (sit == pkg->shared_asset_index_by_uri.end()) {
    return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                      std::string("AddSharedAsset: failed to register URI ") + computed_uri);
  }
  *out_uri = pkg->shared_assets[sit->second]->uri_cache.c_str();
  return nullptr;
}

ModelPackageStatus* ModelPackage_RemoveSharedAsset(ModelPackage* pkg, const char* uri) {
  if (!pkg) return NullArg("pkg");
  if (!uri) return NullArg("uri");
  std::string uri_str(uri);
  if (pkg->manifest.contains("shared_assets") && pkg->manifest["shared_assets"].is_object()) {
    pkg->manifest["shared_assets"].erase(uri_str);
    if (pkg->manifest["shared_assets"].empty()) {
      pkg->manifest.erase("shared_assets");
    }
  }
  pkg->pending_shared_asset_copies.erase(uri_str);
  // Physically remove the on-disk directory at the default convention. If it
  // stays on disk, the next RefreshSharedAssets would auto-discover it again
  // and the removal would be a no-op. We only touch paths that live inside
  // package_root.
  if (!pkg->package_root.empty()) {
    std::string dir_name = mp::DefaultSharedAssetDirName(uri_str);
    if (!dir_name.empty()) {
      std::filesystem::path on_disk = pkg->package_root / "shared_assets" / dir_name;
      if (mp::IsInsidePackageRoot(pkg, on_disk)) {
        std::error_code ec;
        std::filesystem::remove_all(on_disk, ec);
      }
    }
  }
  return PostMutate(pkg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Package metadata
// ─────────────────────────────────────────────────────────────────────────────

namespace {

void SetOrClearString(ordered_json* obj, const char* key, const char* value) {
  if (value == nullptr) return;  // leave untouched
  if (value[0] == '\0') {
    obj->erase(key);
  } else {
    (*obj)[key] = std::string(value);
  }
}

}  // namespace

ModelPackageStatus* ModelPackage_SetMetadata(ModelPackage* pkg,
                                             const char* name_or_null,
                                             const char* version_or_null,
                                             const char* description_or_null) {
  if (!pkg) return NullArg("pkg");
  SetOrClearString(&pkg->manifest, "package_name", name_or_null);
  SetOrClearString(&pkg->manifest, "package_version", version_or_null);
  SetOrClearString(&pkg->manifest, "description", description_or_null);
  return PostMutate(pkg, /*refresh_assets=*/false);
}

ModelPackageStatus* ModelPackage_SetLayout(ModelPackage* pkg, const char* layout) {
  if (!pkg) return NullArg("pkg");
  if (!layout) return NullArg("layout");
  std::string l(layout);
  if (l != "portable" && l != "installed") {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "SetLayout: layout must be 'portable' or 'installed'.");
  }
  pkg->manifest["layout"] = l;
  pkg->layout = l;
  return PostMutate(pkg, /*refresh_assets=*/false);
}

ModelPackageStatus* ModelPackage_SetAdditionalMetadataJson(ModelPackage* pkg,
                                                           const char* scope,
                                                           const char* component_or_null,
                                                           const char* variant_or_null,
                                                           const char* json_or_null) {
  if (!pkg) return NullArg("pkg");
  if (!scope) return NullArg("scope");
  std::string s(scope);
  ordered_json* target = nullptr;
  mp::ComponentRecord* comp = nullptr;
  mp::VariantRecord* var = nullptr;
  if (s == "manifest") {
    if (component_or_null || variant_or_null) {
      return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                        "SetAdditionalMetadataJson: 'manifest' scope takes no component/variant.");
    }
    target = &pkg->manifest;
  } else if (s == "component") {
    if (!component_or_null) return NullArg("component");
    if (variant_or_null) {
      return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                        "SetAdditionalMetadataJson: 'component' scope takes no variant.");
    }
    comp = FindComponentRecord(pkg, component_or_null);
    if (!comp) {
      return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                        std::string("component '") + component_or_null + "' not found.");
    }
    target = &comp->body;
  } else if (s == "variant") {
    if (!component_or_null) return NullArg("component");
    if (!variant_or_null) return NullArg("variant");
    comp = FindComponentRecord(pkg, component_or_null);
    if (!comp) {
      return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                        std::string("component '") + component_or_null + "' not found.");
    }
    var = FindVariantRecord(comp, variant_or_null);
    if (!var) {
      return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                        std::string("variant '") + variant_or_null + "' not found.");
    }
    target = &var->body;
  } else {
    return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                      "SetAdditionalMetadataJson: scope must be 'manifest', 'component', or 'variant'.");
  }
  if (json_or_null == nullptr) {
    target->erase("additional_metadata");
  } else {
    ordered_json body;
    if (auto* st = ParseJsonString(json_or_null, "additional_metadata", &body)) return st;
    (*target)["additional_metadata"] = body;
  }
  if (comp && comp->storage == mp::ComponentStorage::kInline) {
    pkg->manifest["components"][comp->name] = comp->body;
  }
  if (comp) comp->component_json_cache.reset();
  if (var) var->additional_metadata_cache.reset();
  if (comp) comp->additional_metadata_cache.reset();
  return PostMutate(pkg, /*refresh_assets=*/false);
}

}  // extern "C"
