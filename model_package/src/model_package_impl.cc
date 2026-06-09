// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_impl.cc
/// \brief Implementation of the public C API declared in model_package.h.

#include "model_package.h"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <unordered_map>

#include "manifest_parser.h"
#include "model_package_impl.h"
#include "path_resolver.h"
#include "status_impl.h"

namespace mp = model_package_v2;
using model_package::MakeStatus;

namespace {

ModelPackageStatus* NullArg(const char* name) {
  return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                    std::string("model_package: '") + name + "' must not be null.");
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// View cache helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace model_package_v2 {

// Per-package view cache. We store it inside the ModelPackage struct via a
// pImpl-style side map: the ModelPackage struct itself doesn't carry the cache
// to avoid forcing every translation unit to include <map>. For Phase 1 we
// keep it simple and just thread a per-package unique_ptr through a static
// helper. Since each call needs the cache, we store it on the package.

struct PackageViewCache {
  std::vector<std::unique_ptr<ModelComponent>> component_views;
  std::vector<std::vector<std::unique_ptr<ModelVariant>>> variant_views;
};

namespace {

// Use a single side-map keyed by package pointer so we don't have to extend
// the public ModelPackage struct in this phase. Single-threaded model in
// Phase 1 (per the API thread-safety contract: const calls are safe but no
// internal locking).
std::unordered_map<const ModelPackage*, std::unique_ptr<PackageViewCache>> g_view_caches;

PackageViewCache& EnsureCache(const ModelPackage* pkg) {
  auto it = g_view_caches.find(pkg);
  if (it != g_view_caches.end()) return *it->second;
  auto cache = std::make_unique<PackageViewCache>();
  cache->component_views.reserve(pkg->components.size());
  cache->variant_views.resize(pkg->components.size());
  for (size_t ci = 0; ci < pkg->components.size(); ++ci) {
    auto cv = std::make_unique<ModelComponent>();
    cv->owner = const_cast<ModelPackage*>(pkg);
    cv->component_idx = ci;
    cv->record = pkg->components[ci].get();
    cache->component_views.push_back(std::move(cv));
    cache->variant_views[ci].reserve(pkg->components[ci]->variants.size());
    for (size_t vi = 0; vi < pkg->components[ci]->variants.size(); ++vi) {
      auto vv = std::make_unique<ModelVariant>();
      vv->owner = const_cast<ModelPackage*>(pkg);
      vv->component_idx = ci;
      vv->variant_idx = vi;
      vv->component_record = pkg->components[ci].get();
      vv->record = pkg->components[ci]->variants[vi].get();
      cache->variant_views[ci].push_back(std::move(vv));
    }
  }
  auto* raw = cache.get();
  g_view_caches.emplace(pkg, std::move(cache));
  return *raw;
}

void DropCache(const ModelPackage* pkg) {
  g_view_caches.erase(pkg);
}

}  // namespace

}  // namespace model_package_v2

// ─────────────────────────────────────────────────────────────────────────────
// Status helpers
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

const char* ModelPackageStatus_Message(const ModelPackageStatus* s) {
  return ModelPackage_GetErrorMessage(s);
}
ModelPackageErrorCode ModelPackageStatus_Code(const ModelPackageStatus* s) {
  return ModelPackage_GetErrorCode(s);
}
void ModelPackageStatus_Release(ModelPackageStatus* s) {
  ModelPackage_ReleaseStatus(s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* ModelPackage_Open(const char* package_root,
                                      const ModelPackageOpenOptions* opts,
                                      ModelPackage** out) {
  if (!package_root) return NullArg("package_root");
  if (!out) return NullArg("out");
  *out = nullptr;

  ModelPackageOpenOptions effective{};
  effective.struct_size = sizeof(ModelPackageOpenOptions);
  effective.abi_version = 1;
  effective.allow_external_paths = false;
  effective.follow_symlinks = true;
  effective.strict_unknown_fields = true;
  if (opts) {
    // Honor only the fields up to the caller's struct_size.
    if (opts->struct_size >= sizeof(ModelPackageOpenOptions)) {
      effective = *opts;
    } else {
      // Copy by member with bounds-checking against struct_size.
      const char* base = reinterpret_cast<const char*>(opts);
      auto copy_if_fits = [&](size_t offset, size_t size, void* dst) {
        if (offset + size <= opts->struct_size) std::memcpy(dst, base + offset, size);
      };
      copy_if_fits(offsetof(ModelPackageOpenOptions, abi_version),
                   sizeof(int), &effective.abi_version);
      copy_if_fits(offsetof(ModelPackageOpenOptions, allow_external_paths),
                   sizeof(bool), &effective.allow_external_paths);
      copy_if_fits(offsetof(ModelPackageOpenOptions, follow_symlinks),
                   sizeof(bool), &effective.follow_symlinks);
      copy_if_fits(offsetof(ModelPackageOpenOptions, strict_unknown_fields),
                   sizeof(bool), &effective.strict_unknown_fields);
    }
  }

  auto pkg = std::make_unique<ModelPackage>();
  if (auto* s = mp::ParsePackage(std::filesystem::path(package_root), effective, pkg.get())) {
    return s;
  }
  *out = pkg.release();
  return nullptr;
}

ModelPackageStatus* ModelPackage_New(ModelPackage** out) {
  if (!out) return NullArg("out");
  return MakeStatus(MODEL_PACKAGE_ERR_STATE,
                    "ModelPackage_New is not yet implemented (Phase 3).");
}

void ModelPackage_Close(ModelPackage* pkg) {
  if (!pkg) return;
  mp::DropCache(pkg);
  delete pkg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Package-level inspection
// ─────────────────────────────────────────────────────────────────────────────

const ModelPackageInfo* ModelPackage_Info(const ModelPackage* pkg) {
  if (!pkg) return nullptr;
  return &pkg->info_view;
}

const ModelComponent* ModelPackage_GetComponent(const ModelPackage* pkg, size_t idx) {
  if (!pkg || idx >= pkg->components.size()) return nullptr;
  return mp::EnsureCache(pkg).component_views[idx].get();
}

const ModelComponent* ModelPackage_FindComponent(const ModelPackage* pkg, const char* name) {
  if (!pkg || !name) return nullptr;
  auto it = pkg->component_index_by_name.find(name);
  if (it == pkg->component_index_by_name.end()) return nullptr;
  return ModelPackage_GetComponent(pkg, it->second);
}

const char* ModelComponent_Name(const ModelComponent* c) {
  if (!c) return nullptr;
  return c->record->name_cache.c_str();
}

size_t ModelComponent_VariantCount(const ModelComponent* c) {
  if (!c) return 0;
  return c->record->variants.size();
}

const ModelVariant* ModelComponent_GetVariant(const ModelComponent* c, size_t idx) {
  if (!c || idx >= c->record->variants.size()) return nullptr;
  return mp::EnsureCache(c->owner).variant_views[c->component_idx][idx].get();
}

const ModelVariant* ModelComponent_FindVariant(const ModelComponent* c, const char* name) {
  if (!c || !name) return nullptr;
  for (size_t i = 0; i < c->record->variants.size(); ++i) {
    if (c->record->variants[i]->name == name) {
      return ModelComponent_GetVariant(c, i);
    }
  }
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant accessors
// ─────────────────────────────────────────────────────────────────────────────

const char* ModelVariant_Name(const ModelVariant* v) {
  if (!v) return nullptr;
  return v->record->name_cache.c_str();
}

static const char* OptStr(const std::optional<std::string>& s) {
  return s.has_value() ? s->c_str() : nullptr;
}

const char* ModelVariant_EpName(const ModelVariant* v) {
  return v ? OptStr(v->record->ep_cache) : nullptr;
}
const char* ModelVariant_Device(const ModelVariant* v) {
  return v ? OptStr(v->record->device_cache) : nullptr;
}
const char* ModelVariant_CompatibilityString(const ModelVariant* v) {
  return v ? OptStr(v->record->compatibility_string_cache) : nullptr;
}

ModelPackageStatus* ModelVariant_ResolveDirectoryPath(const ModelVariant* v,
                                                      const char** out_path) {
  if (!v) return NullArg("variant");
  if (!out_path) return NullArg("out_path");
  if (!v->record->resolved_directory.has_value()) {
    return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                      "variant '" + v->record->name + "' has no resolvable variant_directory.");
  }
  *out_path = v->record->resolved_directory_cache.value().c_str();
  return nullptr;
}

ModelPackageStatus* ModelVariant_GetExecutorInfoJson(const ModelVariant* v,
                                                     const char* namespace_,
                                                     const char** out_json) {
  if (!v) return NullArg("variant");
  if (!namespace_) return NullArg("namespace_");
  if (!out_json) return NullArg("out_json");
  *out_json = nullptr;

  auto ei_it = v->record->body.find("executor_info");
  if (ei_it == v->record->body.end()) return nullptr;
  auto entry = ei_it->find(namespace_);
  if (entry == ei_it->end()) return nullptr;

  std::string cached;
  if (entry->is_object()) {
    cached = entry->dump();
  } else if (entry->is_string()) {
    // Resolve the file against variant_directory and load contents as JSON text.
    if (!v->record->resolved_directory.has_value()) {
      return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                        "variant '" + v->record->name + "' has no variant_directory for "
                        "external executor_info file.");
    }
    mp::PathResolverOptions opts;
    opts.allow_external_paths = v->owner->allow_external_paths;
    opts.follow_symlinks = v->owner->follow_symlinks;
    std::filesystem::path resolved;
    if (auto* s = mp::ResolvePath(*v->record->resolved_directory,
                                  v->owner->package_root,
                                  entry->get<std::string>(),
                                  opts, /*must_exist=*/true, &resolved)) {
      return s;
    }
    std::ifstream f(resolved, std::ios::binary);
    if (!f) {
      return MakeStatus(MODEL_PACKAGE_ERR_IO,
                        "Cannot open executor_info file: '" + resolved.string() + "'.");
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    cached = buf.str();
    // Validate as JSON for callers' sanity.
    try {
      auto _ = mp::ordered_json::parse(cached);
      (void)_;
    } catch (const std::exception& e) {
      return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                        std::string("Failed to parse executor_info JSON at '") +
                            resolved.string() + "': " + e.what());
    }
  } else {
    return MakeStatus(MODEL_PACKAGE_ERR_SCHEMA,
                      "variant '" + v->record->name + "': executor_info entry must be string or object.");
  }
  auto& slot = v->record->executor_info_json_cache[namespace_];
  slot = std::move(cached);
  *out_json = slot.c_str();
  return nullptr;
}

size_t ModelVariant_UsedAssetCount(const ModelVariant* v) {
  return v ? v->record->used_asset_uri_caches.size() : 0;
}
const char* ModelVariant_UsedAssetUri(const ModelVariant* v, size_t idx) {
  if (!v || idx >= v->record->used_asset_uri_caches.size()) return nullptr;
  return v->record->used_asset_uri_caches[idx].c_str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared assets
// ─────────────────────────────────────────────────────────────────────────────

const ModelSharedAsset* ModelPackage_GetSharedAsset(const ModelPackage* pkg, size_t idx) {
  if (!pkg || idx >= pkg->shared_assets.size()) return nullptr;
  return &pkg->shared_assets[idx]->abi_view;
}

ModelPackageStatus* ModelPackage_ResolveAssetUri(const ModelPackage* pkg,
                                                 const char* uri,
                                                 const char** out_path) {
  if (!pkg) return NullArg("pkg");
  if (!uri) return NullArg("uri");
  if (!out_path) return NullArg("out_path");
  *out_path = nullptr;
  auto it = pkg->shared_asset_index_by_uri.find(uri);
  if (it == pkg->shared_asset_index_by_uri.end()) {
    return MakeStatus(MODEL_PACKAGE_ERR_ASSET_MISSING,
                      std::string("Asset URI not declared in this package: '") + uri + "'.");
  }
  *out_path = pkg->shared_assets[it->second]->resolved_path_cache.c_str();
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Round-trip JSON getters and additional_metadata accessors
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* ModelPackage_GetComponentJson(const ModelPackage* pkg,
                                                  const char* component_name,
                                                  const char** out_json) {
  if (!pkg) return NullArg("pkg");
  if (!component_name) return NullArg("component_name");
  if (!out_json) return NullArg("out_json");
  *out_json = nullptr;
  auto it = pkg->component_index_by_name.find(component_name);
  if (it == pkg->component_index_by_name.end()) {
    return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                      std::string("component '") + component_name + "' not found.");
  }
  auto& rec = pkg->components[it->second];
  if (!rec->component_json_cache.has_value()) {
    rec->component_json_cache = rec->body.dump();
  }
  *out_json = rec->component_json_cache->c_str();
  return nullptr;
}

ModelPackageStatus* ModelPackage_GetVariantJson(const ModelPackage* pkg,
                                                const char* component_name,
                                                const char* variant_name,
                                                const char** out_json) {
  if (!pkg) return NullArg("pkg");
  if (!component_name) return NullArg("component_name");
  if (!variant_name) return NullArg("variant_name");
  if (!out_json) return NullArg("out_json");
  *out_json = nullptr;
  auto it = pkg->component_index_by_name.find(component_name);
  if (it == pkg->component_index_by_name.end()) {
    return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                      std::string("component '") + component_name + "' not found.");
  }
  auto& comp = pkg->components[it->second];
  for (auto& var : comp->variants) {
    if (var->name == variant_name) {
      if (!var->variant_json_cache.has_value()) {
        var->variant_json_cache = var->body.dump();
      }
      *out_json = var->variant_json_cache->c_str();
      return nullptr;
    }
  }
  return MakeStatus(MODEL_PACKAGE_ERR_NOT_FOUND,
                    std::string("variant '") + variant_name + "' not found in component '" +
                        component_name + "'.");
}

static const char* CachedAdditionalMetadata(const mp::ordered_json& body,
                                            std::optional<std::string>& cache) {
  auto it = body.find("additional_metadata");
  if (it == body.end()) return nullptr;
  if (!cache.has_value()) cache = it->dump();
  return cache->c_str();
}

const char* ModelPackage_AdditionalMetadataJson(const ModelPackage* pkg) {
  if (!pkg) return nullptr;
  return CachedAdditionalMetadata(pkg->manifest, pkg->additional_metadata_cache);
}
const char* ModelComponent_AdditionalMetadataJson(const ModelComponent* c) {
  if (!c) return nullptr;
  return CachedAdditionalMetadata(c->record->body, c->record->additional_metadata_cache);
}
const char* ModelVariant_AdditionalMetadataJson(const ModelVariant* v) {
  if (!v) return nullptr;
  return CachedAdditionalMetadata(v->record->body, v->record->additional_metadata_cache);
}

}  // extern "C"
