// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_impl.cc
/// \brief Implementation of the public C API declared in model_package.h.

#include "model_package.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <string>

#include "asset_hasher.h"
#include "manifest_parser.h"
#include "model_package_impl.h"
#include "path_resolver.h"
#include "status_impl.h"

namespace mp = model_package;
using mp::MakeStatus;

namespace {

ModelPackageStatus* NullArg(const char* name) {
  return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                    std::string("model_package: '") + name + "' must not be null.");
}

const char* OptStr(const std::optional<std::string>& s) {
  return s.has_value() ? s->c_str() : nullptr;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// View cache materialization
// ─────────────────────────────────────────────────────────────────────────────

namespace model_package {

void DropViewCache(ModelPackage* pkg) {
  if (!pkg) return;
  pkg->info_cache.reset();
  for (auto& comp : pkg->components) {
    comp->component_json_cache.reset();
    comp->additional_metadata_cache.reset();
    for (auto& var : comp->variants) {
      var->variant_json_cache.reset();
      var->additional_metadata_cache.reset();
    }
  }
  pkg->additional_metadata_cache.reset();
}

const InfoViewCache& BuildOrGetViewCache(const ModelPackage* pkg) {
  if (pkg->info_cache.has_value()) return *pkg->info_cache;

  pkg->info_cache.emplace();
  auto& cache = *pkg->info_cache;
  const size_t num_components = pkg->components.size();

  cache.executor_infos_storage.resize(num_components);
  cache.variants_storage.resize(num_components);
  cache.components.resize(num_components);

  for (size_t ci = 0; ci < num_components; ++ci) {
    const auto& comp = *pkg->components[ci];
    const size_t num_variants = comp.variants.size();
    cache.executor_infos_storage[ci].clear();
    cache.variants_storage[ci].resize(num_variants);

    // Total executor_info entry count across this component's variants.
    size_t total_execs = 0;
    for (const auto& vp : comp.variants) {
      total_execs += vp->executor_info_resolved.size();
    }
    cache.executor_infos_storage[ci].reserve(total_execs);

    // First pass: append all executor_info entries so storage pointers stay
    // stable for the second pass.
    std::vector<std::pair<size_t, size_t>> ei_ranges(num_variants);

    for (size_t vi = 0; vi < num_variants; ++vi) {
      const auto& var = *comp.variants[vi];
      size_t ei_begin = cache.executor_infos_storage[ci].size();
      // executor_info_resolved is populated eagerly by RefreshExecutorInfoCache
      // (at Open and on every mutation); any parse/IO error surfaces there.
      for (const auto& [ns_str, body_json] : var.executor_info_resolved) {
        ModelExecutorInfoEntry entry{};
        entry.namespace_key = ns_str.c_str();
        entry.json = body_json.c_str();
        cache.executor_infos_storage[ci].push_back(entry);
      }
      ei_ranges[vi] = {ei_begin, cache.executor_infos_storage[ci].size()};
    }

    // Additional metadata strings live in the record-level cache; populate it
    // lazily here as well.
    for (size_t vi = 0; vi < num_variants; ++vi) {
      auto& var = *comp.variants[vi];
      auto am_it = var.body.find("additional_metadata");
      if (am_it != var.body.end() && !var.additional_metadata_cache.has_value()) {
        var.additional_metadata_cache = am_it->dump();
      }
    }
    if (auto am_it = comp.body.find("additional_metadata"); am_it != comp.body.end()) {
      if (!comp.additional_metadata_cache.has_value()) {
        comp.additional_metadata_cache = am_it->dump();
      }
    }

    // Second pass: populate ModelVariantInfo entries pointing at the now-stable
    // storage above.
    for (size_t vi = 0; vi < num_variants; ++vi) {
      const auto& var = *comp.variants[vi];
      ModelVariantInfo& vi_out = cache.variants_storage[ci][vi];
      vi_out = ModelVariantInfo{};
      vi_out.name = var.name_cache.c_str();
      vi_out.variant_directory =
          var.resolved_directory_cache.has_value() ? var.resolved_directory_cache->c_str() : nullptr;
      vi_out.ep = OptStr(var.ep_cache);
      vi_out.device = OptStr(var.device_cache);
      vi_out.compatibility_string = OptStr(var.compatibility_string_cache);
      vi_out.additional_metadata_json = OptStr(var.additional_metadata_cache);
      auto [ei_begin, ei_end] = ei_ranges[vi];
      vi_out.num_executor_infos = ei_end - ei_begin;
      vi_out.executor_infos =
          (vi_out.num_executor_infos > 0) ? &cache.executor_infos_storage[ci][ei_begin] : nullptr;
    }

    ModelComponentInfo& ci_out = cache.components[ci];
    ci_out = ModelComponentInfo{};
    ci_out.name = comp.name_cache.c_str();
    ci_out.additional_metadata_json = OptStr(comp.additional_metadata_cache);
    ci_out.num_variants = num_variants;
    ci_out.variants = num_variants > 0 ? cache.variants_storage[ci].data() : nullptr;
  }

  // Shared assets.
  cache.shared_assets.resize(pkg->shared_assets.size());
  for (size_t i = 0; i < pkg->shared_assets.size(); ++i) {
    const auto& rec = *pkg->shared_assets[i];
    ModelSharedAssetInfo& sa = cache.shared_assets[i];
    sa = ModelSharedAssetInfo{};
    sa.uri = rec.uri_cache.c_str();
    sa.resolved_path = rec.resolved_path_cache.c_str();
  }

  ModelPackageInfo& info = cache.info;
  info = ModelPackageInfo{};
  info.schema_version_major = pkg->schema_version_major;
  info.schema_version_minor = pkg->schema_version_minor;
  info.package_name = OptStr(pkg->package_name_cache);
  info.package_version = OptStr(pkg->package_version_cache);
  info.description = OptStr(pkg->description_cache);
  info.layout = pkg->layout_cache.c_str();
  info.additional_metadata_json = OptStr(pkg->additional_metadata_cache);
  info.num_components = cache.components.size();
  info.components = cache.components.empty() ? nullptr : cache.components.data();
  info.num_shared_assets = cache.shared_assets.size();
  info.shared_assets = cache.shared_assets.empty() ? nullptr : cache.shared_assets.data();

  return cache;
}

}  // namespace model_package

// ─────────────────────────────────────────────────────────────────────────────
// Status helpers
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

const char* ModelPackageStatus_Message(const ModelPackageStatus* s) {
  return s ? s->message.c_str() : nullptr;
}
ModelPackageErrorCode ModelPackageStatus_Code(const ModelPackageStatus* s) {
  return s ? s->code : MODEL_PACKAGE_OK;
}
void ModelPackageStatus_Release(ModelPackageStatus* s) {
  delete s;
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
  effective.allow_external_paths = false;
  effective.follow_symlinks = true;
  effective.strict_unknown_fields = true;
  if (opts) {
    effective.allow_external_paths = opts->allow_external_paths;
    effective.follow_symlinks = opts->follow_symlinks;
    effective.strict_unknown_fields = opts->strict_unknown_fields;
  }

  auto pkg = std::make_unique<ModelPackage>();
  if (auto* s = mp::ParsePackage(std::filesystem::path(package_root), effective, pkg.get())) {
    return s;
  }
  *out = pkg.release();
  return nullptr;
}

void ModelPackage_Close(ModelPackage* pkg) {
  if (!pkg) return;
  mp::DropViewCache(pkg);
  delete pkg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Info tree + convenience lookups
// ─────────────────────────────────────────────────────────────────────────────

const ModelPackageInfo* ModelPackage_Info(const ModelPackage* pkg) {
  if (!pkg) return nullptr;
  return &mp::BuildOrGetViewCache(pkg).info;
}

const ModelComponentInfo* ModelPackage_FindComponent(const ModelPackageInfo* info,
                                                     const char* name) {
  if (!info || !name) return nullptr;
  for (size_t i = 0; i < info->num_components; ++i) {
    if (info->components[i].name && std::strcmp(info->components[i].name, name) == 0) {
      return &info->components[i];
    }
  }
  return nullptr;
}

const ModelVariantInfo* ModelComponentInfo_FindVariant(const ModelComponentInfo* comp,
                                                       const char* name) {
  if (!comp || !name) return nullptr;
  for (size_t i = 0; i < comp->num_variants; ++i) {
    if (comp->variants[i].name && std::strcmp(comp->variants[i].name, name) == 0) {
      return &comp->variants[i];
    }
  }
  return nullptr;
}

const ModelExecutorInfoEntry* ModelVariantInfo_FindExecutorInfo(const ModelVariantInfo* var,
                                                                const char* namespace_key) {
  if (!var || !namespace_key) return nullptr;
  for (size_t i = 0; i < var->num_executor_infos; ++i) {
    if (var->executor_infos[i].namespace_key &&
        std::strcmp(var->executor_infos[i].namespace_key, namespace_key) == 0) {
      return &var->executor_infos[i];
    }
  }
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared assets
// ─────────────────────────────────────────────────────────────────────────────

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

ModelPackageStatus* ModelPackage_ResolveStringRef(const ModelPackage* pkg,
                                                  const char* base_dir,
                                                  const char* input,
                                                  bool must_exist,
                                                  const char** out_path) {
  if (!pkg) return NullArg("pkg");
  if (!input) return NullArg("input");
  if (!out_path) return NullArg("out_path");
  *out_path = nullptr;
  static thread_local std::string slot;

  std::string uri_part, tail_part;
  if (mp::TrySplitAssetUriPrefix(std::string(input), uri_part, tail_part)) {
    auto asset_it = pkg->shared_asset_index_by_uri.find(uri_part);
    if (asset_it == pkg->shared_asset_index_by_uri.end()) {
      return MakeStatus(MODEL_PACKAGE_ERR_ASSET_MISSING,
                        std::string("Asset URI not declared in this package: '") + uri_part + "'.");
    }
    const std::string& asset_folder = pkg->shared_assets[asset_it->second]->resolved_path_cache;
    if (tail_part.empty()) {
      slot = asset_folder;
      *out_path = slot.c_str();
      return nullptr;
    }
    // Tail is resolved with portable confinement under the asset folder:
    // no absolute, no `..`. follow_symlinks mirrors the package setting.
    mp::PathResolverOptions tail_opts;
    tail_opts.allow_external_paths = false;
    tail_opts.follow_symlinks = pkg->follow_symlinks;
    std::filesystem::path resolved;
    if (auto* s = mp::ResolvePath(asset_folder, asset_folder, tail_part, tail_opts,
                                  must_exist, &resolved)) {
      return s;
    }
    slot = resolved.string();
    *out_path = slot.c_str();
    return nullptr;
  }

  std::filesystem::path base = base_dir ? std::filesystem::path(base_dir) : pkg->package_root;
  std::filesystem::path resolved;
  if (auto* s = mp::ResolvePath(base, pkg->package_root, std::string(input),
                                mp::PathOptionsFor(pkg), must_exist, &resolved)) {
    return s;
  }
  slot = resolved.string();
  *out_path = slot.c_str();
  return nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Round-trip JSON getters
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

// ─────────────────────────────────────────────────────────────────────────────
// Hashing utility
// ─────────────────────────────────────────────────────────────────────────────

ModelPackageStatus* ModelPackage_ComputeDirectoryHash(const char* source_dir,
                                                      const char** out_uri) {
  if (!source_dir) return NullArg("source_dir");
  if (!out_uri) return NullArg("out_uri");
  *out_uri = nullptr;
  static thread_local std::string slot;
  if (auto* s = mp::ComputeDirectoryAssetUri(std::filesystem::path(source_dir), &slot)) {
    return s;
  }
  *out_uri = slot.c_str();
  return nullptr;
}

}  // extern "C"
