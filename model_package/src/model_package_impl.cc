// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_impl.cc
/// \brief Implementation of the public C API declared in model_package.h.

#include "model_package.h"

#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>

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
// ABI guards
//
// 1. View safety: every accessor reinterpret_casts a public element pointer to its view
//    struct. That is only valid if the public struct is the view's first member (so their
//    addresses coincide) and both are standard-layout. These run on every platform.
// 2. Append-only layout: the field offsets below are pinned so that reordering, removing, or
//    inserting a field (which would silently break already-compiled consumers within a
//    SOVERSION) fails to compile. Appending a new trailing field does not change existing
//    offsets and is therefore allowed. Pins are gated on 64-bit pointers since offsets are
//    pointer-size dependent; appending a field requires no change here.
// ─────────────────────────────────────────────────────────────────────────────

static_assert(std::is_standard_layout<ModelPackageInfo>::value, "ModelPackageInfo must be standard-layout");
static_assert(std::is_standard_layout<ModelComponentInfo>::value, "ModelComponentInfo must be standard-layout");
static_assert(std::is_standard_layout<ModelVariantInfo>::value, "ModelVariantInfo must be standard-layout");
static_assert(std::is_standard_layout<ModelExecutorInfoEntry>::value, "ModelExecutorInfoEntry must be standard-layout");
static_assert(std::is_standard_layout<ModelSharedAssetInfo>::value, "ModelSharedAssetInfo must be standard-layout");
static_assert(std::is_standard_layout<ModelPackageOpenOptions>::value, "ModelPackageOpenOptions must be standard-layout");
static_assert(std::is_standard_layout<mp::PackageInfoView>::value, "PackageInfoView must be standard-layout");
static_assert(std::is_standard_layout<mp::ComponentView>::value, "ComponentView must be standard-layout");
static_assert(std::is_standard_layout<mp::VariantView>::value, "VariantView must be standard-layout");

static_assert(offsetof(mp::PackageInfoView, pub) == 0, "public struct must be the view's first member");
static_assert(offsetof(mp::ComponentView, pub) == 0, "public struct must be the view's first member");
static_assert(offsetof(mp::VariantView, pub) == 0, "public struct must be the view's first member");
static_assert(offsetof(ModelPackageInfo, struct_size) == 0, "struct_size must be the first field");
static_assert(offsetof(ModelComponentInfo, struct_size) == 0, "struct_size must be the first field");
static_assert(offsetof(ModelVariantInfo, struct_size) == 0, "struct_size must be the first field");
static_assert(offsetof(ModelExecutorInfoEntry, struct_size) == 0, "struct_size must be the first field");
static_assert(offsetof(ModelSharedAssetInfo, struct_size) == 0, "struct_size must be the first field");
static_assert(offsetof(ModelPackageOpenOptions, struct_size) == 0, "struct_size must be the first field");

#if defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8
// Append-only field-offset pins (64-bit). Reordering/removing/inserting a field changes one
// of these and fails the build; appending a trailing field does not. Update only when adding
// a field at the end (new offsets) — never to "fix" a reorder.
static_assert(offsetof(ModelPackageInfo, schema_version_major) == 8, "ModelPackageInfo layout changed");
static_assert(offsetof(ModelPackageInfo, schema_version_minor) == 16, "ModelPackageInfo layout changed");
static_assert(offsetof(ModelPackageInfo, package_name) == 24, "ModelPackageInfo layout changed");
static_assert(offsetof(ModelPackageInfo, package_version) == 32, "ModelPackageInfo layout changed");
static_assert(offsetof(ModelPackageInfo, description) == 40, "ModelPackageInfo layout changed");
static_assert(offsetof(ModelPackageInfo, layout) == 48, "ModelPackageInfo layout changed");
static_assert(offsetof(ModelPackageInfo, additional_metadata_json) == 56, "ModelPackageInfo layout changed");
static_assert(offsetof(ModelComponentInfo, name) == 8, "ModelComponentInfo layout changed");
static_assert(offsetof(ModelComponentInfo, additional_metadata_json) == 16, "ModelComponentInfo layout changed");
static_assert(offsetof(ModelVariantInfo, name) == 8, "ModelVariantInfo layout changed");
static_assert(offsetof(ModelVariantInfo, variant_directory) == 16, "ModelVariantInfo layout changed");
static_assert(offsetof(ModelVariantInfo, ep) == 24, "ModelVariantInfo layout changed");
static_assert(offsetof(ModelVariantInfo, device) == 32, "ModelVariantInfo layout changed");
static_assert(offsetof(ModelVariantInfo, compatibility_string) == 40, "ModelVariantInfo layout changed");
static_assert(offsetof(ModelVariantInfo, additional_metadata_json) == 48, "ModelVariantInfo layout changed");
static_assert(offsetof(ModelExecutorInfoEntry, namespace_key) == 8, "ModelExecutorInfoEntry layout changed");
static_assert(offsetof(ModelExecutorInfoEntry, json) == 16, "ModelExecutorInfoEntry layout changed");
static_assert(offsetof(ModelSharedAssetInfo, uri) == 8, "ModelSharedAssetInfo layout changed");
static_assert(offsetof(ModelSharedAssetInfo, resolved_path) == 16, "ModelSharedAssetInfo layout changed");
#endif  // 64-bit pointer

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
        entry.struct_size = sizeof(ModelExecutorInfoEntry);
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

    // Second pass: populate VariantView entries pointing at the now-stable
    // executor-info storage above.
    for (size_t vi = 0; vi < num_variants; ++vi) {
      const auto& var = *comp.variants[vi];
      VariantView& view = cache.variants_storage[ci][vi];
      view = VariantView{};
      ModelVariantInfo& vi_out = view.pub;
      vi_out.struct_size = sizeof(ModelVariantInfo);
      vi_out.name = var.name_cache.c_str();
      vi_out.variant_directory =
          var.resolved_directory_cache.has_value() ? var.resolved_directory_cache->c_str() : nullptr;
      vi_out.ep = OptStr(var.ep_cache);
      vi_out.device = OptStr(var.device_cache);
      vi_out.compatibility_string = OptStr(var.compatibility_string_cache);
      vi_out.additional_metadata_json = OptStr(var.additional_metadata_cache);
      auto [ei_begin, ei_end] = ei_ranges[vi];
      view.num_executor_infos = ei_end - ei_begin;
      view.executor_infos =
          (view.num_executor_infos > 0) ? &cache.executor_infos_storage[ci][ei_begin] : nullptr;
    }

    ComponentView& comp_view = cache.components[ci];
    comp_view = ComponentView{};
    ModelComponentInfo& ci_out = comp_view.pub;
    ci_out.struct_size = sizeof(ModelComponentInfo);
    ci_out.name = comp.name_cache.c_str();
    ci_out.additional_metadata_json = OptStr(comp.additional_metadata_cache);
    comp_view.num_variants = num_variants;
    comp_view.variants = num_variants > 0 ? cache.variants_storage[ci].data() : nullptr;
  }

  // Shared assets (leaf structs: no children, plain storage).
  cache.shared_assets.resize(pkg->shared_assets.size());
  for (size_t i = 0; i < pkg->shared_assets.size(); ++i) {
    const auto& rec = *pkg->shared_assets[i];
    ModelSharedAssetInfo& sa = cache.shared_assets[i];
    sa = ModelSharedAssetInfo{};
    sa.struct_size = sizeof(ModelSharedAssetInfo);
    sa.uri = rec.uri_cache.c_str();
    sa.resolved_path = rec.resolved_path_cache.c_str();
  }

  ModelPackageInfo& info = cache.root.pub;
  info = ModelPackageInfo{};
  info.struct_size = sizeof(ModelPackageInfo);
  info.schema_version_major = pkg->schema_version_major;
  info.schema_version_minor = pkg->schema_version_minor;
  info.package_name = OptStr(pkg->package_name_cache);
  info.package_version = OptStr(pkg->package_version_cache);
  info.description = OptStr(pkg->description_cache);
  info.layout = pkg->layout_cache.c_str();
  info.additional_metadata_json = OptStr(pkg->additional_metadata_cache);
  cache.root.num_components = cache.components.size();
  cache.root.components = cache.components.empty() ? nullptr : cache.components.data();
  cache.root.num_shared_assets = cache.shared_assets.size();
  cache.root.shared_assets = cache.shared_assets.empty() ? nullptr : cache.shared_assets.data();

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
  effective.struct_size = sizeof(ModelPackageOpenOptions);
  effective.allow_external_paths = false;
  effective.follow_symlinks = true;
  effective.strict_unknown_fields = true;
  if (opts) {
    // struct_size is the caller's sizeof(ModelPackageOpenOptions). It must be at least large
    // enough to contain the struct_size field itself; a smaller value means the caller did
    // not initialize it (e.g. forgot the `= {sizeof(...)}` idiom) and we cannot safely read
    // any field. Reject rather than risk an out-of-bounds read.
    if (opts->struct_size < sizeof(opts->struct_size)) {
      return MakeStatus(MODEL_PACKAGE_ERR_INVALID_ARG,
                        "ModelPackage_Open: options struct_size is too small; set it to "
                        "sizeof(ModelPackageOpenOptions).");
    }
    if (opts->struct_size >= sizeof(ModelPackageOpenOptions)) {
      effective = *opts;
    } else {
      const char* base = reinterpret_cast<const char*>(opts);
      auto copy_if_fits = [&](size_t offset, size_t size, void* dst) {
        if (offset + size <= opts->struct_size) std::memcpy(dst, base + offset, size);
      };
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
  return &mp::BuildOrGetViewCache(pkg).root.pub;
}

// Collection accessors. Each public element struct is the first member of its
// view, so the reinterpret_cast from the public pointer to the view is valid.

size_t ModelPackageInfo_GetComponentCount(const ModelPackageInfo* info) {
  if (!info) return 0;
  return reinterpret_cast<const mp::PackageInfoView*>(info)->num_components;
}

const ModelComponentInfo* ModelPackageInfo_GetComponent(const ModelPackageInfo* info, size_t index) {
  if (!info) return nullptr;
  const auto* view = reinterpret_cast<const mp::PackageInfoView*>(info);
  if (index >= view->num_components) return nullptr;
  return &view->components[index].pub;
}

size_t ModelPackageInfo_GetSharedAssetCount(const ModelPackageInfo* info) {
  if (!info) return 0;
  return reinterpret_cast<const mp::PackageInfoView*>(info)->num_shared_assets;
}

const ModelSharedAssetInfo* ModelPackageInfo_GetSharedAsset(const ModelPackageInfo* info, size_t index) {
  if (!info) return nullptr;
  const auto* view = reinterpret_cast<const mp::PackageInfoView*>(info);
  if (index >= view->num_shared_assets) return nullptr;
  return &view->shared_assets[index];
}

size_t ModelComponentInfo_GetVariantCount(const ModelComponentInfo* comp) {
  if (!comp) return 0;
  return reinterpret_cast<const mp::ComponentView*>(comp)->num_variants;
}

const ModelVariantInfo* ModelComponentInfo_GetVariant(const ModelComponentInfo* comp, size_t index) {
  if (!comp) return nullptr;
  const auto* view = reinterpret_cast<const mp::ComponentView*>(comp);
  if (index >= view->num_variants) return nullptr;
  return &view->variants[index].pub;
}

size_t ModelVariantInfo_GetExecutorInfoCount(const ModelVariantInfo* var) {
  if (!var) return 0;
  return reinterpret_cast<const mp::VariantView*>(var)->num_executor_infos;
}

const ModelExecutorInfoEntry* ModelVariantInfo_GetExecutorInfo(const ModelVariantInfo* var, size_t index) {
  if (!var) return nullptr;
  const auto* view = reinterpret_cast<const mp::VariantView*>(var);
  if (index >= view->num_executor_infos) return nullptr;
  return &view->executor_infos[index];
}

const ModelComponentInfo* ModelPackage_FindComponent(const ModelPackageInfo* info,
                                                     const char* name) {
  if (!info || !name) return nullptr;
  const size_t n = ModelPackageInfo_GetComponentCount(info);
  for (size_t i = 0; i < n; ++i) {
    const ModelComponentInfo* comp = ModelPackageInfo_GetComponent(info, i);
    if (comp && comp->name && std::strcmp(comp->name, name) == 0) {
      return comp;
    }
  }
  return nullptr;
}

const ModelVariantInfo* ModelComponentInfo_FindVariant(const ModelComponentInfo* comp,
                                                       const char* name) {
  if (!comp || !name) return nullptr;
  const size_t n = ModelComponentInfo_GetVariantCount(comp);
  for (size_t i = 0; i < n; ++i) {
    const ModelVariantInfo* var = ModelComponentInfo_GetVariant(comp, i);
    if (var && var->name && std::strcmp(var->name, name) == 0) {
      return var;
    }
  }
  return nullptr;
}

const ModelExecutorInfoEntry* ModelVariantInfo_FindExecutorInfo(const ModelVariantInfo* var,
                                                                const char* namespace_key) {
  if (!var || !namespace_key) return nullptr;
  const size_t n = ModelVariantInfo_GetExecutorInfoCount(var);
  for (size_t i = 0; i < n; ++i) {
    const ModelExecutorInfoEntry* e = ModelVariantInfo_GetExecutorInfo(var, i);
    if (e && e->namespace_key && std::strcmp(e->namespace_key, namespace_key) == 0) {
      return e;
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
