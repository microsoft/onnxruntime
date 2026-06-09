// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_impl.h
/// \brief Internal C++ representation of a ModelPackage handle.
///
/// The package stores its parsed manifest plus per-component records as
/// `nlohmann::ordered_json` to preserve declaration order and unknown fields
/// for round-trip. Typed accessors are thin views over the JSON; their string
/// outputs are cached in stable per-entity std::string fields so that
/// `const char*` returns remain valid until the package is closed or the
/// relevant scope is mutated.

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "model_package.h"

namespace model_package_v2 {

using ordered_json = nlohmann::ordered_json;

// ─────────────────────────────────────────────────────────────────────────────
// Records
// ─────────────────────────────────────────────────────────────────────────────

/// How the component's body is stored on disk relative to the manifest.
enum class ComponentStorage {
  kInline,   ///< body lives directly inside the manifest as an object
  kExternal, ///< body lives in a separate file pointed to by a string
};

struct VariantRecord {
  std::string name;
  nlohmann::ordered_json body;                    ///< the full variant JSON object

  // String caches for stable C API pointers.
  std::string name_cache;
  std::optional<std::string> ep_cache;
  std::optional<std::string> device_cache;
  std::optional<std::string> compatibility_string_cache;
  std::optional<std::string> resolved_directory_cache;
  std::vector<std::string> used_asset_uri_caches;
  mutable std::unordered_map<std::string, std::string> executor_info_json_cache;
  mutable std::optional<std::string> additional_metadata_cache;
  mutable std::optional<std::string> variant_json_cache;

  // The variant's resolved variant_directory, if it has one. Lazily filled.
  // std::nullopt means "no resolvable directory" (the directory field is
  // missing and the default <component_dir>/<variant_name>/ doesn't exist).
  // Populated at open for variants that declare any inline executor_info
  // (eager check per §4.2). Otherwise computed on-demand.
  std::optional<std::filesystem::path> resolved_directory;
  bool resolved_directory_attempted{false};
};

struct ComponentRecord {
  std::string name;
  ComponentStorage storage{ComponentStorage::kInline};
  std::filesystem::path external_path;            ///< valid iff storage == kExternal
  std::filesystem::path component_dir;            ///< the directory used as the base for this component's relative paths
  nlohmann::ordered_json body;                    ///< {"component_name": ..., "variants": {...}, "additional_metadata": {...}}
  std::vector<std::unique_ptr<VariantRecord>> variants;

  // String caches.
  std::string name_cache;
  mutable std::optional<std::string> additional_metadata_cache;
  mutable std::optional<std::string> component_json_cache;
};

struct SharedAssetRecord {
  std::string uri;                                ///< "sha256:<hex>"
  std::filesystem::path resolved_path;
  std::string uri_cache;
  std::string resolved_path_cache;
  ModelSharedAsset abi_view{};                    ///< populated to point at the caches above
};

}  // namespace model_package_v2

// ─────────────────────────────────────────────────────────────────────────────
// Public opaque types (live in the global namespace to match the C API)
// ─────────────────────────────────────────────────────────────────────────────

struct ModelPackage {
  std::filesystem::path package_root;
  nlohmann::ordered_json manifest;                ///< the parsed manifest.json, with declarations intact (component values stay in their original string-or-object form)
  std::string layout;                             ///< "portable" | "installed"

  // Open-time options.
  bool allow_external_paths{false};
  bool follow_symlinks{true};
  bool strict_unknown_fields{true};

  // Component and shared-asset records (in declaration order).
  std::vector<std::unique_ptr<model_package_v2::ComponentRecord>> components;
  std::vector<std::unique_ptr<model_package_v2::SharedAssetRecord>> shared_assets;

  // Index for fast name->record lookup.
  std::unordered_map<std::string, size_t> component_index_by_name;
  std::unordered_map<std::string, size_t> shared_asset_index_by_uri;

  // Authoring-time bookkeeping: source directories for copy_in=true shared
  // assets that haven't been committed yet. Keyed by sha256:<hex> URI.
  std::unordered_map<std::string, std::filesystem::path> pending_shared_asset_copies;

  // Package-level string caches and ABI view.
  std::optional<std::string> package_name_cache;
  std::optional<std::string> package_version_cache;
  std::optional<std::string> description_cache;
  std::string layout_cache;
  mutable std::optional<std::string> additional_metadata_cache;
  ModelPackageInfo info_view{};
};

struct ModelComponent {
  ModelPackage* owner{nullptr};
  size_t component_idx{0};
  model_package_v2::ComponentRecord* record{nullptr};
};

struct ModelVariant {
  ModelPackage* owner{nullptr};
  size_t component_idx{0};
  size_t variant_idx{0};
  model_package_v2::ComponentRecord* component_record{nullptr};
  model_package_v2::VariantRecord* record{nullptr};
};

namespace model_package_v2 {

void DropViewCache(const ModelPackage* pkg);

// Stable view handles kept alive by the package so that pointer identity
// matches across repeated lookups (per §7.2 caller contract).
struct ViewCache {
  std::vector<std::unique_ptr<ModelComponent>> component_views;
  std::vector<std::vector<std::unique_ptr<ModelVariant>>> variant_views; // [component_idx][variant_idx]
};

ViewCache& GetViewCache(ModelPackage* pkg);
const ModelComponent* ComponentView(ModelPackage* pkg, size_t idx);
const ModelVariant*   VariantView(ModelPackage* pkg, size_t comp_idx, size_t var_idx);

}  // namespace model_package_v2
