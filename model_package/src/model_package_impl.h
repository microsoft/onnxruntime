// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file model_package_impl.h
/// \brief Internal C++ representation of a ModelPackage handle.
///
/// Records hold the parsed manifest data plus stable per-entity string buffers
/// so that all `const char*` exposed through the C API have package-owned
/// storage. The package builds an `InfoViewCache` lazily that materializes the
/// public POD struct tree returned by `ModelPackage_Info()`; any mutation
/// drops the cache so the next read produces a fresh tree.

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

namespace model_package {

using ordered_json = nlohmann::ordered_json;

/// How the component's body is stored on disk relative to the manifest.
enum class ComponentStorage {
  kInline,    ///< body lives directly inside the manifest as an object
  kExternal,  ///< body lives in a separate file pointed to by a string
};

struct VariantRecord {
  std::string name;
  nlohmann::ordered_json body;  ///< the full variant JSON object

  // Stable string buffers for ABI exposure.
  std::string name_cache;
  std::optional<std::string> ep_cache;
  std::optional<std::string> device_cache;
  std::optional<std::string> compatibility_string_cache;
  std::optional<std::string> resolved_directory_cache;
  mutable std::optional<std::string> additional_metadata_cache;
  mutable std::optional<std::string> variant_json_cache;

  /// Resolved variant_directory for variants that have one. `std::nullopt`
  /// means none was declared and the default location does not exist.
  std::optional<std::filesystem::path> resolved_directory;
  bool resolved_directory_attempted{false};

  /// Pre-resolved executor_info entries. Populated eagerly at Open and
  /// after any mutation that can touch executor_info. The first member is the
  /// namespace key; the second is the serialized JSON body of that entry
  /// (inline bodies are dumped, external file bodies are read + validated).
  std::vector<std::pair<std::string, std::string>> executor_info_resolved;
};

struct ComponentRecord {
  std::string name;
  ComponentStorage storage{ComponentStorage::kInline};
  std::filesystem::path external_path;  ///< valid iff storage == kExternal
  std::filesystem::path component_dir;  ///< base directory for relative paths inside this component
  nlohmann::ordered_json body;          ///< {"component_name": ..., "variants": {...}, "additional_metadata": {...}}
  std::vector<std::unique_ptr<VariantRecord>> variants;

  std::string name_cache;
  mutable std::optional<std::string> additional_metadata_cache;
  mutable std::optional<std::string> component_json_cache;
};

struct SharedAssetRecord {
  std::string uri;  ///< "sha256:<hex>"
  std::filesystem::path resolved_path;
  std::string uri_cache;
  std::string resolved_path_cache;
};

/// Materialized POD-struct tree returned by ModelPackage_Info(). Owns all
/// backing storage (extra strings and array buffers) so pointers stay valid
/// until the next mutation drops the cache.
struct InfoViewCache {
  // Per-variant arrays. Indexed [component_idx][variant_idx].
  std::vector<std::vector<ModelExecutorInfoEntry>> executor_infos_storage;
  std::vector<std::vector<ModelVariantInfo>> variants_storage;

  std::vector<ModelComponentInfo> components;
  std::vector<ModelSharedAssetInfo> shared_assets;
  ModelPackageInfo info{};
};

}  // namespace model_package

// ─────────────────────────────────────────────────────────────────────────────
// Public opaque type (lives in the global namespace to match the C API)
// ─────────────────────────────────────────────────────────────────────────────

struct ModelPackage {
  std::filesystem::path package_root;
  nlohmann::ordered_json manifest;  ///< parsed manifest.json with declarations intact (component values stay in their original string-or-object form)
  std::string layout;               ///< "portable" | "installed"

  // Open-time options.
  bool allow_external_paths{false};
  bool follow_symlinks{true};
  bool strict_unknown_fields{true};

  // Package-level parsed data and stable string buffers.
  int64_t schema_version_major{0};
  int64_t schema_version_minor{0};
  std::optional<std::string> package_name_cache;
  std::optional<std::string> package_version_cache;
  std::optional<std::string> description_cache;
  std::string layout_cache;
  mutable std::optional<std::string> additional_metadata_cache;

  std::vector<std::unique_ptr<model_package::ComponentRecord>> components;
  std::vector<std::unique_ptr<model_package::SharedAssetRecord>> shared_assets;

  std::unordered_map<std::string, size_t> component_index_by_name;
  std::unordered_map<std::string, size_t> shared_asset_index_by_uri;

  /// Authoring-time staging for copy_in=true shared assets that have not been
  /// committed yet. Keyed by sha256:<hex> URI.
  std::unordered_map<std::string, std::filesystem::path> pending_shared_asset_copies;

  /// Paths removed from the live tree, candidates for ModelPackage_Prune.
  /// Populated by the authoring API; never by walking package_root.
  std::vector<std::filesystem::path> pending_orphan_variant_dirs;
  std::vector<std::filesystem::path> pending_orphan_component_dirs;

  /// Cache for the most recent ModelPackage_Validate report JSON.
  mutable std::optional<std::string> last_validate_report;

  /// Lazily built; dropped on any mutation.
  mutable std::optional<model_package::InfoViewCache> info_cache;
};

namespace model_package {

/// Drop the materialized view cache. Call after any mutation that affects the
/// view tree. Safe on a cleared cache.
void DropViewCache(ModelPackage* pkg);

/// Return the package's info view, building it lazily.
const InfoViewCache& BuildOrGetViewCache(const ModelPackage* pkg);

/// Returns true iff `p` is `package_root` or lives under it (lexically).
bool IsInsidePackageRoot(const ModelPackage* pkg, const std::filesystem::path& p);

/// Push the variant's resolved_directory onto the Prune candidates if it's
/// inside package_root. No-op if unresolved.
void RecordOrphanVariantDir(ModelPackage* pkg, const VariantRecord& v);

/// Push every variant_dir of `c`, plus `c.component_dir` if external.
void RecordOrphanComponent(ModelPackage* pkg, const ComponentRecord& c);

}  // namespace model_package
