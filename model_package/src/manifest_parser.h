// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file manifest_parser.h
/// \brief Internal parser that reads a model package from disk into the
///        in-memory representation defined in model_package_impl.h.

#pragma once

#include "model_package_impl.h"
#include "path_resolver.h"

namespace model_package {

/// Parse the manifest at `<package_root>/manifest.json` and all referenced
/// external component files, then populate `*pkg`. Caller owns `pkg`.
ModelPackageStatus* ParsePackage(const std::filesystem::path& package_root,
                                 const ModelPackageOpenOptions& opts,
                                 ModelPackage* pkg);

/// Parse a single variant body into `out`. Used by authoring.
ModelPackageStatus* ParseVariantBody(const std::filesystem::path& component_dir,
                                     const std::filesystem::path& package_root,
                                     const PathResolverOptions& opts,
                                     bool strict,
                                     const std::string& variant_name,
                                     const ordered_json& variant_body,
                                     VariantRecord* out);

/// Parse a single component body. `component_dir` is the directory used as the
/// base for the component's relative paths.
ModelPackageStatus* ParseComponentBody(const std::filesystem::path& package_root,
                                       const PathResolverOptions& opts,
                                       bool strict,
                                       const std::string& component_name,
                                       const ordered_json& body,
                                       const std::filesystem::path& component_dir,
                                       ComponentRecord* out);

/// Re-derive package-level metadata (schema_version, package_name, version,
/// description, layout, additional_metadata) from `pkg->manifest` into the
/// package's stable string buffers.
ModelPackageStatus* RefreshPackageMetadata(ModelPackage* pkg);

/// Re-derive `pkg->shared_assets` from `pkg->manifest.shared_assets` overrides,
/// plus any `sha256-<hex>` subdirectories discovered under
/// `<package_root>/shared_assets/`, plus any pending copy_in entries staged via
/// the authoring API. Clears and replaces the existing shared_assets vector
/// and `shared_asset_index_by_uri`.
ModelPackageStatus* RefreshSharedAssets(ModelPackage* pkg, const PathResolverOptions& opts);

/// Re-resolve every variant's executor_info entries into stable strings on the
/// VariantRecord (inline bodies dumped, external files loaded + JSON-parsed).
/// If `strict_missing_external` is true, missing external files are an error
/// (use at Open: the package is already published, files must be present);
/// if false, missing external files are recorded as an empty body (use during
/// authoring: callers may set the path before writing the file). Parse errors
/// on existing external files are always surfaced.
ModelPackageStatus* RefreshExecutorInfoCache(ModelPackage* pkg, bool strict_missing_external);

/// Build PathResolverOptions appropriate for `pkg` (respects layout).
PathResolverOptions PathOptionsFor(const ModelPackage* pkg);

}  // namespace model_package
