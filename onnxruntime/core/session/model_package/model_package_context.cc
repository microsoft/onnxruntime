// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package/model_package_options.h"
#include "core/session/model_package/model_package_variant_selector.h"
#include "core/session/ort_env.h"
#include "core/session/provider_policy_context.h"
#include "core/session/utils.h"

// We intentionally use the standalone model_package library's internal C++ types directly
// (model_package::ParsePackage, model_package_internal.h) rather than its public C API
// (ModelPackage_* functions). This avoids double-wrapping since ORT compiles the library in-tree.
// The public C API exists for external consumers (GenAI, FL) who link independently.
#include "model_package_internal.h"
#include "parser.h"

namespace onnxruntime {

namespace {

Status FillOptionCachesFromMap(
    const std::optional<std::unordered_map<std::string, std::string>>& options_map,
    std::vector<std::string>& key_cache,
    std::vector<std::string>& value_cache,
    gsl::span<const std::string>& out_keys,
    gsl::span<const std::string>& out_values) {
  key_cache.clear();
  value_cache.clear();

  if (!options_map.has_value()) {
    out_keys = gsl::span<const std::string>{};
    out_values = gsl::span<const std::string>{};
    return Status::OK();
  }

  key_cache.reserve(options_map->size());
  value_cache.reserve(options_map->size());

  for (const auto& kv : *options_map) {
    key_cache.push_back(kv.first);
    value_cache.push_back(kv.second);
  }

  out_keys = gsl::span<const std::string>(key_cache.data(), key_cache.size());
  out_values = gsl::span<const std::string>(value_cache.data(), value_cache.size());
  return Status::OK();
}

}  // namespace

ModelPackageComponentContext::ModelPackageComponentContext(const std::string& component_name,
                                                           const ComponentInfo& component_model_info,
                                                           const ModelPackageOptions& options)
    : component_model_name_(component_name),
      component_model_info_(component_model_info),
      owned_ep_infos_(options.EpInfos()),
      execution_devices_(options.ExecutionDevices()),
      devices_selected_(options.DevicesSelected()),
      from_policy_(options.FromPolicy()) {
  // Point the span at our owned copy.
  ep_infos_ = gsl::span<const VariantSelectionEpInfo>(owned_ep_infos_);
}

ModelPackageComponentContext::ModelPackageComponentContext(const std::string& component_name,
                                                           const ComponentInfo& component_model_info,
                                                           gsl::span<const VariantSelectionEpInfo> ep_infos)
    : component_model_name_(component_name),
      component_model_info_(component_model_info),
      ep_infos_(ep_infos) {
}

Status ModelPackageComponentContext::ResolveVariant() {
  ORT_RETURN_IF(ep_infos_.empty(),
                "ModelPackageComponentContext::ResolveVariant requires non-empty ep_infos "
                "(from ModelPackageOptions or explicit span).");

  return ResolveVariantImpl(ep_infos_);
}

Status ModelPackageComponentContext::ResolveVariantImpl(gsl::span<const VariantSelectionEpInfo> ep_infos) {
  std::optional<VariantInfo> selected_variant;
  VariantSelector selector;
  ORT_RETURN_IF_ERROR(selector.SelectVariant(*this, ep_infos, selected_variant));

  ORT_RETURN_IF(!selected_variant.has_value(),
                "No suitable model variant found for the configured execution providers.");

  ORT_RETURN_IF(selected_variant->component_name != component_model_name_,
                "Selected variant's component model name does not match context's component model name.");

  component_model_info_.selected_variant_index.reset();

  size_t matched_variants = 0;
  for (size_t i = 0; i < component_model_info_.variants.size(); ++i) {
    if (component_model_info_.variants[i].variant_name == selected_variant->variant_name) {
      component_model_info_.selected_variant_index = i;
      component_model_info_.variants[i] = *selected_variant;
      ++matched_variants;
    }
  }

  ORT_RETURN_IF(matched_variants == 0,
                "Selected variant was not found in model package context.");
  ORT_RETURN_IF(matched_variants > 1,
                "Selected variant matched multiple variants; selection is ambiguous.");

  return Status::OK();
}

Status ModelPackageComponentContext::GetSelectedVariantFolderPath(const std::filesystem::path*& out_folder_path) const {
  out_folder_path = nullptr;

  ORT_RETURN_IF(!component_model_info_.selected_variant_index.has_value(),
                "No variant selected for component: ", component_model_name_);

  const size_t selected_idx = *component_model_info_.selected_variant_index;
  ORT_RETURN_IF(selected_idx >= component_model_info_.variants.size(),
                "Selected variant index out of range for component: ", component_model_name_);

  const auto& selected_variant = component_model_info_.variants[selected_idx];

  folder_path_cache_ = selected_variant.folder_path;
  out_folder_path = &folder_path_cache_;
  return Status::OK();
}

Status ModelPackageComponentContext::GetSelectedVariantFilePath(std::filesystem::path& out_path) const {
  out_path.clear();

  ORT_RETURN_IF(!component_model_info_.selected_variant_index.has_value(),
                "No variant selected for component: ", component_model_name_);

  const size_t selected_idx = *component_model_info_.selected_variant_index;
  ORT_RETURN_IF(selected_idx >= component_model_info_.variants.size(),
                "Selected variant index out of range for component: ", component_model_name_);

  const auto& selected_variant = component_model_info_.variants[selected_idx];
  ORT_RETURN_IF(!selected_variant.file.has_value(),
                "Selected variant '", selected_variant.variant_name,
                "' does not have a variant.json descriptor (or it lacks a 'filename' entry). "
                "Component: ",
                component_model_name_);

  out_path = selected_variant.file->model_file_path;
  return Status::OK();
}

Status ModelPackageComponentContext::GetSelectedVariantInfo(const VariantInfo*& out_variant) const {
  out_variant = nullptr;

  if (!component_model_info_.selected_variant_index.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "No variant is selected for component: ", component_model_name_);
  }

  const size_t selected_idx = *component_model_info_.selected_variant_index;
  if (selected_idx >= component_model_info_.variants.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Selected variant index out of range for component: ", component_model_name_);
  }

  out_variant = &component_model_info_.variants[selected_idx];
  return Status::OK();
}

Status ModelPackageComponentContext::GetSelectedVariantFileSessionOptions(gsl::span<const std::string>& out_keys,
                                                                          gsl::span<const std::string>& out_values) const {
  out_keys = {};
  out_values = {};

  const VariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantInfo(selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr, "Selected variant is null for component: ", component_model_name_);
  ORT_RETURN_IF(!selected_variant->file.has_value(), "Selected variant has no file entry.");

  // Fast path: return cached key/value vectors if both exist.
  if (!session_option_keys_cache_.empty() || !session_option_values_cache_.empty()) {
    ORT_RETURN_IF(session_option_keys_cache_.size() != session_option_values_cache_.size(),
                  "Session options key/value cache size mismatch");

    out_keys = gsl::span<const std::string>(session_option_keys_cache_.data(), session_option_keys_cache_.size());
    out_values = gsl::span<const std::string>(session_option_values_cache_.data(), session_option_values_cache_.size());
    return Status::OK();
  }

  const auto& selected_file = *selected_variant->file;
  return FillOptionCachesFromMap(selected_file.session_options,
                                 session_option_keys_cache_,
                                 session_option_values_cache_,
                                 out_keys,
                                 out_values);
}

Status ModelPackageComponentContext::GetSelectedVariantFileProviderOptions(gsl::span<const std::string>& out_keys,
                                                                           gsl::span<const std::string>& out_values) const {
  out_keys = {};
  out_values = {};

  const VariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantInfo(selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr, "Selected variant is null for component: ", component_model_name_);
  ORT_RETURN_IF(!selected_variant->file.has_value(), "Selected variant has no file entry.");

  // Fast path: return cached key/value vectors if both exist.
  if (!provider_option_keys_cache_.empty() || !provider_option_values_cache_.empty()) {
    ORT_RETURN_IF(provider_option_keys_cache_.size() != provider_option_values_cache_.size(),
                  "Provider options key/value cache size mismatch");

    out_keys = gsl::span<const std::string>(provider_option_keys_cache_.data(), provider_option_keys_cache_.size());
    out_values = gsl::span<const std::string>(provider_option_values_cache_.data(), provider_option_values_cache_.size());
    return Status::OK();
  }

  const auto& selected_file = *selected_variant->file;
  return FillOptionCachesFromMap(selected_file.provider_options,
                                 provider_option_keys_cache_,
                                 provider_option_values_cache_,
                                 out_keys,
                                 out_values);
}

namespace {
void BuildPtrCache(gsl::span<const std::string> strings,
                   std::vector<const char*>& ptrs_cache,
                   const char* const*& out_ptrs, size_t& out_count) {
  ptrs_cache.clear();
  ptrs_cache.reserve(strings.size());
  for (const auto& s : strings) {
    ptrs_cache.push_back(s.c_str());
  }
  out_count = ptrs_cache.size();
  out_ptrs = ptrs_cache.empty() ? nullptr : ptrs_cache.data();
}
}  // namespace

Status ModelPackageComponentContext::GetSelectedVariantFileSessionOptionPtrs(
    const char* const*& out_keys,
    const char* const*& out_values,
    size_t& out_count) const {
  out_keys = nullptr;
  out_values = nullptr;
  out_count = 0;

  gsl::span<const std::string> keys;
  gsl::span<const std::string> values;
  ORT_RETURN_IF_ERROR(GetSelectedVariantFileSessionOptions(keys, values));
  ORT_RETURN_IF(keys.size() != values.size(), "Session options keys/values size mismatch.");

  BuildPtrCache(keys, session_option_key_ptrs_cache_, out_keys, out_count);
  size_t dummy;
  BuildPtrCache(values, session_option_value_ptrs_cache_, out_values, dummy);
  return Status::OK();
}

Status ModelPackageComponentContext::GetSelectedVariantFileProviderOptionPtrs(
    const char* const*& out_keys,
    const char* const*& out_values,
    size_t& out_count) const {
  out_keys = nullptr;
  out_values = nullptr;
  out_count = 0;

  gsl::span<const std::string> keys;
  gsl::span<const std::string> values;
  ORT_RETURN_IF_ERROR(GetSelectedVariantFileProviderOptions(keys, values));
  ORT_RETURN_IF(keys.size() != values.size(), "Provider options keys/values size mismatch.");

  BuildPtrCache(keys, provider_option_key_ptrs_cache_, out_keys, out_count);
  size_t dummy;
  BuildPtrCache(values, provider_option_value_ptrs_cache_, out_values, dummy);
  return Status::OK();
}

Status ModelPackageComponentContext::RebuildProviderListForSession(
    const Environment& env, const OrtSessionOptions& effective_options) {
  provider_list_.clear();

  if (owned_ep_infos_.empty()) {
    return Status::OK();
  }

  const auto& ep_info = owned_ep_infos_[0];
  if (ep_info.ep_name == kCpuExecutionProvider || ep_info.ep_devices.empty()) {
    // CPU is built-in; no provider to register.
    return Status::OK();
  }

  std::unique_ptr<IExecutionProviderFactory> provider_factory;
  ORT_RETURN_IF_ERROR(CreateIExecutionProviderFactoryForEpDevices(
      env,
      gsl::span<const OrtEpDevice* const>(ep_info.ep_devices.data(), ep_info.ep_devices.size()),
      provider_factory));

  const auto& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
  provider_list_.push_back(provider_factory->CreateProvider(effective_options, logger));

  return Status::OK();
}

Status ModelPackageComponentContext::GetSelectedVariantConsumerMetadata(const std::string*& out_json_str) const {
  out_json_str = nullptr;

  const VariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantInfo(selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr,
                "Selected variant is null for component: ", component_model_name_);

  if (!consumer_metadata_cache_valid_) {
    if (selected_variant->consumer_metadata.has_value()) {
      consumer_metadata_cache_ = selected_variant->consumer_metadata->dump();
    } else {
      consumer_metadata_cache_.clear();
    }
    consumer_metadata_cache_valid_ = true;
  }

  out_json_str = &consumer_metadata_cache_;
  return Status::OK();
}

Status ModelPackageComponentContext::GetSelectedVariantName(const std::string*& out_name) const {
  out_name = nullptr;

  const VariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantInfo(selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr,
                "Selected variant is null for component: ", component_model_name_);

  out_name = &selected_variant->variant_name;
  return Status::OK();
}

ModelPackageContext::ModelPackageContext(const std::filesystem::path& package_root) {
  // Use the standalone model_package library for parsing.
  model_package::PackageInfo pkg_info;
  std::string error;
  if (!model_package::ParsePackage(package_root, pkg_info, error)) {
    ORT_THROW("Failed to parse model package: ", error);
  }

  // Convert standalone library types to ORT internal types.
  model_package_info_.schema_version = pkg_info.schema_version;
  model_package_info_.components.clear();
  component_name_to_index_.clear();

  for (const auto& component : pkg_info.components) {
    const auto& name = component.name;
    size_t component_idx = model_package_info_.components.size();
    component_name_to_index_[name] = component_idx;

    ComponentInfo ort_component{};
    ort_component.component_name = name;
    ort_component.selected_variant_index.reset();

    for (const auto& variant : component.variants) {
      VariantInfo ort_variant{};
      ort_variant.component_name = name;
      ort_variant.variant_name = variant.name;
      ort_variant.folder_path = variant.folder_path;

      // Convert EP compatibility (single entry per variant).
      ort_variant.ep_compatibility.ep = variant.ep_compatibility.ep;
      ort_variant.ep_compatibility.device = variant.ep_compatibility.device;
      ort_variant.ep_compatibility.compatibility_string = variant.ep_compatibility.compatibility_string;
      ort_variant.ep_compatibility.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

      // Convert file entry (single file per variant).
      if (variant.file.has_value()) {
        VariantModelInfo ort_file{};
        ort_file.identifier = variant.file->filename;
        ort_file.model_file_path = variant.file->resolved_path;
        ort_file.session_options = variant.file->session_options;
        ort_file.provider_options = variant.file->provider_options;
        ort_file.shared_files = variant.file->shared_files;
        ort_variant.file = std::move(ort_file);
      }

      // Consumer metadata.
      if (variant.consumer_metadata_json.has_value()) {
        ort_variant.consumer_metadata = nlohmann::json::parse(*variant.consumer_metadata_json);
      }

      model_variant_infos_.push_back(ort_variant);
      ort_component.variants.push_back(std::move(ort_variant));
    }

    model_package_info_.components.push_back(std::move(ort_component));
  }

  // Create component names cache for quick lookup.
  component_names_cache_.clear();
  component_names_cache_.reserve(model_package_info_.components.size());

  for (const auto& component : model_package_info_.components) {
    component_names_cache_.push_back(component.component_name);
  }
}

size_t ModelPackageContext::GetComponentCount() const noexcept {
  return model_package_info_.components.size();
}

Status ModelPackageContext::GetComponentNames(gsl::span<const std::string>& out_names) const {
  out_names = gsl::span<const std::string>(component_names_cache_.data(),
                                           component_names_cache_.size());
  return Status::OK();
}

void ModelPackageContext::GetComponentNamePtrs(const char* const*& out_ptrs, size_t& out_count) const {
  if (component_name_ptrs_cache_.empty() && !component_names_cache_.empty()) {
    component_name_ptrs_cache_.reserve(component_names_cache_.size());
    for (const auto& s : component_names_cache_) {
      component_name_ptrs_cache_.push_back(s.c_str());
    }
  }
  out_count = component_name_ptrs_cache_.size();
  out_ptrs = component_name_ptrs_cache_.empty() ? nullptr : component_name_ptrs_cache_.data();
}

void ModelPackageContext::GetVariantNamePtrs(const std::string& component_name,
                                             const char* const*& out_ptrs, size_t& out_count) const {
  // Ensure variant name strings are cached first.
  gsl::span<const std::string> variant_names;
  (void)GetVariantNames(component_name, variant_names);

  auto& ptrs = variant_name_ptrs_cache_[component_name];
  ptrs.clear();
  ptrs.reserve(variant_names.size());
  for (const auto& s : variant_names) {
    ptrs.push_back(s.c_str());
  }
  out_count = ptrs.size();
  out_ptrs = ptrs.empty() ? nullptr : ptrs.data();
}

Status ModelPackageContext::GetVariantCount(const std::string& component_name, size_t& out_count) const {
  out_count = 0;

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  out_count = model_package_info_.components[it->second].variants.size();
  return Status::OK();
}

Status ModelPackageContext::GetVariantNames(const std::string& component_name,
                                            gsl::span<const std::string>& out_variant_names) const {
  out_variant_names = gsl::span<const std::string>{};

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  const auto& variants = model_package_info_.components[it->second].variants;
  component_to_variant_names_cache_[component_name].clear();
  component_to_variant_names_cache_[component_name].reserve(variants.size());

  for (const auto& variant : variants) {
    component_to_variant_names_cache_[component_name].push_back(variant.variant_name);
  }

  out_variant_names = gsl::span<const std::string>(component_to_variant_names_cache_[component_name].data(),
                                                   component_to_variant_names_cache_[component_name].size());
  return Status::OK();
}

namespace {
Status FindVariant(const ModelPackageInfo& model_package_info,
                   const std::unordered_map<std::string, size_t>& component_name_to_index,
                   const std::string& component_name,
                   const std::string& variant_name,
                   const VariantInfo*& out_variant) {
  out_variant = nullptr;
  auto it = component_name_to_index.find(component_name);
  if (it == component_name_to_index.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  const auto& variants = model_package_info.components[it->second].variants;
  for (const auto& v : variants) {
    if (v.variant_name == variant_name) {
      out_variant = &v;
      return Status::OK();
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Variant '", variant_name, "' not found in component '", component_name, "'.");
}
}  // namespace

Status ModelPackageContext::GetVariantEpCompatibility(const std::string& component_name,
                                                      const std::string& variant_name,
                                                      const VariantEpCompatibilityInfo*& out_info) const {
  out_info = nullptr;
  const VariantInfo* variant = nullptr;
  ORT_RETURN_IF_ERROR(FindVariant(model_package_info_, component_name_to_index_,
                                  component_name, variant_name, variant));
  out_info = &variant->ep_compatibility;
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)
}
