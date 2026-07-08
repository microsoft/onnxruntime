// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <cctype>
#include <limits>
#include <string>
#include <unordered_set>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
#include "core/providers/providers.h"
#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package/model_package_options.h"
#include "core/session/model_package/model_package_variant_selector.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_env.h"
#include "core/session/provider_policy_context.h"
#include "core/session/utils.h"

// Use the standalone model_package library's public C API. The library has no ORT
// dependency; ORT links it as a static archive (see cmake/onnxruntime_session.cmake)
// and translates the C handles into the ORT-internal C++ types defined in
// model_package_context.h here.
#include "model_package.h"

namespace onnxruntime {

bool IsModelPackagePathSessionOption(std::string_view key) {
  // Session-option config keys whose values are path references (sha256:<hex>, relative, or
  // absolute) that must be resolved against the model package. Add new path-valued keys here.
  return key == kOrtSessionOptionsModelExternalInitializersFileFolderPath ||
         key == kOrtSessionOptionEpContextFilePath;
}

namespace {
// Deleter for the type-erased model_package handle held by ModelPackageContext.
void CloseModelPackageHandle(void* handle) {
  if (handle != nullptr) {
    ::ModelPackage_Close(static_cast<::ModelPackage*>(handle));
  }
}
}  // namespace

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
  ORT_RETURN_IF(!selected_variant.file.has_value() || selected_variant.file->identifier.empty(),
                "Selected variant '", selected_variant.variant_name,
                "' has no executor_info[\"ort\"][\"model_file\"]. Component: ",
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

ModelPackageContext::ModelPackageContext(const std::filesystem::path& package_root)
    : package_handle_(nullptr, &CloseModelPackageHandle), package_root_(package_root) {
  // Open the package via the model_package C API and keep the handle open for this context's
  // lifetime (owned by package_handle_) so path references can be resolved later without
  // reopening. The unique_ptr releases the handle even on exception paths during conversion.
  ::ModelPackage* pkg = nullptr;
  if (::ModelPackageStatus* st = ::ModelPackage_Open(package_root.string().c_str(), nullptr, &pkg)) {
    std::string msg = ::ModelPackageStatus_Message(st) ? ::ModelPackageStatus_Message(st) : "unknown error";
    ::ModelPackageStatus_Release(st);
    ORT_THROW("Failed to open model package at '", package_root.string(), "': ", msg);
  }
  package_handle_.reset(pkg);

  const ::ModelPackageInfo* pkg_info = ::ModelPackage_Info(pkg);
  model_package_info_.schema_version = pkg_info ? pkg_info->schema_version_major : 0;
  model_package_info_.components.clear();
  component_name_to_index_.clear();

  const size_t component_count = pkg_info ? pkg_info->num_components : 0;
  for (size_t ci = 0; ci < component_count; ++ci) {
    const ::ModelComponentInfo* component = &pkg_info->components[ci];

    std::string component_name = component->name ? component->name : "";
    const size_t component_idx = model_package_info_.components.size();
    component_name_to_index_[component_name] = component_idx;

    ComponentInfo ort_component{};
    ort_component.component_name = component_name;
    ort_component.selected_variant_index.reset();

    const size_t variant_count = component->num_variants;
    for (size_t vi = 0; vi < variant_count; ++vi) {
      const ::ModelVariantInfo* variant = &component->variants[vi];

      VariantInfo ort_variant{};
      ort_variant.component_name = component_name;
      ort_variant.variant_name = variant->name ? variant->name : "";

      // Resolve the variant directory. Absence is treated as a soft signal;
      // downstream callers that require a directory surface a clearer error
      // at the point of use.
      if (variant->variant_directory != nullptr) {
        ort_variant.folder_path = std::filesystem::path(variant->variant_directory);
      }

      // EP compatibility (single entry per variant).
      if (variant->ep != nullptr) ort_variant.ep_compatibility.ep = std::string(variant->ep);
      if (variant->device != nullptr) ort_variant.ep_compatibility.device = std::string(variant->device);
      if (variant->compatibility_string != nullptr)
        ort_variant.ep_compatibility.compatibility_string = std::string(variant->compatibility_string);
      ort_variant.ep_compatibility.compiled_model_compatibility =
          OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

      // Resolve the ORT executor_info from the manifest.
      std::optional<json> ort_obj;
      if (const ::ModelExecutorInfoEntry* ei =
              ::ModelVariantInfo_FindExecutorInfo(variant, "ort")) {
        if (ei->json != nullptr && ei->json[0] != '\0') {
          try {
            ort_obj = json::parse(ei->json);
          } catch (const std::exception& e) {
            ORT_THROW("Failed to parse executor_info[\"ort\"] JSON for variant '",
                      ort_variant.variant_name, "' in component '", component_name, "': ", e.what());
          }
        }
      }

      if (ort_obj.has_value()) {
        if (!ort_obj->is_object()) {
          ORT_THROW("ORT variant configuration must be a JSON object for variant '",
                    ort_variant.variant_name, "' in component '", component_name, "'");
        }

        VariantModelInfo ort_file{};

        // Common resolver for ORT-side string refs (model_file, external_data).
        // Delegates to ModelPackage_ResolveStringRef so accepted forms (relative,
        // absolute, '..', sha256: URI, sha256: URI + subpath) and portable/installed
        // confinement match the rest of the model_package library.
        const std::string base_dir_str = ort_variant.folder_path.string();
        const char* base_dir = base_dir_str.empty() ? nullptr : base_dir_str.c_str();
        auto resolve_string_ref = [&](const char* field, const std::string& input,
                                      bool must_exist) -> std::string {
          const char* resolved = nullptr;
          if (::ModelPackageStatus* st = ::ModelPackage_ResolveStringRef(
                  pkg, base_dir, input.c_str(), must_exist, &resolved)) {
            std::string msg = ::ModelPackageStatus_Message(st) ? ::ModelPackageStatus_Message(st)
                                                               : "unknown error";
            ::ModelPackageStatus_Release(st);
            ORT_THROW("Failed to resolve ORT variant '", field, "' = '", input, "' for variant '",
                      ort_variant.variant_name, "' in component '", component_name, "': ", msg);
          }
          return resolved ? std::string(resolved) : std::string{};
        };

        if (auto it = ort_obj->find("model_file"); it != ort_obj->end()) {
          if (!it->is_string()) {
            ORT_THROW("ORT variant configuration: model_file must be a string for variant '",
                      ort_variant.variant_name, "' in component '", component_name, "'");
          }
          const std::string model_file = it->get<std::string>();
          ort_file.identifier = model_file;
          ort_file.model_file_path = resolve_string_ref("model_file", model_file,
                                                        /*must_exist=*/false);
        }

        auto fill_string_map = [&](const char* key,
                                   std::optional<std::unordered_map<std::string, std::string>>& dest) {
          auto it = ort_obj->find(key);
          if (it == ort_obj->end()) return;
          if (!it->is_object()) {
            ORT_THROW("ORT variant configuration: '", key, "' must be a JSON object for variant '",
                      ort_variant.variant_name, "' in component '", component_name, "'");
          }
          std::unordered_map<std::string, std::string> out;
          out.reserve(it->size());
          for (auto kv = it->begin(); kv != it->end(); ++kv) {
            if (!kv.value().is_string()) {
              ORT_THROW("ORT variant configuration: '", key, "' entries must be strings for variant '",
                        ort_variant.variant_name, "' in component '", component_name, "'");
            }
            out.emplace(kv.key(), kv.value().get<std::string>());
          }
          dest = std::move(out);
        };
        fill_string_map("session_options", ort_file.session_options);
        fill_string_map("provider_options", ort_file.provider_options);

        // Resolve path-valued session options (e.g. the external initializers folder) against the
        // package so variants can reference shared assets by sha256: URI or relative path.
        if (ort_file.session_options.has_value()) {
          for (auto& [key, value] : *ort_file.session_options) {
            if (!value.empty() && IsModelPackagePathSessionOption(key)) {
              value = resolve_string_ref(key.c_str(), value, /*must_exist=*/false);
            }
          }
        }

        if (!ort_file.identifier.empty() || ort_file.session_options.has_value() ||
            ort_file.provider_options.has_value()) {
          ort_variant.file = std::move(ort_file);
        }
      }

      // Variant-scope additional_metadata.
      if (variant->additional_metadata_json != nullptr) {
        try {
          ort_variant.consumer_metadata = json::parse(variant->additional_metadata_json);
        } catch (const std::exception& e) {
          ORT_THROW("Failed to parse additional_metadata JSON for variant '", ort_variant.variant_name,
                    "' in component '", component_name, "': ", e.what());
        }
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

Status ModelPackageContext::ResolveStringRef(const std::string& base_dir,
                                             const std::string& input,
                                             bool must_exist,
                                             const char*& out_path) const {
  out_path = nullptr;
  auto* pkg = static_cast<::ModelPackage*>(package_handle_.get());
  if (pkg == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model package handle is not open");
  }
  const char* resolved = nullptr;
  if (::ModelPackageStatus* st = ::ModelPackage_ResolveStringRef(
          pkg, base_dir.empty() ? nullptr : base_dir.c_str(), input.c_str(), must_exist, &resolved)) {
    std::string msg = ::ModelPackageStatus_Message(st) ? ::ModelPackageStatus_Message(st) : "unknown error";
    ::ModelPackageStatus_Release(st);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to resolve '", input, "' in model package: ", msg);
  }
  // Copy out of the library's transient buffer into a context-owned cache so the returned
  // pointer stays valid until the next ResolveStringRef call.
  resolve_string_ref_cache_ = resolved ? resolved : "";
  out_path = resolve_string_ref_cache_.c_str();
  return Status::OK();
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
