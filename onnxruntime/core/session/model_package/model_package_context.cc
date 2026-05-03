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
#include "core/session/model_package/model_package_context.h"
#include "core/session/model_package/model_package_descriptor_parser.h"
#include "core/session/model_package/model_package_options.h"
#include "core/session/model_package/model_package_variant_selector.h"
#include "core/session/ort_env.h"
#include "core/session/provider_policy_context.h"
#include "core/session/utils.h"

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

ModelPackageContext::ModelPackageContext(const std::filesystem::path& package_root) {
  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  std::vector<ModelVariantInfo>& variants = model_variant_infos_;
  ORT_THROW_IF_ERROR(parser.ParseVariantsFromRoot(package_root, variants));

  model_package_info_.component_models.clear();
  component_name_to_index_.clear();

  // Create model package info cache
  for (const auto& variant : variants) {
    const auto& name = variant.component_model_name;
    size_t component_idx = 0;
    auto it = component_name_to_index_.find(name);
    if (it == component_name_to_index_.end()) {
      component_idx = model_package_info_.component_models.size();
      component_name_to_index_[name] = component_idx;

      ComponentModelInfo component{};
      component.component_model_name = name;
      component.selected_variant_index.reset();  // no selection state used anymore
      model_package_info_.component_models.push_back(std::move(component));
    } else {
      component_idx = it->second;
    }

    model_package_info_.component_models[component_idx].model_variants.push_back(variant);
  }

  // Create component names cache for quick lookup.
  component_names_cache_.clear();
  component_names_cache_.reserve(model_package_info_.component_models.size());

  for (const auto& component : model_package_info_.component_models) {
    component_names_cache_.push_back(component.component_model_name);
  }
}

const ModelPackageOptions* ModelPackageContext::Options() const noexcept {
  return options_;
}

std::vector<std::unique_ptr<IExecutionProvider>>& ModelPackageContext::MutableProviderList() {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->MutableProviderList();
}

const std::vector<const OrtEpDevice*>& ModelPackageContext::ExecutionDevices() const {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->ExecutionDevices();
}

const std::vector<const OrtEpDevice*>& ModelPackageContext::DevicesSelected() const {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->DevicesSelected();
}

bool ModelPackageContext::IsFromPolicy() const {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->FromPolicy();
}

Status ModelPackageContext::ResolveVariant(const ModelPackageOptions* options) {
  ORT_RETURN_IF(options == nullptr,
                "ModelPackageContext::ResolveVariant requires non-null ModelPackageOptions.");

  options_ = options;
  return ResolveVariantImpl(options_->EpInfos());
}

Status ModelPackageContext::ResolveVariant(gsl::span<const VariantSelectionEpInfo> ep_infos) {
  return ResolveVariantImpl(ep_infos);
}

Status ModelPackageContext::ResolveVariantImpl(gsl::span<const VariantSelectionEpInfo> ep_infos) {
  std::optional<ModelVariantInfo> selected_variant;
  ModelVariantSelector selector;
  ORT_RETURN_IF_ERROR(selector.SelectVariant(*this, ep_infos, selected_variant));

  ORT_RETURN_IF(!selected_variant.has_value(),
                "No suitable model variant found for the configured execution providers.");

  // Clear prior selection state before applying new selection.
  for (auto& component : model_package_info_.component_models) {
    component.selected_variant_index.reset();
  }

  // Update selected variant cache for the component that owns this selected model_info.
  size_t matched_variants = 0;
  for (auto& component : model_package_info_.component_models) {
    if (component.component_model_name != selected_variant->component_model_name) {
      continue;
    }

    for (size_t i = 0; i < component.model_variants.size(); ++i) {
      if (component.model_variants[i].variant_name == selected_variant->variant_name) {
        component.selected_variant_index = i;
        ++matched_variants;
      }
    }
  }

  ORT_RETURN_IF(matched_variants == 0,
                "Selected variant was not found in model package context.");
  ORT_RETURN_IF(matched_variants > 1,
                "Selected variant matched multiple variants; selection is ambiguous.");

  return Status::OK();
}

size_t ModelPackageContext::GetComponentModelCount() const noexcept {
  return model_package_info_.component_models.size();
}

Status ModelPackageContext::GetComponentModelNames(gsl::span<const std::string>& out_names) const {
  out_names = gsl::span<const std::string>(component_names_cache_.data(),
                                           component_names_cache_.size());
  return Status::OK();
}

Status ModelPackageContext::GetModelVariantCount(const std::string& component_name, size_t& out_count) const {
  out_count = 0;

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  out_count = model_package_info_.component_models[it->second].model_variants.size();
  return Status::OK();
}

Status ModelPackageContext::GetModelVariantNames(const std::string& component_name,
                                                 gsl::span<const std::string>& out_variant_names) const {
  out_variant_names = gsl::span<const std::string>{};

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  const auto& variants = model_package_info_.component_models[it->second].model_variants;
  component_to_variant_names_cache_[component_name].clear();
  component_to_variant_names_cache_[component_name].reserve(variants.size());

  for (const auto& variant : variants) {
    component_to_variant_names_cache_[component_name].push_back(variant.variant_name);
  }

  out_variant_names = gsl::span<const std::string>(component_to_variant_names_cache_[component_name].data(),
                                                   component_to_variant_names_cache_[component_name].size());
  return Status::OK();
}

Status ModelPackageContext::GetFileCount(const std::string& component_name,
                                         const std::string& variant_name,
                                         size_t& out_count) const {
  out_count = 0;

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  const auto& variants = model_package_info_.component_models[it->second].model_variants;
  for (const auto& variant : variants) {
    if (variant.variant_name == variant_name) {
      out_count = variant.model_info.size();
      return Status::OK();
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Variant '", variant_name, "' not found for component '", component_name, "'.");
}

Status ModelPackageContext::GetFileIdentifiers(const std::string& component_name,
                                               const std::string& variant_name,
                                               gsl::span<const std::string>& out_file_identifiers) const {
  out_file_identifiers = gsl::span<const std::string>{};

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  const auto& variants = model_package_info_.component_models[it->second].model_variants;
  for (const auto& variant : variants) {
    if (variant.variant_name == variant_name) {
      std::string key = component_name + ":" + variant_name;
      variant_to_file_identifiers_cache_[key].clear();
      variant_to_file_identifiers_cache_[key].reserve(variant.model_info.size());

      for (const auto& mi : variant.model_info) {
        variant_to_file_identifiers_cache_[key].push_back(mi.identifier);
      }

      out_file_identifiers = gsl::span<const std::string>(variant_to_file_identifiers_cache_[key].data(),
                                                          variant_to_file_identifiers_cache_[key].size());
      return Status::OK();
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Variant '", variant_name, "' not found for component '", component_name, "'.");
}

Status ModelPackageContext::GetFilePath(const std::string& component_name,
                                        const std::string& variant_name,
                                        const char* file_identifier,
                                        std::filesystem::path& out_path) const {
  out_path.clear();

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  const auto& variants = model_package_info_.component_models[it->second].model_variants;
  for (const auto& variant : variants) {
    if (variant.variant_name != variant_name) {
      continue;
    }

    if (file_identifier == nullptr || std::string_view(file_identifier).empty()) {
      if (variant.model_info.size() == 1) {
        out_path = variant.model_info.front().model_file_path;
        return Status::OK();
      }

      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "file_identifier is required when variant has multiple files. component='",
                             component_name, "', variant='", variant_name, "'.");
    }

    for (const auto& mi : variant.model_info) {
      if (mi.identifier == file_identifier) {
        out_path = mi.model_file_path;
        return Status::OK();
      }
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Unknown file_identifier '", file_identifier,
                           "' for component '", component_name,
                           "', variant '", variant_name, "'.");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Variant '", variant_name, "' not found for component '", component_name, "'.");
}

Status ModelPackageContext::GetSelectedVariantForComponent(const std::string& component_name,
                                                           const ModelVariantInfo*& out_variant) const {
  out_variant = nullptr;

  auto it = component_name_to_index_.find(component_name);
  if (it == component_name_to_index_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Component model not found: ", component_name);
  }

  const auto& component = model_package_info_.component_models[it->second];
  if (component.model_variants.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Component has no variants: ", component_name);
  }

  if (!component.selected_variant_index.has_value()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "No variant is selected for component: ", component_name);
  }

  const size_t selected_idx = *component.selected_variant_index;
  if (selected_idx >= component.model_variants.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Selected variant index out of range for component: ", component_name);
  }

  out_variant = &component.model_variants[selected_idx];
  return Status::OK();
}

Status ModelPackageContext::GetSelectedVariantModelInfo(const std::string& component_name,
                                                        const char* file_identifier,
                                                        const VariantModelInfo*& out_model_info) const {
  out_model_info = nullptr;

  const ModelVariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantForComponent(component_name, selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr, "Selected variant is null for component: ", component_name);

  if (file_identifier == nullptr || std::string_view(file_identifier).empty()) {
    if (selected_variant->model_info.size() == 1) {
      out_model_info = &selected_variant->model_info.front();
      return Status::OK();
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "file_identifier is required when selected variant has multiple model_info entries.");
  }

  for (const auto& mi : selected_variant->model_info) {
    if (mi.identifier == file_identifier) {
      out_model_info = &mi;
      return Status::OK();
    }
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                         "Unknown file_identifier '", file_identifier,
                         "' for component '", component_name, "'.");
}

Status ModelPackageContext::GetSelectedVariantFileIdentifiers(const std::string& component_name,
                                                              gsl::span<const std::string>& out_file_identifiers) const {
  out_file_identifiers = gsl::span<const std::string>{};

  const ModelVariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantForComponent(component_name, selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr, "Selected variant is null for component: ", component_name);

  selected_variant_file_identifiers_cache_.clear();
  selected_variant_file_identifiers_cache_.reserve(selected_variant->model_info.size());

  // Preserve schema order (do not sort).
  for (const auto& mi : selected_variant->model_info) {
    selected_variant_file_identifiers_cache_.push_back(mi.identifier);
  }

  out_file_identifiers = gsl::span<const std::string>(
      selected_variant_file_identifiers_cache_.data(),
      selected_variant_file_identifiers_cache_.size());

  return Status::OK();
}

Status ModelPackageContext::ResolveSelectedVariantFilePath(const std::string& component_name,
                                                           const char* file_identifier,
                                                           std::filesystem::path& out_path) const {
  out_path.clear();

  const VariantModelInfo* selected_model_info = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantModelInfo(component_name, file_identifier, selected_model_info));
  ORT_RETURN_IF(selected_model_info == nullptr,
                "Selected model info is null for component: ", component_name);

  out_path = selected_model_info->model_file_path;
  return Status::OK();
}

Status ModelPackageContext::GetSelectedVariantFilePath(std::filesystem::path& out_path) const {
  out_path.clear();

  const size_t component_count = GetComponentModelCount();
  if (component_count != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ResolveSelectedVariantFileForSingleComponent requires exactly one component model, found: ",
                           component_count);
  }

  gsl::span<const std::string> component_names;
  ORT_RETURN_IF_ERROR(GetComponentModelNames(component_names));
  ORT_RETURN_IF(component_names.empty(),
                "Single component model name is missing.");

  const std::string& component_name = component_names.front();

  const ModelVariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantForComponent(component_name, selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr,
                "Selected variant is null for single component model: ", component_name);

  if (selected_variant->model_info.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ResolveSelectedVariantFileForSingleComponent requires selected variant to have exactly one model_info entry, found: ",
                           selected_variant->model_info.size(),
                           ", component: ", component_name,
                           ", variant: ", selected_variant->variant_name);
  }

  out_path = selected_variant->model_info.front().model_file_path;
  return Status::OK();
}

Status ModelPackageContext::GetSelectedVariantFileSessionOptions(const std::string& component_name,
                                                                 const char* file_identifier,
                                                                 gsl::span<const std::string>& out_keys,
                                                                 gsl::span<const std::string>& out_values) const {
  out_keys = {};
  out_values = {};

  const VariantModelInfo* mi = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantModelInfo(component_name, file_identifier, mi));
  ORT_RETURN_IF(mi == nullptr, "Selected model info is null for component: ", component_name);

  ORT_RETURN_IF(mi->ep_compatibility.empty(),
                "Selected model info has no ep_compatibility entries.");

  size_t ec_idx = 0;
  if (mi->selected_ep_compatibility_index.has_value()) {
    ec_idx = *mi->selected_ep_compatibility_index;
    ORT_RETURN_IF(ec_idx >= mi->ep_compatibility.size(),
                  "selected_ep_compatibility_index out of range.");
  } else if (mi->ep_compatibility.size() > 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "selected_ep_compatibility_index is not set for selected model_info with multiple ep_compatibility entries.");
  }

  return FillOptionCachesFromMap(mi->ep_compatibility[ec_idx].session_options,
                                 selected_variant_session_option_keys_cache_,
                                 selected_variant_session_option_values_cache_,
                                 out_keys,
                                 out_values);
}

Status ModelPackageContext::GetSelectedVariantFileProviderOptions(const std::string& component_name,
                                                                  const char* file_identifier,
                                                                  gsl::span<const std::string>& out_keys,
                                                                  gsl::span<const std::string>& out_values) const {
  out_keys = {};
  out_values = {};

  const VariantModelInfo* mi = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantModelInfo(component_name, file_identifier, mi));
  ORT_RETURN_IF(mi == nullptr, "Selected model info is null for component: ", component_name);

  ORT_RETURN_IF(mi->ep_compatibility.empty(),
                "Selected model info has no ep_compatibility entries.");

  size_t ec_idx = 0;
  if (mi->selected_ep_compatibility_index.has_value()) {
    ec_idx = *mi->selected_ep_compatibility_index;
    ORT_RETURN_IF(ec_idx >= mi->ep_compatibility.size(),
                  "selected_ep_compatibility_index out of range.");
  } else if (mi->ep_compatibility.size() > 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "selected_ep_compatibility_index is not set for selected model_info with multiple ep_compatibility entries.");
  }

  return FillOptionCachesFromMap(mi->ep_compatibility[ec_idx].provider_options,
                                 selected_variant_provider_option_keys_cache_,
                                 selected_variant_provider_option_values_cache_,
                                 out_keys,
                                 out_values);
}

#endif  // !defined(ORT_MINIMAL_BUILD)
}
