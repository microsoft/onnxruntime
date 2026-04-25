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
#include "core/session/ort_env.h"
#include "core/session/provider_policy_context.h"
#include "core/session/utils.h"

namespace onnxruntime {
namespace {

struct VariantMatchResult {
  bool matched{false};
  int score{std::numeric_limits<int>::min()};
  std::optional<VariantModelInfo> selected_model_info{};
};

std::string ToLower(std::string_view s) {
  std::string result(s);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

bool MatchesDevice(const OrtHardwareDevice* hd, std::string_view value) {
  if (value.empty() || hd == nullptr) {
    return value.empty();
  }

  const std::string device_type = ToLower(value);
  switch (hd->type) {
    case OrtHardwareDeviceType::OrtHardwareDeviceType_CPU:
      return device_type == "cpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_GPU:
      return device_type == "gpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_NPU:
      return device_type == "npu";
    default:
      return false;
  }
}

const OrtHardwareDevice* FindMatchingHardwareDevice(std::string_view device_constraint,
                                                    gsl::span<const OrtHardwareDevice* const> hardware_devices) {
  if (device_constraint.empty()) {
    return nullptr;
  }

  for (const auto* hd : hardware_devices) {
    if (MatchesDevice(hd, device_constraint)) {
      return hd;
    }
  }

  return nullptr;
}

Status ValidateCompiledModelCompatibilityInfo(const VariantSelectionEpInfo& ep_info,
                                              const std::string& compatibility_info,
                                              std::vector<const OrtHardwareDevice*>& constraint_devices,
                                              OrtCompiledModelCompatibility* compiled_model_compatibility) {
  if (compatibility_info.empty()) {
    LOGS_DEFAULT(INFO) << "No compatibility info constraint for this variant. Skip compatibility validation.";
    return Status::OK();
  }

  auto* ep_factory = ep_info.ep_factory;

  if (ep_factory &&
      ep_factory->ort_version_supported >= 23 &&
      ep_factory->ValidateCompiledModelCompatibilityInfo != nullptr) {
    auto status = ep_factory->ValidateCompiledModelCompatibilityInfo(ep_factory,
                                                                     constraint_devices.data(),
                                                                     constraint_devices.size(),
                                                                     compatibility_info.c_str(),
                                                                     compiled_model_compatibility);
    ORT_RETURN_IF_ERROR(ToStatusAndRelease(status));
  }

  return Status::OK();
}

int ScoreEpCompatibility(const VariantEpCompatibilityInfo& ec) {
  int score = 0;
  if (ec.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) {
    score += 100;
  } else if (ec.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION) {
    score += 50;
  }

  if (ec.ep.has_value() && !ec.ep->empty()) {
    score += 10;
  }

  return score;
}

bool IsUnconstrainedEpCompatibility(const VariantEpCompatibilityInfo& ec) {
  const bool no_ep = !ec.ep.has_value() || ec.ep->empty();
  const bool no_device = !ec.device_type.has_value() || ec.device_type->empty();
  const bool no_compat = !ec.compatibility_info.has_value() || ec.compatibility_info->empty();
  return no_ep && no_device && no_compat;
}

bool TryMatchModelInfoForEp(VariantModelInfo& model_info,
                            const VariantSelectionEpInfo& ep_info,
                            int& best_score_for_model_info) {
  model_info.selected_ep_compatibility_index.reset();
  best_score_for_model_info = std::numeric_limits<int>::min();
  bool matched = false;

  for (size_t ec_idx = 0; ec_idx < model_info.ep_compatibility.size(); ++ec_idx) {
    auto& ec = model_info.ep_compatibility[ec_idx];

    if (ec.ep.has_value() && !ec.ep->empty() && *ec.ep != ep_info.ep_name) {
      continue;
    }

    bool device_ok = !ec.device_type.has_value() || ec.device_type->empty();
    std::vector<const OrtHardwareDevice*> constraint_devices = ep_info.hardware_devices;

    if (ep_info.hardware_devices.empty()) {
      device_ok = true;
    } else if (!device_ok) {
      if (const auto* matched_hd = FindMatchingHardwareDevice(*ec.device_type, ep_info.hardware_devices)) {
        device_ok = true;
        constraint_devices = {matched_hd};
      }
    }

    if (!device_ok) {
      continue;
    }

    ec.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    auto st = ValidateCompiledModelCompatibilityInfo(ep_info,
                                                     ec.compatibility_info.value_or(""),
                                                     constraint_devices,
                                                     &ec.compiled_model_compatibility);
    if (!st.IsOK()) {
      ec.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    }

    if (ec.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
      continue;
    }

    const int score = ScoreEpCompatibility(ec);
    if (!matched || score > best_score_for_model_info) {
      matched = true;
      best_score_for_model_info = score;
      model_info.selected_ep_compatibility_index = ec_idx;
    }
  }

  return matched;
}

VariantMatchResult MatchVariantForEp(ModelVariantInfo& variant, const VariantSelectionEpInfo& ep_info) {
  VariantMatchResult result{};
  if (variant.model_info.empty()) {
    return result;
  }

  int total_score = 0;
  int best_model_info_score = std::numeric_limits<int>::min();
  size_t best_model_info_idx = 0;

  // ALL model_info must match for variant to match.
  for (size_t mi_idx = 0; mi_idx < variant.model_info.size(); ++mi_idx) {
    auto& mi = variant.model_info[mi_idx];
    int best_score_for_mi = std::numeric_limits<int>::min();

    if (!TryMatchModelInfoForEp(mi, ep_info, best_score_for_mi)) {
      return result;  // variant does not match
    }

    total_score += best_score_for_mi;
    if (best_score_for_mi > best_model_info_score) {
      best_model_info_score = best_score_for_mi;
      best_model_info_idx = mi_idx;
    }
  }

  // Normalize score by model_info count.
  const int n = static_cast<int>(variant.model_info.size());
  const int normalized_score = (total_score + n / 2) / n;

  result.matched = true;
  result.score = normalized_score;
  result.selected_model_info = variant.model_info[best_model_info_idx];
  return result;
}

VariantMatchResult MatchUnconstrainedVariant(const ModelVariantInfo& variant) {
  VariantMatchResult result{};

  if (variant.model_info.empty()) {
    return result;
  }

  for (const auto& mi : variant.model_info) {
    std::optional<size_t> unconstrained_ec_index;

    for (size_t ec_idx = 0; ec_idx < mi.ep_compatibility.size(); ++ec_idx) {
      const auto& ec = mi.ep_compatibility[ec_idx];
      const bool no_ep = !ec.ep.has_value() || ec.ep->empty();
      const bool no_device = !ec.device_type.has_value() || ec.device_type->empty();
      const bool no_compat = !ec.compatibility_info.has_value() || ec.compatibility_info->empty();

      if (no_ep && no_device && no_compat) {
        unconstrained_ec_index = ec_idx;
        break;
      }
    }

    // All model_info entries must have an unconstrained ep_compatibility.
    if (!unconstrained_ec_index.has_value()) {
      return result;
    }

    // Pick a deterministic selected_model_info for APIs that need one.
    if (!result.selected_model_info.has_value()) {
      VariantModelInfo selected = mi;
      selected.selected_ep_compatibility_index = unconstrained_ec_index;
      result.selected_model_info = std::move(selected);
    }
  }

  result.matched = true;
  result.score = 0;
  return result;
}

}  // namespace

// Calculate a score for the model variant based on its constraints and metadata.
//
// It's only used to choose the best model variant among multiple candidates that match constraints.
// Higher score means more preferred.
//
// For example:
// If one model variant/EPContext is compatible with the EP and has compatiliby value indicating optimal compatibility
// (i.e. compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL) while another model variant/EPContext
// is also compatible with the EP but has compatibility value indicating prefer recompilation
// (i.e. compiled_model_compatibility == OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION),
// the former will have a higher score and thus be selected.
//
int ModelVariantSelector::CalculateVariantScore(const ModelVariantInfo& variant) const {
  int total = 0;

  for (const auto& mi : variant.model_info) {
    int best_for_mi = std::numeric_limits<int>::min();
    for (const auto& ec : mi.ep_compatibility) {
      best_for_mi = std::max(best_for_mi, ScoreEpCompatibility(ec));
    }

    if (best_for_mi == std::numeric_limits<int>::min()) {
      return std::numeric_limits<int>::min();
    }

    total += best_for_mi;
  }

  return total;
}

Status ModelVariantSelector::SelectVariant(const ModelPackageContext& context,
                                           gsl::span<VariantSelectionEpInfo> ep_infos,
                                           std::optional<VariantModelInfo>& selected_model_info) const {
  selected_model_info.reset();

  std::vector<ModelVariantInfo> variants = context.GetModelVariantInfos();
  if (variants.empty()) {
    return Status::OK();
  }

  const VariantSelectionEpInfo* selected_ep_info = nullptr;
  if (ep_infos.size() > 1) {
    LOGS_DEFAULT(WARNING) << "Multiple EP info provided for model variant selection, but only the first one with ep name '"
                          << ep_infos[0].ep_name << "' will be used.";
  }
  if (!ep_infos.empty()) {
    selected_ep_info = &ep_infos[0];
  }

  std::unordered_set<size_t> candidate_indices_set;
  std::unordered_map<size_t, VariantMatchResult> candidate_matches;

  // 1) Unconstrained variants (all model_info unconstrained).
  for (size_t i = 0, end = variants.size(); i < end; ++i) {
    VariantMatchResult m = MatchUnconstrainedVariant(variants[i]);
    if (m.matched) {
      candidate_indices_set.insert(i);
      candidate_matches[i] = std::move(m);
    }
  }

  // 2) EP/device compatibility: all model_info must match.
  if (selected_ep_info != nullptr) {
    for (size_t i = 0, end = variants.size(); i < end; ++i) {
      VariantMatchResult m = MatchVariantForEp(variants[i], *selected_ep_info);
      if (m.matched) {
        candidate_indices_set.insert(i);
        auto it = candidate_matches.find(i);
        if (it == candidate_matches.end() || m.score > it->second.score) {
          candidate_matches[i] = std::move(m);
        }
      }
    }
  }

  if (candidate_indices_set.empty()) {
    return Status::OK();
  }

  // choose best
  int best_score = std::numeric_limits<int>::min();
  size_t best_index = *candidate_indices_set.begin();

  for (size_t idx : candidate_indices_set) {
    const int score = candidate_matches[idx].score;
    if (score > best_score) {
      best_score = score;
      best_index = idx;
    }
  }

  const auto& best_match = candidate_matches[best_index];
  ORT_RETURN_IF(!best_match.selected_model_info.has_value(),
                "Selected variant has no selected model info.");

  selected_model_info = best_match.selected_model_info;
  return Status::OK();
}

ModelPackageContext::ModelPackageContext(const onnxruntime::Environment& env,
                                         const std::filesystem::path& package_root,
                                         std::vector<VariantSelectionEpInfo> ep_infos) : env_(env), ep_infos_(std::move(ep_infos)) {
  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  ORT_THROW_IF_ERROR(parser.ParseVariantsFromRoot(package_root, model_variant_infos_));
  BuildComponentModelCache();
}

ModelPackageContext::ModelPackageContext(const onnxruntime::Environment& env,
                                         const std::filesystem::path& package_root,
                                         const ModelPackageOptions& options) : env_(env), options_(&options) {
  ModelPackageDescriptorParser parser(logging::LoggingManager::DefaultLogger());
  ORT_THROW_IF_ERROR(parser.ParseVariantsFromPackageRoot(package_root, model_variant_infos_));
  BuildComponentModelCache();
}

const ModelPackageOptions* ModelPackageContext::Options() const noexcept {
  return options_;
}

std::vector<std::unique_ptr<IExecutionProvider>>& ModelPackageContext::MutableProviderList() noexcept {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->MutableProviderList();
}

const std::vector<const OrtEpDevice*>& ModelPackageContext::ExecutionDevices() const noexcept {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->ExecutionDevices();
}

const std::vector<const OrtEpDevice*>& ModelPackageContext::DevicesSelected() const noexcept {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->DevicesSelected();
}

bool ModelPackageContext::IsFromPolicy() const noexcept {
  ORT_ENFORCE(options_ != nullptr, "ModelPackageContext has no associated ModelPackageOptions.");
  return options_->FromPolicy();
}

Status ModelPackageContext::ResolveVariant() {
  // Determine EP infos source:
  // 1) explicit ep_infos passed to context ctor, or
  // 2) resolved infos from options.
  std::vector<VariantSelectionEpInfo> ep_infos;
  if (!ep_infos_.empty()) {
    ep_infos = ep_infos_;
  } else {
    ORT_RETURN_IF(options_ == nullptr,
                  "ModelPackageContext requires either explicit ep_infos or associated ModelPackageOptions.");
    ep_infos = options_->EpInfos();
  }

  std::optional<VariantModelInfo> selected_model_info;
  ModelVariantSelector selector;
  ORT_RETURN_IF_ERROR(selector.SelectVariant(*this, ep_infos, selected_model_info));

  ORT_RETURN_IF(!selected_model_info.has_value(),
                "No suitable model variant found for the configured execution providers.");

  // Clear prior selection state before applying new selection.
  for (auto& component : model_package_info_.component_models) {
    component.selected_variant_index.reset();
  }

  // Update selected variant cache for the component that owns this selected model_info.
  size_t matched_variants = 0;
  for (auto& component : model_package_info_.component_models) {
    for (size_t i = 0; i < component.model_variants.size(); ++i) {
      const auto& variant = component.model_variants[i];
      for (const auto& mi : variant.model_info) {
        if (mi.identifier == selected_model_info->identifier &&
            mi.model_file_path == selected_model_info->model_file_path) {
          component.selected_variant_index = i;
          ++matched_variants;
        }
      }
    }
  }

  ORT_RETURN_IF(matched_variants == 0,
                "Selected model_info was not found in model package context.");
  ORT_RETURN_IF(matched_variants > 1,
                "Selected model_info matched multiple variants; selection is ambiguous.");

  return Status::OK();
}

size_t ModelPackageContext::GetComponentModelCount() const noexcept {
  return model_package_info_.component_models.size();
}

Status ModelPackageContext::GetComponentModelName(size_t component_index, const std::string*& out_name) const {
  out_name = nullptr;

  if (component_index >= model_package_info_.component_models.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "component_index out of range: ", component_index,
                           ", component count: ", model_package_info_.component_models.size());
  }

  out_name = &model_package_info_.component_models[component_index].component_model_name;
  return Status::OK();
}

void ModelPackageContext::BuildComponentModelCache() {
  model_package_info_.component_models.clear();
  component_name_to_index_.clear();

  for (const auto& variant : model_variant_infos_) {
    const auto& name = variant.component_model_name;

    size_t component_idx = 0;
    auto it = component_name_to_index_.find(name);
    if (it == component_name_to_index_.end()) {
      component_idx = model_package_info_.component_models.size();
      component_name_to_index_[name] = component_idx;

      ComponentModelInfo component{};
      component.component_model_name = name;
      component.selected_variant_index = std::nullopt;
      model_package_info_.component_models.push_back(std::move(component));
    } else {
      component_idx = it->second;
    }

    auto& component = model_package_info_.component_models[component_idx];
    component.model_variants.push_back(variant);
  }
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

  const std::string* component_name = nullptr;
  ORT_RETURN_IF_ERROR(GetComponentModelName(0, component_name));
  ORT_RETURN_IF(component_name == nullptr || component_name->empty(),
                "Single component model name is null or empty.");

  const ModelVariantInfo* selected_variant = nullptr;
  ORT_RETURN_IF_ERROR(GetSelectedVariantForComponent(*component_name, selected_variant));
  ORT_RETURN_IF(selected_variant == nullptr,
                "Selected variant is null for single component model: ", *component_name);

  if (selected_variant->model_info.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ResolveSelectedVariantFileForSingleComponent requires selected variant to have exactly one model_info entry, found: ",
                           selected_variant->model_info.size(),
                           ", component: ", *component_name,
                           ", variant: ", selected_variant->variant_name);
  }

  out_path = selected_variant->model_info.front().model_file_path;
  return Status::OK();
}

namespace {

bool IsUnconstrainedVariant(const ModelVariantInfo& variant) {
  for (const auto& model_info : variant.model_info) {
    for (const auto& ec : model_info.ep_compatibility) {
      const bool no_ep = !ec.ep.has_value() || ec.ep->empty();
      const bool no_device = !ec.device_type.has_value() || ec.device_type->empty();
      const bool no_compat = !ec.compatibility_info.has_value() || ec.compatibility_info->empty();
      if (no_ep && no_device && no_compat) {
        return true;
      }
    }
  }

  return false;
}

}  // namespace

#endif  // !defined(ORT_MINIMAL_BUILD)
}
