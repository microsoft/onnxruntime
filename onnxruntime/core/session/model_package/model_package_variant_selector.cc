// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_variant_selector.h"
#include "core/session/model_package/model_package_context.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/utils.h"

namespace onnxruntime {
namespace {

struct VariantMatchResult {
  bool matched{false};
  int score{std::numeric_limits<int>::min()};
  std::vector<VariantModelInfo> selected_model_infos{};
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

  // ALL model_info must match for variant to match.
  for (auto& mi : variant.model_info) {
    int best_score_for_mi = std::numeric_limits<int>::min();
    if (!TryMatchModelInfoForEp(mi, ep_info, best_score_for_mi)) {
      return result;
    }
    total_score += best_score_for_mi;
    result.selected_model_infos.push_back(mi);  // keep selected_ep_compatibility_index for each model_info
  }

  // Normalize score by model_info count.
  const int n = static_cast<int>(variant.model_info.size());
  const int normalized_score = (total_score + n / 2) / n;

  result.matched = true;
  result.score = normalized_score;
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

    VariantModelInfo selected_mi = mi;
    selected_mi.selected_ep_compatibility_index = unconstrained_ec_index;
    result.selected_model_infos.push_back(std::move(selected_mi));
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
                                           std::optional<ModelVariantInfo>& selected_variant) const {
  selected_variant.reset();

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
  ORT_RETURN_IF(best_match.selected_model_infos.empty(),
                "Selected variant has no selected model_infos.");

  selected_variant = variants[best_index];
  selected_variant->model_info = std::move(best_match.selected_model_infos);
  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)