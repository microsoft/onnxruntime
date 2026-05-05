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
  std::optional<size_t> selected_ep_compatibility_index{};
};

std::string ToLower(std::string_view s) {
  std::string result(s);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

bool MatchesDevice(const OrtHardwareDevice* hd, std::string_view value) {
  if (value.empty() || hd == nullptr) {
    return value.empty();
  }

  const std::string device = ToLower(value);
  switch (hd->type) {
    case OrtHardwareDeviceType::OrtHardwareDeviceType_CPU:
      return device == "cpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_GPU:
      return device == "gpu";
    case OrtHardwareDeviceType::OrtHardwareDeviceType_NPU:
      return device == "npu";
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
                                              const std::string& compatibility_string,
                                              std::vector<const OrtHardwareDevice*>& constraint_devices,
                                              OrtCompiledModelCompatibility* compiled_model_compatibility) {
  if (compatibility_string.empty()) {
    LOGS_DEFAULT(INFO) << "No compatibility_string constraint. Skip compatibility validation.";
    return Status::OK();
  }

  auto* ep_factory = ep_info.ep_factory;

  if (ep_factory &&
      ep_factory->ort_version_supported >= 23 &&
      ep_factory->ValidateCompiledModelCompatibilityInfo != nullptr) {
    auto status = ep_factory->ValidateCompiledModelCompatibilityInfo(ep_factory,
                                                                     constraint_devices.data(),
                                                                     constraint_devices.size(),
                                                                     compatibility_string.c_str(),
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

  if (ec.device.has_value() && !ec.device->empty()) {
    score += 5;
  }

  return score;
}

bool IsUnconstrainedEpCompatibility(const VariantEpCompatibilityInfo& ec) {
  const bool no_ep = !ec.ep.has_value() || ec.ep->empty();
  const bool no_device = !ec.device.has_value() || ec.device->empty();
  const bool no_compat = !ec.compatibility_string.has_value() || ec.compatibility_string->empty();
  return no_ep && no_device && no_compat;
}

VariantMatchResult MatchVariantForEp(ModelVariantInfo& variant, const VariantSelectionEpInfo& ep_info) {
  VariantMatchResult result{};
  if (variant.ep_compatibility.empty()) {
    return result;
  }

  int best_score = std::numeric_limits<int>::min();

  for (size_t ec_idx = 0; ec_idx < variant.ep_compatibility.size(); ++ec_idx) {
    auto& ec = variant.ep_compatibility[ec_idx];

    if (ec.ep.has_value() && !ec.ep->empty() && *ec.ep != ep_info.ep_name) {
      continue;
    }

    bool device_ok = !ec.device.has_value() || ec.device->empty();
    std::vector<const OrtHardwareDevice*> constraint_devices = ep_info.hardware_devices;

    if (ep_info.hardware_devices.empty()) {
      device_ok = true;
    } else if (!device_ok) {
      if (const auto* matched_hd = FindMatchingHardwareDevice(*ec.device, ep_info.hardware_devices)) {
        device_ok = true;
        constraint_devices = {matched_hd};
      }
    }

    if (!device_ok) {
      continue;
    }

    ec.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    auto st = ValidateCompiledModelCompatibilityInfo(ep_info,
                                                     ec.compatibility_string.value_or(""),
                                                     constraint_devices,
                                                     &ec.compiled_model_compatibility);
    if (!st.IsOK()) {
      ec.compiled_model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    }

    if (ec.compiled_model_compatibility == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
      continue;
    }

    const int score = ScoreEpCompatibility(ec);
    if (!result.matched || score > best_score) {
      result.matched = true;
      best_score = score;
      result.selected_ep_compatibility_index = ec_idx;
    }
  }

  result.score = best_score;
  return result;
}

VariantMatchResult MatchUnconstrainedVariant(const ModelVariantInfo& variant) {
  VariantMatchResult result{};

  for (size_t ec_idx = 0; ec_idx < variant.ep_compatibility.size(); ++ec_idx) {
    if (IsUnconstrainedEpCompatibility(variant.ep_compatibility[ec_idx])) {
      result.matched = true;
      result.score = 0;
      result.selected_ep_compatibility_index = ec_idx;
      break;
    }
  }

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
  int best = std::numeric_limits<int>::min();
  for (const auto& ec : variant.ep_compatibility) {
    best = std::max(best, ScoreEpCompatibility(ec));
  }
  return best;
}

Status ModelVariantSelector::SelectVariant(const ModelPackageComponentContext& context,
                                           gsl::span<const VariantSelectionEpInfo> ep_infos,
                                           std::optional<ModelVariantInfo>& selected_variant) const {
  selected_variant.reset();

  std::vector<ModelVariantInfo> variants = context.GetModelVariantInfos();
  if (variants.empty()) {
    return Status::OK();
  }

  const VariantSelectionEpInfo* selected_ep_info = nullptr;
  if (ep_infos.size() > 1) {
    LOGS_DEFAULT(WARNING) << "Multiple EP infos provided; only first ep '" << ep_infos[0].ep_name << "' is used.";
  }
  if (!ep_infos.empty()) {
    selected_ep_info = &ep_infos[0];
  }

  std::unordered_set<size_t> candidate_indices_set;
  std::unordered_map<size_t, VariantMatchResult> candidate_matches;

  // 1) Unconstrained variants
  for (size_t i = 0, end = variants.size(); i < end; ++i) {
    VariantMatchResult m = MatchUnconstrainedVariant(variants[i]);
    if (m.matched) {
      candidate_indices_set.insert(i);
      candidate_matches[i] = std::move(m);
    }
  }

  // 2) EP/device compatibility: all constraints in ep compatibility info need to be matched.
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

  selected_variant = std::move(variants[best_index]);
  selected_variant->selected_ep_compatibility_index = candidate_matches[best_index].selected_ep_compatibility_index;
  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)