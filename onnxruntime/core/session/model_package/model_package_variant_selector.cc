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

bool IsUnconstrainedEpCompatibility(const VariantEpCompatibilityInfo& ec) {
  const bool no_ep = !ec.ep.has_value() || ec.ep->empty();
  const bool no_device = !ec.device.has_value() || ec.device->empty();
  const bool no_compat = !ec.compatibility_strings.has_value() || ec.compatibility_strings->empty();
  return no_ep && no_device && no_compat;
}

int CompatibilityToScore(OrtCompiledModelCompatibility compatibility) {
  switch (compatibility) {
    case OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL:
      return 100;
    case OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION:
      return 50;
    case OrtCompiledModelCompatibility_EP_NOT_APPLICABLE:
      return 0;
    case OrtCompiledModelCompatibility_EP_UNSUPPORTED:
    default:
      return -100;
  }
}

Status ValidateCompiledModelCompatibilityInfoSingle(const VariantSelectionEpInfo& ep_info,
                                                    const std::string& compatibility_string,
                                                    std::vector<const OrtHardwareDevice*>& constraint_devices,
                                                    OrtCompiledModelCompatibility* compiled_model_compatibility) {
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

Status ValidateCompiledModelCompatibilityInfos(const VariantSelectionEpInfo& ep_info,
                                               const std::optional<std::vector<std::string>>& compatibility_strings,
                                               std::vector<const OrtHardwareDevice*>& constraint_devices,
                                               std::vector<OrtCompiledModelCompatibility>& out_compats) {
  out_compats.clear();

  if (!compatibility_strings.has_value()) {
    return Status::OK();
  }

  out_compats.assign(compatibility_strings->size(),
                     OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);

  for (size_t i = 0; i < compatibility_strings->size(); ++i) {
    const std::string& compatibility_string = (*compatibility_strings)[i];
    if (compatibility_string.empty()) {
      continue;
    }

    ORT_RETURN_IF_ERROR(ValidateCompiledModelCompatibilityInfoSingle(
        ep_info, compatibility_string, constraint_devices, &out_compats[i]));
  }

  return Status::OK();
}

VariantMatchResult MatchVariantForEp(ModelVariantInfo& variant, const VariantSelectionEpInfo& ep_info) {
  VariantMatchResult result{};
  if (variant.ep_compatibility.empty()) {
    return result;
  }

  for (size_t ec_idx = 0; ec_idx < variant.ep_compatibility.size(); ++ec_idx) {
    auto& ec = variant.ep_compatibility[ec_idx];

    // 1. Match EP
    if (ec.ep.has_value() && !ec.ep->empty() && *ec.ep != ep_info.ep_name) {
      continue;
    }

    // 2. Match device
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

    // 3. Check ep compatibility string for each model in the variant

    // Keep vector size aligned with compatibility_strings size.
    const size_t compat_count = ec.compatibility_strings.has_value() ? ec.compatibility_strings->size() : 0;
    ec.compiled_model_compatibilities.assign(
        compat_count, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);

    auto st = ValidateCompiledModelCompatibilityInfos(
        ep_info, ec.compatibility_strings, constraint_devices, ec.compiled_model_compatibilities);

    if (!st.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to validate compiled model compatibility for variant '"
                            << variant.variant_name << "', ep compatibility index " << ec_idx
                            << ". Error: " << st.ErrorMessage()
                            << " Skip this ep compatibility.";
      continue;
    }

    // Aggregate model(file)-level scores for this ep compatibility.
    int sum = 0;
    if (ec.compiled_model_compatibilities.empty()) {
      sum = CompatibilityToScore(OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
    } else {
      const bool has_unsupported = std::any_of(
          ec.compiled_model_compatibilities.begin(),
          ec.compiled_model_compatibilities.end(),
          [](OrtCompiledModelCompatibility c) {
            return c == OrtCompiledModelCompatibility_EP_UNSUPPORTED;
          });

      if (has_unsupported) {
        continue;
      }

      for (auto c : ec.compiled_model_compatibilities) {
        sum += CompatibilityToScore(c);
      }
    }

    const int denominator = static_cast<int>(
        ec.compiled_model_compatibilities.empty() ? 1 : ec.compiled_model_compatibilities.size());
    const int normalized_score = (sum * 1000) / denominator;

    if (!result.matched || normalized_score > result.score) {
      result.matched = true;
      result.score = normalized_score;
      result.selected_ep_compatibility_index = ec_idx;
    }
  }

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
  //
  // To handle the case where there are multiple models for a variant.
  // First, each model inside a variant should have a model role, for example: prefill, decode, etc.
  //
  // Case A: Variants are structurally aligned
  // (For a given EP, all variants have the same number of models, and roles are fully matched across variants.)
  //
  // 1. For each role, ORT asks EP to choose the best model candidate by calling SelectBestCompiledModelCandidate().
  // 2. For each variant, ORT computes role-level scores:
  //    - 100 if that variant's model is selected for the role
  //    - 0 otherwise
  // 3. ORT then computes:
  //    - variant_min = min(role_scores) (bottleneck quality)
  //    - variant_sum = sum(role_scores) (overall quality, tie-breaker)
  // 4. ORT selects the variant with:
  //    - highest variant_min
  //    - if tied, highest variant_sum
  //    - if still tied, earliest appearance in metadata.json (deterministic tie-break)
  //
  // Case B: Variants are not structurally aligned
  // (Variants have different model counts and/or unmatched roles.)
  //
  // 1. ORT falls back to ValidateCompiledModelCompatibilityInfo().
  // 2. ORT computes a score per model from OrtCompiledModelCompatibility.
  // 3. ORT aggregates per-model scores into a variant score (e.g., normalized average).
  // 4. ORT selects the variant with the highest aggregated score.
  //
  // Currently MatchVariantForEp() handles Case B.
  // TODO: Add support for Case A. See https://github.com/microsoft/onnxruntime/pull/28387.
  //
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