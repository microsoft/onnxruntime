// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_variant_selector.h"
#include "core/session/model_package/model_package_context.h"

#include <algorithm>
#include <cctype>
#include <limits>

#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
#include "core/session/abi_devices.h"
#include "core/session/utils.h"

namespace onnxruntime {
namespace {

struct VariantMatchResult {
  bool matched{false};
  int score{std::numeric_limits<int>::min()};
  // ORT could not determine whether the variant matches the EP: a required device could not be checked because no
  // devices were captured, or the compatibility callback returned an error.
  // The built-in CPU EP can match a CPU requirement without captured devices.
  bool unchecked{false};
};

std::string ToLower(std::string_view s) {
  std::string result(s);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return result;
}

bool MatchesDevice(OrtHardwareDeviceType type, std::string_view value) {
  const std::string device = ToLower(value);
  switch (type) {
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
    if (hd != nullptr && MatchesDevice(hd->type, device_constraint)) {
      return hd;
    }
  }

  return nullptr;
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

// Calls the EP's ValidateCompiledModelCompatibilityInfo for a single compatibility string.
// Writes OrtCompiledModelCompatibility_EP_NOT_APPLICABLE when the EP does not implement the ABI,
// the EP is too old, or the compatibility string is empty.
Status ValidateCompatibilityString(const VariantSelectionEpInfo& ep_info,
                                   const std::string& compatibility_string,
                                   std::vector<const OrtHardwareDevice*>& constraint_devices,
                                   OrtCompiledModelCompatibility& out_compat) {
  out_compat = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

  if (compatibility_string.empty()) {
    return Status::OK();
  }

  auto* ep_factory = ep_info.ep_factory;
  if (ep_factory == nullptr ||
      ep_factory->ort_version_supported < 23 ||
      ep_factory->ValidateCompiledModelCompatibilityInfo == nullptr) {
    return Status::OK();
  }

  auto status = ep_factory->ValidateCompiledModelCompatibilityInfo(ep_factory,
                                                                   constraint_devices.data(),
                                                                   constraint_devices.size(),
                                                                   compatibility_string.c_str(),
                                                                   &out_compat);
  ORT_RETURN_IF_ERROR(ToStatusAndRelease(status));
  return Status::OK();
}

// Checks whether a variant's single EP compatibility entry matches the given EP.
// Returns a match result with a score derived from ValidateCompiledModelCompatibilityInfo().
VariantMatchResult MatchVariantForEp(VariantInfo& variant, const VariantSelectionEpInfo& ep_info) {
  VariantMatchResult result{};
  const auto& ec = variant.ep_compatibility;

  // 1. Match EP name. Required and non-empty per schema.
  if (!ec.ep.has_value() || *ec.ep != ep_info.ep_name) {
    return result;
  }

  // 2. Match device constraint.
  bool device_ok = !ec.device.has_value() || ec.device->empty();
  std::vector<const OrtHardwareDevice*> constraint_devices = ep_info.hardware_devices;

  if (!device_ok) {
    if (ep_info.hardware_devices.empty()) {
      if (ep_info.ep_name != kCpuExecutionProvider) {
        result.unchecked = true;
        return result;
      }
      device_ok = MatchesDevice(OrtHardwareDeviceType_CPU, *ec.device);
    } else if (const auto* matched_hd = FindMatchingHardwareDevice(*ec.device, ep_info.hardware_devices)) {
      device_ok = true;
      constraint_devices = {matched_hd};
    }
  }

  if (!device_ok) {
    return result;
  }

  // 3. Validate compatibility string via EP callback.
  const std::string& compat_str =
      ec.compatibility_string.has_value() ? *ec.compatibility_string : std::string{};

  OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  auto st = ValidateCompatibilityString(ep_info, compat_str, constraint_devices, compat);
  if (!st.IsOK()) {
    LOGS_DEFAULT(WARNING) << "Failed to validate compiled model compatibility for variant '"
                          << variant.variant_name << "'. Error: " << st.ErrorMessage();
    result.unchecked = true;
    return result;
  }

  variant.ep_compatibility.compiled_model_compatibility = compat;

  if (compat == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
    return result;
  }

  result.matched = true;
  result.score = CompatibilityToScore(compat);
  return result;
}

}  // namespace

bool MatchesVariantDevice(const OrtHardwareDevice* hardware_device, std::string_view device) {
  return device.empty() || (hardware_device != nullptr && MatchesDevice(hardware_device->type, device));
}

Status VariantSelector::SelectVariant(const ModelPackageComponentContext& context,
                                      gsl::span<const VariantSelectionEpInfo> ep_infos,
                                      std::optional<VariantSelection>& selection) const {
  selection.reset();

  std::vector<VariantInfo> variants = context.GetVariantInfos();
  if (variants.empty()) {
    return Status::OK();
  }

  // EP/device compatibility pass.
  //
  // Each variant declares a single target EP (ep, device, compatibility_string). ORT does not
  // parse or decompose the compatibility string -- EP validates the compatibility string and
  // returns an OrtCompiledModelCompatibility enum indicating the level of compatibility.
  //
  // Selection algorithm:
  //   The captured EPs are considered in priority order: the order determined by the selection policy
  //   (for example [CoreML, CPU]) or the order in which EPs were added to the session options.
  //   The first EP with a matching variant is selected.
  //   For each variant whose EP name and device requirements match the current EP, call
  //   OrtEpFactory::ValidateCompiledModelCompatibilityInfo() and map the returned OrtCompiledModelCompatibility value
  //   to a numeric score. Select the matching variant with the highest compatibility score for that EP.
  //   If several variants have the same score, select the first one listed in the package manifest.
  //   If an EP has no variants or all its variants are incompatible, consider the next EP.
  //   Compatibility is unknown if a variant requires a device but no devices were captured for the EP, or if
  //   the EP's compatibility callback returns an error.
  //   The built-in CPU EP can match a CPU requirement without captured devices.
  //   If compatibility is unknown and no other variant matches that EP, return an error instead of considering
  //   EPs with lower priority.
  //   This prevents an unchecked variant from being silently replaced by a variant for another EP.
  //
  // Planned fallback ladder (not yet wired -- TODO):
  //   a) If the EP implements OrtEpFactory::SelectBestCompiledModelCandidate() (PR #28387), gather
  //      every matching variant into a candidate list and let the EP pick the best index. SIZE_MAX
  //      means "none acceptable", fall through to (b).
  //   b) Otherwise, use ValidateCompiledModelCompatibilityInfo() per variant as we do today.
  //   c) If neither ABI is implemented, return the first variant whose ep/device matches.
  for (size_t ep_index = 0; ep_index < ep_infos.size(); ++ep_index) {
    int best_score = std::numeric_limits<int>::min();
    std::optional<size_t> best_index;
    bool has_unchecked_variant = false;

    for (size_t i = 0, end = variants.size(); i < end; ++i) {
      VariantMatchResult m = MatchVariantForEp(variants[i], ep_infos[ep_index]);
      has_unchecked_variant = has_unchecked_variant || m.unchecked;
      if (!m.matched) {
        continue;
      }
      // Strict '>' so on ties we keep the first variant seen, giving deterministic
      // tie-break by manifest declaration order.
      if (!best_index.has_value() || m.score > best_score) {
        best_score = m.score;
        best_index = i;
      }
    }

    if (best_index.has_value()) {
      selection = VariantSelection{std::move(variants[*best_index]), ep_index};
      return Status::OK();
    }

    if (has_unchecked_variant) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "No suitable model variant found for the configured execution providers: "
                             "selection stopped at execution provider '",
                             ep_infos[ep_index].ep_name,
                             "' because some variants could not be checked. They require a device but no devices "
                             "were captured for this EP, or its compatibility callback returned an error.");
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
