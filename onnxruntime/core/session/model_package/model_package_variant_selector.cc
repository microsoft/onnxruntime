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

  // 2. Match the device constraint. For the compatibility callback, use the EP's captured hardware devices matching
  // the variant's declared device type, or all captured hardware devices for that EP if no type is declared
  // (see ModelPackageComponentContext::SelectedEpDevices).
  bool device_ok = !ec.device.has_value() || ec.device->empty();
  std::vector<const OrtHardwareDevice*> constraint_devices = ep_info.hardware_devices;

  if (!device_ok) {
    if (ep_info.hardware_devices.empty()) {
      if (ep_info.ep_name != kCpuExecutionProvider) {
        result.unchecked = true;
        return result;
      }
      device_ok = MatchesDevice(OrtHardwareDeviceType_CPU, *ec.device);
    } else {
      constraint_devices.clear();
      for (const OrtHardwareDevice* hardware_device : ep_info.hardware_devices) {
        if (MatchesVariantDevice(hardware_device, *ec.device)) {
          constraint_devices.push_back(hardware_device);
        }
      }
      device_ok = !constraint_devices.empty();
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
  // If ORT cannot check a variant for an EP and no other variant matches that EP, selection returns an error instead
  // of considering EPs with lower priority.
  //
  // Planned fallback ladder (not yet wired -- TODO):
  //   a) If the EP implements OrtEpFactory::SelectBestModelCandidate() (PR #28387), gather
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
