// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_variant_selector.h"
#include "core/session/model_package/model_package_context.h"

#include <algorithm>
#include <cctype>
#include <limits>

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

VariantMatchResult MatchVariantForEp(VariantInfo& variant, const VariantSelectionEpInfo& ep_info) {
  VariantMatchResult result{};
  if (variant.ep_compatibility.empty()) {
    return result;
  }

  for (size_t ec_idx = 0; ec_idx < variant.ep_compatibility.size(); ++ec_idx) {
    auto& ec = variant.ep_compatibility[ec_idx];

    // 1. Match EP. Parser enforces that "ep" is required and non-empty in every entry, so this is
    //    a straightforward string compare.
    if (!ec.ep.has_value() || *ec.ep != ep_info.ep_name) {
      continue;
    }

    // 2. Match device
    bool device_ok = !ec.device.has_value() || ec.device->empty();
    std::vector<const OrtHardwareDevice*> constraint_devices = ep_info.hardware_devices;

    if (!device_ok) {
      // If the EP exposes no hardware devices but the variant declares a device constraint,
      // we cannot verify the constraint, so treat it as non-matching.
      if (ep_info.hardware_devices.empty()) {
        continue;
      }
      if (const auto* matched_hd = FindMatchingHardwareDevice(*ec.device, ep_info.hardware_devices)) {
        device_ok = true;
        constraint_devices = {matched_hd};
      }
    }

    if (!device_ok) {
      continue;
    }

    // 3. Check compatibility string. Each variant carries one opaque compatibility string
    //    owned by the EP; if the EP needs to encode multiple sub-targets it does so internally
    //    (e.g. comma-joined) and parses the value itself.
    const std::string& compat_str =
        ec.compatibility_string.has_value() ? *ec.compatibility_string : std::string{};

    OrtCompiledModelCompatibility compat = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    auto st = ValidateCompatibilityString(ep_info, compat_str, constraint_devices, compat);
    if (!st.IsOK()) {
      LOGS_DEFAULT(WARNING) << "Failed to validate compiled model compatibility for variant '"
                            << variant.variant_name << "', ep compatibility index " << ec_idx
                            << ". Error: " << st.ErrorMessage()
                            << " Skip this ep compatibility.";
      continue;
    }

    ec.compiled_model_compatibility = compat;

    if (compat == OrtCompiledModelCompatibility_EP_UNSUPPORTED) {
      continue;
    }

    const int score = CompatibilityToScore(compat);

    if (!result.matched || score > result.score) {
      result.matched = true;
      result.score = score;
      result.selected_ep_compatibility_index = ec_idx;
    }
  }

  return result;
}

}  // namespace

Status VariantSelector::SelectVariant(const ModelPackageComponentContext& context,
                                      gsl::span<const VariantSelectionEpInfo> ep_infos,
                                      std::optional<VariantInfo>& selected_variant) const {
  selected_variant.reset();

  std::vector<VariantInfo> variants = context.GetVariantInfos();
  if (variants.empty()) {
    return Status::OK();
  }

  const VariantSelectionEpInfo* selected_ep_info = nullptr;
  if (!ep_infos.empty()) {
    // For now we only consider the first captured EP for variant matching.
    //
    // ModelPackageOptions may capture multiple EPs (e.g. when EP selection comes from a policy that
    // returns a ranked list such as [CUDA, WebGPU, CPU]). The full v4 design ranks variants across
    // that list in priority order; until that lands here, we use only the first EP. Callers that
    // need a specific EP should put it first in the captured order.
    // TODO: extend to rank variants across the full EP list, honoring captured priority order.
    selected_ep_info = &ep_infos[0];
    if (ep_infos.size() > 1) {
      LOGS_DEFAULT(INFO) << "Multiple EPs captured for model package; using only the first ('"
                         << ep_infos[0].ep_name << "') for variant matching.";
    }
  }

  // EP/device compatibility pass.
  //
  // Each variant declares a single target EP and carries one opaque compatibility string owned
  // by that EP. ORT does not parse it, decompose it, or aggregate scores across multiple strings
  // -- if the EP needs to encode multiple sub-targets (e.g. several QNN SoC models), it does so
  // internally in this single string and validates the full value itself.
  //
  // If multiple execution providers are needed, separate variants should be created (preferably
  // with shared weights).
  //
  // We do not currently support a "portable" / wildcard variant. If that need arises later, the
  // planned shape is: a variant with EP info omitted is treated as the portable fallback (and
  // likely pinned to CPU at load time).
  //
  // Current implementation:
  //   For each variant whose ep/device constraints match the selected EP, call
  //   OrtEpFactory::ValidateCompiledModelCompatibilityInfo() and map the returned
  //   OrtCompiledModelCompatibility enum to a numeric score. Pick the highest-scoring variant.
  //
  // Planned fallback ladder (not yet wired -- TODO):
  //   a) If the EP implements OrtEpFactory::SelectBestCompiledModelCandidate() (PR #28387), gather
  //      every variant into a candidate list and let the EP pick the best index. SIZE_MAX means
  //      "none acceptable", fall through to (b).
  //   b) Otherwise, use ValidateCompiledModelCompatibilityInfo() per variant as we do today and pick
  //      the highest-ranked one.
  //   c) If neither ABI is implemented, return the first variant whose ep/device constraints match.
  if (selected_ep_info != nullptr) {
    int best_score = std::numeric_limits<int>::min();
    std::optional<size_t> best_index;
    std::optional<size_t> best_ec_index;

    for (size_t i = 0, end = variants.size(); i < end; ++i) {
      VariantMatchResult m = MatchVariantForEp(variants[i], *selected_ep_info);
      if (!m.matched) {
        continue;
      }
      // Strict '>' so on ties we keep the first variant seen, giving deterministic
      // tie-break by manifest declaration order.
      if (!best_index.has_value() || m.score > best_score) {
        best_score = m.score;
        best_index = i;
        best_ec_index = m.selected_ep_compatibility_index;
      }
    }

    if (best_index.has_value()) {
      selected_variant = std::move(variants[*best_index]);
      selected_variant->selected_ep_compatibility_index = best_ec_index;
      return Status::OK();
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
