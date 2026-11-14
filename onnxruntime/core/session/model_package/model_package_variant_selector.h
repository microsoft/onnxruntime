// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <optional>
#include <string_view>
#include <gsl/gsl>
#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// forward declaration
class ModelPackageComponentContext;
struct VariantInfo;
struct VariantModelInfo;
struct VariantEpCompatibilityInfo;
struct VariantSelectionEpInfo;
struct VariantSelection;

// Returns true if hardware_device matches the device type declared by a variant ("cpu", "gpu" or "npu").
// The comparison ignores letter case. An empty device type matches any device.
bool MatchesVariantDevice(const OrtHardwareDevice* hardware_device, std::string_view device);

class VariantSelector {
 public:
  VariantSelector() = default;

  // Selects a variant for the component. ep_infos lists EPs in priority order.
  // Skips EPs with no variants or only incompatible variants. Leaves selection empty if no variant matches.
  // If ORT cannot check a variant's compatibility and no other variant matches that EP, returns an error instead
  // of considering EPs with lower priority.
  Status SelectVariant(const ModelPackageComponentContext& context,
                       gsl::span<const VariantSelectionEpInfo> ep_infos,
                       std::optional<VariantSelection>& selection) const;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
