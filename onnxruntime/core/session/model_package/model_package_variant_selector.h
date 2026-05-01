// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <optional>
#include <gsl/gsl>
#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {

// forward declaration
class ModelPackageContext;
struct ModelVariantInfo;
struct VariantModelInfo;
struct VariantEpCompatibilityInfo;
struct VariantSelectionEpInfo;

class ModelVariantSelector {
 public:
  ModelVariantSelector() = default;

  // Select model variant (finest granularity).
  Status SelectVariant(const ModelPackageContext& context,
                       gsl::span<const VariantSelectionEpInfo> ep_infos,
                       std::optional<ModelVariantInfo>& selected_variant) const;

 private:
  int CalculateVariantScore(const ModelVariantInfo& variant) const;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)