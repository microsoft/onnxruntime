// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/common/common.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace onnxruntime {

struct ModelVariantInfo {
  std::string ep{};
  std::string device{};
  std::string architecture{};
  std::string compatibility_info{};
  std::unordered_map<std::string, std::string> metadata{};
  OrtCompiledModelCompatibility compiled_model_compatibility{};
  std::filesystem::path model_path{};
};

struct VariantSelectionEpInfo {
  std::string ep_name{};
  OrtEpFactory* ep_factory{nullptr};
  std::vector<const OrtEpDevice*> ep_devices{};
  std::vector<const OrtHardwareDevice*> hardware_devices{};
  std::vector<const OrtKeyValuePairs*> ep_metadata{};
  ProviderOptions ep_options{};
};

class ModelPackageContext {
 public:
  ModelPackageContext(const std::filesystem::path& package_root);

  const std::vector<ModelVariantInfo>& GetModelVariantInfos() const noexcept {
    return model_variant_infos_;
  }

 private:
  std::vector<ModelVariantInfo> model_variant_infos_;
};

class ModelVariantSelector {
 public:
  ModelVariantSelector() = default;

  // Select model variants that match the provided EP/device info. If multiple
  // variants match, the one with the highest variant score is chosen.
  Status SelectVariant(const ModelPackageContext& context,
                       gsl::span<VariantSelectionEpInfo> ep_infos,
                       std::optional<std::filesystem::path>& selected_variant_path) const;

 private:
  // Compute a score for a variant
  int CalculateVariantScore(const ModelVariantInfo& variant) const;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
