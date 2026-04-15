// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/common/common.h"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace onnxruntime {

struct ModelVariantInfo {
  std::string component_model_name{};
  std::string variant_name{};
  std::string package_variant_id{};
  std::string ep{};
  std::string device{};
  std::string architecture{};
  std::string compatibility_info{};
  std::unordered_map<std::string, std::string> metadata{};
  ProviderOptions provider_options{};
  std::unordered_map<std::string, std::string> session_config_entries{};
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

struct ResolvedModelComponentInfo {
  std::string component_model_name{};
  std::string variant_name{};
  std::string package_variant_id{};
  std::string ep_name{};
  std::string device{};
  std::string architecture{};
  std::string compatibility_info{};
  ProviderOptions provider_options{};
  std::unordered_map<std::string, std::string> session_config_entries{};
  OrtCompiledModelCompatibility compiled_model_compatibility{};
  std::filesystem::path model_path{};
  const OrtHardwareDevice* hardware_device{nullptr};
  const OrtEpDevice* ep_device{nullptr};
};

struct ResolvedModelPackageInfo {
  std::string package_variant_id{};
  int score{};
  std::vector<ResolvedModelComponentInfo> resolved_components{};
};

class ModelPackageContext {
 public:
  ModelPackageContext(const std::filesystem::path& package_root);

  const std::vector<ModelVariantInfo>& GetModelVariantInfos() const noexcept {
    return model_variant_infos_;
  }

  const std::vector<std::string>& GetComponentModelNames() const noexcept {
    return component_model_names_;
  }

 private:
  std::vector<ModelVariantInfo> model_variant_infos_;
  std::vector<std::string> component_model_names_;
};

class ModelPackageResolver {
 public:
  ModelPackageResolver() = default;

  // Resolve the best package variant and per-component execution plan for the
  // provided EP/device info. If no package variant matches, the output is reset.
  Status Resolve(const ModelPackageContext& context,
                 gsl::span<VariantSelectionEpInfo> ep_infos,
                 std::optional<ResolvedModelPackageInfo>& resolved_package) const;

 private:
  int CalculateVariantScore(const ModelVariantInfo& variant) const;
};

class ModelVariantSelector {
 public:
  ModelVariantSelector() = default;

  // Convenience wrapper for the legacy direct-session path. This only succeeds
  // if the resolved package contains exactly one component model.
  Status SelectVariant(const ModelPackageContext& context,
                       gsl::span<VariantSelectionEpInfo> ep_infos,
                       std::optional<std::filesystem::path>& selected_variant_path) const;

};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
