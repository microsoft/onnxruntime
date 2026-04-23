// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/model_package/model_package_options.h"
#include "core/framework/execution_provider.h"
#include "core/common/common.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

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

class ModelPackageOptions;  // forward declaration

class ModelPackageContext {
 public:
  explicit ModelPackageContext(const onnxruntime::Environment& env, const std::filesystem::path& package_root);

  ModelPackageContext(const onnxruntime::Environment& env, const std::filesystem::path& package_root,
                      const ModelPackageOptions& options);

  Status GetEpInfosAndResolveVariant();

  const std::vector<ModelVariantInfo>& GetModelVariantInfos() const noexcept {
    return model_variant_infos_;
  }

  size_t GetComponentModelCount() const noexcept;
  Status GetComponentModelName(size_t component_index, const std::string*& out_name) const;
  Status GetSelectedVariant(const std::string& component_name,
                            const ModelVariantInfo*& out_variant) const;
  Status GetSelectedVariantFiles(const std::string& component_name,
                                 gsl::span<const std::string>& out_file_identifiers) const;
  Status ResolveSelectedVariantFile(const std::string& component_name,
                                    const char* file_identifier /*may be null*/,
                                    std::filesystem::path& out_path) const;

  const ModelPackageOptions* Options() const noexcept { return options_; }

  // Resolved EP state (taken from ModelPackageOptions at ResolveVariants time).
  std::vector<std::unique_ptr<IExecutionProvider>>& MutableProviderList() noexcept { return provider_list_; }
  const std::vector<const OrtEpDevice*>& ExecutionDevices() const noexcept { return execution_devices_; }
  const std::vector<const OrtEpDevice*>& DevicesSelected() const noexcept { return devices_selected_; }
  bool IsFromPolicy() const noexcept { return from_policy_; }
  const std::optional<std::filesystem::path>& GetSelectedVariantPath() const noexcept {
    return selected_variant_path_;
  }

 private:
  const onnxruntime::Environment& env_;
  std::vector<ModelVariantInfo> model_variant_infos_;
  const ModelPackageOptions* options_{};  // non-owning, immutable config source

  // Cached component model info for query APIs.
  std::vector<std::string> component_model_names_{};
  std::unordered_map<std::string, std::vector<size_t>> component_to_variant_indices_{};

  // Cached file identifiers for the currently selected variant (for query APIs).
  mutable std::vector<std::string> selected_variant_file_identifiers_cache_{};

  // Resolved EP state owned by context
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list_{};
  std::vector<VariantSelectionEpInfo> ep_infos_{};
  std::vector<const OrtEpDevice*> execution_devices_{};
  std::vector<const OrtEpDevice*> devices_selected_{};
  bool from_policy_{false};
  std::optional<std::filesystem::path> selected_variant_path_{};

  void BuildComponentModelCache();
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
