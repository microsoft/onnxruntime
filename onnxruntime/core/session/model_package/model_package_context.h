// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/execution_provider.h"
#include "core/common/common.h"
#include "core/session/model_package/model_package_variant_selector.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace onnxruntime {

struct VariantEpCompatibilityInfo {
  std::optional<std::string> ep;
  std::optional<std::string> device_type;
  std::optional<std::string> compatibility_info;
  std::optional<std::unordered_map<std::string, std::string>> session_options;
  std::optional<std::unordered_map<std::string, std::string>> provider_options;
  OrtCompiledModelCompatibility compiled_model_compatibility{};
};

struct VariantModelInfo {
  std::string identifier;
  std::filesystem::path model_file_path;
  std::vector<VariantEpCompatibilityInfo> ep_compatibility;

  // Selected ep_compatibility entry index after variant matching.
  std::optional<size_t> selected_ep_compatibility_index{};
};

// The finest granularity for variant selection.
struct ModelVariantInfo {
  std::string component_model_name;
  std::string variant_name;
  std::vector<VariantModelInfo> model_info;
  std::optional<json> consumer_metadata;
};

struct ComponentModelInfo {
  std::string component_model_name{};
  std::vector<ModelVariantInfo> model_variants{};
  std::optional<size_t> selected_variant_index{};  // index into model_variants
};

struct ModelPackageInfo {
  std::vector<ComponentModelInfo> component_models{};
};

struct VariantSelectionEpInfo {
  std::string ep_name{};
  OrtEpFactory* ep_factory{nullptr};
  std::vector<const OrtEpDevice*> ep_devices{};
  std::vector<const OrtHardwareDevice*> hardware_devices{};
  std::vector<const OrtKeyValuePairs*> ep_metadata{};
};

class ModelPackageOptions;  // forward declaration

class ModelPackageContext {
 public:
  explicit ModelPackageContext(const std::filesystem::path& package_root);

  size_t GetComponentModelCount() const noexcept;
  Status GetComponentModelNames(gsl::span<const std::string>& out_names) const;

  Status GetModelVariantCount(const std::string& component_name, size_t& out_count) const;
  Status GetModelVariantNames(const std::string& component_name,
                              gsl::span<const std::string>& out_variant_names) const;

  Status GetFileCount(const std::string& component_name,
                      const std::string& variant_name,
                      size_t& out_count) const;
  Status GetFileIdentifiers(const std::string& component_name,
                            const std::string& variant_name,
                            gsl::span<const std::string>& out_file_identifiers) const;
  Status GetFilePath(const std::string& component_name,
                     const std::string& variant_name,
                     const char* file_identifier /*may be null*/,
                     std::filesystem::path& out_path) const;

  Status ResolveVariant(const ModelPackageOptions* options);

  Status ResolveVariant(gsl::span<const VariantSelectionEpInfo> ep_infos);

  const ModelPackageOptions* Options() const noexcept;

  const ModelPackageInfo& GetModelPackageInfo() const noexcept {
    return model_package_info_;
  }

  const std::vector<ModelVariantInfo>& GetModelVariantInfos() const noexcept {
    return model_variant_infos_;
  }

  Status GetSelectedVariantModelInfo(const std::string& component_name,
                                     const char* file_identifier /*may be null*/,
                                     const VariantModelInfo*& out_model_info) const;
  Status GetSelectedVariantFileIdentifiers(const std::string& component_name,
                                           gsl::span<const std::string>& out_file_identifiers) const;
  Status ResolveSelectedVariantFilePath(const std::string& component_name,
                                        const char* file_identifier /*may be null*/,
                                        std::filesystem::path& out_path) const;

  // Convenience API for single-component packages. Will return an error if there are 0 or >1 components.
  Status GetSelectedVariantFilePath(std::filesystem::path& out_path) const;

  // Resolved EP state (taken from ModelPackageOptions).
  std::vector<std::unique_ptr<IExecutionProvider>>& MutableProviderList();
  const std::vector<const OrtEpDevice*>& ExecutionDevices() const;
  const std::vector<const OrtEpDevice*>& DevicesSelected() const;
  bool IsFromPolicy() const;

  Status GetSelectedVariantFileSessionOptions(const std::string& component_name,
                                              const char* file_identifier /*may be null*/,
                                              gsl::span<const std::string>& out_keys,
                                              gsl::span<const std::string>& out_values) const;

  Status GetSelectedVariantFileProviderOptions(const std::string& component_name,
                                               const char* file_identifier /*may be null*/,
                                               gsl::span<const std::string>& out_keys,
                                               gsl::span<const std::string>& out_values) const;

 private:
  const ModelPackageOptions* options_{};  // non-owning, immutable config source
  std::vector<ModelVariantInfo> model_variant_infos_;

  ModelPackageInfo model_package_info_{};
  std::unordered_map<std::string, size_t> component_name_to_index_{};
  std::vector<std::string> component_names_cache_{};
  mutable std::unordered_map<std::string, std::vector<std::string>> component_to_variant_names_cache_{};
  mutable std::unordered_map<std::string, std::vector<std::string>> variant_to_file_identifiers_cache_{};

  mutable std::vector<std::string> selected_variant_file_identifiers_cache_{};
  mutable std::vector<std::string> selected_variant_session_option_keys_cache_{};
  mutable std::vector<std::string> selected_variant_session_option_values_cache_{};
  mutable std::vector<std::string> selected_variant_provider_option_keys_cache_{};
  mutable std::vector<std::string> selected_variant_provider_option_values_cache_{};

  // Resolved EP state owned by context
  std::vector<VariantSelectionEpInfo> ep_infos_{};
  std::vector<const OrtEpDevice*> execution_devices_{};
  std::vector<const OrtEpDevice*> devices_selected_{};
  bool from_policy_{false};
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list_{};

  Status ResolveVariantImpl(gsl::span<const VariantSelectionEpInfo> ep_infos);

  Status GetSelectedVariantForComponent(const std::string& component_name,
                                        const ModelVariantInfo*& out_variant) const;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
