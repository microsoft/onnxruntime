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
  std::optional<std::string> device;
  // Single opaque, EP-owned compatibility token for this (ep, device) entry.
  // If a variant needs to advertise compatibility against multiple sub-targets
  // (e.g. several SoC models or arch variants), the EP encodes that internally
  // in this single string (e.g. comma-joined). ORT does not parse it.
  std::optional<std::string> compatibility_string;
  OrtCompiledModelCompatibility compiled_model_compatibility{OrtCompiledModelCompatibility_EP_NOT_APPLICABLE};
};

struct VariantModelInfo {
  std::string identifier;                 // deterministic id (e.g., filename)
  std::filesystem::path model_file_path;  // resolved path under <component>/<variant>/

  // from variant.json file entry
  std::optional<std::unordered_map<std::string, std::string>> session_options;
  std::optional<std::unordered_map<std::string, std::string>> provider_options;
  std::optional<std::unordered_map<std::string, std::string>> shared_files;  // logical_name -> checksum/path
};

// variant-level info (metadata.json + variant.json)
struct VariantInfo {
  std::string component_name;
  std::string variant_name;

  // from metadata.json: variants.<variant_name>.ep_compatibility
  std::vector<VariantEpCompatibilityInfo> ep_compatibility;

  // selected ep_compatibility entry index after variant matching
  std::optional<size_t> selected_ep_compatibility_index{};

  // from variant.json: files[]
  std::vector<VariantModelInfo> files;

  // from variant.json
  std::optional<json> consumer_metadata;
};

struct ComponentInfo {
  std::string component_name{};
  std::vector<VariantInfo> variants{};
  std::optional<size_t> selected_variant_index{};  // index into variants
};

struct ModelPackageInfo {
  std::vector<ComponentInfo> components{};
};

struct VariantSelectionEpInfo {
  std::string ep_name{};
  OrtEpFactory* ep_factory{nullptr};
  std::vector<const OrtEpDevice*> ep_devices{};
  std::vector<const OrtHardwareDevice*> hardware_devices{};
  std::vector<const OrtKeyValuePairs*> ep_metadata{};
};

class ModelPackageOptions;  // forward declaration

class ModelPackageComponentContext {
 public:
  explicit ModelPackageComponentContext(const std::string& component_name,
                                        const ComponentInfo& component_model_info,
                                        const ModelPackageOptions* options);

  explicit ModelPackageComponentContext(const std::string& component_name,
                                        const ComponentInfo& component_model_info,
                                        gsl::span<const VariantSelectionEpInfo> ep_infos);

  Status ResolveVariant();

  const std::vector<VariantInfo>& GetVariantInfos() const noexcept {
    return component_model_info_.variants;
  }

  const ModelPackageOptions* Options() const noexcept {
    return options_;
  }

  Status GetSelectedVariantFilePaths(gsl::span<const std::filesystem::path>& out_file_paths) const;

  Status GetSelectedVariantFolderPath(const std::filesystem::path*& out_folder_path) const;

  // Convenience API for single-file selected variants. Will return an error if there are 0 or >1 components.
  Status GetSelectedVariantFilePath(std::filesystem::path& out_path) const;

  Status GetSelectedVariantFileSessionOptions(size_t file_idx,
                                              gsl::span<const std::string>& out_keys,
                                              gsl::span<const std::string>& out_values) const;

  Status GetSelectedVariantFileProviderOptions(size_t file_idx,
                                               gsl::span<const std::string>& out_keys,
                                               gsl::span<const std::string>& out_values) const;

  // Returns the consumer_metadata JSON object (from variant.json) serialized to a string for the
  // selected variant. Returns an empty string if the variant did not declare consumer_metadata.
  // Pointer lifetime is owned by this context.
  Status GetSelectedVariantConsumerMetadata(const std::string*& out_json_str) const;

  std::vector<std::unique_ptr<IExecutionProvider>>& MutableProviderList();
  const std::vector<const OrtEpDevice*>& ExecutionDevices() const;
  const std::vector<const OrtEpDevice*>& DevicesSelected() const;
  bool IsFromPolicy() const;

 private:
  std::string component_model_name_;
  ComponentInfo component_model_info_{};

  const ModelPackageOptions* options_{};                // non-owning, immutable config source for EP intent
  gsl::span<const VariantSelectionEpInfo> ep_infos_{};  // non-owning EP intent when options_ is not used
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list_{};

  // optional runtime state mirrors (if needed by callers)
  std::vector<const OrtEpDevice*> execution_devices_{};
  std::vector<const OrtEpDevice*> devices_selected_{};
  bool from_policy_{false};

  // Caches for selected variant info.
  mutable std::string consumer_metadata_cache_{};
  mutable bool consumer_metadata_cache_valid_{false};
  mutable std::filesystem::path folder_path_cache_{};
  mutable std::vector<std::filesystem::path> file_paths_cache_{};
  mutable std::unordered_map<size_t, std::vector<std::string>> file_id_to_session_option_keys_cache_{};
  mutable std::unordered_map<size_t, std::vector<std::string>> file_id_to_session_option_values_cache_{};
  mutable std::unordered_map<size_t, std::vector<std::string>> file_id_to_provider_option_keys_cache_{};
  mutable std::unordered_map<size_t, std::vector<std::string>> file_id_to_provider_option_values_cache_{};

  Status ResolveVariantImpl(gsl::span<const VariantSelectionEpInfo> ep_infos);
  Status GetSelectedVariantInfo(const VariantInfo*& out_variant) const;
};

class ModelPackageContext {
 public:
  explicit ModelPackageContext(const std::filesystem::path& package_root);

  size_t GetComponentCount() const noexcept;
  Status GetComponentNames(gsl::span<const std::string>& out_names) const;

  Status GetVariantCount(const std::string& component_name, size_t& out_count) const;
  Status GetVariantNames(const std::string& component_name,
                              gsl::span<const std::string>& out_variant_names) const;

  // Pre-selection traversal of the (ep, device, compatibility_string) tuples declared on a variant.
  // Lets callers (e.g. GenAI defaulting logic) inspect what EPs a package advertises support for
  // before any EP has been resolved / before SelectComponent has been called.
  Status GetVariantEpCompatibilityCount(const std::string& component_name,
                                        const std::string& variant_name,
                                        size_t& out_count) const;
  Status GetVariantEpCompatibilityInfo(const std::string& component_name,
                                       const std::string& variant_name,
                                       size_t ep_idx,
                                       const VariantEpCompatibilityInfo*& out_info) const;

  const ModelPackageInfo& GetModelPackageInfo() const noexcept {
    return model_package_info_;
  }

  const std::vector<VariantInfo>& GetVariantInfos() const noexcept {
    return model_variant_infos_;
  }

 private:
  ModelPackageInfo model_package_info_{};
  std::vector<VariantInfo> model_variant_infos_;

  std::unordered_map<std::string, size_t> component_name_to_index_{};
  std::vector<std::string> component_names_cache_{};
  mutable std::unordered_map<std::string, std::vector<std::string>> component_to_variant_names_cache_{};
  mutable std::unordered_map<std::string, std::vector<std::string>> variant_to_file_identifiers_cache_{};
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
