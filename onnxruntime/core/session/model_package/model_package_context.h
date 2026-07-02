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

  // from variant.json file entry. Values of path-valued session option keys (see
  // IsModelPackagePathSessionOption) are resolved to absolute paths at parse time.
  std::optional<std::unordered_map<std::string, std::string>> session_options;
  std::optional<std::unordered_map<std::string, std::string>> provider_options;
};

// True if the given ORT session-option config key holds a file/folder path reference that must be
// resolved against the model package (sha256:<hex>, relative, or absolute) before use.
bool IsModelPackagePathSessionOption(std::string_view key);

// variant-level info (metadata.json + variant.json)
struct VariantInfo {
  std::string component_name;
  std::string variant_name;

  // The variant's root folder (always available regardless of variant.json presence).
  std::filesystem::path folder_path;

  // from metadata.json: single EP target per variant (ep, device, compatibility_string)
  VariantEpCompatibilityInfo ep_compatibility;

  // from variant.json: single model file entry. Empty when variant.json is absent.
  std::optional<VariantModelInfo> file;

  // from variant.json
  std::optional<json> consumer_metadata;
};

struct ComponentInfo {
  std::string component_name{};
  std::vector<VariantInfo> variants{};
  std::optional<size_t> selected_variant_index{};  // index into variants
};

struct ModelPackageInfo {
  int64_t schema_version{0};
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
                                        const ModelPackageOptions& options);

  explicit ModelPackageComponentContext(const std::string& component_name,
                                        const ComponentInfo& component_model_info,
                                        gsl::span<const VariantSelectionEpInfo> ep_infos);

  Status ResolveVariant();

  const std::vector<VariantInfo>& GetVariantInfos() const noexcept {
    return component_model_info_.variants;
  }

  Status GetSelectedVariantFolderPath(const std::filesystem::path*& out_folder_path) const;

  // Get the single model file path for the selected variant.
  Status GetSelectedVariantFilePath(std::filesystem::path& out_path) const;

  Status GetSelectedVariantFileSessionOptions(gsl::span<const std::string>& out_keys,
                                              gsl::span<const std::string>& out_values) const;

  Status GetSelectedVariantFileProviderOptions(gsl::span<const std::string>& out_keys,
                                               gsl::span<const std::string>& out_values) const;

  // Returns the consumer_metadata JSON object (from variant.json) serialized to a string for the
  // selected variant. Returns an empty string if the variant did not declare consumer_metadata.
  // Pointer lifetime is owned by this context.

  // C API helpers: return const char* pointer arrays with context-owned lifetime.
  Status GetSelectedVariantFileSessionOptionPtrs(const char* const*& out_keys,
                                                 const char* const*& out_values,
                                                 size_t& out_count) const;
  Status GetSelectedVariantFileProviderOptionPtrs(const char* const*& out_keys,
                                                  const char* const*& out_values,
                                                  size_t& out_count) const;

  Status GetSelectedVariantConsumerMetadata(const std::string*& out_json_str) const;

  Status GetSelectedVariantName(const std::string*& out_name) const;

  std::vector<std::unique_ptr<IExecutionProvider>>& MutableProviderList() { return provider_list_; }
  const std::vector<const OrtEpDevice*>& ExecutionDevices() const { return execution_devices_; }
  const std::vector<const OrtEpDevice*>& DevicesSelected() const { return devices_selected_; }
  gsl::span<const VariantSelectionEpInfo> EpInfos() const { return ep_infos_; }
  bool IsFromPolicy() const { return from_policy_; }

  // Rebuild the provider list for a new session creation call (providers are consumed/moved
  // when registered, so they must be rebuilt for each session).
  // Uses the provided session options for provider creation (should include merged provider options).
  Status RebuildProviderListForSession(const Environment& env, const OrtSessionOptions& effective_options);

 private:
  std::string component_model_name_;
  ComponentInfo component_model_info_{};

  gsl::span<const VariantSelectionEpInfo> ep_infos_{};    // non-owning EP intent when options are not used
  std::vector<VariantSelectionEpInfo> owned_ep_infos_{};  // owned copy when constructed from ModelPackageOptions
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list_{};

  // optional runtime state mirrors (if needed by callers)
  std::vector<const OrtEpDevice*> execution_devices_{};
  std::vector<const OrtEpDevice*> devices_selected_{};
  bool from_policy_{false};

  // Caches for selected variant info.
  mutable std::string consumer_metadata_cache_{};
  mutable bool consumer_metadata_cache_valid_{false};
  mutable std::filesystem::path folder_path_cache_{};
  mutable std::vector<std::string> session_option_keys_cache_{};
  mutable std::vector<std::string> session_option_values_cache_{};
  mutable std::vector<std::string> provider_option_keys_cache_{};
  mutable std::vector<std::string> provider_option_values_cache_{};

  // C API pointer caches for session/provider options, owned by context for stable lifetime.
  mutable std::vector<const char*> session_option_key_ptrs_cache_{};
  mutable std::vector<const char*> session_option_value_ptrs_cache_{};
  mutable std::vector<const char*> provider_option_key_ptrs_cache_{};
  mutable std::vector<const char*> provider_option_value_ptrs_cache_{};

  Status ResolveVariantImpl(gsl::span<const VariantSelectionEpInfo> ep_infos);
  Status GetSelectedVariantInfo(const VariantInfo*& out_variant) const;
};

class ModelPackageContext {
 public:
  explicit ModelPackageContext(const std::filesystem::path& package_root);

  size_t GetComponentCount() const noexcept;
  Status GetComponentNames(gsl::span<const std::string>& out_names) const;

  // C API helpers: return const char* pointer arrays with context-owned lifetime.
  void GetComponentNamePtrs(const char* const*& out_ptrs, size_t& out_count) const;
  void GetVariantNamePtrs(const std::string& component_name,
                          const char* const*& out_ptrs, size_t& out_count) const;

  Status GetVariantCount(const std::string& component_name, size_t& out_count) const;
  Status GetVariantNames(const std::string& component_name,
                         gsl::span<const std::string>& out_variant_names) const;

  // Get the EP compatibility info declared on a variant.
  // Lets callers inspect what EP a variant targets
  // before any EP has been resolved / before SelectComponent has been called.
  Status GetVariantEpCompatibility(const std::string& component_name,
                                   const std::string& variant_name,
                                   const VariantEpCompatibilityInfo*& out_info) const;

  const ModelPackageInfo& GetModelPackageInfo() const noexcept {
    return model_package_info_;
  }

  const std::vector<VariantInfo>& GetVariantInfos() const noexcept {
    return model_variant_infos_;
  }

  // Resolves a path reference from the package against the model_package library's rules:
  // a "sha256:<hex>[/tail]" content-addressed shared-asset reference (honoring manifest
  // overrides), or a plain relative path resolved against `base_dir` (empty base_dir falls
  // back to the package root). When `must_exist` is true the resolved path must exist on
  // disk. The returned pointer is owned by this context and stays valid until the next
  // ResolveStringRef call. The underlying package handle is kept open for the context's
  // lifetime so no reopen/reparse happens per call.
  Status ResolveStringRef(const std::string& base_dir, const std::string& input,
                          bool must_exist, const char*& out_path) const;

 private:
  // The open model_package library handle, kept alive for this context's lifetime so path
  // references can be resolved on demand. Stored type-erased (void*) to keep the
  // model_package C header out of this ORT header; the deleter defined in the .cc closes it
  // via ModelPackage_Close.
  std::unique_ptr<void, void (*)(void*)> package_handle_;
  std::filesystem::path package_root_{};
  mutable std::string resolve_string_ref_cache_{};

  ModelPackageInfo model_package_info_{};
  std::vector<VariantInfo> model_variant_infos_;

  std::unordered_map<std::string, size_t> component_name_to_index_{};
  std::vector<std::string> component_names_cache_{};
  mutable std::unordered_map<std::string, std::vector<std::string>> component_to_variant_names_cache_{};
  mutable std::unordered_map<std::string, std::vector<std::string>> variant_to_file_identifiers_cache_{};

  // C API pointer caches: owned by the context so their lifetime matches the documented contract.
  mutable std::vector<const char*> component_name_ptrs_cache_{};
  mutable std::unordered_map<std::string, std::vector<const char*>> variant_name_ptrs_cache_{};
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
