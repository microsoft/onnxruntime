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

class ModelPackageOptions;  // forward declaration

class ModelPackageContext {
 public:
  // New primary ctor: package is opened using EP selection captured on `options`.
  ModelPackageContext(const std::filesystem::path& package_root, const ModelPackageOptions& options);

  // Existing ctor kept for tests that don't need EP selection.
  explicit ModelPackageContext(const std::filesystem::path& package_root);

  // Parses variants (if not already) and runs variant selection using options_->EpInfos().
  Status ResolveVariants();

  // Accessors used by the query APIs:
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
  const std::vector<ModelVariantInfo>& GetModelVariantInfos() const noexcept { return model_variant_infos_; }

 private:
  std::vector<ModelVariantInfo> model_variant_infos_;
  const ModelPackageOptions* options_{};  // non-owning

  // TODO: per-component grouping + selected-variant cache for the query APIs.
  // This depends on extending the descriptor parser to surface
  //   - the list of component models, and
  //   - the list of file identifiers (logical names) within each variant.
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
