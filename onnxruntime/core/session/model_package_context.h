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

//
// Keys for parsing the model package manifest json and metadata json.
//
static constexpr const char* kModelPackageManifestFileName = "manifest.json";
static constexpr const char* kModelNameKey = "name";
static constexpr const char* kComponentModelsKey = "component_models";
static constexpr const char* kComponentModelMetadataFileName = "metadata.json";
static constexpr const char* kModelVariantsKey = "model_variants";
static constexpr const char* kVariantNameKey = "variant_name";
static constexpr const char* kFileKey = "file";
static constexpr const char* kConstraintsKey = "constraints";
static constexpr const char* kEpKey = "ep";
static constexpr const char* kDeviceKey = "device";
static constexpr const char* kArchitectureKey = "architecture";
static constexpr const char* kEpCompatibilityInfoKey = "ep_compatibility_info";
static constexpr const char* kSdkVersionKey = "sdk_version";

struct ModelVariantInfo {
  std::string ep;
  std::string device;
  std::string architecture;
  std::string compatibility_info;
  std::unordered_map<std::string, std::string> metadata;
  OrtCompiledModelCompatibility compiled_model_compatibility;
  std::filesystem::path model_path;
};

struct SelectionEpInfo {
  std::string ep_name;
  OrtEpFactory* ep_factory;
  std::vector<const OrtEpDevice*> ep_devices;
  std::vector<const OrtHardwareDevice*> hardware_devices;
  std::vector<const OrtKeyValuePairs*> ep_metadata;
  ProviderOptions ep_options;
};

class ModelPackageDescriptorParser {
 public:
  explicit ModelPackageDescriptorParser(const logging::Logger& logger) : logger_(logger) {}

  Status ParseManifest(const std::filesystem::path& package_root,
                       /*out*/ std::vector<ModelVariantInfo>& components) const;

  Status ParseModelVariantConstraints(const json& constraints, ModelVariantInfo& variant) const;

 private:
  const logging::Logger& logger_;
};

class ModelVariantSelector {
 public:
  ModelVariantSelector() = default;

  // Select model variants that match the provided EP/device info. If multiple
  // variants match, the one with the highest variant score is chosen.
  Status SelectVariant(gsl::span<ModelVariantInfo> variants,
                       gsl::span<SelectionEpInfo> ep_infos,
                       std::optional<std::filesystem::path>& selected_variant_path) const;

 private:
  // Compute a score for a variant
  int CalculateVariantScore(const ModelVariantInfo& variant) const;
};

class ModelPackageContext {
 public:
  ModelPackageContext(const std::filesystem::path& package_root);

  // Select the most suitable model variant
  Status SelectModelVariant(gsl::span<SelectionEpInfo> ep_infos);

  std::optional<std::filesystem::path> GetSelectedModelVariantPath() const {
    return selected_model_variant_path_;
  }

 private:
  std::vector<ModelVariantInfo> model_variant_infos_;
  std::optional<std::filesystem::path> selected_model_variant_path_;
};

}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
