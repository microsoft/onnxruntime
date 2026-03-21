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
// Keys for parsing the model package manifest json
//
static constexpr const char* kModelPackageManifestFileName = "manifest.json";
static constexpr const char* kModelNameKey = "name";
static constexpr const char* kComponentsKey = "components";
static constexpr const char* kVariantNameKey = "variant_name";
static constexpr const char* kFileKey = "file";
static constexpr const char* kConstraintsKey = "constraints";
static constexpr const char* kEpKey = "ep";
static constexpr const char* kDeviceKey = "device";
static constexpr const char* kArchitectureKey = "architecture";

struct EpContextVariantInfo {
  std::string ep;
  std::string device;
  std::string architecture;
  OrtKeyValuePairs metadata;
  std::filesystem::path model_path;
};

// Same as the `SelecionInfo` in provider_policy_context.h but with
// additional fields for ep name and provider options.
struct SelectionEpInfo {
  std::string ep_name;
  OrtEpFactory* ep_factory;
  std::vector<const OrtEpDevice*> ep_devices;
  std::vector<const OrtHardwareDevice*> hardware_devices;
  std::vector<const OrtKeyValuePairs*> ep_metadata;
  ProviderOptions ep_options;
};

class ModelPackageManifestParser {
 public:
  explicit ModelPackageManifestParser(const logging::Logger& logger) : logger_(logger) {}

  Status ParseManifest(const std::filesystem::path& package_root,
                       /*out*/ std::vector<EpContextVariantInfo>& components);

 private:
  const logging::Logger& logger_;
};

class ModelPackageContext {
 public:
  ModelPackageContext() = default;

  // Select components that match the provided EP/device info. If multiple
  // components match, the one with the highest device score is chosen.
  Status SelectComponent(gsl::span<EpContextVariantInfo> components,
                         gsl::span<SelectionEpInfo> ep_infos,
                         std::optional<std::filesystem::path>& selected_component_path) const;

 private:
  // Compute a score for a component relative to an EP/device selection.
  int CalculateComponentScore(const EpContextVariantInfo& component) const;
};

}  // namespace onnxruntime

#endif  // !ORT_MINIMAL_BUILD
