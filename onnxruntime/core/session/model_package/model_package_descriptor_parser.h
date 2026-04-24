// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_context.h"

namespace onnxruntime {

//
// Keys for parsing the model package manifest json and component model metadata json.
//
static constexpr const char* kModelPackageManifestFileName = "manifest.json";
static constexpr const char* kModelNameKey = "model_name";
static constexpr const char* kModelVersionKey = "model_version";
static constexpr const char* kComponentModelsKey = "component_models";
static constexpr const char* kComponentModelNameInMetadataKey = "component_model_name";
static constexpr const char* kComponentModelMetadataFileName = "metadata.json";
static constexpr const char* kModelVariantsKey = "model_variants";
static constexpr const char* kVariantNameKey = "variant_name";
static constexpr const char* kModelInfoKey = "model_info";
static constexpr const char* kModelFileKey = "model_file";
static constexpr const char* kIdentifierKey = "identifier";
static constexpr const char* kEpCompatibilityKey = "ep_compatibility";
static constexpr const char* kEpKey = "ep";
static constexpr const char* kDeviceTypeKey = "device_type";
static constexpr const char* kCompatibilityInfoKey = "compatibility_info";
static constexpr const char* kSessionOptionsKey = "session_options";
static constexpr const char* kProviderOptionsKey = "provider_options";
static constexpr const char* kConsumerMetadataKey = "consumer_metadata";

static constexpr const char* kFilesKey = "files";
static constexpr const char* kPathKey = "path";
static constexpr const char* kFileTypeKey = "type";

static constexpr const char* kEpKey = "ep";
static constexpr const char* kDeviceKey = "device";
static constexpr const char* kArchitectureKey = "architecture";
static constexpr const char* kEpCompatibilityInfoKey = "ep_compatibility_info";

class ModelPackageDescriptorParser {
 public:
  explicit ModelPackageDescriptorParser(const logging::Logger& logger) : logger_(logger) {}

  // Legacy mixed-mode API:
  // - component root if metadata.json exists at root
  // - package root otherwise
  Status ParseVariantsFromRoot(const std::filesystem::path& package_root,
                               /*out*/ std::vector<ModelVariantInfo>& components) const;

  // Explicit package-root API (Mode 2).
  Status ParseVariantsFromPackageRoot(const std::filesystem::path& package_root,
                                      /*out*/ std::vector<ModelVariantInfo>& variants) const;

 private:
  Status ParseVariantsFromComponent(const std::string& component_model_name,
                                    const std::filesystem::path& component_model_root,
                                    const json* metadata_variants_obj,
                                    /*in,out*/ std::vector<ModelVariantInfo>& variants) const;

  const logging::Logger& logger_;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)