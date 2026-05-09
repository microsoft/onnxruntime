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
static constexpr const char* kComponentModelMetadataFileName = "metadata.json";
static constexpr const char* kModelVariantsKey = "model_variants";
static constexpr const char* kVariantNameKey = "variant_name";
static constexpr const char* kModelTypeKey = "model_type";
static constexpr const char* kModelFileKey = "model_file";
static constexpr const char* kModelIdKey = "model_id";
static constexpr const char* kConstraintsKey = "constraints";
static constexpr const char* kEpKey = "ep";
static constexpr const char* kDeviceKey = "device";
static constexpr const char* kArchitectureKey = "architecture";
static constexpr const char* kEpCompatibilityInfoKey = "ep_compatibility_info";

class ModelPackageDescriptorParser {
 public:
  explicit ModelPackageDescriptorParser(const logging::Logger& logger) : logger_(logger) {}

  Status ParseVariantsFromRoot(const std::filesystem::path& package_root,
                               /*out*/ std::vector<ModelVariantInfo>& components) const;

 private:
  Status ParseVariantsFromComponent(const std::string& component_model_name,
                                    const std::filesystem::path& component_model_root,
                                    const json* metadata_variants_obj,
                                    /*in,out*/ std::vector<ModelVariantInfo>& variants) const;

  const logging::Logger& logger_;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)