// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/model_package/model_package_context.h"

namespace onnxruntime {

//
// Keys for parsing model package JSON files.
//
static constexpr const char* kModelPackageManifestFileName = "manifest.json";
static constexpr const char* kComponentNameInMetadataKey = "component_name";
static constexpr const char* kComponentMetadataFileName = "metadata.json";

static constexpr const char* kEpCompatibilityKey = "ep_compatibility";
static constexpr const char* kEpKey = "ep";
static constexpr const char* kDeviceKey = "device";
static constexpr const char* kCompatibilityStringKey = "compatibility_string";

static constexpr const char* kSessionOptionsKey = "session_options";
static constexpr const char* kProviderOptionsKey = "provider_options";
static constexpr const char* kConsumerMetadataKey = "consumer_metadata";

static constexpr const char* kFilesKey = "files";
static constexpr const char* kFilenameKey = "filename";
static constexpr const char* kSharedFilesKey = "shared_files";

static constexpr const char* kSchemaVersionKey = "schema_version";
static constexpr const char* kComponentsKey = "components";
static constexpr const char* kVariantsKey = "variants";
static constexpr const char* kVariantDescriptorFileName = "variant.json";

class ModelPackageDescriptorParser {
 public:
  explicit ModelPackageDescriptorParser(const logging::Logger& logger) : logger_(logger) {}

  // Legacy mixed-mode API:
  // - component root if metadata.json exists at root
  // - package root otherwise
  Status ParseVariantsFromRoot(const std::filesystem::path& package_root,
                               /*out*/ std::vector<VariantInfo>& components) const;

  // Explicit package-root API (Mode 2).
  Status ParseVariantsFromPackageRoot(const std::filesystem::path& package_root,
                                      /*out*/ std::vector<VariantInfo>& variants) const;

 private:
  Status ParseVariantsFromComponent(const std::string& component_name,
                                    const std::filesystem::path& component_model_root,
                                    const json* metadata_variants_obj,
                                    /*in,out*/ std::vector<VariantInfo>& variants) const;

  const logging::Logger& logger_;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
