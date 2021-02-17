// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "pch.h"

namespace _winml {

struct OnnxruntimeValueInfoWrapper {
  OnnxruntimeValueInfoWrapper() : type_info_(UniqueOrtTypeInfo(nullptr, nullptr)) {}
  const char* name_ = nullptr;
  size_t name_length_ = 0;
  const char* description_ = nullptr;
  size_t description_length_ = 0;
  UniqueOrtTypeInfo type_info_;
};

class OnnxruntimeEngineFactory;

struct OnnxruntimeDescriptorConverter {
  OnnxruntimeDescriptorConverter(
      OnnxruntimeEngineFactory* engine_factory,
      const std::unordered_map<std::string, std::string>& model_metadata);

  wfc::IVector<winml::ILearningModelFeatureDescriptor>
  ConvertToLearningModelDescriptors(const OnnxruntimeValueInfoWrapper* descriptors, size_t num_descriptors);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  const std::unordered_map<std::string, std::string>& metadata_;
};

}  // namespace _winml