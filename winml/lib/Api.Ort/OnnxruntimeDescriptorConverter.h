// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "pch.h"

#include "core/session/winml_adapter_c_api.h"

namespace Windows::AI::MachineLearning {

struct OnnxruntimeValueInfoWrapper {
  const char* name_;
  size_t name_length_;
  const char* description_;
  size_t description_length_;
  OrtTypeInfo* type_info_;
};

class OnnxruntimeEngineFactory;

struct OnnxruntimeDescriptorConverter {
  OnnxruntimeDescriptorConverter(
      OnnxruntimeEngineFactory* engine_factory,
      const std::unordered_map<std::string, std::string>& model_metadata);

  wfc::IVector<winml::ILearningModelFeatureDescriptor>
  ConvertToLearningModelDescriptors(const std::vector<OnnxruntimeValueInfoWrapper>& descriptors);

 private:
  Microsoft::WRL::ComPtr<OnnxruntimeEngineFactory> engine_factory_;
  const std::unordered_map<std::string, std::string>& metadata_;
};

}  // namespace Windows::AI::MachineLearning