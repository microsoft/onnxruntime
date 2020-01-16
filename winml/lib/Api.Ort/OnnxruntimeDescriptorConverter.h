// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "pch.h"

#include "adapter/winml_adapter_c_api.h"

namespace Windows::AI::MachineLearning {

struct OnnxruntimeValueInfoWrapper {
  OnnxruntimeValueInfoWrapper() : type_info_(std::unique_ptr<OrtTypeInfo, void (*)(OrtTypeInfo*)>(nullptr, nullptr)) {}
  const char* name_ = nullptr;
  size_t name_length_ = 0;
  const char* description_ = nullptr;
  size_t description_length_ = 0;
  std::unique_ptr<OrtTypeInfo, void (*)(OrtTypeInfo*)> type_info_;
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