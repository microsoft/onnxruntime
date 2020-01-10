// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "pch.h"

#include "core/session/winml_adapter_c_api.h"

namespace Windows::AI::MachineLearning {

struct FeatureDescriptor {
  const char* name_;
  size_t name_length_;
  const char* description;
  size_t description_length_;
  OrtTypeInfo* type_info_;
};

struct FeatureDescriptorFactory {
  FeatureDescriptorFactory(
      const std::unordered_map<std::string, std::string>& model_metadata);

  wfc::IVector<winml::ILearningModelFeatureDescriptor>
  CreateLearningModelFeatureDescriptors(
      const std::vector<FeatureDescriptor>& descriptors);

 private:
  const std::unordered_map<std::string, std::string>& metadata_;
};

}  // namespace Windows::AI::MachineLearning