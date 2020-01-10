// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "pch.h"

#include "core/session/winml_adapter_c_api.h"

namespace Windows::AI::MachineLearning {

struct FeatureDescriptor
{
  const char* name_;
  const char* description;
  const OrtTypeInfo* type_info_;
};

struct FeatureDescriptorFactory {
  FeatureDescriptorFactory(
      const std::unordered_map<std::string, std::string>& model_metadata);

  wfc::IVector<winml::ILearningModelFeatureDescriptor>
  CreateLearningModelFeatureDescriptors(
      const std::vector<const FeatureDescriptor>& descriptors);

 private:
  const std::unordered_map<std::string, std::string>& metadata_;
};

}  // namespace Windows::AI::MachineLearning