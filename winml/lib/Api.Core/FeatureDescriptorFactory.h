#pragma once
#include "pch.h"

namespace Windows::AI::MachineLearning {

struct FeatureDescriptorFactory {
  FeatureDescriptorFactory(
      const std::unordered_map<std::string, std::string>& model_metadata);

  wfc::IVector<winml::ILearningModelFeatureDescriptor>
  CreateDescriptorsFromValueInfoProtos(
      const std::vector<const onnx::ValueInfoProto*>& value_info_protos);

 private:
  const std::unordered_map<std::string, std::string>& metadata_;
};

}  // namespace Windows::AI::MachineLearning