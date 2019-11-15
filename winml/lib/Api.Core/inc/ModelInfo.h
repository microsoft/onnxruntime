// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "WinMLAdapter.h"

namespace Windows::AI::MachineLearning {

class ModelInfo {
 public:
  ModelInfo(const onnx::ModelProto* model_proto);

 public:
  // model metadata
  std::string author_;
  std::string name_;
  std::string domain_;
  std::string description_;
  int64_t version_;
  std::unordered_map<std::string, std::string> model_metadata_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> input_features_;
  wfc::IVector<winml::ILearningModelFeatureDescriptor> output_features_;

 private:
  void Initialize(const onnx::ModelProto* model_proto);
};

}  // namespace Windows::AI::MachineLearning