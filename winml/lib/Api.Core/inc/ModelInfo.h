// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.#pragma once

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

// factory methods for creating an ort model from a path
std::unique_ptr<onnx::ModelProto> CreateModelProto(const char* path);

// factory methods for creating an ort model from a stream
std::unique_ptr<onnx::ModelProto> CreateModelProto(const wss::IRandomAccessStreamReference& stream_reference);

}  // namespace Windows::AI::MachineLearning