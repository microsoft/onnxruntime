// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace nnapi {

class ModelBuilder;

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  // Check if an operator is supported
  virtual bool IsOpSupported(ModelBuilder& model_builder, const Node& node) = 0;

  // Check if the initializers of this operator need preprocess
  // which will not be copied
  virtual void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) = 0;

  // Add the operator to NNAPI model
  virtual void AddToModelBuilder(ModelBuilder& model_builder, const Node& node) = 0;
};

// Generate a lookup table with IOpBuilder delegates
// for different onnx operators
std::unordered_map<std::string, std::shared_ptr<IOpBuilder>> CreateOpBuilders();

// Transpose the NHWCinput to NCHW output
void TransposeNHWCToNCHW(ModelBuilder& model_builder, const std::string& input, const std::string& output);

}  // namespace nnapi
}  // namespace onnxruntime
