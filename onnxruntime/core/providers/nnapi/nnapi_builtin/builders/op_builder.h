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
  virtual std::pair<bool, std::string> IsOpSupported(
      ModelBuilder& model_builder,
      const ONNX_NAMESPACE::NodeProto& node) = 0;

  // Check if the initializers of this operator need preprocess
  // which will not be copied
  virtual void AddInitializersToSkip(ModelBuilder& model_builder,
                                     const ONNX_NAMESPACE::NodeProto& node) = 0;

  // Add the operator to NNAPI model
  virtual void AddOperator(ModelBuilder& model_builder,
                           const ONNX_NAMESPACE::NodeProto& node) = 0;
};

std::unordered_map<std::string, std::shared_ptr<IOpBuilder>>
CreateOpBuilders();

}  // namespace nnapi
}  // namespace onnxruntime