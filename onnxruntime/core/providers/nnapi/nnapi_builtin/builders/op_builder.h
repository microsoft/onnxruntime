// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace nnapi {

class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;
  virtual std::pair<bool, std::string> IsOpSupported() = 0;
  virtual void AddInitializersToSkip() = 0;
  virtual void AddOperator() = 0;
};

std::unordered_map<std::string, std::unique_ptr<IOpBuilder>>
CreateOpBuilders();

std::unique_ptr<IOpBuilder> CreateOpBuilder(
    ModelBuilder& model_builder,
    const ONNX_NAMESPACE::NodeProto& node);

}  // namespace nnapi
}  // namespace onnxruntime