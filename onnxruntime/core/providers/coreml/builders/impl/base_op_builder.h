// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/span_utils.h"
#include "core/providers/coreml/builders/coreml_spec.h"
#include "core/providers/coreml/builders/op_builder.h"

namespace onnxruntime {
namespace coreml {

class ModelBuilder;

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;

  // does the operator implementation support creating an ML Program
  bool SupportsMLProgram() const override { return false; }

  bool IsOpSupported(const Node& node, const OpBuilderInputParams& input_params,
                     const logging::Logger& logger) const override final;

  Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                           const logging::Logger& logger) const override final;

  void AddInitializersToSkip(ModelBuilder& /*model_builder*/, const Node& /*node*/) const override {}

 protected:
  // currently we only support float
  static bool IsInputFloat(const Node& node, size_t idx, const OpBuilderInputParams& input_params,
                           const logging::Logger& logger);

 private:
  virtual bool IsOpSupportedImpl(const Node& /*node*/, const OpBuilderInputParams& /*input_params*/,
                                 const logging::Logger& /*logger*/) const {
    return true;
  }

  virtual bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const;

  virtual int GetMinSupportedOpSet(const Node& /*node*/) const { return 1; }
  virtual int GetMaxSupportedOpSet(const Node& /*node*/) const { return 20; }

  bool HasSupportedOpSet(const Node& node, const logging::Logger& logger) const;
  bool HasSupportedInputs(const Node& node, const OpBuilderInputParams& input_params,
                          const logging::Logger& logger) const;

  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                       const logging::Logger& logger) const = 0;
};

}  // namespace coreml
}  // namespace onnxruntime
