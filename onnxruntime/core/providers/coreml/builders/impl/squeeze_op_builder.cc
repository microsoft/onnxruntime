// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/cpu/tensor/unsqueeze.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

namespace onnxruntime {
namespace coreml {

class SqueezeOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool SupportsMLProgram() const override { return true; }
};

namespace {
void GetAxes(ModelBuilder& model_builder, const Node& node, TensorShapeVector& axes) {
  // Squeeze opset 13 use input as axes
  if (node.SinceVersion() > 12) {
    // If axes is not provided, return an empty axes as default to squeeze all
    if (node.InputDefs().size() > 1) {
      const auto& axes_tensor = *model_builder.GetConstantInitializer(node.InputDefs()[1]->Name());
      Initializer unpacked_tensor(axes_tensor);
      auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
      const auto size = SafeInt<size_t>(axes_tensor.dims()[0]);
      axes.reserve(size);
      axes.insert(axes.end(), raw_axes.begin(), raw_axes.end());
    }
  } else {
    NodeAttrHelper helper(node);
    auto axes_attr = helper.Get("axes", std::vector<int64_t>());
    axes.assign(axes_attr.begin(), axes_attr.end());
  }
}
}  // namespace

void SqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  if (node.SinceVersion() > 12 && node.InputDefs().size() > 1) {
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
  }
}

#if defined(COREML_ENABLE_MLPROGRAM)
void HandleX86ArchUnsqueezeScalarInput(ModelBuilder& model_builder,
                                       const Node& node, const logging::Logger& logger) {
  const auto& input_defs(node.InputDefs());
  TensorShapeVector axes;
  GetAxes(model_builder, node, axes);

  std::vector<int64_t> input_shape;
  GetShape(*input_defs[0], input_shape, logger);
  auto op = model_builder.CreateOperation(node, "reshape");
  AddOperationInput(*op, "x", input_defs[0]->Name());
  TensorShapeVector output_shape = UnsqueezeBase::ComputeOutputShape(TensorShape(input_shape), axes);
  AddOperationInput(*op, "shape", model_builder.AddConstant(op->type(), "shape", AsSpan(output_shape)));
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
}
#endif

Status SqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               [[maybe_unused]] const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);
  auto* coreml_squeeze = layer->mutable_squeeze();
  TensorShapeVector axes;
  GetAxes(model_builder, node, axes);
#if defined(COREML_ENABLE_MLPROGRAM)
  const auto& input_defs(node.InputDefs());
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

#if defined(TARGET_CPU_X86_64) && TARGET_CPU_X86_64
    // expand_dims has limited requirements for static shape, however, X86_64 has a bug that it can't handle scalar input
    if (node.OpType() == "Unsqueeze" && input_defs[0]->Shape()->dim_size() < 2) {
      HandleX86ArchUnsqueezeScalarInput(model_builder, node, logger);
      return Status::OK();
    }
#endif
    std::string_view coreml_op_type = node.OpType() == "Squeeze" ? "squeeze" : "expand_dims";
    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, coreml_op_type);
    AddOperationInput(*op, "x", input_defs[0]->Name());

    if (!axes.empty()) {
      // coreml supports negative axes
      AddOperationInput(*op, "axes", model_builder.AddConstant(op->type(), "axes", AsSpan(axes)));
    }
    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  } else  // NOLINT
#endif
  {
    if (axes.empty()) {
      coreml_squeeze->set_squeezeall(true);
    } else {
      *coreml_squeeze->mutable_axes() = {axes.cbegin(), axes.cend()};
      coreml_squeeze->set_squeezeall(false);
    }

    *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool SqueezeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                         const logging::Logger& logger) const {
  // Squeeze opset 13 uses input 1 as axes, if we have input 1 then it needs to be an initializer
  const auto& input_defs = node.InputDefs();
  if (node.SinceVersion() > 12 && input_defs.size() > 1) {
    const auto& axes_name = input_defs[1]->Name();
    if (!input_params.graph_viewer.GetConstantInitializer(axes_name)) {
      LOGS(logger, VERBOSE) << "Input axes must be known";
      return false;
    }
  }

  if (node.OpType() == "Unsqueeze") {
    if (!input_params.create_mlprogram) {
      return false;
    }

    int64_t num_of_new_dims = 0;
    if (node.SinceVersion() > 12) {
      num_of_new_dims = node.InputDefs()[1]->Shape()->dim(0).dim_value();
    } else {
      NodeAttrHelper helper(node);
      auto axes = helper.Get("axes", std::vector<int64_t>());
      num_of_new_dims = static_cast<int64_t>(axes.size());
    }

    std::vector<int64_t> input_shape;
    if (!GetShape(*input_defs[0], input_shape, logger) || input_shape.size() + num_of_new_dims > 5) {
      LOGS(logger, VERBOSE) << "Unsqueeze to output shape with > 5 dimensions is not supported";
      return false;
    }
  }
  return true;
}

void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
