// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"
#include <numeric>

namespace onnxruntime {
namespace coreml {

class NormalizationOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
  Status AddGroupNormToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                        const logging::Logger& logger) const;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 1; }

 public:
  bool SupportsMLProgram() const override { return true; }
};

void NormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // skip everything except input0 for Normalization
  const auto& input_defs = node.InputDefs();
  model_builder.AddInitializerToSkip(input_defs[1]->Name());  // scale
  if (input_defs.size() > 2) {
    model_builder.AddInitializerToSkip(input_defs[2]->Name());  // B
  }
}

Status NormalizationOpBuilder::AddToModelBuilderImpl(
    [[maybe_unused]] ModelBuilder& model_builder,
    [[maybe_unused]] const Node& node,
    [[maybe_unused]] const logging::Logger& logger) const {
  if (node.OpType() == "GroupNormalization") {
    return AddGroupNormToModelBuilderImpl(model_builder, node, logger);
  }
#if defined(COREML_ENABLE_MLPROGRAM)
  const auto& input_defs = node.InputDefs();
  NodeAttrHelper helper(node);
  const auto& scale_tensor = *model_builder.GetConstantInitializer(input_defs[1]->Name());

  const auto eps = helper.Get("epsilon", 1e-5f);

  std::vector<int64_t> input_shape;
  // GetShape will never fail as we have already verified the input shape in IsOpSupportedImpl
  GetShape(*input_defs[0], input_shape, logger);

  const auto rank = input_shape.size();
  auto axis = static_cast<size_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  std::vector<int64_t> axes(rank - axis);
  std::iota(axes.begin(), axes.end(), axis);
  auto input_dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    std::string_view layer_input_name_x = node.InputDefs()[0]->Name();
    std::string_view op_name = (node.OpType() == "InstanceNormalization") ? "instance_norm" : "layer_norm";
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.normalization.layer_norm

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, op_name);
    AddOperationInput(*op, "x", layer_input_name_x);
    if (op_name == "layer_norm") {
      AddOperationInput(*op, "axes", model_builder.AddConstant(op->type(), input_defs[0]->Name() + "axes", axes));
    }
    AddOperationInput(*op, "gamma", model_builder.AddConstant(op->type(), input_defs[1]->Name() + "gamma", scale_tensor));
    if (input_defs.size() > 2) {
      const auto& bias_tensor = *model_builder.GetConstantInitializer(input_defs[2]->Name());
      AddOperationInput(*op, "beta", model_builder.AddConstant(op->type(), input_defs[2]->Name() + "beta", bias_tensor));
    }

    if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      MLFloat16 epsilon_fp16(eps);
      AddOperationInput(*op, "epsilon", model_builder.AddScalarConstant(op->type(), "epsilon", epsilon_fp16));
    } else {
      AddOperationInput(*op, "epsilon", model_builder.AddScalarConstant(op->type(), "epsilon", eps));
    }

    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  }
#endif  // (COREML_ENABLE_MLPROGRAM)

  return Status::OK();
}

Status NormalizationOpBuilder::AddGroupNormToModelBuilderImpl(
    [[maybe_unused]] ModelBuilder& model_builder,
    [[maybe_unused]] const Node& node,
    [[maybe_unused]] const logging::Logger& logger) const {
#if defined(COREML_ENABLE_MLPROGRAM)
  const auto& input_defs = node.InputDefs();
  NodeAttrHelper helper(node);
  // Coreml hasn't supported GroupNorm yet.
  // we decompose GroupNorm to sub ops and levrage LayerNorm to implement GroupNorm.
  // groupnorm --> reshape [b, num_groups, c // (num_groups), h, w] --> layer_norm --> reshape [b, c, h, w]->mul(scale)->add(bias)

  // scale and bias is required for group-norm by the onnx spec
  const auto& scale_tensor = *model_builder.GetConstantInitializer(input_defs[1]->Name());
  const auto& bias_tensor = *model_builder.GetConstantInitializer(input_defs[2]->Name());

  const auto eps = helper.Get("epsilon", 1e-5f);
  int64_t num_groups = helper.Get("num_groups", 1);  // GroupNorm

  std::vector<int64_t> input_shape;
  GetShape(*input_defs[0], input_shape, logger);

  const auto input_size = input_shape.size();
  int64_t axis = 2;
  std::vector<int64_t> axes(input_size + 1 - axis);  // Group add one more dim
  std::iota(axes.begin(), axes.end(), axis);
  auto input_dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int64_t channel_dims = input_shape[1];

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    std::string_view layer_input_name_x = node.InputDefs()[0]->Name();
    const int32_t elem_type = static_cast<int32_t>(input_dtype);

    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.normalization.layer_norm
    // https://github.com/apple/coremltools/blob/9827d424b3c5b5fbb6ddc8891a000d87a188c84f/coremltools/converters/mil/frontend/torch/ops.py#L1354
    // reshape to [b, num_groups, c // (num_groups), h, w]
    auto reshape1 = model_builder.CreateOperation(node, "reshape", "pre");
    std::vector<int64_t> shape1 = input_shape;
    shape1.insert(shape1.begin() + 1, num_groups);
    shape1[2] = input_shape[1] / num_groups;
    std::vector<int64_t> shape_scale_bias(input_shape.size(), 1);
    shape_scale_bias[1] = channel_dims;
    AddOperationInput(*reshape1, "x", node.InputDefs()[0]->Name());
    AddOperationInput(*reshape1, "shape", model_builder.AddConstant(reshape1->type(), "shape1", shape1));
    layer_input_name_x = model_builder.GetUniqueName(node, "ln_reshape1_");
    AddIntermediateOperationOutput(*reshape1, layer_input_name_x, elem_type, shape1);

    std::unique_ptr<Operation> layer_norm = model_builder.CreateOperation(node, "layer_norm");
    AddOperationInput(*layer_norm, "x", layer_input_name_x);
    AddOperationInput(*layer_norm, "axes", model_builder.AddConstant(layer_norm->type(), "axes", axes));

    if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      MLFloat16 epsilon_fp16(eps);
      AddOperationInput(*layer_norm, "epsilon", model_builder.AddScalarConstant(layer_norm->type(), "epsilon", epsilon_fp16));
    } else {
      AddOperationInput(*layer_norm, "epsilon", model_builder.AddScalarConstant(layer_norm->type(), "epsilon", eps));
    }

    const auto& ln_output_name = model_builder.GetUniqueName(node, "ln_output_");
    AddIntermediateOperationOutput(*layer_norm, ln_output_name, elem_type, shape1);

    auto reshape2 = model_builder.CreateOperation(node, "reshape", "post");
    AddOperationInput(*reshape2, "x", ln_output_name);
    AddOperationInput(*reshape2, "shape", model_builder.AddConstant(reshape2->type(), "shape2", input_shape));

    const auto& reshape2_output_name = model_builder.GetUniqueName(node, "gn_reshape_output_");
    AddIntermediateOperationOutput(*reshape2, reshape2_output_name, elem_type, input_shape);

    auto mul = model_builder.CreateOperation(node, "mul", "post_mul");
    AddOperationInput(*mul, "x", reshape2_output_name);
    AddOperationInput(*mul, "y", model_builder.AddConstant(mul->type(), "mul1", scale_tensor, shape_scale_bias));
    const auto& mul_output_name = model_builder.GetUniqueName(node, "mul_output_");
    AddIntermediateOperationOutput(*mul, mul_output_name, elem_type, input_shape);

    auto add = model_builder.CreateOperation(node, "add", "post_add");
    AddOperationInput(*add, "x", mul_output_name);
    AddOperationInput(*add, "y", model_builder.AddConstant(add->type(), "add1", bias_tensor, shape_scale_bias));
    AddOperationOutput(*add, *node.OutputDefs()[0]);

    model_builder.AddOperation(std::move(reshape1));
    model_builder.AddOperation(std::move(layer_norm));
    model_builder.AddOperation(std::move(reshape2));
    model_builder.AddOperation(std::move(mul));
    model_builder.AddOperation(std::move(add));
  }
#endif  // (COREML_ENABLE_MLPROGRAM)
  return Status::OK();
}

bool NormalizationOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                               const logging::Logger& logger) const {
  // LayerNormalization may have three output in the training mode, but we only support the inference mode
  // for InstanceNormalization and GroupNormalization, they only have one output, so this check will always return true
  if (node.OutputDefs().size() != 1) {
    LOGS(logger, VERBOSE) << "Your onnx model (with LayerNormalization) may be in training mode,"
                          << " please export it for inferencing.";
    return false;
  }
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  // groupnorm and layernorm has attribute "stash_type", while InstanceNormalization doesn't have this attribute
  // Type of Mean and InvStdDev. This also specifies stage one’s computation precision.
  //  if stash_type is 1, this operator casts all input variables to 32-bit float,
  // perform the computation, and finally cast Normalized back to the original type of X
  // coreml didn't have a similiar attribute to stash_type, for now, we support the default value
  if (node.OpType() != "InstanceNormalization") {
    NodeAttrHelper helper(node);
    const auto stash_type = helper.Get("stash_type", 1);
    if (stash_type != 1) {
      LOGS(logger, VERBOSE) << "stash_type != 1 is not supported";
      return false;
    }
  }

  const auto& scale_name = input_defs[1]->Name();
  const auto* scale_tensor = input_params.graph_viewer.GetConstantInitializer(scale_name);
  if (!scale_tensor) {
    LOGS(logger, VERBOSE) << "Scale must be a constant initializer";
    return false;
  }

  if (input_defs.size() > 2) {
    const auto& b_name = input_defs[2]->Name();
    const auto& b_tensor = input_params.graph_viewer.GetConstantInitializer(b_name);
    if (!b_tensor) {
      LOGS(logger, VERBOSE) << "Bias must be a constant initializer";
      return false;
    }
  }

  return true;
}

bool NormalizationOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                                    const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    return false;
  }
  // We only check the type of input 0,1,2
  const auto& input_0 = *node.InputDefs()[0];
  const auto& input_1 = *node.InputDefs()[1];
  const auto& input_2 = node.InputDefs().size() > 2 ? *node.InputDefs()[2] : input_0;
  int32_t input_type_0, input_type_1, input_type_2;
  if (!GetType(input_0, input_type_0, logger)) {
    return false;
  }
  if (!GetType(input_1, input_type_1, logger)) {
    return false;
  }
  if (!GetType(input_2, input_type_2, logger)) {
    return false;
  }
  if (input_type_0 != input_type_1 || input_type_0 != input_type_2) {
    LOGS(logger, VERBOSE) << "Input types of LayerNorm must be the same";
    return false;
  }

  if (input_type_0 != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type_0 != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << "Input types of LayerNorm must be float or float16";
    return false;
  }
  return true;
}

void CreateNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<NormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
