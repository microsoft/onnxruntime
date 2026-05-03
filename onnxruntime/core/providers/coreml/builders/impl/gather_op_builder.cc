// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime::coreml {

class GatherOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool SupportsMLProgram() const override { return true; }
};

namespace {
int64_t GetAxisAttribute(const Node& node) {
  NodeAttrHelper node_attr_helper{node};
  return node_attr_helper.Get("axis", int64_t{0});
}
}  // namespace

Status GatherOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& logger) const {
  const auto axis = GetAxisAttribute(node);
  const auto& data_def = *node.InputDefs()[0];
  const auto& indices_def = *node.InputDefs()[1];
  const auto& output_def = *node.OutputDefs()[0];

  std::vector<int64_t> data_shape, indices_shape;
  ORT_RETURN_IF_NOT(GetShape(data_def, data_shape, logger), "Failed to get 'data' shape");
  ORT_RETURN_IF_NOT(GetShape(indices_def, indices_shape, logger), "Failed to get 'indices' shape");

  // ONNX Gather: out_shape = data_shape[:axis] + indices_shape + data_shape[axis+1:]
  // CoreML's gather requires rank-1+ indices, so for scalar indices we promote
  // them to [1], gather, and then squeeze the resulting axis to restore the
  // original output rank. The positive axis after wrapping is needed for the
  // squeeze axis below regardless of path.
  const bool scalar_indices = indices_shape.empty();
  const int64_t pos_axis = HandleNegativeAxis(axis, data_shape.size());

  if (model_builder.CreateMLProgram()) {
    using CoreML::Specification::MILSpec::Operation;
    constexpr int32_t kInt32 = static_cast<int32_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    constexpr int32_t kInt64 = static_cast<int32_t>(ONNX_NAMESPACE::TensorProto_DataType_INT64);

    int32_t indices_dtype = kInt32;
    GetType(indices_def, indices_dtype, logger);
    const int32_t output_dtype = static_cast<int32_t>(output_def.TypeAsProto()->tensor_type().elem_type());

    std::string indices_name = indices_def.Name();

    if (scalar_indices) {
      // [] -> [1] via reshape. We use reshape rather than expand_dims because
      // CoreML internally pads scalars; expand_dims on the padded tensor can
      // push the apparent rank past the rank-5 limit on high-rank `data`.
      auto reshape = model_builder.CreateOperation(node, "reshape", "indices");
      AddOperationInput(*reshape, "x", indices_def.Name());
      const std::vector<int64_t> indices_1d_shape = {1};
      AddOperationInput(*reshape, "shape",
                        model_builder.AddConstant(reshape->type(), "shape", indices_1d_shape));

      indices_name = model_builder.GetUniqueName(node, "indices_1d");
      AddIntermediateOperationOutput(*reshape, indices_name,
                                     indices_dtype == kInt64 ? kInt64 : kInt32,
                                     indices_1d_shape);
      model_builder.AddOperation(std::move(reshape));
    }

    std::unique_ptr<Operation> gather = model_builder.CreateOperation(node, "gather");
    constexpr bool validate_indices = false;
    AddOperationInput(*gather, "x", data_def.Name());
    AddOperationInput(*gather, "indices", indices_name);
    AddOperationInput(*gather, "axis", model_builder.AddScalarConstant(gather->type(), "axis", axis));
    AddOperationInput(*gather, "validate_indices",
                      model_builder.AddScalarConstant(gather->type(), "validate_indices", validate_indices));

    if (!scalar_indices) {
      AddOperationOutput(*gather, output_def);
      model_builder.AddOperation(std::move(gather));
    } else {
      // gather output here has the data's rank (one more than ONNX scalar-gather output);
      // squeeze the inserted axis to recover the original output shape.
      std::vector<int64_t> gather_shape = data_shape;
      gather_shape[pos_axis] = 1;
      const std::string& gather_out_name = model_builder.GetUniqueName(node, "gather_out");
      AddIntermediateOperationOutput(*gather, gather_out_name, output_dtype, gather_shape);
      model_builder.AddOperation(std::move(gather));

      auto squeeze = model_builder.CreateOperation(node, "squeeze", "post");
      AddOperationInput(*squeeze, "x", gather_out_name);
      const std::vector<int64_t> sq_axes = {pos_axis};
      AddOperationInput(*squeeze, "axes", model_builder.AddConstant(squeeze->type(), "axes", sq_axes));
      AddOperationOutput(*squeeze, output_def);
      model_builder.AddOperation(std::move(squeeze));
    }
  } else {
    if (!scalar_indices) {
      auto layer = model_builder.CreateNNLayer(node);
      layer->mutable_gather()->set_axis(axis);
      *layer->mutable_input()->Add() = data_def.Name();
      *layer->mutable_input()->Add() = indices_def.Name();
      *layer->mutable_output()->Add() = output_def.Name();
      model_builder.AddLayer(std::move(layer));
    } else {
      // expand_dims indices: [] -> [1]
      const std::string& indices_1d_name = model_builder.GetUniqueName(node, "indices_1d");
      {
        auto expand_layer = model_builder.CreateNNLayer(node, "_indices_expand");
        expand_layer->mutable_expanddims()->add_axes(0);
        *expand_layer->mutable_input()->Add() = indices_def.Name();
        *expand_layer->mutable_output()->Add() = indices_1d_name;
        model_builder.AddLayer(std::move(expand_layer));
      }

      // gather with the promoted indices
      const std::string& gather_out_name = model_builder.GetUniqueName(node, "gather_out");
      {
        auto gather_layer = model_builder.CreateNNLayer(node);
        gather_layer->mutable_gather()->set_axis(axis);
        *gather_layer->mutable_input()->Add() = data_def.Name();
        *gather_layer->mutable_input()->Add() = indices_1d_name;
        *gather_layer->mutable_output()->Add() = gather_out_name;
        model_builder.AddLayer(std::move(gather_layer));
      }

      // squeeze the inserted axis
      {
        auto squeeze_layer = model_builder.CreateNNLayer(node, "_post_squeeze");
        squeeze_layer->mutable_squeeze()->add_axes(pos_axis);
        squeeze_layer->mutable_squeeze()->set_squeezeall(false);
        *squeeze_layer->mutable_input()->Add() = gather_out_name;
        *squeeze_layer->mutable_output()->Add() = output_def.Name();
        model_builder.AddLayer(std::move(squeeze_layer));
      }
    }
  }
  return Status::OK();
}

bool GatherOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                             const logging::Logger& logger) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
       !input_params.create_mlprogram || input_params.coreml_version < 6) &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool GatherOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                        const logging::Logger& logger) const {
  std::vector<int64_t> data_shape, indices_shape;
  if (!GetShape(*node.InputDefs()[0], data_shape, logger)) {
    LOGS(logger, VERBOSE) << "Failed to get 'data' shape";
    return false;
  }

  if (!GetShape(*node.InputDefs()[1], indices_shape, logger)) {
    LOGS(logger, VERBOSE) << "Failed to get 'indices' shape";
    return false;
  }

  // For scalar indices we internally emit gather with promoted [1] indices
  // then squeeze. That requires us to claim a static intermediate shape, so
  // we only handle scalar indices when the data shape itself is fully
  // static. (Dynamic-shape scalar Gather still falls back to CPU.)
  if (indices_shape.empty()) {
    if (!IsStaticShape(data_shape)) {
      LOGS(logger, VERBOSE) << "Gather with scalar 'indices' requires static 'data' shape";
      return false;
    }
    // The pre-squeeze intermediate has the same rank as `data`. CoreML's
    // compiler treats rank-5 intermediates as exceeding its internal
    // rank-5 limit when produced via reshape+gather (compiler reports
    // "Invalid rank: 6"), so cap scalar-indices Gather at data rank 4.
    if (data_shape.size() > 4) {
      LOGS(logger, VERBOSE) << "Gather with scalar 'indices' supports 'data' rank up to 4";
      return false;
    }
  }

  // Output rank = data_rank + indices_rank - 1. The rank-5 limit applies.
  if (data_shape.size() + indices_shape.size() - 1 > 5) {
    LOGS(logger, VERBOSE) << "Gather does not support output with rank greater than 5";
    return false;
  }

  return true;
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::coreml
