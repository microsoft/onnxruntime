// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"  // for NodeAttrHelper

namespace onnxruntime::coreml {

class ShapeOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;
  bool SupportsMLProgram() const override { return true; }
};

Status ShapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /*logger*/) const {
  const auto& input_defs = node.InputDefs();

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    NodeAttrHelper node_attr_helper{node};
    int64_t size = -1;
    int64_t num_dims = 0;
    int64_t start = node_attr_helper.Get("start", 0);
    // If the input shape is not available, size is -1 and start is 0
    if (input_defs[0]->Shape()) {
      num_dims = input_defs[0]->Shape()->dim_size();
      start = HandleNegativeAxis(start, num_dims);
      if (node_attr_helper.HasAttr("end")) {
        int64_t end = HandleNegativeAxis(node_attr_helper.Get("end", -1), num_dims);
        size = end - start;
      }
    }

    int32_t output_datatype = ONNX_NAMESPACE::TensorProto_DataType_INT32;
    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "shape");
    AddOperationInput(*op, "x", input_defs[0]->Name());
    if (size != -1 || start != 0) {
      std::string_view layer_input_name_x = model_builder.GetUniqueName(node, "slice_by_size");
      std::vector<int64_t> x0_shape{num_dims};
      AddIntermediateOperationOutput(*op, layer_input_name_x, output_datatype, x0_shape);
      model_builder.AddOperation(std::move(op));

      auto slice_op = model_builder.CreateOperation(node, "slice_by_size");
      AddOperationInput(*slice_op, "x", layer_input_name_x);
      std::vector<int64_t> starts = {start};
      std::vector<int64_t> sizes = {size};
      AddOperationInput(*slice_op, "begin", model_builder.AddConstant(slice_op->type(), "begin", starts));
      AddOperationInput(*slice_op, "size", model_builder.AddConstant(slice_op->type(), "size", sizes));
      AddOperationOutput(*slice_op, *node.OutputDefs()[0], output_datatype);
      model_builder.AddOperation(std::move(slice_op));
    } else {
      AddOperationOutput(*op, *node.OutputDefs()[0], output_datatype);
      model_builder.AddOperation(std::move(op));
    }
  } else  // NOLINT
#endif
  {
    auto layer = model_builder.CreateNNLayer(node);
    layer->mutable_getshape();
    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();
    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool ShapeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  const auto* tensor_shape = node.InputDefs()[0]->Shape();

  NodeAttrHelper node_attr_helper{node};
  if (!input_params.create_mlprogram) {
    if (node_attr_helper.HasAttr("end")) {
      LOGS(logger, VERBOSE) << "Shape does not support 'end' attribute";
      return false;
    }

    if (node_attr_helper.Get("start", 0) != 0) {
      LOGS(logger, VERBOSE) << "Shape does not support 'start' attribute with value other than 0";
      return false;
    }
  } else {
    int64_t end = node_attr_helper.HasAttr("end")
                      ? node_attr_helper.Get("end", -1)
                      : std::numeric_limits<int64_t>::max();
    int64_t start = node_attr_helper.Get("start", 0);
    // no need to slice if start is 0 and end is max
    if (end == std::numeric_limits<int64_t>::max() && start == 0) {
    } else if (tensor_shape == nullptr) {
      LOGS(logger, VERBOSE) << "Shape does not support slicing when tensor_shape is not available";
      return false;
    }
    int64_t dim_size = tensor_shape->dim_size();
    int64_t size = node_attr_helper.HasAttr("end")
                       ? HandleNegativeAxis(node_attr_helper.Get("end", -1), dim_size)
                       : dim_size;
    start = HandleNegativeAxis(start, dim_size);
    size = size - start;
    if (size == 0) {
      LOGS(logger, VERBOSE) << "Shape does not support slicing when size is 0";
      return false;
    }
  }

  return true;
}

bool ShapeOpBuilder::HasSupportedInputsImpl(const Node& node,
                                            [[maybe_unused]] const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  // We only check the type of input 0
  const auto& input = *node.InputDefs()[0];

  int32_t input_type;
  if (!GetType(input, input_type, logger)) {
    return false;
  }

  if (input_params.create_mlprogram) {
    if ((input_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
         input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
         input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)) {
      return true;
    } else {
      LOGS(logger, VERBOSE) << "[" << node.OpType()
                            << "] Input type: [" << input_type
                            << "] is not supported.";
      return false;
    }
  } else if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported.";
    return false;
  }

  return true;
}

void CreateShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ShapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::coreml
