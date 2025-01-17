// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class CastOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

 public:
  bool SupportsMLProgram() const override { return true; }
};

Status CastOpBuilder::AddToModelBuilderImpl([[maybe_unused]] ModelBuilder& model_builder,
                                            [[maybe_unused]] const Node& node,
                                            [[maybe_unused]] const logging::Logger& logger) const {
// This is a special handling case for ArgMax Op, where argmax is followed by a cast to int32 type.
// The ArgMax is fused with the Cast node and produces an int32 output.
#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_unary.cast

    NodeAttrHelper helper(node);
    auto cast_to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
    std::string to_dtype = "";
    if (cast_to_type == ONNX_NAMESPACE::TensorProto::INT32 || cast_to_type == ONNX_NAMESPACE::TensorProto::INT64) {
      to_dtype = "int32";
      // CoreML doesn't support int64, while ONNX uses int64 for indices and as well as data values.
      // We convert the data inputs/outputs between int64 and int32 when calling onnxruntime::coreml::Model::Predict,
      // and when adding int64 initializers to the CoreML model.
      // CoreML operators can only produce int32 and not int64 values.
      // Due to that there should be no actual int64 values inside the CoreML model and we can infer any
      // ONNX_NAMESPACE::TensorProto::INT64 values to be int32.
      cast_to_type = ONNX_NAMESPACE::TensorProto::INT32;
    } else if (cast_to_type == ONNX_NAMESPACE::TensorProto::FLOAT) {
      to_dtype = "fp32";
    } else if (cast_to_type == ONNX_NAMESPACE::TensorProto::FLOAT16) {
      to_dtype = "fp16";
    } else if (cast_to_type == ONNX_NAMESPACE::TensorProto::BOOL) {
      to_dtype = "bool";
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported cast type: ", cast_to_type);
    }

    std::string_view op_type = "cast";
    auto input_dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (((input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
          input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT32) &&
         to_dtype == "int32") ||
        cast_to_type == input_dtype) {
      op_type = "identity";
    }

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, op_type);
    AddOperationInput(*op, "x", node.InputDefs()[0]->Name());
    if (op_type == "cast") {
      AddOperationInput(*op, "dtype", model_builder.AddScalarConstant(op->type(), "dtype", std::string(to_dtype)));
    }
    AddOperationOutput(*op, *node.OutputDefs()[0], cast_to_type);
    model_builder.AddOperation(std::move(op));
  }
#endif

  return Status::OK();
}

bool CastOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  if (node.GetInputEdgesCount() == 0) {
    LOGS(logger, VERBOSE) << "Cast has no preceding nodes.";
    return false;
  }

  if (input_params.create_mlprogram) {
    return true;
  }

  const auto& prec_node = node.InputEdgesBegin()->GetNode();

  /*Cast node is only aimed for supporting argmax and we are only handling the case where an argmax
    followed by a cast node. We need to check if the preceding node is an argmax and also if it's a
    supported argmax op type.*/
  if (prec_node.OpType() != "ArgMax") {
    LOGS(logger, VERBOSE) << "Cast's producing node is not ArgMax is not supported."
                          << "Current producing node: [" << prec_node.OpType()
                          << "]";
    return false;
  }
  if (!IsNodeSupported(prec_node, input_params, logger)) {
    LOGS(logger, VERBOSE) << "Cast's producing node ["
                          << prec_node.OpType()
                          << "] is not a supported op.";
    return false;
  }

  // Check if the output type of cast node is int32
  NodeAttrHelper helper(node);
  const auto cast_to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
  if (cast_to_type != ONNX_NAMESPACE::TensorProto::INT32) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Output type: [" << cast_to_type
                          << "] is not supported.";
    return false;
  }

  return true;
}

bool CastOpBuilder::HasSupportedInputsImpl(const Node& node, [[maybe_unused]] const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  // We only check the type of input 0
  const auto& input = *node.InputDefs()[0];
  const auto& output = *node.OutputDefs()[0];

  int32_t input_type, output_type;
  if (!GetType(input, input_type, logger)) {
    return false;
  }
  if (!GetType(output, output_type, logger)) {
    return false;
  }

#if defined(COREML_ENABLE_MLPROGRAM)
  if (input_params.create_mlprogram) {
    if ((input_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
         input_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
         input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
         input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) &&
        (output_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
         output_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
         output_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
         output_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)) {
      return true;
    } else {
      LOGS(logger, VERBOSE) << "[" << node.OpType()
                            << "] Input type: [" << input_type
                            << "] is not supported.";
      return false;
    }
  }
#endif

  // only support int64 coming from ArgMax (check for ArgMax is done in IsOpSupportedImpl())
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported.";
    return false;
  }

  return true;
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CastOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
