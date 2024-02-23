// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

class GatherOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

// Add operator related

void GatherOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  const auto& indices_name = inputs[1].node_arg.Name();
  int32_t indices_data_type;
  GetType(node_unit.Inputs()[1].node_arg, indices_data_type);
  if (Contains(model_builder.GetInitializerTensors(), indices_name) &&
      indices_data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    // Skip the second input `indices` for Gather if it is an initializer
    model_builder.AddInitializerToSkip(indices_name);
  }
}

Status GatherOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& input1 = node_unit.Inputs()[0].node_arg.Name();
  const auto& input2 = node_unit.Inputs()[1].node_arg.Name();  // "indices"
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  NodeAttrHelper helper(node_unit);
  int32_t rank = static_cast<int32_t>(shaper[input1].size());
  int32_t axis = static_cast<int32_t>(HandleNegativeAxis(helper.Get("axis", 0), rank));

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  auto output_shape = shaper[output];
  bool need_squeeze = false;
  int32_t indices_data_type;
  GetType(node_unit.Inputs()[1].node_arg, indices_data_type);
  if (Contains(model_builder.GetInitializerTensors(), input2) &&
      indices_data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    // Add indices operand into nnapi
    const auto& indices_tensor = *initializers.at(input2);
    Initializer unpacked_tensor(indices_tensor);

    const auto data_type = indices_tensor.data_type();
    const auto indices_shape = indices_tensor.dims();
    Shape indices_dimen;
    indices_dimen.reserve(indices_tensor.dims_size());
    for (auto i = 0; i < indices_tensor.dims_size(); i++) {
      indices_dimen.push_back(static_cast<uint32_t>(indices_shape[i]));
    }

    std::vector<int32_t> indices(unpacked_tensor.size());

    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      auto indice_span = unpacked_tensor.DataAsSpan<int64_t>();
      std::transform(indice_span.begin(), indice_span.end(), indices.begin(),
                     [](int64_t indice_n) -> int32_t { return SafeInt<int32_t>(indice_n); });
    } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      auto indice_span = unpacked_tensor.DataAsSpan<int32_t>();
      indices.assign(indice_span.begin(), indice_span.end());
    }

    // If indices is a scalar, insert a dimension of 1
    // ONNX spec says that indices can be q-D tensor, q could be any value >= 0
    // When indices is a scalar, output shape would be automatically squeezed.
    // Add a fake output dimension to avoid NNAPI error.
    if (indices_dimen.empty()) {
      indices_dimen.push_back(1);
      output_shape.insert(output_shape.begin() + axis, 1);
      need_squeeze = true;
    }

    OperandType indices_operand_type(Type::TENSOR_INT32, indices_dimen);
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(input2, indices.data(), indices_operand_type));
  }
  input_indices.push_back(operand_indices.at(input2));

  const OperandType output_operand_type(operand_types.at(input1).type, output_shape);
  if (!need_squeeze) {
    return model_builder.AddOperation(ANEURALNETWORKS_GATHER, input_indices,
                                      {output}, {output_operand_type});
  }

  std::string intermediate_output = model_builder.GetUniqueName(node_unit.Name() + "_need_squeeze");
  shaper.AddShape(intermediate_output, output_shape);

  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_GATHER, input_indices,
                                                 {intermediate_output}, {output_operand_type}));

  std::string squeeze_op_name = model_builder.GetUniqueName(node_unit.Name() + "_squeeze");
  return op_builder_helpers::AddSqueezeOp(model_builder, squeeze_op_name, intermediate_output, output, {axis});
}

// Operator support related

bool GatherOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                        const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;

  if (!GetShape(inputs[0].node_arg, input_shape)) {
    return false;
  }

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Gather only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](int32_t i) { return i == 0; })) {
    LOGS_DEFAULT(VERBOSE) << "Gather doesn't support dynamic input shape";
    return false;
  }

  // Here in GatherOpBuilder::IsOpSupportedImpl, we removed the restriction that 2nd input "indices" must be an
  // initializer to accommodate the support for some models such as mobileBERT.
  // It doesn't need to be an initializer for int32 type as NNAPI Gather uses int32 for indices so the type matches.
  // However, we still require indices of other types to be an initializer as we convert the data to int32
  // during model building.
  // TODO: We could potentially support non-initializer inputs for the other types if we inserted a cast.
  const auto& indices_name = inputs[1].node_arg.Name();

  int32_t indices_type;
  if (!GetType(node_unit.Inputs()[1].node_arg, indices_type))
    return false;

  if (indices_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    if (!graph_viewer.GetConstantInitializer(indices_name)) {
      LOGS_DEFAULT(VERBOSE) << "Indices of Gather must be a constant initializer.";
      return false;
    }
  }

  return true;
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
