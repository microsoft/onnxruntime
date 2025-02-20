// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_gemm_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <string>

#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

namespace {

const NodeUnit* GetReshapeNodeUnit(
    const GraphViewer& graph_viewer, const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Node& gemm_node) {
  if (gemm_node.OpType() != "Gemm") {
    return nullptr;
  }

  for (auto edge = gemm_node.InputEdgesBegin(); edge != gemm_node.InputEdgesEnd(); ++edge) {
    if (edge->GetDstArgIndex() == 0) {
      const Node& reshape_node = edge->GetNode();
      if (reshape_node.OpType() == "Reshape" && !graph_viewer.NodeProducesGraphOutput(reshape_node) &&
          reshape_node.GetOutputEdgesCount() == 1) {
        const auto it = node_to_node_unit.find(&reshape_node);
        if (it != node_to_node_unit.end()) {
          const NodeUnit* reshape_node_unit = it->second;
          if (reshape_node_unit && node_unit_to_qnn_node_group.count(reshape_node_unit) == 0 &&
              reshape_node_unit->UnitType() == NodeUnit::Type::SingleNode) {
            return reshape_node_unit;
          }
        }
      }
    }
  }

  return nullptr;
}

// Reshape from [x0, x1, ..., xn, k] to [x0 * x1 * ... * xn, k].
bool CheckShape(const Node& reshape_node) {
  auto input_shape_proto = reshape_node.InputDefs()[0]->Shape();
  auto output_shape_proto = reshape_node.OutputDefs()[0]->Shape();
  if (!input_shape_proto || !output_shape_proto) {
    return false;
  }

  auto input_shape = utils::GetTensorProtoShape(*input_shape_proto);
  auto output_shape = utils::GetTensorProtoShape(*output_shape_proto);
  auto input_rank = input_shape.NumDimensions();
  auto output_rank = output_shape.NumDimensions();
  return input_shape.Size() != -1 && output_shape.Size() != -1 && output_rank == 2 &&
         input_shape.SizeToDimension(input_rank - 1) == output_shape[0] &&
         input_shape[input_rank - 1] == output_shape[1];
}

Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& reshape_node_unit,
                             const NodeUnit& gemm_node_unit, bool validate) {
  assert(reshape_node_unit.OpType() == "Reshape" && gemm_node_unit.OpType() == "Gemm");
  const auto& node_name = utils::GetNodeName(gemm_node_unit);
  const NodeUnitIODef& input_def = reshape_node_unit.Inputs()[0];
  const NodeUnitIODef& weight_def = gemm_node_unit.Inputs()[1];
  const NodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = gemm_node_unit.Inputs().size() == 3 && gemm_node_unit.Inputs()[2].node_arg.Exists();
  if (has_bias) {
    bias_def_ptr = &gemm_node_unit.Inputs()[2];
  }
  const NodeUnitIODef& output_def = gemm_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper bias_tensor;
  QnnTensorWrapper output_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  std::vector<uint32_t> weight_shape;
  std::vector<uint8_t> unpacked_tensor;
  std::string weight_tensor_name = weight_def.node_arg.Name();
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(weight_def.node_arg, weight_shape), "Failed to get weight shape");
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  Qnn_DataType_t data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false, weight_def.node_arg.TypeAsProto(), data_type));
  const auto& weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  ORT_RETURN_IF_ERROR(
      utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, *weight_tensor_proto, unpacked_tensor));
  QnnTensorWrapper weight_tensor(weight_tensor_name, tensor_type, data_type, QnnQuantParamsWrapper(),
                                 std::move(weight_shape), std::move(unpacked_tensor));
  if (has_bias) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(*bias_def_ptr, bias_tensor));
  }
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    std::vector<Qnn_Tensor_t> input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_FULLY_CONNECTED, std::move(input_tensors),
                                                          {output_tensor.GetQnnTensor()}, {}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
    if (has_bias) {
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    std::vector<std::string> input_names = {input_def.node_arg.Name(), weight_tensor_name};
    if (has_bias) {
      input_names.emplace_back(bias_def_ptr->node_arg.Name());
    }
    ORT_RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                        std::move(input_names), {output_def.node_arg.Name()}, {}, validate),
        "Failed to add fused Gemm node.");
  }

  return Status::OK();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper, const NodeUnit& gemm_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& /*logger*/) {
  if (gemm_node_unit.OpType() != "Gemm" || gemm_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& gemm_node = gemm_node_unit.GetNode();
  NodeAttrHelper helper(gemm_node);
  auto transA = helper.Get("transA", static_cast<int64_t>(0));
  auto transB = helper.Get("transB", static_cast<int64_t>(0));
  const auto& weight_input = gemm_node_unit.Inputs()[1];
  // The pattern is from MatMul->Add, so the transA and transB should be false, and weight should be initializer.
  // Currently we don't handle quantized weight.
  if (transA != 0 || transB != 0 || !qnn_model_wrapper.IsConstantInput(weight_input.node_arg.Name()) ||
      weight_input.quant_param.has_value()) {
    return nullptr;
  }

  const NodeUnit* reshape_node_unit =
      GetReshapeNodeUnit(graph_viewer, node_to_node_unit, node_unit_to_qnn_node_group, gemm_node);
  if (!reshape_node_unit) {
    return nullptr;
  }

  if (!CheckShape(reshape_node_unit->GetNode())) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusion>(*reshape_node_unit, gemm_node_unit);
}

ReshapeGemmFusion::ReshapeGemmFusion(const NodeUnit& reshape_node_unit, const NodeUnit& gemm_node_unit)
    : node_units_{} {
  node_units_[0] = &reshape_node_unit;
  node_units_[1] = &gemm_node_unit;
}

Status ReshapeGemmFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& /*logger*/) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], true);
}

Status ReshapeGemmFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& /*logger*/) const {
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], false);
}

gsl::span<const NodeUnit* const> ReshapeGemmFusion::GetNodeUnits() const {
  return gsl::make_span<const NodeUnit* const>(node_units_.data(), 2);
}

const NodeUnit* ReshapeGemmFusion::GetTargetNodeUnit() const { return node_units_[1]; }

}  // namespace qnn
}  // namespace onnxruntime
