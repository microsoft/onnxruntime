// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_gemm_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <string>

#include "core/graph/graph_utils.h"
#include "core/framework/node_unit.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace qnn {

static const NodeUnit* GetReshapeNodeUnit(
    const GraphViewer& graph_viewer, const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Node& gemm_node) {
  if (gemm_node.OpType() != "Gemm") {
    return nullptr;
  }
  for (auto it = gemm_node.InputEdgesBegin(); it != gemm_node.InputEdgesEnd(); it++) {
    if (it->GetDstArgIndex() == 0) {
      const Node& reshape_node = it->GetNode();
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

static bool CheckShape(const GraphViewer& graph_viewer, const Node& reshape_node) {
  auto tensor_shape = reshape_node.InputDefs()[0]->Shape();
  if (!tensor_shape) return false;
  InlinedVector<int64_t> input_shape;
  for (const auto& dim : tensor_shape->dim()) {
    if (dim.value_case() != ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue) return false;
    input_shape.emplace_back(dim.dim_value());
  }

  const ONNX_NAMESPACE::TensorProto* shape_proto =
      graph_viewer.GetConstantInitializer(reshape_node.InputDefs()[1]->Name());
  if (!shape_proto) return false;
  const auto* dtype = DataTypeImpl::TensorTypeFromONNXEnum(shape_proto->data_type())->GetElementType();
  TensorShape shape = onnxruntime::utils::GetTensorShapeFromTensorProto(*shape_proto);
  Tensor tensor(dtype, shape, std::make_shared<CPUAllocator>());
  if (onnxruntime::utils::TensorProtoToTensor(onnxruntime::Env::Default(), graph_viewer.ModelPath(), *shape_proto,
                                              tensor) != Status::OK()) {
    return false;
  }

  InlinedVector<int64_t> output_shape;
  if (tensor.IsDataType<int64_t>()) {
    gsl::span<const int64_t> tensor_elems = tensor.DataAsSpan<int64_t>();
    output_shape.insert(output_shape.end(), tensor_elems.begin(), tensor_elems.end());
  } else if (tensor.IsDataType<int32_t>()) {
    gsl::span<const int32_t> tensor_elems = tensor.DataAsSpan<int32_t>();
    for (int32_t elem : tensor_elems) {
      output_shape.emplace_back(static_cast<int64_t>(elem));
    }
  }

  return !input_shape.empty() && output_shape.size() == 2 && input_shape.back() == output_shape.back();
}

#define ValidateOnQnn(qnn_model_wrapper, reshape_node_unit, gemm_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reshape_node_unit), (gemm_node_unit), true)
#define CreateOnQnn(qnn_model_wrapper, reshape_node_unit, gemm_node_unit) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (reshape_node_unit), (gemm_node_unit), false)
static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& reshape_node_unit,
                                    const NodeUnit& gemm_node_unit, bool validate) {
  assert(reshape_node_unit.OpType() == "Reshape" && gemm_node_unit.OpType() == "Gemm");
  const auto& node_name = utils::GetNodeName(gemm_node_unit);
  const NodeUnitIODef& input_def = reshape_node_unit.Inputs()[0];
  const NodeUnitIODef& weight_def = gemm_node_unit.Inputs()[1];
  const NodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = gemm_node_unit.Inputs().size() == 3;
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
  const auto& weight_tensor_proto = qnn_model_wrapper.GetInitializerTensors().at(weight_tensor_name);
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

std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper, const NodeUnit& gemm_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
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
  if (transA != 0 || transB != 0 || !qnn_model_wrapper.IsInitializerInput(weight_input.node_arg.Name()) ||
      weight_input.quant_param.has_value()) {
    return nullptr;
  }

  const NodeUnit* reshape_node_unit =
      GetReshapeNodeUnit(graph_viewer, node_to_node_unit, node_unit_to_qnn_node_group, gemm_node);
  if (!reshape_node_unit) {
    return nullptr;
  }

  if (!CheckShape(graph_viewer, reshape_node_unit->GetNode())) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusion>(*reshape_node_unit, gemm_node_unit);
}

ReshapeGemmFusion::ReshapeGemmFusion(const NodeUnit& reshape_node_unit, const NodeUnit& gemm_node_unit)
    : node_units_{} {
  node_units_[0] = &reshape_node_unit;
  node_units_[1] = &gemm_node_unit;
}

Status ReshapeGemmFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

Status ReshapeGemmFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1]);
}

gsl::span<const NodeUnit* const> ReshapeGemmFusion::GetNodeUnits() const {
  return gsl::make_span<const NodeUnit* const>(node_units_.data(), 2);
}

const NodeUnit* ReshapeGemmFusion::GetTargetNodeUnit() const { return node_units_[1]; }

}  // namespace qnn
}  // namespace onnxruntime
