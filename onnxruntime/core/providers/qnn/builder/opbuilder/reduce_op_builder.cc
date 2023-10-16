// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <array>
#include <vector>

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/endian_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

enum ReduceOpType {
  REDUCE_OP_TYPE_MAX = 0,
  REDUCE_OP_TYPE_MIN,
  REDUCE_OP_TYPE_MEAN,
  REDUCE_OP_TYPE_PROD,
  REDUCE_OP_TYPE_SUM,

  REDUCE_OP_TYPE_COUNT,
  REDUCE_OP_TYPE_UNKNOWN,
};

ReduceOpType GetReduceOpType(const std::string& op_type) {
  if (op_type == "ReduceMax") {
    return REDUCE_OP_TYPE_MAX;
  } else if (op_type == "ReduceMin") {
    return REDUCE_OP_TYPE_MIN;
  } else if (op_type == "ReduceMean") {
    return REDUCE_OP_TYPE_MEAN;
  } else if (op_type == "ReduceProd") {
    return REDUCE_OP_TYPE_PROD;
  } else if (op_type == "ReduceSum") {
    return REDUCE_OP_TYPE_SUM;
  } else {
    return REDUCE_OP_TYPE_UNKNOWN;
  }
}

class ReduceOpBuilder : public BaseOpBuilder {
 public:
  ReduceOpBuilder() : BaseOpBuilder("ReduceOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ReduceOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  using AxesOnnxIntType = int64_t;
  using AxesQnnIntType = uint32_t;

  Status GetAxesSet(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                    InlinedHashSet<AxesOnnxIntType>& axes_set) const;

  // Maps an operator type to the opset in which "axes" became an input instead of an attribute.
  static const std::array<int, REDUCE_OP_TYPE_COUNT> opset_with_axes_as_input;
};

const std::array<int, REDUCE_OP_TYPE_COUNT> ReduceOpBuilder::opset_with_axes_as_input = {
    18,  // ReduceMax
    18,  // ReduceMin
    18,  // ReduceMean
    18,  // ReduceProd
    13   // ReduceSum
};

Status ReduceOpBuilder::GetAxesSet(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                   InlinedHashSet<AxesOnnxIntType>& axes_set) const {
  ReduceOpType reduce_op_type = GetReduceOpType(node_unit.OpType());
  if (reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_UNKNOWN) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unknown reduce operator ", node_unit.OpType());
  }

  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  const size_t input_rank = input_shape.size();

  std::vector<AxesOnnxIntType> reduce_axes;

  const int opset_axes_as_input = ReduceOpBuilder::opset_with_axes_as_input[reduce_op_type];
  const int opset = node_unit.SinceVersion();
  NodeAttrHelper node_helper(node_unit);

  // Extract the axes values from either the attribute or initializer input (depending on opset).
  if (opset < opset_axes_as_input) {  // Axes is in ONNX node attribute.
    reduce_axes = node_helper.Get(QNN_OP_REDUCE_MAX_PARAM_AXES, reduce_axes);
  } else if (inputs.size() > 1) {  // Axes is in ONNX input[1] initializer.
    const auto& axes_input = inputs[1];

    std::vector<uint32_t> axes_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(axes_input.node_arg, axes_shape),
                      "Cannot get shape of axes input");

    if (axes_shape.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "QNN EP: \"axes\" input must have shape [M] where 0 < M <= rank(input[0])");
    }

    bool noop_with_empty_axes = static_cast<bool>(node_helper.Get("noop_with_empty_axes", (int64_t)0));
    if (axes_shape[0] == 0 && noop_with_empty_axes) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "QNN EP: does not support NoOp for reduction operators with empty axes.");
    }

    // Empty axes means to use default axes (when noop_with_empty_axes is 0).
    if (axes_shape[0] > 0) {
      const std::string& axes_input_name = inputs[1].node_arg.Name();

      // Check that the axes input is an initializer.
      if (!qnn_model_wrapper.IsInitializerInput(axes_input_name)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: \"axes\" input for reduce operator must be an initializer");
      }

      // Get axes initializer bytes.
      const auto& axes_tensor = qnn_model_wrapper.GetInitializerTensors().at(axes_input_name);
      std::vector<uint8_t> axes_bytes;

      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*axes_tensor, axes_bytes));
      ORT_ENFORCE(input_rank * sizeof(AxesOnnxIntType) >= axes_bytes.size(),
                  "Expect QNN Reduce* operator to have at most rank(input[0]) axes elements.");
      reduce_axes.resize(axes_bytes.size() / sizeof(AxesOnnxIntType));

      auto src_span = gsl::make_span(axes_bytes.data(), axes_bytes.size());
      auto dst_span = gsl::make_span(reduce_axes.data(), reduce_axes.size());

      // Copy initializer bytes (stored in little-endian order) to vector of int64_t.
      // ReadLittleEndian returns a status error if the source and destination spans do not have
      // matching byte sizes.
      ORT_RETURN_IF_ERROR(onnxruntime::utils::ReadLittleEndian(src_span, dst_span));
    }
  }

  if (reduce_axes.size() == 0) {
    // Use default axes of (0, 1, 2, ..., input_rank - 1)
    for (size_t i = 0; i < input_rank; ++i) {
      axes_set.insert(static_cast<AxesOnnxIntType>(i));
    }
  } else {
    // QNN does not support negative axes values. Fix negative values by adding the input rank.
    for (auto ax : reduce_axes) {
      AxesOnnxIntType positive_axis = (ax < 0) ? (ax + static_cast<AxesOnnxIntType>(input_rank)) : ax;
      axes_set.insert(positive_axis);
    }
  }

  if (axes_set.size() > input_rank) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "QNN EP: \"axes\" input must have shape [M] where 0 < M <= rank(input[0])");
  }

  return Status::OK();
}

Status ReduceOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger) const {
  ReduceOpType reduce_op_type = GetReduceOpType(node_unit.OpType());
  if (reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_UNKNOWN) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unknown reduce operator ", node_unit.OpType());
  }

  bool is_npu_backend = IsNpuBackend(qnn_model_wrapper.GetQnnBackendType());
  if (reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_PROD && is_npu_backend) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: ReduceProd operator not supported by HTP backend.");
  }

  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status ReduceOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();

  // Only need to process input[0]. In newer opset versions, input[1] corresponds to the reduce axes,
  // which needs to be set as a QNN parameter.
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));

  return Status::OK();
}

Status ReduceOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  NodeAttrHelper node_attr_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  //
  // Handle axes param.
  //
  InlinedHashSet<AxesOnnxIntType> axes_set;
  ORT_RETURN_IF_ERROR(GetAxesSet(qnn_model_wrapper, node_unit, axes_set));
  const size_t num_axes = axes_set.size();

  // Truncate int64 ONNX axes values to QNN's required type (uint32_t).
  std::vector<AxesQnnIntType> axes_shape{SafeInt<AxesQnnIntType>(num_axes)};
  std::vector<AxesQnnIntType> axes_data;
  axes_data.resize(num_axes);
  std::transform(axes_set.begin(), axes_set.end(), axes_data.begin(),
                 [](AxesOnnxIntType item) { return SafeInt<AxesQnnIntType>(item); });

  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), QNN_OP_REDUCE_MAX_PARAM_AXES,
                             std::move(axes_shape), std::move(axes_data));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  //
  // Handle keepdims param.
  //
  auto onnx_keepdims = node_attr_helper.Get("keepdims", (int32_t)1);
  Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
  scalar_param.dataType = QNN_DATATYPE_BOOL_8;
  scalar_param.bool8Value = static_cast<uint8_t>(onnx_keepdims == 0 ? 0 : 1);
  QnnParamWrapper keep_dims_param(node_unit.Index(), node_unit.Name(), QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS, scalar_param);
  param_tensor_names.push_back(keep_dims_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(keep_dims_param));

  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));

  return Status::OK();
}

void CreateReduceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ReduceOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
