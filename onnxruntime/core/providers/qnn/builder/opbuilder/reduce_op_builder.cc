// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <array>
#include <vector>

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/endian_utils.h"
#include "core/framework/tensorprotoutils.h"
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
                       const logging::Logger& logger,
                       bool is_quantized_model) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status SetAxesParam(QnnModelWrapper& qnn_model_wrapper,
                      const NodeUnit& node_unit,
                      std::vector<std::string>& param_tensor_names) const;
  Status IsTypeAllowed(ONNX_NAMESPACE::DataType type, const std::vector<const char*>& allowed_types,
                       const char* err_prefix) const;

  // Maps an operator type to the opset in which "axes" became an input instead of an attribute.
  static const std::array<int, REDUCE_OP_TYPE_COUNT> opset_with_axes_as_input;
  static const std::array<std::vector<const char*>, REDUCE_OP_TYPE_COUNT> data_input_types;
  static const std::array<std::vector<const char*>, REDUCE_OP_TYPE_COUNT> data_input_quant_types;
};

const std::array<int, REDUCE_OP_TYPE_COUNT> ReduceOpBuilder::opset_with_axes_as_input = {
  18,  // ReduceMax
  18,  // ReduceMin
  18,  // ReduceMean
  18,  // ReduceProd
  13   // ReduceSum
};

const std::array<std::vector<const char*>, REDUCE_OP_TYPE_COUNT> ReduceOpBuilder::data_input_types = {
    std::vector<const char*>{"float"},  // ReduceMax
    std::vector<const char*>{"float"},  // ReduceMin
    std::vector<const char*>{"float"},  // ReduceMean
    std::vector<const char*>{"float"},  // ReduceProd
    std::vector<const char*>{"float", "int32"},  // ReduceSum
};

const std::array<std::vector<const char*>, REDUCE_OP_TYPE_COUNT> ReduceOpBuilder::data_input_quant_types = {
    std::vector<const char*>{"uint8", "int8"},  // ReduceMax
    std::vector<const char*>{"uint8", "int8"},  // ReduceMax
    std::vector<const char*>{"uint8", "int8"},  // ReduceMean
    std::vector<const char*>{"uint8", "int8"},  // ReduceProd
    std::vector<const char*>{"uint8", "int8", "int32"},  // ReduceSum
};

Status ReduceOpBuilder::IsTypeAllowed(ONNX_NAMESPACE::DataType type, const std::vector<const char*>& allowed_types,
                                      const char* err_prefix) const {
  for (auto allowed_type : allowed_types) {
    if (ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(allowed_type) == type) {
      return Status::OK();
    }
  }

  std::string err_msg = err_prefix;
  err_msg += " Type ";
  err_msg += *type;
  err_msg += " is not supported.";

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, err_msg);
}

Status ReduceOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      bool is_quantized_model) const {
  ORT_UNUSED_PARAMETER(logger);
  ReduceOpType reduce_op_type = GetReduceOpType(node_unit.OpType());
  if (reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_UNKNOWN) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unknown reduce operator ", node_unit.OpType());
  }
  
  const int opset_axes_as_input = ReduceOpBuilder::opset_with_axes_as_input[reduce_op_type];
  const int opset = node_unit.SinceVersion();

  NodeAttrHelper node_attr_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto& data_input = inputs[0];
  std::vector<uint32_t> input_data_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(data_input.node_arg, input_data_shape),
                    "Cannot get shape of data input");
  const size_t input_data_rank = input_data_shape.size();

  // Check allowed data input type.
  const std::vector<const char*>& allowed_data_types = is_quantized_model ? data_input_quant_types[reduce_op_type] : data_input_types[reduce_op_type];
  ONNX_NAMESPACE::DataType input_data_type = data_input.node_arg.Type();
  ORT_RETURN_IF_ERROR(IsTypeAllowed(input_data_type, allowed_data_types,
                                    "Invalid data input type for QNN reduce operator."));
  
  size_t num_axes = input_data_rank;

  if (opset >= opset_axes_as_input) {  // This Reduce* operator has an axes input.
    if (inputs.size() > 1) {
      const auto& axes_input = inputs[1];
      const auto& axes_input_name = axes_input.node_arg.Name();

      // Check that the axes input is an initializer.
      if (!qnn_model_wrapper.IsInitializerInput(axes_input_name)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: \"axes\" input for reduce operator must be an initializer");
      }

      // Check that the axes shape is 1D with shape [M] where 0 < M <= rank(input_data)
      std::vector<uint32_t> axes_shape;
      ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(axes_input.node_arg, axes_shape),
                        "Cannot get shape of axes input");

      if (axes_shape.size() != 1 || axes_shape[0] > input_data_rank) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "QNN EP: \"axes\" input must have shape [M] where 0 < M <= rank(input[0])");
      }

      bool noop_with_empty_axes = static_cast<bool>(node_attr_helper.Get("noop_with_empty_axes", (int64_t)0));
      if (axes_shape[0] == 0 && noop_with_empty_axes) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "QNN EP: does not support NoOp for reduction operators with empty axes.");
      }

      num_axes = (axes_shape[0] == 0) ? input_data_rank : axes_shape[0];
    }
  } else {
    std::vector<int64_t> reduce_axes(input_data_rank);
    reduce_axes = node_attr_helper.Get(qnn_def::axes, reduce_axes);
    num_axes = reduce_axes.size();
  }

  // TODO: Check that each axes value is within range [-input_data_rank, input_data_rank - 1],
  // and that each value is unique.

  // Check that the output data type matches the input data type.
  const auto& reduced_output = node_unit.Outputs()[0];
  ONNX_NAMESPACE::DataType output_data_type = reduced_output.node_arg.Type();
  if (output_data_type != input_data_type) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output data for QNN reduce operator must match the input data type");
  }

  const bool keep_dims = static_cast<bool>(node_attr_helper.Get("keepdims", (int32_t)1));
  std::vector<uint32_t> output_data_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(reduced_output.node_arg, output_data_shape),
                    "Cannot get shape of reduced output");
  const size_t output_data_rank = output_data_shape.size();

  // Output shape: K-dimensional, where K = rank(input[0]) if keep_dims is true,
  // or K = max(1, rank(input[0]) - num_axes) otherwise.
  if ((keep_dims && output_data_rank != input_data_rank) ||
      (!keep_dims && output_data_rank != std::max(static_cast<size_t>(1), input_data_rank - num_axes))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "QNN EP: reduce output must have K-dimensional shape, "
                           "where K = rank(input[0]) if keepdims is true, or "
                           "K = max(1, rank(input[0]) - num_axes) if keepdims is false");
  }

  return Status::OK();
}

Status ReduceOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      bool is_quantized_model,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();

  // Only need to process input[0]. In newer opset versions, input[1] corresponds to the reduce axes,
  // which needs to be set as a QNN parameter.
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, is_quantized_model, input_names));

  return Status::OK();
}

Status ReduceOpBuilder::SetAxesParam(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>& param_tensor_names) const {
  ReduceOpType reduce_op_type = GetReduceOpType(node_unit.OpType());
  if (reduce_op_type == ReduceOpType::REDUCE_OP_TYPE_UNKNOWN) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unknown reduce operator ", node_unit.OpType());
  }

  const auto& inputs = node_unit.Inputs();
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  const uint32_t rank = static_cast<uint32_t>(input_shape.size());

  // Create an array of default axes values.
  std::vector<int64_t> reduce_axes(rank);
  for (uint32_t i = 0; i < rank; ++i) {
    reduce_axes[i] = i;
  }

  const int opset_axes_as_input = ReduceOpBuilder::opset_with_axes_as_input[reduce_op_type];
  const int opset = node_unit.SinceVersion();

  // Extract the axes values from either the attribute or initializer input (depending on opset).
  if (opset < opset_axes_as_input) {  // Axes is in ONNX node attribute.
    NodeAttrHelper node_helper(node_unit);
    reduce_axes = node_helper.Get(qnn_def::axes, reduce_axes);
  } else if (inputs.size() > 1) {  // Axes is in ONNX input[1] initializer.
    const auto& axes_input = inputs[1];

    std::vector<uint32_t> axes_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(axes_input.node_arg, axes_shape),
                      "Cannot get shape of axes input");

    // Empty axes means to use default axes (when noop_with_empty_axes is 0).
    if (axes_shape[0] > 0) {
      const std::string& axes_input_name = inputs[1].node_arg.Name();
      ORT_ENFORCE(qnn_model_wrapper.IsInitializerInput(axes_input_name),
                  "Exepect QNN Reduce* operator's axes input to be an initializer.");

      // Get axes initializer bytes.
      const auto& axes_tensor = qnn_model_wrapper.GetInitializerTensors().at(axes_input_name);
      std::vector<uint8_t> axes_bytes;

      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*axes_tensor, axes_bytes));
      ORT_ENFORCE(reduce_axes.size() * sizeof(reduce_axes[0]) >= axes_bytes.size(),
                  "Expect QNN Reduce* operator to have at most rank(input[0]) axes elements.");
      reduce_axes.resize(axes_bytes.size() / sizeof(reduce_axes[0]));

      auto src_span = gsl::make_span(axes_bytes.data(), axes_bytes.size());
      auto dst_span = gsl::make_span(reduce_axes.data(), reduce_axes.size());

      // Copy initializer bytes (stored in little-endian order) to vector of int64_t.
      // ReadLittleEndian returns a status error if the source and destination spans do not have
      // matching byte sizes.
      ORT_RETURN_IF_ERROR(onnxruntime::utils::ReadLittleEndian(src_span, dst_span));
    }
  }

  // QNN does not support negative axes values.
  // Fix negative values by adding the input rank.
  size_t num_axes = reduce_axes.size();
  for (size_t i = 0; i < num_axes; ++i) {
    if (reduce_axes.at(i) < 0) {
      reduce_axes[i] += rank;
    }
  }

  // Truncate int64 ONNX axes values to QNN's required type (uint32_t).
  std::vector<uint32_t> axes_shape{SafeInt<uint32_t>(num_axes)};
  std::vector<uint32_t> axes_data;
  axes_data.resize(num_axes);
  std::transform(reduce_axes.begin(), reduce_axes.end(), axes_data.begin(),
                 [](int64_t item) { return SafeInt<uint32_t>(item); });

  // Create the QNN axes parameter.
  QnnParamWrapper axes_param(node_unit.Index(), node_unit.Name(), qnn_def::axes,
                             std::move(axes_shape), std::move(axes_data));
  param_tensor_names.push_back(axes_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(axes_param));

  return Status::OK();
}

Status ReduceOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool is_quantized_model,
                                                    bool do_op_validation) const {
  NodeAttrHelper node_attr_helper(node_unit);

  std::vector<std::string> param_tensor_names;

  // Handle axes
  ORT_RETURN_IF_ERROR(SetAxesParam(qnn_model_wrapper, node_unit, param_tensor_names));

  // Handle keepdims attribute.
  auto onnx_keepdims = node_attr_helper.Get("keepdims", (int32_t)1);
  Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
  scalar_param.dataType = QNN_DATATYPE_BOOL_8;
  scalar_param.bool8Value = static_cast<uint8_t>(onnx_keepdims == 0 ? 0 : 1);
  QnnParamWrapper keep_dims_param(node_unit.Index(), node_unit.Name(), qnn_def::keep_dims, scalar_param);
  param_tensor_names.push_back(keep_dims_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(keep_dims_param));


  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                     std::move(input_names),
                                     std::move(param_tensor_names),
                                     logger, is_quantized_model, do_op_validation));

  return Status::OK();
}

void CreateReduceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ReduceOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
