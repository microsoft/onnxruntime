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

using namespace op_builder_helpers;

class ReductionOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                           const OpSupportCheckParams& params) const override;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

// Add operator related

void ReductionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (node_unit.SinceVersion() > 13 && inputs.size() > 1) {
    model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
  }
}

Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& op_type(node_unit.OpType());
  const auto& inputs = node_unit.Inputs();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto& shaper(model_builder.GetShaper());
  const auto input_shape = shaper[inputs[0].node_arg.Name()];
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  NodeAttrHelper helper(node_unit);

  int32_t op_code;
  if (op_type == "ReduceMean") {
    op_code = ANEURALNETWORKS_MEAN;
  } else {
    // TODO: Add more reduction ops support
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ReductionOpBuilder, unknown op: ", op_type);
  }

  if (op_type == "ReduceMean") {
    // Get axes for ReduceMean
    std::vector<int32_t> axes;
    if (inputs.size() > 1 && inputs[1].node_arg.Exists()) {
      // ReduceMean-18 uses the second optional input as axes
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& axes_tensor = *initializers.at(inputs[1].node_arg.Name());
      Initializer unpacked_tensor(axes_tensor);
      auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
      const auto size = SafeInt<uint32_t>(axes_tensor.dims()[0]);
      axes.resize(size);
      for (uint32_t i = 0; i < size; i++) {
        // it is unlikely we have an axis value overflow for int32
        axes[i] = static_cast<int32_t>(raw_axes[i]);
      }
    } else if (helper.HasAttr("axes")) {
      // For ReduceMean-13 or eariler, retrieve axes from the attribute
      const auto& axes_int64 = helper.Get("axes", std::vector<int64_t>{});
      axes.reserve(axes_int64.size());
      for (auto& axis : axes_int64) {
        axes.push_back(static_cast<int32_t>(axis));
      }
    }

    // Add ReduceMean operation
    const auto keepdims = helper.Get("keepdims", 1);

    InlinedVector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(inputs[0].node_arg.Name()));  // data

    if (!axes.empty()) {
      for (auto& axis : axes) {
        axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_shape.size()));
      }

      const auto axes_name = model_builder.GetUniqueName(node_unit.Name() + inputs[0].node_arg.Name() + "_axes");
      Shape axes_dimen = {static_cast<uint32_t>(axes.size())};
      const OperandType axes_operand_type(Type::TENSOR_INT32, axes_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(axes_name, axes.data(), axes_operand_type));

      input_indices.push_back(operand_indices.at(axes_name));  // axes

      int32_t input_size = static_cast<int32_t>(input_shape.size());
      std::unordered_set<int32_t> axes_to_be_reduced;

      for (const auto& axis : axes) {
        axes_to_be_reduced.insert(axis);
      }

      // Make output dimensions
      InlinedVector<uint32_t> output_dimen;
      if (keepdims == 1) {
        output_dimen.reserve(input_size);
      } else {
        output_dimen.reserve(input_size - axes_to_be_reduced.size());
      }

      for (int32_t i = 0; i < input_size; i++) {
        if (!Contains(axes_to_be_reduced, i)) {
          output_dimen.push_back(input_shape[i]);
        } else {
          if (keepdims == 1) {
            output_dimen.push_back(1);
          }
        }
      }

      // In case of a tensor has all 1's in dimension such as {1,1,1,1} and gets all reduced,
      // NNAPI requires the output shape to be {1}. (otherwise NNAPI will treat it as dynamic shape.)
      if (output_dimen.empty())
        output_dimen.push_back(1);

      shaper.AddShape(output, output_dimen);

      ADD_SCALAR_OPERAND(model_builder, input_indices, keepdims);

      const OperandType output_operand_type(operand_types.at(inputs[0].node_arg.Name()).type, output_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                     {output}, {output_operand_type}));
    } else {
      // For case that `axes` is empty, act as an Identity op
      const OperandType output_operand_type(operand_types.at(inputs[0].node_arg.Name()).type, input_shape);
      model_builder.RegisterOperand(output, operand_indices.at(inputs[0].node_arg.Name()), output_operand_type);
    }
  }

  return Status::OK();
}

// Operator support related

int32_t ReductionOpBuilder::GetMinSupportedNNAPIFeatureLevel(
    const NodeUnit& node_unit, const OpSupportCheckParams& /* params */) const {
  const auto& op(node_unit.OpType());
  if (op == "ReduceMean") {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  return ANEURALNETWORKS_FEATURE_LEVEL_3;
}

bool ReductionOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                           const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  const auto& op(node_unit.OpType());

  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "NNAPI reduction ops only support 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  NodeAttrHelper helper(node_unit);

  if (op == "ReduceMean") {
    // ONNX `ReduceMean` will reduce by default all dimensions if axes is not provided. However, NNAPI does support the behavior
    // to reduce all dimensions by default when 'axes' is empty/not provided.
    // Notes from NNAPI doc:
    // https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a047fe95a35b27f45c05432b6ca18eb6c
    if (node_unit.SinceVersion() < 18 && !helper.HasAttr("axes")) {
      // ONNX doc does not specify `axes` attribute to be optional, but we have unit test cases for ReduceMean 13- where axes attributes not provided
      // and treated as default behavior to reduce all.
      LOGS_DEFAULT(VERBOSE) << "For ReduceMean op version earlier than 18, `axes` is required to be provided as an attribute for NNAPI.";
      return false;
    }

    if (node_unit.SinceVersion() > 13) {
      if (inputs.size() > 1 && inputs[1].node_arg.Exists()) {
        const auto& axes_name = inputs[1].node_arg.Name();
        if (!Contains(initializers, axes_name)) {
          LOGS_DEFAULT(VERBOSE) << "Axes of ReduceMean must be a constant initializer.";
          return false;
        }

        // When `axes` is empty and `noop_with_empty_axes` equals 1, the case is supported and will be treated as an identity op.
        if (initializers.at(axes_name)->int64_data_size() == 0 && helper.Get("noop_with_empty_axes", 0) == 0) {
          LOGS_DEFAULT(VERBOSE) << "NNAPI ReduceMean doesn't support the behavior by default to reduce all dimensions when 'axes' is empty.";
          return false;
        }
      }

      if (inputs.size() < 2 && helper.Get("noop_with_empty_axes", 0) == 0) {
        LOGS_DEFAULT(VERBOSE) << "For ReduceMean-18, NNAPI does not support the default behavior to reduce all dimensions when `axes` is not provided and `noop_with_empty_axes` is false.";
      }
    }
  }
  return true;
}

void CreateReductionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<ReductionOpBuilder>(
      op_type, op_registrations,
      {
          // TODO: Add more reduction ops support
          "ReduceMean",
      });
}

}  // namespace nnapi
}  // namespace onnxruntime
