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
  if (inputs.size() > 1 && inputs[1].node_arg.Exists()) {
    model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
  }
}

Status ReductionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& op_type(node_unit.OpType());
  const auto& inputs = node_unit.Inputs();
  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto& shaper(model_builder.GetShaper());
  const auto input_shape = shaper[input];
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

  const bool keepdims = helper.Get("keepdims", 1) != 0;
  const bool noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0) != 0;

  // Get axes for ReduceMean
  // Note: ONNX `ReduceMean` will reduce by default all dimensions if axes is not provided/provided as empty. However, NNAPI doesn't implement the behavior
  // to reduce all dimensions by default when 'axes' is empty/not provided. We will convert the case by providing an input with all axes for NNAPI here.
  // Notes from NNAPI doc:
  // https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a047fe95a35b27f45c05432b6ca18eb6c
  std::vector<int32_t> axes;
  if (node_unit.SinceVersion() >= 18) {
    if (inputs.size() > 1 && inputs[1].node_arg.Exists()) {
      // ReduceMean-18 uses the second optional input as axes
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& axes_tensor = *initializers.at(inputs[1].node_arg.Name());
      Initializer unpacked_tensor(axes_tensor);
      auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
      axes = OnnxAxesToNnapi(raw_axes, input_shape.size());
    }
  } else {
    // For ReduceMean-13 or earlier, retrieve axes from the attribute
    const auto axes_int64 = helper.Get("axes", std::vector<int64_t>{});
    axes = OnnxAxesToNnapi(axes_int64, input_shape.size());
  }

  if (axes.empty() && !noop_with_empty_axes) {
    // we provide an input with all axes for NNAPI here to simulate this default behavior to reduce all dimensions
    axes.resize(input_shape.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Add ReduceMean operation
  if (!axes.empty()) {
    InlinedVector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));  // data

    const auto axes_name = model_builder.GetUniqueName(node_unit.Name() + inputs[0].node_arg.Name() + "_axes");
    Shape axes_dimen = {static_cast<uint32_t>(axes.size())};
    const OperandType axes_operand_type(Type::TENSOR_INT32, axes_dimen);
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(axes_name, axes.data(), axes_operand_type));

    input_indices.push_back(operand_indices.at(axes_name));  // axes

    int32_t input_rank = static_cast<int32_t>(input_shape.size());

    // Make output dimensions
    InlinedVector<uint32_t> output_dimen;
    if (keepdims) {
      output_dimen.reserve(input_rank);
    } else {
      output_dimen.reserve(input_rank - axes.size());
    }

    for (int32_t i = 0; i < input_rank; i++) {
      if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
        output_dimen.push_back(input_shape[i]);
      } else {
        if (keepdims) {
          output_dimen.push_back(1);
        }
      }
    }

    // In case of a tensor has all 1's in dimension such as {1,1,1,1} and gets all reduced,
    // NNAPI requires the output shape to be {1}. (otherwise NNAPI will treat it as dynamic shape.)
    if (output_dimen.empty())
      output_dimen.push_back(1);

    shaper.AddShape(output, output_dimen);

    ADD_SCALAR_OPERAND(model_builder, input_indices, keepdims ? 1 : 0);

    const OperandType output_operand_type(operand_types.at(inputs[0].node_arg.Name()).type, output_dimen);
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                   {output}, {output_operand_type}));
  } else {
    // Note: If `axes` is still empty at this point, meaning it's ReduceMean-18 and attribute `noop_with_empty_axes`
    // specifies as 1. We treat this case as an Identity op in NNAPI EP.
    // However, we hit an issue while adding no-ops operation in NNAPI because it doesn't allow adding an operand both as
    // an input and output.
    // Currently, we return not supported in NNAPI EP when `noop_with_empty_axes` is true.

    // const OperandType output_operand_type(operand_types.at(input).type, input_shape);
    // model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
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

  NodeAttrHelper helper(node_unit);

  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "NNAPI reduction ops only support 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  if (op == "ReduceMean") {
    const bool noop_with_empty_axes = helper.Get("noop_with_empty_axes", 0) != 0;
    if (inputs.size() > 1 && inputs[1].node_arg.Exists()) {
      const auto& axes_name = inputs[1].node_arg.Name();
      if (!Contains(initializers, axes_name)) {
        LOGS_DEFAULT(VERBOSE) << "Axes of ReduceMean must be a constant initializer.";
        return false;
      }
    }
    // Note: For the case - ReduceMean 18+ with noop_with_empty_axes attribute set as 1,
    // currently we hit an issue in NNAPI where it does not allow adding an operand as both an input and output.
    // This issue may arise from handling no-ops like Identity and ReduceX with noop_with_empty_axes set.
    // TODO: Support the case when a more complete solution is available.
    if (node_unit.SinceVersion() >= 18 && noop_with_empty_axes) {
      LOGS_DEFAULT(VERBOSE)
          << "ReduceMean 18+ with noop_with_empty_axes attribute set as 1 is not supported for now.";
      return false;
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
