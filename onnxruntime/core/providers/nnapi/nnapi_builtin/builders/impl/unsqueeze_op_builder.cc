// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
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

class UnsqueezeOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

// Add operator related

void UnsqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // Unsqueeze opset 13 uses input 1 as axes, add it to initializer skip list
  if (node_unit.SinceVersion() > 12 && node_unit.Inputs().size() > 1) {
    model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());  // "axes"
  }
}

Status UnsqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& shaper(model_builder.GetShaper());
  const auto& input = node_unit.Inputs()[0].node_arg.Name();

  // NNAPI does not support unsqueeze, here we utilize unsqueeze's axes input to compute output shape
  // And add equivalent operation as ANEURALNETWORKS_RESHAPE to nnapi model
  std::vector<int32_t> axes;
  ORT_RETURN_IF_ERROR(GetAxesForSqueezeAndUnSqueeze(model_builder, node_unit, axes));

  Shape input_shape = shaper[input];
  auto input_dims = input_shape.size();
  std::vector<int32_t> shape;
  const auto size = SafeInt<uint32_t>(input_dims + axes.size());  // "output rank"
  shape.reserve(size);
  for (auto& axis : axes) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, size));
  }
  std::sort(axes.begin(), axes.end());
  std::copy(input_shape.cbegin(), input_shape.cend(), std::back_inserter(shape));
  for (size_t i = 0; i < axes.size(); i++) {
    auto iter = shape.cbegin() + axes[i];
    shape.insert(iter, SafeInt<int32_t>(1));
  }

  return AddReshapeOperator(model_builder, node_unit, input, shape);
}

// Operator support related

bool UnsqueezeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                           const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  // This limitation actually comes from Reshape op.
  // We are adding ANEURALNETWORKS_RESHAPE as an equivalent operation for Unsqueeze as it's not supported by nnapi
  const auto input_rank = input_shape.size();
  if (input_rank > 4 || input_rank == 0) {
    LOGS_DEFAULT(VERBOSE) << "Unsqueeze only supports 1-4d shape, input is "
                          << input_rank << "d shape";
    return false;
  }

  // Unsqueeze opset 13 uses input 1 as axes, if we have input 1 then it needs to be an initializer
  if (node_unit.SinceVersion() > 12 && inputs.size() > 1) {
    const auto& axes_name = inputs[1].node_arg.Name();
    if (!graph_viewer.GetConstantInitializer(axes_name)) {
      LOGS_DEFAULT(VERBOSE) << "Input axes of Unsqueeze must be a constant initializer";
      return false;
    }
  }

  return true;
}

void CreateUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<UnsqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
