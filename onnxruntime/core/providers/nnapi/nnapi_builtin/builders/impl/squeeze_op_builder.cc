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

class SqueezeOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
};

// Add operator related

void SqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (node_unit.SinceVersion() > 12 && node_unit.Inputs().size() > 1) {
    model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());
  }
}

Status SqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto input = node_unit.Inputs()[0].node_arg.Name();

  std::vector<int32_t> axes;
  ORT_RETURN_IF_ERROR(GetAxesForSqueezeAndUnSqueeze(model_builder, node_unit, axes));
  return AddSqueezeOp(model_builder, node_unit.Name(), input, node_unit.Outputs()[0].node_arg.Name(), axes);
}

// Operator support related

bool SqueezeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                         const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  const auto input_rank = input_shape.size();
  if (input_rank > 4 || input_rank == 0) {
    LOGS_DEFAULT(VERBOSE) << "Squeeze only supports 1-4d shape, input is "
                          << input_rank << "d shape";
    return false;
  }

  // Squeeze opset 13 use input 1 as axes, if we have input 1 then it need to be an initializer
  if (node_unit.SinceVersion() > 12 && inputs.size() > 1) {
    const auto& axes_name = inputs[1].node_arg.Name();
    if (!graph_viewer.GetConstantInitializer(axes_name)) {
      LOGS_DEFAULT(VERBOSE) << "Input axes of Squeeze must be a constant initializer";
      return false;
    }
  }

  return true;
}

void CreateSqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
