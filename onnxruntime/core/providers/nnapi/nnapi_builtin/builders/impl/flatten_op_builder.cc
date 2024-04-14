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

namespace {
void GetFlattenOutputShape(const NodeUnit& node_unit, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2) {
  int32_t rank = static_cast<int>(input_shape.size());
  NodeAttrHelper helper(node_unit);
  int32_t axis = helper.Get("axis", 1);
  // axis == rank is a valid input, but invalid for HandleNegativeAxis
  // Skip non-negative axis here
  if (axis < 0)
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, rank));

  dim_1 = std::accumulate(input_shape.cbegin(), input_shape.cbegin() + axis, 1, std::multiplies<int32_t>());
  dim_2 = std::accumulate(input_shape.cbegin() + axis, input_shape.cend(), 1, std::multiplies<int32_t>());
}
}  // namespace

class FlattenOpBuilder : public BaseOpBuilder {
  // Add operator relate
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

// Add operator related

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto input = node_unit.Inputs()[0].node_arg.Name();

  // Flatten is basically a reshape to 2d tensor
  // Get the shape for Reshape here
  Shape input_shape;
  GetShape(node_unit.Inputs()[0].node_arg, input_shape);
  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenOutputShape(node_unit, input_shape, dim_1, dim_2);
  // If the input is of dynamic shape, replace 0 (dynamic) dimension with -1
  // We cannot have dim_1 and dim_2 both be 0 here, it was checked in IsOpSupportedImpl
  dim_1 = dim_1 == 0 ? -1 : dim_1;
  dim_2 = dim_2 == 0 ? -1 : dim_2;
  std::vector<int32_t> shape{dim_1, dim_2};
  return AddReshapeOperator(model_builder, node_unit, input, shape);
}

// Operator support related

bool FlattenOpBuilder::IsOpSupportedImpl(const GraphViewer& /* graph_viewer */, const NodeUnit& node_unit,
                                         const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Flatten only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenOutputShape(node_unit, input_shape, dim_1, dim_2);

  if (dim_1 == 0 && dim_2 == 0) {
    LOGS_DEFAULT(VERBOSE) << "The dynamic input shape " << Shape2String(input_shape)
                          << " is not supported";
    return false;
  }

  return true;
}

void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
