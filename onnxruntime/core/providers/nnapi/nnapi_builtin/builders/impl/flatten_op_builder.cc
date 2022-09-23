// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class FlattenOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

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

}  // namespace nnapi
}  // namespace onnxruntime
