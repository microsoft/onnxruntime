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

#include "base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class UnsqueezeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<UnsqueezeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

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

}  // namespace nnapi
}  // namespace onnxruntime
