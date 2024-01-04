// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>
#include <algorithm>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/optimizer/initializer.h"
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

class SplitOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related

 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // Split opset 13- uses "split" as attribute. Currently it's not supported.
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 13; }

  // NNAPI Split is available since NNAPI feature level 3
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }
};

// Add operator related

void SplitOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& input_defs = node_unit.Inputs();

  if (input_defs.size() > 1 && input_defs[1].node_arg.Exists()) {  // optional second input "split"
    model_builder.AddInitializerToSkip(input_defs[1].node_arg.Name());
  }
}

Status SplitOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& input_name = node_unit.Inputs()[0].node_arg.Name();
  const auto& outputs = node_unit.Outputs();

  NodeAttrHelper helper(node_unit);
  const auto axis = helper.Get("axis", 0);

  int32_t num_outputs;
  if (node_unit.SinceVersion() >= 18) {
    num_outputs = SafeInt<int32_t>(*helper.GetInt("num_outputs"));
  } else {
    num_outputs = SafeInt<int32_t>(node_unit.Outputs().size());
  }

  std::vector<std::string> output_names;
  output_names.reserve(num_outputs);
  for (int32_t i = 0; i < num_outputs; ++i) {
    output_names.push_back(outputs[i].node_arg.Name());
  }

  ORT_RETURN_IF_ERROR(op_builder_helpers::AddNnapiSplit(model_builder, input_name, axis, output_names));

  return Status::OK();
}

// Operator support related

bool SplitOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                       const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto& input_defs = node_unit.Inputs();
  NodeAttrHelper helper(node_unit);
  const auto axis = helper.Get("axis", 0);

  const auto split_dims_at_axis = input_shape[SafeInt<uint32_t>(HandleNegativeAxis(axis, input_shape.size()))];
  if (input_defs.size() > 1 && input_defs[1].node_arg.Exists()) {
    // if optional input `split` is provided
    auto split_initializer_it = initializers.find(input_defs[1].node_arg.Name());
    if (split_initializer_it == initializers.end()) {
      LOGS_DEFAULT(VERBOSE) << "Optional input 'split' must be initializer if provided.";
      return false;
    }
    const auto& splits_tensor = *split_initializer_it->second;
    Initializer unpacked_tensor(splits_tensor);
    auto splits_span = unpacked_tensor.DataAsSpan<int64_t>();
    uint32_t sum_of_splits = std::accumulate(splits_span.begin(), splits_span.end(), SafeInt<uint32_t>(0));
    if (sum_of_splits != split_dims_at_axis) {
      LOGS_DEFAULT(VERBOSE) << "Sum of the 'split' input values must equal to the dim value at 'axis' specified. "
                            << "dim value at 'axis' specified: "
                            << split_dims_at_axis
                            << ", sum of 'split' input values: "
                            << sum_of_splits;
      return false;
    }

    auto it = std::adjacent_find(splits_span.begin(), splits_span.end(), [](const auto& a, const auto& b) {
      return a != b;
    });
    if (it != splits_span.end()) {
      LOGS_DEFAULT(VERBOSE) << "NNAPI only supports the case that number of splits evenly divides split axis size";
      return false;
    }
  } else {
    uint32_t num_outputs;
    if (node_unit.SinceVersion() >= 18) {
      auto num_outputs_attr = helper.GetInt("num_outputs");
      if (!num_outputs_attr.has_value()) {
        LOGS_DEFAULT(VERBOSE) << "No 'num_outputs' provided. For split 18+, num_outputs is a required attribute.";
        return false;
      }
      num_outputs = SafeInt<uint32_t>(*num_outputs_attr);
      if (num_outputs != SafeInt<uint32_t>(node_unit.Outputs().size()) || num_outputs > split_dims_at_axis) {
        LOGS_DEFAULT(VERBOSE) << "Invalid num_outputs provided. "
                              << "The value should be less than or equal to the size of dimension being split "
                              << "and align with the size of output nodes. Current num_outputs: "
                              << num_outputs;
        return false;
      }
    } else {
      num_outputs = SafeInt<uint32_t>(node_unit.Outputs().size());
    }
    // NNAPI only supports the case where axis can be evenly divided by num of splits
    if (split_dims_at_axis % num_outputs != 0) {
      LOGS_DEFAULT(VERBOSE) << "split count: " << num_outputs << " doesn't evenly divide split dimension: "
                            << split_dims_at_axis;
      return false;
    }
  }
  return true;
}

void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SplitOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
