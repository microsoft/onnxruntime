// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/node_attr_utils.h"

namespace onnxruntime {
namespace xnnpack {

bool IsPaddingTypeSupported(AutoPadType auto_pad) {
  return auto_pad == AutoPadType::NOTSET ||
         auto_pad == AutoPadType::VALID ||
         auto_pad == AutoPadType::SAME_UPPER;
}

// Fuse activation with node. Currently Conv and MaxPool are supported.
std::unique_ptr<IndexedSubGraph::MetaDef> FuseActivation(const Node& node, const Node& activation,
                                                         const GraphViewer& graph) {
  std::unique_ptr<IndexedSubGraph::MetaDef> metadef = std::make_unique<IndexedSubGraph::MetaDef>();
  IndexedSubGraph::MetaDef& def = *metadef;

  // we use the op type/domain to match the static xnnpack Conv or MaxPool kernel
  // registration
  def.name = node.OpType();
  def.domain = node.Domain();  // should always be kMSInternalNHWCDomain
  def.since_version = node.SinceVersion();

  // inputs
  const auto& inputs = node.InputDefs();
  def.inputs.reserve(inputs.size());
  std::for_each(inputs.cbegin(), inputs.cend(),
                [&def](const NodeArg* arg) {
                  // keep the number of inputs the same by inserting an empty string for a missing optional input
                  def.inputs.push_back(arg ? arg->Name() : "");
                });

  // outputs
  def.outputs.push_back(activation.OutputDefs()[0]->Name());

  // attributes
  // copy existing and add the activation info
  def.attributes = node.GetAttributes();

  // use infinity as the default as that's what xnnpack uses if min/max are not set
  float min = -INFINITY;
  float max = INFINITY;

  const auto& activation_type = activation.OpType();
  if (activation_type == "Clip") {
    min = std::numeric_limits<float>::min();
    max = std::numeric_limits<float>::max();
    bool min_max_are_attributes = activation.SinceVersion() == 1 || activation.SinceVersion() == 6;

    if (min_max_are_attributes) {
      ProtoHelperNodeContext nc(activation);
      OpNodeProtoHelper info(&nc);
      min = info.GetAttrOrDefault<float>("min", min);
      max = info.GetAttrOrDefault<float>("max", max);
    } else {
      const auto& clip_inputs = activation.InputDefs();
      const auto num_inputs = clip_inputs.size();

      const auto update_value = [&](size_t idx, float& value_to_set) {
        if (num_inputs > idx) {
          const NodeArg& arg = *clip_inputs[idx];
          if (arg.Exists()) {
            const auto& value = *graph.GetConstantInitializer(arg.Name(), true);
            // these should never be in external data as it makes no sense to put scalars there.
            ORT_ENFORCE(utils::HasExternalData(value) == false,
                        "External data is not supported for the scalar min/max Clip values");

            value_to_set = utils::HasRawData(value)
                               ? *reinterpret_cast<const float*>(value.raw_data().data())
                               : value.float_data()[0];
          }
        }
      };

      update_value(1, min);
      update_value(2, max);
    }
  } else if (activation_type == "Relu") {
    min = 0.f;
  } else {
    ORT_NOT_IMPLEMENTED("No support for fusion of ", node.OpType(), " with ", activation_type);
  }

  InlinedVector<float> activation_params{min, max};
  def.attributes.insert({"activation", utils::MakeAttribute("activation", activation_type)});
  def.attributes.insert({"activation_params", utils::MakeAttribute("activation_params", activation_params)});

  return metadef;
}
}  // namespace xnnpack
}  // namespace onnxruntime
