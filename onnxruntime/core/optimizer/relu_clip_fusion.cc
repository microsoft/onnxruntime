// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/relu_clip_fusion.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/graph/op.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

Status FuseReluClip::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  const auto& next_node = *node.OutputNodesBegin();

  // Clip opset 6 has min and max as attributes. they're inputs from opset 11 on.
  bool min_max_are_attributes = graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6});

  // if Clip had a min < 0 we need to replace that value with 0 to do what Relu would have done using just Clip
  bool replace_min = false;
  ONNX_NAMESPACE::TensorProto replacement_min;

  if (min_max_are_attributes) {
    replace_min = graph_utils::GetNodeAttribute(next_node, "min")->f() < 0.f;
  } else {
    // we can fuse if the optional 'min' input is not provided, or if it is provided via a constant initializer
    const auto& clip_inputs = next_node.InputDefs();
    const NodeArg* min_input = (clip_inputs.size() > 1) ? clip_inputs[1] : nullptr;
    const ONNX_NAMESPACE::TensorProto* initializer = nullptr;

    if (min_input && min_input->Exists()) {
      initializer = graph_utils::GetConstantInitializer(graph, min_input->Name());
      if (!initializer) {
        // non-const initializer. can't proceed
        return Status::OK();
      }
    }

    int32_t data_type;

    if (!initializer) {
      // 'min' is using the default value of std::numeric_limits<>::lowest so we can fuse and provide a constant
      // value of '0' for 'min'

      // we need to know the correct data type to create a valid initializer for the value 0.
      // get that from 'input' as that must match the 'min' input type.
      const auto* input_type = next_node.InputDefs()[0]->TypeAsProto();
      if (input_type == nullptr || !input_type->tensor_type().has_elem_type()) {
        return Status::OK();
      }

      data_type = input_type->tensor_type().elem_type();
      replace_min = true;
    } else {
      // 'min' is provided by a constant initializer so we can fuse.
      // see if we need to replace with an initializer with a value of 0

      data_type = initializer->data_type();
      // construct an initializer to gracefully handle typed or raw data in the TensorProto
      Initializer i(*initializer, graph.ModelPath());
      switch (data_type) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          if (*i.data<float>() < 0.f) {
            replace_min = true;
          }
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
          if (math::halfToFloat(i.data<MLFloat16>()->val) < 0.f) {
            replace_min = true;
          }
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
          if (i.data<BFloat16>()->ToFloat() < 0.f) {
            replace_min = true;
          }
          break;
        default:
          ORT_THROW("Unexpected data type for Clip 'min' input of ", initializer->data_type());
      }
    }

    if (replace_min) {
      // create a new TensorProto with value of 0 and unique name in replacement_min
      auto new_name = graph.GenerateNodeArgName("FuseReluClip_" + node.Name() + "_min_zero_constant");
      Initializer(static_cast<ONNX_NAMESPACE::TensorProto::DataType>(data_type), new_name, {})
          .ToProto(replacement_min);
    }
  }

  // Remove the Relu node, and update the following Clip node if the 'min' is < 0.f, to set it to 0.f.
  // This essentially fuses the Relu and Clip. If the Clip 'min' is >= 0.f no change is required to the Clip node
  // as Relu would have set a lower min of 0.f.
  if (graph_utils::RemoveNode(graph, node)) {
    if (replace_min) {
      auto* mutable_next_node = graph.GetNode(next_node.Index());
      if (min_max_are_attributes) {
        mutable_next_node->ClearAttribute("min");
        mutable_next_node->AddAttribute("min", 0.f);
      } else {
        graph.AddInitializedTensor(replacement_min);
        auto& mutable_input_defs = mutable_next_node->MutableInputDefs();
        NodeArg* replacement_min_nodearg = graph.GetNodeArg(replacement_min.name());
        if (mutable_input_defs.size() == 1) {  // Clip node only has the required 'input' so add optional 'min' input
          mutable_input_defs.push_back(replacement_min_nodearg);
          mutable_next_node->MutableInputArgsCount().push_back(1);
        } else {
          mutable_input_defs[1] = graph.GetNodeArg(replacement_min.name());
        }
      }
    }

    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool FuseReluClip::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13, 14})) {
    return false;
  }

  if (node.GetOutputEdgesCount() != 1) {
    return false;
  }

  // If the Relu is followed by a Clip node the Relu is redundant and can be removed
  // as Clip will apply the minimum. If the Clip 'min' value is < 0 we need
  // to update it to 0 to apply what the Relu would have done. We do that in Apply.
  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6, 11, 12, 13}) ||
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  if (!graph_utils::CanRemoveNode(graph, node, logger)) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
