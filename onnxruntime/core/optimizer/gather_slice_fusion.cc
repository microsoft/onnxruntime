// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gather_slice_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

bool GatherSliceToSplitFusion::IsSupportedGather(const Graph& graph, const Node& node, int64_t& index,
                                                 int64_t& axis, int64_t& indices_n_dims) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gather", {1, 11, 13}) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
    return false;
  }

  const NodeArg& input_arg = *(node.InputDefs()[1]);

  if (!optimizer_utils::IsScalar(input_arg)) return false;

  const ONNX_NAMESPACE::TensorProto* indices_init = graph_utils::GetConstantInitializer(graph, input_arg.Name());

  if (!indices_init) return false;

  if (indices_init->data_type() != ONNX_NAMESPACE::TensorProto::INT64) return false;

  // get the index value
  Initializer init_const(*indices_init, graph.ModelPath());
  index = *(init_const.data<int64_t>());

  // get attributes value
  axis = 0;
  auto& attrs = node.GetAttributes();
  if (attrs.find("axis") != attrs.end()) {
    auto& axis_attr = attrs.at("axis");
    if (utils::HasInt(axis_attr)) axis = axis_attr.i();
  }

  indices_n_dims = indices_init->dims_size();
  return true;
}

bool GatherSliceToSplitFusion::IsSupportedSlice(const Graph& graph, const Node& node,
                                                InlinedVector<int64_t>& starts,
                                                InlinedVector<int64_t>& ends,
                                                InlinedVector<int64_t>& axes,
                                                InlinedVector<int64_t>& steps) const {
  // check the version of Slice ops
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {1, 10, 11, 13}) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
    return false;
  }

  // get the opset version
  int onnx_opset_version = -1;
  if (graph.DomainToVersionMap().find(kOnnxDomain) != graph.DomainToVersionMap().end()) {
    onnx_opset_version = graph.DomainToVersionMap().at(kOnnxDomain);
  }

  // If Slice op of opset version 1
  if (onnx_opset_version == 1) {
    if (!graph_utils::GetRepeatedNodeAttributeValues(node, "starts", starts) ||
        !graph_utils::GetRepeatedNodeAttributeValues(node, "ends", ends) ||
        starts.size() != ends.size()) {
      return false;
    }

    if (graph_utils::GetRepeatedNodeAttributeValues(node, "axes", axes) && (axes.size() != starts.size())) {
      return false;
    }
  }

  // If Slice op of opset version >= 10
  if (onnx_opset_version >= 10) {
    // node inputs include: starts - ends - axes - steps

    // return a pointer to the corresponding NodeArg if input of the node at the index exists
    auto get_input_if_exists = [&node](size_t input_index) -> const NodeArg* {
      const auto& input_defs = node.InputDefs();
      const NodeArg* input = (input_defs.size() > input_index) ? input_defs[input_index] : nullptr;
      return (input == nullptr || !input->Exists()) ? nullptr : input;
    };

    // return a pointer to the initializer if it is constant; otherwise, a nullptr
    auto get_initializer_if_constant =
        [&graph, get_input_if_exists](size_t input_index) -> const ONNX_NAMESPACE::TensorProto* {
      const NodeArg* input = get_input_if_exists(input_index);
      return input ? graph_utils::GetConstantInitializer(graph, input->Name()) : nullptr;
    };

    // return the initialization data if it is constant
    auto get_initializer_data =
        [&graph](const ONNX_NAMESPACE::TensorProto* slice_initializer) -> InlinedVector<int64_t> {
      Initializer init(*slice_initializer, graph.ModelPath());
      if (slice_initializer->data_type() == ONNX_NAMESPACE::TensorProto::INT32) {
        int32_t* init_data = init.data<int32_t>();
        return InlinedVector<int64_t>(init_data, init_data + init.size());
      }

      if (slice_initializer->data_type() == ONNX_NAMESPACE::TensorProto::INT64) {
        int64_t* init_data = init.data<int64_t>();
        return InlinedVector<int64_t>(init_data, init_data + init.size());
      }
      return {};
    };

    // starts and ends inputs have to exist, be constants and be of the same size.
    const ONNX_NAMESPACE::TensorProto* starts_init = get_initializer_if_constant(1);
    const ONNX_NAMESPACE::TensorProto* ends_init = get_initializer_if_constant(2);
    const ONNX_NAMESPACE::TensorProto* axes_init = get_initializer_if_constant(3);
    const ONNX_NAMESPACE::TensorProto* steps_init = get_initializer_if_constant(4);

    if (!starts_init || !ends_init || !axes_init || !steps_init) {
      return false;
    }

    starts = get_initializer_data(starts_init);
    ends = get_initializer_data(ends_init);
    axes = get_initializer_data(axes_init);
    steps = get_initializer_data(steps_init);

    if (starts.size() == 0 || ends.size() == 0 || starts.size() != ends.size()) {
      return false;
    }

    if (axes_init->dims_size() != 1 || static_cast<size_t>(axes_init->dims().Get(0)) != starts.size()) {
      return false;
    }

    // if steps exists, it should be constant and all value should be 1
    if (steps.size() != starts.size()) {
      return false;
    }

    for (int64_t step : steps) {
      if (step != 1) {
        return false;
      }
    }
  }

  return true;
}

/*
GatherToSplitFusion is to fuse:
    Node
        |-> Gather(index=0, axis=axis)
        |-> Gather(index=1, axis=axis)
        |-> Slice(index=2, axis=axis)
To
    Node
        |-> Split(index=0)
So that we can use one kernel to finish the job.
*/

Status GatherSliceToSplitFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                           const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);

  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  InlinedVector<const NodeArg*> output_args;

  // Iterate the topological order and get Reshape ops
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);

    if (p_node == nullptr) continue;

    Node& node = *p_node;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // Currently only catch after Reshape ops, optimize in the future
    if (node.OpType() != "Reshape") continue;

    size_t output_count = node.GetOutputEdgesCount();

    // We only catch 1 scenario for Multi Query Attention for now.
    //         |---> Gather
    // Reshape |---> Gather
    //         |---> Slice
    //         |... or (other ops)

    // Get the output into node args
    if (output_count < 3) continue;

    output_args.push_back(node.OutputDefs()[0]);
  }

  // iterate the children of Reshape node
  for (const NodeArg* node_arg : output_args) {
    auto shape = node_arg->Shape();
    if (!shape) continue;

    auto consumers = graph.GetConsumerNodes(node_arg->Name());
    size_t consumer_count = consumers.size();

    // get the tensor rank
    int64_t rank = static_cast<int64_t>(shape->dim_size());

    bool can_fuse = true;
    bool first_edge = true;
    int64_t split_axis = 0;
    int64_t indices_n_dims = -1;

    // Fuse 2 Gathers and 1 slice to Split
    // Get those outputs as Split outputs
    InlinedVector<NodeArg*> split_outputs(3);

    InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse;
    size_t gather_node_count = 0, slice_node_count = 0;

    // find the nodes to be merged
    for (auto consumer : consumers) {
      int64_t index, axis, dims;
      InlinedVector<int64_t> starts, ends, axes, steps;

      bool IsSupportedGatherOps = IsSupportedGather(graph, *consumer, index, axis, dims);
      bool IsSupportedSliceOps = IsSupportedSlice(graph, *consumer, starts, ends, axes, steps);

      if ((!consumer || consumer->InputDefs()[0] != node_arg) ||
          (!IsSupportedGatherOps && !IsSupportedSliceOps)) {
        break;
      }

      if (IsSupportedGatherOps) {
        if (indices_n_dims == -1) {
          indices_n_dims = dims;
        } else if (indices_n_dims != dims) {
          // Not the same number of dimensions (0 or 1) for all scalar indices.
          can_fuse = false;
          break;
        }

        if (axis < 0) axis += rank;

        if (first_edge) {
          auto dim = shape->dim(static_cast<int>(axis));
          // dim.dim_value() = 73
          if (!utils::HasDimValue(dim)) {
            can_fuse = false;
            break;
          }
          split_axis = axis;
          first_edge = false;
        } else if (axis != split_axis) {
          can_fuse = false;
          break;
        }

        if (index < 0) index += static_cast<int64_t>(consumer_count);
        if (index < 0 || index >= static_cast<int64_t>(consumer_count)) {
          can_fuse = false;
          break;
        }

        Node& gather_node = *graph.GetNode(consumer->Index());
        nodes_to_fuse.push_back(gather_node);
        NodeArg* gather_output_args = gather_node.MutableOutputDefs()[0];
        split_outputs[++gather_node_count] = gather_output_args;
      }

      // check the Slice Ops
      if (IsSupportedSliceOps) {
        if (axes[0] != axis && !first_edge) {
          can_fuse = false;
          break;
        }

        Node& slice_node = *graph.GetNode(consumer->Index());
        NodeArg* slice_output_args = slice_node.MutableOutputDefs()[0];
        nodes_to_fuse.push_back(slice_node);
        split_outputs[slice_node_count] = slice_output_args;
        slice_node_count++;
      }
    }

    // condition check
    if (!can_fuse || gather_node_count != 2 || slice_node_count != 1) continue;

    // generate the split node and merge the kernel
    ONNX_NAMESPACE::TypeProto split_output_type;
    const ONNX_NAMESPACE::TensorProto_DataType element_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
        node_arg->TypeAsProto()->tensor_type().elem_type());

    split_output_type.mutable_tensor_type()->set_elem_type(element_type);

    for (int64_t i = 0; i < rank; i++) {
      if (i == split_axis)
        split_output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1LL);
      else
        *(split_output_type.mutable_tensor_type()->mutable_shape()->add_dim()) = shape->dim(static_cast<int>(i));
    }

    InlinedVector<NodeArg*> split_output_types;

    for (size_t i = 0; i < consumer_count; ++i) {
      split_output_types.push_back(
          &graph.GetOrCreateNodeArg(
              graph.GenerateNodeArgName("fused_split_" + std::to_string(i)), &split_output_type));
    }

    // Generate the Split Node
    ONNX_NAMESPACE::TensorProto split_initializer_proto;
    split_initializer_proto.set_name(graph.GenerateNodeName("fused_Split"));
    split_initializer_proto.add_dims(static_cast<int64_t>(3));
    split_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

    auto dim_value = shape->dim(static_cast<int>(split_axis)).dim_value();
    int64_t slice_dim = static_cast<int64_t>(dim_value - gather_node_count);
    InlinedVector<int64_t> split_value{{slice_dim, 1, 1}};
    split_initializer_proto.set_raw_data(split_value.data(), split_value.size() * sizeof(int64_t));
    NodeArg* split_arg = &graph_utils::AddInitializer(graph, split_initializer_proto);

    Node& split_node =
        graph.AddNode(graph.GenerateNodeName("Split"), "Split", "Split for fused Gather-Slice fusion",
                      {graph.GetNodeArg(node_arg->Name()), split_arg}, split_outputs);

    split_node.AddAttribute("axis", split_axis);

    split_node.SetExecutionProviderType(nodes_to_fuse[0].get().GetExecutionProviderType());

    int onnx_opset_version = -1;
    if (graph.DomainToVersionMap().find(kOnnxDomain) != graph.DomainToVersionMap().end()) {
      onnx_opset_version = graph.DomainToVersionMap().at(kOnnxDomain);
    }

    if (onnx_opset_version >= 18) {
      split_node.AddAttribute("num_outputs", static_cast<int64_t>(consumer_count));
    }

    for (Node& node_to_fuse : nodes_to_fuse) {
      graph_utils::RemoveNodeOutputEdges(graph, node_to_fuse);
      graph.RemoveNode(node_to_fuse.Index());
    }
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
