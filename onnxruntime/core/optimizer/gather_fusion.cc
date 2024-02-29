// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gather_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {
static int64_t GetGatherAxis(const Node& node, int64_t rank) {
  int64_t axis = 0;
  auto& attrs = node.GetAttributes();
  if (attrs.find("axis") != attrs.end()) {
    auto& axis_attr = attrs.at("axis");
    if (utils::HasInt(axis_attr)) {
      axis = axis_attr.i();
      if (axis < 0) axis += rank;
    }
  }
  return axis;
}

static bool GetScalarInt64Initializer(const Graph& graph, const NodeArg& node_arg, int64_t& value, int64_t& rank) {
  if (!optimizer_utils::IsScalar(node_arg)) return false;
  const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node_arg.Name());
  if (!tensor_proto || tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto::INT64) return false;
  Initializer init_const{*tensor_proto, graph.ModelPath()};
  value = *(init_const.data<int64_t>());
  rank = tensor_proto->dims_size();
  return true;
}

static bool GetSliceAxis(const Graph& graph, const Node& node, int64_t rank, int64_t& axis) {
  if (node.InputDefs().size() < 4) return false;
  int64_t unused = 0;
  if (!GetScalarInt64Initializer(graph, *node.InputDefs()[3], axis, unused)) return false;
  if (axis < 0) axis += rank;
  return true;
}

static bool GetAxis(const Graph& graph, const Node& node, int64_t rank, int64_t& axis) {
  if (node.OpType() == "Gather") {
    axis = GetGatherAxis(node, rank);
    return true;
  }
  if (node.OpType() == "Slice") {
    return GetSliceAxis(graph, node, rank, axis);
  }
  return false;
}

}  // namespace

bool GatherSliceToSplitFusion::IsSupportedGather(const Graph& graph, const Node& node, int64_t rank,
                                                 int64_t target_axis, int64_t dim_size, InlinedVector<bool>& consumed,
                                                 int64_t& start, bool& need_squeeze) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gather", {13}) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
    return false;
  }

  if (GetGatherAxis(node, rank) != target_axis) return false;
  // Require the indices input to be a scalar tensor for now. Normally if not, the exporter will choose Slice.
  // We can relax this later if needed.
  int64_t indices_n_dims = 0;
  if (!GetScalarInt64Initializer(graph, *(node.InputDefs()[1]), start, indices_n_dims)) return false;
  if (start < 0) start += dim_size;
  if (start < 0 || start >= dim_size || consumed[static_cast<size_t>(start)]) return false;
  consumed[static_cast<size_t>(start)] = true;
  need_squeeze = indices_n_dims == 0;
  return true;
}

bool GatherSliceToSplitFusion::IsSupportedSlice(const Graph& graph, const Node& node, int64_t rank, int64_t target_axis,
                                                int64_t dim_size, InlinedVector<bool>& consumed, int64_t& start,
                                                int64_t& end) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {13}) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
    return false;
  }

  int64_t axis = 0;
  if (!GetSliceAxis(graph, node, rank, axis) || axis != target_axis) return false;
  int64_t unused = 0;
  if (!GetScalarInt64Initializer(graph, *node.InputDefs()[1], start, unused) ||
      !GetScalarInt64Initializer(graph, *node.InputDefs()[2], end, unused)) {
    return false;
  }
  // Handling start and end according to schema definition.
  if (start < 0) start += dim_size;
  if (end < 0) end += dim_size;
  if (start < 0)
    start = 0;
  else if (start > dim_size)
    start = dim_size;
  if (end < 0)
    end = 0;
  else if (end > dim_size)
    end = dim_size;
  if (start >= end) return false;
  if (node.InputDefs().size() >= 5) {
    int64_t step = 0;
    if (!GetScalarInt64Initializer(graph, *node.InputDefs()[4], step, unused) || step != 1) return false;
  }
  for (int64_t i = start; i < end; ++i) {
    if (consumed[static_cast<size_t>(i)]) return false;
    consumed[static_cast<size_t>(i)] = true;
  }
  return true;
}

/*
GatherSliceToSplitFusion is to fuse:
Node -> Gather(indices=0, axis=axis)
    |-> Gather(indices=[1], axis=axis)
    |-> Slice(start=2, end=3, axes=[axis])
    |...

To

Node -> Split -> Squeeze(axis=axis)
             |->
             |->
             |...

So that we can use one kernel to finish the job.
The fusion requires that the indices of Gather nodes and start/end of Slice nodes are not overlapping and cover
all the elements in the target axis. Step of Slice node should be 1.
*/
Status GatherSliceToSplitFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                           const logging::Logger& logger) const {
  // Squeeze, Gather, Slice and Split have different schemas before and after OpSet 13.
  // To make code simple, support OpSet >= 13 only.
  int onnx_opset_version = -1;
  if (graph.DomainToVersionMap().find(kOnnxDomain) != graph.DomainToVersionMap().end()) {
    onnx_opset_version = graph.DomainToVersionMap().at(kOnnxDomain);
  }
  if (onnx_opset_version < 13) return Status::OK();

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  InlinedVector<const NodeArg*> candidate_args;
  for (auto node_arg : graph.GetInputs()) {
    if (node_arg && graph.GetConsumerNodes(node_arg->Name()).size() > 1) {
      candidate_args.push_back(node_arg);
    }
  }

  for (auto entry : graph.GetAllInitializedTensors()) {
    if (graph.GetConsumerNodes(entry.first).size() > 1) {
      auto node_arg = graph.GetNodeArg(entry.first);
      if (node_arg) {
        candidate_args.push_back(node_arg);
      }
    }
  }

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr) continue;  // we removed the node as part of an earlier fusion
    Node& node = *p_node;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // Gather following Shape is a common case but not the target case to fuse here as its compute is normally very
    // quick.
    if (node.OpType() == "Shape") continue;

    // Ideally it's possible that the node has multiple outputs and the Gather nodes can consume one of them.
    // To make the fusion simple, here we require the node has only one output, if in the future we observe
    // new pattern we can optimize the fusion here.
    if (node.MutableOutputDefs().size() > 1) continue;

    // No need to fuse if there is only one output or no output.
    size_t output_count = node.GetOutputEdgesCount();
    if (output_count <= 1) continue;

    candidate_args.push_back(node.OutputDefs()[0]);
  }

  for (const NodeArg* node_arg : candidate_args) {
    auto shape = node_arg->Shape();
    if (!shape) continue;
    int64_t rank = static_cast<int64_t>(shape->dim_size());
    auto consumers = graph.GetConsumerNodes(node_arg->Name());
    InlinedVector<const Node*> condidate_consumers;
    for (auto consumer : consumers) {
      if (consumer && consumer->InputDefs()[0] == node_arg &&
          (consumer->OpType() == "Gather" || consumer->OpType() == "Slice")) {
        condidate_consumers.emplace_back(consumer);
      }
    }
    if (condidate_consumers.size() < 2) continue;
    int64_t axis = 0;
    if (!GetAxis(graph, *condidate_consumers[0], rank, axis)) continue;
    auto dim = shape->dim(static_cast<int>(axis));
    if (!utils::HasDimValue(dim)) continue;
    int64_t dim_size = dim.dim_value();
    InlinedVector<bool> consumed(static_cast<size_t>(dim_size), false);
    bool can_fuse = true;
    InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse;
    InlinedVector<int64_t> starts;
    InlinedHashMap<int64_t, std::tuple<NodeArg*, int64_t, bool>> output_info_map;
    for (auto consumer : condidate_consumers) {
      if (!consumer || consumer->InputDefs()[0] != node_arg) {
        can_fuse = false;
        break;
      }
      int64_t start = 0, end = 0;
      bool need_squeeze = false;
      if (IsSupportedGather(graph, *consumer, rank, axis, dim_size, consumed, start, need_squeeze)) {
        Node& gather_node = *graph.GetNode(consumer->Index());
        nodes_to_fuse.emplace_back(gather_node);
        starts.emplace_back(start);
        output_info_map[start] = std::make_tuple(gather_node.MutableOutputDefs()[0], 1, need_squeeze);
      } else if (IsSupportedSlice(graph, *consumer, rank, axis, dim_size, consumed, start, end)) {
        Node& slice_node = *graph.GetNode(consumer->Index());
        nodes_to_fuse.emplace_back(slice_node);
        starts.emplace_back(start);
        output_info_map[start] = std::make_tuple(slice_node.MutableOutputDefs()[0], end - start, false);
      } else {
        can_fuse = false;
        break;
      }
    }

    if (!can_fuse || std::find(consumed.begin(), consumed.end(), false) != consumed.end()) continue;
    std::sort(starts.begin(), starts.end());
    InlinedVector<NodeArg*> split_outputs;
    InlinedVector<int64_t> split_values;
    for (int64_t start : starts) {
      auto& output_info = output_info_map[start];
      NodeArg* original_output_arg = std::get<0>(output_info);
      int64_t split_value = std::get<1>(output_info);
      split_values.emplace_back(split_value);
      if (std::get<2>(output_info)) {
        ONNX_NAMESPACE::TypeProto split_output_type;
        const ONNX_NAMESPACE::TensorProto_DataType element_type =
            static_cast<ONNX_NAMESPACE::TensorProto_DataType>(node_arg->TypeAsProto()->tensor_type().elem_type());
        split_output_type.mutable_tensor_type()->set_elem_type(element_type);
        for (int64_t i = 0; i < rank; ++i) {
          if (i == axis) {
            split_output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(split_value);
          } else {
            *(split_output_type.mutable_tensor_type()->mutable_shape()->add_dim()) = shape->dim(static_cast<int>(i));
          }
        }
        NodeArg* split_output_arg =
            &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("split_output"), &split_output_type);
        ONNX_NAMESPACE::TensorProto axes_initializer_proto;
        axes_initializer_proto.set_name(graph.GenerateNodeName("squeeze_axes"));
        axes_initializer_proto.add_dims(static_cast<int64_t>(1));
        axes_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        axes_initializer_proto.add_int64_data(axis);
        NodeArg* axes_arg = &graph_utils::AddInitializer(graph, axes_initializer_proto);
        Node& squeeze_node =
            graph.AddNode(graph.GenerateNodeName("Squeeze"), "Squeeze", "Squeeze for Fused Gather nodes",
                          {split_output_arg, axes_arg}, {original_output_arg});
        squeeze_node.SetExecutionProviderType(nodes_to_fuse[0].get().GetExecutionProviderType());
        split_outputs.emplace_back(split_output_arg);
      } else {
        split_outputs.emplace_back(original_output_arg);
      }
    }

    ONNX_NAMESPACE::TensorProto split_initializer_proto;
    split_initializer_proto.set_name(graph.GenerateNodeName("splits"));
    split_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    split_initializer_proto.add_dims(static_cast<int64_t>(split_values.size()));
    split_initializer_proto.mutable_int64_data()->Add(split_values.begin(), split_values.end());
    NodeArg* split_initializer_arg = &graph_utils::AddInitializer(graph, split_initializer_proto);
    Node& split_node = graph.AddNode(graph.GenerateNodeName("Split"), "Split", "Split for Fused Gather nodes",
                                     {graph.GetNodeArg(node_arg->Name()), split_initializer_arg}, split_outputs);
    split_node.AddAttribute("axis", axis);
    split_node.SetExecutionProviderType(nodes_to_fuse[0].get().GetExecutionProviderType());

    for (Node& node : nodes_to_fuse) {
      graph_utils::RemoveNodeOutputEdges(graph, node);
      graph.RemoveNode(node.Index());
    }

    modified = true;
  }

  return Status::OK();
}

/*
Fuse Range->Gather to Slice. Slice kernel is faster than Gather kernel in this case,
and SliceGrad is much faster than GatherGrad.
*/
Status GatherToSliceFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                      const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr) continue;  // we removed the node as part of an earlier fusion
    Node& node = *p_node;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Range", {1, 11}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) || node.GetOutputEdgesCount() != 1) {
      continue;
    }

    Node& gather_node = *graph.GetNode(node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(gather_node, "Gather", {1, 11, 13}) ||
        !graph_utils::IsSupportedProvider(gather_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Range's output is Gather's input[1].
    if (node.MutableOutputDefs()[0] != gather_node.MutableInputDefs()[1]) {
      continue;
    }

    InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse{node, gather_node};

    auto& range_input_defs = node.MutableInputDefs();
    ORT_ENFORCE(range_input_defs.size() == 3);
    // Range's inputs are scalar, need unsqueeze to 1-D tensors.
    ONNX_NAMESPACE::TypeProto unsqueeze_output_type;
    const ONNX_NAMESPACE::TensorProto_DataType element_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
        node.MutableOutputDefs()[0]->TypeAsProto()->tensor_type().elem_type());
    unsqueeze_output_type.mutable_tensor_type()->set_elem_type(element_type);
    unsqueeze_output_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1LL);

    InlinedVector<NodeArg*> unsqueeze_outputs;
    for (size_t i = 0; i < range_input_defs.size(); ++i) {
      unsqueeze_outputs.emplace_back(&graph.GetOrCreateNodeArg(
          graph.GenerateNodeArgName("unsqueeze_output_" + std::to_string(i)), &unsqueeze_output_type));
    }

    // Unsqueeze before and after OpSet-13 have different schemas.
    int onnx_opset_version = -1;
    if (graph.DomainToVersionMap().find(kOnnxDomain) != graph.DomainToVersionMap().end()) {
      onnx_opset_version = graph.DomainToVersionMap().at(kOnnxDomain);
    }

    if (onnx_opset_version < 13) {
      for (size_t i = 0; i < range_input_defs.size(); ++i) {
        Node& unsqueeze_node =
            graph.AddNode(graph.GenerateNodeName("Unsqueeze_" + std::to_string(i)), "Unsqueeze",
                          "Unsqueeze for Fused Gather nodes", {range_input_defs[i]}, {unsqueeze_outputs[i]});
        unsqueeze_node.AddAttribute("axes", std::vector<int64_t>{static_cast<int64_t>(0)});
        unsqueeze_node.SetExecutionProviderType(node.GetExecutionProviderType());
      }
    } else {
      ONNX_NAMESPACE::TensorProto unsqueeze_axes_initializer_proto;
      unsqueeze_axes_initializer_proto.set_name(graph.GenerateNodeName("UnsqueezeAxesInitializer"));
      unsqueeze_axes_initializer_proto.add_dims(static_cast<int64_t>(1));
      unsqueeze_axes_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
      unsqueeze_axes_initializer_proto.add_int64_data(static_cast<int64_t>(0));
      NodeArg* unsqueeze_axes_arg = &graph_utils::AddInitializer(graph, unsqueeze_axes_initializer_proto);

      for (size_t i = 0; i < range_input_defs.size(); ++i) {
        Node& unsqueeze_node = graph.AddNode(graph.GenerateNodeName("Unsqueeze_" + std::to_string(i)), "Unsqueeze",
                                             "Unsqueeze for Fused Gather nodes",
                                             {range_input_defs[i], unsqueeze_axes_arg}, {unsqueeze_outputs[i]});
        unsqueeze_node.SetExecutionProviderType(node.GetExecutionProviderType());
      }
    }

    int64_t axis = 0;  // Default value.
    auto& attrs = gather_node.GetAttributes();
    if (attrs.find("axis") != attrs.end()) {
      auto& axis_attr = attrs.at("axis");
      if (utils::HasInt(axis_attr)) axis = axis_attr.i();
    }

    ONNX_NAMESPACE::TensorProto slice_axes_initializer_proto;
    slice_axes_initializer_proto.set_name(graph.GenerateNodeName("SliceAxesInitializer"));
    slice_axes_initializer_proto.add_dims(static_cast<int64_t>(1));
    slice_axes_initializer_proto.set_data_type(element_type);
    // Tind of Slice can only support int32 and int64.
    if (element_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      slice_axes_initializer_proto.add_int64_data(axis);
    } else {
      slice_axes_initializer_proto.add_int32_data(static_cast<int32_t>(axis));
    }
    NodeArg* slice_axes_arg = &graph_utils::AddInitializer(graph, slice_axes_initializer_proto);
    Node& slice_node = graph.AddNode(graph.GenerateNodeName("Slice"), "Slice", "Slice for Fused Gather nodes",
                                     {gather_node.MutableInputDefs()[0], unsqueeze_outputs[0], unsqueeze_outputs[1],
                                      slice_axes_arg, unsqueeze_outputs[2]},
                                     {gather_node.MutableOutputDefs()[0]});
    slice_node.SetExecutionProviderType(gather_node.GetExecutionProviderType());

    for (Node& n : nodes_to_fuse) {
      graph_utils::RemoveNodeOutputEdges(graph, n);
      graph.RemoveNode(n.Index());
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
