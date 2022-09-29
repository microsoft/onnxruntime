// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

// Supports limited data types.
InlinedVector<int32_t> supported_data_types{
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
    ONNX_NAMESPACE::TensorProto_DataType_DOUBLE,
    ONNX_NAMESPACE::TensorProto_DataType_INT32,
    ONNX_NAMESPACE::TensorProto_DataType_INT64,
};

bool IsSupportedDataType(int32_t data_type) {
  if (std::find(supported_data_types.begin(), supported_data_types.end(), data_type) == supported_data_types.end()) {
    return false;
  }

  return true;
}

bool IsSingleValueShape(const ONNX_NAMESPACE::TensorShapeProto* input_shape) {
  if (input_shape == nullptr) {
    return false;
  }

  size_t dim_size = static_cast<size_t>(input_shape->dim_size());
  if (dim_size == 0 ||
      (dim_size == 1 && utils::HasDimValue(input_shape->dim(0)) && input_shape->dim(0).dim_value() == 1)) {
    return true;
  }

  return false;
}

static constexpr char SHARED_INITIALIZER_PREFIX[] = "ortshared_";
bool IsSharedInitializer(std::string_view initializer_name) {
  return initializer_name.rfind(SHARED_INITIALIZER_PREFIX, 0) == 0;
}

// Replace all consumer nodes to use shared initializers.
void ReplaceInputsToUseSharedInitializer(Graph& graph,
                                         const InlinedHashMap<const Node*, InlinedVector<int>>&
                                             consumer_node_to_input_index_map,
                                         const NodeArg* origin_initializer_node_arg,
                                         NodeArg* shared_initializer_node_arg) {
  for (auto it = consumer_node_to_input_index_map.begin(), end = consumer_node_to_input_index_map.end();
       it != end; ++it) {
    Node* node = graph.GetNode(it->first->Index());
    // Iterate all input defs to replace those that are equal to origin_initializer_node_arg,
    // Then it would be safe to remove the consumer node.
    for (int input_index : it->second) {
      graph_utils::ReplaceNodeInput(*node, input_index, *shared_initializer_node_arg);
    }
    graph.RemoveConsumerNode(origin_initializer_node_arg->Name(), node);

    //  Add consumer ref count for shared scalar initializer.
    std::vector<const Node*> consumers = graph.GetConsumerNodes(shared_initializer_node_arg->Name());
    if (std::find(consumers.begin(), consumers.end(), node) == consumers.end()) {
      graph.AddConsumerNode(shared_initializer_node_arg->Name(), node);
    }
  }

  // Remove the initializer if no other consumer nodes.
  if (graph.GetConsumerNodes(origin_initializer_node_arg->Name()).size() == 0) {
    graph.RemoveInitializedTensor(origin_initializer_node_arg->Name());
  }
}

int32_t GenerateUniqueValueId() {
  static int32_t value_key_generator = 1;
  return value_key_generator++;
}

}  // namespace

Status ConstantSharing::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/,
                                  const logging::Logger& /*logger*/) const {
  // Accumulated map from type/value/rank to initializer:
  //   The key is a string representation of initializer's data type, value and rank.
  //   The value is newly created initializer NodeArg* to be shared.
  std::map<std::string, NodeArg*> pattern_key_to_shared_arg_map;
  const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  InlinedVector<std::string> original_initializer_names;
  original_initializer_names.reserve(initialized_tensor_set.size());
  for (const auto& entry : initialized_tensor_set) {
    // Ignore if the initializer already handled, or not a constant initializer.
    if (IsSharedInitializer(entry.first) || !graph_utils::IsConstantInitializer(graph, entry.first)) {
      continue;
    }

    if (excluded_initializers_.find(entry.first) != excluded_initializers_.end()) {
      continue;
    }

    original_initializer_names.push_back(entry.first);
  }

  // We avoid using the scalar value directly in pattern_key because the value for example INT_MAX can be super big
  // and it will be hard to read. Instead, we use a unique id for each scalar value, and a map from the value to
  // unique id.
  InlinedHashMap<int32_t, int32_t> int32_value_to_value_id_map;
  InlinedHashMap<int64_t, int32_t> int64_value_to_value_id_map;
  InlinedHashMap<float, int32_t> float_value_to_value_id_map;
  InlinedHashMap<float, int32_t> half_value_to_value_id_map;
  InlinedHashMap<double, int32_t> double_value_to_value_id_map;
  for (const auto& initializer_name : original_initializer_names) {
    NodeArg* origin_initializer_node_arg = graph.GetNodeArg(initializer_name);

    // Already handled, skip it.
    if (origin_initializer_node_arg == nullptr) {
      continue;
    }

    auto input_shape = origin_initializer_node_arg->Shape();
    if (input_shape == nullptr || !IsSingleValueShape(input_shape)) {
      continue;
    }

    // Ignore if not constant initializers.
    const ONNX_NAMESPACE::TensorProto* tensor_proto = graph.GetConstantInitializer(origin_initializer_node_arg->Name(),
                                                                                   true);

    int32_t data_type = tensor_proto->data_type();
    if (!IsSupportedDataType(data_type)) {
      continue;
    }

    std::vector<const Node*> consumers = graph.GetConsumerNodes(origin_initializer_node_arg->Name());
    InlinedHashMap<const Node*, InlinedVector<int>> consumer_node_to_input_index_map;
    // If usage is from subgraph, skip it now, can be extended to support if there is a need.
    bool found_subgraph_usage = false;
    for (const Node* const_node : consumers) {
      for (int i = 0; i < static_cast<int>(const_node->ImplicitInputDefs().size()); ++i) {
        if (const_node->ImplicitInputDefs()[i] == origin_initializer_node_arg) {
          found_subgraph_usage = true;
          break;
        }
      }

      if (found_subgraph_usage) {
        break;
      }

      // Iterate all input defs to replace those that are equal to origin_initializer_node_arg,
      // Then it would be safe to remove the consumer node aferwards.
      for (int i = 0; i < static_cast<int>(const_node->InputDefs().size()); ++i) {
        if (const_node->InputDefs()[i] == origin_initializer_node_arg) {
          consumer_node_to_input_index_map[const_node].push_back(i);
        }
      }
    }

    if (found_subgraph_usage || consumer_node_to_input_index_map.size() == 0) {
      continue;
    }

    onnxruntime::Initializer initializer{*tensor_proto, graph.ModelPath()};
    std::int32_t value_id;
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        float flt_val = *initializer.data<float>();
        if (float_value_to_value_id_map.find(flt_val) == float_value_to_value_id_map.end()) {
          float_value_to_value_id_map[flt_val] = GenerateUniqueValueId();
        }
        value_id = float_value_to_value_id_map[flt_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {
        float flt16_val = math::halfToFloat(initializer.data<MLFloat16>()->val);
        if (half_value_to_value_id_map.find(flt16_val) == half_value_to_value_id_map.end()) {
          half_value_to_value_id_map[flt16_val] = GenerateUniqueValueId();
        }
        value_id = half_value_to_value_id_map[flt16_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE: {
        double dbl_val = *initializer.data<double>();
        if (double_value_to_value_id_map.find(dbl_val) == double_value_to_value_id_map.end()) {
          double_value_to_value_id_map[dbl_val] = GenerateUniqueValueId();
        }
        value_id = double_value_to_value_id_map[dbl_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
        int32_t int32_val = *initializer.data<int32_t>();
        if (int32_value_to_value_id_map.find(int32_val) == int32_value_to_value_id_map.end()) {
          int32_value_to_value_id_map[int32_val] = GenerateUniqueValueId();
        }
        value_id = int32_value_to_value_id_map[int32_val];
      } break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
        int64_t int64_val = *initializer.data<int64_t>();
        if (int64_value_to_value_id_map.find(int64_val) == int64_value_to_value_id_map.end()) {
          int64_value_to_value_id_map[int64_val] = GenerateUniqueValueId();
        }
        value_id = int64_value_to_value_id_map[int64_val];
      } break;
      default:
        ORT_THROW("Unsupported data type [",
                  std::to_string(data_type) + "] found: for initializer: " + initializer_name);
    }

    // Construct a string by data type, value, and rank. Used as a key in pattern_key_to_shared_arg_map.
    const std::string& pattern_key = MakeString(SHARED_INITIALIZER_PREFIX, value_id, "_", data_type, "_",
                                                origin_initializer_node_arg->Shape()->dim_size());

    // If there is no such existing scalar pattern, add a new one.
    if (pattern_key_to_shared_arg_map.find(pattern_key) == pattern_key_to_shared_arg_map.end()) {
      // Do a copy and rename the TensorProto.
      ONNX_NAMESPACE::TensorProto constant_tensor_proto_as_replacement(*tensor_proto);
      constant_tensor_proto_as_replacement.set_name(graph.GenerateNodeArgName(pattern_key));
      NodeArg& shared_scalar_initializer_node_arg = graph_utils::AddInitializer(graph,
                                                                                constant_tensor_proto_as_replacement);
      pattern_key_to_shared_arg_map[pattern_key] = &shared_scalar_initializer_node_arg;
    }

    ReplaceInputsToUseSharedInitializer(graph, consumer_node_to_input_index_map, origin_initializer_node_arg,
                                        pattern_key_to_shared_arg_map[pattern_key]);

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
