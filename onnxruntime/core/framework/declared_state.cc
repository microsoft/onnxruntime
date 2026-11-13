// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/declared_state.h"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <utility>

#include "core/common/safeint.h"
#include "core/graph/graph_viewer.h"
#include "nlohmann/json.hpp"

namespace onnxruntime {
namespace {

using Json = nlohmann::json;

Status CheckFields(const Json& object, std::initializer_list<std::string_view> required,
                   std::initializer_list<std::string_view> optional = {}) {
  if (!object.is_object()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared state requires a JSON object.");
  }
  for (const auto field : required) {
    if (!object.contains(std::string(field))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Missing declared-state field: ", field);
    }
  }
  for (const auto& [key, value] : object.items()) {
    (void)value;
    if (std::find(required.begin(), required.end(), key) == required.end() &&
        std::find(optional.begin(), optional.end(), key) == optional.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown declared-state field: ", key);
    }
  }
  return Status::OK();
}

Status ReadInteger(const Json& value, std::string_view field, int64_t minimum, int64_t& result) {
  if (!value.is_number_integer() ||
      (value.is_number_unsigned() &&
       value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state ", field, " must be an int64 integer.");
  }
  const int64_t number = value.get<int64_t>();
  if (number < minimum) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state ", field, " must be at least ", minimum, ".");
  }
  result = number;
  return Status::OK();
}

Status ReadString(const Json& value, std::string_view field, std::string& result) {
  if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state ", field, " must be a nonempty string.");
  }
  result = value.get_ref<const std::string&>();
  return Status::OK();
}

Status ExpandName(std::string_view pattern, int64_t layer, std::string& name) {
  constexpr std::string_view token = "{layer}";
  const size_t position = pattern.find(token);
  name = pattern;
  if (position != std::string_view::npos) {
    name.replace(position, token.size(), std::to_string(layer));
  }
  if (name.empty() || name.find_first_of("{}") != std::string::npos) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid declared-state name template: ", pattern);
  }
  return Status::OK();
}

Status ValidateContract(const DeclaredStateContract& contract) {
  if (contract.version < 1 || contract.version > static_cast<int64_t>(kMaxDeclaredStateContractVersion)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unsupported declared-state version: ", contract.version);
  }
  if (contract.max_batch_size <= 0 || contract.max_sequence_length <= 0 || contract.max_live_instances <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state envelope bounds must be positive.");
  }
  InlinedHashSet<std::string> ids;
  InlinedHashSet<std::string> names;
  InlinedHashSet<std::string> tables;
  for (const auto& state : contract.states) {
    if (state.id.empty() || !ids.insert(state.id).second || state.layers.empty() || state.capacity <= 0 ||
        (state.window.has_value() && *state.window < 0) ||
        (state.alias != CapabilityInputOutputAlias::Disjoint && state.alias != CapabilityInputOutputAlias::Identical) ||
        (state.lifetime != DeclaredStateLifetime::Generator && state.lifetime != DeclaredStateLifetime::Session)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid or duplicate declared state: ", state.id);
    }
    if (state.paging &&
        (contract.version != 2 || state.paging->block_size <= 0 || state.paging->num_blocks <= 0 ||
         state.alias != CapabilityInputOutputAlias::Identical)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Paged state requires version 2, positive block bounds, and past_present alias: ", state.id);
    }
    InlinedHashSet<int64_t> layers;
    for (const int64_t layer : state.layers) {
      if (layer < 0 || !layers.insert(layer).second) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid or duplicate layer in declared state: ", state.id);
      }
      for (const auto* pattern : {&state.input, &state.output}) {
        std::string name;
        if (auto status = ExpandName(*pattern, layer, name); !status.IsOK()) {
          return status;
        }
        if (!names.insert(name).second) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Duplicate declared-state tensor: ", name);
        }
      }
      if (state.paging) {
        std::string name;
        if (auto status = ExpandName(state.paging->block_table, layer, name); !status.IsOK()) {
          return status;
        }
        tables.insert(std::move(name));
      }
    }
  }
  for (const auto& table : tables) {
    if (names.contains(table)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Block table cannot also be a cache tensor: ", table);
    }
  }
  return Status::OK();
}

int64_t IntegerAttribute(const Node& node, const std::string& name, int64_t fallback) {
  const auto found = node.GetAttributes().find(name);
  return found == node.GetAttributes().end() ? fallback : found->second.i();
}

std::string_view StringAttribute(const Node& node, const std::string& name, std::string_view fallback) {
  const auto found = node.GetAttributes().find(name);
  return found == node.GetAttributes().end() ? fallback : std::string_view(found->second.s());
}

int64_t BlocksPerSequence(int64_t capacity, int64_t block_size) {
  return capacity / block_size + (capacity % block_size != 0 ? 1 : 0);
}

Status TensorByteBound(gsl::span<const int64_t> shape, size_t element_bytes, std::string_view name, size_t& result) {
  size_t bytes = element_bytes;
  for (const int64_t dimension : shape) {
    size_t converted = 0;
    if (dimension <= 0 || !SafeCast(dimension, converted) || !SafeMultiply(bytes, converted, bytes)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state byte bound overflows size_t: ", name);
    }
  }
  result = bytes;
  return Status::OK();
}

struct StatePort {
  const Node* node;
  size_t input_index;
  size_t output_index;
};

bool HasValue(gsl::span<const NodeArg* const> values, const std::string& name) {
  return std::any_of(values.begin(), values.end(), [&](const NodeArg* value) { return value->Name() == name; });
}

Status ResolveElementType(const StatePort& port, ResolvedDeclaredStateTensor& state, size_t& element_bytes) {
  const auto* input = port.node->InputDefs()[port.input_index];
  const auto* output = port.node->OutputDefs()[port.output_index];
  const auto* input_type = input->TypeAsProto();
  const auto* output_type = output->TypeAsProto();
  if (!input_type || !output_type || !input_type->has_tensor_type() || !output_type->has_tensor_type() ||
      input_type->tensor_type().elem_type() != output_type->tensor_type().elem_type()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "State input/output tensor types must agree: ", state.input);
  }
  state.element_type = input_type->tensor_type().elem_type();
  switch (state.element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      element_bytes = 4;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      element_bytes = 2;
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 requires an unquantized floating-point cache.");
  }
  return Status::OK();
}

Status ResolveDenseTensor(const DeclaredStateContract& contract, const DeclaredStateSpec& spec, const StatePort& port,
                          ResolvedDeclaredStateTensor& state) {
  const auto* input = port.node->InputDefs()[port.input_index];
  const auto* output = port.node->OutputDefs()[port.output_index];
  size_t element_bytes = 0;
  if (auto status = ResolveElementType(port, state, element_bytes); !status.IsOK()) {
    return status;
  }
  const int64_t heads = IntegerAttribute(*port.node, "kv_num_heads", 0);
  const int64_t local_window = IntegerAttribute(*port.node, "local_window_size", -1);
  const int64_t sliding = IntegerAttribute(*port.node, "sliding_window_cache", 0);
  if (heads <= 0 || local_window < -1 || (sliding != 0 && sliding != 1)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid GQA state attributes: ", state.input);
  }
  state.window = std::max<int64_t>(local_window, 0);
  state.sliding_window_cache = sliding == 1;
  if ((spec.window.has_value() && *spec.window != state.window) ||
      (state.sliding_window_cache && (state.window == 0 || spec.capacity < state.window)) ||
      (!state.sliding_window_cache && spec.capacity < contract.max_sequence_length)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state capacity/window disagrees with GQA: ",
                           state.input);
  }

  int64_t head_size = 0;
  for (const auto* value : {input, output}) {
    const auto* shape = value->Shape();
    if (!shape || shape->dim_size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 dense state requires rank-four BNSH shapes: ", value->Name());
    }
    const auto& dims = shape->dim();
    if ((dims[0].has_dim_value() && (dims[0].dim_value() <= 0 || dims[0].dim_value() > contract.max_batch_size)) ||
        (dims[1].has_dim_value() && dims[1].dim_value() != heads) ||
        (dims[2].has_dim_value() &&
         (dims[2].dim_value() < 0 || dims[2].dim_value() > spec.capacity ||
          (spec.alias == CapabilityInputOutputAlias::Identical && dims[2].dim_value() != spec.capacity)))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state dimensions disagree with graph: ",
                             value->Name());
    }
    if (dims[3].has_dim_value()) {
      const int64_t current = dims[3].dim_value();
      if (current <= 0 || (head_size != 0 && current != head_size)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "State input/output head sizes must agree: ", state.input);
      }
      head_size = current;
    }
  }
  if (head_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 requires a graph-resolved cache head size: ", state.input);
  }
  state.max_shape = {contract.max_batch_size, heads, spec.capacity, head_size};
  if (auto status = TensorByteBound(state.max_shape, element_bytes, state.input, state.size_bound_bytes);
      !status.IsOK()) {
    return status;
  }
  state.alias = spec.alias;
  state.lifetime = spec.lifetime;
  return Status::OK();
}

Status ResolvePagedTensor(const DeclaredStateContract& contract, const DeclaredStateSpec& spec, const StatePort& port,
                          ResolvedDeclaredStateTensor& state) {
  const auto& node = *port.node;
  const auto& paging = *spec.paging;
  size_t element_bytes = 0;
  if (auto status = ResolveElementType(port, state, element_bytes); !status.IsOK()) {
    return status;
  }
  if (state.element_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      StringAttribute(node, "kv_cache_layout", "SEPARATE") != "SEPARATE" ||
      StringAttribute(node, "k_quant_type", "NONE") != "NONE" ||
      StringAttribute(node, "v_quant_type", "NONE") != "NONE") {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "K1 paged state requires SEPARATE, unquantized FP16/BF16 pools.");
  }
  const auto dtype = StringAttribute(node, port.input_index == 3 ? "k_cache_dtype" : "v_cache_dtype", "");
  const auto expected_dtype = state.element_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ? "float16" : "bfloat16";
  if (!dtype.empty() && dtype != expected_dtype) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Paged cache dtype disagrees with graph: ", state.input);
  }
  const int64_t heads = IntegerAttribute(node, "kv_num_heads", 0);
  const int64_t local_window = IntegerAttribute(node, "local_window_size", -1);
  state.window = std::max<int64_t>(local_window, 0);
  if (heads <= 0 || local_window < -1 || (spec.window && *spec.window != state.window) ||
      spec.capacity < contract.max_sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid paged capacity/window: ", state.input);
  }
  if (paging.num_blocks > std::numeric_limits<int32_t>::max() ||
      paging.block_size > std::numeric_limits<int32_t>::max() || spec.capacity > std::numeric_limits<int32_t>::max()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 paged block/sequence bounds must fit int32.");
  }

  size_t required_blocks = 0;
  size_t batch = 0;
  size_t instances = 0;
  size_t available_blocks = 0;
  if (!SafeCast(BlocksPerSequence(spec.capacity, paging.block_size), required_blocks) ||
      !SafeCast(contract.max_batch_size, batch) ||
      !SafeCast(spec.lifetime == DeclaredStateLifetime::Session ? contract.max_live_instances : 1, instances) ||
      !SafeCast(paging.num_blocks, available_blocks) || !SafeMultiply(required_blocks, batch, required_blocks) ||
      !SafeMultiply(required_blocks, instances, required_blocks) || available_blocks < required_blocks) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Paged pool does not cover the resident envelope without sharing credit: ", state.input);
  }

  int64_t head_size = 0;
  for (const auto* value : {node.InputDefs()[3], node.InputDefs()[4],
                            node.OutputDefs()[1], node.OutputDefs()[2]}) {
    const auto* shape = value->Shape();
    if (!shape || shape->dim_size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 requires rank-four paged pool shapes: ", value->Name());
    }
    const auto& dims = shape->dim();
    for (const auto& [index, expected] :
         {std::pair{0, paging.num_blocks}, std::pair{1, paging.block_size}, std::pair{2, heads}}) {
      if (dims[index].has_dim_value() && dims[index].dim_value() != expected) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Paged pool dimensions disagree with graph: ", value->Name());
      }
    }
    if (dims[3].has_dim_value()) {
      const int64_t current = dims[3].dim_value();
      if (current <= 0 || (head_size != 0 && current != head_size)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Paged K/V head sizes must agree: ", state.input);
      }
      head_size = current;
    }
  }
  if (head_size == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 requires a graph-resolved paged head size: ", state.input);
  }
  state.max_shape = {paging.num_blocks, paging.block_size, heads, head_size};
  if (auto status = TensorByteBound(state.max_shape, element_bytes, state.input, state.size_bound_bytes);
      !status.IsOK()) {
    return status;
  }
  state.alias = spec.alias;
  state.lifetime = spec.lifetime;
  return Status::OK();
}

Status ResolveBlockTable(const DeclaredStateContract& contract, const DeclaredStateSpec& spec, const StatePort& port,
                         const GraphViewer& graph, const ResolvedDeclaredStateTensor& state,
                         ResolvedDeclaredStateContract& result) {
  const auto& inputs = port.node->InputDefs();
  const auto& paging = *state.paging;
  if (inputs.size() <= 7 || !inputs[7]->Exists() || inputs[7]->Name() != paging.block_table ||
      !HasValue(graph.GetInputs(), paging.block_table) || graph.IsInitializedTensor(paging.block_table)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Block table must match PagedAttention graph input 7: ",
                           paging.block_table);
  }
  const auto* value = inputs[7];
  const auto* type = value->TypeAsProto();
  const auto* shape = value->Shape();
  if (!type || !type->has_tensor_type() ||
      type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
      !shape || shape->dim_size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 requires a rank-two int32 block table: ", paging.block_table);
  }
  ResolvedDeclaredStateBlockTable table;
  table.input = paging.block_table;
  table.block_size = paging.block_size;
  table.max_shape = {contract.max_batch_size, BlocksPerSequence(spec.capacity, paging.block_size)};
  const auto& dims = shape->dim();
  if ((dims[0].has_dim_value() && (dims[0].dim_value() <= 0 || dims[0].dim_value() > contract.max_batch_size)) ||
      (dims[1].has_dim_value() && dims[1].dim_value() != table.max_shape[1])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Block-table dimensions disagree with envelope: ", table.input);
  }
  if (auto status = TensorByteBound(table.max_shape, sizeof(int32_t), table.input, table.size_bound_bytes);
      !status.IsOK()) {
    return status;
  }
  auto existing = std::find_if(result.block_tables.begin(), result.block_tables.end(),
                               [&](const auto& item) { return item.input == table.input; });
  if (existing == result.block_tables.end()) {
    result.block_tables.push_back(std::move(table));
    existing = std::prev(result.block_tables.end());
  } else if (existing->block_size != table.block_size || existing->max_shape != table.max_shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Conflicting shared block-table geometry: ", table.input);
  }
  if (std::find(existing->consumer_node_indices.begin(), existing->consumer_node_indices.end(),
                state.consumer_node_index) == existing->consumer_node_indices.end()) {
    existing->consumer_node_indices.push_back(state.consumer_node_index);
  }
  return Status::OK();
}

Status MergeShape(CapabilityPlanningQuery& query, const std::string& name, const TensorShapeVector& dimensions) {
  const TensorShape shape(dimensions);
  const auto [existing, inserted] = query.max_input_shapes.emplace(name, shape);
  if (!inserted && existing->second != shape) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Conflicting declared-state input capacity: ", name);
  }
  return Status::OK();
}

Status MergeBound(CapabilityPlanningQuery& query, std::string_view name, int64_t lower, int64_t upper) {
  const auto found = std::find_if(query.workload_bounds.begin(), query.workload_bounds.end(),
                                  [&](const auto& bound) { return bound.name == name; });
  if (found != query.workload_bounds.end()) {
    if (found->lower_bound != lower || found->upper_bound != upper) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Conflicting declared-state workload bound: ", name);
    }
  } else {
    query.workload_bounds.push_back({std::string(name), lower, upper});
  }
  return Status::OK();
}

}  // namespace

Status ParseDeclaredStateContract(std::string_view text, DeclaredStateContract& contract) {
  bool duplicate_key = false;
  InlinedVector<InlinedHashSet<std::string>, 1> object_keys;
  const auto callback = [&](int, Json::parse_event_t event, Json& value) {
    if (event == Json::parse_event_t::object_start) {
      object_keys.emplace_back();
    } else if (event == Json::parse_event_t::key) {
      duplicate_key |= !object_keys.back().insert(value.get_ref<const std::string&>()).second;
    } else if (event == Json::parse_event_t::object_end) {
      object_keys.pop_back();
    }
    return true;
  };
  const auto json = Json::parse(text, callback, false);
  if (json.is_discarded() || duplicate_key) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid declared-state JSON or duplicate object key.");
  }
  if (auto status = CheckFields(json, {"version", "max_batch_size", "max_sequence_length", "max_live_instances", "states"});
      !status.IsOK()) {
    return status;
  }
  int64_t version = 0;
  if (auto status = ReadInteger(json["version"], "version", 1, version); !status.IsOK()) {
    return status;
  }
  if (version != 1 && version != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Unsupported declared-state version: ", version);
  }
  DeclaredStateContract parsed;
  parsed.version = version;
  for (const auto& [field, destination] :
       {std::pair{"max_batch_size", &parsed.max_batch_size},
        std::pair{"max_sequence_length", &parsed.max_sequence_length},
        std::pair{"max_live_instances", &parsed.max_live_instances}}) {
    if (auto status = ReadInteger(json[field], field, 1, *destination); !status.IsOK()) {
      return status;
    }
  }
  if (!json["states"].is_array()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state states must be an array.");
  }
  for (const auto& item : json["states"]) {
    if (auto status = CheckFields(item, {"id", "kind", "input", "output", "layers", "capacity", "alias", "lifetime", "location"},
                                  version == 1 ? std::initializer_list<std::string_view>{"window"}
                                               : std::initializer_list<std::string_view>{"window", "paging"});
        !status.IsOK()) {
      return status;
    }
    if (!item["kind"].is_string() || !item["location"].is_string()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state kind and location must be strings.");
    }
    if (item["kind"] != "kv_cache" || item["location"] != "follow_consumer") {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 supports kv_cache and follow_consumer only.");
    }
    DeclaredStateSpec spec;
    for (const auto& [field, destination] :
         {std::pair{"id", &spec.id}, std::pair{"input", &spec.input}, std::pair{"output", &spec.output}}) {
      if (auto status = ReadString(item[field], field, *destination); !status.IsOK()) {
        return status;
      }
    }
    if (auto status = ReadInteger(item["capacity"], "capacity", 1, spec.capacity); !status.IsOK()) {
      return status;
    }
    if (item.contains("window")) {
      int64_t window = 0;
      if (auto status = ReadInteger(item["window"], "window", 0, window); !status.IsOK()) {
        return status;
      }
      spec.window = window;
    }
    if (item["alias"] == "past_present") {
      spec.alias = CapabilityInputOutputAlias::Identical;
    } else if (item["alias"] == "none") {
      spec.alias = CapabilityInputOutputAlias::Disjoint;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state alias must be past_present or none.");
    }
    if (item["lifetime"] == "per_generator") {
      spec.lifetime = DeclaredStateLifetime::Generator;
    } else if (item["lifetime"] == "per_session") {
      spec.lifetime = DeclaredStateLifetime::Session;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid declared-state lifetime.");
    }
    if (!item["layers"].is_array()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared-state layers must be an array.");
    }
    for (const auto& value : item["layers"]) {
      int64_t layer = 0;
      if (auto status = ReadInteger(value, "layer", 0, layer); !status.IsOK()) {
        return status;
      }
      spec.layers.push_back(layer);
    }
    if (item.contains("paging")) {
      const auto& object = item["paging"];
      if (auto status = CheckFields(object, {"block_size", "num_blocks", "block_table"}); !status.IsOK()) {
        return status;
      }
      DeclaredStatePaging paging;
      if (auto status = ReadInteger(object["block_size"], "block_size", 1, paging.block_size); !status.IsOK()) {
        return status;
      }
      if (auto status = ReadInteger(object["num_blocks"], "num_blocks", 1, paging.num_blocks); !status.IsOK()) {
        return status;
      }
      if (auto status = ReadString(object["block_table"], "block_table", paging.block_table); !status.IsOK()) {
        return status;
      }
      spec.paging = std::move(paging);
    }
    parsed.states.push_back(std::move(spec));
  }
  if (auto status = ValidateContract(parsed); !status.IsOK()) {
    return status;
  }
  contract = std::move(parsed);
  return Status::OK();
}

Status ResolveDeclaredStateContract(const DeclaredStateContract& contract, const GraphViewer& graph,
                                    std::string_view graph_id, uint64_t graph_revision,
                                    ResolvedDeclaredStateContract& resolved) {
  if (auto status = ValidateContract(contract); !status.IsOK()) {
    return status;
  }
  if (graph_id.empty() || graph.GetFilterInfo() != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared state requires an identified, unfiltered graph.");
  }
  InlinedHashMap<std::string, size_t> consumers;
  InlinedHashMap<std::string, StatePort> ports;
  for (const auto& node : graph.Nodes()) {
    for (const auto* input : node.InputDefs()) {
      if (input->Exists()) {
        ++consumers[input->Name()];
      }
    }
    for (const auto* input : node.ImplicitInputDefs()) {
      ++consumers[input->Name()];
    }
    if (node.Domain() != kMSDomain ||
        (node.OpType() != "GroupQueryAttention" && node.OpType() != "PagedAttention")) {
      continue;
    }
    if (node.OpType() == "PagedAttention" &&
        (contract.version != 2 || StringAttribute(node, "kv_cache_layout", "SEPARATE") != "SEPARATE")) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 paged state requires version 2 and SEPARATE layout.");
    }
    if (node.OpType() == "PagedAttention" &&
        (node.InputDefs().size() <= 4 || !node.InputDefs()[3]->Exists() || !node.InputDefs()[4]->Exists() ||
         node.OutputDefs().size() <= 2 || !node.OutputDefs()[1]->Exists() || !node.OutputDefs()[2]->Exists())) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 requires both paged K/V input/output pairs.");
    }
    for (const size_t input_index : {3u, 4u}) {
      const size_t output_index = input_index - 2;
      const auto& inputs = node.InputDefs();
      const auto& outputs = node.OutputDefs();
      const bool has_input = inputs.size() > input_index && inputs[input_index]->Exists();
      const bool has_output = outputs.size() > output_index && outputs[output_index]->Exists();
      if (!has_input && !has_output) {
        continue;
      }
      if (!has_input || !has_output || !HasValue(graph.GetInputs(), inputs[input_index]->Name()) ||
          !HasValue(graph.GetOutputs(), outputs[output_index]->Name()) ||
          graph.IsInitializedTensor(inputs[input_index]->Name())) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                               "K1 requires cache input/output pairs at graph inputs/outputs.");
      }
      if (!ports.emplace(inputs[input_index]->Name(), StatePort{&node, input_index, output_index}).second) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "K1 requires one consumer per cache tensor.");
      }
    }
  }
  ResolvedDeclaredStateContract result;
  result.graph_id = graph_id;
  result.graph_revision = graph_revision;
  result.max_batch_size = contract.max_batch_size;
  result.max_sequence_length = contract.max_sequence_length;
  result.max_live_instances = contract.max_live_instances;
  InlinedHashSet<std::string> declared_inputs;
  InlinedHashMap<size_t, TensorShapeVector> paged_shapes;
  for (const auto& spec : contract.states) {
    for (const int64_t layer : spec.layers) {
      ResolvedDeclaredStateTensor state;
      state.id = spec.id;
      state.layer = layer;
      if (auto status = ExpandName(spec.input, layer, state.input); !status.IsOK()) {
        return status;
      }
      if (auto status = ExpandName(spec.output, layer, state.output); !status.IsOK()) {
        return status;
      }
      state.paging = spec.paging;
      if (state.paging) {
        if (auto status = ExpandName(spec.paging->block_table, layer, state.paging->block_table); !status.IsOK()) {
          return status;
        }
      }
      const auto found = ports.find(state.input);
      if (found == ports.end() ||
          found->second.node->OutputDefs()[found->second.output_index]->Name() != state.output) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared pair does not match attention wiring: ", state.input,
                               " -> ", state.output);
      }
      if (consumers[state.input] != 1) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "State tensor has multiple consumers: ", state.input);
      }
      state.consumer_node_index = found->second.node->Index();
      state.input_index = found->second.input_index;
      state.output_index = found->second.output_index;
      const bool paged_node = found->second.node->OpType() == "PagedAttention";
      if (paged_node != state.paging.has_value()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Paging declaration must match the attention operator.");
      }
      if (auto status = paged_node ? ResolvePagedTensor(contract, spec, found->second, state)
                                   : ResolveDenseTensor(contract, spec, found->second, state);
          !status.IsOK()) {
        return status;
      }
      if (paged_node) {
        const auto [shape, inserted] = paged_shapes.emplace(state.consumer_node_index, state.max_shape);
        if (!inserted && shape->second != state.max_shape) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Paged K/V declarations must have identical pool shapes.");
        }
        if (auto status = ResolveBlockTable(contract, spec, found->second, graph, state, result); !status.IsOK()) {
          return status;
        }
      }
      declared_inputs.insert(state.input);
      result.states.push_back(std::move(state));
    }
  }
  for (const auto& [name, port] : ports) {
    (void)port;
    if (!declared_inputs.count(name)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Undeclared attention state: ", name);
    }
  }
  for (const auto& table : result.block_tables) {
    if (consumers[table.input] != table.consumer_node_indices.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Block table has an unsupported consumer: ", table.input);
    }
  }
  resolved = std::move(result);
  return Status::OK();
}

Status ComputeDeclaredStateBackingBounds(const ResolvedDeclaredStateContract& contract,
                                         DeclaredStateBackingBounds& bounds) {
  size_t instances = 0;
  if (contract.graph_id.empty() || contract.max_batch_size <= 0 || contract.max_sequence_length <= 0 ||
      contract.max_live_instances <= 0 || !SafeCast(contract.max_live_instances, instances)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Backing bounds require a resolved positive envelope.");
  }
  DeclaredStateBackingBounds result;
  for (const auto& state : contract.states) {
    if (state.size_bound_bytes == 0 ||
        (state.alias != CapabilityInputOutputAlias::Identical && state.alias != CapabilityInputOutputAlias::Disjoint) ||
        (state.lifetime != DeclaredStateLifetime::Generator && state.lifetime != DeclaredStateLifetime::Session)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Backing bounds require resolved cache facts: ", state.input);
    }
    size_t bytes = state.size_bound_bytes;
    if (!SafeMultiply(bytes, state.alias == CapabilityInputOutputAlias::Identical ? size_t{1} : size_t{2}, bytes) ||
        !SafeMultiply(bytes, state.lifetime == DeclaredStateLifetime::Generator ? instances : size_t{1}, bytes) ||
        !SafeAdd(result.cache_bytes, bytes, result.cache_bytes)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared cache backing overflows size_t.");
    }
  }
  for (const auto& table : contract.block_tables) {
    size_t bytes = 0;
    if (table.size_bound_bytes == 0 || !SafeMultiply(table.size_bound_bytes, instances, bytes) ||
        !SafeAdd(result.block_table_bytes, bytes, result.block_table_bytes)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared block-table backing overflows size_t.");
    }
  }
  if (!SafeAdd(result.cache_bytes, result.block_table_bytes, result.total_bytes)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Declared state backing overflows size_t.");
  }
  bounds = result;
  return Status::OK();
}

Status ProjectDeclaredStateToPlanningQuery(const ResolvedDeclaredStateContract& contract,
                                           CapabilityPlanningQuery& query) {
  if (query.mode != CapabilityQueryMode::Planning || query.graph_id != contract.graph_id ||
      query.graph_revision != contract.graph_revision || contract.graph_id.empty() ||
      contract.max_batch_size <= 0 || contract.max_sequence_length <= 0 || contract.max_live_instances <= 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Declared-state projection requires a resolved envelope and matching graph/revision.");
  }
  auto projected = query;
  if (auto status = MergeBound(projected, "declared_state.batch_size", 1, contract.max_batch_size); !status.IsOK()) {
    return status;
  }
  if (auto status = MergeBound(projected, "declared_state.total_sequence_length", 0, contract.max_sequence_length);
      !status.IsOK()) {
    return status;
  }
  for (const auto& state : contract.states) {
    if (auto status = MergeShape(projected, state.input, state.max_shape); !status.IsOK()) {
      return status;
    }
    auto node = std::find_if(projected.node_constraints.begin(), projected.node_constraints.end(),
                             [&](const auto& constraint) { return constraint.node_index == state.consumer_node_index; });
    if (node == projected.node_constraints.end()) {
      CapabilityNodeConstraint constraint;
      constraint.node_index = state.consumer_node_index;
      projected.node_constraints.push_back(std::move(constraint));
      node = std::prev(projected.node_constraints.end());
    }
    auto& aliases = node->input_output_aliases;
    const auto alias = std::find_if(aliases.begin(), aliases.end(), [&](const auto& constraint) {
      return constraint.input_index == state.input_index && constraint.output_index == state.output_index;
    });
    if (alias != aliases.end()) {
      if (alias->alias == CapabilityInputOutputAlias::Unknown) {
        alias->alias = state.alias;
      } else if (alias->alias != state.alias) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Conflicting declared-state alias: ", state.input);
      }
    } else {
      aliases.push_back({state.input_index, state.output_index, state.alias});
    }
  }
  for (const auto& table : contract.block_tables) {
    if (auto status = MergeShape(projected, table.input, table.max_shape); !status.IsOK()) {
      return status;
    }
  }
  query = std::move(projected);
  return Status::OK();
}

}  // namespace onnxruntime
