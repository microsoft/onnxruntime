// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// DO NOT include ORT header files as this is meant to be a header-only utility that can be copied
// to other projects.

/*
 SUMMARY:
   Utilities to serialize an OrtGraph into an ONNX GraphProto or ModelProto. Can be used by execution provider
   implementations that need to convert an OrtGraph instance into an ONNX protobuf model.

   Users may copy this file and modify as needed.

 USAGE:
   This is a header-only implementation that includes both the function declarations and definitions. Copy this file
   into a project that links with both ONNX Runtime and ONNX.

   Define the ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL preprocessor macro before the #include statement in exactly one C++
   file to define the implementation. Example:

     #define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
     #include "ort_graph_to_proto.h"

   Other compilation units that depend on these utilities should include this file without defining the
   preprocessor macro.

   Example program snippets are shown below. Refer to the function declarations for detailed usage information.

 EXAMPLE SNIPPET (initializers stored within TensorProto):

   ```C++
   #define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
   #include "ort_graph_to_proto.h"

   OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                             OrtEpGraphSupportInfo* graph_support_info) {
     onnx::GraphProto graph_proto;
     OrtEpUtils::OrtGraphToProto(*ort_graph, graph_proto);

     // graph_proto stores initializers internally
   }
   ```

 EXAMPLE SNIPPET (large initializers stored in external file):

   ```C++
   #define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
   #include "ort_graph_to_proto.h"

   OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                             OrtEpGraphSupportInfo* graph_support_info) {
     std::string external_file_path = "weights.bin";
     std::ofstream out_file(external_file_path, std::ios::binary);

     auto handle_initializer_data = [&external_file_path, &out_file](const OrtValueInfo* value_info,
                                                                     const void* data, size_t bytes,
                                                                     bool& is_external, std::string& location,
                                                                     int64_t& offset) -> Ort::Status {
       // OrtValueInfo* could be used to query initializer's name, type, shape, consumers, etc.
       (void)value_info;

       if (bytes <= 127) {
         is_external = false;  // Keep small initializers stored inside the TensorProto.
         return Ort::Status{nullptr};
       }

       offset = out_file.tellp();
       location = external_file_path;
       out_file.write(static_cast<const char*>(data), bytes);
       out_file.flush();
       is_external = true;  // True if is external initializer
       return Ort::Status{nullptr};
     }

     ONNX_NAMESPACE::GraphProto graph_proto;
     OrtEpUtils::OrtGraphToProto(*ort_graph, graph_proto, handle_initializer_data);

     // graph_proto stores large initializers in an external file
   }
   ```

 EXAMPLE SNIPPET (external initializers that point to data in memory, not officially supported by ONNX spec):

   This example stores initializers externally. However, instead of storing the initializers in a separate
   file, the onnx::TensorProto objects point directly to memory addresses. This requires setting the initializer's
   location to a special tag like "_MEM_ADDR_" (instead of a file path). The offset is set to the pointer to the
   initializer's data in memory (instead of an offset into a file).

   Because this is not standard ONNX, such a onnx::GraphProto should not be saved as an ONNX file.
   However, it allows custom tools that operate directly on a onnx::GraphProto to get the initializer data
   if it has already been loaded into memory.

   ```C++
   #define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
   #include "ort_graph_to_proto.h"

   OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                             OrtEpGraphSupportInfo* graph_support_info) {
     auto handle_initializer_data = [](const OrtValueInfo* value_info,
                                       const void* data, size_t bytes,
                                       bool& is_external, std::string& location,
                                       int64_t& offset) -> Ort::Status {
       (void)value_info;
       (void)bytes;

       offset = reinterpret_cast<int64_t>(data);
       location = "_MEM_ADDR_";  // Some special location tag that indicates the offset is a pointer.
       is_external = true;  // True if is external initializer
       return Ort::Status{nullptr};
     }

     ONNX_NAMESPACE::GraphProto graph_proto;
     OrtEpUtils::OrtGraphToProto(*ort_graph, graph_proto, handle_initializer_data);

     // graph_proto has initializers that look like they are stored in an external file,
     // but they are actually pointing to the data in memory.
   }
   ```
*/

#ifndef INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_ORT_GRAPH_TO_PROTO_H_
#define INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_ORT_GRAPH_TO_PROTO_H_

#include <functional>
#include "core/session/onnxruntime_cxx_api.h"
#include "onnx/onnx_pb.h"

namespace OrtEpUtils {

/// <summary>
/// Signature of user-provided function to handle initializer data. Called by OrtGraphToProto() for every initializer.
///
/// If the function sets the `is_external` output parameter to false, OrtGraphToProto() stores initializer data
/// within the TensorProto as raw_data.
///
/// Otherwise, if the function sets `is_external` to true, OrtGraphToProto() assumes that this function stores the
/// initializer data in a file. In this case, OrtGraphToProto() configures the corresponding TensorProto to point the
/// location and offset returned via the `location` and `offset` output parameters.
///
/// It is recommended to keep small initializers with byte size <= 127 stored inline the TensorProto to ensure
/// ONNX shape inference works correctly with the serialized ONNX model.
/// </summary>
/// <param name="value_info">OrtValueInfo for the initializer. Can be used to query name, type, shape,
///                           and consumer nodes.</param>
/// <param name="data">Opaque pointer to the initializer data.</param>
/// <param name="size">Size in bytes of the initializer data.</param>
/// <param name="is_external">Output parameter set to true if the initializer data is stored externally. The
///                           implementer is responsible for writing the initializer data to file. If set to false,
///                           the initializer will be stored within the TensorProto.</param>
/// <param name="location">Output parameter set to the location (e.g., file) into which the initializer is stored
///                        by the implementer of this function. Ignored if `is_external` is set to false.</param>
/// <param name="offset">Output parameter set to the offset (e.g., file offset) into which the initializer is stored
///                      by the implementer of this function. Ignored if `is_external` is set to false.</param>
/// <returns>An Ort::Status indicating success or an error. Serialization exits if this returns an error.</returns>
using HandleInitializerDataFunc = std::function<Ort::Status(const OrtValueInfo* value_info,
                                                            const void* data, size_t size,
                                                            /*out*/ bool& is_external, /*out*/ std::string& location,
                                                            /*out*/ int64_t& offset)>;

/// <summary>
/// Serializes the provided OrtGraph to a onnx::GraphProto.
/// Allows the caller to provide a function that specifies whether an initializer should be stored
/// within a TensorProto, written to a file, or remain as an in-memory external initializer (not valid ONNX).
/// </summary>
/// <param name="ort_graph">OrtGraph instance to serialize.</param>
/// <param name="graph_proto">Destination GraphProto into which to serialize the input OrtGraph.</param>
/// <param name="handle_initializer_data_func">Optional function called to allow the user to determine
///                                            where the initializer data is stored.</param>
/// <returns>An Ort::Status indicating success or an error.</returns>
Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::GraphProto& graph_proto,
                            HandleInitializerDataFunc handle_initializer_data_func = nullptr);

/// <summary>
/// Serializes the provided top-level OrtGraph to a onnx::ModelProto.
/// Allows the caller to provide a function that specifies whether an initializer should be stored
/// within a TensorProto, written to a file, or remain as an in-memory external initializer (not valid ONNX).
/// </summary>
/// <param name="ort_graph">OrtGraph instance to serialize.</param>
/// <param name="model_proto">Destination ModelProto into which to serialize the input OrtGraph.</param>
/// <param name="handle_initializer_data_func">Optional function called to allow the user to determine
///                                            where the initializer data is stored.</param>
/// <returns>An Ort::Status indicating success or an error.</returns>
Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::ModelProto& model_proto,
                            HandleInitializerDataFunc handle_initializer_data_func = nullptr);
}  // namespace OrtEpUtils

// End of header
#endif  // INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_ORT_GRAPH_TO_PROTO_H_

//
// IMPLEMENTATION BELOW
//
#ifdef ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <map>
#include <vector>

#define ORT_EP_UTILS_C_RETURN_IF_ERROR(fn) \
  do {                                     \
    OrtStatus* _status = (fn);             \
    if (_status != nullptr) {              \
      return Ort::Status{_status};         \
    }                                      \
  } while (0)

#define ORT_EP_UTILS_CXX_RETURN_IF_ERROR(fn) \
  do {                                       \
    Ort::Status _status = (fn);              \
    if (!_status.IsOK()) {                   \
      return _status;                        \
    }                                        \
  } while (0)

#define ORT_EP_UTILS_C_RETURN_IF(cond, ort_api, msg)               \
  do {                                                             \
    if ((cond)) {                                                  \
      return Ort::Status{(ort_api).CreateStatus(ORT_FAIL, (msg))}; \
    }                                                              \
  } while (0)

namespace OrtEpUtils {

static Ort::Status GetOrtValueInfoTensorTypeShape(const OrtValueInfo& ort_value_info,
                                                  bool get_symbolic_dims,
                                                  /*out*/ ONNXTensorElementDataType& elem_type,
                                                  /*out*/ std::vector<int64_t>& dims,
                                                  /*out*/ std::vector<std::string>& symbolic_dims);
static Ort::Status OrtValueInfoToProto(const OrtValueInfo& ort_value_info, onnx::ValueInfoProto& value_info_proto);
static Ort::Status OrtOpAttrToProto(const OrtNode& ort_node, const OrtOpAttr& ort_attr, onnx::AttributeProto& attr_proto);

Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::GraphProto& graph_proto,
                            HandleInitializerDataFunc handle_initializer_data_func) {
  const OrtApi& ort_api = Ort::GetApi();

  //
  // Set GraphProto metadata
  //
  const char* graph_name = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetName(&ort_graph, &graph_name));
  graph_proto.set_name(graph_name);
  graph_proto.set_doc_string("Serialized from OrtGraph");

  //
  // Set GraphProto inputs and outputs
  //
  size_t num_graph_inputs = 0;
  size_t num_graph_outputs = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetNumInputs(&ort_graph, &num_graph_inputs));
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetNumOutputs(&ort_graph, &num_graph_outputs));

  std::vector<const OrtValueInfo*> graph_inputs(num_graph_inputs);
  std::vector<const OrtValueInfo*> graph_outputs(num_graph_outputs);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetInputs(&ort_graph, graph_inputs.data(), graph_inputs.size()));
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetOutputs(&ort_graph, graph_outputs.data(), graph_outputs.size()));

  for (const OrtValueInfo* ort_value_info : graph_inputs) {
    onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_input()->Add();
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtValueInfoToProto(*ort_value_info, *value_info_proto));
  }

  for (const OrtValueInfo* ort_value_info : graph_outputs) {
    onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_output()->Add();
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtValueInfoToProto(*ort_value_info, *value_info_proto));
  }

  //
  // Set GraphProto nodes, value_infos, and initializers.
  //

  // Use std::maps to store OrtValueInfos for GraphProto.value_info and GraphProto.initializer.
  // A std::map maintains its elements in a stable ordering.
  std::map<std::string_view, const OrtValueInfo*> value_infos;              // For GraphProto.value_info
  std::map<std::string_view, const OrtValueInfo*> initializer_value_infos;  // For GraphProto.initializer

  // Helper function to collect an OrtValueInfo into `value_infos` or `initializer_value_infos`.
  // Optionally returns the OrtValueInfo name to the caller.
  auto collect_value_info = [&ort_api, &value_infos,
                             &initializer_value_infos](const OrtValueInfo& ort_value_info,
                                                       /*out*/ const char** value_name_out = nullptr) -> Ort::Status {
    const char* value_name = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetValueInfoName(&ort_value_info, &value_name));

    if (value_name_out != nullptr) {
      *value_name_out = value_name;
    }

    if (value_infos.count(value_name) != 0 || initializer_value_infos.count(value_name) != 0) {
      return Ort::Status{nullptr};  // Already processed this OrtValueInfo.
    }

    bool is_required_graph_input = false;
    bool is_optional_graph_input = false;
    bool is_graph_output = false;
    bool is_constant_initializer = false;
    bool is_from_outer_scope = false;

    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsRequiredGraphInput(&ort_value_info, &is_required_graph_input));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsOptionalGraphInput(&ort_value_info, &is_optional_graph_input));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsGraphOutput(&ort_value_info, &is_graph_output));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsConstantInitializer(&ort_value_info, &is_constant_initializer));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsFromOuterScope(&ort_value_info, &is_from_outer_scope));

    // Don't add graph inputs or graph outputs to GraphProto's list of value_infos.
    // Do add initializers (constant and non-constant) to GraphProto's list of initializer tensors.
    // For values defined in an outer scope, just add the value info but not the initializer.
    if (is_from_outer_scope) {
      value_infos.emplace(value_name, &ort_value_info);
    } else if (is_optional_graph_input) {
      initializer_value_infos.emplace(value_name, &ort_value_info);
    } else if (is_constant_initializer) {
      value_infos.emplace(value_name, &ort_value_info);
      initializer_value_infos.emplace(value_name, &ort_value_info);
    } else if (!is_required_graph_input && !is_graph_output) {
      value_infos.emplace(value_name, &ort_value_info);  // This is an internal OrtValueInfo.
    }

    return Ort::Status{nullptr};
  };

  size_t num_nodes = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(&ort_graph, &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetNodes(&ort_graph, nodes.data(), nodes.size()));

  // Loop through all nodes (topological order): add NodeProto instances to GraphProto and track OrtValueInfos
  // that will be stored in GraphProto.value_info and GraphProto.initializer.
  for (size_t i = 0; i < num_nodes; i++) {
    const OrtNode* ort_node = nodes[i];
    onnx::NodeProto* node_proto = graph_proto.add_node();

    const char* node_name = nullptr;
    const char* node_domain = nullptr;
    const char* node_op_type = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetName(ort_node, &node_name));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetDomain(ort_node, &node_domain));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(ort_node, &node_op_type));

    node_proto->set_name(node_name);
    node_proto->set_domain(node_domain);
    node_proto->set_op_type(node_op_type);

    size_t num_inputs = 0;
    size_t num_implicit_inputs = 0;
    size_t num_outputs = 0;
    size_t num_attrs = 0;
    size_t num_subgraphs = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumInputs(ort_node, &num_inputs));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumImplicitInputs(ort_node, &num_implicit_inputs));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(ort_node, &num_outputs));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumAttributes(ort_node, &num_attrs));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumSubgraphs(ort_node, &num_subgraphs));

    // Handle node attributes
    if (num_attrs > 0) {
      std::vector<const OrtOpAttr*> ort_attrs(num_attrs);
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetAttributes(ort_node, ort_attrs.data(), ort_attrs.size()));

      for (const OrtOpAttr* ort_attr : ort_attrs) {
        OrtOpAttrType attr_type = OrtOpAttrType::ORT_OP_ATTR_UNDEFINED;

        Ort::Status attr_type_status{ort_api.OpAttr_GetType(ort_attr, &attr_type)};
        if (attr_type == OrtOpAttrType::ORT_OP_ATTR_GRAPH) {
          // ORT does not support reading subgraphs via ReadOpAttr(), so skip it.
          // Can use Node_GetSubgraphs to get subgraphs.
          continue;
        }

        if (!attr_type_status.IsOK()) {
          // Unsupported attribute type.
          return attr_type_status;
        }

        onnx::AttributeProto* attr_proto = node_proto->add_attribute();
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtOpAttrToProto(*ort_node, *ort_attr, *attr_proto));
      }
    }

    // Handle node subgraphs
    if (num_subgraphs > 0) {
      std::vector<const OrtGraph*> ort_subgraphs(num_subgraphs);
      std::vector<const char*> subgraph_attr_names(num_subgraphs);
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetSubgraphs(ort_node, ort_subgraphs.data(), ort_subgraphs.size(),
                                                               subgraph_attr_names.data()));

      for (size_t subgraph_idx = 0; subgraph_idx < num_subgraphs; subgraph_idx++) {
        const OrtGraph* ort_subgraph = ort_subgraphs[subgraph_idx];
        const char* subgraph_attr_name = subgraph_attr_names[subgraph_idx];

        onnx::AttributeProto* attr_proto = node_proto->add_attribute();
        onnx::GraphProto* subgraph_proto = attr_proto->mutable_g();

        attr_proto->set_name(subgraph_attr_name);
        attr_proto->set_type(onnx::AttributeProto_AttributeType_GRAPH);
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtGraphToProto(*ort_subgraph, *subgraph_proto));
      }
    }

    // Handle node inputs
    if (num_inputs > 0) {
      std::vector<const OrtValueInfo*> ort_inputs(num_inputs);
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetInputs(ort_node, ort_inputs.data(), ort_inputs.size()));

      for (const OrtValueInfo* ort_value_info : ort_inputs) {
        if (ort_value_info == nullptr) {
          // missing optional input.
          node_proto->add_input("");
          continue;
        }

        const char* value_name = nullptr;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(collect_value_info(*ort_value_info, &value_name));

        node_proto->add_input(value_name);
      }
    }

    // Handle implicit inputs to this node.
    if (num_implicit_inputs > 0) {
      std::vector<const OrtValueInfo*> ort_implicit_inputs(num_implicit_inputs);
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetImplicitInputs(ort_node, ort_implicit_inputs.data(),
                                                                    ort_implicit_inputs.size()));

      for (const OrtValueInfo* ort_value_info : ort_implicit_inputs) {
        assert(ort_value_info != nullptr);
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(collect_value_info(*ort_value_info, /*value_name_out*/ nullptr));
      }
    }

    // Handle node outputs
    if (num_outputs > 0) {
      std::vector<const OrtValueInfo*> ort_outputs(num_outputs);
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOutputs(ort_node, ort_outputs.data(), ort_outputs.size()));

      for (const OrtValueInfo* ort_value_info : ort_outputs) {
        if (ort_value_info == nullptr) {
          // missing optional output.
          node_proto->add_output("");
          continue;
        }

        const char* value_name = nullptr;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(collect_value_info(*ort_value_info, &value_name));

        node_proto->add_output(value_name);
      }
    }
  }

  // Add value_infos to GraphProto as ValueInfoProto objects.
  for (const std::pair<const std::string_view, const OrtValueInfo*>& entry : value_infos) {
    onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_value_info()->Add();
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtValueInfoToProto(*entry.second, *value_info_proto));
  }

  // Add initializers to GraphProto as TensorProto objects.
  for (const std::pair<const std::string_view, const OrtValueInfo*>& entry : initializer_value_infos) {
    const OrtValueInfo* initializer_value_info = entry.second;
    std::string initializer_name = std::string{entry.first};  // Need a null-terminated string.
    std::vector<int64_t> initializer_dims;
    std::vector<std::string> initializer_sym_dims;
    ONNXTensorElementDataType initializer_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetOrtValueInfoTensorTypeShape(*initializer_value_info, /*get_sym_dims*/ false,
                                                                    initializer_elem_type, initializer_dims,
                                                                    initializer_sym_dims));

    onnx::TensorProto* tensor_proto = graph_proto.add_initializer();
    tensor_proto->set_name(initializer_name);
    tensor_proto->set_data_type(initializer_elem_type);

    auto* tensor_proto_dims = tensor_proto->mutable_dims();
    for (int64_t dim : initializer_dims) {
      tensor_proto_dims->Add(dim);
    }

    const OrtValue* ort_value = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetInitializerValue(initializer_value_info, &ort_value));

    const void* data = nullptr;
    size_t data_bytes = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorData(ort_value, &data));
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(ort_value, &data_bytes));

    std::string ext_location;
    int64_t ext_offset = 0;
    bool is_external = false;

    if (handle_initializer_data_func != nullptr) {
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(handle_initializer_data_func(initializer_value_info, data, data_bytes,
                                                                    is_external, ext_location, ext_offset));
    }

    if (is_external) {
      tensor_proto->set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
      auto* ext_data_entries = tensor_proto->mutable_external_data();
      onnx::StringStringEntryProto* location_entry = ext_data_entries->Add();
      onnx::StringStringEntryProto* offset_entry = ext_data_entries->Add();
      onnx::StringStringEntryProto* length_entry = ext_data_entries->Add();

      location_entry->set_key("location");
      location_entry->set_value(ext_location);
      offset_entry->set_key("offset");
      offset_entry->set_value(std::to_string(ext_offset));
      length_entry->set_key("length");
      length_entry->set_value(std::to_string(data_bytes));
    } else {
      // User wants to store data inline the TensorProto's raw_data
      tensor_proto->set_data_location(onnx::TensorProto_DataLocation_DEFAULT);
      tensor_proto->set_raw_data(data, data_bytes);
    }
  }

  return Ort::Status{nullptr};
}

Ort::Status OrtGraphToProto(const OrtGraph& ort_graph,
                            onnx::ModelProto& model_proto,
                            HandleInitializerDataFunc handle_initializer_data_func) {
  const OrtApi& ort_api = Ort::GetApi();

  // Check that OrtGraph is a top-level graph (no parent node).
  const OrtNode* parent_node = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetParentNode(&ort_graph, &parent_node));
  ORT_EP_UTILS_C_RETURN_IF(parent_node != nullptr, ort_api, "Cannot serialize nested OrtGraph into a ModelProto");

  // Set model description.
  model_proto.set_doc_string("Serialized from OrtGraph");
  model_proto.set_producer_name("ort_ep_utils::OrtGraphToProto");

  // Set ir version.
  int64_t ir_version = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetOnnxIRVersion(&ort_graph, &ir_version));
  model_proto.set_ir_version(ir_version);

  // Set operator sets.
  size_t num_operator_sets = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetNumOperatorSets(&ort_graph, &num_operator_sets));
  ORT_EP_UTILS_C_RETURN_IF(num_operator_sets == 0, ort_api, "OrtGraph should have at least one operator set.");

  std::vector<const char*> domains(num_operator_sets, nullptr);
  std::vector<int64_t> opset_versions(num_operator_sets);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetOperatorSets(&ort_graph, domains.data(), opset_versions.data(),
                                                               num_operator_sets));

  auto* operator_sets = model_proto.mutable_opset_import();

  for (size_t i = 0; i < num_operator_sets; ++i) {
    onnx::OperatorSetIdProto* operator_set = operator_sets->Add();
    operator_set->set_domain(domains[i]);
    operator_set->set_version(opset_versions[i]);
  }

  model_proto.clear_graph();
  onnx::GraphProto* graph_proto = model_proto.mutable_graph();

  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtGraphToProto(ort_graph, *graph_proto, handle_initializer_data_func));

  return Ort::Status{nullptr};
}

static Ort::Status GetOrtValueInfoTensorTypeShape(const OrtValueInfo& ort_value_info,
                                                  bool get_symbolic_dims,
                                                  /*out*/ ONNXTensorElementDataType& elem_type,
                                                  /*out*/ std::vector<int64_t>& dims,
                                                  /*out*/ std::vector<std::string>& symbolic_dims) {
  const OrtApi& ort_api = Ort::GetApi();

  const OrtTypeInfo* ort_type_info = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(&ort_value_info, &ort_type_info));

  ONNXType ort_onnx_type = ONNX_TYPE_UNKNOWN;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetOnnxTypeFromTypeInfo(ort_type_info, &ort_onnx_type));
  ORT_EP_UTILS_C_RETURN_IF(ort_onnx_type != ONNX_TYPE_TENSOR, ort_api, "Expected OrtValueInfo to represent a Tensor");

  const OrtTensorTypeAndShapeInfo* ort_type_shape = nullptr;
  ONNXTensorElementDataType ort_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.CastTypeInfoToTensorInfo(ort_type_info, &ort_type_shape));
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorElementType(ort_type_shape, &ort_elem_type));

  size_t num_dims = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetDimensionsCount(ort_type_shape, &num_dims));

  std::vector<int64_t> ort_dims(num_dims, 0);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetDimensions(ort_type_shape, ort_dims.data(), ort_dims.size()));

  elem_type = ort_elem_type;
  dims = std::move(ort_dims);

  if (get_symbolic_dims) {
    std::vector<const char*> ort_dim_syms(num_dims, nullptr);
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetSymbolicDimensions(ort_type_shape, ort_dim_syms.data(),
                                                                 ort_dim_syms.size()));

    symbolic_dims.reserve(num_dims);
    for (const char* sym_dim : ort_dim_syms) {
      symbolic_dims.push_back(sym_dim);
    }
  }

  return Ort::Status{nullptr};
}

// Create an onnx::ValueInfoProto from an OrtValueInfo (name, type, shape).
static Ort::Status OrtValueInfoToProto(const OrtValueInfo& ort_value_info,
                                       onnx::ValueInfoProto& value_info_proto) {
  const OrtApi& ort_api = Ort::GetApi();

  std::vector<int64_t> ort_dims;
  std::vector<std::string> ort_dim_syms;
  ONNXTensorElementDataType ort_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  // We currently only support ONNX tensors. Support for other types (e.g., ONNX_TYPE_SEQUENCE) can be added later.
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetOrtValueInfoTensorTypeShape(ort_value_info, /*get_sym_dims*/ true,
                                                                  ort_elem_type, ort_dims, ort_dim_syms));

  const char* value_name = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetValueInfoName(&ort_value_info, &value_name));
  value_info_proto.set_name(value_name);

  onnx::TypeProto_Tensor* type_proto_tensor = value_info_proto.mutable_type()->mutable_tensor_type();
  type_proto_tensor->set_elem_type(ort_elem_type);

  // If there are no dimensions in the shape, do not set a TensorShapeProto. Otherwise, it always looks
  // like a scalar value.
  if (!ort_dims.empty()) {
    onnx::TensorShapeProto* shape_proto = type_proto_tensor->mutable_shape();

    for (size_t dim_idx = 0; dim_idx < ort_dims.size(); dim_idx++) {
      onnx::TensorShapeProto_Dimension* dim_proto = shape_proto->add_dim();

      if (ort_dims[dim_idx] >= 0) {
        dim_proto->set_dim_value(ort_dims[dim_idx]);
      } else {
        const std::string& dim_param = ort_dim_syms[dim_idx];

        // If dim_param is empty, leave dim_proto with neither the dim_value or dim_param set,
        // which represents an unknown dimension.
        if (!dim_param.empty()) {
          dim_proto->set_dim_param(dim_param);
        }
      }
    }
  }

  return Ort::Status{nullptr};
}

static Ort::Status OrtOpAttrToProto(const OrtNode& ort_node, const OrtOpAttr& ort_attr, onnx::AttributeProto& attr_proto) {
  const OrtApi& ort_api = Ort::GetApi();

  const char* attr_name = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.OpAttr_GetName(&ort_attr, &attr_name));
  attr_proto.set_name(attr_name);

  size_t total_attr_bytes = 0;
  OrtOpAttrType attr_type = OrtOpAttrType::ORT_OP_ATTR_UNDEFINED;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.OpAttr_GetType(&ort_attr, &attr_type));

  switch (attr_type) {
    case OrtOpAttrType::ORT_OP_ATTR_INT: {
      attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);

      int64_t i_val = 0;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ReadOpAttr(&ort_attr, attr_type, &i_val, sizeof(i_val), &total_attr_bytes));
      attr_proto.set_i(i_val);
      break;
    }
    case OrtOpAttrType::ORT_OP_ATTR_INTS: {
      attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);

      // First call to ReadOpAttr gets the total byte size. Second call reads the data.
      Ort::Status status{ort_api.ReadOpAttr(&ort_attr, attr_type, nullptr, 0, &total_attr_bytes)};
      std::vector<int64_t> i_vals(total_attr_bytes / sizeof(int64_t));
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ReadOpAttr(&ort_attr, attr_type, i_vals.data(), total_attr_bytes,
                                                        &total_attr_bytes));

      auto* ints = attr_proto.mutable_ints();
      for (int64_t val : i_vals) {
        ints->Add(val);
      }
      break;
    }
    case OrtOpAttrType::ORT_OP_ATTR_FLOAT: {
      attr_proto.set_type(onnx::AttributeProto_AttributeType_FLOAT);

      float f_val = 0.0f;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ReadOpAttr(&ort_attr, attr_type, &f_val, sizeof(f_val), &total_attr_bytes));
      attr_proto.set_f(f_val);
      break;
    }
    case OrtOpAttrType::ORT_OP_ATTR_FLOATS: {
      attr_proto.set_type(onnx::AttributeProto_AttributeType_FLOATS);

      // First call to ReadOpAttr gets the total byte size. Second call reads the data.
      Ort::Status status{ort_api.ReadOpAttr(&ort_attr, attr_type, nullptr, 0, &total_attr_bytes)};
      std::vector<float> f_vals(total_attr_bytes / sizeof(float));

      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ReadOpAttr(&ort_attr, attr_type, f_vals.data(), total_attr_bytes,
                                                        &total_attr_bytes));

      auto* floats = attr_proto.mutable_floats();
      for (float val : f_vals) {
        floats->Add(val);
      }
      break;
    }
    case OrtOpAttrType::ORT_OP_ATTR_STRING: {
      attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);

      // First call to ReadOpAttr gets the total byte size. Second call reads the data.
      Ort::Status status{ort_api.ReadOpAttr(&ort_attr, attr_type, nullptr, 0, &total_attr_bytes)};
      std::string* str = attr_proto.mutable_s();

      str->resize(total_attr_bytes);
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ReadOpAttr(&ort_attr, attr_type, str->data(), total_attr_bytes,
                                                        &total_attr_bytes));

      str->resize(total_attr_bytes);
      break;
    }
    case OrtOpAttrType::ORT_OP_ATTR_STRINGS: {
      attr_proto.set_type(onnx::AttributeProto_AttributeType_STRINGS);

      // First call to ReadOpAttr gets the total byte size. Second call reads the data.
      Ort::Status status{ort_api.ReadOpAttr(&ort_attr, attr_type, nullptr, 0, &total_attr_bytes)};
      std::vector<char> chars(total_attr_bytes, '\0');

      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ReadOpAttr(&ort_attr, attr_type, chars.data(), total_attr_bytes,
                                                        &total_attr_bytes));

      auto* strs = attr_proto.mutable_strings();

      // Strings are all in a single buffer, each separated with a '\0'.
      // Extract each string and add it to the STRINGS attribute array.
      char* at = chars.data();
      char* end = at + chars.size();

      while (at < end) {
        char* str_begin = at;

        while (*at && at < end) {
          at++;
        }

        strs->Add()->assign(str_begin, at - str_begin);
        if (at < end) {
          assert(*at == '\0');
          at++;  // Skip '\0' to get to the beginning of the next string.
        }
      }

      break;
    }
    case OrtOpAttrType::ORT_OP_ATTR_TENSOR: {
      attr_proto.set_type(onnx::AttributeProto_AttributeType_TENSOR);

      onnx::TensorProto tensor_proto;
      std::string name = std::string(attr_name) + "_tensor_proto";
      tensor_proto.set_name(name);
      tensor_proto.add_dims(2);
      tensor_proto.add_dims(3);

      const OrtValue* ort_value = nullptr;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetTensorAttributeAsOrtValue(&ort_node, &ort_attr, &ort_value));

      // Get tensor type and shape info
      OrtTensorTypeAndShapeInfo* type_shape_info;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorTypeAndShape(ort_value, &type_shape_info));

      // Get tensor type
      ONNXTensorElementDataType element_type;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorElementType(type_shape_info, &element_type));

      // Set tensor type
      switch (element_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_FLOAT);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT8);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_INT8);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT16);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_INT16);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_INT32);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_INT64);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_BOOL);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_DOUBLE);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT32);
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
          tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT64);
        }
        default: {
          std::string err_msg = "Unexpected ONNXTensorElementDataType with value " + std::to_string(static_cast<int>(element_type));
          return Ort::Status(err_msg.c_str(), ORT_FAIL);
        }
      }

      // Get rank
      size_t num_dims;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetDimensionsCount(type_shape_info, &num_dims));

      // Get dimensions
      std::vector<int64_t> dims(num_dims);
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetDimensions(type_shape_info, dims.data(), num_dims));

      // Set dimensions
      for (auto& dim : dims) {
        tensor_proto.add_dims(dim);
      }

      const void* data = nullptr;
      size_t data_bytes = 0;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorData(ort_value, &data));
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(ort_value, &data_bytes));

      // Copy the Ortvalue to TensorProto as raw data
      tensor_proto.set_raw_data(data, data_bytes);

      ort_api.ReleaseTensorTypeAndShapeInfo(type_shape_info);
    }
    default: {
      std::string err_msg = "Unexpected OrtOpAttrType with value " + std::to_string(static_cast<int>(attr_type));
      return Ort::Status(err_msg.c_str(), ORT_FAIL);
    }
  }

  return Ort::Status{nullptr};
}

}  // namespace OrtEpUtils
#endif  // ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
