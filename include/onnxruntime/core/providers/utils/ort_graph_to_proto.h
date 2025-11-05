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
/// <summary>
/// Convert the endianess of data based of tensor element type. Mainly used in BE systems.
/// </summary>
/// <param name="value_info">OrtValueInfo for the initializer. Can be used to query name, type, shape,
///                           and consumer nodes.</param>
/// <param name="data">Pointer to data buffer.</param>
/// <param name="bytes">Length of data buffer.</param>
/// <returns>An Ort::Status indicating success or an error.</returns>
Ort::Status ConvertExternalData(const OrtValueInfo* value_info, void* data, size_t bytes);

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
    Ort::Status _status{(fn)};             \
    if (!_status.IsOK()) {                 \
      return _status;                      \
    }                                      \
  } while (0)

#define ORT_EP_UTILS_CXX_RETURN_IF_ERROR(fn) \
  ORT_EP_UTILS_C_RETURN_IF_ERROR(fn)

#define ORT_EP_UTILS_C_RETURN_IF(cond, msg) \
  do {                                      \
    if ((cond)) {                           \
      return Ort::Status{msg, ORT_FAIL};    \
    }                                       \
  } while (0)

namespace OrtEpUtils {

static Ort::Status GetOrtValueInfoTensorTypeShape(Ort::ConstValueInfo vi,
                                                  bool get_symbolic_dims,
                                                  /*out*/ ONNXTensorElementDataType& elem_type,
                                                  /*out*/ std::vector<int64_t>& dims,
                                                  /*out*/ std::vector<std::string>& symbolic_dims,
                                                  /*out*/ bool& has_shape);
static Ort::Status OrtValueInfoToProto(Ort::ConstValueInfo ort_value_info, onnx::ValueInfoProto& value_info_proto);
static Ort::Status OrtOpAttrToProto(Ort::ConstOpAttr ort_attr, onnx::AttributeProto& attr_proto);
static Ort::Status GetTensorElementSize(const ONNXTensorElementDataType& element_type, size_t& element_size);
static void SwapByteOrderInplace(void* data, const size_t& data_len, const size_t& element_size);

// Below endian enum class is referenced from include/onnxruntime/core/framework/endian.h
enum class endian {
#if defined(_WIN32)
  little = 0,
  big = 1,
  native = little,
#elif defined(__GNUC__) || defined(__clang__)
  little = __ORDER_LITTLE_ENDIAN__,
  big = __ORDER_BIG_ENDIAN__,
  native = __BYTE_ORDER__,
#else
#error onnxruntime::endian is not implemented in this environment.
#endif
};

Ort::Status OrtGraphToProto(const OrtGraph& graph,
                            onnx::GraphProto& graph_proto,
                            HandleInitializerDataFunc handle_initializer_data_func) {
  try {
    Ort::ConstGraph ort_graph{&graph};
    //
    // Set GraphProto metadata
    //
    auto graph_name = ort_graph.GetName();
    graph_proto.set_name(graph_name);
    graph_proto.set_doc_string("Serialized from OrtGraph");

    //
    // Set GraphProto inputs and outputs
    //
    std::vector<Ort::ConstValueInfo> graph_inputs = ort_graph.GetInputs();
    std::vector<Ort::ConstValueInfo> graph_outputs = ort_graph.GetOutputs();

    for (const auto& ort_value_info : graph_inputs) {
      onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_input()->Add();
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtValueInfoToProto(ort_value_info, *value_info_proto));
    }

    for (const auto& ort_value_info : graph_outputs) {
      onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_output()->Add();
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtValueInfoToProto(ort_value_info, *value_info_proto));
    }

    //
    // Set GraphProto nodes, value_infos, and initializers.
    //

    // Use std::maps to store OrtValueInfos for GraphProto.value_info and GraphProto.initializer.
    // A std::map maintains its elements in a stable ordering.
    std::map<std::string, Ort::ConstValueInfo> value_infos;              // For GraphProto.value_info
    std::map<std::string, Ort::ConstValueInfo> initializer_value_infos;  // For GraphProto.initializer

    // Helper function to collect an OrtValueInfo into `value_infos` or `initializer_value_infos`.
    // Optionally returns the OrtValueInfo name to the caller.
    auto collect_value_info = [&value_infos,
                               &initializer_value_infos](Ort::ConstValueInfo ort_value_info,
                                                         /*out*/ std::optional<std::string>& value_name_out) {
      auto value_name = ort_value_info.GetName();

      if (value_name_out) {
        *value_name_out = value_name;
      }

      if (value_infos.count(value_name) != 0 || initializer_value_infos.count(value_name) != 0) {
        return;  // Already processed this OrtValueInfo.
      }

      bool is_required_graph_input = ort_value_info.IsRequiredGraphInput();
      bool is_optional_graph_input = ort_value_info.IsOptionalGraphInput();
      bool is_graph_output = ort_value_info.IsGraphOutput();
      bool is_constant_initializer = ort_value_info.IsConstantInitializer();
      bool is_from_outer_scope = ort_value_info.IsFromOuterScope();

      // Don't add graph inputs or graph outputs to GraphProto's list of value_infos.
      // Do add initializers (constant and non-constant) to GraphProto's list of initializer tensors.
      // For values defined in an outer scope, just add the value info but not the initializer.
      if (is_from_outer_scope) {
        value_infos.emplace(value_name, ort_value_info);
      } else if (is_optional_graph_input) {
        initializer_value_infos.emplace(value_name, ort_value_info);
      } else if (is_constant_initializer) {
        value_infos.emplace(value_name, ort_value_info);
        initializer_value_infos.emplace(value_name, ort_value_info);
      } else if (!is_required_graph_input && !is_graph_output) {
        value_infos.emplace(value_name, ort_value_info);  // This is an internal OrtValueInfo.
      }
    };

    std::vector<Ort::ConstNode> nodes = ort_graph.GetNodes();
    // Loop through all nodes (topological order): add NodeProto instances to GraphProto and track OrtValueInfos
    // that will be stored in GraphProto.value_info and GraphProto.initializer.
    for (const auto& ort_node : nodes) {
      onnx::NodeProto* node_proto = graph_proto.add_node();

      std::string node_name = ort_node.GetName();
      std::string node_domain = ort_node.GetDomain();
      std::string node_op_type = ort_node.GetOperatorType();

      node_proto->set_name(node_name);
      node_proto->set_domain(node_domain);
      node_proto->set_op_type(node_op_type);

      // Handle node attributes
      std::vector<Ort::ConstOpAttr> ort_attrs = ort_node.GetAttributes();
      for (const auto& attr : ort_attrs) {
        OrtOpAttrType attr_type = attr.GetType();
        if (attr_type == OrtOpAttrType::ORT_OP_ATTR_GRAPH) {
          // ORT does not support reading subgraphs via ReadOpAttr(), so skip it.
          // Can use Node_GetSubgraphs to get subgraphs.
          continue;
        }

        onnx::AttributeProto* attr_proto = node_proto->add_attribute();
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtOpAttrToProto(attr, *attr_proto));
      }

      // Handle node subgraphs
      std::vector<Ort::AttrNameSubgraph> ort_subgraphs = ort_node.GetSubgraphs();
      for (const auto& [subgraph_attr_name, ort_subgraph] : ort_subgraphs) {
        onnx::AttributeProto* attr_proto = node_proto->add_attribute();
        onnx::GraphProto* subgraph_proto = attr_proto->mutable_g();
        attr_proto->set_name(subgraph_attr_name);
        attr_proto->set_type(onnx::AttributeProto_AttributeType_GRAPH);
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtGraphToProto(*ort_subgraph, *subgraph_proto));
      }

      // Handle node inputs
      std::vector<Ort::ConstValueInfo> ort_inputs = ort_node.GetInputs();
      for (const auto& vi : ort_inputs) {
        if (vi == nullptr) {
          // missing optional input.
          node_proto->add_input("");
          continue;
        }

        std::optional<std::string> value_name;
        value_name.emplace();
        collect_value_info(vi, value_name);
        node_proto->add_input(*value_name);
      }

      // Handle implicit inputs to this node.
      std::vector<Ort::ConstValueInfo> ort_implicit_inputs = ort_node.GetImplicitInputs();
      for (const auto& vi : ort_implicit_inputs) {
        assert(vi != nullptr);
        std::optional<std::string> value_name;
        collect_value_info(vi, value_name);
      }

      // Handle node outputs
      std::vector<Ort::ConstValueInfo> ort_outputs = ort_node.GetOutputs();
      for (const auto& vi : ort_outputs) {
        if (vi == nullptr) {
          // missing optional output.
          node_proto->add_output("");
          continue;
        }

        std::optional<std::string> value_name;
        value_name.emplace();
        collect_value_info(vi, value_name);
        node_proto->add_output(*value_name);
      }
    }

    // Add value_infos to GraphProto as ValueInfoProto objects.
    for (const auto& [value_name, value_info] : value_infos) {
      onnx::ValueInfoProto* value_info_proto = graph_proto.mutable_value_info()->Add();
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtValueInfoToProto(value_info, *value_info_proto));
    }

    // Add initializers to GraphProto as TensorProto objects.
    for (const auto& [initializer_name, initializer_value_info] : initializer_value_infos) {
      std::vector<int64_t> initializer_dims;
      std::vector<std::string> initializer_sym_dims;
      ONNXTensorElementDataType initializer_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      bool has_shape = false;
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetOrtValueInfoTensorTypeShape(initializer_value_info, /*get_sym_dims*/ false,
                                                                      initializer_elem_type, initializer_dims,
                                                                      initializer_sym_dims, has_shape));

      onnx::TensorProto* tensor_proto = graph_proto.add_initializer();
      tensor_proto->set_name(initializer_name);
      tensor_proto->set_data_type(initializer_elem_type);

      auto* tensor_proto_dims = tensor_proto->mutable_dims();
      for (int64_t dim : initializer_dims) {
        tensor_proto_dims->Add(dim);
      }

      Ort::ConstValue ort_value{nullptr};
      ORT_EP_UTILS_C_RETURN_IF_ERROR(initializer_value_info.GetInitializer(ort_value));

      assert(ort_value.IsTensor());
      const void* data = ort_value.GetTensorRawData();
      const size_t data_bytes = ort_value.GetTensorSizeInBytes();

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
        if constexpr (endian::native == endian::big) {
          size_t element_size = 0;
          GetTensorElementSize(initializer_elem_type, element_size);
          // create local copy of data and do endianess conversion
          auto raw_data_buf = std::make_unique<unsigned char[]>(data_bytes);
          std::memcpy(raw_data_buf.get(), data, data_bytes);
          SwapByteOrderInplace(raw_data_buf.get(), data_bytes, element_size);
          tensor_proto->set_raw_data(raw_data_buf.get(), data_bytes);
        } else {
          tensor_proto->set_raw_data(data, data_bytes);
        }
      }
    }
  } catch (const Ort::Exception& ex) {
    return Ort::Status{ex};
  } catch (const std::exception& ex) {
    return Ort::Status{ex.what(), ORT_FAIL};
  }

  return Ort::Status{nullptr};
}

Ort::Status OrtGraphToProto(const OrtGraph& graph,
                            onnx::ModelProto& model_proto,
                            HandleInitializerDataFunc handle_initializer_data_func) {
  try {
    // Check that OrtGraph is a top-level graph (no parent node).
    Ort::ConstGraph ort_graph{&graph};
    Ort::ConstNode parent_node = ort_graph.GetParentNode();
    ORT_EP_UTILS_C_RETURN_IF(parent_node != nullptr, "Cannot serialize nested OrtGraph into a ModelProto");

    // Set model description.
    model_proto.set_doc_string("Serialized from OrtGraph");
    model_proto.set_producer_name("ort_ep_utils::OrtGraphToProto");

    // Set ir version.
    int64_t ir_version = ort_graph.GetOnnxIRVersion();
    model_proto.set_ir_version(ir_version);

    // Set operator sets.
    std::vector<Ort::OperatorSet> op_sets = ort_graph.GetOperatorSets();
    ORT_EP_UTILS_C_RETURN_IF(op_sets.empty(), "OrtGraph should have at least one operator set.");

    auto* operator_sets = model_proto.mutable_opset_import();

    for (const auto& op_set : op_sets) {
      onnx::OperatorSetIdProto* operator_set = operator_sets->Add();
      operator_set->set_domain(op_set.domain);
      operator_set->set_version(op_set.version);
    }

    model_proto.clear_graph();
    onnx::GraphProto* graph_proto = model_proto.mutable_graph();
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(OrtGraphToProto(*ort_graph, *graph_proto, handle_initializer_data_func));

  } catch (const Ort::Exception& ex) {
    return Ort::Status(ex);
  } catch (const std::exception& ex) {
    return Ort::Status(ex.what(), ORT_EP_FAIL);
  }

  return Ort::Status{nullptr};
}

static Ort::Status GetOrtValueInfoTensorTypeShape(Ort::ConstValueInfo vi,
                                                  bool get_symbolic_dims,
                                                  /*out*/ ONNXTensorElementDataType& elem_type,
                                                  /*out*/ std::vector<int64_t>& dims,
                                                  /*out*/ std::vector<std::string>& symbolic_dims,
                                                  /*out*/ bool& has_shape) {
  try {
    Ort::ConstTypeInfo ort_type_info = vi.TypeInfo();
    ONNXType ort_onnx_type = ort_type_info.GetONNXType();
    ORT_EP_UTILS_C_RETURN_IF(ort_onnx_type != ONNX_TYPE_TENSOR, "Expected OrtValueInfo to represent a Tensor");

    Ort::ConstTensorTypeAndShapeInfo ort_type_shape = ort_type_info.GetTensorTypeAndShapeInfo();
    elem_type = ort_type_shape.GetElementType();
    has_shape = ort_type_shape.HasShape();

    if (has_shape) {
      const size_t num_dims = ort_type_shape.GetDimensionsCount();
      dims = ort_type_shape.GetShape();

      if (get_symbolic_dims) {
        std::vector<const char*> ort_dim_syms(num_dims, nullptr);
        ort_type_shape.GetSymbolicDimensions(ort_dim_syms.data(), ort_dim_syms.size());

        symbolic_dims.reserve(num_dims);
        for (const char* sym_dim : ort_dim_syms) {
          symbolic_dims.push_back(sym_dim);
        }
      }
    }
  } catch (const Ort::Exception& ex) {
    return Ort::Status{ex};
  } catch (const std::exception& ex) {
    return Ort::Status{ex.what(), ORT_EP_FAIL};
  }
  return Ort::Status{nullptr};
}

// Create an onnx::ValueInfoProto from an OrtValueInfo (name, type, shape).
static Ort::Status OrtValueInfoToProto(Ort::ConstValueInfo ort_value_info,
                                       onnx::ValueInfoProto& value_info_proto) {
  std::vector<int64_t> ort_dims;
  std::vector<std::string> ort_dim_syms;
  ONNXTensorElementDataType ort_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  // We currently only support ONNX tensors. Support for other types (e.g., ONNX_TYPE_SEQUENCE) can be added later.
  bool has_shape = false;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetOrtValueInfoTensorTypeShape(ort_value_info, /*get_sym_dims*/ true,
                                                                  ort_elem_type, ort_dims, ort_dim_syms,
                                                                  has_shape));

  value_info_proto.set_name(ort_value_info.GetName());

  onnx::TypeProto_Tensor* type_proto_tensor = value_info_proto.mutable_type()->mutable_tensor_type();
  type_proto_tensor->set_elem_type(ort_elem_type);

  // If there is no shape, do not set a TensorShapeProto.
  if (has_shape) {
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

static Ort::Status OrtOpAttrToProto(Ort::ConstOpAttr attr, onnx::AttributeProto& attr_proto) {
  try {
    std::string attr_name = attr.GetName();
    attr_proto.set_name(attr_name);

    OrtOpAttrType attr_type = attr.GetType();

    switch (attr_type) {
      case OrtOpAttrType::ORT_OP_ATTR_INT: {
        int64_t i_val = 0;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(attr.GetValue(i_val));
        attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
        attr_proto.set_i(i_val);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_INTS: {
        std::vector<int64_t> i_vals;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(attr.GetValueArray(i_vals));
        auto* ints = attr_proto.mutable_ints();
        ints->Assign(i_vals.begin(), i_vals.end());
        attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_FLOAT: {
        float f_val = 0.0f;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(attr.GetValue(f_val));
        attr_proto.set_type(onnx::AttributeProto_AttributeType_FLOAT);
        attr_proto.set_f(f_val);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_FLOATS: {
        std::vector<float> f_vals;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(attr.GetValueArray(f_vals));
        auto* floats = attr_proto.mutable_floats();
        floats->Assign(f_vals.begin(), f_vals.end());
        attr_proto.set_type(onnx::AttributeProto_AttributeType_FLOATS);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_STRING: {
        std::string* str = attr_proto.mutable_s();
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(attr.GetValue(*str));
        attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_STRINGS: {
        std::vector<std::string> result;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(attr.GetValueArray(result));
        auto* strs = attr_proto.mutable_strings();
        strs->Assign(result.begin(), result.end());
        attr_proto.set_type(onnx::AttributeProto_AttributeType_STRINGS);
        break;
      }
      case OrtOpAttrType::ORT_OP_ATTR_TENSOR: {
        attr_proto.set_type(onnx::AttributeProto_AttributeType_TENSOR);

        onnx::TensorProto tensor_proto;

        // TensorProto as an attribute value doesn't require a name.

        Ort::Value tensor;
        ORT_EP_UTILS_C_RETURN_IF_ERROR(attr.GetTensorAttributeAsOrtValue(tensor));

        // Get tensor type and shape info
        Ort::TensorTypeAndShapeInfo type_shape_info = tensor.GetTensorTypeAndShapeInfo();

        // Get tensor type
        ONNXTensorElementDataType element_type = type_shape_info.GetElementType();

        switch (element_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_FLOAT);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT8);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_INT8);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT16);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_INT16);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_INT32);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_INT64);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_BOOL);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_DOUBLE);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT32);
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
            tensor_proto.set_data_type(onnx::TensorProto_DataType_UINT64);
            break;
          }
          default: {
            std::string err_msg = "Unexpected ONNXTensorElementDataType with value " + std::to_string(static_cast<int>(element_type));
            return Ort::Status(err_msg.c_str(), ORT_FAIL);
          }
        }

        auto shape = type_shape_info.GetShape();

        for (auto& dim : shape) {
          tensor_proto.add_dims(dim);
        }

        const void* data = tensor.GetTensorRawData();
        const size_t data_bytes = tensor.GetTensorSizeInBytes();

        // Copy the Ortvalue to TensorProto as raw data
        if constexpr (endian::native == endian::big) {
          size_t element_size = 0;
          GetTensorElementSize(element_type, element_size);
          // create local copy of data and do endianess conversion
          auto raw_data_buf = std::make_unique<unsigned char[]>(data_bytes);
          std::memcpy(raw_data_buf.get(), data, data_bytes);
          SwapByteOrderInplace(raw_data_buf.get(), data_bytes, element_size);
          tensor_proto.set_raw_data(raw_data_buf.get(), data_bytes);
        } else {
          tensor_proto.set_raw_data(data, data_bytes);
        }

        *(attr_proto.mutable_t()) = std::move(tensor_proto);
        break;
      }
      default: {
        std::string err_msg = "Unexpected OrtOpAttrType with value " + std::to_string(static_cast<int>(attr_type));
        return Ort::Status(err_msg.c_str(), ORT_FAIL);
      }
    }
  } catch (const Ort::Exception& ex) {
    return Ort::Status{ex};
  } catch (const std::exception& ex) {
    return Ort::Status{ex.what(), ORT_FAIL};
  }

  return Ort::Status{nullptr};
}

Ort::Status ConvertExternalData(const OrtValueInfo* value_info, void* data, size_t bytes) {
#if !defined(_WIN32)
  if constexpr (endian::native == endian::little) {
    return Ort::Status{nullptr};
  }
  std::vector<int64_t> initializer_dims;
  std::vector<std::string> initializer_sym_dims;
  ONNXTensorElementDataType initializer_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t element_size = 0;
  Ort::ConstValueInfo ort_value_info{value_info};
  bool has_shape{false};
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetOrtValueInfoTensorTypeShape(ort_value_info, false,
                                                                  initializer_elem_type, initializer_dims,
                                                                  initializer_sym_dims, has_shape));
  GetTensorElementSize(initializer_elem_type, element_size);
  if (element_size != 1) {
    SwapByteOrderInplace(data, bytes, element_size);
  }
#else
  (value_info);
  (data);
  (bytes);
#endif
  return Ort::Status{nullptr};
}

static Ort::Status GetTensorElementSize(const ONNXTensorElementDataType& element_type, size_t& element_size) {
  using TensorElemDataMap = std::unordered_map<ONNXTensorElementDataType, size_t>;
  static TensorElemDataMap tensor_elem_data_size{
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, sizeof(uint16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, sizeof(int16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, sizeof(uint16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, sizeof(uint16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, sizeof(uint64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, sizeof(double)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, sizeof(uint8_t)},
  };
  auto pos = tensor_elem_data_size.find(element_type);
  if (pos == tensor_elem_data_size.end()) {
    std::string err_msg = "Unexpected ONNXTensorElementDataType with value " + std::to_string(static_cast<int>(element_type));
    return Ort::Status(err_msg.c_str(), ORT_FAIL);
  }
  element_size = pos->second;
  return Ort::Status{nullptr};
}

static void SwapByteOrderInplace(void* data, const size_t& data_len, const size_t& element_size) {
  char* bytes = reinterpret_cast<char*>(data);
  size_t num_elements = data_len / element_size;
  for (size_t i = 0; i < num_elements; ++i) {
    char* start_byte = bytes + i * element_size;
    char* end_byte = start_byte + element_size - 1;
    for (size_t count = 0; count < element_size / 2; ++count) {
      std::swap(*start_byte++, *end_byte--);
    }
  }
}

}  // namespace OrtEpUtils
#endif  // ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
