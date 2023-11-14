// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "./dll_safe.h"
#include "./my_ort.h"
#include <gsl/gsl>
#include <cassert>
#include <functional>
#include <vector>

namespace vaip_core {

struct OrtApiForVaip {
  ProviderHost* host_;
  // model
  Model* (*model_load)(const std::string& file);  // [0]
  void (*model_delete)(Model* model);             // [1]
  Model* (*model_clone)(const Model& model);      // [2]
  Graph& (*model_main_graph)(Model& model);       // [3]
  void (*model_set_meta_data)(Model& model, const std::string& key,
                              const std::string& value);  // [4]
  DllSafe<std::string> (*model_get_meta_data)(const Model& model,
                                              const std::string& key);  // [5]
  int (*model_has_meta_data)(const Model& model,
                             const std::string& key);  // [6]
  // graph
  const std::string& (*graph_get_name)(const Graph& graph);  // [7]
  const Model& (*graph_get_model)(const Graph& graph);       // [8]
  DllSafe<std::vector<const Node*>> (*graph_nodes_unsafe)(
      const Graph& graph);  // [9]
  DllSafe<std::vector<const NodeArg*>> (*graph_get_inputs_unsafe)(
      const Graph& graph);  // [10]
  DllSafe<std::vector<const NodeArg*>> (*graph_get_outputs_unsafe)(
      const Graph& graph);  // [11]
  void (*graph_set_outputs)(Graph& graph,
                            gsl::span<const NodeArg* const> outputs);  // [12]
  const Node* (*graph_get_node)(const Graph& graph, size_t index);     // [13]
  const Node* (*graph_producer_node)(const Graph& graph,
                                     const std::string& node_arg_name);  // [14]
  const NodeArg* (*graph_get_node_arg)(const Graph& graph,
                                       const std::string& name);  // [15]
  const InitializedTensorSet& (*graph_get_all_initialized_tensors)(
      const Graph& graph);                                               // [16]
  void (*graph_remove_node)(Graph& graph, const NodeInput& node_input);  // [17]
  Node& (*graph_add_node)(Graph& graph, const std::string& name,
                          const std::string& op_type,
                          const std::string& description,
                          const std::vector<const NodeArg*>& input_args,
                          const std::vector<const NodeArg*>& output_args,
                          NodeAttributes& attributes,
                          const std::string& domain);  // [18]
  void (*graph_save)(const Graph& graph, const std::string& filename,
                     const std::string& dat_filename,
                     size_t external_data_threshold);  // [19]
  Node& (*graph_fuse)(
      Graph& graph, const std::string& name, const std::string& op_type,
      const std::vector<size_t>& nodes, const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs,
      const std::vector<std::string>& constant_initializers);  // [20]
  int (*graph_resolve)(Graph& graph, bool force);              // [21]
  DllSafe<std::vector<const Node*>> (*graph_get_consumer_nodes_unsafe)(
      const Graph& graph, const std::string& node_arg_name);  // [22]
  void (*graph_reverse_dfs_from)(
      const Graph& graph, gsl::span<const Node* const> from,
      const std::function<void(const Node*)>& enter,
      const std::function<void(const Node*)>& leave,
      const std::function<bool(const Node* from, const Node* to)>&
          stop);  // [23]

  /// node
  //
  const std::string& (*node_get_name)(const Node& node);     // [24]
  const std::string& (*node_description)(const Node& node);  // [25]
  size_t (*node_get_index)(const Node& node);                // [26]
  const std::string& (*node_op_type)(const Node& node);      // [27]
  const std::string& (*node_op_domain)(const Node& node);    // [28]
  DllSafe<std::vector<NodeInput>> (*node_get_inputs_unsafe)(
      const Node& node);  // [29]
  DllSafe<std::vector<const NodeArg*>> (*node_get_output_node_args_unsafe)(
      const Node& node);  // [30]

  NodeAttributes& (*node_get_attributes)(Node& node);  // [31]

  const Graph& (*node_get_function_body)(const Node& node);  // [32]
  bool (*node_type_is_fused)(const Node& node);              // [33]

  /// node args
  const std::string& (*node_arg_get_name_unsafe)(
      const NodeArg& node_arg);                         // [34]
  bool (*node_arg_is_exists)(const NodeArg& node_arg);  // [35]
  bool (*node_arg_is_constant)(const Graph& graph,
                               const NodeArg& node_arg);  // [36]

  NodeArg& (*node_arg_clone)(Graph& graph, const NodeArg& node_arg,
                             const std::string& name);  // [37]
  NodeArg& (*node_arg_new)(Graph& graph, const std::string& name,
                           const std::vector<int64_t>* shape,
                           int element_type);  // [38]
  DllSafe<std::vector<int64_t>> (*node_arg_get_shape_i64_unsafe)(
      const NodeArg& node_arg);  // [39]
  DllSafe<std::vector<std::string>> (*node_arg_get_denotation_unsafe)(
      const NodeArg& node_arg);  // [40]
  /// here, it is cheating so that even `node_arg` is a const reference,
  /// it still change the shape internally. But we cannot change the
  /// rank, i.e. the size of shape must be same.
  void (*node_arg_set_shape_i64)(const NodeArg& node_arg,
                                 const std::vector<int64_t>& shape);  // [41]
  void (*node_arg_set_denotation)(
      const NodeArg& node_arg,
      const std::vector<std::string>& denotation);            // [42]
  int (*node_arg_get_element_type)(const NodeArg& node_arg);  // [43]
  void (*node_arg_set_element_type)(
      NodeArg& node_arg, int /*TensorProto::DataType*/ data_type);  // [44]
  const TensorProto& (*node_arg_get_const_data_as_tensor)(
      const Graph& graph, const NodeArg& node_arg);                       // [45]
                                                                          /// node attributes.
  NodeAttributes* (*node_attributes_new)();                               // [46]
  void (*node_attributes_delete)(NodeAttributes* p);                      // [47]
  void (*node_attributes_add)(NodeAttributes& p, AttributeProto&& attr);  // [48]
  AttributeProto* (*node_attributes_get)(NodeAttributes& p,
                                         const std::string& name);  // [49]
  DllSafe<std::vector<std::string>> (*node_attributes_get_keys)(
      NodeAttributes& p);  // [50]
  /// attr proto
  void (*attr_proto_delete)(AttributeProto* attr);                        // [51]
  AttributeProto* (*attr_proto_clone)(const AttributeProto& attr);        // [52]
  const std::string& (*attr_proto_get_name)(const AttributeProto& attr);  // [53]
  int (*attr_proto_get_type)(const AttributeProto& attr);                 // [54]
  void (*attr_proto_set_name)(AttributeProto* attr,
                              const std::string& name);  // [55]
  AttributeProto* (*attr_proto_new_int)(const std::string& name,
                                        int64_t value);  // [56]
  AttributeProto* (*attr_proto_new_float)(const std::string& name,
                                          float value);  // [57]
  AttributeProto* (*attr_proto_new_string)(const std::string& name,
                                           const std::string& value);  // [58]
  AttributeProto* (*attr_proto_new_tensor)(const std::string& name,
                                           const TensorProto& value);  // [59]
  AttributeProto* (*attr_proto_new_ints)(
      const std::string& name,
      const std::vector<int64_t>& value);  // [60]
  AttributeProto* (*attr_proto_new_floats)(
      const std::string& name,
      const std::vector<float>& value);  // [61]
  AttributeProto* (*attr_proto_new_strings)(
      const std::string& name, const std::vector<std::string>& value);  // [62]
  int64_t (*attr_proto_get_int)(const AttributeProto& attr);            // [63]
  float (*attr_proto_get_float)(const AttributeProto& attr);            // [64]
  const std::string& (*attr_proto_get_string)(
      const AttributeProto& attr);  // [65]
  const TensorProto& (*attr_proto_get_tensor)(
      const AttributeProto& attr);  // [66]
  gsl::span<const int64_t> (*attr_proto_get_ints)(
      const AttributeProto& attr);  // [67]
  gsl::span<const float> (*attr_proto_get_floats)(
      const AttributeProto& attr);  // [68]
  std::vector<std::string> (*attr_proto_get_strings)(
      const AttributeProto& attr);  // [69]

  // tensor_proto
  void (*tensor_proto_delete)(TensorProto* tp);  // [70]
  DllSafe<std::vector<int64_t>> (*tensor_proto_get_shape_unsafe)(
      const TensorProto& tensor_proto);                            // [71]
  int (*tensor_proto_data_type)(const TensorProto& tensor_proto);  // [72]
  TensorProto* (*tensor_proto_new_floats)(
      const std::string& name, const std::vector<int64_t>& shape,
      const std::vector<float>& data);  // [73]
  TensorProto* (*tensor_proto_new_i64)(
      const std::string& name, const std::vector<int64_t>& shape,
      const std::vector<int64_t>& data);  // [74]
  TensorProto* (*tensor_proto_new_i32)(
      const std::string& name, const std::vector<int64_t>& shape,
      const std::vector<int32_t>& data);  // [75]
  TensorProto* (*tensor_proto_new_i8)(const std::string& name,
                                      const std::vector<int64_t>& shape,
                                      const std::vector<int8_t>& data);  // [76]
  const std::string& (*tensor_proto_get_name)(
      const TensorProto& tensor_proto);                             // [77]
  size_t (*tensor_proto_raw_data_size)(const TensorProto& tensor);  // [78]
  gsl::span<const char> (*tensor_proto_as_raw)(
      const TensorProto& tensor);  // [79]

  DllSafe<std::string> (*get_lib_id)();    // [80]
  DllSafe<std::string> (*get_lib_name)();  // [81]
};

#ifndef USE_VITISAI
VAIP_DLL_SPEC const OrtApiForVaip* api();
// avoid macro redefinitions in vitisai ort ep.
#define VAIP_ORT_API(name)             \
  (::vaip_core::api()->name != nullptr \
       ? ::vaip_core::api()->name      \
       : (assert(false && #name " is not set"), nullptr))
#endif
}  // namespace vaip_core
