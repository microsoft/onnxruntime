// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License

#pragma once

// // This compilation unit (ort_api.h/.cc) encapsulates the interface between the EP and ORT in a manner
// // that allows QNN EP to built either as a static library or a dynamic shared library.
// // The preprocessor macro `BUILD_QNN_EP_STATIC_LIB` is defined and set to 1 if QNN EP
// // is built as a static library.


// #define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
// #endif

// #include "core/common/inlined_containers.h"
// #include "core/session/onnxruntime_session_options_config_keys.h"
// #include "core/session/onnxruntime_run_options_config_keys.h"

// #include <memory>
// #include <vector>

// namespace onnxruntime {

// inline void InitOrtCppApi() {
//   // Call util function in provider bridge that initializes the global api_ object.
//   InitProviderOrtApi();
// }

// /// <summary>
// /// Creates an onnxruntime or onnx object. Works for both static and shared library builds of QNN EP.
// /// <!-- Example: auto model = Factory<Model>::Create(/* args ... */); -->
// /// Example: auto model = Factory&lt;Model&gt;::Create(/* args ... */);
// /// </summary>
// /// <typeparam name="T">Type of the object to create</typeparam>
// template <typename T>
// struct Factory {
//   template <typename... Params>
//   static inline std::unique_ptr<T> Create(Params&&... params) {
//     return T::Create(std::forward<Params>(params)...);
//   }
// };

// inline const ConfigOptions& RunOptions__GetConfigOptions(const RunOptions& run_options) {
//   return run_options.GetConfigOptions();
// }

// inline std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability& compute_cability) {
//   return compute_cability.SubGraph();
// }

// inline std::vector<NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph& indexed_sub_graph) {
//   return indexed_sub_graph.Nodes();
// }

// std::vector<const Node*> Graph__Nodes(const Graph& graph);

// inline std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
// GetQDQNodeUnits(const GraphViewer& graph_viewer, const logging::Logger& logger) {
//   return QDQ::GetAllNodeUnits(&graph_viewer, logger);
// }

// /**
//  * Wrapping onnxruntime::Node for retrieving attribute values
//  */
// class NodeAttrHelper {
//  public:
//   explicit NodeAttrHelper(const Node& node);

//   // Get the attributes from the target node of the node_unit
//   explicit NodeAttrHelper(const NodeUnit& node_unit);

//   /*
//    * Get with default
//    */
//   float Get(const std::string& key, float def_val) const;
//   std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;

//   int64_t Get(const std::string& key, int64_t def_val) const;
//   std::vector<int64_t> Get(const std::string& key, const std::vector<int64_t>& def_val) const;

//   const std::string& Get(const std::string& key, const std::string& def_val) const;
//   std::vector<std::string> Get(const std::string& key, const std::vector<std::string>& def_val) const;

//   // Convert the i() or ints() of the attribute from int64_t to int32_t
//   int32_t Get(const std::string& key, int32_t def_val) const;
//   std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;

//   // Convert the i() or ints() of the attribute from int64_t to uint32_t
//   uint32_t Get(const std::string& key, uint32_t def_val) const;
//   std::vector<uint32_t> Get(const std::string& key, const std::vector<uint32_t>& def_val) const;

//   /*
//    * Get without default.
//    */
//   std::optional<float> GetFloat(const std::string& key) const;
//   std::optional<std::vector<float>> GetFloats(const std::string& key) const;

//   std::optional<int64_t> GetInt64(const std::string& key) const;
//   std::optional<std::vector<int64_t>> GetInt64s(const std::string& key) const;

//   std::optional<std::string> GetString(const std::string& key) const;

//   bool HasAttr(const std::string& key) const;

//  private:
//   const NodeAttributes& node_attributes_;
// };
// }  // namespace onnxruntime
