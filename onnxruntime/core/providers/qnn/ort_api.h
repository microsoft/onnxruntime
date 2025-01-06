// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

// This compilation unit (ort_api.h/.cc) encapsulates the interface between the EP and ORT in a manner
// that allows QNN EP to built either as a static library or a dynamic shared library.
// The preprocessor macro `BUILD_QNN_EP_STATIC_LIB` is defined and set to 1 if QNN EP
// is built as a static library.

#if BUILD_QNN_EP_STATIC_LIB
// Includes when building QNN EP statically
#ifdef _WIN32
#include <Windows.h>
#include <winmeta.h>
#include "core/platform/tracing.h"
#include "core/platform/windows/logging/etw_sink.h"
#endif

#include "onnx/defs/data_type_utils.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/capture.h"
#include "core/common/path_string.h"
#include "core/platform/env.h"
#include "core/framework/data_types.h"
#include "core/framework/float16.h"
#include "core/framework/run_options.h"
#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/node_unit.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/constants.h"
#include "core/graph/basic_types.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/common.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#else
// Includes when building QNN EP as a shared library
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#endif

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

#include <memory>
#include <vector>

namespace onnxruntime {
#if BUILD_QNN_EP_STATIC_LIB
using Node_EdgeEnd = Node::EdgeEnd;
#endif

#if BUILD_QNN_EP_STATIC_LIB
void RunOnUnload(std::function<void()> function);
inline const Env& GetDefaultEnv() { return Env::Default(); }
#endif

inline void InitOrtCppApi() {
#if BUILD_QNN_EP_STATIC_LIB
  // Do nothing. Including "onnxruntime_cxx_api.h" normally initializes the global api_ object.
#else
  // Call util function in provider bridge that initializes the global api_ object.
  InitProviderOrtApi();
#endif
}

/// <summary>
/// Creates an onnxruntime or onnx object. Works for both static and shared library builds of QNN EP.
/// <!-- Example: auto model = Factory<Model>::Create(/* args ... */); -->
/// Example: auto model = Factory&lt;Model&gt;::Create(/* args ... */);
/// </summary>
/// <typeparam name="T">Type of the object to create</typeparam>
template <typename T>
struct Factory {
  template <typename... Params>
  static inline std::unique_ptr<T> Create(Params&&... params) {
#if BUILD_QNN_EP_STATIC_LIB
    return std::make_unique<T>(std::forward<Params>(params)...);
#else
    return T::Create(std::forward<Params>(params)...);
#endif
  }
};

// Specialization of Factory for creating NodeUnit objects.
template <>
struct Factory<NodeUnit> {
  static std::unique_ptr<NodeUnit> Create(gsl::span<const Node* const> dq_nodes,
                                          const Node& target_node,
                                          gsl::span<const Node* const> q_nodes,
                                          NodeUnit::Type unit_type,
                                          gsl::span<const NodeUnitIODef> inputs,
                                          gsl::span<const NodeUnitIODef> outputs,
                                          size_t input_edge_count,
                                          gsl::span<const Node_EdgeEnd* const> output_edges);
};

inline const ConfigOptions& RunOptions__GetConfigOptions(const RunOptions& run_options) {
#if BUILD_QNN_EP_STATIC_LIB
  return run_options.config_options;
#else
  return run_options.GetConfigOptions();
#endif
}

inline std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability& compute_cability) {
#if BUILD_QNN_EP_STATIC_LIB
  return compute_cability.sub_graph;
#else
  return compute_cability.SubGraph();
#endif
}

inline std::vector<NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph& indexed_sub_graph) {
#if BUILD_QNN_EP_STATIC_LIB
  return indexed_sub_graph.nodes;
#else
  return indexed_sub_graph.Nodes();
#endif
}

std::vector<const Node*> Graph__Nodes(const Graph& graph);

inline std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
GetQDQNodeUnits(const GraphViewer& graph_viewer, const logging::Logger& logger) {
#if BUILD_QNN_EP_STATIC_LIB
  return QDQ::GetAllNodeUnits(graph_viewer, logger);
#else
  return QDQ::GetAllNodeUnits(&graph_viewer, logger);
#endif
}

/**
 * Wrapping onnxruntime::Node for retrieving attribute values
 */
class NodeAttrHelper {
 public:
  explicit NodeAttrHelper(const Node& node);

  // Get the attributes from the target node of the node_unit
  explicit NodeAttrHelper(const NodeUnit& node_unit);

  /*
   * Get with default
   */
  float Get(const std::string& key, float def_val) const;
  std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;

  int64_t Get(const std::string& key, int64_t def_val) const;
  std::vector<int64_t> Get(const std::string& key, const std::vector<int64_t>& def_val) const;

  const std::string& Get(const std::string& key, const std::string& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to int32_t
  int32_t Get(const std::string& key, int32_t def_val) const;
  std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;

  // Convert the i() or ints() of the attribute from int64_t to uint32_t
  uint32_t Get(const std::string& key, uint32_t def_val) const;
  std::vector<uint32_t> Get(const std::string& key, const std::vector<uint32_t>& def_val) const;

  /*
   * Get without default.
   */
  std::optional<float> GetFloat(const std::string& key) const;
  std::optional<std::vector<float>> GetFloats(const std::string& key) const;

  std::optional<int64_t> GetInt64(const std::string& key) const;
  std::optional<std::vector<int64_t>> GetInt64s(const std::string& key) const;

  std::optional<std::string> GetString(const std::string& key) const;

  bool HasAttr(const std::string& key) const;

 private:
  const NodeAttributes& node_attributes_;
};

namespace logging {
inline std::unique_ptr<Capture> Capture__Create(const Logger& logger, logging::Severity severity, const char* category,
                                                logging::DataType data_type, const CodeLocation& location) {
#if BUILD_QNN_EP_STATIC_LIB
  return std::make_unique<Capture>(logger, severity, category, data_type, location);
#else
  return Capture::Create(logger, severity, category, data_type, location);
#endif
}
}  // namespace logging
}  // namespace onnxruntime
