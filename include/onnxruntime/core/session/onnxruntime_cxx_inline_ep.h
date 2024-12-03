// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "onnxruntime_cxx_api_ep.h" instead.

namespace Ort{
namespace PluginEP {

static const OrtGraphApi* ort_graph_api = GetApi().GetGraphApi(ORT_API_VERSION);

inline Graph::Graph(const OrtGraphViewer* graph) : graph_(graph) {}

inline const char* Graph::GetName() {
  const char* graph_name = nullptr;
  ThrowOnError(ort_graph_api->OrtGraph_GetName(graph_, &graph_name));
  return graph_name;
}

inline Node::Node(const OrtNode* node) : node_(node) {}

inline const char* Node::GetName() {
  const char* node_name = nullptr;
  ThrowOnError(ort_graph_api->OrtNode_GetName(node_, &node_name));
  return node_name;
}

}
}
