#include "core/session/onnxruntime_c_api_ep.h"
#include "ort_apis_ep.h"
#include "core/graph/graph_viewer.h"

ORT_API_STATUS_IMPL(OrtGraphApis::OrtGraph_PlaceHolder, const OrtGraphViewer* graph, _Out_ int* out) {
  const ::onnxruntime::GraphViewer* graph_viewer = reinterpret_cast<const ::onnxruntime::GraphViewer*>(graph);
  *out = graph_viewer->NumberOfNodes();
  return nullptr;
}

static constexpr OrtGraphApi ort_graph_api = {
    &OrtGraphApis::OrtGraph_PlaceHolder,
};

ORT_API(const OrtGraphApi*, OrtGraphApis::GetGraphApi, uint32_t) {
  // No constraints on the API version yet.
  return &ort_graph_api;
}
