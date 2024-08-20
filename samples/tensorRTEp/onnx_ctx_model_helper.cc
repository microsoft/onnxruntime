#include "onnx_ctx_model_helper.h"
namespace onnxruntime {
bool GraphHasCtxNode(const OrtGraphViewer* graph_viewer) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int maxNodeIndex = 0;
  api->OrtGraph_MaxNodeIndex(graph_viewer, &maxNodeIndex);
  for (int i = 0; i < maxNodeIndex; ++i) {
    const OrtNode* node = nullptr;
    api->OrtGraph_GetOrtNode(graph_viewer, i, &node);
    const char* opType = nullptr;
    api->OrtNode_GetOpType(node, &opType);
    if (node != nullptr && strcmp(opType, EPCONTEXT_OP.c_str()) == 0) {
      return true;
    }
  }
  return false;
}
}
