#pragma once

namespace OrtGraphApis {
ORT_API(const OrtGraphApi*, GetGraphApi, uint32_t version);
ORT_API_STATUS_IMPL(OrtGraph_PlaceHolder, const OrtGraphViewer* graph, _Out_ int* out);
}
