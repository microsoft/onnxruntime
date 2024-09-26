#pragma once
#include "onnxruntime_c_api.h"

struct OrtGraphApi {
ORT_API2_STATUS(OrtGraph_PlaceHolder, const OrtGraphViewer* graph, _Out_ int* out);
};
typedef struct OrtGraphApi OrtGraphApi;
