#pragma once

namespace OrtGraphApis {
ORT_API(const OrtGraphApi*, GetGraphApi, uint32_t version);

ORT_API(const char*, OrtGraph_GetName, const OrtGraphViewer*) ORT_ALL_ARGS_NONNULL;

ORT_API(bool, OrtGraph_IsConstantInitializer, const OrtGraphViewer* graph, const char* name, bool check_outer_scope)ORT_ALL_ARGS_NONNULL;

ORT_API(size_t, OrtGraph_GetNodesIndexInTopologicalOrder, const OrtGraphViewer* graph, int execution_order, _Out_ const size_t** nodes_index_in_topological_order);

ORT_API(bool, OrtGraph_IsSubgraph, const OrtGraph* graph);

ORT_API(const OrtGraph*, OrtGraph_GetParentGraph, const OrtGraph* graph);

ORT_API(const OrtNode*, OrtGraph_GetParenNode, const OrtGraphViewer* graph);

ORT_API(const void*, OrtGraph_GetModelPath, const OrtGraphViewer* graph);

ORT_API(const OrtGraph*, OrtGraph_GetOrtGraph, const OrtGraphViewer* graph_viewer);

ORT_API(size_t, OrtGraph_GetInputsIncludingInitializers, const OrtGraphViewer* graph, _Outptr_ const char*** input_names);

ORT_API(const OrtNode*, OrtGraph_GetOrtNode, const OrtGraphViewer* graph, size_t node_index);

ORT_API(size_t, OrtGraph_GetNodesConsumingInput, const OrtGraphViewer* graph, const char* input_name, _Outptr_ const OrtNode*** consumers);

ORT_API(const OrtNode*, OrtGraph_GetNodeProducingOutput, const OrtGraphViewer* graph, const char* output_name);

ORT_API(int, OrtGraph_NumberOfNodes, const OrtGraphViewer*) ORT_ALL_ARGS_NONNULL;

ORT_API(int, OrtGraph_MaxNodeIndex, const OrtGraphViewer* graph);

ORT_API(size_t, OrtGraph_GetOutputSize, const OrtGraphViewer*) ORT_ALL_ARGS_NONNULL;

ORT_API(const char*, OrtGraph_GetIthOutputName, const OrtGraphViewer*, size_t i) ORT_ALL_ARGS_NONNULL;

ORT_API(int32_t, OrtGraph_GetIthOutputElemType, const OrtGraphViewer*, size_t i) ORT_ALL_ARGS_NONNULL;

ORT_API(bool, OrtGraph_GetInitializerTensor, const OrtGraphViewer* graph, const char* initializer_name, _Outptr_ OrtTensorRef**);

ORT_API(bool, OrtGraph_GetValueInfo, const OrtGraphViewer* graph, const char* name, _Outptr_ OrtValueInfoRef**);

ORT_API(size_t, OrtGraph_SerializeToArray, const OrtGraphViewer*, _Out_ void** data);

ORT_API_STATUS_IMPL(OrtGraph_GetSubGraph, const OrtGraphViewer* graph, const int node_num, const size_t* node_indices, _Outptr_ const OrtGraphViewer** subgraph);

ORT_API_STATUS_IMPL(OrtGraph_ReleaseGraph, const OrtGraphViewer* graph);

ORT_API(const char*, OrtNode_GetName, const OrtNode* node);

ORT_API(const char*, OrtNode_GetDescription, const OrtNode* node);

ORT_API(const char*, OrtNode_GetDomain, const OrtNode* node);

ORT_API(int, OrtNode_SinceVersion, const OrtNode* node);

ORT_API(const char*, OrtNode_GetExecutionProviderType, const OrtNode* node);

ORT_API(const char*, OrtNode_GetOpType, const OrtNode* node);

ORT_API(size_t, OrtNode_GetImplicitInputSize, const OrtNode* node);

ORT_API(const char*, OrtNode_GetIthImplicitInputName, const OrtNode* node, size_t i);

ORT_API(size_t, OrtNode_GetInputSize, const OrtNode* node);

ORT_API(const char*, OrtNode_GetIthInputName, const OrtNode* node, size_t i);

ORT_API(size_t, OrtNode_GetOutputSize, const OrtNode* node);

ORT_API(const char*, OrtNode_GetIthOutputName, const OrtNode* node, size_t i);

ORT_API(size_t, OrtNode_GetIndex, const OrtNode* node);

ORT_API(size_t, OrtNode_GetAttributeNames, const OrtNode* node, const char*** names);

ORT_API(size_t, OrtNode_GetAttributeSize, const OrtNode* node);

ORT_API(int, OrtNode_GetAttributeType, const OrtNode* node, const char* attribute) ORT_ALL_ARGS_NONNULL;

ORT_API(size_t, OrtNode_GetAttributeKeyCount, const OrtNode* node, const char* key);

ORT_API(int, OrtNode_GetAttributeIntSize, const OrtNode* node, const char* key);

ORT_API(int, OrtNode_GetAttributeFloatSize, const OrtNode* node, const char* key);

ORT_API(int, OrtNode_GetAttributeStringSize, const OrtNode* node, const char* key);

ORT_API(int64_t, OrtNode_GetAttributeIthInt, const OrtNode* node, const char* key, int i);

ORT_API(float, OrtNode_GetAttributeIthFloat, const OrtNode* node, const char* key, int i);

ORT_API(const char*, OrtNode_GetAttributeIthStr, const OrtNode* node, const char* key, int i);

ORT_API(const char*, OrtNode_GetAttributeStr, const OrtNode* node, const char* key) ORT_ALL_ARGS_NONNULL;

ORT_API(int64_t, OrtNode_GetAttributeInt, const OrtNode* node, const char* key) ORT_ALL_ARGS_NONNULL;

ORT_API(float, OrtNode_GetAttributeFloat, const OrtNode* node, const char* key) ORT_ALL_ARGS_NONNULL;

ORT_API(size_t, OrtNode_GetSubgraphs, const OrtNode* node, _Outptr_ const OrtGraphViewer*** subgraphs);

ORT_API_STATUS_IMPL(OrtFreeMem, void* p);

}
